"""
DeepEval RAG metrics evaluation.

Metrics implemented:
  - ContextualPrecision  : retrieved chunks ranked correctly (relevant > irrelevant)
  - ContextualRecall     : retrieved context covers the expected answer
  - ContextualRelevancy  : retrieved context is actually on-topic for the query
  - AnswerRelevancy      : the generated answer addresses the question
  - Faithfulness         : answer is grounded in context (no hallucinations)

OTel spans: rag.evaluate — total eval time, per-metric scores.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

from src.rag.pipeline import RAGPipeline, RAGResponse

logger = logging.getLogger(__name__)
_tracer = trace.get_tracer("rag.evaluation")


@dataclass
class RAGEvalSample:
    """One question-answer pair with optional reference for evaluation."""
    query: str
    expected_answer: str           # ground-truth / reference answer
    actual_answer: str = ""        # filled after RAG query
    retrieved_contexts: list[str] = field(default_factory=list)
    citations: list[str] = field(default_factory=list)


@dataclass
class MetricScore:
    name: str
    score: float
    reason: str
    passed: bool
    threshold: float


@dataclass
class RAGEvalResult:
    sample: RAGEvalSample
    metrics: list[MetricScore]
    overall_pass: bool
    latency_ms: float
    error: Optional[str] = None

    @property
    def avg_score(self) -> float:
        if not self.metrics:
            return 0.0
        return sum(m.score for m in self.metrics) / len(self.metrics)


_MAX_RETRIES = 2


async def _run_with_retry(fn, *args, **kwargs):
    last_err = None
    for attempt in range(_MAX_RETRIES + 1):
        try:
            return await fn(*args, **kwargs)
        except Exception as e:
            last_err = e
            if attempt < _MAX_RETRIES:
                logger.warning("RAG eval metric attempt %d failed: %s — retrying", attempt + 1, e)
    raise last_err


async def evaluate_rag_response(
    pipeline: RAGPipeline,
    sample: RAGEvalSample,
    thresholds: Optional[dict[str, float]] = None,
) -> RAGEvalResult:
    """
    Run all five DeepEval RAG metrics against a single sample.

    Args:
        pipeline: the RAGPipeline that produced the answer
        sample: query + expected_answer + actual_answer + retrieved contexts
        thresholds: per-metric pass/fail thresholds (default 0.5 each)

    Returns:
        RAGEvalResult with metric scores, pass/fail, and latency.
    """
    t0 = time.monotonic()
    thresholds = thresholds or {}
    with _tracer.start_as_current_span("rag.evaluate") as span:
        span.set_attribute("rag.query", sample.query[:200])
        span.set_attribute("rag.config_id", pipeline.config.config_id)

        try:
            from deepeval.test_case import LLMTestCase
            from deepeval.metrics import (
                ContextualPrecisionMetric,
                ContextualRecallMetric,
                ContextualRelevancyMetric,
                AnswerRelevancyMetric,
                FaithfulnessMetric,
            )
            from src.evaluation.integrations import build_deepeval_model  # reuse existing wrapper

            judge = build_deepeval_model()
            tc = LLMTestCase(
                input=sample.query,
                actual_output=sample.actual_answer,
                expected_output=sample.expected_answer,
                retrieval_context=sample.retrieved_contexts,
            )

            metrics_to_run = [
                ("contextual_precision", ContextualPrecisionMetric(
                    threshold=thresholds.get("contextual_precision", 0.5),
                    model=judge,
                    include_reason=True,
                )),
                ("contextual_recall", ContextualRecallMetric(
                    threshold=thresholds.get("contextual_recall", 0.5),
                    model=judge,
                    include_reason=True,
                )),
                ("contextual_relevancy", ContextualRelevancyMetric(
                    threshold=thresholds.get("contextual_relevancy", 0.5),
                    model=judge,
                    include_reason=True,
                )),
                ("answer_relevancy", AnswerRelevancyMetric(
                    threshold=thresholds.get("answer_relevancy", 0.5),
                    model=judge,
                    include_reason=True,
                )),
                ("faithfulness", FaithfulnessMetric(
                    threshold=thresholds.get("faithfulness", 0.5),
                    model=judge,
                    include_reason=True,
                )),
            ]

            scores: list[MetricScore] = []
            for metric_name, metric in metrics_to_run:
                try:
                    await _run_with_retry(metric.a_measure, tc)
                    scores.append(MetricScore(
                        name=metric_name,
                        score=metric.score,
                        reason=metric.reason or "",
                        passed=metric.is_successful(),
                        threshold=thresholds.get(metric_name, 0.5),
                    ))
                    span.set_attribute(f"rag.metric.{metric_name}", metric.score)
                except Exception as e:
                    err_msg = f"ERROR: {e}"
                    logger.error("RAG metric '%s' failed: %s", metric_name, e)
                    scores.append(MetricScore(
                        name=metric_name,
                        score=0.0,
                        reason=err_msg,
                        passed=False,
                        threshold=thresholds.get(metric_name, 0.5),
                    ))

            duration_ms = (time.monotonic() - t0) * 1000
            all_pass = all(m.passed for m in scores)
            span.set_attribute("rag.eval_pass", all_pass)
            span.set_attribute("rag.eval_ms", duration_ms)
            return RAGEvalResult(
                sample=sample,
                metrics=scores,
                overall_pass=all_pass,
                latency_ms=round(duration_ms),
            )

        except Exception as e:
            span.set_status(Status(StatusCode.ERROR, str(e)))
            span.record_exception(e)
            duration_ms = (time.monotonic() - t0) * 1000
            return RAGEvalResult(
                sample=sample,
                metrics=[],
                overall_pass=False,
                latency_ms=round(duration_ms),
                error=str(e),
            )


async def batch_evaluate(
    pipeline: RAGPipeline,
    samples: list[RAGEvalSample],
    thresholds: Optional[dict[str, float]] = None,
) -> list[RAGEvalResult]:
    """Run evaluate_rag_response for a list of samples sequentially."""
    results = []
    for i, sample in enumerate(samples):
        logger.info("RAG eval sample %d/%d: %s", i + 1, len(samples), sample.query[:60])
        # If actual_answer not yet filled, run the pipeline
        if not sample.actual_answer:
            try:
                resp: RAGResponse = await pipeline.query(sample.query)
                sample.actual_answer = resp.answer
                sample.retrieved_contexts = [c.snippet for c in resp.citations]
                sample.citations = [c.source for c in resp.citations]
            except Exception as e:
                sample.actual_answer = f"ERROR: {e}"
        result = await evaluate_rag_response(pipeline, sample, thresholds)
        results.append(result)
    return results
