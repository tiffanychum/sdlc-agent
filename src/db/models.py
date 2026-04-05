"""
SQLAlchemy models for the SDLC Agent platform.

Covers: Teams, Agents, Skills, Tool Mappings, Traces, Spans, and Evaluation Runs.
"""

import uuid
from datetime import datetime

from sqlalchemy import (
    Column, String, Text, Float, Integer, Boolean,
    DateTime, ForeignKey, JSON,
)
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


def _uuid() -> str:
    return uuid.uuid4().hex[:12]


# ── Teams & Agents ──────────────────────────────────────────────

class Team(Base):
    __tablename__ = "teams"

    id = Column(String, primary_key=True, default=_uuid)
    name = Column(String, nullable=False)
    description = Column(Text, default="")
    decision_strategy = Column(
        String, default="router_decides",
    )
    created_at = Column(DateTime, default=datetime.utcnow)

    agents = relationship("Agent", back_populates="team", cascade="all, delete-orphan")


class Agent(Base):
    __tablename__ = "agents"

    id = Column(String, primary_key=True, default=_uuid)
    team_id = Column(String, ForeignKey("teams.id"), nullable=False)
    name = Column(String, nullable=False)
    role = Column(String, nullable=False)
    description = Column(Text, default="")
    system_prompt = Column(Text, default="")
    model = Column(String, default="")
    decision_strategy = Column(String, default="react")
    created_at = Column(DateTime, default=datetime.utcnow)

    team = relationship("Team", back_populates="agents")
    tool_mappings = relationship("AgentToolMapping", back_populates="agent", cascade="all, delete-orphan")
    skill_mappings = relationship("AgentSkillMapping", back_populates="agent", cascade="all, delete-orphan")


class AgentToolMapping(Base):
    __tablename__ = "agent_tool_mappings"

    id = Column(String, primary_key=True, default=_uuid)
    agent_id = Column(String, ForeignKey("agents.id"), nullable=False)
    tool_group = Column(String, nullable=False)

    agent = relationship("Agent", back_populates="tool_mappings")


class Skill(Base):
    __tablename__ = "skills"

    id = Column(String, primary_key=True, default=_uuid)
    name = Column(String, nullable=False)
    description = Column(Text, default="")
    instructions = Column(Text, nullable=False)
    trigger_pattern = Column(String, default="")
    created_at = Column(DateTime, default=datetime.utcnow)

    agent_mappings = relationship("AgentSkillMapping", back_populates="skill", cascade="all, delete-orphan")


class AgentSkillMapping(Base):
    __tablename__ = "agent_skill_mappings"

    id = Column(String, primary_key=True, default=_uuid)
    agent_id = Column(String, ForeignKey("agents.id"), nullable=False)
    skill_id = Column(String, ForeignKey("skills.id"), nullable=False)

    agent = relationship("Agent", back_populates="skill_mappings")
    skill = relationship("Skill", back_populates="agent_mappings")


# ── Tracing ─────────────────────────────────────────────────────

class Trace(Base):
    __tablename__ = "traces"

    id = Column(String, primary_key=True, default=_uuid)
    team_id = Column(String, ForeignKey("teams.id"), nullable=True)
    user_prompt = Column(Text, default="")
    agent_used = Column(String, default="")
    agent_response = Column(Text, default="")
    tool_calls_json = Column(JSON, default=list)
    total_latency_ms = Column(Float, default=0.0)
    total_tokens = Column(Integer, default=0)
    total_cost = Column(Float, default=0.0)
    status = Column(String, default="running")
    eval_scores = Column(JSON, default=dict)
    eval_status = Column(String, default="pending")  # pending | evaluated
    created_at = Column(DateTime, default=datetime.utcnow)

    spans = relationship("Span", back_populates="trace", cascade="all, delete-orphan")


class Span(Base):
    __tablename__ = "spans"

    id = Column(String, primary_key=True, default=_uuid)
    trace_id = Column(String, ForeignKey("traces.id"), nullable=False)
    parent_span_id = Column(String, nullable=True)
    name = Column(String, nullable=False)
    span_type = Column(String, nullable=False)  # routing | llm_call | tool_call
    start_time = Column(DateTime, default=datetime.utcnow)
    end_time = Column(DateTime, nullable=True)
    input_data = Column(JSON, default=dict)
    output_data = Column(JSON, default=dict)
    tokens_in = Column(Integer, default=0)
    tokens_out = Column(Integer, default=0)
    cost = Column(Float, default=0.0)
    model = Column(String, default="")
    status = Column(String, default="running")
    error = Column(Text, nullable=True)

    trace = relationship("Trace", back_populates="spans")


# ── Evaluation ──────────────────────────────────────────────────

class EvalRun(Base):
    __tablename__ = "eval_runs"

    id = Column(String, primary_key=True, default=_uuid)
    model = Column(String, nullable=False)
    prompt_version = Column(String, default="v1")
    team_id = Column(String, ForeignKey("teams.id"), nullable=True)
    num_tasks = Column(Integer, default=0)
    task_completion_rate = Column(Float, default=0.0)
    routing_accuracy = Column(Float, default=0.0)
    avg_tool_call_accuracy = Column(Float, default=0.0)
    avg_failure_recovery_rate = Column(Float, default=0.0)
    avg_latency_ms = Column(Float, nullable=True)
    total_cost = Column(Float, default=0.0)
    total_tokens = Column(Integer, default=0)
    results_json = Column(JSON, default=dict)
    created_at = Column(DateTime, default=datetime.utcnow)

    regression_results = relationship("RegressionResult", back_populates="eval_run", cascade="all, delete-orphan")


# ── Golden Dataset & Regression ──────────────────────────────────

class GoldenTestCase(Base):
    __tablename__ = "golden_test_cases"

    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    prompt = Column(Text, nullable=False)
    expected_agent = Column(String, default="")
    expected_tools = Column(JSON, default=list)
    expected_output_keywords = Column(JSON, default=list)
    expected_delegation_pattern = Column(JSON, default=list)
    quality_thresholds = Column(JSON, default=dict)
    max_llm_calls = Column(Integer, default=15)
    max_tool_calls = Column(Integer, default=10)
    max_tokens = Column(Integer, default=8000)
    max_latency_ms = Column(Integer, default=120000)
    complexity = Column(String, default="quick")
    version = Column(String, default="1.0")
    reference_output = Column(Text, default="")
    # "auto", "supervisor", "sequential", "parallel", "router_decides", or None (team default)
    strategy = Column(String, nullable=True)
    expected_strategy = Column(String, nullable=True)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class RegressionResult(Base):
    __tablename__ = "regression_results"

    id = Column(String, primary_key=True, default=_uuid)
    run_id = Column(String, ForeignKey("eval_runs.id"), nullable=False)
    golden_case_id = Column(String, nullable=False)
    golden_case_name = Column(String, default="")

    actual_output = Column(Text, default="")
    actual_agent = Column(String, default="")
    actual_tools = Column(JSON, default=list)
    actual_delegation_pattern = Column(JSON, default=list)
    full_trace = Column(JSON, default=list)
    span_data = Column(JSON, default=list)

    actual_llm_calls = Column(Integer, default=0)
    actual_tool_calls = Column(Integer, default=0)
    actual_tokens_in = Column(Integer, default=0)
    actual_tokens_out = Column(Integer, default=0)
    actual_latency_ms = Column(Float, default=0.0)
    actual_cost = Column(Float, default=0.0)

    semantic_similarity = Column(Float, default=0.0)
    quality_scores = Column(JSON, default=dict)
    deepeval_scores = Column(JSON, default=dict)
    trace_assertions = Column(JSON, default=dict)
    eval_reasoning = Column(JSON, default=dict)

    cost_regression = Column(Boolean, default=False)
    latency_regression = Column(Boolean, default=False)
    quality_regression = Column(Boolean, default=False)
    trace_regression = Column(Boolean, default=False)
    overall_pass = Column(Boolean, default=True)

    rca_analysis = Column(JSON, nullable=True)
    model_used = Column(String, default="")
    prompt_version = Column(String, default="v1")
    # Strategy fields: what was configured vs what was actually executed
    expected_strategy = Column(String, nullable=True)
    actual_strategy = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    eval_run = relationship("EvalRun", back_populates="regression_results")


# ── RAG Pipeline ─────────────────────────────────────────────────

class RagConfig(Base):
    """Stores the full configuration for one RAG pipeline instance."""
    __tablename__ = "rag_configs"

    id = Column(String, primary_key=True, default=_uuid)
    name = Column(String, nullable=False)
    description = Column(Text, default="")
    embedding_model = Column(String, default="openai/text-embedding-3-small")
    vector_store = Column(String, default="chroma")
    llm_model = Column(String, nullable=True)
    chunk_size = Column(Integer, default=1000)
    chunk_overlap = Column(Integer, default=200)
    chunk_strategy = Column(String, default="recursive")
    retrieval_strategy = Column(String, default="similarity")
    top_k = Column(Integer, default=5)
    mmr_lambda = Column(Float, default=0.5)
    multi_query_n = Column(Integer, default=3)
    system_prompt = Column(Text, nullable=True)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    sources = relationship("RagSource", back_populates="config", cascade="all, delete-orphan")
    queries = relationship("RagQuery", back_populates="config", cascade="all, delete-orphan")


class RagSource(Base):
    """One document source ingested into a RAG pipeline."""
    __tablename__ = "rag_sources"

    id = Column(String, primary_key=True, default=_uuid)
    config_id = Column(String, ForeignKey("rag_configs.id"), nullable=False)
    source_type = Column(String, nullable=False)   # text | file | url
    content = Column(Text, nullable=False)          # raw text, file path, or URL
    label = Column(String, default="")              # user-friendly name
    chunks_count = Column(Integer, default=0)
    tokens_estimated = Column(Integer, default=0)
    status = Column(String, default="pending")      # pending | ingested | error
    error_message = Column(Text, nullable=True)
    ingested_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    config = relationship("RagConfig", back_populates="sources")


class RagQuery(Base):
    """One RAG chat query log with citations, metrics, and evaluation."""
    __tablename__ = "rag_queries"

    id = Column(String, primary_key=True, default=_uuid)
    config_id = Column(String, ForeignKey("rag_configs.id"), nullable=False)
    query = Column(Text, nullable=False)
    answer = Column(Text, default="")
    citations = Column(JSON, default=list)          # list of {source, chunk, page, score, snippet}
    strategy_used = Column(String, default="similarity")
    chunks_retrieved = Column(Integer, default=0)
    tokens_in = Column(Integer, default=0)
    tokens_out = Column(Integer, default=0)
    latency_ms = Column(Float, default=0.0)
    # Evaluation results (populated async after query)
    eval_scores = Column(JSON, nullable=True)       # {metric: {score, passed, reason}}
    eval_status = Column(String, default="pending") # pending | running | done | error
    eval_error = Column(Text, nullable=True)
    # Tracing linkage
    trace_id = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    config = relationship("RagConfig", back_populates="queries")


# ── Prompt Versioning ────────────────────────────────────────────

class PromptVersion(Base):
    __tablename__ = "prompt_versions"

    id = Column(String, primary_key=True, default=_uuid)
    version_label = Column(String, nullable=False, unique=True)
    description = Column(Text, default="")
    agent_prompts = Column(JSON, default=dict)
    team_strategy = Column(String, default="")
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
