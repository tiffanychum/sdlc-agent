"""
SQLAlchemy models for the SDLC Agent platform.

Covers: Teams, Agents, Skills, Tool Mappings, Traces, Spans, and Evaluation Runs.
"""

import uuid
from datetime import datetime

from sqlalchemy import (
    Column, String, Text, Float, Integer, Boolean,
    DateTime, ForeignKey, Enum, JSON, create_engine,
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
    total_latency_ms = Column(Float, default=0.0)
    total_tokens = Column(Integer, default=0)
    total_cost = Column(Float, default=0.0)
    status = Column(String, default="running")
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
