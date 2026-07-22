"""
Langfuse-based Observability & Evaluation for SoulNPC Agent.

Provides:
- Full-chain tracing of Agent execution steps
- Prompt version management
- Token & latency monitoring
- Bad Case collection & annotation
- Evaluation metrics (retrieval recall, generation quality)

Integration is lite - the tracer works even without Langfuse server.
When Langfuse is unavailable, traces are logged locally.
"""
from __future__ import annotations

import os
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field

ROOT = Path(__file__).resolve().parents[2]

# Try importing Langfuse SDK
try:
    import langfuse
    LANGFUSE_AVAILABLE = True
except ImportError:
    LANGFUSE_AVAILABLE = False
    langfuse = None


@dataclass
class TraceStep:
    """A single step in the Agent execution trace."""
    step_name: str
    timestamp: float = field(default_factory=time.time)
    input_data: Dict[str, Any] = field(default_factory=dict)
    output_data: Dict[str, Any] = field(default_factory=dict)
    duration_ms: float = 0
    error: Optional[str] = None


@dataclass
class ExecutionTrace:
    """Full execution trace for one Agent run."""
    trace_id: str
    thread_id: str
    player_input: str
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    steps: List[TraceStep] = field(default_factory=list)
    total_tokens: int = 0
    errors: List[str] = field(default_factory=list)
    
    @property
    def total_duration_ms(self) -> float:
        if self.end_time:
            return (self.end_time - self.start_time) * 1000
        return (time.time() - self.start_time) * 1000


class SoulNPCTracer:
    """
    Lightweight tracer for SoulNPC Agent.
    
    Usage:
        tracer = SoulNPCTracer()
        tracer.start_trace("thread-1", "玩家走进酒馆")
        tracer.trace_step("parse_event", {"input": "..."})
        # ... agent runs ...
        tracer.end_trace(final_state)
    """
    
    def __init__(
        self,
        enable_remote: bool = False,
        log_dir: Optional[str] = None,
    ):
        """
        Args:
            enable_remote: Send traces to Langfuse server
            log_dir: Local directory for trace logs (default: data/traces/)
        """
        self.enable_remote = enable_remote
        self.log_dir = Path(log_dir) if log_dir else ROOT / "data" / "traces"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Current trace
        self._current_trace: Optional[ExecutionTrace] = None
        self._current_step: Optional[TraceStep] = None
        self._step_timer: float = 0
        
        # Bad Case storage
        self.bad_cases: List[Dict[str, Any]] = []
        self._load_bad_cases()
        
        # Langfuse client (lazy init)
        self._langfuse_client = None
        if enable_remote and LANGFUSE_AVAILABLE:
            self._init_langfuse()
    
    def _init_langfuse(self):
        """Initialize Langfuse client if credentials are available."""
        public_key = os.getenv("LANGFUSE_PUBLIC_KEY", "")
        secret_key = os.getenv("LANGFUSE_SECRET_KEY", "")
        host = os.getenv("LANGFUSE_HOST", "http://localhost:3000")
        
        if public_key and secret_key:
            try:
                self._langfuse_client = langfuse.Langfuse(
                    public_key=public_key,
                    secret_key=secret_key,
                    host=host,
                )
                print(f"[Tracer] Langfuse connected: {host}")
            except Exception as e:
                print(f"[Tracer] Langfuse init failed: {e}")
    
    # ---- Tracing API ----
    
    def start_trace(self, thread_id: str, player_input: str):
        """Start a new execution trace."""
        trace_id = f"trace_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        self._current_trace = ExecutionTrace(
            trace_id=trace_id,
            thread_id=thread_id,
            player_input=player_input,
        )
        print(f"[Tracer] Trace started: {trace_id}")
    
    def trace_step(self, step_name: str, input_data: Dict[str, Any] = None):
        """Start tracing a new step in the execution."""
        if not self._current_trace:
            return
        
        # Finish previous step if exists
        if self._current_step:
            self._current_step.duration_ms = (time.time() - self._step_timer) * 1000
            self._current_trace.steps.append(self._current_step)
        
        # Start new step
        self._current_step = TraceStep(
            step_name=step_name,
            input_data=input_data or {},
        )
        self._step_timer = time.time()
    
    def end_trace(self, final_state: Any = None):
        """End the current trace and persist."""
        if not self._current_trace:
            return
        
        # Finish last step
        if self._current_step:
            self._current_step.duration_ms = (time.time() - self._step_timer) * 1000
            if "output" in self._current_step.__dict__:
                pass
            self._current_trace.steps.append(self._current_step)
            self._current_step = None
        
        # Collect errors from final state
        if final_state and hasattr(final_state, 'errors'):
            self._current_trace.errors.extend(final_state.errors)
        
        self._current_trace.end_time = time.time()
        
        # Persist locally
        self._save_trace(self._current_trace)
        
        # Send to Langfuse if enabled
        if self.enable_remote and self._langfuse_client:
            self._send_to_langfuse(self._current_trace)
        
        trace = self._current_trace
        self._current_trace = None
        
        print(f"[Tracer] Trace complete: {trace.trace_id} "
              f"({len(trace.steps)} steps, {trace.total_duration_ms:.0f}ms)")
    
    def _save_trace(self, trace: ExecutionTrace):
        """Save trace to local JSON file."""
        trace_file = self.log_dir / f"{trace.trace_id}.json"
        trace_data = {
            "trace_id": trace.trace_id,
            "thread_id": trace.thread_id,
            "player_input": trace.player_input,
            "start_time": trace.start_time,
            "end_time": trace.end_time,
            "total_duration_ms": trace.total_duration_ms,
            "total_tokens": trace.total_tokens,
            "errors": trace.errors,
            "steps": [
                {
                    "name": s.step_name,
                    "duration_ms": s.duration_ms,
                    "input": s.input_data,
                    "output": s.output_data,
                    "error": s.error,
                }
                for s in trace.steps
            ],
        }
        with open(trace_file, "w", encoding="utf-8") as f:
            json.dump(trace_data, f, ensure_ascii=False, indent=2)
    
    def _send_to_langfuse(self, trace: ExecutionTrace):
        """Send trace to Langfuse server."""
        if not self._langfuse_client:
            return
        
        try:
            langfuse_trace = self._langfuse_client.trace(
                name="soulnpc_agent_run",
                id=trace.trace_id,
                metadata={
                    "thread_id": trace.thread_id,
                    "player_input": trace.player_input,
                },
            )
            
            for step in trace.steps:
                langfuse_trace.span(
                    name=step.step_name,
                    input=step.input_data,
                    output=step.output_data,
                )
            
            self._langfuse_client.flush()
        except Exception as e:
            print(f"[Tracer] Langfuse send failed: {e}")
    
    # ---- Bad Case Management ----
    
    def record_bad_case(
        self,
        trace_id: str,
        issue_type: str,
        description: str,
        expected_behavior: str = "",
        severity: str = "medium",
    ):
        """
        Record a bad case for later analysis.
        
        Args:
            trace_id: Trace ID of the bad interaction
            issue_type: e.g., 'hallucination', 'wrong_emotion', 'poor_dialogue', 'memory_error'
            description: What went wrong
            expected_behavior: What should have happened
            severity: 'low', 'medium', 'high', 'critical'
        """
        bad_case = {
            "id": f"bc_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "trace_id": trace_id,
            "issue_type": issue_type,
            "description": description,
            "expected_behavior": expected_behavior,
            "severity": severity,
            "timestamp": datetime.now().isoformat(),
            "resolved": False,
        }
        self.bad_cases.append(bad_case)
        self._save_bad_cases()
        print(f"[Tracer] Bad case recorded: {bad_case['id']} ({issue_type})")
    
    def _load_bad_cases(self):
        """Load bad cases from file."""
        bc_file = self.log_dir / "bad_cases.json"
        if bc_file.exists():
            try:
                with open(bc_file, "r", encoding="utf-8") as f:
                    self.bad_cases = json.load(f)
            except Exception:
                self.bad_cases = []
    
    def _save_bad_cases(self):
        """Save bad cases to file."""
        bc_file = self.log_dir / "bad_cases.json"
        with open(bc_file, "w", encoding="utf-8") as f:
            json.dump(self.bad_cases, f, ensure_ascii=False, indent=2)
    
    def get_bad_case_summary(self) -> Dict[str, Any]:
        """Get summary statistics of bad cases."""
        by_type = {}
        by_severity = {}
        for bc in self.bad_cases:
            by_type[bc["issue_type"]] = by_type.get(bc["issue_type"], 0) + 1
            by_severity[bc["severity"]] = by_severity.get(bc["severity"], 0) + 1
        
        return {
            "total": len(self.bad_cases),
            "unresolved": sum(1 for bc in self.bad_cases if not bc["resolved"]),
            "by_type": by_type,
            "by_severity": by_severity,
        }
    
    # ---- Evaluation Metrics ----
    
    @staticmethod
    def calculate_retrieval_metrics(
        retrieved: List[Dict[str, Any]],
        relevant_ids: List[str],
    ) -> Dict[str, float]:
        """Calculate recall, precision for memory retrieval."""
        if not relevant_ids:
            return {"recall": 0, "precision": 0, "f1": 0}
        
        retrieved_ids = {r.get("id", "") for r in retrieved}
        relevant_set = set(relevant_ids)
        
        tp = len(retrieved_ids & relevant_set)
        recall = tp / len(relevant_set) if relevant_set else 0
        precision = tp / len(retrieved_ids) if retrieved_ids else 0
        f1 = 2 * recall * precision / (recall + precision) if (recall + precision) > 0 else 0
        
        return {
            "recall": round(recall, 4),
            "precision": round(precision, 4),
            "f1": round(f1, 4),
        }
