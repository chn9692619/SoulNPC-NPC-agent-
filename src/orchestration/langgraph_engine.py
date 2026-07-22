"""
LangGraph-based Agent Orchestration Engine for SoulNPC.

Replaces the linear runtime pipeline with a state graph that supports
conditional branching, parallel execution, and tool calling.

Architecture:
    START -> parse_event -> [retrieve_memory, update_state] (parallel)
        -> decide_action -> call_tool? -> generate_dialogue -> write_memory -> END
"""
from __future__ import annotations

from typing import TypedDict, List, Dict, Any, Optional, Literal
from dataclasses import dataclass, field
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# Optional LangGraph import - graceful fallback
try:
    from langgraph.graph import StateGraph, END
    from langgraph.checkpoint.memory import MemorySaver
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    StateGraph = None
    END = None
    MemorySaver = None


@dataclass
class AgentState:
    """Core state that flows through the LangGraph nodes."""
    # Input
    player_event_text: str = ""
    
    # Parsed event
    event_id: str = ""
    event_primitives: Dict[str, float] = field(default_factory=dict)  # {primitive_name: weight}
    
    # Memory & context
    retrieved_memories: List[Dict[str, Any]] = field(default_factory=list)
    recalled_count: int = 0
    
    # Emotional & relationship state
    emotion_before: Dict[str, float] = field(default_factory=dict)
    emotion_after: Dict[str, float] = field(default_factory=dict)
    relation_before: Dict[str, float] = field(default_factory=dict)
    relation_after: Dict[str, float] = field(default_factory=dict)
    
    # Action decision
    action_label: str = ""
    action_confidence: float = 0.0
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    
    # Generated output
    dialogue_text: str = ""
    lora_prompt: Dict[str, Any] = field(default_factory=dict)
    
    # Metadata
    step_number: int = 0
    trace_id: str = ""
    errors: List[str] = field(default_factory=list)


class SoulNPCGraphEngine:
    """
    LangGraph-based orchestration engine for SoulNPC Agent.
    
    Usage:
        engine = SoulNPCGraphEngine(character_config="configs/character_ella.yaml")
        state = engine.run("玩家走近吧台，低声说：'来一杯最烈的酒'")
        print(state.dialogue_text)
    """
    
    def __init__(
        self,
        character_config: Optional[str] = None,
        enable_memory: bool = True,
        enable_tool_calling: bool = True,
        enable_tracing: bool = False,
    ):
        self.enable_memory = enable_memory
        self.enable_tool_calling = enable_tool_calling
        self.enable_tracing = enable_tracing
        
        # Lazy-loaded components
        self._runtime: Optional[Any] = None
        self._tools: Optional[Dict[str, Any]] = None
        self._tracer: Optional[Any] = None
        self._graph = None
        
        # Load character config
        self.character_config = character_config
        self._init_runtime()
        self._init_tools()
        self._init_tracing()
        
        if LANGGRAPH_AVAILABLE:
            self._graph = self._build_graph()
    
    def _init_runtime(self):
        """Initialize the SoulNPC Runtime Engine."""
        try:
            from src.agent_runtime.runtime_engine import SoulNPCRuntime
            self._runtime = SoulNPCRuntime(
                character_config_path=self.character_config,
                data_dir=str(ROOT / "data" / "runtime"),
            )
        except Exception as e:
            print(f"[Orchestration] Runtime init warning: {e}")
            self._runtime = None
    
    def _init_tools(self):
        """Initialize Tool Calling registry."""
        try:
            from src.tools.npc_tools import NPC_TOOLS, invoke_tool
            self._tools = NPC_TOOLS
            self._invoke_tool = invoke_tool
        except ImportError:
            print("[Orchestration] Tools module not available, using stub.")
            self._tools = {}
            self._invoke_tool = lambda name, **kwargs: {"error": "tools not loaded"}
    
    def _init_tracing(self):
        """Initialize Langfuse tracing if enabled."""
        if not self.enable_tracing:
            return
        try:
            from src.observability.tracer import SoulNPCTracer
            self._tracer = SoulNPCTracer()
        except ImportError:
            print("[Orchestration] Tracer not available.")
    
    # ---- Graph Nodes ----
    
    def _node_parse_event(self, state: AgentState) -> AgentState:
        """Parse player input into event primitives."""
        if self._tracer:
            self._tracer.trace_step("parse_event", {"input": state.player_event_text})
        
        if self._runtime:
            try:
                parsed = self._runtime.parse_event(state.player_event_text)
                state.event_id = parsed.get("event_id", "")
                state.event_primitives = parsed.get("primitives", {})
            except Exception as e:
                state.errors.append(f"parse_event: {e}")
        else:
            # Fallback: use rule-based parser directly
            from src.agent_runtime.event_parser import EventParser
            parser = EventParser()
            result = parser.parse(state.player_event_text)
            state.event_id = result.get("event_id", "unknown_event")
            state.event_primitives = result.get("primitives", {})
        
        return state
    
    def _node_retrieve_memory(self, state: AgentState) -> AgentState:
        """Retrieve relevant memories via weighted RAG."""
        if not self.enable_memory:
            return state
        
        if self._tracer:
            self._tracer.trace_step("retrieve_memory", {"event_id": state.event_id})
        
        if self._runtime:
            try:
                memories = self._runtime.retrieve_memories(
                    event_id=state.event_id,
                    event_primitives=state.event_primitives,
                    top_k=5,
                )
                state.retrieved_memories = memories
                state.recalled_count = len(memories)
            except Exception as e:
                state.errors.append(f"retrieve_memory: {e}")
        
        return state
    
    def _node_update_state(self, state: AgentState) -> AgentState:
        """Update emotional and relationship state vectors."""
        if self._runtime:
            try:
                # Capture state before update
                if hasattr(self._runtime, 'state_store'):
                    current = self._runtime.state_store.get_current_state()
                    state.emotion_before = current.get("emotion", {})
                    state.relation_before = current.get("relation", {})
                
                # Apply state update
                self._runtime.update_state(
                    event_id=state.event_id,
                    event_primitives=state.event_primitives,
                    memories=state.retrieved_memories,
                )
                
                # Capture state after update
                if hasattr(self._runtime, 'state_store'):
                    current = self._runtime.state_store.get_current_state()
                    state.emotion_after = current.get("emotion", {})
                    state.relation_after = current.get("relation", {})
                    
            except Exception as e:
                state.errors.append(f"update_state: {e}")
        
        return state
    
    def _node_decide_action(self, state: AgentState) -> AgentState:
        """Decide NPC action and potentially call tools."""
        if self._tracer:
            self._tracer.trace_step("decide_action", {
                "emotion": state.emotion_after,
                "relation": state.relation_after,
            })
        
        if self._runtime:
            try:
                from src.character.policy import choose_action
                action = choose_action(
                    emotion=state.emotion_after,
                    relation=state.relation_after,
                    event_id=state.event_id,
                )
                state.action_label = action.get("label", "neutral")
                state.action_confidence = action.get("confidence", 0.5)
            except Exception as e:
                state.errors.append(f"decide_action: {e}")
                state.action_label = "observe"
                state.action_confidence = 0.5
        
        # If tool calling is enabled, determine which tools to call
        if self.enable_tool_calling and self._tools:
            tool_calls = self._route_tools(state)
            state.tool_calls = tool_calls
        
        return state
    
    def _route_tools(self, state: AgentState) -> List[Dict[str, Any]]:
        """Determine which tools to call based on state."""
        calls = []
        
        # Tool: calculate_emotion_intensity
        if self._tools.get("calculate_emotion_intensity"):
            calls.append({
                "tool": "calculate_emotion_intensity",
                "params": {
                    "emotion_vector": state.emotion_after,
                    "event_id": state.event_id,
                }
            })
        
        # Tool: assess_relationship_change
        if self._tools.get("assess_relationship_change"):
            calls.append({
                "tool": "assess_relationship_change",
                "params": {
                    "relation_before": state.relation_before,
                    "relation_after": state.relation_after,
                }
            })
        
        return calls
    
    def _node_execute_tools(self, state: AgentState) -> AgentState:
        """Execute pending tool calls."""
        if self._tracer:
            self._tracer.trace_step("execute_tools", {"calls": len(state.tool_calls)})
        
        for tool_call in state.tool_calls:
            tool_name = tool_call["tool"]
            params = tool_call.get("params", {})
            try:
                result = self._invoke_tool(tool_name, **params)
                tool_call["result"] = result
            except Exception as e:
                tool_call["error"] = str(e)
                state.errors.append(f"tool:{tool_name}: {e}")
        
        return state
    
    def _node_generate_dialogue(self, state: AgentState) -> AgentState:
        """Generate NPC dialogue based on state and action."""
        if self._tracer:
            self._tracer.trace_step("generate_dialogue", {"action": state.action_label})
        
        if self._runtime:
            try:
                # Build context
                context = {
                    "emotion": state.emotion_after,
                    "relation": state.relation_after,
                    "memories": state.retrieved_memories[-3:],
                    "action": state.action_label,
                    "event_id": state.event_id,
                }
                
                # Generate via LoRA or rule-based
                dialogue = self._runtime.generate_response(context)
                state.dialogue_text = dialogue.get("text", "")
                state.lora_prompt = dialogue.get("prompt", {})
                
            except Exception as e:
                # Fallback to rule-based
                from src.character.policy import generate_rule_dialogue
                state.dialogue_text = generate_rule_dialogue(
                    action=state.action_label,
                    character_config=self.character_config,
                )
        
        return state
    
    def _node_write_memory(self, state: AgentState) -> AgentState:
        """Write this interaction to persistent memory."""
        if not self.enable_memory:
            return state
        
        try:
            memory_entry = {
                "event_id": state.event_id,
                "player_text": state.player_event_text,
                "npc_action": state.action_label,
                "npc_dialogue": state.dialogue_text,
                "emotion_change": {
                    k: state.emotion_after.get(k, 0) - state.emotion_before.get(k, 0)
                    for k in state.emotion_after
                },
            }
            if self._runtime and hasattr(self._runtime, 'write_memory'):
                self._runtime.write_memory(memory_entry)
        except Exception as e:
            state.errors.append(f"write_memory: {e}")
        
        return state
    
    # ---- Graph Construction ----
    
    def _build_graph(self):
        """Build the LangGraph StateGraph."""
        if not LANGGRAPH_AVAILABLE:
            return None
        
        graph = StateGraph(AgentState)
        
        # Add nodes
        graph.add_node("parse_event", self._node_parse_event)
        graph.add_node("retrieve_memory", self._node_retrieve_memory)
        graph.add_node("update_state", self._node_update_state)
        graph.add_node("decide_action", self._node_decide_action)
        graph.add_node("execute_tools", self._node_execute_tools)
        graph.add_node("generate_dialogue", self._node_generate_dialogue)
        graph.add_node("write_memory", self._node_write_memory)
        
        # Define edges
        graph.set_entry_point("parse_event")
        graph.add_edge("parse_event", "retrieve_memory")
        graph.add_edge("parse_event", "update_state")  # parallel conceptual
        graph.add_edge("retrieve_memory", "decide_action")
        graph.add_edge("update_state", "decide_action")
        
        # Conditional: execute tools if available
        def should_call_tools(state: AgentState) -> str:
            if state.tool_calls:
                return "execute_tools"
            return "generate_dialogue"
        
        graph.add_conditional_edges(
            "decide_action",
            should_call_tools,
            {"execute_tools": "execute_tools", "generate_dialogue": "generate_dialogue"}
        )
        graph.add_edge("execute_tools", "generate_dialogue")
        graph.add_edge("generate_dialogue", "write_memory")
        graph.add_edge("write_memory", END)
        
        # Compile with memory for multi-turn support
        memory = MemorySaver()
        return graph.compile(checkpointer=memory)
    
    # ---- Public API ----
    
    def run(self, player_event_text: str, thread_id: str = "default") -> AgentState:
        """
        Run a single agent step.
        
        Args:
            player_event_text: Natural language player action description
            thread_id: Conversation thread ID for multi-turn memory
        
        Returns:
            AgentState with all outputs
        """
        initial_state = AgentState(
            player_event_text=player_event_text,
            step_number=getattr(self, '_step_counter', 0) + 1,
        )
        self._step_counter = initial_state.step_number
        
        if self._tracer:
            self._tracer.start_trace(thread_id, initial_state.player_event_text)
        
        if self._graph and LANGGRAPH_AVAILABLE:
            config = {"configurable": {"thread_id": thread_id}}
            result = self._graph.invoke(initial_state, config)
        else:
            # Fallback: sequential pipeline (original behavior)
            result = self._run_sequential(initial_state)
        
        if self._tracer:
            self._tracer.end_trace(result)
        
        return result
    
    def _run_sequential(self, state: AgentState) -> AgentState:
        """Fallback sequential pipeline when LangGraph is unavailable."""
        state = self._node_parse_event(state)
        state = self._node_retrieve_memory(state)
        state = self._node_update_state(state)
        state = self._node_decide_action(state)
        state = self._node_execute_tools(state)
        state = self._node_generate_dialogue(state)
        state = self._node_write_memory(state)
        return state
    
    async def arun(self, player_event_text: str, thread_id: str = "default") -> AgentState:
        """Async version of run()."""
        if self._graph and LANGGRAPH_AVAILABLE:
            config = {"configurable": {"thread_id": thread_id}}
            initial_state = AgentState(player_event_text=player_event_text)
            result = await self._graph.ainvoke(initial_state, config)
            return result
        return self.run(player_event_text, thread_id)
    
    def stream(self, player_event_text: str, thread_id: str = "default"):
        """Stream agent execution step by step."""
        if not self._graph or not LANGGRAPH_AVAILABLE:
            state = self._run_sequential(
                AgentState(player_event_text=player_event_text)
            )
            yield {"step": "complete", "state": state}
            return
        
        config = {"configurable": {"thread_id": thread_id}}
        initial_state = AgentState(player_event_text=player_event_text)
        
        for event in self._graph.stream(initial_state, config):
            yield event
