"""
NPC Tool Calling System for SoulNPC Agent.

Defines Function Calling tools that the Agent can invoke during
the decision-making process. Each tool has a structured schema
(name, description, parameters) that the LLM/Agent uses for selection.

Design principles for stable Function Calling:
1. Clear, specific descriptions - the LLM uses these to decide which tool to call
2. Typed parameters with constraints - prevents malformed calls
3. Side-effect labeling - marks tools that modify state
4. Parameter validation - programmatic safety net before execution
"""
from __future__ import annotations

from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
import math


# ---- Tool Schema Definition ----

@dataclass
class ToolParameter:
    """Parameter definition for a tool."""
    name: str
    type: str  # "float", "str", "int", "list", "dict"
    description: str
    required: bool = True
    default: Any = None
    constraints: Optional[Dict[str, Any]] = None  # e.g., {"min": 0, "max": 1}


@dataclass
class ToolSchema:
    """Complete tool schema with metadata for LLM function calling."""
    name: str
    description: str
    parameters: List[ToolParameter]
    has_side_effects: bool = False
    requires_permission: bool = False
    
    def to_openai_schema(self) -> Dict[str, Any]:
        """Convert to OpenAI Function Calling schema format."""
        properties = {}
        required = []
        for p in self.parameters:
            properties[p.name] = {
                "type": p.type,
                "description": p.description,
            }
            if p.constraints:
                properties[p.name].update(p.constraints)
            if p.required:
                required.append(p.name)
        
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            }
        }


# ---- Tool Implementations ----

def tool_calculate_emotion_intensity(
    emotion_vector: Dict[str, float],
    event_id: str = "",
) -> Dict[str, Any]:
    """
    Calculate emotional intensity metrics from the current emotion vector.
    
    Args:
        emotion_vector: 6-dim emotion vector {valence, arousal, dominance, safety, stress, curiosity}
        event_id: Event identifier for context
    
    Returns:
        Intensity metrics including dominant emotion, intensity score, and classification
    """
    if not emotion_vector:
        return {"error": "empty emotion vector", "dominant": "unknown", "intensity": 0}
    
    # Find dominant emotion (highest absolute value)
    dominant = max(emotion_vector.items(), key=lambda x: abs(x[1]))
    
    # Calculate overall intensity (L2 norm normalized)
    intensity = math.sqrt(sum(v**2 for v in emotion_vector.values())) / math.sqrt(len(emotion_vector))
    
    # Classify emotional state
    valence = emotion_vector.get("valence", 0)
    arousal = emotion_vector.get("arousal", 0)
    
    if valence > 0.3 and arousal > 0.3:
        state = "excited"
    elif valence > 0.3 and arousal < -0.3:
        state = "calm"
    elif valence < -0.3 and arousal > 0.3:
        state = "distressed"
    elif valence < -0.3 and arousal < -0.3:
        state = "depressed"
    else:
        state = "neutral"
    
    return {
        "dominant_emotion": dominant[0],
        "dominant_value": round(dominant[1], 4),
        "overall_intensity": round(intensity, 4),
        "emotional_state": state,
        "event_id": event_id,
    }


def tool_assess_relationship_change(
    relation_before: Dict[str, float],
    relation_after: Dict[str, float],
) -> Dict[str, Any]:
    """
    Assess how the NPC-player relationship changed after an interaction.
    
    Args:
        relation_before: 6-dim relation vector before interaction
        relation_after: 6-dim relation vector after interaction
    
    Returns:
        Change analysis for each relationship dimension
    """
    if not relation_before or not relation_after:
        return {"error": "missing relation vectors"}
    
    changes = {}
    all_keys = set(relation_before.keys()) | set(relation_after.keys())
    
    for key in all_keys:
        before = relation_before.get(key, 0)
        after = relation_after.get(key, 0)
        delta = after - before
        
        if abs(delta) < 0.05:
            trend = "stable"
        elif delta > 0:
            trend = "increasing"
        else:
            trend = "decreasing"
        
        changes[key] = {
            "before": round(before, 4),
            "after": round(after, 4),
            "delta": round(delta, 4),
            "trend": trend,
        }
    
    # Identify most significant change
    max_change = max(changes.items(), key=lambda x: abs(x[1]["delta"]))
    
    return {
        "dimension_changes": changes,
        "most_significant_change": max_change[0],
        "most_significant_delta": max_change[1]["delta"],
        "overall_stability": "stable" if all(
            abs(v["delta"]) < 0.1 for v in changes.values()
        ) else "evolving",
    }


def tool_retrieve_relevant_lore(
    query: str,
    top_k: int = 3,
) -> Dict[str, Any]:
    """
    Retrieve relevant world lore or character background based on query.
    
    Args:
        query: Search query about world or character lore
        top_k: Number of results to return (1-10)
    
    Returns:
        Retrieved lore entries with relevance scores
    """
    top_k = max(1, min(top_k, 10))
    
    # Placeholder - in production, this would query a vector store
    return {
        "query": query,
        "results": [],
        "count": 0,
        "top_k": top_k,
        "note": "Lore retrieval requires vector store setup. See docs for integration guide.",
    }


def tool_generate_dialogue_variations(
    base_dialogue: str,
    tone: str = "neutral",
    count: int = 3,
) -> Dict[str, Any]:
    """
    Generate variations of NPC dialogue in different tones.
    
    Args:
        base_dialogue: Original dialogue line
        tone: Desired tone (friendly, hostile, sarcastic, neutral, worried, excited)
        count: Number of variations (1-5)
    
    Returns:
        List of dialogue variations
    """
    valid_tones = ["friendly", "hostile", "sarcastic", "neutral", "worried", "excited"]
    if tone not in valid_tones:
        tone = "neutral"
    count = max(1, min(count, 5))
    
    return {
        "base_dialogue": base_dialogue,
        "tone": tone,
        "variations": [
            f"[{tone}] {base_dialogue}"
            for _ in range(count)
        ],
        "note": "Dialogue variation generation via LLM. In production, use a text generation model.",
    }


# ---- Tool Registry ----

# All available NPC tools with their schemas
NPC_TOOL_SCHEMAS = [
    ToolSchema(
        name="calculate_emotion_intensity",
        description="Calculate emotional intensity metrics from the current NPC emotion vector. Returns dominant emotion, overall intensity score, and emotional state classification (excited/calm/distressed/depressed/neutral).",
        parameters=[
            ToolParameter("emotion_vector", "object", "6-dim emotion vector with keys: valence, arousal, dominance, safety, stress, curiosity"),
            ToolParameter("event_id", "string", "Event identifier for context tracking", required=False, default=""),
        ],
        has_side_effects=False,
    ),
    ToolSchema(
        name="assess_relationship_change",
        description="Analyze how the NPC-player relationship changed across 6 dimensions (trust, intimacy, dependence, conflict, boundary, commitment). Returns per-dimension delta and overall stability assessment.",
        parameters=[
            ToolParameter("relation_before", "object", "6-dim relation vector before the interaction"),
            ToolParameter("relation_after", "object", "6-dim relation vector after the interaction"),
        ],
        has_side_effects=False,
    ),
    ToolSchema(
        name="retrieve_relevant_lore",
        description="Search the world lore and character background knowledge base for relevant information. Useful for grounding NPC responses in established lore.",
        parameters=[
            ToolParameter("query", "string", "Natural language query about world or character lore"),
            ToolParameter("top_k", "integer", "Number of results to return (1-10)", required=False, default=3),
        ],
        has_side_effects=False,
    ),
    ToolSchema(
        name="generate_dialogue_variations",
        description="Generate multiple variations of an NPC dialogue line in a specified tone. Useful for A/B testing dialogue quality.",
        parameters=[
            ToolParameter("base_dialogue", "string", "Original dialogue line to vary"),
            ToolParameter("tone", "string", "Desired tone: friendly, hostile, sarcastic, neutral, worried, excited", required=False, default="neutral"),
            ToolParameter("count", "integer", "Number of variations to generate (1-5)", required=False, default=3),
        ],
        has_side_effects=False,
    ),
]

# Tool implementation registry
NPC_TOOLS: Dict[str, Callable] = {
    "calculate_emotion_intensity": tool_calculate_emotion_intensity,
    "assess_relationship_change": tool_assess_relationship_change,
    "retrieve_relevant_lore": tool_retrieve_relevant_lore,
    "generate_dialogue_variations": tool_generate_dialogue_variations,
}


def invoke_tool(name: str, **kwargs) -> Dict[str, Any]:
    """
    Invoke a tool by name with validated parameters.
    
    This is the main entry point for Tool Calling in the Agent.
    Includes parameter validation and error handling.
    """
    if name not in NPC_TOOLS:
        return {"error": f"Unknown tool: {name}", "available_tools": list(NPC_TOOLS.keys())}
    
    func = NPC_TOOLS[name]
    try:
        result = func(**kwargs)
        return {"tool": name, "status": "success", "result": result}
    except TypeError as e:
        return {"tool": name, "status": "error", "error": f"Invalid parameters: {e}"}
    except Exception as e:
        return {"tool": name, "status": "error", "error": str(e)}


def get_openai_tool_schemas() -> List[Dict[str, Any]]:
    """Get all tool schemas in OpenAI Function Calling format."""
    return [s.to_openai_schema() for s in NPC_TOOL_SCHEMAS]
