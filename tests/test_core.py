from src.character.npc_agent import SoulNPC
from src.data_generation.generator import generate_samples


def test_npc_process_event():
    npc = SoulNPC()
    result = npc.process_event("player_broke_promise", "I am sorry.")
    data = result.to_dict()
    assert data["emotion"] == "disappointed"
    assert data["action"] == "ask_for_explanation"
    assert data["npc_dialogue"]


def test_generate_samples():
    samples = generate_samples(10)
    assert len(samples) == 10
    assert "event_type" in samples[0]
    assert "sft_prompt" in samples[0]
