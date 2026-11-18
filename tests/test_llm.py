import json
from types import SimpleNamespace

from payroll_agent.llm import ModelTurn, ChatLLM, ToolCall, ToolResults, UserMessage
from payroll_agent.schemas import ModelReply


def tool(name: str, amount: float) -> dict:
    """Return a sample amount."""
    return {"amount": amount}


class FakeCompletions:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.requests = []

    def create(self, **request):
        self.requests.append(request)
        return next(self.responses)


def response(message):
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def assistant_message(content=None, tool_calls=None):
    return SimpleNamespace(content=content, tool_calls=tool_calls or [])


def test_step_declares_tools_and_preserves_call_ids_for_tool_results():
    tool_call = SimpleNamespace(
        id="call_123",
        function=SimpleNamespace(name="tool", arguments='{"amount": 12.5}'),
    )
    completions = FakeCompletions([response(assistant_message(tool_calls=[tool_call]))])
    llm = ChatLLM([tool], "system prompt", client=SimpleNamespace(
        chat=SimpleNamespace(completions=completions)
    ))

    turn = llm.step([UserMessage("run the tool")])

    assert turn.tool_calls == [ToolCall(name="tool", args={"amount": 12.5}, id="call_123")]
    request = completions.requests[0]
    assert request["model"] == "openai/gpt-oss-20b"
    assert request["reasoning_effort"] == "low"
    assert request["top_p"] == 0.7
    assert request["tools"][0]["function"]["name"] == "tool"
    assert request["messages"] == [
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": "run the tool"},
    ]

    next_messages = llm._to_messages([
        UserMessage("run the tool"),
        turn,
        ToolResults([("tool", {"amount": 12.5})], ["call_123"]),
    ])
    assert next_messages[-2]["tool_calls"][0]["id"] == "call_123"
    assert next_messages[-1] == {
        "role": "tool",
        "tool_call_id": "call_123",
        "content": json.dumps({"amount": 12.5}, separators=(",", ":")),
    }


def test_structure_validates_final_text_without_an_extra_api_request():
    completions = FakeCompletions([])
    llm = ChatLLM([], "system prompt", client=SimpleNamespace(
        chat=SimpleNamespace(completions=completions)
    ))

    result = llm.structure([
        UserMessage("How old are you?"),
        ModelTurn(text="How old are you?"),
    ])

    assert result == ModelReply(status="needs_clarification", message="How old are you?")
    assert completions.requests == []
