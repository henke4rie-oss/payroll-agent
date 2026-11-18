"""A fake model for tests: it plays back scripted turns, so tests need no API key or internet."""

from payroll_agent.llm import ModelTurn, ToolCall
from payroll_agent.schemas import ModelReply


def call(tool_name: str, /, **args) -> ModelTurn:
    """A model turn that asks for one tool. (Positional-only so a tool arg called 'name' can't clash.)"""
    return ModelTurn(text="", tool_calls=[ToolCall(name=tool_name, args=args)])


def final(text: str = "done") -> ModelTurn:
    """A model turn with no tool request (the model is finished)."""
    return ModelTurn(text=text, tool_calls=[])


class FakeLLM:
    def __init__(self, turns, reply=None):
        self.turns = list(turns)
        self.reply = reply or ModelReply(status="answered", message="ok")
        self.step_calls = 0
        self.step_histories = []
        self.step_tool_names = []
        self.history_seen_by_structure = None

    def step(self, history, tool_names=None):
        self.step_calls += 1
        self.step_histories.append(list(history))
        self.step_tool_names.append(tool_names)
        # If the script runs out, keep repeating its last turn (used to test the step limit)
        return self.turns.pop(0) if len(self.turns) > 1 else self.turns[0]

    def structure(self, history):
        self.history_seen_by_structure = history
        return self.reply
