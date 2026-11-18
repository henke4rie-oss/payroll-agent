import inspect
import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Optional, get_args, get_origin

from openai import APIConnectionError, OpenAI

from .schemas import ModelReply

RETRY_CODES = {429, 500, 502, 503, 504}
MAX_ATTEMPTS = 4
DEFAULT_TIMEOUT_SECONDS = 60.0
DEFAULT_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_MODEL = "openai/gpt-oss-20b"


def with_retry(
    make_call: Callable,
    sleep: Callable[[float], None] = time.sleep,
    on_retry: Optional[Callable[[str], None]] = None,
):
    """Run make_call(). On temporary API error, back off and retry."""
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            return make_call()
        except Exception as err:
            status_code = getattr(err, "status_code", None)
            # APITimeoutError subclasses APIConnectionError; neither carries a status code.
            network_error = isinstance(err, APIConnectionError)
            if (status_code not in RETRY_CODES and not network_error) or attempt == MAX_ATTEMPTS:
                raise
            body = getattr(err, "body", None)
            details = f"{body!r} {err}".lower()
            if status_code == 429 and any(
                marker in details
                for marker in ("insufficient_quota", "quota exceeded", "daily limit")
            ):
                raise
            delay = 2 ** attempt
            if on_retry is not None:
                reason = type(err).__name__ if network_error else f"HTTP {status_code}"
                on_retry(
                    f"Transient model API error ({reason}); "
                    f"retrying in {delay}s ({attempt + 1}/{MAX_ATTEMPTS})."
                )
            sleep(delay)


@dataclass
class UserMessage:
    text: str


@dataclass
class ToolCall:
    name: str
    args: dict[str, Any]
    id: Optional[str] = None


@dataclass
class ModelTurn:
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    raw: Any = None
    usage: Optional[tuple[int, int]] = None  # (prompt_tokens, completion_tokens)


@dataclass
class ToolResults:
    results: list[tuple[str, dict]]
    call_ids: list[str] = field(default_factory=list)


def _function_schema(function: Callable) -> dict[str, Any]:
    """Build the OpenAI function schema from a toolbox function's signature."""
    properties: dict[str, dict[str, Any]] = {}
    required = []
    type_names = {str: "string", int: "integer",
                  float: "number", bool: "boolean"}
    for name, parameter in inspect.signature(function).parameters.items():
        parameter_type = parameter.annotation
        if get_origin(parameter_type) is Literal:
            properties[name] = {"type": "string", "enum": list(get_args(parameter_type))}
        elif get_origin(parameter_type) is list and get_origin(get_args(parameter_type)[0]) is Literal:
            properties[name] = {"type": "array",
                                "items": {"type": "string", "enum": list(get_args(get_args(parameter_type)[0]))}}
        elif parameter_type in type_names:
            properties[name] = {"type": type_names[parameter_type]}
        else:
            raise TypeError(
                f"Unsupported type for tool parameter {name!r}: {parameter_type!r}")
        if parameter.default is inspect.Parameter.empty:
            required.append(name)

    return {
        "type": "function",
        "function": {
            "name": function.__name__,
            "description": inspect.getdoc(function) or "",
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            },
        },
    }


class ChatLLM:
    def __init__(
        self,
        tools: list[Callable],
        system_instruction: str,
        model: str = DEFAULT_MODEL,
        client: Optional[OpenAI] = None,
        log: Callable[[str], None] = lambda message: None,
    ):
        api_key = os.environ.get("LLM_API_KEY")
        if client is None and not api_key:
            raise ValueError("Missing LLM_API_KEY.")
        self.client = client if client is not None else OpenAI(
            base_url=os.environ.get("LLM_BASE_URL") or DEFAULT_BASE_URL,
            api_key=api_key,
            max_retries=0,
            # The SDK default is 600s, so a stalled request on a shared free tier hangs for minutes.
            timeout=float(os.environ.get("LLM_TIMEOUT", DEFAULT_TIMEOUT_SECONDS)),
        )
        self.model = model
        self.system_instruction = system_instruction
        self.tools = {function.__name__: _function_schema(function) for function in tools}
        self.log = log

    def _to_messages(self, history: list) -> list[dict[str, Any]]:
        """Convert neutral history into OpenAI-compatible message format."""
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.system_instruction}
        ]
        for item in history:
            if isinstance(item, UserMessage):
                messages.append({"role": "user", "content": item.text})
            elif isinstance(item, ModelTurn):
                if item.raw is not None:
                    messages.append(item.raw)
            elif isinstance(item, ToolResults):
                for index, (name, result) in enumerate(item.results):
                    call_id = item.call_ids[index] if index < len(
                        item.call_ids) else name
                    messages.append({
                        "role": "tool",
                        "tool_call_id": call_id,
                        "content": json.dumps(result, ensure_ascii=False, separators=(",", ":")),
                    })
        return messages

    def step(self, history: list, tool_names: Optional[list[str]] = None) -> ModelTurn:
        """One model call. tool_names limits which tool schemas are sent (None = all)."""
        request: dict[str, Any] = {
            "model": self.model,
            "messages": self._to_messages(history),
            "temperature": 0.2,
            "top_p": 0.7,
            "reasoning_effort": "low",
        }
        names = list(self.tools) if tool_names is None else [n for n in tool_names if n in self.tools]
        if names:
            request["tools"] = [self.tools[n] for n in names]
            request["tool_choice"] = "auto"

        response = with_retry(
            lambda: self.client.chat.completions.create(**request),
            on_retry=self.log,
        )
        usage = getattr(response, "usage", None)
        token_counts = None
        if usage is not None:
            token_counts = (usage.prompt_tokens, usage.completion_tokens)
            self.log(f"tokens: prompt={usage.prompt_tokens} completion={usage.completion_tokens}")
        message = response.choices[0].message
        text = message.content or ""
        calls = []
        raw: dict[str, Any] = {"role": "assistant", "content": message.content}

        if message.tool_calls:
            raw_calls = []
            for index, tool_call in enumerate(message.tool_calls):
                call_id = tool_call.id or f"call_{index}"
                arguments = tool_call.function.arguments
                args = json.loads(arguments) if isinstance(
                    arguments, str) else arguments
                if not isinstance(args, dict):
                    raise ValueError(
                        f"Tool arguments for {tool_call.function.name!r} must be a JSON object.")
                calls.append(
                    ToolCall(name=tool_call.function.name, args=args, id=call_id))
                raw_calls.append({
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": tool_call.function.name,
                        "arguments": json.dumps(args, ensure_ascii=False),
                    },
                })
            raw["tool_calls"] = raw_calls

        return ModelTurn(text=text, tool_calls=calls, raw=raw, usage=token_counts)

    def structure(self, history: list) -> ModelReply:
        """Validate the final model text locally without another API request."""
        final_turn = next(
            (item for item in reversed(history) if isinstance(item, ModelTurn)),
            None,
        )
        if final_turn is None or not final_turn.text.strip():
            raise ValueError("The model returned no final answer.")
        message = final_turn.text.strip()
        status = "needs_clarification" if message.endswith("?") else "answered"
        return ModelReply(status=status, message=message)
