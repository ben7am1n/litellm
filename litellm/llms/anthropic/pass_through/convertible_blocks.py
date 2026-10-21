import json
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from functools import partial, reduce
from itertools import chain, groupby
from operator import itemgetter
from types import MappingProxyType
from typing import Final, TypeAlias

from pydantic import JsonValue, TypeAdapter, ValidationError
from typing_extensions import assert_never

import litellm
from litellm.litellm_core_utils.core_helpers import as_str_mapping, drops_unsupported_params
from litellm.litellm_core_utils.prompt_templates.factory import sanitize_anthropic_tool_use_id
from litellm.llms.anthropic.pass_through.messages.utils import normalize_native_messages
from litellm.types.utils import TokenCountMessagesFormat

_CONVERTIBLE_BLOCK_TYPES: Final = MappingProxyType(
    {
        "user": frozenset({"text", "image", "document", "tool_result"}),
        "assistant": frozenset({"text", "tool_use", "thinking", "redacted_thinking"}),
    }
)
_SOURCED_BLOCK_TYPES: Final = frozenset({"image", "document"})
_TOOL_RESULT_BLOCK_TYPES: Final = frozenset({"text", "image", "document"})


_JSON_MESSAGES: Final = TypeAdapter(tuple[Mapping[str, JsonValue], ...])


def _json_messages(messages: Sequence[Mapping[str, object]]) -> tuple[Mapping[str, JsonValue], ...]:
    try:
        return _JSON_MESSAGES.validate_python(messages)
    except ValidationError:
        return ()


def _source_label(block_type: str, source: JsonValue) -> str | None:
    if not isinstance(source, dict):
        return f"{block_type} without a source"
    match source.get("type"):
        case "base64" if source.get("data"):
            return None
        case "url" if source.get("url"):
            return None
        case "base64" | "url" as source_type:
            return f"{block_type} with an empty {source_type} source"
        case str() as source_type:
            return f"{block_type} with a {source_type} source"
        case _:
            return f"{block_type} with an untyped source"


def _block_label(block: JsonValue, convertible: frozenset[str]) -> str | None:
    if not isinstance(block, dict):
        return None
    block_type: Final = block.get("type")
    if not isinstance(block_type, str):
        return "a block without a type"
    if block_type not in convertible:
        return block_type
    return _source_label(block_type, block.get("source")) if block_type in _SOURCED_BLOCK_TYPES else None


def _tool_result_labels(block: JsonValue) -> Iterator[str]:
    if not isinstance(block, dict) or block.get("type") != "tool_result":
        return iter(())
    inner: Final = block.get("content")
    if not isinstance(inner, list):
        return iter(())
    return (
        f"{label} inside a tool_result"
        for item in inner
        if (label := _block_label(item, _TOOL_RESULT_BLOCK_TYPES)) is not None
    )


def _content_labels(block: JsonValue, convertible: frozenset[str]) -> Iterator[str]:
    label: Final = _block_label(block, convertible)
    return chain(() if label is None else (label,), _tool_result_labels(block))


def _message_labels(message: Mapping[str, JsonValue]) -> Iterator[str]:
    role: Final = message.get("role")
    convertible: Final = _CONVERTIBLE_BLOCK_TYPES.get(role) if isinstance(role, str) else None
    content: Final = message.get("content")
    if convertible is None or not isinstance(content, list):
        return iter(())
    return chain.from_iterable(_content_labels(block, convertible) for block in content)


def unconvertible_anthropic_blocks(messages: Sequence[Mapping[str, object]]) -> tuple[str, ...]:
    normalized: Final = _json_messages(normalize_native_messages(messages))
    return tuple(sorted(set(chain.from_iterable(_message_labels(message) for message in normalized))))


def unconvertible_blocks_error(
    messages: Sequence[Mapping[str, object]] | None, messages_format: TokenCountMessagesFormat
) -> str | None:
    match messages_format:
        case "openai":
            return None
        case "anthropic":
            labels: Final = unconvertible_anthropic_blocks(messages or ())
            return (
                "count_tokens can't convert these content blocks for this model, so the count would leave them out: "
                + ", ".join(labels)
                if labels
                else None
            )
        case _:
            assert_never(messages_format)


_OBJECTS: Final = TypeAdapter(tuple[object, ...])


def _list_items(value: object) -> tuple[object, ...]:
    return _OBJECTS.validate_python(value) if isinstance(value, list) else ()


@dataclass(frozen=True, slots=True)
class _KeptPart:
    block: object
    scanned_text: str


_KeptParts: TypeAlias = Mapping[int, _KeptPart]
_NO_KEPT_PARTS: Final[_KeptParts] = MappingProxyType({})


def _scannable_text(text: object) -> str:
    return json.dumps(text, default=str, ensure_ascii=False)


def _non_str_text(block: object) -> _KeptPart | None:
    match as_str_mapping(block):
        case {"type": "text", "text": str()}:
            return None
        case {"type": "text", "text": text}:
            return _KeptPart(block=block, scanned_text=_scannable_text(text))
        case _:
            return None


def tool_result_text_part(block: object) -> str | None:
    match block:
        case {"type": "text", "text": str(text)}:
            return text
        case {"type": "text", "text": object()}:
            return None
        case {"type": "text"}:
            return ""
        case _:
            return None


def tool_result_text_parts(tool_result: Mapping[str, object]) -> tuple[str, ...]:
    parts: Final = map(tool_result_text_part, _list_items(tool_result.get("content")))
    return tuple(part for part in parts if part is not None)


def _user_content_blocks(message: object) -> tuple[object, ...]:
    match as_str_mapping(message):
        case {"role": "user", "content": content}:
            return _list_items(content)
        case _:
            return ()


def _with_user_blocks(message: object, transform: Callable[[object], object]) -> object:
    match as_str_mapping(message):
        case {"role": "user", "content": list()} as user_message:
            return {**user_message, "content": [transform(block) for block in _user_content_blocks(user_message)]}
        case _:
            return message


def _non_string_text_errors(label: str, content: object) -> Iterator[str]:
    return (
        f"{label} content.{index}.text: Input should be a valid string"
        for index, text_block in enumerate(_list_items(content))
        if _non_str_text(text_block) is not None
    )


def _tool_result_text_errors(block: object) -> Iterator[str]:
    match block:
        case {"type": "tool_result", "tool_use_id": object() as tool_use_id, "content": object() as content}:
            return _non_string_text_errors(f"tool_result {tool_use_id}", content)
        case {"type": "tool_result", "content": object() as content}:
            return _non_string_text_errors("tool_result", content)
        case _:
            return iter(())


def malformed_tool_result_text_error(messages: Iterable[object]) -> str | None:
    blocks: Final = chain.from_iterable(map(_user_content_blocks, messages))
    return next(chain.from_iterable(map(_tool_result_text_errors, blocks)), None)


def malformed_tool_result_count_error(
    messages: Iterable[Mapping[str, object]] | None,
    messages_format: TokenCountMessagesFormat,
    deployment: Mapping[str, object] | None,
) -> str | None:
    if messages_format != "anthropic" or deployment_drops_params(deployment):
        return None
    return malformed_tool_result_text_error(messages or ())


@dataclass(frozen=True, slots=True)
class _ToolChangeBlockState:
    tools: tuple[Mapping[str, object], ...]
    blocks: tuple[object, ...]
    changed: bool


@dataclass(frozen=True, slots=True)
class _ToolChangeMessageState:
    tools: tuple[Mapping[str, object], ...]
    messages: tuple[dict[str, object], ...]
    changed: bool


def _tool_change_name(value: Mapping[str, object]) -> str | None:
    return next(
        (candidate for key in ("name", "tool_name") if isinstance(candidate := value.get(key), str) and candidate),
        None,
    )


def _tool_addition_definition(block: Mapping[str, object]) -> Mapping[str, object] | None:
    tool: Final = block.get("tool")
    if isinstance(tool, Mapping):
        if tool.get("type") == "tool_definition":
            definition: Final = tool.get("definition")
            if isinstance(definition, Mapping):
                return definition
        if "input_schema" in tool or "parameters" in tool:
            return tool
    definition = block.get("definition")
    return definition if isinstance(definition, Mapping) else None


def _tool_removal_name(block: Mapping[str, object]) -> str | None:
    direct_name: Final = _tool_change_name(block)
    if direct_name is not None:
        return direct_name
    tool: Final = block.get("tool")
    if isinstance(tool, str):
        return tool
    return _tool_change_name(tool) if isinstance(tool, Mapping) else None


def _apply_tool_change_block(state: _ToolChangeBlockState, block: object) -> _ToolChangeBlockState:
    block_map: Final = as_str_mapping(block)
    if block_map is None or block_map.get("type") not in ("tool_addition", "tool_removal"):
        return _ToolChangeBlockState(state.tools, (*state.blocks, block), state.changed)
    if block_map.get("type") == "tool_addition":
        tool: Final = _tool_addition_definition(block_map)
        tool_name: Final = None if tool is None else _tool_change_name(tool)
        if tool is None or tool_name is None or any(_tool_change_name(existing) == tool_name for existing in state.tools):
            return _ToolChangeBlockState(state.tools, state.blocks, True)
        return _ToolChangeBlockState((*state.tools, tool), state.blocks, True)
    tool_name = _tool_removal_name(block_map)
    remaining_tools: Final = (
        state.tools
        if tool_name is None
        else tuple(existing for existing in state.tools if _tool_change_name(existing) != tool_name)
    )
    return _ToolChangeBlockState(remaining_tools, state.blocks, True)


def _apply_anthropic_tool_change_message(
    state: _ToolChangeMessageState, message: dict[str, object]
) -> _ToolChangeMessageState:
    if message.get("role") != "system" or not isinstance(message.get("content"), list):
        return _ToolChangeMessageState(state.tools, (*state.messages, message), state.changed)
    blocks: Final = tuple(message["content"])
    block_state: Final = reduce(
        _apply_tool_change_block,
        blocks,
        _ToolChangeBlockState(state.tools, (), False),
    )
    if not block_state.changed:
        return _ToolChangeMessageState(state.tools, (*state.messages, message), state.changed)
    if not block_state.blocks:
        return _ToolChangeMessageState(block_state.tools, state.messages, True)
    return _ToolChangeMessageState(
        block_state.tools,
        (*state.messages, {**message, "content": list(block_state.blocks)}),
        True,
    )


def apply_anthropic_tool_changes(
    messages: Sequence[dict[str, object]] | None,
    tools: Sequence[Mapping[str, object]] | None,
) -> tuple[Sequence[dict[str, object]] | None, Sequence[Mapping[str, object]] | None]:
    if messages is None:
        return None, tools
    initial_tools: Final = tuple(tool for tool in (tools or ()) if isinstance(tool, Mapping))
    result: Final = reduce(
        _apply_anthropic_tool_change_message,
        messages,
        _ToolChangeMessageState(initial_tools, (), False),
    )
    counted_tools: Final = result.tools if result.changed or tools is not None else None
    return result.messages, counted_tools


def _scannable_part(part: object) -> object:
    match as_str_mapping(part):
        case {"type": "text", "text": str()}:
            return part
        case {"type": "text", "text": text} as text_block:
            return {**text_block, "text": _scannable_text(text)}
        case _:
            return part


def _scannable_block(block: object) -> object:
    match as_str_mapping(block):
        case {"type": "tool_result", "content": list()} as tool_result:
            return {
                **tool_result,
                "content": [_scannable_part(part) for part in _list_items(tool_result.get("content"))],
            }
        case _:
            return block


def _scannable_messages(messages: object) -> object:
    items: Final = _list_items(messages)
    return [_with_user_blocks(message, _scannable_block) for message in items] if items else messages


def scannable_tool_result_request(request: object) -> object:
    match as_str_mapping(request):
        case {"messages": messages} as request_fields:
            return {**request_fields, "messages": _scannable_messages(messages)}
        case None:
            return request
        case request_fields:
            return {**request_fields}


def _non_string_text_parts(content: object) -> _KeptParts:
    return MappingProxyType(
        {index: part for index, block in enumerate(_list_items(content)) if (part := _non_str_text(block)) is not None}
    )


def _tool_result_non_string_texts(block: object) -> tuple[tuple[str, _KeptParts], ...]:
    match as_str_mapping(block):
        case {"type": "tool_result", "tool_use_id": str(tool_use_id)} as tool_result:
            return ((sanitize_anthropic_tool_use_id(tool_use_id), _non_string_text_parts(tool_result.get("content"))),)
        case _:
            return ()


def _content_blocks(content: object) -> tuple[object, ...]:
    match content:
        case str() if content:
            return ({"type": "text", "text": content},)
        case _:
            return _list_items(content)


def _restored_part(block: object, kept: _KeptPart | None) -> object:
    return kept.block if kept is not None and tool_result_text_part(block) == kept.scanned_text else block


def _with_kept_parts(block: object, tool_result: Mapping[str, object], parts: _KeptParts) -> object:
    if not parts:
        return block
    blocks: Final = _content_blocks(tool_result.get("content"))
    return {**tool_result, "content": [_restored_part(part, parts.get(index)) for index, part in enumerate(blocks)]}


def _restored_block(block: object, kept: Mapping[str, Iterator[_KeptParts]]) -> object:
    match as_str_mapping(block):
        case {"type": "tool_result", "tool_use_id": str(tool_use_id)} as tool_result if (
            sanitize_anthropic_tool_use_id(tool_use_id) in kept
        ):
            return _with_kept_parts(
                block, tool_result, next(kept[sanitize_anthropic_tool_use_id(tool_use_id)], _NO_KEPT_PARTS)
            )
        case _:
            return block


def keep_non_string_tool_result_texts(rewritten: Sequence[object], original: object) -> Sequence[object]:
    blocks: Final = chain.from_iterable(map(_user_content_blocks, _list_items(original)))
    by_id: Final = tuple(sorted(chain.from_iterable(map(_tool_result_non_string_texts, blocks)), key=itemgetter(0)))
    if not any(parts for _, parts in by_id):
        return rewritten
    kept_in_order: Final = MappingProxyType(
        {
            tool_use_id: iter(tuple(map(itemgetter(1), group)))
            for tool_use_id, group in groupby(by_id, key=itemgetter(0))
        }
    )
    return [_with_user_blocks(message, partial(_restored_block, kept=kept_in_order)) for message in rewritten]


def _without_non_string_text(block: object) -> object:
    match as_str_mapping(block):
        case {"type": "tool_result", "content": list()} as tool_result:
            parts: Final = _list_items(tool_result.get("content"))
            return {**tool_result, "content": [part for part in parts if _non_str_text(part) is None]}
        case _:
            return block


def _without_non_string_texts(
    message: dict[str, object],  # mutable-ok: AnthropicMessagesRequest.messages is typed list[dict]
) -> dict[str, object]:  # mutable-ok: AnthropicMessagesRequest.messages is typed list[dict]
    match message:
        case {"role": "user", "content": list()}:
            return {**message, "content": [_without_non_string_text(block) for block in _user_content_blocks(message)]}
        case _:
            return message


def drop_non_string_tool_result_texts(
    messages: Sequence[dict[str, object]],  # mutable-ok: AnthropicMessagesRequest.messages is typed list[dict]
    drop_params: object,
) -> Sequence[dict[str, object]]:  # mutable-ok: AnthropicMessagesRequest.messages is typed list[dict]
    if not drops_unsupported_params(drop_params) or malformed_tool_result_text_error(messages) is None:
        return messages
    return [_without_non_string_texts(message) for message in messages]


def deployment_drops_params(deployment: Mapping[str, object] | None) -> bool:
    match deployment:
        case {"litellm_params": {"drop_params": object() as drop_params}}:
            return drops_unsupported_params(drop_params)
        case _:
            return drops_unsupported_params(None)


def reject_malformed_tool_result_text(
    messages: Iterable[Mapping[str, object]], model: str, forwarded_kwargs: Mapping[str, object]
) -> None:
    if drops_unsupported_params(forwarded_kwargs.get("drop_params")):
        return
    error: Final = malformed_tool_result_text_error(messages)
    if error is None:
        return
    provider: Final = forwarded_kwargs.get("custom_llm_provider")
    raise litellm.BadRequestError(
        message=error, model=model, llm_provider=provider if isinstance(provider, str) else ""
    )
