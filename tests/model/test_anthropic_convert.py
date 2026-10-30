"""Tests for Anthropic model API conversion functions."""

import json
from typing import Any, cast

import pytest
from anthropic.types import (
    AssistantMessageParam,
    CitationsConfigParam,
    DocumentBlockParam,
    Message,
    MessageParam,
    PlainTextSourceParam,
    TextBlock,
    ThinkingBlock,
    ToolResultBlockParam,
    ToolUseBlock,
    Usage,
    UserMessageParam,
)
from pydantic import TypeAdapter

from inspect_ai._util.content import (
    ContentDocument,
    ContentReasoning,
    ContentText,
    ContentToolUse,
)
from inspect_ai.model import ChatMessage, model_output_from_anthropic
from inspect_ai.model._assistant_internal import init_sample_assistant_internal
from inspect_ai.model._chat_message import ChatMessageAssistant
from inspect_ai.model._compaction.edit import TOOL_RESULT_REMOVED
from inspect_ai.model._model_output import ModelOutput
from inspect_ai.model._providers.anthropic import (
    assistant_message_block_params,
    message_block_params,
    normalize_document_citations,
)


async def test_model_output_from_anthropic_basic() -> None:
    """Test basic Message conversion to ModelOutput."""
    message = Message(
        id="msg_123",
        model="claude-3-5-sonnet-20241022",
        role="assistant",
        content=[TextBlock(type="text", text="Hello! How can I help you today?")],
        type="message",
        stop_reason="end_turn",
        stop_sequence=None,
        usage=Usage(
            input_tokens=10,
            output_tokens=20,
        ),
    )

    result = await model_output_from_anthropic(message)

    assert isinstance(result, ModelOutput)
    assert result.model == "claude-3-5-sonnet-20241022"
    assert len(result.choices) == 1
    assert isinstance(result.choices[0].message, ChatMessageAssistant)
    assert result.choices[0].stop_reason == "stop"
    assert result.usage is not None
    assert result.usage.input_tokens == 10
    assert result.usage.output_tokens == 20
    assert result.usage.total_tokens == 30

    # Check content
    message_obj = result.choices[0].message
    assert isinstance(message_obj.content, list)
    assert len(message_obj.content) == 1
    assert isinstance(message_obj.content[0], ContentText)
    assert message_obj.content[0].text == "Hello! How can I help you today?"


async def test_message_block_params_enables_document_citations() -> None:
    document = ContentDocument(
        document="data:text/plain;base64,SGVsbG8=",
        citations=True,
    )

    document_block = (await message_block_params(document))[0]

    assert document_block["type"] == "document"
    assert document_block.get("citations") == {"enabled": True}


async def test_message_block_params_omits_document_citations_by_default() -> None:
    document = ContentDocument(document="data:text/plain;base64,SGVsbG8=")

    document_block = (await message_block_params(document))[0]

    assert document_block["type"] == "document"
    assert "citations" not in document_block


async def test_message_block_params_omits_citations_for_image_documents() -> None:
    document = ContentDocument(
        document="data:image/png;base64,iVBORw0KGgo=",
        citations=True,
    )

    document_block = (await message_block_params(document))[0]

    assert document_block["type"] == "document"
    assert "citations" not in document_block


def test_normalize_document_citations_enables_all_history_documents() -> None:
    first_document = DocumentBlockParam(
        type="document",
        source=PlainTextSourceParam(type="text", media_type="text/plain", data="Hello"),
    )
    last_document = DocumentBlockParam(
        type="document",
        source=PlainTextSourceParam(type="text", media_type="text/plain", data="World"),
        citations=CitationsConfigParam(enabled=True),
    )
    messages: list[MessageParam] = [
        UserMessageParam(role="user", content=[first_document]),
        AssistantMessageParam(role="assistant", content="Acknowledged."),
        UserMessageParam(role="user", content=[last_document]),
    ]

    normalize_document_citations(messages)

    assert first_document.get("citations") == {"enabled": True}
    assert last_document.get("citations") == {"enabled": True}


def test_normalize_document_citations_enables_tool_result_documents() -> None:
    history_document = DocumentBlockParam(
        type="document",
        source=PlainTextSourceParam(type="text", media_type="text/plain", data="Hello"),
    )
    tool_document = DocumentBlockParam(
        type="document",
        source=PlainTextSourceParam(type="text", media_type="text/plain", data="World"),
        citations=CitationsConfigParam(enabled=True),
    )
    tool_result = ToolResultBlockParam(
        type="tool_result",
        tool_use_id="tool-1",
        content=[tool_document],
    )
    messages: list[MessageParam] = [
        UserMessageParam(role="user", content=[history_document]),
        UserMessageParam(role="user", content=[tool_result]),
    ]

    normalize_document_citations(messages)

    assert history_document.get("citations") == {"enabled": True}
    assert tool_document.get("citations") == {"enabled": True}


async def test_model_output_from_anthropic_with_tool_use() -> None:
    """Test Message with tool use conversion."""
    message = Message(
        id="msg_456",
        model="claude-3-5-sonnet-20241022",
        role="assistant",
        content=[
            TextBlock(type="text", text="Let me check the weather for you."),
            ToolUseBlock(
                type="tool_use",
                id="toolu_123",
                name="get_weather",
                input={"location": "San Francisco"},
            ),
        ],
        type="message",
        stop_reason="tool_use",
        stop_sequence=None,
        usage=Usage(
            input_tokens=15,
            output_tokens=25,
        ),
    )

    result = await model_output_from_anthropic(message)

    assert isinstance(result, ModelOutput)
    assert result.choices[0].stop_reason == "tool_calls"

    message_obj = result.choices[0].message
    assert isinstance(message_obj, ChatMessageAssistant)
    assert isinstance(message_obj.content, list)
    assert len(message_obj.content) == 1

    # Check text content
    assert isinstance(message_obj.content[0], ContentText)
    assert message_obj.content[0].text == "Let me check the weather for you."

    # Check tool calls are stored separately
    assert message_obj.tool_calls is not None
    assert len(message_obj.tool_calls) == 1
    assert message_obj.tool_calls[0].id == "toolu_123"
    assert message_obj.tool_calls[0].function == "get_weather"
    assert message_obj.tool_calls[0].arguments == {"location": "San Francisco"}


async def test_model_output_from_anthropic_with_thinking() -> None:
    """Test Message with thinking blocks (reasoning) conversion."""
    message = Message(
        id="msg_789",
        model="claude-sonnet-4-6",
        role="assistant",
        content=[
            ThinkingBlock(
                type="thinking",
                thinking="Let me carefully consider this problem...",
                signature="thinking-sig-1",
            ),
            TextBlock(type="text", text="The answer is 42."),
        ],
        type="message",
        stop_reason="end_turn",
        stop_sequence=None,
        usage=Usage(
            input_tokens=50,
            output_tokens=150,
        ),
    )

    result = await model_output_from_anthropic(message)

    assert isinstance(result, ModelOutput)
    message_obj = result.choices[0].message
    assert isinstance(message_obj, ChatMessageAssistant)
    assert isinstance(message_obj.content, list)
    assert len(message_obj.content) == 2

    # Check reasoning content
    assert isinstance(message_obj.content[0], ContentReasoning)
    assert "carefully consider" in (message_obj.content[0].summary or "")

    # Check text content
    assert isinstance(message_obj.content[1], ContentText)
    assert message_obj.content[1].text == "The answer is 42."


async def test_model_output_from_anthropic_dict_input() -> None:
    """Test conversion from dict representation of Message."""
    message_dict = {
        "id": "msg_dict",
        "model": "claude-3-5-sonnet-20241022",
        "role": "assistant",
        "content": [{"type": "text", "text": "Dict test response"}],
        "type": "message",
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {
            "input_tokens": 5,
            "output_tokens": 10,
        },
    }

    result = await model_output_from_anthropic(message_dict)

    assert isinstance(result, ModelOutput)
    assert result.model == "claude-3-5-sonnet-20241022"
    message_obj = result.choices[0].message
    assert isinstance(message_obj.content, list)
    assert len(message_obj.content) == 1
    assert isinstance(message_obj.content[0], ContentText)
    assert message_obj.content[0].text == "Dict test response"


async def test_model_output_from_anthropic_max_tokens_stop_reason() -> None:
    """Test Message with max_tokens stop reason."""
    message = Message(
        id="msg_max",
        model="claude-3-5-sonnet-20241022",
        role="assistant",
        content=[
            TextBlock(type="text", text="This response was cut off due to token limit")
        ],
        type="message",
        stop_reason="max_tokens",
        stop_sequence=None,
        usage=Usage(
            input_tokens=10,
            output_tokens=100,
        ),
    )

    result = await model_output_from_anthropic(message)

    assert isinstance(result, ModelOutput)
    assert result.choices[0].stop_reason == "max_tokens"


async def test_model_output_from_anthropic_stop_sequence() -> None:
    """Test Message with stop_sequence stop reason."""
    message = Message(
        id="msg_stop_seq",
        model="claude-3-5-sonnet-20241022",
        role="assistant",
        content=[TextBlock(type="text", text="Response stopped early")],
        type="message",
        stop_reason="stop_sequence",
        stop_sequence="STOP",
        usage=Usage(
            input_tokens=10,
            output_tokens=15,
        ),
    )

    result = await model_output_from_anthropic(message)

    assert isinstance(result, ModelOutput)
    assert result.choices[0].stop_reason == "stop"


async def test_model_output_from_anthropic_multiple_thinking_blocks() -> None:
    """Test Message with multiple thinking blocks."""
    message = Message(
        id="msg_multi_think",
        model="claude-sonnet-4-6",
        role="assistant",
        content=[
            ThinkingBlock(
                type="thinking",
                thinking="First reasoning step...",
                signature="thinking-sig-1",
            ),
            ThinkingBlock(
                type="thinking",
                thinking="Second reasoning step...",
                signature="thinking-sig-2",
            ),
            TextBlock(type="text", text="Final answer based on reasoning."),
        ],
        type="message",
        stop_reason="end_turn",
        stop_sequence=None,
        usage=Usage(
            input_tokens=30,
            output_tokens=80,
        ),
    )

    result = await model_output_from_anthropic(message)

    assert isinstance(result, ModelOutput)
    message_obj = result.choices[0].message
    assert isinstance(message_obj.content, list)
    assert len(message_obj.content) == 3

    # Check both reasoning blocks
    assert isinstance(message_obj.content[0], ContentReasoning)
    assert "First reasoning step" in (message_obj.content[0].summary or "")

    assert isinstance(message_obj.content[1], ContentReasoning)
    assert "Second reasoning step" in (message_obj.content[1].summary or "")

    # Check text content
    assert isinstance(message_obj.content[2], ContentText)
    assert message_obj.content[2].text == "Final answer based on reasoning."


async def test_model_output_from_anthropic_mixed_content() -> None:
    """Test Message with mixed content types."""
    message = Message(
        id="msg_mixed",
        model="claude-3-5-sonnet-20241022",
        role="assistant",
        content=[
            TextBlock(type="text", text="I'll help with that."),
            ToolUseBlock(
                type="tool_use",
                id="toolu_456",
                name="search",
                input={"query": "test"},
            ),
            TextBlock(type="text", text="Based on the results..."),
        ],
        type="message",
        stop_reason="tool_use",
        stop_sequence=None,
        usage=Usage(
            input_tokens=20,
            output_tokens=40,
        ),
    )

    result = await model_output_from_anthropic(message)

    assert isinstance(result, ModelOutput)
    message_obj = result.choices[0].message
    assert isinstance(message_obj.content, list)
    assert len(message_obj.content) == 2

    # Verify content types in order (tool use is moved to tool_calls)
    assert isinstance(message_obj.content[0], ContentText)
    assert message_obj.content[0].text == "I'll help with that."
    assert isinstance(message_obj.content[1], ContentText)
    assert message_obj.content[1].text == "Based on the results..."

    # Verify tool call is stored separately
    assert message_obj.tool_calls is not None
    assert len(message_obj.tool_calls) == 1
    assert message_obj.tool_calls[0].function == "search"


# --- server tool rebuild from a log round-trip --------------------------------
#
# When assistant messages are restored from a log into a new sample, the
# recorded server tool spans are gone and the web_search / web_fetch blocks are
# rebuilt from the logged ContentToolUse items. The rebuild must reproduce the
# blocks that the in-process recorded state replays.


def _server_tool_use(id: str, name: str, input: dict[str, Any]) -> dict[str, Any]:
    return {"type": "server_tool_use", "id": id, "name": name, "input": input}


def _web_search_result(id: str, url: str) -> dict[str, Any]:
    return {
        "type": "web_search_tool_result",
        "tool_use_id": id,
        "content": [
            {
                "type": "web_search_result",
                "title": "Example",
                "url": url,
                "encrypted_content": f"FAKE_ENCRYPTED_{id}",
                "page_age": "1 day ago",
            }
        ],
    }


def _thinking(signature: str) -> dict[str, Any]:
    return {"type": "thinking", "thinking": "Let me look.", "signature": signature}


WEB_SEARCH_BLOCKS = [
    _server_tool_use("srvtoolu_ws1", "web_search", {"query": "example"}),
    _web_search_result("srvtoolu_ws1", "https://example.com/a"),
]

WEB_SEARCH_ERROR_BLOCKS = [
    _server_tool_use("srvtoolu_ws2", "web_search", {"query": "example"}),
    {
        "type": "web_search_tool_result",
        "tool_use_id": "srvtoolu_ws2",
        "content": {
            "type": "web_search_tool_result_error",
            "error_code": "max_uses_exceeded",
        },
    },
]

WEB_FETCH_BLOCKS = [
    _server_tool_use("srvtoolu_wf1", "web_fetch", {"url": "https://example.com/b"}),
    {
        "type": "web_fetch_tool_result",
        "tool_use_id": "srvtoolu_wf1",
        "content": {
            "type": "web_fetch_result",
            "url": "https://example.com/b",
            "retrieved_at": "2026-01-01T00:00:00Z",
            "content": {
                "type": "document",
                "source": {
                    "type": "text",
                    "media_type": "text/plain",
                    "data": "Example page text.",
                },
                "title": "Example B",
            },
        },
    },
]

WEB_FETCH_ERROR_BLOCKS = [
    _server_tool_use("srvtoolu_wf2", "web_fetch", {"url": "https://example.com/c"}),
    {
        "type": "web_fetch_tool_result",
        "tool_use_id": "srvtoolu_wf2",
        "content": {
            "type": "web_fetch_tool_result_error",
            "error_code": "url_not_accessible",
        },
    },
]

INTERLEAVED_BLOCKS = [
    _thinking("FAKE_SIG_1"),
    _server_tool_use("srvtoolu_ws3", "web_search", {"query": "first"}),
    _web_search_result("srvtoolu_ws3", "https://example.com/1"),
    _thinking("FAKE_SIG_2"),
    _server_tool_use("srvtoolu_ws4", "web_search", {"query": "second"}),
    _web_search_result("srvtoolu_ws4", "https://example.com/2"),
    *WEB_FETCH_BLOCKS,
    _thinking("FAKE_SIG_3"),
    {"type": "text", "text": "Here is what I found."},
]


async def _live_and_log_round_trip_block_params(
    content: list[dict[str, Any]], clear_results: bool = False
) -> tuple[list[Any], list[Any]]:
    """Block params replayed from recorded state and rebuilt after a log round-trip."""
    init_sample_assistant_internal()
    output = await model_output_from_anthropic(
        {
            "id": "msg_replay",
            "type": "message",
            "role": "assistant",
            "model": "claude-sonnet-4-6",
            "content": content,
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }
    )
    if clear_results:
        for c in output.message.content:
            if isinstance(c, ContentToolUse):
                c.result = TOOL_RESULT_REMOVED
    live = list(await assistant_message_block_params(output.message))

    # log round-trip into a new sample (no recorded assistant internal state)
    adapter = TypeAdapter(list[ChatMessage])
    [restored] = adapter.validate_json(adapter.dump_json([output.message]))
    assert isinstance(restored, ChatMessageAssistant)
    init_sample_assistant_internal()
    rebuilt = list(await assistant_message_block_params(restored))
    return live, rebuilt


@pytest.mark.parametrize(
    "content",
    [
        pytest.param([*WEB_SEARCH_BLOCKS, {"type": "text", "text": "ok"}], id="search"),
        pytest.param(
            [*WEB_SEARCH_ERROR_BLOCKS, {"type": "text", "text": "ok"}],
            id="search_error",
        ),
        pytest.param([*WEB_FETCH_BLOCKS, {"type": "text", "text": "ok"}], id="fetch"),
        pytest.param(
            [*WEB_FETCH_ERROR_BLOCKS, {"type": "text", "text": "ok"}],
            id="fetch_error",
        ),
        pytest.param(INTERLEAVED_BLOCKS, id="interleaved_thinking"),
    ],
)
async def test_server_tool_blocks_rebuilt_from_log_match_recorded(
    content: list[dict[str, Any]],
) -> None:
    live, rebuilt = await _live_and_log_round_trip_block_params(content)
    # the recorded state replays the wire blocks verbatim (text gains citations)
    assert [{k: v for k, v in b.items() if k != "citations"} for b in live] == content
    # the rebuild matches it and serializes (a lazy iterator or a malformed
    # result fails to serialize or is sent empty)
    assert json.loads(json.dumps(rebuilt)) == json.loads(json.dumps(live))


@pytest.mark.parametrize(
    "content",
    [
        pytest.param(WEB_SEARCH_BLOCKS, id="search"),
        pytest.param(WEB_FETCH_BLOCKS, id="fetch"),
    ],
)
async def test_cleared_server_tool_blocks_rebuilt_from_log_match_recorded(
    content: list[dict[str, Any]],
) -> None:
    live, rebuilt = await _live_and_log_round_trip_block_params(
        content, clear_results=True
    )
    assert [b["type"] for b in live] == [b["type"] for b in content]
    assert rebuilt == live


@pytest.mark.parametrize(
    "blocks", [WEB_FETCH_BLOCKS, WEB_FETCH_ERROR_BLOCKS], ids=["fetch", "fetch_error"]
)
async def test_web_fetch_rebuilt_from_pre_0_3_180_log(
    blocks: list[dict[str, Any]],
) -> None:
    """Older logs keep only the fetch result's inner content."""
    use, result = blocks
    logged = ContentToolUse(
        tool_type="web_search",
        id=use["id"],
        name="web_fetch",
        arguments=json.dumps(use["input"]),
        result=json.dumps(result["content"]),
    )
    init_sample_assistant_internal()
    assert await message_block_params(logged) == [use, result]


async def test_unreadable_web_fetch_result_replayed_as_unavailable(
    caplog: pytest.LogCaptureFixture,
) -> None:
    logged = ContentToolUse(
        tool_type="web_search",
        id="srvtoolu_bad",
        name="web_fetch",
        arguments=json.dumps({"url": "https://example.com/d"}),
        result="not json",
    )
    init_sample_assistant_internal()
    _, result = await message_block_params(logged)
    assert cast(dict[str, Any], result)["content"] == {
        "type": "web_fetch_tool_result_error",
        "error_code": "unavailable",
    }
    assert "srvtoolu_bad" in caplog.text
