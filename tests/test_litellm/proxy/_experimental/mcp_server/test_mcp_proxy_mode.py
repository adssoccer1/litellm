import json
from unittest.mock import AsyncMock, patch

import pytest
from mcp.types import CallToolResult, TextContent, Tool

from litellm.proxy._experimental.mcp_server import server
from litellm.proxy._experimental.mcp_server.faults.list_outcomes import AggregateToolListing
from litellm.proxy._experimental.mcp_server.tool_search import (
    MCP_PROXY_CALL_TOOL_NAME,
    MCP_PROXY_SCHEMA_TOOL_NAME,
    MCP_PROXY_SEARCH_TOOL_NAME,
    handle_mcp_proxy_tool,
    mcp_proxy_tool_id,
)
from litellm.proxy._types import UserAPIKeyAuth

TOOL = Tool.model_validate(
    {
        "name": "math_stdio-add",
        "description": "Add two numbers",
        "inputSchema": {
            "type": "object",
            "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
            "required": ["a", "b"],
        },
        "outputSchema": {"type": "object"},
        "_meta": {"litellm.ai/proxy_tool_identity": {"server_id": "server-1", "tool_name": "math_stdio-add"}},
    }
)
MASTER_KEY = "test-master-key"
AUTH = UserAPIKeyAuth(api_key="key")


def _text(result: CallToolResult) -> object:
    return json.loads(result.content[0].text)


@pytest.mark.asyncio
async def test_proxy_search_returns_opaque_id_and_schema() -> None:
    with (
        patch(
            "litellm.proxy._experimental.mcp_server.server._list_mcp_tools",
            new_callable=AsyncMock,
            return_value=AggregateToolListing(tools=[TOOL], outcomes={}),
        ),
        patch("litellm.proxy.proxy_server.master_key", MASTER_KEY),
    ):
        result = await handle_mcp_proxy_tool(MCP_PROXY_SEARCH_TOOL_NAME, {"query": "add"}, AUTH)

    item = _text(result)[0]
    assert item["tool_id"] == mcp_proxy_tool_id(TOOL)
    assert item["name"] == TOOL.name
    assert "inputSchema" not in item
    assert "outputSchema" not in item
    assert len(item["tool_id"]) == 32


@pytest.mark.asyncio
async def test_proxy_schema_and_call_resolve_current_authorized_catalog() -> None:
    executed = CallToolResult(content=[TextContent(type="text", text="3")], isError=False)
    with (
        patch(
            "litellm.proxy._experimental.mcp_server.server._list_mcp_tools",
            new_callable=AsyncMock,
            return_value=AggregateToolListing(tools=[TOOL], outcomes={}),
        ),
        patch("litellm.proxy.proxy_server.master_key", MASTER_KEY),
        patch(
            "litellm.proxy._experimental.mcp_server.tool_search.handle_mcp_tool_call",
            new_callable=AsyncMock,
            return_value=executed,
        ) as call,
    ):
        schema = await handle_mcp_proxy_tool(
            MCP_PROXY_SCHEMA_TOOL_NAME,
            {"tool_id": mcp_proxy_tool_id(TOOL)},
            AUTH,
        )
        result = await handle_mcp_proxy_tool(
            MCP_PROXY_CALL_TOOL_NAME,
            {"tool_id": mcp_proxy_tool_id(TOOL), "arguments": {"a": 1, "b": 2}},
            AUTH,
        )

    assert _text(schema)["inputSchema"] == TOOL.inputSchema
    assert result is executed
    assert call.await_args.kwargs["tool_name"] == TOOL.name
    assert call.await_args.kwargs["arguments"] == {"a": 1, "b": 2}


@pytest.mark.asyncio
async def test_proxy_rejects_stale_id_and_invalid_arguments_before_dispatch() -> None:
    with (
        patch(
            "litellm.proxy._experimental.mcp_server.server._list_mcp_tools",
            new_callable=AsyncMock,
            return_value=AggregateToolListing(tools=[TOOL], outcomes={}),
        ),
        patch("litellm.proxy.proxy_server.master_key", MASTER_KEY),
        patch(
            "litellm.proxy._experimental.mcp_server.tool_search.handle_mcp_tool_call", new_callable=AsyncMock
        ) as call,
    ):
        stale = await handle_mcp_proxy_tool(MCP_PROXY_SCHEMA_TOOL_NAME, {"tool_id": "stale"}, AUTH)
        invalid = await handle_mcp_proxy_tool(
            MCP_PROXY_CALL_TOOL_NAME,
            {"tool_id": mcp_proxy_tool_id(TOOL), "arguments": "wrong"},
            AUTH,
        )
        falsy = await handle_mcp_proxy_tool(
            MCP_PROXY_CALL_TOOL_NAME,
            {"tool_id": mcp_proxy_tool_id(TOOL), "arguments": False},
            AUTH,
        )
        invalid_schema = await handle_mcp_proxy_tool(
            MCP_PROXY_CALL_TOOL_NAME,
            {"tool_id": mcp_proxy_tool_id(TOOL), "arguments": {"a": "wrong"}},
            AUTH,
        )

    assert stale.isError is True
    assert invalid.isError is True
    assert falsy.isError is True
    assert invalid_schema.isError is True
    call.assert_not_awaited()


@pytest.mark.asyncio
async def test_proxy_call_builds_logging_object() -> None:
    from litellm.proxy._experimental.mcp_server.mcp_context import _mcp_proxy_mode

    sentinel = object()
    result = CallToolResult(content=[TextContent(type="text", text="ok")], isError=False)
    token = _mcp_proxy_mode.set(True)
    try:
        with (
            patch.object(
                server, "_build_virtual_call_logging_obj", new_callable=AsyncMock, return_value=sentinel
            ) as build,
            patch(
                "litellm.proxy._experimental.mcp_server.tool_search.handle_mcp_proxy_tool",
                new_callable=AsyncMock,
                return_value=result,
            ) as handle,
        ):
            actual = await server._dispatch_virtual_mcp_tool(
                name=MCP_PROXY_CALL_TOOL_NAME,
                arguments={"tool_id": "id", "arguments": {}},
                user_api_key_auth=AUTH,
                client_ip=None,
            )
    finally:
        _mcp_proxy_mode.reset(token)

    assert actual is result
    build.assert_awaited_once()
    assert handle.await_args.kwargs["litellm_logging_obj"] is sentinel


@pytest.mark.asyncio
async def test_proxy_list_mode_has_fixed_definitions_without_search_flag() -> None:
    from litellm.proxy._experimental.mcp_server.mcp_context import _mcp_proxy_mode

    token = _mcp_proxy_mode.set(True)
    try:
        with patch(
            "litellm.proxy._experimental.mcp_server.server.get_or_extract_auth_context",
            new_callable=AsyncMock,
            return_value=(AUTH, None, None, None, None, None, None),
        ):
            tools = await server.handle_list_tools()
    finally:
        _mcp_proxy_mode.reset(token)

    assert {tool.name for tool in tools} == {
        MCP_PROXY_SEARCH_TOOL_NAME,
        MCP_PROXY_SCHEMA_TOOL_NAME,
        MCP_PROXY_CALL_TOOL_NAME,
    }
