from __future__ import annotations

import asyncio
import json
from typing import Any

import structlog
from fastapi import Request
from fastapi.responses import StreamingResponse

logger = structlog.get_logger()


class MCPServer:
    def __init__(self) -> None:
        self._tools: list[dict[str, Any]] = []
        self._resources: list[dict[str, Any]] = []
        logger.info("mcp_server_initialized")

    def register_tool(
        self, name: str, description: str, handler: Any, schema: dict[str, Any]
    ) -> None:
        self._tools.append(
            {
                "name": name,
                "description": description,
                "handler": handler,
                "inputSchema": schema,
            }
        )
        logger.info("mcp_tool_registered", name=name)

    def register_resource(self, uri: str, name: str, mime_type: str, handler: Any) -> None:
        self._resources.append(
            {
                "uri": uri,
                "name": name,
                "mimeType": mime_type,
                "handler": handler,
            }
        )
        logger.info("mcp_resource_registered", uri=uri)

    async def _handle_sse(self, request: Request) -> StreamingResponse:
        async def event_stream():
            yield f"data: {json.dumps({'jsonrpc': '2.0', 'method': 'initialize', 'params': {'protocolVersion': '2024-11-05', 'capabilities': {}, 'serverInfo': {'name': 'enaya-mcp', 'version': '0.1.0'}}})}\n\n"
            while True:
                yield f"data: {json.dumps({'jsonrpc': '2.0', 'method': 'ping'})}\n\n"
                await asyncio.sleep(30)

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    async def _handle_message(self, request: Request) -> Any:
        try:
            body = await request.json()
            method = body.get("method")
            params = body.get("params", {})
            msg_id = body.get("id")

            if method == "tools/list":
                result = {
                    "tools": [
                        {"name": t["name"], "description": t["description"], "inputSchema": t["inputSchema"]}
                        for t in self._tools
                    ]
                }
            elif method == "tools/call":
                tool_name = params.get("name", "")
                tool_args = params.get("arguments", {})
                tool = next((t for t in self._tools if t["name"] == tool_name), None)
                if tool:
                    try:
                        result_data = await tool["handler"](**tool_args)
                        result = {"content": [{"type": "text", "text": json.dumps(result_data)}]}
                    except Exception as e:
                        result = {"isError": True, "content": [{"type": "text", "text": str(e)}]}
                else:
                    result = {
                        "isError": True,
                        "content": [{"type": "text", "text": f"Tool {tool_name} not found"}],
                    }
            elif method == "resources/list":
                result = {
                    "resources": [
                        {"uri": r["uri"], "name": r["name"], "mimeType": r["mimeType"]}
                        for r in self._resources
                    ]
                }
            else:
                result = {"error": {"code": -32601, "message": f"Method {method} not found"}}

            response_body = {"jsonrpc": "2.0", "id": msg_id, "result": result}
            from fastapi import Response

            return Response(content=json.dumps(response_body), media_type="application/json")
        except Exception as e:
            logger.error("mcp_message_error", error=str(e))
            from fastapi import Response

            return Response(
                content=json.dumps({"jsonrpc": "2.0", "error": {"code": -32700, "message": str(e)}}),
                media_type="application/json",
                status_code=400,
            )
