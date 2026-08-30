from __future__ import annotations

import asyncio

from enaya.tools.registry import tool, ToolRegistry


@tool(name="shell", description="Execute a shell command safely")
async def shell_command(command: str, timeout: int = 30) -> dict[str, Any]:
    try:
        proc = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        return {
            "stdout": stdout.decode().strip(),
            "stderr": stderr.decode().strip(),
            "returncode": proc.returncode,
        }
    except asyncio.TimeoutError:
        proc.kill()
        return {"error": "Command timed out"}


@tool(name="read_file", description="Read contents of a file")
async def read_file(path: str) -> str:
    try:
        with open(path, "r") as f:
            return f.read()
    except Exception as e:
        return f"Error: {e}"


@tool(name="write_file", description="Write content to a file")
async def write_file(path: str, content: str) -> str:
    try:
        with open(path, "w") as f:
            f.write(content)
        return f"Written {len(content)} chars to {path}"
    except Exception as e:
        return f"Error: {e}"


@tool(name="python", description="Execute Python code and return the result")
async def python_code(code: str) -> str:
    try:
        local_ns: dict[str, Any] = {}
        exec(code, {}, local_ns)
        result = local_ns.get("_result", local_ns)
        return str(result)
    except Exception as e:
        return f"Error: {e}"


@tool(name="http_get", description="Make an HTTP GET request")
async def http_get(url: str) -> dict[str, Any]:
    import httpx

    async with httpx.AsyncClient() as client:
        resp = await client.get(url, timeout=30)
        return {
            "status": resp.status_code,
            "headers": dict(resp.headers),
            "text": resp.text[:5000],
        }


@tool(name="search_memory", description="Search the agent's memory for relevant information")
async def search_memory(query: str, k: int = 5) -> list[dict[str, Any]]:
    return [{"note": "Memory search requires initialized agent"}]


def register_builtin_tools(registry: ToolRegistry) -> None:
    registry.register(shell_command)
    registry.register(read_file)
    registry.register(write_file)
    registry.register(python_code)
    registry.register(http_get)
    registry.register(search_memory)
