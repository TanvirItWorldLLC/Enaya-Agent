from __future__ import annotations

import asyncio
import inspect
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine

import structlog

logger = structlog.get_logger()


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any] = field(default_factory=dict)
    handler: Callable[..., Coroutine[Any, Any, Any]] | None = None

    async def execute(self, **kwargs: Any) -> Any:
        if self.handler is None:
            raise RuntimeError(f"Tool {self.name} has no handler")
        if asyncio.iscoroutinefunction(self.handler):
            return await self.handler(**kwargs)
        return self.handler(**kwargs)


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}
        logger.info("tool_registry_initialized")

    def register(self, tool: ToolSpec | Callable[..., Any]) -> ToolSpec:
        if isinstance(tool, ToolSpec):
            self._tools[tool.name] = tool
            return tool
        spec = _spec_from_callable(tool)
        self._tools[spec.name] = spec
        return spec

    def get(self, name: str) -> ToolSpec | None:
        return self._tools.get(name)

    def describe_all(self) -> list[dict[str, Any]]:
        return [
            {
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters,
            }
            for t in self._tools.values()
        ]

    def list_tools(self) -> list[str]:
        return list(self._tools.keys())

    def unregister(self, name: str) -> None:
        self._tools.pop(name, None)


def _spec_from_callable(fn: Callable[..., Any]) -> ToolSpec:
    sig = inspect.signature(fn)
    params = {}
    for pname, p in sig.parameters.items():
        if pname == "self":
            continue
        param_info: dict[str, Any] = {
            "type": str(p.annotation) if p.annotation != inspect.Parameter.empty else "any"
        }
        if p.default != inspect.Parameter.empty:
            param_info["default"] = p.default
        params[pname] = param_info
    return ToolSpec(
        name=fn.__name__,
        description=fn.__doc__ or "No description",
        parameters={"properties": params},
        handler=fn,
    )


def tool(name: str | None = None, description: str | None = None) -> Callable:
    def decorator(fn: Callable[..., Any]) -> ToolSpec:
        spec = _spec_from_callable(fn)
        if name:
            spec.name = name
        if description:
            spec.description = description
        fn._enaya_tool_spec = spec  # type: ignore
        return spec

    return decorator
