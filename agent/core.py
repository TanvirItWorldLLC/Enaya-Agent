from __future__ import annotations

import asyncio
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Callable, Coroutine

import structlog

from enaya.config.settings import EnayaConfig, get_config
from enaya.llm.providers import BaseLLMProvider, LLMProviderFactory
from enaya.memory.store import MemoryStore
from enaya.tools.registry import ToolRegistry
from enaya.skills.manager import SkillManager

logger = structlog.get_logger()


class AgentStatus(Enum):
    IDLE = auto()
    PLANNING = auto()
    EXECUTING = auto()
    WAITING = auto()
    ERROR = auto()
    COMPLETED = auto()


@dataclass
class Thought:
    content: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    step: int = 0


@dataclass
class Action:
    tool_name: str
    arguments: dict[str, Any]
    thought: str | None = None
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class Observation:
    action: Action
    result: Any
    success: bool
    error: str | None = None
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class AgentState:
    task: str
    thoughts: list[Thought] = field(default_factory=list)
    actions: list[Action] = field(default_factory=list)
    observations: list[Observation] = field(default_factory=list)
    status: AgentStatus = AgentStatus.IDLE
    iteration: int = 0
    max_iterations: int = 25
    context: dict[str, Any] = field(default_factory=dict)

    @property
    def history(self) -> list[dict[str, Any]]:
        h = []
        for t, a, o in zip(self.thoughts, self.actions, self.observations):
            h.append({"role": "assistant", "content": t.content})
            h.append({"role": "tool", "name": a.tool_name, "content": str(o.result)})
        return h


class Agent:
    def __init__(
        self,
        config: EnayaConfig | None = None,
        llm: BaseLLMProvider | None = None,
        memory: MemoryStore | None = None,
        tools: ToolRegistry | None = None,
        skills: SkillManager | None = None,
    ) -> None:
        self.config = config or get_config()
        if llm is None:
            llm = LLMProviderFactory.create(
                provider=self.config.llm.provider,
                model=self.config.llm.model,
                api_key=self.config.llm.api_key,
                base_url=self.config.llm.base_url,
                temperature=self.config.llm.temperature,
                max_tokens=self.config.llm.max_tokens,
                timeout=self.config.llm.timeout,
            )
        self.llm = llm
        self.memory = memory or MemoryStore(self.config.memory)
        self.tools = tools or ToolRegistry()
        self.skills = skills or SkillManager()
        self.state: AgentState | None = None
        self.id = str(uuid.uuid4())
        self._lock = asyncio.Lock()
        self._observers: list[Callable[[AgentState], Coroutine[Any, Any, None]]] = []
        logger.info(
            "agent_initialized",
            agent_id=self.id,
            llm_provider=self.config.llm.provider,
            model=self.config.llm.model,
        )

    async def run(self, task: str, context: dict[str, Any] | None = None) -> AgentState:
        async with self._lock:
            self.state = AgentState(
                task=task,
                max_iterations=self.config.agent.max_iterations,
                context=context or {},
                status=AgentStatus.PLANNING,
            )
            await self._notify_observers()
            await self.memory.add_interaction(role="user", content=task)
            try:
                while self.state.iteration < self.state.max_iterations:
                    self.state.status = AgentStatus.PLANNING
                    thought = await self._think()
                    self.state.thoughts.append(thought)
                    self.state.status = AgentStatus.EXECUTING
                    action = await self._plan_action(thought)
                    if action is None:
                        break
                    self.state.actions.append(action)
                    observation = await self._act(action)
                    self.state.observations.append(observation)
                    await self.memory.add_interaction(
                        role="assistant",
                        content=f"[{action.tool_name}] {observation.result}",
                    )
                    if await self._is_complete():
                        break
                    self.state.iteration += 1
                self.state.status = AgentStatus.COMPLETED
            except Exception as e:
                logger.error("agent_error", error=str(e), exc_info=True)
                self.state.status = AgentStatus.ERROR
                self.state.context["error"] = str(e)
            finally:
                await self._notify_observers()
                await self.memory.save_session(self.state)
            return self.state

    async def _think(self) -> Thought:
        prompt = self._build_thought_prompt()
        response = await self.llm.complete(prompt, temperature=self.config.agent.temperature)
        return Thought(content=response, step=self.state.iteration)

    async def _plan_action(self, thought: Thought) -> Action | None:
        available_tools = self.tools.describe_all() + self.skills.describe_all()
        prompt = self._build_action_prompt(thought, available_tools)
        response = await self.llm.complete(prompt, temperature=0.2)
        try:
            parsed = self._parse_action(response)
            if parsed["tool_name"] in ("done", "finish", "complete", "end"):
                return None
            return Action(
                tool_name=parsed["tool_name"],
                arguments=parsed.get("arguments", {}),
                thought=thought.content,
            )
        except Exception as e:
            logger.warning("action_parse_failed", raw=response, error=str(e))
            return Action(tool_name="noop", arguments={}, thought=thought.content)

    async def _act(self, action: Action) -> Observation:
        tool = self.tools.get(action.tool_name) or self.skills.get(action.tool_name)
        if not tool:
            return Observation(
                action=action,
                result=None,
                success=False,
                error=f"Tool '{action.tool_name}' not found",
            )
        try:
            result = await tool.execute(**action.arguments)
            return Observation(action=action, result=result, success=True)
        except Exception as e:
            logger.error("tool_execution_failed", tool=action.tool_name, error=str(e))
            return Observation(action=action, result=None, success=False, error=str(e))

    async def _is_complete(self) -> bool:
        prompt = self._build_completion_prompt()
        response = await self.llm.complete(prompt, temperature=0.0)
        return "yes" in response.lower() or "true" in response.lower()

    def _build_thought_prompt(self) -> list[dict[str, Any]]:
        system = {
            "role": "system",
            "content": (
                "You are Enaya, an autonomous agent. Analyze the task, "
                "review past actions, and think step-by-step. "
                f"Current task: {self.state.task}"
            ),
        }
        history = self.state.history
        user_msg = {
            "role": "user",
            "content": (
                f"Iteration: {self.state.iteration}\n"
                "What should you think about next?\n"
                f"Available context: {json.dumps(self.state.context, default=str)}"
            ),
        }
        return [system] + history + [user_msg]

    def _build_action_prompt(self, thought: Thought, tools: list[dict]) -> list[dict[str, Any]]:
        system = {
            "role": "system",
            "content": (
                "Given your thought, choose the next action. "
                "Respond in JSON: {\"tool_name\": ..., \"arguments\": {...}}\n"
                "Available tools: " + json.dumps(tools, indent=2) + "\n\n"
                "Use tool_name 'done' or 'finish' when the task is complete."
            ),
        }
        return [
            system,
            {"role": "assistant", "content": thought.content},
            {"role": "user", "content": "What action do you take? Respond with JSON only."},
        ]

    def _build_completion_prompt(self) -> list[dict[str, Any]]:
        return [
            {
                "role": "system",
                "content": "Has the task been completed successfully? Respond yes or no only.",
            },
            {
                "role": "user",
                "content": (
                    f"Task: {self.state.task}\n"
                    f"History: {json.dumps(self.state.history, default=str)}\n"
                    "Completed?"
                ),
            },
        ]

    def _parse_action(self, raw: str) -> dict[str, Any]:
        cleaned = raw.strip()
        if cleaned.startswith("{") and cleaned.endswith("}"):
            return json.loads(cleaned)
        if "```json" in cleaned:
            json_part = cleaned.split("```json")[1].split("```")[0]
            return json.loads(json_part.strip())
        for line in cleaned.split("\n"):
            line = line.strip()
            if line.startswith("{") and line.endswith("}"):
                return json.loads(line)
        raise ValueError(f"Could not parse action: {raw}")

    def add_observer(self, callback: Callable[[AgentState], Coroutine[Any, Any, None]]) -> None:
        self._observers.append(callback)

    async def _notify_observers(self) -> None:
        if self.state:
            for obs in self._observers:
                try:
                    await obs(self.state)
                except Exception:
                    pass

    async def chat(self, message: str) -> str:
        await self.memory.add_interaction(role="user", content=message)
        context = await self.memory.retrieve_relevant(message, k=5)
        prompt = [
            {
                "role": "system",
                "content": (
                    "You are Enaya, a helpful assistant. "
                    "Use retrieved context if relevant. Be concise and accurate."
                ),
            },
            *[{"role": m["role"], "content": m["content"]} for m in context],
            {"role": "user", "content": message},
        ]
        response = await self.llm.complete(prompt, temperature=self.config.agent.temperature)
        await self.memory.add_interaction(role="assistant", content=response)
        return response
