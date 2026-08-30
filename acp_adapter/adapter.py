from __future__ import annotations

import asyncio
import json
import uuid
from typing import Any

import structlog

logger = structlog.get_logger()


class ACPMessage:
    def __init__(
        self,
        msg_type: str,
        payload: dict[str, Any],
        sender: str,
        recipient: str | None = None,
        msg_id: str | None = None,
    ) -> None:
        self.type = msg_type
        self.payload = payload
        self.sender = sender
        self.recipient = recipient
        self.id = msg_id or str(uuid.uuid4())
        self.timestamp = asyncio.get_event_loop().time()

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "payload": self.payload,
            "sender": self.sender,
            "recipient": self.recipient,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ACPMessage":
        return cls(
            msg_type=data["type"],
            payload=data.get("payload", {}),
            sender=data["sender"],
            recipient=data.get("recipient"),
            msg_id=data.get("id"),
        )


class ACPAdapter:
    def __init__(self, agent_id: str | None = None) -> None:
        self.agent_id = agent_id or str(uuid.uuid4())
        self._inbox: asyncio.Queue[ACPMessage] = asyncio.Queue()
        self._handlers: dict[str, list[Any]] = {}
        logger.info("acp_adapter_initialized", agent_id=self.agent_id)

    async def send(
        self, msg_type: str, payload: dict[str, Any], recipient: str | None = None
    ) -> ACPMessage:
        msg = ACPMessage(msg_type, payload, self.agent_id, recipient)
        logger.info("acp_message_sent", type=msg_type, to=recipient)
        return msg

    async def receive(self) -> ACPMessage:
        return await self._inbox.get()

    async def broadcast(self, msg_type: str, payload: dict[str, Any]) -> ACPMessage:
        return await self.send(msg_type, payload, recipient=None)

    def on(self, msg_type: str, handler: Any) -> None:
        if msg_type not in self._handlers:
            self._handlers[msg_type] = []
        self._handlers[msg_type].append(handler)

    async def process_inbox(self) -> None:
        while True:
            msg = await self.receive()
            handlers = self._handlers.get(msg.type, [])
            for h in handlers:
                try:
                    await h(msg)
                except Exception as e:
                    logger.error("acp_handler_error", type=msg.type, error=str(e))

    async def request(
        self, recipient: str, msg_type: str, payload: dict[str, Any], timeout: float = 30.0
    ) -> ACPMessage | None:
        msg = await self.send(msg_type, payload, recipient)
        try:
            return await asyncio.wait_for(self._wait_for_response(msg.id), timeout=timeout)
        except asyncio.TimeoutError:
            logger.warning("acp_request_timeout", msg_id=msg.id)
            return None

    async def _wait_for_response(self, msg_id: str) -> ACPMessage:
        while True:
            msg = await self.receive()
            if msg.payload.get("in_reply_to") == msg_id:
                return msg
