"""WebSocket JSON-RPC client for ControlServer.

Implements the same protocol the web UI uses: ws://host:port/ws?token=...
Messages are {"method": "...", "params": {...}, "id": N}
Responses are {"id": N, "result": ...} or {"id": N, "error": "..."}

Status pushes are unsolicited {"method":"status","result":{...}}
"""
from __future__ import annotations

import asyncio
import json
import os
from typing import Any, Dict, Optional

import websockets
from websockets.exceptions import ConnectionClosed

from dotenv import load_dotenv

load_dotenv()

def _is_closed(ws) -> bool:
    if ws is None:
        return True
    state = getattr(ws, "state", None)
    if state is not None:
        return state.name == "CLOSED"
    return getattr(ws, "closed", True)

class ControlClient:
    def __init__(self):
        self.host = os.getenv("CONTROL_HOST", "localhost")
        self.port = int(os.getenv("CONTROL_PORT", "9013"))
        self.token = os.getenv("CONTROL_TOKEN", "")
        self.ws_url = f"ws://{self.host}:{self.port}/ws"
        if self.token:
            self.ws_url += f"?token={self.token}"
        self._ws: Optional[websockets.WebSocketClientProtocol] = None
        self._next_id = 1
        self._pending: Dict[int, asyncio.Future] = {}
        self._status: Optional[Dict[str, Any]] = None
        self._status_lock = asyncio.Lock()
        self._running = False

    async def connect(self):
        self._running = True
        self._ws = await websockets.connect(self.ws_url)
        asyncio.create_task(self._receiver())

    async def _receiver(self):
        try:
            async for msg in self._ws:
                try:
                    data = json.loads(msg)
                except Exception:
                    continue
                # Response to RPC
                if "id" in data:
                    fut = self._pending.pop(data.get("id"), None)
                    if fut and not fut.done():
                        if "error" in data:
                            fut.set_exception(RuntimeError(data["error"]))
                        else:
                            fut.set_result(data.get("result"))
                    continue
                # Unsolicited status push
                if data.get("method") == "status" and "result" in data:
                    async with self._status_lock:
                        self._status = data["result"]
        except ConnectionClosed:
            self._running = False

    async def call(self, method: str, params: Optional[Dict[str, Any]] = None) -> Any:
        if _is_closed(self._ws):
            await self.connect()
        params = params or {}
        msg_id = self._next_id
        self._next_id += 1
        payload = {"method": method, "params": params, "id": msg_id}
        fut = asyncio.get_event_loop().create_future()
        self._pending[msg_id] = fut
        await self._ws.send(json.dumps(payload))
        return await fut

    async def get_status(self) -> Optional[Dict[str, Any]]:
        async with self._status_lock:
            return dict(self._status) if self._status else None

    async def close(self):
        self._running = False
        if self._ws:
            await self._ws.close()
