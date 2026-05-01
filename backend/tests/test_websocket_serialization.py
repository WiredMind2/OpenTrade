from datetime import datetime

import pytest

from backend.routes.websocket import active_connections, broadcast_websocket_message


class _DummyWebSocket:
    def __init__(self):
        self.messages = []

    async def send_json(self, payload):
        self.messages.append(payload)


@pytest.mark.asyncio
async def test_broadcast_websocket_message_serializes_datetimes():
    ws = _DummyWebSocket()
    active_connections.add(ws)
    result = await broadcast_websocket_message(
        {
            "type": "backtest_status",
            "data": {
                "timestamp": datetime(2026, 4, 29, 9, 41, 55),
                "end_date": datetime(2026, 4, 29, 9, 41, 55),
            },
        }
    )

    try:
        assert result["sent"] is True
        assert ws.messages, "Expected websocket to receive serialized payload"
        payload = ws.messages[0]
        assert isinstance(payload["data"]["timestamp"], str)
        assert isinstance(payload["data"]["end_date"], str)
    finally:
        active_connections.discard(ws)
