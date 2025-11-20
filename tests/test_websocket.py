"""
Unit tests for WebSocket endpoints.
"""
import pytest
import json
from unittest.mock import Mock, patch, AsyncMock
import asyncio
from fastapi.testclient import TestClient
from fastapi import WebSocketDisconnect

from main import app


@pytest.mark.unit
class TestWebSocketEndpoints:
    """Test WebSocket endpoint functionality."""

    def setup_method(self):
        """Set up test client."""
        self.client = TestClient(app)

    @pytest.mark.asyncio
    async def test_broadcast_websocket_message_no_connections(self):
        """Test broadcasting message when no clients are connected."""
        from backend.routes.websocket import broadcast_websocket_message, active_connections

        # Ensure no connections
        active_connections.clear()
        
        message = {"type": "test", "data": "hello"}
        # Should not raise any exception
        result = await broadcast_websocket_message(message)
        assert result["clients"] == 0

    @pytest.mark.asyncio
    async def test_broadcast_websocket_message_with_connections(self):
        """Test broadcasting message to connected clients."""
        from backend.routes.websocket import broadcast_websocket_message, active_connections

        # Mock websocket and add to active_connections
        mock_ws = AsyncMock()
        active_connections.clear()
        active_connections.add(mock_ws)

        message = {"type": "test", "data": "hello"}
        result = await broadcast_websocket_message(message)

        # Verify message was sent
        mock_ws.send_json.assert_called_once_with(message)
        assert result["clients"] == 1
        assert result["successful"] == 1
        
        # Clean up
        active_connections.clear()

    @pytest.mark.asyncio
    async def test_broadcast_websocket_message_with_disconnected_client(self):
        """Test broadcasting message when a client disconnects."""
        from backend.routes.websocket import broadcast_websocket_message, active_connections

        # Mock websockets - one working, one failing
        mock_ws_good = AsyncMock()
        mock_ws_bad = AsyncMock()
        mock_ws_bad.send_json.side_effect = Exception("Connection lost")

        active_connections.clear()
        active_connections.add(mock_ws_good)
        active_connections.add(mock_ws_bad)

        message = {"type": "test", "data": "hello"}
        result = await broadcast_websocket_message(message)

        # Verify good websocket received message
        mock_ws_good.send_json.assert_called_once_with(message)

        # Verify bad websocket was removed
        assert mock_ws_bad not in active_connections
        assert mock_ws_good in active_connections
        assert result["successful"] == 1
        assert result["failed"] == 1
        
        # Clean up
        active_connections.clear()

    @pytest.mark.asyncio
    async def test_websocket_endpoint_connection(self):
        """Test WebSocket endpoint accepts connections."""
        from routes.websocket import websocket_endpoint

        # Mock websocket
        mock_ws = AsyncMock()
        mock_ws.receive_text.side_effect = WebSocketDisconnect()

        active_sockets = set()

        with patch('main.app_state', {'active_websockets': active_sockets}):
            # Test connection acceptance
            await websocket_endpoint(mock_ws)

            # Verify websocket was accepted
            mock_ws.accept.assert_called_once()

    @pytest.mark.asyncio
    async def test_websocket_endpoint_message_handling(self):
        """Test WebSocket endpoint handles messages."""
        from routes.websocket import websocket_endpoint

        # Mock websocket
        mock_ws = AsyncMock()
        mock_ws.receive_text.side_effect = ["hello", WebSocketDisconnect()]

        active_sockets = set()

        with patch('main.app_state', {'active_websockets': active_sockets}):
            await websocket_endpoint(mock_ws)

            # Verify websocket was accepted
            mock_ws.accept.assert_called_once()

    @pytest.mark.asyncio
    async def test_chart_subscription(self):
        """Test chart subscription functionality."""
        from backend.routes.websocket import subscribe_chart, chart_subscriptions

        # Clear subscriptions
        chart_subscriptions.clear()

        # Mock websocket
        mock_ws = AsyncMock()

        # Subscribe to chart
        subscribe_chart(mock_ws, "AAPL", "1D", "guid123")

        # Verify subscription was added
        assert mock_ws in chart_subscriptions
        assert ("AAPL", "1D", "guid123") in chart_subscriptions[mock_ws]

        # Clean up
        chart_subscriptions.clear()

    @pytest.mark.asyncio
    async def test_chart_unsubscription(self):
        """Test chart unsubscription functionality."""
        from backend.routes.websocket import subscribe_chart, unsubscribe_chart, chart_subscriptions

        # Clear subscriptions
        chart_subscriptions.clear()

        # Mock websocket
        mock_ws = AsyncMock()

        # Subscribe first
        subscribe_chart(mock_ws, "AAPL", "1D", "guid123")
        subscribe_chart(mock_ws, "GOOGL", "1H", "guid123")

        # Verify subscriptions exist
        assert ("AAPL", "1D", "guid123") in chart_subscriptions[mock_ws]
        assert ("GOOGL", "1H", "guid123") in chart_subscriptions[mock_ws]

        # Unsubscribe
        unsubscribe_chart(mock_ws, "guid123")

        # Verify websocket is no longer in subscriptions (all subscriptions removed)
        assert mock_ws not in chart_subscriptions

        # Clean up
        chart_subscriptions.clear()

    @pytest.mark.asyncio
    async def test_broadcast_chart_update(self):
        """Test broadcasting chart updates to subscribed clients."""
        from backend.routes.websocket import broadcast_chart_update, subscribe_chart, chart_subscriptions, active_connections

        # Clear data
        chart_subscriptions.clear()
        active_connections.clear()

        # Mock websockets
        mock_ws1 = AsyncMock()
        mock_ws2 = AsyncMock()
        mock_ws3 = AsyncMock()  # Not subscribed

        active_connections.add(mock_ws1)
        active_connections.add(mock_ws2)
        active_connections.add(mock_ws3)

        # Subscribe clients
        subscribe_chart(mock_ws1, "AAPL", "1D", "guid1")
        subscribe_chart(mock_ws2, "AAPL", "1D", "guid2")
        subscribe_chart(mock_ws2, "GOOGL", "1H", "guid3")  # Different symbol

        # Broadcast update
        bar_data = {
            "time": 1640995200,
            "open": 100.0,
            "high": 105.0,
            "low": 95.0,
            "close": 102.0,
            "volume": 1000000
        }
        result = await broadcast_chart_update("AAPL", "1D", bar_data)

        # Verify correct clients received the message
        expected_message = {
            "type": "chart_update",
            "data": {
                "symbol": "AAPL",
                "resolution": "1D",
                "bar": bar_data
            }
        }

        mock_ws1.send_json.assert_called_once_with(expected_message)
        mock_ws2.send_json.assert_called_once_with(expected_message)
        mock_ws3.send_json.assert_not_called()  # Not subscribed

        assert result["clients"] == 2
        assert result["successful"] == 2

        # Clean up
        chart_subscriptions.clear()
        active_connections.clear()

    @pytest.mark.asyncio
    async def test_websocket_chart_subscription_messages(self):
        """Test WebSocket endpoint handles chart subscription messages."""
        from routes.websocket import websocket_endpoint, chart_subscriptions, subscribe_chart

        # Clear subscriptions
        chart_subscriptions.clear()

        # Mock websocket
        mock_ws = AsyncMock()
        subscription_message = {
            "type": "subscribe_chart",
            "data": {
                "symbol": "AAPL",
                "resolution": "1D",
                "listenerGuid": "guid123"
            }
        }
        unsubscription_message = {
            "type": "unsubscribe_chart",
            "data": {
                "listenerGuid": "guid123"
            }
        }
        mock_ws.receive_text.side_effect = [json.dumps(subscription_message), json.dumps(unsubscription_message), WebSocketDisconnect()]

        active_sockets = set()

        with patch('main.app_state', {'active_websockets': active_sockets}):
            await websocket_endpoint(mock_ws)

            # Verify websocket was accepted
            mock_ws.accept.assert_called_once()

        # Verify subscription was processed and then cleaned up on disconnect
        # The subscription should not persist after disconnect
        assert mock_ws not in chart_subscriptions

        # Clean up
        chart_subscriptions.clear()