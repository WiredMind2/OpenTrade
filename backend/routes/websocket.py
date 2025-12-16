"""
WebSocket endpoints for the Trading Backtester API.
"""
import json
from typing import Dict, Set, Tuple
from fastapi import WebSocket, WebSocketDisconnect

from backend.logging_config import get_component_logger


logger = get_component_logger(__file__)

# Active WebSocket connections (for tests to access directly)
active_connections = set()

# Chart subscriptions: websocket -> set of (symbol, resolution, listener_guid) tuples
chart_subscriptions: Dict[WebSocket, Set[Tuple[str, str, str]]] = {}


async def broadcast_websocket_message(message: dict):
    """Broadcast a message to all connected WebSocket clients.

    Returns a dict with broadcast statistics for testing/monitoring.
    """
    from backend.main import app_state

    # Use module-level set or app_state set (whichever has connections)
    connections = active_connections or app_state.get("active_websockets", set())

    if not connections:
        return {"sent": False, "clients": 0, "failed": 0}

    # Log message content for debugging
    logger.debug(f"Broadcasting websocket message: {message}")

    sent_count = 0
    failed_count = 0
    disconnected = set()

    for websocket in connections:
        try:
            await websocket.send_json(message)
            sent_count += 1
        except Exception as e:
            logger.warning("Failed to send websocket message", websocket=websocket, error=e)
            disconnected.add(websocket)
            failed_count += 1

    # Clean up disconnected clients
    for ws in disconnected:
        connections.discard(ws)
        if app_state.get("active_websockets"):
            app_state["active_websockets"].discard(ws)
    
    return {
        "sent": True,
        "clients": len(connections),
        "successful": sent_count,
        "failed": failed_count
    }


def subscribe_chart(websocket: WebSocket, symbol: str, resolution: str, listener_guid: str):
    """Subscribe a websocket connection to chart updates for a symbol/resolution."""
    if websocket not in chart_subscriptions:
        chart_subscriptions[websocket] = set()

    chart_subscriptions[websocket].add((symbol.upper(), resolution, listener_guid))
    logger.info(f"[WebSocket] Subscribed {websocket} to {symbol}:{resolution} (guid: {listener_guid})")


def unsubscribe_chart(websocket: WebSocket, listener_guid: str):
    """Unsubscribe a websocket connection from chart updates."""
    if websocket in chart_subscriptions:
        # Remove all subscriptions with this listener_guid
        subscriptions_to_remove = {sub for sub in chart_subscriptions[websocket] if sub[2] == listener_guid}
        chart_subscriptions[websocket] -= subscriptions_to_remove

        if not chart_subscriptions[websocket]:
            del chart_subscriptions[websocket]

        logger.info(f"[WebSocket] Unsubscribed {websocket} from {len(subscriptions_to_remove)} chart subscriptions (guid: {listener_guid})")


async def broadcast_chart_update(symbol: str, resolution: str, bar_data: dict):
    """Broadcast chart update to all clients subscribed to the symbol/resolution."""
    symbol = symbol.upper()
    logger.info(f"[WebSocket] Broadcasting chart update for {symbol}:{resolution} with bar data: {bar_data}")
    message = {
        "type": "chart_update",
        "data": {
            "symbol": symbol,
            "resolution": resolution,
            "bar": bar_data
        }
    }

    sent_count = 0
    failed_count = 0
    disconnected = set()

    # Find all websockets subscribed to this symbol/resolution
    for websocket, subscriptions in chart_subscriptions.items():
        if any(sub[0] == symbol and sub[1] == resolution for sub in subscriptions):
            try:
                await websocket.send_json(message)
                sent_count += 1
            except Exception as e:
                logger.error(f"[WebSocket] Failed to send chart update to {websocket}: {e}")
                disconnected.add(websocket)
                failed_count += 1

    # Clean up disconnected clients
    for ws in disconnected:
        if ws in chart_subscriptions:
            del chart_subscriptions[ws]
        active_connections.discard(ws)
        if app_state.get("active_websockets"):
            app_state["active_websockets"].discard(ws)

    logger.debug(f"[WebSocket] Broadcast chart update for {symbol}:{resolution} to {sent_count} clients ({failed_count} failed)")

    return {
        "sent": True,
        "clients": sent_count,
        "successful": sent_count,
        "failed": failed_count
    }


async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time trading updates."""
    from backend.main import app_state

    logger.info("[WebSocket Backend] Accepting connection...")
    await websocket.accept()
    
    # Add to both sets for compatibility
    app_state["active_websockets"].add(websocket)
    active_connections.add(websocket)
    
    logger.info(f"[WebSocket Backend] Connection established. Active connections: {len(app_state['active_websockets'])}")

    try:
        while True:
            logger.debug("[WebSocket Backend] Waiting for client message...")
            # Keep connection alive and wait for client messages
            data = await websocket.receive_text()
            logger.info(f"[WebSocket Backend] Received message: {data}")

            try:
                message = json.loads(data)
                message_type = message.get("type")

                if message_type == "subscribe_chart":
                    # Handle chart subscription
                    subscription_data = message.get("data", {})
                    symbol = subscription_data.get("symbol")
                    resolution = subscription_data.get("resolution")
                    listener_guid = subscription_data.get("listenerGuid")

                    if symbol and resolution and listener_guid:
                        subscribe_chart(websocket, symbol, resolution, listener_guid)
                        logger.info(f"[WebSocket Backend] Processed subscribe_chart: {symbol}:{resolution} for listener {listener_guid}")
                        # Log current subscriptions for this websocket
                        current_subs = list(chart_subscriptions.get(websocket, set()))
                        logger.info(f"[WebSocket Backend] Current subscriptions for this client: {len(current_subs)} total")
                    else:
                        logger.warning(f"[WebSocket Backend] Invalid subscribe_chart message: {message}")

                elif message_type == "unsubscribe_chart":
                    # Handle chart unsubscription
                    subscription_data = message.get("data", {})
                    listener_guid = subscription_data.get("listenerGuid")

                    if listener_guid:
                        unsubscribe_chart(websocket, listener_guid)
                        logger.info(f"[WebSocket Backend] Processed unsubscribe_chart: {listener_guid}")
                        # Log remaining subscriptions
                        remaining_subs = list(chart_subscriptions.get(websocket, set()))
                        logger.info(f"[WebSocket Backend] Remaining subscriptions for this client: {len(remaining_subs)}")
                    else:
                        logger.warning(f"[WebSocket Backend] Invalid unsubscribe_chart message: {message}")

                else:
                    # Handle other message types (existing functionality)
                    logger.debug(f"[WebSocket Backend] Unknown message type: {message_type}")

            except json.JSONDecodeError:
                logger.warning(f"[WebSocket Backend] Invalid JSON received: {data}")

    except WebSocketDisconnect:
        logger.info("[WebSocket Backend] Client disconnected normally")
    except Exception as e:
        logger.error(f"[WebSocket Backend] Error: {str(e)}", exc_info=True)
    finally:
        # Clean up subscriptions for this websocket
        if websocket in chart_subscriptions:
            del chart_subscriptions[websocket]

        app_state["active_websockets"].discard(websocket)
        active_connections.discard(websocket)
        logger.info(f"[WebSocket Backend] Connection closed. Active connections: {len(app_state['active_websockets'])}")