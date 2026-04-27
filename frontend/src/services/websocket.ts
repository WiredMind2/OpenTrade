import { WebSocketMessage, ScriptStatusMessage, PipelineStatusMessage, BacktestStatusMessage, ChartUpdateMessage, TrainingProgressMessage } from '../types'

type MessageType = 'script_status' | 'pipeline_status' | 'backtest_status' | 'chart_update' | 'training_progress'
type MessageListener<T = any> = (message: WebSocketMessage & { data: T }) => void

interface ListenerMap {
  script_status: MessageListener<ScriptStatusMessage['data']>[]
  pipeline_status: MessageListener<PipelineStatusMessage['data']>[]
  backtest_status: MessageListener<BacktestStatusMessage['data']>[]
  chart_update: MessageListener<ChartUpdateMessage['data']>[]
  training_progress: MessageListener<TrainingProgressMessage['data']>[]
}

class WebSocketService {
  private ws: WebSocket | null = null
  private reconnectAttempts = 0
  private maxReconnectAttempts = 5
  private reconnectDelay = 1000 // Start with 1 second
  private maxReconnectDelay = 30000 // Max 30 seconds
  private reconnectTimeout: NodeJS.Timeout | null = null
  private isConnecting = false
  private listeners: ListenerMap = {
    script_status: [],
    pipeline_status: [],
    backtest_status: [],
    chart_update: [],
    training_progress: []
  }

  private getWebSocketUrl(): string {
    const apiBase = import.meta.env.VITE_API_BASE
    if (apiBase) {
      // Convert explicit HTTP(S) API base to WS(S)
      return `${apiBase.replace(/^http/, 'ws')}/ws`
    }

    // Same-origin fallback keeps Docker and local dev consistent.
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
    return `${protocol}//${window.location.host}/ws`
  }

  connect(): void {
    if (this.ws?.readyState === WebSocket.OPEN || this.isConnecting) {
      return
    }

    this.isConnecting = true
    const url = this.getWebSocketUrl()

    try {
      this.ws = new WebSocket(url)

      this.ws.onopen = () => {
        console.log('[WebSocket] Connected to', url)
        this.isConnecting = false
        this.reconnectAttempts = 0
        this.reconnectDelay = 1000
      }

      this.ws.onmessage = (event) => {
        try {
          const message: WebSocketMessage = JSON.parse(event.data)
          this.handleMessage(message)
        } catch (error) {
          console.error('[WebSocket] Failed to parse message:', error)
        }
      }

      this.ws.onclose = (event) => {
        console.log('[WebSocket] Connection closed', event.code, event.reason)
        this.isConnecting = false
        this.ws = null

        // Attempt reconnection unless it was a clean close
        if (event.code !== 1000) {
          this.scheduleReconnect()
        }
      }

      this.ws.onerror = (error) => {
        console.error('[WebSocket] Connection error:', error)
        this.isConnecting = false
      }

    } catch (error) {
      console.error('[WebSocket] Failed to create connection:', error)
      this.isConnecting = false
      this.scheduleReconnect()
    }
  }

  disconnect(): void {
    if (this.reconnectTimeout) {
      clearTimeout(this.reconnectTimeout)
      this.reconnectTimeout = null
    }

    if (this.ws) {
      this.ws.close(1000, 'Client disconnect')
      this.ws = null
    }

    this.isConnecting = false
    this.reconnectAttempts = 0
  }

  private scheduleReconnect(): void {
    if (this.reconnectAttempts >= this.maxReconnectAttempts) {
      console.error('[WebSocket] Max reconnection attempts reached')
      return
    }

    this.reconnectAttempts++
    const delay = Math.min(this.reconnectDelay * Math.pow(2, this.reconnectAttempts - 1), this.maxReconnectDelay)

    console.log(`[WebSocket] Scheduling reconnection in ${delay}ms (attempt ${this.reconnectAttempts}/${this.maxReconnectAttempts})`)

    this.reconnectTimeout = setTimeout(() => {
      this.connect()
    }, delay)
  }

  private handleMessage(message: WebSocketMessage): void {
    const { type } = message

    if (type in this.listeners) {
      const typeListeners = this.listeners[type as MessageType]
      typeListeners.forEach(listener => {
        try {
          listener(message as any)
        } catch (error) {
          console.error(`[WebSocket] Error in ${type} listener:`, error)
        }
      })
    } else {
      console.warn('[WebSocket] Received message with unknown type:', type)
    }
  }

  registerListener<T extends MessageType>(
    type: T,
    listener: MessageListener<ListenerMap[T][0] extends MessageListener<infer U> ? U : any>
  ): () => void {
    if (!this.listeners[type]) {
      console.warn(`[WebSocket] Unknown message type: ${type}`)
      return () => {}
    }

    this.listeners[type].push(listener as any)

    // Ensure connection is active
    this.connect()

    // Return unsubscribe function
    return () => {
      const index = this.listeners[type].indexOf(listener as any)
      if (index > -1) {
        this.listeners[type].splice(index, 1)
      }
    }
  }

  unregisterListener<T extends MessageType>(
    type: T,
    listener: MessageListener<ListenerMap[T][0] extends MessageListener<infer U> ? U : any>
  ): void {
    if (!this.listeners[type]) {
      return
    }

    const index = this.listeners[type].indexOf(listener as any)
    if (index > -1) {
      this.listeners[type].splice(index, 1)
    }
  }

  unregisterAllListeners(type?: MessageType): void {
    if (type) {
      this.listeners[type] = []
    } else {
      this.listeners = {
        script_status: [],
        pipeline_status: [],
        backtest_status: [],
        chart_update: [],
        training_progress: []
      }
    }
  }

  getConnectionStatus(): 'connecting' | 'connected' | 'disconnected' {
    if (this.isConnecting) return 'connecting'
    if (this.ws?.readyState === WebSocket.OPEN) return 'connected'
    return 'disconnected'
  }

  sendMessage(message: any): void {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(message))
    } else {
      console.warn('[WebSocket] Cannot send message: not connected')
    }
  }

  isConnected(): boolean {
    return this.ws?.readyState === WebSocket.OPEN
  }
}

// Export singleton instance
export const websocketService = new WebSocketService()
export default websocketService