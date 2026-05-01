import type {
  IDatafeedChartApi,
  DatafeedConfiguration,
  LibrarySymbolInfo,
  Bar,
  SearchSymbolResultItem,
  ResolutionString,
  PeriodParams,
  HistoryCallback,
  ErrorCallback,
  ResolveCallback,
  SearchSymbolsCallback,
  SubscribeBarsCallback,
  SymbolResolveExtension
} from '../../public/charting_library/datafeed-api.d.ts'
import instance from './api'
import websocketService from './websocket'

interface SubscriptionInfo {
  symbolInfo: LibrarySymbolInfo
  resolution: ResolutionString
  onTick: SubscribeBarsCallback
  listenerGuid: string
}

class TradingViewUDFDatafeed implements IDatafeedChartApi {
  private subscriptions: Map<string, SubscriptionInfo> = new Map()
  private readonly baseUrl: string
  private chartUpdateUnsubscribe?: () => void

  private normalizeTimestampToSeconds(value: unknown): number {
    const ts = Number(value)
    if (!Number.isFinite(ts)) return 0
    // TradingView periodParams are typically seconds, but be defensive if ms are provided.
    return ts > 1e11 ? Math.floor(ts / 1000) : Math.floor(ts)
  }

  private normalizeTimestampToMs(value: unknown): number {
    const ts = Number(value)
    if (!Number.isFinite(ts)) return 0
    // TradingView expects bar.time in milliseconds.
    // Some backend paths still return Unix seconds.
    return ts < 1e11 ? ts * 1000 : ts
  }

  constructor() {
    this.baseUrl = import.meta.env.VITE_API_BASE || 'http://localhost:8000'

    // Register listener for chart updates
    this.chartUpdateUnsubscribe = websocketService.registerListener('chart_update', (message) => {
      this.handleChartUpdate(message.data)
    })
  }

  /**
   * Called when the chart needs the configuration data
   */
  onReady(callback: (configuration: DatafeedConfiguration) => void): void {
    instance.get('/udf/config')
      .then(response => {
        const config: DatafeedConfiguration & Record<string, any> = {
          exchanges: response.data.exchanges || [],
          symbols_types: response.data.symbols_types || [],
          supported_resolutions: response.data.supported_resolutions || [],
          supports_marks: response.data.supports_marks || false,
          supports_timescale_marks: response.data.supports_timescale_marks || false,
          supports_time: response.data.supports_time || true,
          supports_search: response.data.supports_search || true,
          supports_group_request: response.data.supports_group_request || true,
          currency_codes: response.data.currency_codes,
          units: response.data.units,
          symbols_grouping: response.data.symbols_grouping
        }
        callback(config)
      })
      .catch(error => {
        console.error('Failed to load datafeed configuration:', error)
        // Provide fallback configuration
        callback({
          exchanges: [{ value: 'NASDAQ', name: 'NASDAQ', desc: 'NASDAQ Stock Exchange' }],
          symbols_types: [{ name: 'Stock', value: 'stock' }],
          supported_resolutions: ['1', '5', '15', '30', '60', '240', '1D', '1W', '1M'] as ResolutionString[],
          supports_marks: false,
          supports_timescale_marks: false,
          supports_time: true,
          supports_search: true,
          supports_group_request: true
        } as DatafeedConfiguration)
      })
  }

  /**
   * Called when the chart needs symbol information
   */
  resolveSymbol(
    symbolName: string,
    onResolve: ResolveCallback,
    onError: ErrorCallback,
    extension?: SymbolResolveExtension
  ): void {
    instance.get(`/udf/symbols?symbol=${encodeURIComponent(symbolName)}`)
      .then(response => {
        if (response.data.s === 'error') {
          onError(response.data.errmsg || 'Symbol not found')
          return
        }

        const symbolInfo: LibrarySymbolInfo = {
          name: response.data.name,
          ticker: response.data.ticker,
          description: response.data.description,
          type: response.data.type,
          session: response.data.session,
          timezone: response.data.timezone,
          exchange: response.data.exchange,
          listed_exchange: response.data.listed_exchange,
          format: 'price',
          minmov: response.data.minmov,
          pricescale: response.data.pricescale,
          has_intraday: response.data.has_intraday,
          supported_resolutions: response.data.supported_resolutions,
          has_daily: response.data.has_daily,
          has_weekly_and_monthly: response.data.has_weekly_and_monthly,
          data_status: response.data.data_status,
          sector: response.data.sector,
          industry: response.data.industry,
          currency_code: response.data.currency_code,
          original_currency_code: response.data.original_currency_code,
          unit_id: response.data.unit_id,
          original_unit_id: response.data.original_unit_id,
          unit_conversion_types: response.data.unit_conversion_types,
          subsession_id: response.data.subsession_id,
          subsessions: response.data.subsessions,
          price_source_id: response.data.price_source_id,
          price_sources: response.data.price_sources,
          logo_urls: response.data.logo_urls,
          exchange_logo: response.data.exchange_logo
        }

        onResolve(symbolInfo)
      })
      .catch(error => {
        console.error('Failed to resolve symbol:', error)
        onError('Failed to resolve symbol')
      })
  }

  /**
    * Called when the chart needs historical bars
    */
   getBars(
     symbolInfo: LibrarySymbolInfo,
     resolution: ResolutionString,
     periodParams: PeriodParams,
     onResult: HistoryCallback,
     onError: ErrorCallback
   ): void {
     const fromSeconds = this.normalizeTimestampToSeconds(periodParams.from)
     const toSeconds = this.normalizeTimestampToSeconds(periodParams.to)

     const params = new URLSearchParams({
       symbol: symbolInfo.ticker || symbolInfo.name,
       resolution: resolution,
       from_ts: fromSeconds.toString(),
       to_ts: toSeconds.toString()
     })

     if (periodParams.countBack) {
       params.set('countback', periodParams.countBack.toString())
     }

     instance.get(`/udf/history?${params.toString()}`)
       .then(response => {
         if (response.data.s === 'error') {
           onError(response.data.errmsg || 'Failed to get historical data')
           return
         }

         if (response.data.s === 'no_data') {
           console.warn(
             '[TradingViewUDF] no_data response',
             { symbol: symbolInfo.ticker || symbolInfo.name, resolution, fromSeconds, toSeconds, countBack: periodParams.countBack }
           )
           onResult([], { noData: true })
           return
         }

         // Convert UDF format to Bar array
         const bars: Bar[] = []
         const { t, o, h, l, c, v } = response.data

         for (let i = 0; i < t.length; i++) {
           bars.push({
            time: this.normalizeTimestampToMs(t[i]),
             open: o[i],
             high: h[i],
             low: l[i],
             close: c[i],
             volume: v[i]
           })
         }

         onResult(bars, { noData: false })
       })
       .catch(error => {
         console.error('Failed to get bars:', error)
         onError('Failed to get historical data')
       })
   }

  /**
   * Called when the chart needs to search for symbols
   */
  searchSymbols(
    userInput: string,
    exchange: string,
    symbolType: string,
    onResult: SearchSymbolsCallback
  ): void {
    const params = new URLSearchParams({
      q: userInput,
      type: symbolType,
      exchange: exchange,
      limit: '50'
    })

    instance.get(`/udf/search?${params.toString()}`)
      .then(response => {
        if (response.data.s === 'error') {
          onResult([])
          return
        }

        const symbols: SearchSymbolResultItem[] = (response.data || []).map((sym: any) => ({
          symbol: sym.ticker || sym.name,
          full_name: `${sym.exchange}:${sym.ticker || sym.name}`,
          description: sym.description,
          exchange: sym.exchange,
          ticker: sym.ticker,
          type: sym.type
        }))

        onResult(symbols)
      })
      .catch(error => {
        console.error('Failed to search symbols:', error)
        onResult([])
      })
  }

  /**
   * Called when the chart wants to receive real-time updates
   */
  subscribeBars(
    symbolInfo: LibrarySymbolInfo,
    resolution: ResolutionString,
    onTick: SubscribeBarsCallback,
    listenerGuid: string,
    onResetCacheNeededCallback: () => void
  ): void {
    const subscriptionKey = listenerGuid

    // Store subscription info
    this.subscriptions.set(subscriptionKey, {
      symbolInfo,
      resolution,
      onTick,
      listenerGuid
    })

    console.log(`[TradingViewUDF] Subscribing to ${symbolInfo.ticker || symbolInfo.name}:${resolution} with listener ${listenerGuid}`)

    // Send subscription message via websocket
    if (websocketService.isConnected()) {
      websocketService.sendMessage({
        type: 'subscribe_chart',
        data: {
          symbol: symbolInfo.ticker || symbolInfo.name,
          resolution: resolution,
          listenerGuid: listenerGuid
        }
      })
      console.log(`[TradingViewUDF] Subscribed to ${symbolInfo.ticker} with listener ${listenerGuid} via websocket`)
    } else {
      console.warn('[TradingViewUDF] WebSocket not connected, chart subscription not sent')
    }
  }

  /**
   * Called when the chart no longer wants updates
   */
  unsubscribeBars(listenerGuid: string): void {
    const subscription = this.subscriptions.get(listenerGuid)
    if (subscription) {
       // Send unsubscription message via websocket
       if (websocketService.isConnected()) {
         websocketService.sendMessage({
           type: 'unsubscribe_chart',
           data: {
             listenerGuid: listenerGuid
           }
         })
         console.log(`Unsubscribed from listener ${listenerGuid} via websocket`)
       }

       this.subscriptions.delete(listenerGuid)
     }
   }

  private handleChartUpdate(updateData: any): void {
    const { symbol, resolution, bar } = updateData

    console.log(`[TradingViewUDF] Received chart update: ${symbol}:${resolution}`, bar)

    // Find subscriptions that match this symbol and resolution
    let deliveredCount = 0
    for (const [listenerGuid, subscription] of this.subscriptions) {
      if ((subscription.symbolInfo.ticker === symbol || subscription.symbolInfo.name === symbol) &&
          subscription.resolution === resolution) {
        try {
          // Call the onTick callback with the bar data
          const normalizedBar = {
            ...bar,
            time: this.normalizeTimestampToMs(bar?.time)
          }
          subscription.onTick(normalizedBar)
          console.log(`[TradingViewUDF] Delivered chart update for ${symbol}:${resolution} to listener ${listenerGuid}`)
          deliveredCount++
        } catch (error) {
          console.error(`[TradingViewUDF] Error calling onTick for listener ${listenerGuid}:`, error)
        }
      }
    }

    if (deliveredCount === 0) {
      console.warn(`[TradingViewUDF] No matching subscriptions found for ${symbol}:${resolution}`)
    }
  }
}

export default TradingViewUDFDatafeed