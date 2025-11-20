import React, { useEffect, useRef } from "react";
import type {
  IChartingLibraryWidget,
  ChartingLibraryWidgetOptions,
} from "../../public/charting_library/charting_library.d.ts";
import TradingViewUDFDatafeed from "../services/tradingViewUDF";

/**
 * Candle data point
 */
interface CandleData {
  date: Date;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
  predicted?: number;
}

/**
 * Props for the OHLCChart component
 */
interface OHLCChartProps {
  /** Trading symbol (e.g., 'AAPL') */
  symbol?: string;
  /** Show volume histogram */
  showVolume?: boolean;
  /** Show confidence bands */
  showConfidence?: boolean;
  /** Chart height as CSS string. Default: '400px' */
  height?: string;
  /** Color for bullish candles */
  bullishColor?: string;
  /** Color for bearish candles */
  bearishColor?: string;
}

/**
 * OHLCChart Component
 *
 * A TradingView advanced chart embeddable widget component.
 * Displays candlestick charts for the specified trading symbol.
 *
 * @param symbol - Trading symbol (e.g., 'NASDAQ:AAPL')
 * @param height - Chart container height as CSS string
 *
 * @example
 * ```tsx
 * <OHLCChart
 *   symbol="NASDAQ:AAPL"
 *   height="500px"
 * />
 * ```
 */
const OHLCChart = React.memo<OHLCChartProps>(
  ({
    symbol = "AAPL",
    showVolume = false,
    showConfidence = false,
    height = "400px",
    bullishColor = "#10b981",
    bearishColor = "#ef4444",
  }) => {
    const containerRef = useRef<HTMLDivElement>(null);
    const widgetRef = useRef<IChartingLibraryWidget | null>(null);
    const containerIdRef = useRef<string>(
      `tradingview_${Math.random().toString(36).substr(2, 9)}`
    );

    useEffect(() => {
      if (!containerRef.current) return;

      const initializeChart = async () => {
        try {
          console.log(
            "[OHLCChart] Initializing TradingView chart for symbol:",
            symbol
          );

          // Import the full charting library
          const TradingView = await import(
            "../../public/charting_library/charting_library.esm.js"
          );

          if (widgetRef.current) {
            widgetRef.current.remove();
          }

          // Create datafeed instance
          const datafeed = new TradingViewUDFDatafeed();
          console.log("[OHLCChart] Created datafeed instance");

          // Chart configuration
          const widgetOptions: ChartingLibraryWidgetOptions = {
            symbol: symbol,
            datafeed: datafeed,
            interval: "1D" as any, // ResolutionString
            container: containerIdRef.current,
            library_path: "/charting_library/",
            locale: "en",
            disabled_features: [
              "header_symbol_search" as any,
              "header_compare" as any,
              "header_undo_redo" as any,
              "header_screenshot" as any,
              "header_fullscreen_button" as any,
            ],
            enabled_features: [
              "study_templates" as any,
              "save_chart_properties_to_local_storage" as any,
              "use_localstorage_for_settings" as any,
              "left_toolbar" as any,
            ],
            charts_storage_url: "https://saveload.tradingview.com",
            charts_storage_api_version: "1.1",
            client_id: "tradingview.com",
            user_id: "public_user_id",
            fullscreen: false,
            autosize: true,
            studies_overrides: {},
            theme: "dark",
            timezone: "Etc/UTC",
            toolbar_bg: "#1E222D",
            loading_screen: {
              backgroundColor: "#0D1421",
              foregroundColor: "#2962FF",
            },
            overrides: {
              "paneProperties.background": "#0D1421",
              "paneProperties.backgroundType": "solid",
              "paneProperties.vertGridProperties.color": "#2A2E39",
              "paneProperties.horzGridProperties.color": "#2A2E39",
              "mainSeriesProperties.style": 2, // Candlestick
              "mainSeriesProperties.candleStyle.upColor":
                bullishColor || "#089981",
              "mainSeriesProperties.candleStyle.downColor":
                bearishColor || "#F23645",
              "mainSeriesProperties.candleStyle.borderUpColor":
                bullishColor || "#089981",
              "mainSeriesProperties.candleStyle.borderDownColor":
                bearishColor || "#F23645",
              "mainSeriesProperties.candleStyle.wickUpColor":
                bullishColor || "#089981",
              "mainSeriesProperties.candleStyle.wickDownColor":
                bearishColor || "#F23645",
              "scalesProperties.textColor": "#787B86",
              "scalesProperties.lineColor": "#2A2E39",
            },
          };
// Create the chart widget
widgetRef.current = new TradingView.widget(widgetOptions);

          
        } catch (error) {
          console.error("Failed to initialize TradingView chart:", error);
        }
      };

      initializeChart();

      return () => {
        if (widgetRef.current) {
          widgetRef.current.remove();
          widgetRef.current = null;
        }
      };
    }, [
      symbol,
      showVolume,
      showConfidence,
      height,
      bullishColor,
      bearishColor,
    ]);

    return (
      <div>
        <div
          ref={containerRef}
          id={containerIdRef.current}
          style={{ width: "100%", height }}
        />
      </div>
    );
  }
);

export default OHLCChart;
