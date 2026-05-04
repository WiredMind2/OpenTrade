import React, { useEffect, useRef, useState, forwardRef, useImperativeHandle } from "react";
import { useTheme } from "./ThemeProvider";
import type {
  IChartingLibraryWidget,
  ChartingLibraryWidgetOptions,
} from "../../public/charting_library/charting_library.d.ts";
import TradingViewUDFDatafeed from "../services/tradingViewUDF";
import { attachProjectionManager, detachProjectionManager } from "../lib/ChartProjectionManager";
import { projectStrategy } from "../api/strategies";
import type { ProjectionPoint, PredictionProjection } from "../types";
import type { NewsArticle } from "@/api/news";

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
   /** Strategy name for projections */
   strategyName?: string;
   /** Strategy parameters */
   params?: Record<string, any>;
   /** Projection horizon in days */
   horizon?: number;
   /** Projection mode */
   mode?: string;
   /** Show prediction projections */
   showPredictionProjections?: boolean;
   /** Prediction projection data */
   predictionProjections?: PredictionProjection[];
   /** Callback when symbol changes in TradingView */
   onSymbolChange?: (symbol: string) => void;
   /** News articles to display as markers on chart */
   newsArticles?: NewsArticle[];
   /** Callback when a news marker is clicked */
   onNewsClick?: (article: NewsArticle) => void;
}

/**
 * Public API exposed via ref
 */
export interface OHLCChartRef {
  setProjectionStrategy: (strategyName: string, params?: Record<string, any>, horizon?: number, mode?: string) => void;
  clearProjections: () => void;
  getLatestPrice: () => number | null;
  getLatestTime: () => number | null;
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
const OHLCChart = forwardRef<OHLCChartRef, OHLCChartProps>(
   ({
     symbol = "AAPL",
     showVolume = false,
     showConfidence = false,
     height = "400px",
     bullishColor = "#10b981",
     bearishColor = "#ef4444",
     strategyName = "moving_average",
     params = {},
     horizon = 30,
     mode = "price",
     showPredictionProjections = false,
     predictionProjections = [],
     onSymbolChange,
   }, ref) => {
       const { theme } = useTheme();
       const isDark = theme === 'dark';

       const containerRef = useRef<HTMLDivElement>(null);
       const widgetRef = useRef<IChartingLibraryWidget | null>(null);
       const containerIdRef = useRef<string>(
         `tradingview_${Math.random().toString(36).substr(2, 9)}`
       );
       const projectionEntitiesRef = useRef<any[]>([]);
       const predictionEntitiesRef = useRef<any[]>([]);
      const normalizeTimeToSeconds = (value: unknown): number | null => {
        const ts = Number(value)
        if (!Number.isFinite(ts)) return null
        return ts > 1e11 ? Math.floor(ts / 1000) : Math.floor(ts)
      }

       // State for projection settings
       const [projectionSettings, setProjectionSettings] = useState({
         strategyName,
         params,
         horizon,
         mode,
       });

       // State for prediction projections
       const [predictionSettings, setPredictionSettings] = useState({
         showPredictionProjections,
         predictionProjections,
       });

       /**
        * Render prediction projections on the chart
        */
       const renderPredictionProjections = () => {
         if (!widgetRef.current) return;

          

        let chart;
        try {
          chart = widgetRef.current.chart();
        } catch (error) {
          console.warn("[OHLCChart] Chart not ready or already removed");
          return;
        }
        if (!chart) return;

         // Clear existing prediction entities
         predictionEntitiesRef.current.forEach(entityId => {
           try {
             chart.removeEntity(entityId);
           } catch (e) {
             console.warn("[OHLCChart] Failed to remove existing prediction entity:", e);
           }
         });
         predictionEntitiesRef.current = [];

         if (!predictionSettings.showPredictionProjections) return;

         // Filter projections for current symbol
         const relevantProjections = predictionSettings.predictionProjections.filter(
           p => p.ticker === symbol
         );

         relevantProjections.forEach(projection => {
          const drawProjectionLine = (
            valueKey: 'price' | 'upperBound' | 'lowerBound',
            style: { linestyle: number; linewidth: number; linecolor: string; transparency: number }
          ) => {
            for (let i = 1; i < projection.points.length; i++) {
              const start = projection.points[i - 1];
              const end = projection.points[i];
              const startValue = start[valueKey];
              const endValue = end[valueKey];
              if (typeof startValue !== 'number' || typeof endValue !== 'number') continue;

              const segmentId = chart.createMultipointShape([
                { time: start.time, price: startValue },
                { time: end.time, price: endValue },
              ], {
                shape: 'trend_line',
                lock: true,
                disableSelection: true,
                disableSave: true,
                overrides: style,
              });

              if (segmentId) {
                predictionEntitiesRef.current.push(segmentId);
              }
            }
          };

          // Draw explicit line segments to avoid any closed polygon behavior.
          drawProjectionLine('price', {
            linestyle: 0, // SOLID
            linewidth: 2,
            linecolor: projection.color,
            transparency: 0
          });
          drawProjectionLine('upperBound', {
            linestyle: 2, // DASHED
            linewidth: 1,
            linecolor: projection.color,
            transparency: 35
          });
          drawProjectionLine('lowerBound', {
            linestyle: 2, // DASHED
            linewidth: 1,
            linecolor: projection.color,
            transparency: 35
          });

           // Add start marker for each projection
           if (projection.points.length > 0) {
             const startPoint = projection.points[0];
             const startMarkerId = chart.createShape({
               time: startPoint.time,
               price: startPoint.price
             }, {
               shape: 'arrow_right',
               lock: true,
               disableSelection: true,
               disableSave: true,
               overrides: {
                 color: projection.color,
                 transparency: 0,
                 size: 1
               }
             });

             if (startMarkerId) {
               predictionEntitiesRef.current.push(startMarkerId);
             }
           }
         });
       };

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
              "header_compare" as any,
              "header_undo_redo" as any,
              "header_screenshot" as any,
              "header_fullscreen_button" as any,
            ],
            enabled_features: [
              "study_templates" as any,
              "left_toolbar" as any,
            ],
            custom_css_url: 'tv-theme.css',
            client_id: "tradingview.com",
            user_id: "public_user_id",
            fullscreen: false,
            autosize: true,
            studies_overrides: {},
            theme: isDark ? "dark" : "light",
            timezone: "Etc/UTC",
            toolbar_bg: isDark ? "#171717" : "#F5F5F5",
            loading_screen: {
              backgroundColor: isDark ? "#171717" : "#FFFFFF",
              foregroundColor: isDark ? "#a3a3a3" : "#555555",
            },
            overrides: {
              "paneProperties.background": isDark ? "#171717" : "#FFFFFF",
              "paneProperties.backgroundType": "solid",
              "paneProperties.vertGridProperties.color": isDark ? "#2a2a2a" : "#E0E3EB",
              "paneProperties.horzGridProperties.color": isDark ? "#2a2a2a" : "#E0E3EB",
              "mainSeriesProperties.style": 2,
              "mainSeriesProperties.candleStyle.upColor": bullishColor || "#089981",
              "mainSeriesProperties.candleStyle.downColor": bearishColor || "#F23645",
              "mainSeriesProperties.candleStyle.borderUpColor": bullishColor || "#089981",
              "mainSeriesProperties.candleStyle.borderDownColor": bearishColor || "#F23645",
              "mainSeriesProperties.candleStyle.wickUpColor": bullishColor || "#089981",
              "mainSeriesProperties.candleStyle.wickDownColor": bearishColor || "#F23645",
              "scalesProperties.textColor": isDark ? "#787B86" : "#555555",
              "scalesProperties.lineColor": isDark ? "#2A2E39" : "#E0E3EB",
            },
          };
// Create the chart widget
widgetRef.current = new TradingView.widget(widgetOptions);

          // Attach projection manager when chart is ready
          widgetRef.current.onChartReady(() => {
            console.log("[OHLCChart] Chart ready, attaching projection manager");

            // Force theme overrides after localStorage restore
            try {
              const chart = widgetRef.current?.chart() as any
              if (chart) {
                chart.applyOverrides({
                  'paneProperties.background': isDark ? '#171717' : '#FFFFFF',
                  'paneProperties.backgroundType': 'solid',
                  'paneProperties.vertGridProperties.color': isDark ? '#2a2a2a' : '#E0E3EB',
                  'paneProperties.horzGridProperties.color': isDark ? '#2a2a2a' : '#E0E3EB',
                  'scalesProperties.textColor': isDark ? '#a3a3a3' : '#555555',
                  'scalesProperties.lineColor': isDark ? '#2a2a2a' : '#E0E3EB',
                })
              }
            } catch { /* ignore */ }

            // Listen for symbol changes in TradingView
            if (onSymbolChange) {
              widgetRef.current!.activeChart().onSymbolChanged().subscribe(null, (() => {
                const currentSymbol = widgetRef.current!.activeChart().symbol();
                console.log("[OHLCChart] Symbol changed to:", currentSymbol);
                // Extract just the ticker symbol (e.g., "AAPL" from "NASDAQ:AAPL")
                const ticker = currentSymbol.split(':').pop() || currentSymbol;
                onSymbolChange(ticker);
              }) as any);
            }

            const projectionOptions = {
              onProjectionRequest: async (startPoint: { time: number; price: number }) => {
                try {
                  console.log("[OHLCChart] Requesting projection:", startPoint, projectionSettings);

                  const response = await projectStrategy(
                    projectionSettings.strategyName,
                    symbol,
                    new Date(startPoint.time * 1000).toISOString(),
                    startPoint.price,
                    projectionSettings.params,
                    projectionSettings.horizon
                  );

                  // Convert response to ProjectionPoint format
                  const points: ProjectionPoint[] = response.map((point: any) => ({
                    time: new Date(point.time).getTime() / 1000,
                    open: point.price,
                    high: point.price,
                    low: point.price,
                    close: point.price,
                    predicted: true,
                  }));

                  return points;
                } catch (error) {
                  console.error("[OHLCChart] Projection request failed:", error);
                  throw error;
                }
              },
              onProjectionRendered: (points: ProjectionPoint[]) => {
                console.log("[OHLCChart] Rendering projection points:", points);

                if (!widgetRef.current) return;

                const chart = widgetRef.current.chart();
                if (!chart) return;

                // Clear existing projection entities
                projectionEntitiesRef.current.forEach(entityId => {
                  try {
                    chart.removeEntity(entityId);
                  } catch (e) {
                    console.warn("[OHLCChart] Failed to remove existing projection entity:", e);
                  }
                });
                projectionEntitiesRef.current = [];

                // Draw line segments instead of polyline to prevent closed shapes.
                for (let i = 1; i < points.length; i++) {
                  const previous = points[i - 1];
                  const current = points[i];
                  const segmentId = chart.createMultipointShape([
                    { time: previous.time, price: previous.close },
                    { time: current.time, price: current.close },
                  ], {
                    shape: 'trend_line',
                    lock: true,
                    disableSelection: true,
                    disableSave: true,
                    overrides: {
                      linestyle: 2, // DASHED
                      linewidth: 2,
                      linecolor: '#FF6B35', // Orange color for projections
                      transparency: 0
                    }
                  });

                  if (segmentId) {
                    projectionEntitiesRef.current.push(segmentId);
                  }
                }

                // Add start point marker
                if (points.length > 0) {
                  const startPoint = points[0];
                  const startMarkerId = chart.createShape({
                    time: startPoint.time,
                    price: startPoint.close
                  }, {
                    shape: 'arrow_up',
                    lock: true,
                    disableSelection: true,
                    disableSave: true,
                    overrides: {
                      color: '#10B981', // Green for start
                      transparency: 0
                    }
                  });

                  if (startMarkerId) {
                    projectionEntitiesRef.current.push(startMarkerId);
                  }
                }

                // Add buy/sell signal markers (simplified: mark significant price changes)
                for (let i = 1; i < points.length; i++) {
                  const current = points[i];
                  const previous = points[i - 1];
                  const priceChange = current.close - previous.close;
                  const threshold = Math.abs(previous.close * 0.005); // 0.5% change threshold

                  if (Math.abs(priceChange) > threshold) {
                    const signalShape = priceChange > 0 ? 'arrow_up' : 'arrow_down';
                    const signalColor = priceChange > 0 ? '#10B981' : '#EF4444'; // Green for buy, red for sell

                    const signalId = chart.createShape({
                      time: current.time,
                      price: current.close
                    }, {
                      shape: signalShape,
                      lock: true,
                      disableSelection: true,
                      disableSave: true,
                      overrides: {
                        color: signalColor,
                        transparency: 0
                      }
                    });

                    if (signalId) {
                      projectionEntitiesRef.current.push(signalId);
                    }
                  }
                }
              },
            };

            attachProjectionManager(widgetRef.current, projectionOptions);

            // Render prediction projections if enabled
            renderPredictionProjections();
          });

        } catch (error) {
          console.error("Failed to initialize TradingView chart:", error);
        }
      };

      initializeChart();

      return () => {
        if (widgetRef.current) {
          detachProjectionManager();
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
      isDark,
      projectionSettings.strategyName,
      projectionSettings.params,
      projectionSettings.horizon,
      projectionSettings.mode,
    ]);

    // Apply theme to existing widget without full reinit
    useEffect(() => {
      if (!widgetRef.current) return;
      try {
        (widgetRef.current as any).changeTheme(isDark ? 'dark' : 'light').then(() => {
          const chart = widgetRef.current?.chart() as any
          if (!chart) return
          chart.applyOverrides({
            'paneProperties.background': isDark ? '#171717' : '#FFFFFF',
            'paneProperties.backgroundType': 'solid',
            'paneProperties.vertGridProperties.color': isDark ? '#2a2a2a' : '#E0E3EB',
            'paneProperties.horzGridProperties.color': isDark ? '#2a2a2a' : '#E0E3EB',
            'scalesProperties.textColor': isDark ? '#a3a3a3' : '#555555',
            'scalesProperties.lineColor': isDark ? '#2a2a2a' : '#E0E3EB',
          })
        })
      } catch {
        // changeTheme not available on this widget version — full reinit handles it
      }
    }, [isDark])

    // Update prediction settings when props change
    useEffect(() => {
      setPredictionSettings({
        showPredictionProjections,
        predictionProjections,
      });
    }, [showPredictionProjections, predictionProjections]);

    // Re-render prediction projections when settings change
    useEffect(() => {
     if (!widgetRef.current) return;

  const timeout = setTimeout(() => {
    renderPredictionProjections();
    }, 300);
    return () => clearTimeout(timeout);
    }, [predictionSettings, symbol]);

    // Expose public API via ref
    useImperativeHandle(ref, () => ({
      setProjectionStrategy: (strategyName: string, params?: Record<string, any>, horizon?: number, mode?: string) => {
        setProjectionSettings({
          strategyName,
          params: params || {},
          horizon: horizon || 30,
          mode: mode || "price",
        });
      },
      clearProjections: () => {
        // Clear interactive projections
        if (projectionEntitiesRef.current.length > 0 && widgetRef.current) {
          const chart = widgetRef.current.chart();
          if (chart) {
            projectionEntitiesRef.current.forEach(entityId => {
              try {
                chart.removeEntity(entityId);
              } catch (e) {
                console.warn("[OHLCChart] Failed to remove projection entity:", e);
              }
            });
            projectionEntitiesRef.current = [];
          }
        }

        // Clear prediction projections
        if (predictionEntitiesRef.current.length > 0 && widgetRef.current) {
          const chart = widgetRef.current.chart();
          if (chart) {
            predictionEntitiesRef.current.forEach(entityId => {
              try {
                chart.removeEntity(entityId);
              } catch (e) {
                console.warn("[OHLCChart] Failed to remove prediction entity:", e);
              }
            });
            predictionEntitiesRef.current = [];
          }
        }
      },
      getLatestPrice: () => {
        if (!widgetRef.current) return null;

        try {
          const chart = widgetRef.current.chart() as any;
          if (!chart) return null;

          // Get the last bar from the chart
          const lastBar = chart.lastBar();
          if (lastBar && typeof lastBar.close === 'number') {
            return lastBar.close;
          }
          console.warn("[OHLCChart] Latest bar price unavailable");

          // Fallback: try to get price from the price scale
          const priceScale = chart.priceScale();
          if (priceScale) {
            // This might not be the most recent price, but it's a fallback
            return null; // For now, return null if we can't get the last bar
          }
        } catch (error) {
          console.warn("[OHLCChart] Failed to get latest price:", error);
        }

        return null;
      },
      getLatestTime: () => {
        if (!widgetRef.current) return null;

        try {
          const chart = widgetRef.current.chart() as any;
          if (!chart) return null;

          // Get the last bar from the chart
          const lastBar = chart.lastBar();
          if (lastBar && typeof lastBar.time === 'number') {
            const normalizedTime = normalizeTimeToSeconds(lastBar.time);
            if (normalizedTime !== null) {
              return normalizedTime;
            }
          }
          console.warn("[OHLCChart] Latest bar time unavailable or invalid");
        } catch (error) {
          console.warn("[OHLCChart] Failed to get latest time:", error);
        }

        return null;
      },
    }));

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
