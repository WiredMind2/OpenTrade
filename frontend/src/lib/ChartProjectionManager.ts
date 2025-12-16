import type { IChartingLibraryWidget } from '../../public/charting_library/charting_library.d.ts';
import type { ProjectionPoint } from '../types';

export type TradingViewWidget = IChartingLibraryWidget;

export interface ProjectionManagerOptions {
  onProjectionRequest: (
    startPoint: { time: number; price: number },
    strategy: string,
    params: object,
    horizon: number
  ) => Promise<ProjectionPoint[]>;
  onProjectionRendered: (points: ProjectionPoint[]) => void;
}

let attachedWidget: IChartingLibraryWidget | null = null;
let chartUnsubscribe: (() => void) | null = null;

export function attachProjectionManager(
  widget: TradingViewWidget,
  options: ProjectionManagerOptions
): void {
  if (attachedWidget) {
    detachProjectionManager();
  }

  attachedWidget = widget;

  widget.onChartReady(() => {
    const chart = widget.chart();
    if (!chart) return;

    const handleClick = (point: { x: number; y: number }) => {
      // Convert pixel coordinates to time and price
      const time = chart.timeScale().coordinateToTime(point.x);
      const price = chart.priceScale().coordinateToPrice(point.y);

      if (time === null || price === null) return;

      // Get symbol metadata for precision
      const symbolInfo = chart.symbolExt();
      const tickSize = symbolInfo?.tick_size || 0.01; // Default to 0.01 if not available

      // Round price to appropriate precision
      const roundedPrice = Math.round(price / tickSize) * tickSize;

      const startPoint = { time, price: roundedPrice };

      // Default values - could be made configurable
      const strategy = 'default';
      const params = {};
      const horizon = 10;

      options.onProjectionRequest(startPoint, strategy, params, horizon)
        .then((points) => {
          options.onProjectionRendered(points);
        })
        .catch((error) => {
          console.error('Projection request failed:', error);
        });
    };

    // Subscribe to click events by adding DOM event listener to widget container
    // The widget has access to its container through the widget API
    const widgetContainer = document.getElementById(attachedWidget._options.container as string);

    if (widgetContainer) {
      const clickHandler = (event: MouseEvent) => {
        // Get the bounding rectangle of the widget container
        const rect = widgetContainer.getBoundingClientRect();
        // Calculate relative coordinates within the widget
        const x = event.clientX - rect.left;
        const y = event.clientY - rect.top;

        handleClick({ x, y });
      };

      widgetContainer.addEventListener('click', clickHandler);

      // Store the cleanup function
      chartUnsubscribe = () => {
        widgetContainer.removeEventListener('click', clickHandler);
      };
    } else {
      console.warn('[ChartProjectionManager] Could not find widget container for click events');
    }
  });
}

export function detachProjectionManager(): void {
  if (chartUnsubscribe) {
    chartUnsubscribe();
    chartUnsubscribe = null;
  }
  attachedWidget = null;
}