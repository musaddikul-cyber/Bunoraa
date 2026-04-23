/**
 * Frontend Performance Monitoring
 * Tracks Core Web Vitals, errors, and API performance
 */

import { onCLS, onFCP, onFCP as onFID, onLCP, onTTFB, type Metric } from "web-vitals";

// Performance metrics config
const CONFIG = {
  // Only report in production unless debug mode is enabled
  debug: process.env.NODE_ENV === "development",
  // Report URL - can be configured to send to backend or analytics
  reportEndpoint: "/api/analytics/performance",
  // Sampling rate (1.0 = 100%)
  sampleRate: 0.1,
  // Metrics threshold for warnings (in ms)
  thresholds: {
    LCP: 2500,
    FID: 100,
    CLS: 0.1,
    FCP: 1800,
    TTFB: 800,
  },
};

// Store metrics locally
interface PerformanceReport {
  timestamp: number;
  url: string;
  userAgent: string;
  metrics: Record<string, Metric | null>;
  errors: PerformanceError[];
  apiCalls: ApiMetrics[];
}

interface PerformanceError {
  message: string;
  stack?: string;
  timestamp: number;
  url: string;
}

interface ApiMetrics {
  endpoint: string;
  duration: number;
  status: number;
  timestamp: number;
}

// In-memory store for current session
const sessionMetrics: ApiMetrics[] = [];
const sessionErrors: PerformanceError[] = [];

/**
 * Initialize performance monitoring
 */
export function initPerformanceMonitoring() {
  if (typeof window === "undefined") return;

  // Sample users to prevent overwhelming analytics
  if (Math.random() > CONFIG.sampleRate) return;

  // Report Core Web Vitals
  onLCP((metric) => reportMetric("LCP", metric));
  onFID((metric) => reportMetric("FID", metric));
  onCLS((metric) => reportMetric("CLS", metric));
  onFCP((metric) => reportMetric("FCP", metric));
  onTTFB((metric) => reportMetric("TTFB", metric));

  // Track JavaScript errors
  window.addEventListener("error", (event) => {
    trackError({
      message: event.message,
      stack: event.error?.stack,
      timestamp: Date.now(),
      url: window.location.href,
    });
  });

  // Track unhandled promise rejections
  window.addEventListener("unhandledrejection", (event) => {
    trackError({
      message: String(event.reason),
      stack: event.reason?.stack,
      timestamp: Date.now(),
      url: window.location.href,
    });
  });

  // Track resource loading
  if (window.performance && "getEntriesByType" in window.performance) {
    window.addEventListener("load", () => {
      // Allow time for all resources to complete
      setTimeout(() => {
        trackResourceMetrics();
      }, 5000);
    });
  }

  // Periodic batch report
  if (CONFIG.debug) {
    setInterval(() => {
      console.log("[Performance] Session metrics:", {
        apiCalls: sessionMetrics.length,
        errors: sessionErrors.length,
      });
    }, 60000);
  }
}

/**
 * Report a performance metric
 */
function reportMetric(name: string, metric: Metric) {
  // Check if metric exceeds threshold
  const threshold = CONFIG.thresholds[name as keyof typeof CONFIG.thresholds];
  const isPoor = threshold && metric.value > threshold;

  if (CONFIG.debug) {
    console.log(`[Performance] ${name}:`, metric.value, isPoor ? "(POOR)" : "(GOOD)");
  }

  // Send to analytics
  const payload = {
    name,
    value: metric.value,
    rating: metric.rating,
    delta: metric.delta,
    id: metric.id,
    navigationType: metric.navigationType,
    url: window.location.href,
    timestamp: Date.now(),
  };

  // Use sendBeacon if available for reliable delivery
  if (navigator.sendBeacon) {
    navigator.sendBeacon(CONFIG.reportEndpoint, JSON.stringify(payload));
  } else {
    // Fallback to fetch with timeout
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 2000); // 2s timeout
    
    fetch(CONFIG.reportEndpoint, {
      method: "POST",
      body: JSON.stringify(payload),
      keepalive: true,
      headers: { "Content-Type": "application/json" },
      signal: controller.signal,
    }).catch(() => {
      // Silent fail - analytics shouldn't break functionality
    }).finally(() => {
      clearTimeout(timeout);
    });
  }

  // Report to console for poor metrics in production
  if (!CONFIG.debug && isPoor) {
    console.warn(`[Performance] Poor ${name}: ${metric.value}`);
  }
}

/**
 * Track an error
 */
function trackError(error: PerformanceError) {
  sessionErrors.push(error);

  if (sessionErrors.length > 100) {
    sessionErrors.shift(); // Keep only recent 100 errors
  }

  // Send to analytics
  if (navigator.sendBeacon) {
    navigator.sendBeacon("/api/analytics/error", JSON.stringify(error));
  }
}

/**
 * API call tracking
 */
export function trackApiCall(endpoint: string, duration: number, status: number) {
  sessionMetrics.push({
    endpoint,
    duration,
    status,
    timestamp: Date.now(),
  });

  // Keep only last 100 calls
  if (sessionMetrics.length > 100) {
    sessionMetrics.shift();
  }

  // Log slow API calls
  if (duration > 1000) {
    if (CONFIG.debug) {
      console.warn(`[Performance] Slow API call: ${endpoint} took ${duration}ms`);
    }
  }
}

/**
 * Track resource loading metrics
 */
function trackResourceMetrics() {
  const resources = window.performance.getEntriesByType("resource");
  
  const metrics = {
    totalResources: resources.length,
    totalTransferSize: 0,
    slowResources: [] as Array<{ name: string; duration: number }>,
  };

  resources.forEach((resource) => {
    const r = resource as PerformanceResourceTiming;
    if (r.transferSize) {
      metrics.totalTransferSize += r.transferSize;
    }

    // Track slow resources (> 1s)
    if (r.duration > 1000) {
      metrics.slowResources.push({
        name: r.name.split("/").pop() || r.name,
        duration: Math.round(r.duration),
      });
    }
  });

  if (CONFIG.debug && metrics.slowResources.length > 0) {
    console.log("[Performance] Slow resources:", metrics.slowResources);
  }
}

/**
 * Get current performance report
 */
export function getPerformanceReport(): PerformanceReport {
  return {
    timestamp: Date.now(),
    url: typeof window !== "undefined" ? window.location.href : "",
    userAgent: typeof navigator !== "undefined" ? navigator.userAgent : "",
    metrics: {},
    errors: [...sessionErrors],
    apiCalls: [...sessionMetrics],
  };
}

/**
 * Measure function execution time
 */
// eslint-disable-next-line @typescript-eslint/no-explicit-any
export function measure<T extends (...args: any[]) => any>(
  name: string,
  fn: T
): T {
  return ((...args: Parameters<T>): ReturnType<T> => {
    const start = performance.now();
    const result = fn(...args);

    // Handle both sync and async
    const measureEnd = () => {
      const duration = performance.now() - start;
      if (CONFIG.debug) {
        console.log(`[Performance] ${name} took ${duration.toFixed(2)}ms`);
      }
    };

    if (result instanceof Promise) {
      result.then(measureEnd, measureEnd);
    } else {
      measureEnd();
    }

    return result;
  }) as T;
}

/**
 * Hook for tracking react component render performance
 */
export function usePerformanceMark(componentName: string) {
  if (typeof window === "undefined") return;

  const start = performance.now();
  
  return () => {
    const duration = performance.now() - start;
    if (CONFIG.debug && duration > 16) {
      console.warn(`[Performance] ${componentName} render took ${duration.toFixed(2)}ms`);
    }
  };
}
