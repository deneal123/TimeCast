import { useEffect, useRef } from "react";
import { baseUrl } from "../API/apiConsts";

const INITIAL_DELAY = 1500;
const MAX_DELAY     = 30000;

/**
 * Connects to the SSE log stream and calls onNewLog for each event.
 * Reconnects with exponential backoff (1.5s → 30s) on error or disconnect.
 */
const useLogStream = (onNewLog) => {
  // Keep a stable ref so the SSE handler never captures a stale closure.
  const onNewLogRef = useRef(onNewLog);
  useEffect(() => { onNewLogRef.current = onNewLog; }, [onNewLog]);

  useEffect(() => {
    let cancelled = false;
    let delay     = INITIAL_DELAY;
    let timer     = null;
    let es        = null;

    const connect = () => {
      if (cancelled) return;

      es = new EventSource(`${baseUrl}/stream-logs`);

      es.onopen = () => { delay = INITIAL_DELAY; };

      es.onmessage = (event) => {
        onNewLogRef.current?.(event.data);
        delay = INITIAL_DELAY;
      };

      es.onerror = () => {
        es.close();
        es = null;
        if (!cancelled) {
          timer = setTimeout(connect, delay);
          delay = Math.min(delay * 2, MAX_DELAY);
        }
      };
    };

    connect();

    return () => {
      cancelled = true;
      clearTimeout(timer);
      es?.close();
    };
  }, []); // intentionally empty — reconnect logic is self-contained
};

export default useLogStream;
