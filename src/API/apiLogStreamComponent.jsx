import { useEffect, useRef } from "react";
import { baseUrl } from "./apiConsts";

const INITIAL_DELAY = 1500;
const MAX_DELAY     = 30000;

// Streams server-sent log events to the parent via onNewLog callback.
// Reconnects with exponential backoff on disconnect or error.
const LogStreamComponent = ({ onNewLog }) => {
  const esRef    = useRef(null);
  const onNewLogRef = useRef(onNewLog);

  useEffect(() => {
    onNewLogRef.current = onNewLog;
  }, [onNewLog]);

  useEffect(() => {
    let cancelled = false;
    let delay     = INITIAL_DELAY;
    let timer     = null;

    const connect = () => {
      if (cancelled) return;

      const es = new EventSource(`${baseUrl}/stream-logs`);
      esRef.current = es;

      es.onopen = () => {
        delay = INITIAL_DELAY;
      };

      es.onmessage = (event) => {
        onNewLogRef.current?.(event.data);
        delay = INITIAL_DELAY;
      };

      es.onerror = () => {
        es.close();
        esRef.current = null;
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
      esRef.current?.close();
      esRef.current = null;
    };
  }, []);

  return null;
};

export default LogStreamComponent;
