import { useEffect, useState, useRef, useCallback } from 'react';

export interface SSEOptions {
  onMessage?: (event: string, data: any) => void;
  onConnect?: () => void;
  onError?: (err: Event) => void;
  onDisconnect?: () => void;
  autoConnect?: boolean;
}

export function useSSE(url: string | null, options: SSEOptions = {}) {
  const [status, setStatus] = useState<'idle' | 'connecting' | 'connected' | 'error'>('idle');
  const eventSourceRef = useRef<EventSource | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const heartbeatTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const reconnectAttemptRef = useRef<number>(0);

  const { onMessage, onConnect, onError, onDisconnect, autoConnect = true } = options;

  const disconnect = useCallback(() => {
    if (eventSourceRef.current) {
      eventSourceRef.current.close();
      eventSourceRef.current = null;
    }
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }
    if (heartbeatTimeoutRef.current) {
      clearTimeout(heartbeatTimeoutRef.current);
      heartbeatTimeoutRef.current = null;
    }
    setStatus('idle');
    onDisconnect?.();
  }, [onDisconnect]);

  const connect = useCallback(() => {
    if (!url) return;
    disconnect();
    
    setStatus('connecting');
    const es = new EventSource(url);
    eventSourceRef.current = es;

    const resetHeartbeat = () => {
      if (heartbeatTimeoutRef.current) {
        clearTimeout(heartbeatTimeoutRef.current);
      }
      // Reconnect if no event (including keepalive) received for 35s
      heartbeatTimeoutRef.current = setTimeout(() => {
        console.warn('SSE heartbeat lost. Reconnecting...');
        connect();
      }, 35000);
    };

    es.onopen = () => {
      setStatus('connected');
      reconnectAttemptRef.current = 0;
      resetHeartbeat();
      onConnect?.();
    };

    es.onerror = (e) => {
      setStatus('error');
      onError?.(e);
      es.close();

      // Exponential backoff reconnect
      const delay = Math.min(1000 * Math.pow(2, reconnectAttemptRef.current), 30000);
      reconnectAttemptRef.current += 1;
      
      console.log(`SSE connection error. Retrying in ${delay}ms...`);
      reconnectTimeoutRef.current = setTimeout(() => {
        connect();
      }, delay);
    };

    // Listen to standard message events
    es.onmessage = (event) => {
      resetHeartbeat();
      try {
        const parsed = JSON.parse(event.data);
        onMessage?.('message', parsed);
      } catch (err) {
        onMessage?.('message', event.data);
      }
    };

    // Generic event dispatcher to support custom events like "token", "retrieval_start", etc.
    const customEvents = [
      'retrieval_start',
      'retrieval_complete',
      'token',
      'done',
      'error',
      'keepalive',
      'evaluation',
    ];
    customEvents.forEach((evtName) => {
      es.addEventListener(evtName, (event: MessageEvent) => {
        resetHeartbeat();
        let parsed = event.data;
        try {
          parsed = JSON.parse(event.data);
        } catch (_) {}
        onMessage?.(evtName, parsed);
      });
    });

  }, [url, disconnect, onConnect, onError, onMessage]);

  useEffect(() => {
    if (autoConnect && url) {
      connect();
    }
    return () => {
      disconnect();
    };
  }, [url, autoConnect, connect, disconnect]);

  return { status, connect, disconnect };
}
