import { useState, useCallback, useRef, useEffect } from 'react';
import { createPortal } from 'react-dom';

export function useTooltip() {
  const [tooltipState, setTooltipState] = useState({
    visible: false,
    x: 0,
    y: 0,
    title: '',
    body: '',
    position: 'top',
  });

  const showTooltip = useCallback((event, { title, body, position = 'top' }) => {
    const rect = event.currentTarget.getBoundingClientRect();
    let x = rect.left + rect.width / 2;
    let y = rect.top;

    if (position === 'bottom') {
      y = rect.bottom;
    } else if (position === 'right') {
      x = rect.right;
      y = rect.top + rect.height / 2;
    } else if (position === 'left') {
      x = rect.left;
      y = rect.top + rect.height / 2;
    }

    setTooltipState({
      visible: true,
      x,
      y,
      title,
      body,
      position,
    });
  }, []);

  const hideTooltip = useCallback(() => {
    setTooltipState((prev) => ({ ...prev, visible: false }));
  }, []);

  return {
    tooltipState,
    showTooltip,
    hideTooltip,
  };
}

export function TooltipPortal({ tooltipState }) {
  const { visible, x, y, title, body, position } = tooltipState;
  const tooltipRef = useRef(null);
  const [coords, setCoords] = useState({ top: -9999, left: -9999, ready: false });

  useEffect(() => {
    if (visible && tooltipRef.current) {
      const rect = tooltipRef.current.getBoundingClientRect();
      let top = y;
      let left = x;

      if (position === 'top') {
        top = y - rect.height - 8;
        left = x - rect.width / 2;
      } else if (position === 'bottom') {
        top = y + 8;
        left = x - rect.width / 2;
      } else if (position === 'right') {
        top = y - rect.height / 2;
        left = x + 8;
      } else if (position === 'left') {
        top = y - rect.height / 2;
        left = x - rect.width - 8;
      }

      // Viewport safety check
      const padding = 10;
      if (left < padding) left = padding;
      if (left + rect.width > window.innerWidth - padding) {
        left = window.innerWidth - rect.width - padding;
      }
      if (top < padding) top = y + 20; // Flip to bottom if top clips

      setCoords({ top, left, ready: true });
    } else {
      setCoords({ top: -9999, left: -9999, ready: false });
    }
  }, [visible, x, y, position]);

  if (!visible || (!title && !body)) return null;

  return createPortal(
    <div
      ref={tooltipRef}
      className={`tooltip-bubble visible pos-${position}`}
      style={{
        position: 'fixed',
        top: `${coords.top}px`,
        left: `${coords.left}px`,
        zIndex: 99999,
        pointerEvents: 'none',
        opacity: coords.ready ? 1 : 0,
        transition: 'opacity 0.12s ease',
      }}
    >
      {title && <div className="tooltip-title">{title}</div>}
      {body && <div className="tooltip-body">{body}</div>}
    </div>,
    document.body
  );
}
