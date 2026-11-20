import React, { useCallback, useEffect, useRef, useState } from 'react';
import {
  Delete,
  ExternalLink,
  Search,
  Sparkles,
  Trash2,
} from 'lucide-react';
import { useVision } from '../vision/VisionContext';
import { VisionViewport } from './VisionViewport';
import { Point2D } from '../types/vision';

const KEY_ROWS: string[][] = [
  ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0'],
  ['Q', 'W', 'E', 'R', 'T', 'Y', 'U', 'I', 'O', 'P'],
  ['A', 'S', 'D', 'F', 'G', 'H', 'J', 'K', 'L'],
  ['SHIFT', 'Z', 'X', 'C', 'V', 'B', 'N', 'M', 'BACK'],
  ['CLEAR', 'SPACE', 'SEARCH'],
];

interface SearchResultItem {
  title: string;
  description: string;
  url: string;
}

export const VirtualSearchView: React.FC = () => {
  const {
    frameStateRef,
    hands,
    inputMode,
    setInputMode,
    cameraActive,
    settings,
    updateSimConfig,
    addLog,
  } = useVision();

  const [query, setQuery] = useState('');
  const [shiftActive, setShiftActive] = useState(false);
  const [typingEnabled, setTypingEnabled] = useState(true);
  const [selectionMode, setSelectionMode] = useState<'pinch' | 'dwell'>('dwell');
  const [dwellMs, setDwellMs] = useState(650);

  const [hoveredKey, setHoveredKey] = useState<string | null>(null);
  const [dwellProgress, setDwellProgress] = useState(0);
  const [cursorNorm, setCursorNorm] = useState<Point2D | null>(null);

  const [results, setResults] = useState<SearchResultItem[]>([]);
  const [searching, setSearching] = useState(false);
  const [searchError, setSearchError] = useState<string | null>(null);
  const [lastTypedKey, setLastTypedKey] = useState<string | null>(null);

  const keyboardContainerRef = useRef<HTMLDivElement | null>(null);
  const keyRefs = useRef<Record<string, HTMLButtonElement | null>>({});
  const hoverStartRef = useRef<{ key: string; ts: number } | null>(null);
  const wasPinchingRef = useRef(false);
  const lastTypeTsRef = useRef(0);

  const runLiveSearch = useCallback(
    async (text: string) => {
      const clean = text.trim();
      if (!clean) return;
      setSearching(true);
      setSearchError(null);
      addLog('Keyboard', 'Search Submitted', `Query: "${clean}"`);

      try {
        const res = await fetch(
          `https://en.wikipedia.org/w/api.php?action=opensearch&origin=*&limit=6&search=${encodeURIComponent(clean)}`
        );
        if (!res.ok) throw new Error('Search request failed');
        const data = (await res.json()) as [string, string[], string[], string[]];
        const titles = data[1] ?? [];
        const descs = data[2] ?? [];
        const urls = data[3] ?? [];
        const mapped: SearchResultItem[] = titles.map((t, i) => ({
          title: t,
          description: descs[i] || 'Open full reference or Google Search results.',
          url: urls[i] || `https://www.google.com/search?q=${encodeURIComponent(t)}`,
        }));
        setResults(mapped);
      } catch {
        setSearchError('Could not fetch inline knowledge preview. Use the Google Search link above.');
      } finally {
        setSearching(false);
      }
    },
    [addLog]
  );

  const handleKeyPress = useCallback(
    (key: string) => {
      setLastTypedKey(key);
      if (key === 'BACK') {
        setQuery((q) => q.slice(0, -1));
      } else if (key === 'CLEAR') {
        setQuery('');
        setResults([]);
      } else if (key === 'SPACE') {
        setQuery((q) => q + ' ');
      } else if (key === 'SHIFT') {
        setShiftActive((s) => !s);
      } else if (key === 'SEARCH') {
        setQuery((current) => {
          void runLiveSearch(current);
          return current;
        });
      } else {
        const char = shiftActive ? key.toUpperCase() : key.toLowerCase();
        setQuery((q) => q + char);
      }
    },
    [runLiveSearch, shiftActive]
  );

  // Hand-tracking hit-testing across virtual keyboard buttons
  useEffect(() => {
    let rafId = 0;
    const mirror = inputMode === 'webcam' ? settings.mirror : false;

    const tick = () => {
      const hand = frameStateRef.current.hands[0];
      const kbEl = keyboardContainerRef.current;

      if (typingEnabled && hand && kbEl && (cameraActive || inputMode === 'simulation')) {
        // Map normalized hand index tip [0..1] across keyboard container
        const nx = Math.min(1, Math.max(0, mirror ? 1 - hand.indexTip.x : hand.indexTip.x));
        const ny = Math.min(1, Math.max(0, hand.indexTip.y));
        setCursorNorm({ x: nx, y: ny });

        const kbRect = kbEl.getBoundingClientRect();
        const px = kbRect.left + nx * kbRect.width;
        const py = kbRect.top + ny * kbRect.height;

        let foundKey: string | null = null;
        for (const [k, btn] of Object.entries(keyRefs.current)) {
          if (!btn) continue;
          const r = btn.getBoundingClientRect();
          if (px >= r.left && px <= r.right && py >= r.top && py <= r.bottom) {
            foundKey = k;
            break;
          }
        }

        setHoveredKey(foundKey);
        const now = performance.now();

        if (foundKey) {
          if (selectionMode === 'dwell') {
            if (!hoverStartRef.current || hoverStartRef.current.key !== foundKey) {
              hoverStartRef.current = { key: foundKey, ts: now };
              setDwellProgress(0);
            } else {
              const elapsed = now - hoverStartRef.current.ts;
              const pct = Math.min(1, elapsed / dwellMs);
              setDwellProgress(pct);
              if (pct >= 1 && now - lastTypeTsRef.current > 450) {
                lastTypeTsRef.current = now;
                hoverStartRef.current = { key: foundKey, ts: now + 350 };
                setDwellProgress(0);
                handleKeyPress(foundKey);
              }
            }
          } else {
            // Pinch-to-Type mode
            setDwellProgress(0);
            if (hand.isPinching && !wasPinchingRef.current && now - lastTypeTsRef.current > 350) {
              lastTypeTsRef.current = now;
              handleKeyPress(foundKey);
            }
          }
        } else {
          hoverStartRef.current = null;
          setDwellProgress(0);
        }

        wasPinchingRef.current = hand.isPinching;
      } else {
        setCursorNorm(null);
        setHoveredKey(null);
        setDwellProgress(0);
        hoverStartRef.current = null;
        wasPinchingRef.current = false;
      }

      rafId = requestAnimationFrame(tick);
    };

    rafId = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(rafId);
  }, [
    cameraActive,
    dwellMs,
    frameStateRef,
    handleKeyPress,
    inputMode,
    selectionMode,
    settings.mirror,
    typingEnabled,
  ]);

  const googleSearchHref = `https://www.google.com/search?q=${encodeURIComponent(query.trim() || 'Computer Vision MediaPipe')}`;

  return (
    <div className="space-y-6">
      {/* Top Query Bar & Typing Mode Controls */}
      <div className="flex flex-wrap items-center justify-between gap-4 rounded-lg border border-slate-200 dark:border-slate-800 bg-white dark:bg-[#0E131D] px-4 py-3">
        <div className="flex flex-wrap items-center gap-3">
          <button
            type="button"
            onClick={() => setTypingEnabled((v) => !v)}
            className={`rounded-md px-3.5 py-1.5 text-xs font-semibold transition-colors cursor-pointer whitespace-nowrap shrink-0 ${
              typingEnabled
                ? 'bg-cyan-500 text-slate-950 hover:bg-cyan-400'
                : 'border border-slate-300 dark:border-slate-700 text-slate-700 dark:text-slate-300'
            }`}
          >
            {typingEnabled ? 'Hand Typing Active' : 'Hand Typing Paused'}
          </button>

          <div className="flex items-center gap-2 text-xs">
            <span className="text-slate-500 dark:text-slate-400">Selection:</span>
            <select
              value={selectionMode}
              onChange={(e) => setSelectionMode(e.target.value as 'pinch' | 'dwell')}
              className="rounded border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-[#090C12] px-2.5 py-1 text-xs text-slate-800 dark:text-slate-200"
            >
              <option value="dwell">Hover Dwell Timer</option>
              <option value="pinch">Index Aim + Pinch</option>
            </select>
          </div>

          {selectionMode === 'dwell' && (
            <label className="flex items-center gap-2 text-xs text-slate-600 dark:text-slate-300">
              <span>Dwell</span>
              <input
                type="range"
                min={350}
                max={1200}
                step={50}
                value={dwellMs}
                onChange={(e) => setDwellMs(Number(e.target.value))}
                className="w-20 accent-cyan-500"
              />
              <span className="font-mono tabular-nums">{dwellMs}ms</span>
            </label>
          )}
        </div>

        <div className="text-xs font-mono text-slate-500 dark:text-slate-400">
          {lastTypedKey ? `Last key: ${lastTypedKey}` : 'Point index finger over keyboard'}
        </div>
      </div>

      {/* Main Split: Keyboard + Query Stage (8 cols) & Camera Tracking Feed (4 cols) */}
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-12 items-start">
        <div className="lg:col-span-8 space-y-4">
          {/* Live Search Input Box */}
          <form
            onSubmit={(e) => {
              e.preventDefault();
              void runLiveSearch(query);
            }}
            className="flex flex-wrap items-center gap-2 rounded-lg border border-slate-200 dark:border-slate-800 bg-white dark:bg-[#0E131D] p-3"
          >
            <Search className="h-4 w-4 text-slate-400 ml-1 shrink-0" />
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Point at keys below to compose your Google Search query…"
              className="min-w-0 flex-1 bg-transparent px-2 py-1 text-sm font-medium text-slate-900 dark:text-slate-100 focus:outline-none"
            />
            <button
              type="button"
              onClick={() => handleKeyPress('BACK')}
              aria-label="Backspace"
              className="rounded border border-slate-200 dark:border-slate-700 p-2 text-slate-500 hover:text-slate-900 dark:hover:text-white cursor-pointer"
            >
              <Delete className="h-3.5 w-3.5" />
            </button>
            <button
              type="button"
              onClick={() => handleKeyPress('CLEAR')}
              aria-label="Clear query"
              className="rounded border border-slate-200 dark:border-slate-700 p-2 text-slate-500 hover:text-rose-400 cursor-pointer"
            >
              <Trash2 className="h-3.5 w-3.5" />
            </button>
            <button
              type="submit"
              disabled={!query.trim() || searching}
              className="rounded-md bg-cyan-500 px-3.5 py-2 text-xs font-semibold text-slate-950 hover:bg-cyan-400 disabled:opacity-40 cursor-pointer whitespace-nowrap shrink-0"
            >
              {searching ? 'Searching…' : 'Search Preview'}
            </button>
            <a
              href={googleSearchHref}
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-1.5 rounded-md border border-slate-300 dark:border-slate-700 bg-slate-100 dark:bg-slate-800 px-3.5 py-2 text-xs font-medium text-slate-900 dark:text-slate-100 hover:border-cyan-500 whitespace-nowrap shrink-0"
            >
              Open Google
              <ExternalLink className="h-3.5 w-3.5" />
            </a>
          </form>

          {/* Virtual Keyboard Grid with Live Hand Cursor */}
          <div
            ref={keyboardContainerRef}
            onMouseMove={(e) => {
              if (inputMode === 'simulation') {
                const rect = e.currentTarget.getBoundingClientRect();
                updateSimConfig({
                  gesture: 'Pointing Index',
                  shapeTrajectory: 'none',
                  pointerTarget: {
                    x: (e.clientX - rect.left) / rect.width,
                    y: (e.clientY - rect.top) / rect.height,
                  },
                });
              }
            }}
            className="relative select-none rounded-lg border border-slate-200 dark:border-slate-800 bg-white dark:bg-[#0E131D] p-4 space-y-2"
          >
            {KEY_ROWS.map((row, rowIdx) => (
              <div key={rowIdx} className="flex justify-center gap-1.5">
                {row.map((key) => {
                  const isHovered = hoveredKey === key;
                  const isWide =
                    key === 'SPACE'
                      ? 'flex-1 max-w-xs'
                      : key === 'SHIFT' || key === 'BACK' || key === 'CLEAR' || key === 'SEARCH'
                      ? 'px-4'
                      : 'w-10 sm:w-12';

                  const displayLabel =
                    key.length === 1
                      ? shiftActive
                        ? key.toUpperCase()
                        : key.toLowerCase()
                      : key === 'BACK'
                      ? '⌫ Back'
                      : key === 'SPACE'
                      ? 'Space'
                      : key === 'SHIFT'
                      ? shiftActive
                        ? '⇧ CAPS'
                        : '⇧ Shift'
                      : key === 'CLEAR'
                      ? 'Clear'
                      : 'Search';

                  return (
                    <button
                      key={key}
                      ref={(el) => {
                        keyRefs.current[key] = el;
                      }}
                      type="button"
                      onClick={() => handleKeyPress(key)}
                      className={`relative overflow-hidden h-11 ${isWide} rounded-md border text-xs font-mono font-semibold transition-colors cursor-pointer whitespace-nowrap ${
                        isHovered
                          ? 'border-cyan-400 bg-cyan-500/20 text-cyan-200'
                          : key === 'SEARCH'
                          ? 'border-cyan-500/50 bg-cyan-500/15 text-cyan-600 dark:text-cyan-300 hover:bg-cyan-500/25'
                          : key === 'SHIFT' && shiftActive
                          ? 'border-amber-500 bg-amber-500/15 text-amber-400'
                          : 'border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-[#090C12] text-slate-800 dark:text-slate-200 hover:border-slate-400 dark:hover:border-slate-500'
                      }`}
                    >
                      {isHovered && selectionMode === 'dwell' && dwellProgress > 0 && (
                        <span
                          className="pointer-events-none absolute inset-y-0 left-0 bg-cyan-500/30"
                          style={{ width: `${Math.round(dwellProgress * 100)}%` }}
                        />
                      )}
                      <span className="relative z-10">{displayLabel}</span>
                    </button>
                  );
                })}
              </div>
            ))}

            {/* Floating Hand Cursor Indicator */}
            {cursorNorm && (
              <div
                className="pointer-events-none absolute z-20 -translate-x-1/2 -translate-y-1/2"
                style={{
                  left: `${cursorNorm.x * 100}%`,
                  top: `${cursorNorm.y * 100}%`,
                }}
              >
                <div className="h-5 w-5 rounded-full border-2 border-cyan-400 bg-cyan-400/20 shadow-sm" />
              </div>
            )}
          </div>

          {/* Search Preview Results */}
          <section className="rounded-lg border border-slate-200 dark:border-slate-800 bg-white dark:bg-[#0E131D] p-4">
            <div className="flex items-center justify-between border-b border-slate-100 dark:border-slate-800 pb-2.5">
              <div className="flex items-center gap-2">
                <Sparkles className="h-4 w-4 text-cyan-500" />
                <h2 className="font-display text-sm font-semibold text-slate-900 dark:text-slate-100">
                  Instant Search Results
                </h2>
              </div>
              {query.trim() && (
                <a
                  href={googleSearchHref}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-xs text-cyan-500 hover:underline inline-flex items-center gap-1"
                >
                  View on Google
                  <ExternalLink className="h-3 w-3" />
                </a>
              )}
            </div>

            {searchError && (
              <p className="mt-3 text-xs text-amber-400">{searchError}</p>
            )}

            {results.length === 0 ? (
              <p className="py-5 text-center text-xs text-slate-500 dark:text-slate-400">
                Type a query using your index finger and hit &ldquo;Search&rdquo; to inspect live web knowledge results or launch Google Search.
              </p>
            ) : (
              <div className="mt-3 divide-y divide-slate-100 dark:divide-slate-800">
                {results.map((item) => (
                  <a
                    key={item.url}
                    href={item.url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="block py-2.5 hover:bg-slate-50 dark:hover:bg-slate-800/40 rounded px-2 transition-colors"
                  >
                    <div className="text-xs font-semibold text-cyan-600 dark:text-cyan-400">
                      {item.title}
                    </div>
                    <div className="mt-0.5 text-xs text-slate-600 dark:text-slate-400 line-clamp-2">
                      {item.description}
                    </div>
                  </a>
                ))}
              </div>
            )}
          </section>
        </div>

        {/* Right: Docked Hand Tracking Feed */}
        <div className="lg:col-span-4 space-y-4">
          <VisionViewport
            aspectClass="aspect-video"
            cursorPoint={cursorNorm}
            compact
          />

          <div className="rounded-lg border border-slate-200 dark:border-slate-800 bg-white dark:bg-[#0E131D] p-4 space-y-2.5 text-xs">
            <div className="font-display font-semibold text-slate-900 dark:text-slate-100">
              Touchless Typing Telemetry
            </div>
            <div className="flex justify-between">
              <span className="text-slate-500 dark:text-slate-400">Hovered Key</span>
              <span className="font-mono font-semibold text-cyan-500">
                {hoveredKey ?? 'None'}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-500 dark:text-slate-400">Hand Pose</span>
              <span className="font-mono text-slate-700 dark:text-slate-200">
                {hands[0]?.gesture ?? 'No Hand'}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-500 dark:text-slate-400">Pinch State</span>
              <span className="font-mono text-slate-700 dark:text-slate-200">
                {hands[0]?.isPinching ? 'Pinched' : 'Open'}
              </span>
            </div>
            {inputMode !== 'simulation' && !cameraActive && (
              <button
                type="button"
                onClick={() => setInputMode('simulation')}
                className="w-full rounded-md border border-slate-700 py-1.5 text-xs font-medium text-slate-300 hover:bg-slate-800 cursor-pointer"
              >
                Use Simulator + Mouse Aim
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};
