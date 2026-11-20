import React, { useCallback, useEffect, useRef, useState } from 'react';
import {
  Download,
  Eraser,
  Pencil,
  Save,
  Sparkles,
  Trash2,
  Undo2,
  Wand2,
} from 'lucide-react';
import { useVision } from '../vision/VisionContext';
import { VisionViewport } from './VisionViewport';
import {
  Point2D,
  SavedCustomShape,
  ShapeRecognitionResult,
  Stroke,
} from '../types/vision';
import {
  loadCustomShapes,
  recognizeAirShape,
  saveCustomShapes,
} from '../vision/shapeRecognition';
import { SimShapeTrajectory } from '../vision/syntheticSimulator';

const BRUSH_COLORS = [
  { name: 'Cyan', hex: '#22D3EE' },
  { name: 'Emerald', hex: '#10B981' },
  { name: 'Amber', hex: '#F59E0B' },
  { name: 'Rose', hex: '#F43F5E' },
  { name: 'White', hex: '#F8FAFC' },
];

export const AirStudioView: React.FC = () => {
  const {
    frameStateRef,
    hands,
    inputMode,
    setInputMode,
    cameraActive,
    settings,
    simConfig,
    updateSimConfig,
    addLog,
  } = useVision();

  const [strokes, setStrokes] = useState<Stroke[]>([]);
  const [activeStroke, setActiveStroke] = useState<Stroke | null>(null);
  const activeStrokeRef = useRef<Stroke | null>(null);

  const [brushColor, setBrushColor] = useState('#22D3EE');
  const [brushWidth, setBrushWidth] = useState(5);
  const [eraserMode, setEraserMode] = useState(false);
  const [drawTrigger, setDrawTrigger] = useState<'index' | 'pinch'>('index');
  const [cursorPoint, setCursorPoint] = useState<Point2D | null>(null);
  const [isDrawingNow, setIsDrawingNow] = useState(false);

  const [recognition, setRecognition] = useState<ShapeRecognitionResult | null>(null);
  const [customNameInput, setCustomNameInput] = useState('');
  const [customShapes, setCustomShapes] = useState<SavedCustomShape[]>(() => loadCustomShapes());

  const customShapesRef = useRef(customShapes);
  customShapesRef.current = customShapes;

  const isMouseDrawingRef = useRef(false);
  const drawContainerRef = useRef<HTMLDivElement | null>(null);

  const runShapeRecognition = useCallback(
    (points: Point2D[]) => {
      if (points.length < 8) return;
      const res = recognizeAirShape(points, customShapesRef.current);
      if (!res) return;
      setRecognition(res);

      if (res.customMatchName) {
        setCustomShapes((prev) => {
          const updated = prev.map((s) =>
            s.customName === res.customMatchName
              ? { ...s, matchCount: s.matchCount + 1 }
              : s
          );
          saveCustomShapes(updated);
          return updated;
        });
        addLog(
          'AirDraw',
          'Custom Shape Matched',
          `"${res.customMatchName}" (${res.geometricShape}, ${Math.round((res.customMatchConfidence ?? 0) * 100)}%)`
        );
      } else {
        addLog(
          'AirDraw',
          'Shape Classified',
          `${res.geometricShape} · ${(res.confidence * 100).toFixed(0)}% conf`
        );
      }
    },
    [addLog]
  );

  // Air-drawing tracking loop reading hand landmarks at ~60fps
  useEffect(() => {
    let rafId = 0;
    const mirror = inputMode === 'webcam' ? settings.mirror : false;

    const step = () => {
      const hand = frameStateRef.current.hands[0];
      if (hand && !isMouseDrawingRef.current) {
        const pt: Point2D = {
          x: mirror ? 1 - hand.indexTip.x : hand.indexTip.x,
          y: hand.indexTip.y,
        };
        setCursorPoint(pt);

        const wantsDraw =
          drawTrigger === 'pinch'
            ? hand.isPinching
            : hand.fingerStates.index &&
              !hand.fingerStates.middle &&
              !hand.fingerStates.ring &&
              !hand.fingerStates.pinky;

        if (wantsDraw) {
          setIsDrawingNow(true);
          const current = activeStrokeRef.current;
          if (!current) {
            const started: Stroke = {
              id: `stroke-${Date.now()}`,
              points: [pt],
              color: brushColor,
              width: brushWidth,
              isEraser: eraserMode,
            };
            activeStrokeRef.current = started;
            setActiveStroke(started);
          } else {
            const last = current.points[current.points.length - 1];
            const d = Math.hypot(pt.x - last.x, pt.y - last.y);
            if (d > 0.004) {
              const updated: Stroke = {
                ...current,
                points: [...current.points, pt],
              };
              activeStrokeRef.current = updated;
              setActiveStroke(updated);
            }
          }
        } else {
          setIsDrawingNow(false);
          if (activeStrokeRef.current) {
            const finished = activeStrokeRef.current;
            activeStrokeRef.current = null;
            setActiveStroke(null);
            if (finished.points.length >= 6) {
              setStrokes((prev) => [...prev, finished]);
              if (!finished.isEraser) {
                runShapeRecognition(finished.points);
              }
            }
          }
        }
      } else if (!hand && !isMouseDrawingRef.current) {
        setCursorPoint(null);
        setIsDrawingNow(false);
      }

      rafId = requestAnimationFrame(step);
    };

    if (cameraActive || inputMode === 'simulation') {
      rafId = requestAnimationFrame(step);
    }
    return () => cancelAnimationFrame(rafId);
  }, [
    brushColor,
    brushWidth,
    cameraActive,
    drawTrigger,
    eraserMode,
    frameStateRef,
    inputMode,
    runShapeRecognition,
    settings.mirror,
  ]);

  // Pointer fallback handlers so users can also draw directly on canvas with mouse/touch
  const getRelPoint = (e: React.PointerEvent<HTMLDivElement>): Point2D | null => {
    const el = drawContainerRef.current;
    if (!el) return null;
    const rect = el.getBoundingClientRect();
    return {
      x: Math.min(1, Math.max(0, (e.clientX - rect.left) / rect.width)),
      y: Math.min(1, Math.max(0, (e.clientY - rect.top) / rect.height)),
    };
  };

  const handlePointerDown = (e: React.PointerEvent<HTMLDivElement>) => {
    const pt = getRelPoint(e);
    if (!pt) return;
    isMouseDrawingRef.current = true;
    setIsDrawingNow(true);
    const started: Stroke = {
      id: `stroke-${Date.now()}`,
      points: [pt],
      color: brushColor,
      width: brushWidth,
      isEraser: eraserMode,
    };
    activeStrokeRef.current = started;
    setActiveStroke(started);
  };

  const handlePointerMove = (e: React.PointerEvent<HTMLDivElement>) => {
    if (!isMouseDrawingRef.current || !activeStrokeRef.current) return;
    const pt = getRelPoint(e);
    if (!pt) return;
    const last = activeStrokeRef.current.points[activeStrokeRef.current.points.length - 1];
    if (Math.hypot(pt.x - last.x, pt.y - last.y) > 0.004) {
      const updated = {
        ...activeStrokeRef.current,
        points: [...activeStrokeRef.current.points, pt],
      };
      activeStrokeRef.current = updated;
      setActiveStroke(updated);
    }
  };

  const handlePointerUp = () => {
    if (!isMouseDrawingRef.current) return;
    isMouseDrawingRef.current = false;
    setIsDrawingNow(false);
    const finished = activeStrokeRef.current;
    activeStrokeRef.current = null;
    setActiveStroke(null);
    if (finished && finished.points.length >= 6) {
      setStrokes((prev) => [...prev, finished]);
      if (!finished.isEraser) {
        runShapeRecognition(finished.points);
      }
    }
  };

  const handleSaveCustomShape = (e: React.FormEvent) => {
    e.preventDefault();
    const trimmed = customNameInput.trim();
    if (!trimmed || !recognition) return;

    const newEntry: SavedCustomShape = {
      id: `shape-${Date.now()}`,
      customName: trimmed,
      baseShape: recognition.geometricShape,
      normalizedPoints: recognition.normalizedPoints,
      aspectRatio: recognition.metrics.aspectRatio,
      circularity: recognition.metrics.circularity,
      vertices: recognition.metrics.vertices,
      createdAt: Date.now(),
      matchCount: 0,
    };

    const next = [newEntry, ...customShapes];
    setCustomShapes(next);
    saveCustomShapes(next);
    setCustomNameInput('');
    setRecognition({
      ...recognition,
      customMatchName: trimmed,
      customMatchConfidence: 1,
    });
    addLog('AirDraw', 'Custom Shape Saved', `Named ${recognition.geometricShape} as "${trimmed}"`);
  };

  const handleDeleteCustomShape = (id: string) => {
    const next = customShapes.filter((s) => s.id !== id);
    setCustomShapes(next);
    saveCustomShapes(next);
  };

  const handleExportPNG = () => {
    const canvas = document.createElement('canvas');
    canvas.width = 1280;
    canvas.height = 720;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.fillStyle = '#090C12';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    for (const s of strokes) {
      if (s.points.length < 2) continue;
      ctx.strokeStyle = s.isEraser ? '#090C12' : s.color;
      ctx.lineWidth = s.width * 1.5;
      ctx.lineCap = 'round';
      ctx.lineJoin = 'round';
      ctx.beginPath();
      ctx.moveTo(s.points[0].x * canvas.width, s.points[0].y * canvas.height);
      for (let i = 1; i < s.points.length; i++) {
        ctx.lineTo(s.points[i].x * canvas.width, s.points[i].y * canvas.height);
      }
      ctx.stroke();
    }
    const link = document.createElement('a');
    link.download = `visionx-air-drawing-${Date.now()}.png`;
    link.href = canvas.toDataURL('image/png');
    link.click();
    addLog('AirDraw', 'Exported Drawing', 'Saved air canvas snapshot as PNG');
  };

  const triggerSimulatedShape = (shape: SimShapeTrajectory) => {
    setInputMode('simulation');
    setStrokes([]);
    setRecognition(null);
    updateSimConfig({ gesture: 'Pointing Index', shapeTrajectory: shape });
    addLog('AirDraw', 'Simulating Shape', `Tracing air trajectory: ${shape}`);
  };

  const primaryHand = hands[0];

  return (
    <div className="space-y-6">
      {/* Toolbar */}
      <div className="flex flex-wrap items-center justify-between gap-4 rounded-lg border border-slate-200 dark:border-slate-800 bg-white dark:bg-[#0E131D] px-4 py-3">
        <div className="flex flex-wrap items-center gap-3">
          {/* Colors */}
          <div className="flex items-center gap-1.5">
            {BRUSH_COLORS.map((c) => (
              <button
                key={c.hex}
                type="button"
                aria-label={`Brush color ${c.name}`}
                onClick={() => {
                  setBrushColor(c.hex);
                  setEraserMode(false);
                }}
                className={`h-6 w-6 rounded-full border-2 transition-transform cursor-pointer ${
                  brushColor === c.hex && !eraserMode
                    ? 'scale-110 border-white ring-2 ring-cyan-500'
                    : 'border-slate-700 hover:scale-105'
                }`}
                style={{ backgroundColor: c.hex }}
              />
            ))}
          </div>

          <span className="h-5 w-px bg-slate-200 dark:bg-slate-800" />

          <label className="flex items-center gap-2 text-xs text-slate-600 dark:text-slate-300">
            <span>Size</span>
            <input
              type="range"
              min={2}
              max={16}
              value={brushWidth}
              onChange={(e) => setBrushWidth(Number(e.target.value))}
              className="w-20 accent-cyan-500"
            />
            <span className="font-mono w-5 tabular-nums">{brushWidth}</span>
          </label>

          <span className="h-5 w-px bg-slate-200 dark:bg-slate-800" />

          <div className="flex items-center gap-1.5 text-xs">
            <span className="text-slate-500 dark:text-slate-400">Draw Trigger:</span>
            <select
              value={drawTrigger}
              onChange={(e) => setDrawTrigger(e.target.value as 'index' | 'pinch')}
              className="rounded border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-[#090C12] px-2 py-1 text-xs text-slate-800 dark:text-slate-200"
            >
              <option value="index">Pointing Index</option>
              <option value="pinch">Thumb-Index Pinch</option>
            </select>
          </div>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          <button
            type="button"
            onClick={() => setEraserMode((v) => !v)}
            className={`inline-flex items-center gap-1.5 rounded-md border px-3 py-1.5 text-xs font-medium transition-colors cursor-pointer whitespace-nowrap shrink-0 ${
              eraserMode
                ? 'border-amber-500 bg-amber-500/15 text-amber-400'
                : 'border-slate-200 dark:border-slate-700 text-slate-700 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-800'
            }`}
          >
            <Eraser className="h-3.5 w-3.5" />
            Eraser
          </button>
          <button
            type="button"
            onClick={() => setStrokes((prev) => prev.slice(0, -1))}
            disabled={strokes.length === 0}
            className="inline-flex items-center gap-1.5 rounded-md border border-slate-200 dark:border-slate-700 px-3 py-1.5 text-xs font-medium text-slate-700 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-800 disabled:opacity-40 cursor-pointer whitespace-nowrap shrink-0"
          >
            <Undo2 className="h-3.5 w-3.5" />
            Undo
          </button>
          <button
            type="button"
            onClick={() => {
              const lastStroke = strokes[strokes.length - 1];
              if (lastStroke) runShapeRecognition(lastStroke.points);
            }}
            disabled={strokes.length === 0}
            className="inline-flex items-center gap-1.5 rounded-md border border-cyan-500/50 bg-cyan-500/10 px-3 py-1.5 text-xs font-medium text-cyan-600 dark:text-cyan-300 hover:bg-cyan-500/20 disabled:opacity-40 cursor-pointer whitespace-nowrap shrink-0"
          >
            <Wand2 className="h-3.5 w-3.5" />
            Classify Last
          </button>
          <button
            type="button"
            onClick={() => {
              setStrokes([]);
              setRecognition(null);
              updateSimConfig({ shapeTrajectory: 'none' });
            }}
            className="inline-flex items-center gap-1.5 rounded-md border border-slate-200 dark:border-slate-700 px-3 py-1.5 text-xs font-medium text-slate-700 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-800 cursor-pointer whitespace-nowrap shrink-0"
          >
            <Trash2 className="h-3.5 w-3.5" />
            Clear
          </button>
          <button
            type="button"
            onClick={handleExportPNG}
            disabled={strokes.length === 0}
            className="inline-flex items-center gap-1.5 rounded-md bg-cyan-500 px-3 py-1.5 text-xs font-semibold text-slate-950 hover:bg-cyan-400 disabled:opacity-40 cursor-pointer whitespace-nowrap shrink-0"
          >
            <Download className="h-3.5 w-3.5" />
            Export PNG
          </button>
        </div>
      </div>

      {/* Main Split: Interactive Air Canvas + Shape Recognition & Custom Naming */}
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-12 items-start">
        <div className="lg:col-span-8 space-y-3">
          <div
            ref={drawContainerRef}
            onPointerDown={handlePointerDown}
            onPointerMove={handlePointerMove}
            onPointerUp={handlePointerUp}
            onPointerLeave={handlePointerUp}
            className="relative cursor-crosshair select-none touch-none"
          >
            <VisionViewport
              aspectClass="aspect-video"
              strokes={strokes}
              activeStroke={activeStroke}
              cursorPoint={cursorPoint}
              isDrawing={isDrawingNow}
            />
          </div>

          <div className="flex flex-wrap items-center justify-between gap-3 rounded-lg border border-slate-200 dark:border-slate-800 bg-white dark:bg-[#0E131D] px-4 py-2.5 text-xs">
            <div className="flex items-center gap-3 text-slate-600 dark:text-slate-300">
              <span className="font-mono">
                Status:{' '}
                <strong className={isDrawingNow ? 'text-emerald-400' : 'text-slate-300'}>
                  {isDrawingNow ? 'Drawing Active' : 'Hover / Pen Up'}
                </strong>
              </span>
              <span className="text-slate-400">·</span>
              <span>
                {primaryHand
                  ? `Hand: ${primaryHand.gesture}`
                  : 'Use hand gesture or drag mouse directly on canvas'}
              </span>
            </div>

            <div className="flex flex-wrap items-center gap-1.5">
              <span className="text-slate-400 mr-1">Test Trajectory:</span>
              {(['circle', 'triangle', 'square', 'star', 'arrow'] as const).map((shape) => (
                <button
                  key={shape}
                  type="button"
                  onClick={() => triggerSimulatedShape(shape)}
                  className={`rounded border px-2 py-0.5 text-[11px] capitalize transition-colors cursor-pointer ${
                    simConfig.shapeTrajectory === shape && inputMode === 'simulation'
                      ? 'border-cyan-500 bg-cyan-500/15 text-cyan-300'
                      : 'border-slate-200 dark:border-slate-700 text-slate-600 dark:text-slate-400 hover:border-slate-500'
                  }`}
                >
                  {shape}
                </button>
              ))}
              {simConfig.shapeTrajectory !== 'none' && (
                <button
                  type="button"
                  onClick={() => {
                    updateSimConfig({ shapeTrajectory: 'none', gesture: 'Open Palm' });
                  }}
                  className="rounded border border-amber-500/40 bg-amber-500/10 px-2 py-0.5 text-[11px] text-amber-300 cursor-pointer"
                >
                  Lift Pen & Classify
                </button>
              )}
            </div>
          </div>
        </div>

        {/* Right Panel: Shape Classification & Custom Naming */}
        <div className="lg:col-span-4 space-y-4">
          <section className="rounded-lg border border-slate-200 dark:border-slate-800 bg-white dark:bg-[#0E131D] p-4">
            <div className="flex items-center justify-between border-b border-slate-100 dark:border-slate-800/80 pb-2.5">
              <div className="flex items-center gap-2">
                <Sparkles className="h-4 w-4 text-cyan-500" />
                <h2 className="font-display text-sm font-semibold text-slate-900 dark:text-slate-100">
                  Geometric & Custom Recognition
                </h2>
              </div>
              <span className="font-mono text-[11px] text-slate-400">CV Contour</span>
            </div>

            {!recognition ? (
              <div className="py-6 text-center text-xs text-slate-500 dark:text-slate-400">
                Draw a shape in the air (or drag with your pointer) and pause/open palm to classify geometry and assign a custom name.
              </div>
            ) : (
              <div className="mt-3 space-y-3">
                {recognition.customMatchName && (
                  <div className="rounded-md border border-emerald-500/40 bg-emerald-500/10 p-3">
                    <div className="text-[11px] font-mono uppercase tracking-wider text-emerald-400">
                      Custom Shape Library Match
                    </div>
                    <div className="mt-0.5 flex items-baseline justify-between">
                      <span className="font-display text-lg font-bold text-slate-900 dark:text-white">
                        {recognition.customMatchName}
                      </span>
                      <span className="font-mono text-xs text-emerald-400 tabular-nums">
                        {Math.round((recognition.customMatchConfidence ?? 0) * 100)}% match
                      </span>
                    </div>
                  </div>
                )}

                <div className="flex items-baseline justify-between">
                  <div>
                    <span className="block text-[11px] text-slate-500 dark:text-slate-400">
                      Geometric Primitive
                    </span>
                    <span className="font-display text-xl font-bold text-slate-900 dark:text-slate-100">
                      {recognition.geometricShape}
                    </span>
                  </div>
                  <span className="font-mono text-sm font-semibold text-cyan-600 dark:text-cyan-400 tabular-nums">
                    {(recognition.confidence * 100).toFixed(1)}%
                  </span>
                </div>

                <div className="grid grid-cols-3 gap-2 rounded-md border border-slate-100 dark:border-slate-800 bg-slate-50 dark:bg-[#090C12] p-2.5 font-mono text-[11px] tabular-nums">
                  <div>
                    <span className="block text-slate-500">Vertices</span>
                    <span className="text-slate-800 dark:text-slate-200">{recognition.metrics.vertices}</span>
                  </div>
                  <div>
                    <span className="block text-slate-500">Circularity</span>
                    <span className="text-slate-800 dark:text-slate-200">{recognition.metrics.circularity}</span>
                  </div>
                  <div>
                    <span className="block text-slate-500">Aspect</span>
                    <span className="text-slate-800 dark:text-slate-200">{recognition.metrics.aspectRatio}</span>
                  </div>
                </div>

                {/* Custom Shape Naming Form */}
                <form onSubmit={handleSaveCustomShape} className="space-y-2 border-t border-slate-100 dark:border-slate-800 pt-3">
                  <label className="block text-xs font-medium text-slate-700 dark:text-slate-300">
                    Assign Custom Name to This Shape
                  </label>
                  <div className="flex gap-2">
                    <input
                      type="text"
                      value={customNameInput}
                      onChange={(e) => setCustomNameInput(e.target.value)}
                      placeholder='e.g. "My Planet"'
                      maxLength={32}
                      className="min-w-0 flex-1 rounded-md border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-[#090C12] px-3 py-1.5 text-xs text-slate-900 dark:text-slate-100 focus:border-cyan-500 focus:outline-none"
                    />
                    <button
                      type="submit"
                      disabled={!customNameInput.trim()}
                      className="inline-flex items-center gap-1.5 rounded-md bg-cyan-500 px-3 py-1.5 text-xs font-semibold text-slate-950 hover:bg-cyan-400 disabled:opacity-40 cursor-pointer whitespace-nowrap shrink-0"
                    >
                      <Save className="h-3.5 w-3.5" />
                      Save
                    </button>
                  </div>
                </form>
              </div>
            )}
          </section>

          {/* Saved Custom Shape Library */}
          <section className="rounded-lg border border-slate-200 dark:border-slate-800 bg-white dark:bg-[#0E131D] p-4">
            <div className="flex items-center justify-between border-b border-slate-100 dark:border-slate-800/80 pb-2.5">
              <div className="flex items-center gap-2">
                <Pencil className="h-4 w-4 text-cyan-500" />
                <h2 className="font-display text-sm font-semibold text-slate-900 dark:text-slate-100">
                  Saved Shape Memory
                </h2>
              </div>
              <span className="font-mono text-[11px] text-slate-400 tabular-nums">
                {customShapes.length} saved
              </span>
            </div>

            {customShapes.length === 0 ? (
              <p className="py-5 text-center text-xs text-slate-500 dark:text-slate-400">
                No named shapes saved yet. Draw any shape and save a custom name above.
              </p>
            ) : (
              <div className="mt-3 max-h-64 space-y-2 overflow-y-auto pr-1">
                {customShapes.map((item) => {
                  const svgPath = item.normalizedPoints
                    .map((p, idx) => `${idx === 0 ? 'M' : 'L'}${(p.x * 16 + 20).toFixed(1)},${(p.y * 16 + 20).toFixed(1)}`)
                    .join(' ');
                  return (
                    <div
                      key={item.id}
                      className="flex items-center justify-between gap-3 rounded-md border border-slate-100 dark:border-slate-800 bg-slate-50 dark:bg-[#090C12] p-2.5"
                    >
                      <div className="flex items-center gap-3 min-w-0">
                        <svg viewBox="0 0 40 40" className="h-9 w-9 shrink-0 rounded border border-slate-200 dark:border-slate-800 bg-white dark:bg-[#0E131D]">
                          <path d={svgPath} fill="none" stroke="#22D3EE" strokeWidth="2" strokeLinecap="round" />
                        </svg>
                        <div className="min-w-0">
                          <div className="truncate text-xs font-semibold text-slate-900 dark:text-slate-100">
                            {item.customName}
                          </div>
                          <div className="text-[11px] text-slate-500 dark:text-slate-400 tabular-nums">
                            {item.baseShape} · {item.matchCount} matches
                          </div>
                        </div>
                      </div>
                      <button
                        type="button"
                        onClick={() => handleDeleteCustomShape(item.id)}
                        aria-label={`Delete ${item.customName}`}
                        className="rounded p-1.5 text-slate-400 hover:bg-rose-500/10 hover:text-rose-400 cursor-pointer shrink-0"
                      >
                        <Trash2 className="h-3.5 w-3.5" />
                      </button>
                    </div>
                  );
                })}
              </div>
            )}
          </section>
        </div>
      </div>
    </div>
  );
};
