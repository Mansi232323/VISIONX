import React, { useEffect, useRef } from 'react';
import { Camera, Cpu, Play } from 'lucide-react';
import { useVision } from '../vision/VisionContext';
import { renderVisionOverlay } from '../vision/canvasRenderer';
import { Point2D, Stroke } from '../types/vision';

interface VisionViewportProps {
  className?: string;
  aspectClass?: string;
  strokes?: Stroke[];
  activeStroke?: Stroke | null;
  cursorPoint?: Point2D | null;
  isDrawing?: boolean;
  compact?: boolean;
}

export const VisionViewport: React.FC<VisionViewportProps> = ({
  className = '',
  aspectClass = 'aspect-video',
  strokes,
  activeStroke,
  cursorPoint,
  isDrawing,
  compact = false,
}) => {
  const {
    videoRef,
    frameStateRef,
    inputMode,
    setInputMode,
    cameraActive,
    cameraError,
    modelsLoading,
    modelLoadError,
    startCamera,
    settings,
    fps,
    inferenceMs,
  } = useVision();

  const videoCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const overlayCanvasRef = useRef<HTMLCanvasElement | null>(null);

  const strokesRef = useRef(strokes);
  strokesRef.current = strokes;
  const activeStrokeRef = useRef(activeStroke);
  activeStrokeRef.current = activeStroke;
  const cursorRef = useRef(cursorPoint);
  cursorRef.current = cursorPoint;
  const isDrawingRef = useRef(isDrawing);
  isDrawingRef.current = isDrawing;

  useEffect(() => {
    let rafId = 0;

    const draw = () => {
      const vCanvas = videoCanvasRef.current;
      const oCanvas = overlayCanvasRef.current;
      if (!vCanvas || !oCanvas) {
        rafId = requestAnimationFrame(draw);
        return;
      }

      const rect = vCanvas.getBoundingClientRect();
      const targetW = Math.max(320, Math.round(rect.width));
      const targetH = Math.max(180, Math.round(rect.height));

      if (vCanvas.width !== targetW || vCanvas.height !== targetH) {
        vCanvas.width = targetW;
        vCanvas.height = targetH;
        oCanvas.width = targetW;
        oCanvas.height = targetH;
      }

      const vCtx = vCanvas.getContext('2d');
      const oCtx = oCanvas.getContext('2d');

      if (vCtx) {
        vCtx.clearRect(0, 0, targetW, targetH);
        const video = videoRef.current;

        if (inputMode === 'webcam' && cameraActive && video && video.readyState >= 2) {
          vCtx.save();
          if (settings.mirror) {
            vCtx.translate(targetW, 0);
            vCtx.scale(-1, 1);
          }
          // Cover-fit video into canvas preserving aspect ratio
          const vw = video.videoWidth || 1280;
          const vh = video.videoHeight || 720;
          const scale = Math.max(targetW / vw, targetH / vh);
          const dw = vw * scale;
          const dh = vh * scale;
          const dx = (targetW - dw) / 2;
          const dy = (targetH - dh) / 2;
          vCtx.drawImage(video, dx, dy, dw, dh);
          vCtx.restore();
        } else {
          // Precision calibrated dark grid backdrop
          vCtx.fillStyle = '#090C12';
          vCtx.fillRect(0, 0, targetW, targetH);
          vCtx.strokeStyle = 'rgba(148, 163, 184, 0.08)';
          vCtx.lineWidth = 1;
          const step = compact ? 28 : 40;
          for (let x = step; x < targetW; x += step) {
            vCtx.beginPath();
            vCtx.moveTo(x, 0);
            vCtx.lineTo(x, targetH);
            vCtx.stroke();
          }
          for (let y = step; y < targetH; y += step) {
            vCtx.beginPath();
            vCtx.moveTo(0, y);
            vCtx.lineTo(targetW, y);
            vCtx.stroke();
          }
        }
      }

      if (oCtx) {
        const snap = frameStateRef.current;
        renderVisionOverlay(oCtx, {
          width: targetW,
          height: targetH,
          mirror: inputMode === 'webcam' ? settings.mirror : false,
          showFaceMesh: settings.showFaceMesh,
          showHandSkeleton: settings.showHandSkeleton,
          face: snap.face,
          hands: snap.hands,
          strokes: strokesRef.current,
          activeStroke: activeStrokeRef.current,
          cursorPoint: cursorRef.current,
          isDrawing: isDrawingRef.current,
        });
      }

      rafId = requestAnimationFrame(draw);
    };

    rafId = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(rafId);
  }, [cameraActive, compact, frameStateRef, inputMode, settings.mirror, settings.showFaceMesh, settings.showHandSkeleton, videoRef]);

  const isRunning = cameraActive || inputMode === 'simulation';

  return (
    <div
      className={`relative overflow-hidden rounded-lg border border-slate-200 dark:border-slate-800 bg-[#090C12] ${aspectClass} ${className}`}
    >
      <canvas ref={videoCanvasRef} className="absolute inset-0 h-full w-full block" />
      <canvas ref={overlayCanvasRef} className="absolute inset-0 h-full w-full block pointer-events-none" />

      {/* Top calibrated HUD strip */}
      <div className="pointer-events-none absolute inset-x-0 top-0 flex items-center justify-between bg-gradient-to-b from-black/75 via-black/30 to-transparent px-3.5 py-2.5 text-xs font-mono text-slate-200">
        <div className="flex items-center gap-2.5">
          <span className={isRunning ? 'text-emerald-400' : 'text-slate-400'}>
            {isRunning ? '● Live Stream' : '○ Standby'}
          </span>
          <span aria-hidden="true" className="text-slate-600">/</span>
          <span className="text-slate-300">
            {inputMode === 'simulation' ? 'Synthetic Simulator' : settings.resolution}
          </span>
        </div>
        <div className="flex items-center gap-3 tabular-nums">
          <span>{fps} FPS</span>
          <span aria-hidden="true" className="text-slate-600">·</span>
          <span>{inferenceMs.toFixed(1)} ms</span>
        </div>
      </div>

      {/* Standby / Permission / Error state */}
      {!isRunning && (
        <div className="absolute inset-0 flex flex-col items-center justify-center bg-[#090C12]/85 p-6 text-center">
          <p className="font-display text-base font-semibold text-slate-100">
            {modelsLoading ? 'Initializing MediaPipe Vision Models…' : 'Vision Pipeline Ready'}
          </p>
          <p className="mt-1.5 max-w-md text-xs leading-relaxed text-slate-400">
            {cameraError ||
              modelLoadError ||
              'Activate your webcam for real-time 478-point facial expression and dual-hand tracking, or launch the synthetic simulator to test every CV/ML module immediately.'}
          </p>
          <div className="mt-4 flex flex-wrap items-center justify-center gap-3">
            <button
              type="button"
              onClick={() => void startCamera()}
              disabled={modelsLoading}
              className="inline-flex items-center gap-2 rounded-md bg-cyan-500 px-4 py-2 text-xs font-semibold text-slate-950 transition-colors hover:bg-cyan-400 disabled:opacity-60 cursor-pointer whitespace-nowrap shrink-0"
            >
              <Camera className="h-3.5 w-3.5" />
              {modelsLoading ? 'Loading Weights…' : 'Enable Webcam'}
            </button>
            <button
              type="button"
              onClick={() => setInputMode('simulation')}
              className="inline-flex items-center gap-2 rounded-md border border-slate-700 bg-slate-900 px-4 py-2 text-xs font-medium text-slate-200 transition-colors hover:border-slate-500 hover:bg-slate-800 cursor-pointer whitespace-nowrap shrink-0"
            >
              <Cpu className="h-3.5 w-3.5" />
              Launch Simulator
            </button>
          </div>
        </div>
      )}

      {inputMode === 'simulation' && !compact && (
        <div className="pointer-events-none absolute bottom-2.5 left-3 flex items-center gap-2 rounded bg-black/70 px-2.5 py-1 text-[11px] font-mono text-cyan-300">
          <Play className="h-3 w-3" />
          <span>Synthetic Landmark Feed Active</span>
        </div>
      )}
    </div>
  );
};
