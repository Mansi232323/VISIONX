import React from 'react';
import {
  Camera,
  CameraOff,
  Cpu,
  Eye,
  Hand,
  RefreshCw,
  Sliders,
} from 'lucide-react';
import { useVision } from '../vision/VisionContext';
import { VisionViewport } from './VisionViewport';
import { ExpressionLabel } from '../types/vision';
import { CORE_GESTURE_LABELS } from '../ml/gestureModels';

const EXPRESSIONS: ExpressionLabel[] = [
  'Neutral',
  'Happy',
  'Surprised',
  'Angry',
  'Sad',
  'Fearful',
  'Disgusted',
];

export const OverviewView: React.FC = () => {
  const {
    face,
    hands,
    fps,
    fpsHistory,
    inferenceMs,
    expressionHistory,
    gestureCounts,
    logs,
    inputMode,
    setInputMode,
    cameraActive,
    startCamera,
    stopCamera,
    settings,
    updateSettings,
    simConfig,
    updateSimConfig,
    classifierMode,
    setClassifierMode,
  } = useVision();

  const primaryHand = hands[0];
  const sortedExpressions = Object.entries(face.expression.scores).sort(
    ([, a], [, b]) => b - a
  );
  const topGestures = Object.entries(gestureCounts)
    .sort(([, a], [, b]) => b - a)
    .slice(0, 6);

  // Build SVG sparkline points for FPS
  const fpsPath = React.useMemo(() => {
    if (fpsHistory.length < 2) return '';
    const w = 260;
    const h = 56;
    const maxFps = Math.max(60, ...fpsHistory);
    return fpsHistory
      .map((val, i) => {
        const x = (i / (fpsHistory.length - 1)) * w;
        const y = h - (val / maxFps) * (h - 8) - 4;
        return `${i === 0 ? 'M' : 'L'}${x.toFixed(1)},${y.toFixed(1)}`;
      })
      .join(' ');
  }, [fpsHistory]);

  // Build SVG waveform for Expression confidence
  const exprPath = React.useMemo(() => {
    if (expressionHistory.length < 2) return '';
    const w = 260;
    const h = 56;
    return expressionHistory
      .map((item, i) => {
        const x = (i / (expressionHistory.length - 1)) * w;
        const y = h - item.confidence * (h - 8) - 4;
        return `${i === 0 ? 'M' : 'L'}${x.toFixed(1)},${y.toFixed(1)}`;
      })
      .join(' ');
  }, [expressionHistory]);

  return (
    <div className="space-y-6">
      {/* Calibration & Source Toolbar */}
      <div className="flex flex-wrap items-center justify-between gap-4 rounded-lg border border-slate-200 dark:border-slate-800 bg-white dark:bg-[#0E131D] px-4 py-3">
        <div className="flex flex-wrap items-center gap-3">
          <div className="flex items-center rounded-md border border-slate-200 dark:border-slate-800 bg-slate-100 dark:bg-[#090C12] p-0.5">
            <button
              type="button"
              onClick={() => (cameraActive ? stopCamera() : void startCamera())}
              className={`inline-flex items-center gap-1.5 rounded px-3 py-1.5 text-xs font-medium transition-colors cursor-pointer whitespace-nowrap shrink-0 ${
                inputMode === 'webcam'
                  ? 'bg-white dark:bg-slate-800 text-slate-900 dark:text-slate-100 shadow-xs'
                  : 'text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-slate-200'
              }`}
            >
              {cameraActive ? <CameraOff className="h-3.5 w-3.5" /> : <Camera className="h-3.5 w-3.5" />}
              {cameraActive ? 'Stop Webcam' : 'Live Webcam'}
            </button>
            <button
              type="button"
              onClick={() => setInputMode('simulation')}
              className={`inline-flex items-center gap-1.5 rounded px-3 py-1.5 text-xs font-medium transition-colors cursor-pointer whitespace-nowrap shrink-0 ${
                inputMode === 'simulation'
                  ? 'bg-white dark:bg-slate-800 text-slate-900 dark:text-slate-100 shadow-xs'
                  : 'text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-slate-200'
              }`}
            >
              <Cpu className="h-3.5 w-3.5" />
              Simulator
            </button>
          </div>

          <div className="flex items-center gap-2 text-xs">
            <label className="inline-flex items-center gap-1.5 text-slate-600 dark:text-slate-300 cursor-pointer select-none">
              <input
                type="checkbox"
                checked={settings.showFaceMesh}
                onChange={(e) => updateSettings({ showFaceMesh: e.target.checked })}
                className="accent-cyan-500 rounded"
              />
              Face Mesh
            </label>
            <span className="text-slate-300 dark:text-slate-700">/</span>
            <label className="inline-flex items-center gap-1.5 text-slate-600 dark:text-slate-300 cursor-pointer select-none">
              <input
                type="checkbox"
                checked={settings.showHandSkeleton}
                onChange={(e) => updateSettings({ showHandSkeleton: e.target.checked })}
                className="accent-cyan-500 rounded"
              />
              Hand Skeleton
            </label>
            <span className="text-slate-300 dark:text-slate-700">/</span>
            <label className="inline-flex items-center gap-1.5 text-slate-600 dark:text-slate-300 cursor-pointer select-none">
              <input
                type="checkbox"
                checked={settings.mirror}
                onChange={(e) => updateSettings({ mirror: e.target.checked })}
                className="accent-cyan-500 rounded"
              />
              Mirror
            </label>
          </div>
        </div>

        <div className="flex items-center gap-3">
          <span className="text-xs text-slate-500 dark:text-slate-400">Gesture Classifier:</span>
          <select
            value={classifierMode}
            onChange={(e) => setClassifierMode(e.target.value as 'rules' | 'knn' | 'mlp')}
            aria-label="Gesture Classifier Engine"
            className="rounded-md border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-[#090C12] px-2.5 py-1.5 text-xs font-medium text-slate-800 dark:text-slate-200 focus:outline-none focus:border-cyan-500"
          >
            <option value="rules">Geometric Rules</option>
            <option value="knn">KNN (k=5, 28-D)</option>
            <option value="mlp">MLP Neural Network</option>
          </select>
        </div>
      </div>

      {/* Simulation Controls Bar (only shown in simulation mode) */}
      {inputMode === 'simulation' && (
        <div className="flex flex-wrap items-center gap-4 rounded-lg border border-cyan-500/30 bg-cyan-950/15 px-4 py-2.5 text-xs">
          <span className="font-mono font-semibold text-cyan-400">Simulator Controls:</span>
          <div className="flex items-center gap-2">
            <span className="text-slate-400">Expression</span>
            <select
              value={simConfig.expression}
              onChange={(e) => updateSimConfig({ expression: e.target.value as ExpressionLabel })}
              className="rounded border border-slate-700 bg-[#090C12] px-2 py-1 text-xs text-slate-200"
            >
              {EXPRESSIONS.map((exp) => (
                <option key={exp} value={exp}>
                  {exp}
                </option>
              ))}
            </select>
          </div>
          <div className="flex items-center gap-2">
            <span className="text-slate-400">Hand Pose</span>
            <select
              value={simConfig.gesture}
              onChange={(e) => updateSimConfig({ gesture: e.target.value })}
              className="rounded border border-slate-700 bg-[#090C12] px-2 py-1 text-xs text-slate-200"
            >
              {CORE_GESTURE_LABELS.map((g) => (
                <option key={g} value={g}>
                  {g}
                </option>
              ))}
            </select>
          </div>
          <button
            type="button"
            onClick={() => updateSimConfig({ shapeTrajectory: 'none', pointerTarget: null })}
            className="inline-flex items-center gap-1 rounded border border-slate-700 px-2.5 py-1 text-slate-300 hover:bg-slate-800 cursor-pointer"
          >
            <RefreshCw className="h-3 w-3" />
            Reset Pose
          </button>
        </div>
      )}

      {/* Main Split Console: Live Video Stage (7 cols) + Face & Hand Telemetry (5 cols) */}
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-12 items-start">
        <div className="lg:col-span-7 space-y-4">
          <VisionViewport aspectClass="aspect-video" />

          {/* Calibrated 4-metric KPI strip directly below viewport */}
          <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
            <div className="rounded-lg border border-slate-200 dark:border-slate-800 bg-white dark:bg-[#0E131D] p-3.5">
              <div className="text-[11px] font-medium text-slate-500 dark:text-slate-400">
                Expression Pattern
              </div>
              <div className="mt-1 font-display text-lg font-bold text-slate-900 dark:text-slate-100 truncate">
                {face.detected ? face.expression.label : 'No Face'}
              </div>
              <div className="mt-0.5 font-mono text-xs text-cyan-600 dark:text-cyan-400 tabular-nums">
                {face.detected ? `${(face.expression.confidence * 100).toFixed(1)}% conf` : '0.0%'}
              </div>
            </div>

            <div className="rounded-lg border border-slate-200 dark:border-slate-800 bg-white dark:bg-[#0E131D] p-3.5">
              <div className="text-[11px] font-medium text-slate-500 dark:text-slate-400">
                Primary Gesture
              </div>
              <div className="mt-1 font-display text-lg font-bold text-slate-900 dark:text-slate-100 truncate">
                {primaryHand ? primaryHand.gesture : 'No Hand'}
              </div>
              <div className="mt-0.5 font-mono text-xs text-emerald-600 dark:text-emerald-400 tabular-nums">
                {primaryHand
                  ? `${primaryHand.handedness} · ${(primaryHand.gestureConfidence * 100).toFixed(0)}%`
                  : 'Standby'}
              </div>
            </div>

            <div className="rounded-lg border border-slate-200 dark:border-slate-800 bg-white dark:bg-[#0E131D] p-3.5">
              <div className="text-[11px] font-medium text-slate-500 dark:text-slate-400">
                Pipeline Throughput
              </div>
              <div className="mt-1 font-mono text-lg font-bold text-slate-900 dark:text-slate-100 tabular-nums">
                {fps} <span className="text-xs font-normal text-slate-400">FPS</span>
              </div>
              <div className="mt-0.5 font-mono text-xs text-slate-500 dark:text-slate-400 tabular-nums">
                {inferenceMs.toFixed(1)} ms / frame
              </div>
            </div>

            <div className="rounded-lg border border-slate-200 dark:border-slate-800 bg-white dark:bg-[#0E131D] p-3.5">
              <div className="text-[11px] font-medium text-slate-500 dark:text-slate-400">
                Eye Blinks & Motion
              </div>
              <div className="mt-1 font-mono text-lg font-bold text-slate-900 dark:text-slate-100 tabular-nums">
                {face.blinkCount} <span className="text-xs font-normal text-slate-400">blinks</span>
              </div>
              <div className="mt-0.5 font-mono text-xs text-amber-600 dark:text-amber-400 truncate">
                {primaryHand ? primaryHand.motion : 'Idle'}
              </div>
            </div>
          </div>
        </div>

        {/* Right Analysis Column */}
        <div className="lg:col-span-5 space-y-4">
          {/* Facial Expression & Pose Panel */}
          <section className="rounded-lg border border-slate-200 dark:border-slate-800 bg-white dark:bg-[#0E131D] p-4">
            <div className="flex items-center justify-between border-b border-slate-100 dark:border-slate-800/80 pb-2.5">
              <div className="flex items-center gap-2">
                <Eye className="h-4 w-4 text-cyan-500" />
                <h2 className="font-display text-sm font-semibold text-slate-900 dark:text-slate-100">
                  Facial Landmark & Expression Analysis
                </h2>
              </div>
              <span className="font-mono text-[11px] text-slate-500 dark:text-slate-400 tabular-nums">
                478 pts
              </span>
            </div>

            <div className="mt-3 space-y-2">
              {sortedExpressions.map(([label, score]) => {
                const pct = Math.min(100, Math.round(score * 100));
                const isTop = label === face.expression.label && face.detected;
                return (
                  <div key={label} className="grid grid-cols-12 items-center gap-2 text-xs">
                    <span
                      className={`col-span-3 truncate ${
                        isTop
                          ? 'font-semibold text-cyan-600 dark:text-cyan-400'
                          : 'text-slate-600 dark:text-slate-400'
                      }`}
                    >
                      {label}
                    </span>
                    <div className="col-span-7 h-1.5 overflow-hidden rounded-full bg-slate-100 dark:bg-slate-800">
                      <div
                        className={`h-full transition-all duration-150 ${
                          isTop ? 'bg-cyan-500' : 'bg-slate-400 dark:bg-slate-600'
                        }`}
                        style={{ width: `${pct}%` }}
                      />
                    </div>
                    <span className="col-span-2 text-right font-mono text-[11px] text-slate-600 dark:text-slate-400 tabular-nums">
                      {pct}%
                    </span>
                  </div>
                );
              })}
            </div>

            <div className="mt-4 grid grid-cols-3 gap-2 border-t border-slate-100 dark:border-slate-800/80 pt-3 font-mono text-xs tabular-nums">
              <div>
                <span className="block text-[10px] text-slate-500">EAR (L / R)</span>
                <span className="text-slate-800 dark:text-slate-200">
                  {face.leftEAR.toFixed(2)} / {face.rightEAR.toFixed(2)}
                </span>
              </div>
              <div>
                <span className="block text-[10px] text-slate-500">Mouth Ratio</span>
                <span className="text-slate-800 dark:text-slate-200">
                  {face.mouthOpenRatio.toFixed(2)}
                </span>
              </div>
              <div>
                <span className="block text-[10px] text-slate-500">Head Y / P / R</span>
                <span className="text-slate-800 dark:text-slate-200">
                  {face.headPose.yaw.toFixed(0)}° / {face.headPose.pitch.toFixed(0)}° /{' '}
                  {face.headPose.roll.toFixed(0)}°
                </span>
              </div>
            </div>

            <p className="mt-3 text-[11px] leading-relaxed text-slate-500 dark:text-slate-400">
              Note: Expression outputs classify visible facial muscle & blendshape patterns (52 ARKit coefficients), not verified psychological states.
            </p>
          </section>

          {/* Dual-Hand Tracking Panel */}
          <section className="rounded-lg border border-slate-200 dark:border-slate-800 bg-white dark:bg-[#0E131D] p-4">
            <div className="flex items-center justify-between border-b border-slate-100 dark:border-slate-800/80 pb-2.5">
              <div className="flex items-center gap-2">
                <Hand className="h-4 w-4 text-cyan-500" />
                <h2 className="font-display text-sm font-semibold text-slate-900 dark:text-slate-100">
                  Dual-Hand Pose & Finger States
                </h2>
              </div>
              <span className="font-mono text-[11px] text-slate-500 dark:text-slate-400 tabular-nums">
                {hands.length} / 2 hands
              </span>
            </div>

            {hands.length === 0 ? (
              <div className="py-6 text-center text-xs text-slate-500 dark:text-slate-400">
                Show one or both hands to the camera to inspect 21-point joint states and pinch metrics.
              </div>
            ) : (
              <div className="mt-3 space-y-3">
                {hands.map((hand) => (
                  <div
                    key={hand.id}
                    className="rounded-md border border-slate-100 dark:border-slate-800 bg-slate-50 dark:bg-[#090C12] p-3"
                  >
                    <div className="flex items-center justify-between text-xs">
                      <span className="font-semibold text-slate-900 dark:text-slate-100">
                        {hand.handedness} Hand · {hand.gesture}
                      </span>
                      <span className="font-mono text-cyan-600 dark:text-cyan-400 tabular-nums">
                        {(hand.gestureConfidence * 100).toFixed(0)}%
                      </span>
                    </div>

                    <div className="mt-2.5 grid grid-cols-5 gap-1.5 text-center font-mono text-[11px]">
                      {(['thumb', 'index', 'middle', 'ring', 'pinky'] as const).map((finger) => {
                        const up = hand.fingerStates[finger];
                        return (
                          <div
                            key={finger}
                            className={`rounded border py-1 capitalize ${
                              up
                                ? 'border-cyan-500/40 bg-cyan-500/10 text-cyan-600 dark:text-cyan-300'
                                : 'border-slate-200 dark:border-slate-800 text-slate-400'
                            }`}
                          >
                            {finger.slice(0, 3)}
                          </div>
                        );
                      })}
                    </div>

                    <div className="mt-2.5 flex items-center justify-between font-mono text-[11px] text-slate-500 dark:text-slate-400 tabular-nums">
                      <span>Index ({hand.indexTip.x.toFixed(2)}, {hand.indexTip.y.toFixed(2)})</span>
                      <span>Pinch Δ {hand.pinchDistance.toFixed(3)}</span>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </section>
        </div>
      </div>

      {/* Bottom Time-Series & Analytics Row */}
      <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
        <div className="rounded-lg border border-slate-200 dark:border-slate-800 bg-white dark:bg-[#0E131D] p-4">
          <div className="flex items-center justify-between text-xs">
            <span className="font-medium text-slate-700 dark:text-slate-300">
              Expression Confidence Timeline
            </span>
            <span className="font-mono text-slate-400 tabular-nums">
              {expressionHistory.length} samples
            </span>
          </div>
          <svg viewBox="0 0 260 56" className="mt-3 h-14 w-full overflow-visible">
            <line x1="0" y1="28" x2="260" y2="28" stroke="currentColor" className="text-slate-200 dark:text-slate-800" strokeDasharray="3 3" />
            {exprPath && (
              <path d={exprPath} fill="none" stroke="#06B6D4" strokeWidth="1.75" />
            )}
          </svg>
        </div>

        <div className="rounded-lg border border-slate-200 dark:border-slate-800 bg-white dark:bg-[#0E131D] p-4">
          <div className="flex items-center justify-between text-xs">
            <span className="font-medium text-slate-700 dark:text-slate-300">
              Frame Rate Stability (FPS)
            </span>
            <span className="font-mono text-slate-400 tabular-nums">{fps} FPS</span>
          </div>
          <svg viewBox="0 0 260 56" className="mt-3 h-14 w-full overflow-visible">
            <line x1="0" y1="28" x2="260" y2="28" stroke="currentColor" className="text-slate-200 dark:text-slate-800" strokeDasharray="3 3" />
            {fpsPath && (
              <path d={fpsPath} fill="none" stroke="#10B981" strokeWidth="1.75" />
            )}
          </svg>
        </div>

        <div className="rounded-lg border border-slate-200 dark:border-slate-800 bg-white dark:bg-[#0E131D] p-4">
          <div className="flex items-center justify-between text-xs">
            <span className="font-medium text-slate-700 dark:text-slate-300">
              Session Gesture Counts & Parameters
            </span>
            <Sliders className="h-3.5 w-3.5 text-slate-400" />
          </div>
          {topGestures.length === 0 ? (
            <p className="mt-3 text-xs text-slate-500 dark:text-slate-400">
              Perform hand gestures to populate frequency distribution.
            </p>
          ) : (
            <div className="mt-3 grid grid-cols-2 gap-x-4 gap-y-1.5 text-xs">
              {topGestures.map(([name, count]) => (
                <div key={name} className="flex items-center justify-between">
                  <span className="truncate text-slate-600 dark:text-slate-300">{name}</span>
                  <span className="font-mono text-cyan-600 dark:text-cyan-400 tabular-nums">
                    {count}
                  </span>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Recent Interaction Log */}
      <section className="rounded-lg border border-slate-200 dark:border-slate-800 bg-white dark:bg-[#0E131D] p-4">
        <h3 className="font-display text-xs font-semibold uppercase tracking-wider text-slate-500 dark:text-slate-400">
          Pipeline Event History
        </h3>
        {logs.length === 0 ? (
          <p className="mt-2 text-xs text-slate-500">No events recorded yet in this session.</p>
        ) : (
          <div className="mt-2.5 max-h-36 overflow-y-auto divide-y divide-slate-100 dark:divide-slate-800/70">
            {logs.slice(0, 8).map((entry) => (
              <div key={entry.id} className="flex items-center justify-between py-1.5 text-xs">
                <div className="flex items-center gap-3 min-w-0">
                  <span className="font-mono text-[11px] text-slate-400 tabular-nums shrink-0">
                    {new Date(entry.timestamp).toLocaleTimeString()}
                  </span>
                  <span className="font-mono text-[11px] text-cyan-600 dark:text-cyan-400 shrink-0">
                    {entry.module}
                  </span>
                  <span className="font-medium text-slate-800 dark:text-slate-200 truncate">
                    {entry.event}
                  </span>
                  <span className="text-slate-500 dark:text-slate-400 truncate">
                    {entry.detail}
                  </span>
                </div>
              </div>
            ))}
          </div>
        )}
      </section>
    </div>
  );
};
