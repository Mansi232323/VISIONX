import React, { useMemo, useState } from 'react';
import {
  Activity,
  Brain,
  Database,
  Play,
  PlusCircle,
  RotateCcw,
} from 'lucide-react';
import { useVision } from '../vision/VisionContext';
import { VisionViewport } from './VisionViewport';
import { CORE_GESTURE_LABELS } from '../ml/gestureModels';

export const MLLabView: React.FC = () => {
  const {
    frameStateRef,
    hands,
    classifierMode,
    setClassifierMode,
    dataset,
    addGestureSamples,
    clearCustomSamples,
    trainAndEvaluate,
    evaluationReport,
    cameraActive,
    inputMode,
  } = useVision();

  const [selectedLabel, setSelectedLabel] = useState<string>(CORE_GESTURE_LABELS[0]);
  const [customGestureName, setCustomGestureName] = useState('');
  const [recording, setRecording] = useState(false);
  const [evalTarget, setEvalTarget] = useState<'mlp' | 'knn'>('mlp');

  const classCounts = useMemo(() => {
    const counts: Record<string, { total: number; webcam: number }> = {};
    for (const sample of dataset) {
      if (!counts[sample.label]) counts[sample.label] = { total: 0, webcam: 0 };
      counts[sample.label].total += 1;
      if (sample.source === 'webcam') counts[sample.label].webcam += 1;
    }
    return counts;
  }, [dataset]);

  const allLabels = useMemo(
    () => Array.from(new Set([...CORE_GESTURE_LABELS, ...Object.keys(classCounts)])),
    [classCounts]
  );

  const handleRecordBatch = async () => {
    const targetLabel = customGestureName.trim() || selectedLabel;
    if (!targetLabel || recording) return;
    setRecording(true);
    const collected: number[][] = [];

    for (let i = 0; i < 15; i++) {
      const hand = frameStateRef.current.hands[0];
      if (hand) {
        collected.push([...hand.features]);
      }
      await new Promise((r) => setTimeout(r, 70));
    }

    if (collected.length > 0) {
      addGestureSamples(targetLabel, collected);
      if (customGestureName.trim()) {
        setSelectedLabel(targetLabel);
        setCustomGestureName('');
      }
    }
    setRecording(false);
  };

  const handleTrainModel = (modelType: 'knn' | 'mlp') => {
    setEvalTarget(modelType);
    trainAndEvaluate(modelType);
    setClassifierMode(modelType);
  };

  const primaryHand = hands[0];
  const report = evaluationReport;

  // Epoch loss SVG path
  const lossPath = useMemo(() => {
    if (!report || report.lossHistory.length < 2) return '';
    const w = 300;
    const h = 90;
    const maxLoss = Math.max(0.5, ...report.lossHistory);
    return report.lossHistory
      .map((val, i) => {
        const x = (i / (report.lossHistory.length - 1)) * w;
        const y = h - (val / maxLoss) * (h - 12) - 6;
        return `${i === 0 ? 'M' : 'L'}${x.toFixed(1)},${y.toFixed(1)}`;
      })
      .join(' ');
  }, [report]);

  const accPath = useMemo(() => {
    if (!report || report.accuracyHistory.length < 2) return '';
    const w = 300;
    const h = 90;
    return report.accuracyHistory
      .map((val, i) => {
        const x = (i / (report.accuracyHistory.length - 1)) * w;
        const y = h - val * (h - 12) - 6;
        return `${i === 0 ? 'M' : 'L'}${x.toFixed(1)},${y.toFixed(1)}`;
      })
      .join(' ');
  }, [report]);

  return (
    <div className="space-y-6">
      {/* Top Inference Engine & Pipeline Strip */}
      <div className="flex flex-wrap items-center justify-between gap-4 rounded-lg border border-slate-200 dark:border-slate-800 bg-white dark:bg-[#0E131D] px-4 py-3">
        <div className="flex flex-wrap items-center gap-3">
          <span className="text-xs font-medium text-slate-600 dark:text-slate-300">
            Active Live Classifier:
          </span>
          <div className="flex items-center rounded-md border border-slate-200 dark:border-slate-800 bg-slate-100 dark:bg-[#090C12] p-0.5">
            {([
              { id: 'rules', label: 'Rule-Based' },
              { id: 'knn', label: 'KNN (k=5)' },
              { id: 'mlp', label: 'MLP Neural Net' },
            ] as const).map((m) => (
              <button
                key={m.id}
                type="button"
                onClick={() => setClassifierMode(m.id)}
                className={`rounded px-3 py-1.5 text-xs font-medium transition-colors cursor-pointer whitespace-nowrap shrink-0 ${
                  classifierMode === m.id
                    ? 'bg-white dark:bg-slate-800 text-slate-900 dark:text-cyan-300 shadow-xs'
                    : 'text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-slate-200'
                }`}
              >
                {m.label}
              </button>
            ))}
          </div>
        </div>

        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={() => handleTrainModel('knn')}
            className="inline-flex items-center gap-1.5 rounded-md border border-slate-300 dark:border-slate-700 px-3 py-1.5 text-xs font-medium text-slate-800 dark:text-slate-200 hover:border-cyan-500 cursor-pointer whitespace-nowrap shrink-0"
          >
            <Play className="h-3.5 w-3.5" />
            Evaluate KNN
          </button>
          <button
            type="button"
            onClick={() => handleTrainModel('mlp')}
            className="inline-flex items-center gap-1.5 rounded-md bg-cyan-500 px-3.5 py-1.5 text-xs font-semibold text-slate-950 hover:bg-cyan-400 cursor-pointer whitespace-nowrap shrink-0"
          >
            <Brain className="h-3.5 w-3.5" />
            Train MLP (60 Epochs)
          </button>
        </div>
      </div>

      {/* Main Grid: Dataset Capture + Live 28D Features (5 cols) & Evaluation Report (7 cols) */}
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-12 items-start">
        {/* Left: Live Recorder & 28D Feature Vector */}
        <div className="lg:col-span-5 space-y-4">
          <VisionViewport aspectClass="aspect-video" compact />

          {/* Live Sample Recorder */}
          <section className="rounded-lg border border-slate-200 dark:border-slate-800 bg-white dark:bg-[#0E131D] p-4 space-y-3">
            <div className="flex items-center justify-between border-b border-slate-100 dark:border-slate-800 pb-2.5">
              <div className="flex items-center gap-2">
                <Database className="h-4 w-4 text-cyan-500" />
                <h2 className="font-display text-sm font-semibold text-slate-900 dark:text-slate-100">
                  Custom Gesture Dataset Recorder
                </h2>
              </div>
              <span className="font-mono text-[11px] text-slate-400 tabular-nums">
                {dataset.length} samples
              </span>
            </div>

            <div className="grid grid-cols-1 gap-2.5 sm:grid-cols-2">
              <div>
                <label className="block text-[11px] text-slate-500 dark:text-slate-400 mb-1">
                  Target Gesture Class
                </label>
                <select
                  value={selectedLabel}
                  onChange={(e) => setSelectedLabel(e.target.value)}
                  className="w-full rounded border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-[#090C12] px-2.5 py-1.5 text-xs text-slate-800 dark:text-slate-200"
                >
                  {allLabels.map((l) => (
                    <option key={l} value={l}>
                      {l} ({classCounts[l]?.total ?? 0})
                    </option>
                  ))}
                </select>
              </div>
              <div>
                <label className="block text-[11px] text-slate-500 dark:text-slate-400 mb-1">
                  Or New Custom Class
                </label>
                <input
                  type="text"
                  value={customGestureName}
                  onChange={(e) => setCustomGestureName(e.target.value)}
                  placeholder="e.g. Shaka Sign"
                  maxLength={24}
                  className="w-full rounded border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-[#090C12] px-2.5 py-1.5 text-xs text-slate-800 dark:text-slate-200 focus:border-cyan-500 focus:outline-none"
                />
              </div>
            </div>

            <div className="flex items-center gap-2 pt-1">
              <button
                type="button"
                onClick={() => void handleRecordBatch()}
                disabled={recording || (!cameraActive && inputMode !== 'simulation')}
                className="flex-1 inline-flex items-center justify-center gap-1.5 rounded-md bg-cyan-500 px-3 py-2 text-xs font-semibold text-slate-950 hover:bg-cyan-400 disabled:opacity-40 cursor-pointer"
              >
                <PlusCircle className="h-3.5 w-3.5" />
                {recording ? 'Capturing 15 Frames…' : 'Capture 15 Samples'}
              </button>
              <button
                type="button"
                onClick={clearCustomSamples}
                className="inline-flex items-center gap-1 rounded-md border border-slate-300 dark:border-slate-700 px-3 py-2 text-xs text-slate-600 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-800 cursor-pointer"
              >
                <RotateCcw className="h-3.5 w-3.5" />
                Reset
              </button>
            </div>
          </section>

          {/* 28-D Feature Vector Live Monitor */}
          <section className="rounded-lg border border-slate-200 dark:border-slate-800 bg-white dark:bg-[#0E131D] p-4">
            <div className="flex items-center justify-between border-b border-slate-100 dark:border-slate-800 pb-2.5">
              <div className="flex items-center gap-2">
                <Activity className="h-4 w-4 text-cyan-500" />
                <h2 className="font-display text-sm font-semibold text-slate-900 dark:text-slate-100">
                  28-D Engineered Landmark Features
                </h2>
              </div>
              <span className="font-mono text-[11px] text-slate-400">
                {primaryHand ? primaryHand.gesture : 'No Hand'}
              </span>
            </div>
            <div className="mt-3 flex h-20 items-end gap-1">
              {(primaryHand?.features ?? new Array(28).fill(0.05)).map((val, idx) => (
                <div
                  key={idx}
                  title={`Feature[${idx}] = ${val.toFixed(3)}`}
                  className="flex-1 rounded-t bg-cyan-500/75 transition-all duration-75"
                  style={{ height: `${Math.max(4, Math.round(val * 100))}%` }}
                />
              ))}
            </div>
            <div className="mt-2 flex justify-between font-mono text-[10px] text-slate-400">
              <span>Tip Ratios [0..4]</span>
              <span>Angles [5..9]</span>
              <span>Pinch [10..13]</span>
              <span>Offsets [17..27]</span>
            </div>
          </section>
        </div>

        {/* Right: Neural Net Evaluation, Curves & Confusion Matrix */}
        <div className="lg:col-span-7 space-y-4">
          {report && (
            <>
              {/* 4 Evaluation KPI Cards */}
              <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
                <div className="rounded-lg border border-slate-200 dark:border-slate-800 bg-white dark:bg-[#0E131D] p-3.5">
                  <div className="text-[11px] text-slate-500 dark:text-slate-400">Test Accuracy</div>
                  <div className="mt-1 font-mono text-2xl font-bold text-slate-900 dark:text-slate-100 tabular-nums">
                    {(report.accuracy * 100).toFixed(1)}%
                  </div>
                  <div className="mt-0.5 font-mono text-[11px] text-cyan-500 uppercase">
                    {evalTarget} ({report.testCount} eval)
                  </div>
                </div>

                <div className="rounded-lg border border-slate-200 dark:border-slate-800 bg-white dark:bg-[#0E131D] p-3.5">
                  <div className="text-[11px] text-slate-500 dark:text-slate-400">Macro Precision</div>
                  <div className="mt-1 font-mono text-2xl font-bold text-slate-900 dark:text-slate-100 tabular-nums">
                    {(report.macroPrecision * 100).toFixed(1)}%
                  </div>
                  <div className="mt-0.5 font-mono text-[11px] text-slate-400">
                    {report.labels.length} classes
                  </div>
                </div>

                <div className="rounded-lg border border-slate-200 dark:border-slate-800 bg-white dark:bg-[#0E131D] p-3.5">
                  <div className="text-[11px] text-slate-500 dark:text-slate-400">Macro Recall</div>
                  <div className="mt-1 font-mono text-2xl font-bold text-slate-900 dark:text-slate-100 tabular-nums">
                    {(report.macroRecall * 100).toFixed(1)}%
                  </div>
                  <div className="mt-0.5 font-mono text-[11px] text-slate-400">
                    75 / 25 stratified
                  </div>
                </div>

                <div className="rounded-lg border border-slate-200 dark:border-slate-800 bg-white dark:bg-[#0E131D] p-3.5">
                  <div className="text-[11px] text-slate-500 dark:text-slate-400">Macro F1-Score</div>
                  <div className="mt-1 font-mono text-2xl font-bold text-emerald-500 tabular-nums">
                    {(report.macroF1 * 100).toFixed(1)}%
                  </div>
                  <div className="mt-0.5 font-mono text-[11px] text-slate-400">
                    {report.trainCount} train
                  </div>
                </div>
              </div>

              {/* MLP Loss & Accuracy Training Curves */}
              <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
                <div className="rounded-lg border border-slate-200 dark:border-slate-800 bg-white dark:bg-[#0E131D] p-4">
                  <div className="flex items-center justify-between text-xs">
                    <span className="font-semibold text-slate-800 dark:text-slate-200">
                      MLP Cross-Entropy Loss (60 Epochs)
                    </span>
                    <span className="font-mono text-amber-400 tabular-nums">
                      {report.lossHistory[report.lossHistory.length - 1]?.toFixed(3)}
                    </span>
                  </div>
                  <svg viewBox="0 0 300 90" className="mt-3 h-24 w-full overflow-visible">
                    <line x1="0" y1="84" x2="300" y2="84" stroke="currentColor" className="text-slate-200 dark:text-slate-800" />
                    {lossPath && <path d={lossPath} fill="none" stroke="#F59E0B" strokeWidth="2" />}
                  </svg>
                </div>

                <div className="rounded-lg border border-slate-200 dark:border-slate-800 bg-white dark:bg-[#0E131D] p-4">
                  <div className="flex items-center justify-between text-xs">
                    <span className="font-semibold text-slate-800 dark:text-slate-200">
                      MLP Training Accuracy Curve
                    </span>
                    <span className="font-mono text-emerald-400 tabular-nums">
                      {((report.accuracyHistory[report.accuracyHistory.length - 1] ?? 0) * 100).toFixed(1)}%
                    </span>
                  </div>
                  <svg viewBox="0 0 300 90" className="mt-3 h-24 w-full overflow-visible">
                    <line x1="0" y1="84" x2="300" y2="84" stroke="currentColor" className="text-slate-200 dark:text-slate-800" />
                    {accPath && <path d={accPath} fill="none" stroke="#10B981" strokeWidth="2" />}
                  </svg>
                </div>
              </div>

              {/* Confusion Matrix Heatmap */}
              <section className="rounded-lg border border-slate-200 dark:border-slate-800 bg-white dark:bg-[#0E131D] p-4">
                <div className="flex items-center justify-between border-b border-slate-100 dark:border-slate-800 pb-2.5">
                  <h2 className="font-display text-sm font-semibold text-slate-900 dark:text-slate-100">
                    Evaluation Confusion Matrix ({report.modelType.toUpperCase()})
                  </h2>
                  <span className="font-mono text-[11px] text-slate-400">
                    Rows: Actual · Cols: Predicted
                  </span>
                </div>

                <div className="mt-3 overflow-x-auto">
                  <table className="w-full border-collapse font-mono text-[11px] tabular-nums">
                    <thead>
                      <tr>
                        <th className="p-1.5 text-left text-slate-400">Actual \ Pred</th>
                        {report.labels.map((lbl) => (
                          <th key={lbl} className="p-1.5 text-center text-slate-400 truncate max-w-[70px]" title={lbl}>
                            {lbl.slice(0, 6)}
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {report.labels.map((rowLbl, rIdx) => (
                        <tr key={rowLbl}>
                          <td className="p-1.5 font-medium text-slate-700 dark:text-slate-300 whitespace-nowrap">
                            {rowLbl}
                          </td>
                          {report.confusionMatrix[rIdx].map((val, cIdx) => {
                            const isDiag = rIdx === cIdx;
                            const bg =
                              val === 0
                                ? 'bg-slate-50 dark:bg-[#090C12] text-slate-500'
                                : isDiag
                                ? 'bg-cyan-500/25 text-cyan-700 dark:text-cyan-300 font-bold'
                                : 'bg-rose-500/25 text-rose-500 font-semibold';
                            return (
                              <td
                                key={cIdx}
                                className={`border border-slate-200/60 dark:border-slate-800 p-1.5 text-center ${bg}`}
                              >
                                {val}
                              </td>
                            );
                          })}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </section>
            </>
          )}
        </div>
      </div>
    </div>
  );
};
