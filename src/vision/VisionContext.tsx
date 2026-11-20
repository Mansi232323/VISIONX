import React, {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useRef,
  useState,
} from 'react';
import {
  FaceLandmarker,
  FilesetResolver,
  HandLandmarker,
} from '@mediapipe/tasks-vision';
import {
  ExpressionLabel,
  ExpressionPrediction,
  FaceMetrics,
  GestureClassifierMode,
  GestureSample,
  InputSourceMode,
  InteractionLogEntry,
  ModelEvaluationReport,
  TrackedHand,
} from '../types/vision';
import { FaceExpressionAnalyzer } from './faceAnalysis';
import {
  classifyGestureRules,
  HandTrackingProcessor,
} from './handAnalysis';
import {
  evaluateClassifier,
  generateSeededDataset,
  KNNClassifier,
  MLPGestureNetwork,
} from '../ml/gestureModels';
import {
  generateSyntheticFaceResult,
  generateSyntheticHandResult,
  SimulationConfig,
} from './syntheticSimulator';

export interface VisionSettings {
  mirror: boolean;
  showFaceMesh: boolean;
  showHandSkeleton: boolean;
  minDetectionConfidence: number;
  minTrackingConfidence: number;
  expressionSmoothing: number;
  predictionIntervalMs: number;
  resolution: '640x480' | '1280x720';
}

interface VisionFrameState {
  face: FaceMetrics;
  hands: TrackedHand[];
  fps: number;
  inferenceMs: number;
}

interface VisionContextValue {
  videoRef: React.RefObject<HTMLVideoElement | null>;
  frameStateRef: React.MutableRefObject<VisionFrameState>;
  // UI reactive snapshots (throttled to ~12-15Hz so React UI stays fast while canvas runs at 60fps)
  face: FaceMetrics;
  hands: TrackedHand[];
  fps: number;
  fpsHistory: number[];
  inferenceMs: number;
  expressionHistory: ExpressionPrediction[];
  gestureCounts: Record<string, number>;
  logs: InteractionLogEntry[];
  addLog: (module: InteractionLogEntry['module'], event: string, detail: string) => void;

  // Camera & Model lifecycle
  inputMode: InputSourceMode;
  setInputMode: (mode: InputSourceMode) => void;
  cameraActive: boolean;
  cameraError: string | null;
  modelsReady: boolean;
  modelsLoading: boolean;
  modelLoadError: string | null;
  startCamera: () => Promise<void>;
  stopCamera: () => void;

  // Settings
  settings: VisionSettings;
  updateSettings: (patch: Partial<VisionSettings>) => void;

  // Simulation config
  simConfig: SimulationConfig;
  updateSimConfig: (patch: Partial<SimulationConfig>) => void;

  // ML / DL Pipeline
  classifierMode: GestureClassifierMode;
  setClassifierMode: (mode: GestureClassifierMode) => void;
  dataset: GestureSample[];
  addGestureSamples: (label: string, samples: number[][]) => void;
  clearCustomSamples: () => void;
  trainAndEvaluate: (modelType: 'knn' | 'mlp') => ModelEvaluationReport;
  evaluationReport: ModelEvaluationReport | null;
}

const DEFAULT_FACE: FaceMetrics = {
  detected: false,
  confidence: 0,
  landmarks: [],
  leftEAR: 0,
  rightEAR: 0,
  avgEAR: 0,
  isBlinking: false,
  blinkCount: 0,
  mouthOpenRatio: 0,
  headPose: { pitch: 0, yaw: 0, roll: 0 },
  expression: {
    label: 'Neutral',
    confidence: 0,
    scores: {
      Neutral: 1,
      Happy: 0,
      Surprised: 0,
      Angry: 0,
      Sad: 0,
      Fearful: 0,
      Disgusted: 0,
    },
    timestamp: Date.now(),
  },
  topBlendshapes: [],
};

const VisionContext = createContext<VisionContextValue | null>(null);

export const VisionProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);

  const faceLandmarkerRef = useRef<FaceLandmarker | null>(null);
  const handLandmarkerRef = useRef<HandLandmarker | null>(null);

  const faceAnalyzerRef = useRef(new FaceExpressionAnalyzer());
  const handProcessorRef = useRef(new HandTrackingProcessor());

  const frameStateRef = useRef<VisionFrameState>({
    face: DEFAULT_FACE,
    hands: [],
    fps: 0,
    inferenceMs: 0,
  });

  const [inputMode, setInputModeState] = useState<InputSourceMode>('webcam');
  const [cameraActive, setCameraActive] = useState(false);
  const [cameraError, setCameraError] = useState<string | null>(null);
  const [modelsReady, setModelsReady] = useState(false);
  const [modelsLoading, setModelsLoading] = useState(false);
  const [modelLoadError, setModelLoadError] = useState<string | null>(null);

  const [settings, setSettings] = useState<VisionSettings>({
    mirror: true,
    showFaceMesh: true,
    showHandSkeleton: true,
    minDetectionConfidence: 0.5,
    minTrackingConfidence: 0.5,
    expressionSmoothing: 0.28,
    predictionIntervalMs: 80,
    resolution: '1280x720',
  });
  const settingsRef = useRef(settings);
  settingsRef.current = settings;

  const [simConfig, setSimConfig] = useState<SimulationConfig>({
    expression: 'Happy',
    gesture: 'Open Palm',
    shapeTrajectory: 'none',
    pointerTarget: null,
    triggerPinch: false,
  });
  const simConfigRef = useRef(simConfig);
  simConfigRef.current = simConfig;

  const [face, setFace] = useState<FaceMetrics>(DEFAULT_FACE);
  const [hands, setHands] = useState<TrackedHand[]>([]);
  const [fps, setFps] = useState(0);
  const [fpsHistory, setFpsHistory] = useState<number[]>([]);
  const [inferenceMs, setInferenceMs] = useState(0);
  const [expressionHistory, setExpressionHistory] = useState<ExpressionPrediction[]>([]);
  const [gestureCounts, setGestureCounts] = useState<Record<string, number>>({});
  const [logs, setLogs] = useState<InteractionLogEntry[]>([]);

  // ML & DL State
  const [classifierMode, setClassifierMode] = useState<GestureClassifierMode>('rules');
  const classifierModeRef = useRef<GestureClassifierMode>('rules');
  classifierModeRef.current = classifierMode;

  const [dataset, setDataset] = useState<GestureSample[]>(() => generateSeededDataset(16));
  const knnRef = useRef<KNNClassifier>(new KNNClassifier(5));
  const mlpRef = useRef<MLPGestureNetwork>(new MLPGestureNetwork());
  const [evaluationReport, setEvaluationReport] = useState<ModelEvaluationReport | null>(null);

  // Train initial KNN & MLP on seeded dataset immediately on boot
  useEffect(() => {
    const initial = evaluateClassifier(dataset, 'mlp');
    knnRef.current = initial.knn;
    mlpRef.current = initial.mlp;
    setEvaluationReport(initial.report);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const addLog = useCallback(
    (module: InteractionLogEntry['module'], event: string, detail: string) => {
      setLogs((prev) => [
        {
          id: `${Date.now()}-${Math.random().toString(36).slice(2, 7)}`,
          timestamp: Date.now(),
          module,
          event,
          detail,
        },
        ...prev.slice(0, 79),
      ]);
    },
    []
  );

  const loadModels = useCallback(async () => {
    if (faceLandmarkerRef.current && handLandmarkerRef.current) {
      setModelsReady(true);
      return true;
    }
    setModelsLoading(true);
    setModelLoadError(null);
    try {
      const vision = await FilesetResolver.forVisionTasks('/mediapipe/wasm');

      const createTasks = async (delegate: 'GPU' | 'CPU') => {
        const fl = await FaceLandmarker.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath: '/mediapipe/models/face_landmarker.task',
            delegate,
          },
          runningMode: 'VIDEO',
          numFaces: 1,
          minFaceDetectionConfidence: settingsRef.current.minDetectionConfidence,
          minFacePresenceConfidence: settingsRef.current.minDetectionConfidence,
          minTrackingConfidence: settingsRef.current.minTrackingConfidence,
          outputFaceBlendshapes: true,
          outputFacialTransformationMatrixes: true,
        });

        const hl = await HandLandmarker.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath: '/mediapipe/models/hand_landmarker.task',
            delegate,
          },
          runningMode: 'VIDEO',
          numHands: 2,
          minHandDetectionConfidence: settingsRef.current.minDetectionConfidence,
          minHandPresenceConfidence: settingsRef.current.minDetectionConfidence,
          minTrackingConfidence: settingsRef.current.minTrackingConfidence,
        });

        return { fl, hl };
      };

      let tasks;
      try {
        tasks = await createTasks('GPU');
      } catch {
        tasks = await createTasks('CPU');
      }

      faceLandmarkerRef.current = tasks.fl;
      handLandmarkerRef.current = tasks.hl;
      setModelsReady(true);
      setModelsLoading(false);
      addLog('MLLab', 'Models Ready', 'MediaPipe FaceLandmarker & HandLandmarker initialized');
      return true;
    } catch (err) {
      const msg = err instanceof Error ? err.message : 'Failed to initialize MediaPipe models';
      setModelLoadError(msg);
      setModelsLoading(false);
      return false;
    }
  }, [addLog]);

  const stopCamera = useCallback(() => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    setCameraActive(false);
    faceAnalyzerRef.current.reset();
    handProcessorRef.current.reset();
  }, []);

  const startCamera = useCallback(async () => {
    setCameraError(null);
    setInputModeState('webcam');

    const ok = await loadModels();
    if (!ok) return;

    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      setCameraError('Browser camera API unavailable. Use Simulation mode to test all pipelines.');
      return;
    }

    try {
      const [w, h] = settingsRef.current.resolution === '1280x720' ? [1280, 720] : [640, 480];
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: w },
          height: { ideal: h },
          facingMode: 'user',
        },
        audio: false,
      });

      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }
      setCameraActive(true);
      addLog('Face', 'Camera Active', `Live stream started (${w}x${h})`);
    } catch (err) {
      const msg =
        err instanceof Error
          ? err.message
          : 'Camera permission denied or hardware not found.';
      setCameraError(msg);
      setCameraActive(false);
    }
  }, [addLog, loadModels]);

  const setInputMode = useCallback(
    (mode: InputSourceMode) => {
      setInputModeState(mode);
      if (mode === 'simulation') {
        stopCamera();
        setCameraError(null);
        addLog('MLLab', 'Simulation Active', 'Synthetic 478-pt face & 21-pt hand stream enabled');
      }
    },
    [addLog, stopCamera]
  );

  const updateSettings = useCallback((patch: Partial<VisionSettings>) => {
    setSettings((prev) => ({ ...prev, ...patch }));
  }, []);

  const updateSimConfig = useCallback((patch: Partial<SimulationConfig>) => {
    setSimConfig((prev) => ({ ...prev, ...patch }));
  }, []);

  const addGestureSamples = useCallback(
    (label: string, featureVectors: number[][]) => {
      const now = Date.now();
      const newSamples: GestureSample[] = featureVectors.map((features, i) => ({
        id: `cam-${label}-${now}-${i}`,
        label,
        features,
        source: 'webcam',
        createdAt: now,
      }));
      setDataset((prev) => [...prev, ...newSamples]);
      addLog('MLLab', 'Samples Recorded', `Added ${newSamples.length} frames for "${label}"`);
    },
    [addLog]
  );

  const clearCustomSamples = useCallback(() => {
    setDataset(generateSeededDataset(16));
    addLog('MLLab', 'Dataset Reset', 'Reverted to 160 canonical seeded gesture prototypes');
  }, [addLog]);

  const trainAndEvaluate = useCallback(
    (modelType: 'knn' | 'mlp') => {
      const res = evaluateClassifier(dataset, modelType);
      knnRef.current = res.knn;
      mlpRef.current = res.mlp;
      setEvaluationReport(res.report);
      addLog(
        'MLLab',
        `${modelType.toUpperCase()} Trained`,
        `Accuracy ${(res.report.accuracy * 100).toFixed(1)}% · F1 ${(res.report.macroF1 * 100).toFixed(1)}%`
      );
      return res.report;
    },
    [addLog, dataset]
  );

  // Main real-time CV/ML processing loop
  useEffect(() => {
    let rafId = 0;
    let lastVideoTime = -1;
    let frameCount = 0;
    let fpsWindowStart = performance.now();
    let lastUiSync = 0;
    let lastLoggedGesture = '';
    let lastLoggedExpr: ExpressionLabel | '' = '';

    const classifyHand = (
      features: number[],
      landmarks: TrackedHand['landmarks'],
      fingers: TrackedHand['fingerStates'],
      pinchDist: number
    ) => {
      const mode = classifierModeRef.current;
      if (mode === 'knn') {
        return knnRef.current.predict(features);
      }
      if (mode === 'mlp' && mlpRef.current.ready()) {
        return mlpRef.current.predict(features);
      }
      return classifyGestureRules(landmarks, fingers, pinchDist);
    };

    const tick = () => {
      const now = performance.now();
      const tStart = now;

      let faceResult = null;
      let handResult = null;

      if (inputMode === 'simulation') {
        const tSec = now / 1000;
        faceResult = generateSyntheticFaceResult(simConfigRef.current.expression, tSec);
        handResult = generateSyntheticHandResult(simConfigRef.current, tSec);
      } else if (
        cameraActive &&
        videoRef.current &&
        videoRef.current.readyState >= 2 &&
        faceLandmarkerRef.current &&
        handLandmarkerRef.current
      ) {
        const video = videoRef.current;
        if (video.currentTime !== lastVideoTime) {
          lastVideoTime = video.currentTime;
          try {
            faceResult = faceLandmarkerRef.current.detectForVideo(video, now);
            handResult = handLandmarkerRef.current.detectForVideo(video, now);
          } catch {
            // Skip transient frame errors on stream changes
          }
        } else {
          rafId = requestAnimationFrame(tick);
          return;
        }
      }

      if (faceResult || handResult) {
        const nextFace = faceAnalyzerRef.current.analyze(
          faceResult,
          settingsRef.current.predictionIntervalMs,
          settingsRef.current.expressionSmoothing
        );
        const nextHands = handProcessorRef.current.process(handResult, classifyHand);
        const dtInf = performance.now() - tStart;

        frameCount++;
        let currentFps = frameStateRef.current.fps;
        if (now - fpsWindowStart >= 500) {
          currentFps = Math.round((frameCount * 1000) / (now - fpsWindowStart));
          frameCount = 0;
          fpsWindowStart = now;
        }

        frameStateRef.current = {
          face: nextFace,
          hands: nextHands,
          fps: currentFps,
          inferenceMs: Number(dtInf.toFixed(1)),
        };

        // Sync React dashboard state at ~12Hz to avoid React re-render bottlenecks
        if (now - lastUiSync >= 85) {
          lastUiSync = now;
          setFace(nextFace);
          setHands(nextHands);
          setFps(currentFps);
          setInferenceMs(Number(dtInf.toFixed(1)));

          if (currentFps > 0) {
            setFpsHistory((prev) => [...prev.slice(-29), currentFps]);
          }

          if (nextFace.detected) {
            setExpressionHistory((prev) => [...prev.slice(-39), nextFace.expression]);
            if (
              nextFace.expression.label !== lastLoggedExpr &&
              nextFace.expression.confidence > 0.45
            ) {
              lastLoggedExpr = nextFace.expression.label;
            }
          }

          if (nextHands.length > 0) {
            const g = nextHands[0].gesture;
            if (g && g !== 'Unknown') {
              if (g !== lastLoggedGesture) {
                lastLoggedGesture = g;
                setGestureCounts((prev) => ({
                  ...prev,
                  [g]: (prev[g] ?? 0) + 1,
                }));
              }
            }
          }
        }
      }

      rafId = requestAnimationFrame(tick);
    };

    if (cameraActive || inputMode === 'simulation') {
      rafId = requestAnimationFrame(tick);
    } else {
      frameStateRef.current = {
        face: DEFAULT_FACE,
        hands: [],
        fps: 0,
        inferenceMs: 0,
      };
      setFace(DEFAULT_FACE);
      setHands([]);
      setFps(0);
    }

    return () => {
      if (rafId) cancelAnimationFrame(rafId);
    };
  }, [cameraActive, inputMode]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      stopCamera();
      faceLandmarkerRef.current?.close();
      handLandmarkerRef.current?.close();
    };
  }, [stopCamera]);

  return (
    <VisionContext.Provider
      value={{
        videoRef,
        frameStateRef,
        face,
        hands,
        fps,
        fpsHistory,
        inferenceMs,
        expressionHistory,
        gestureCounts,
        logs,
        addLog,
        inputMode,
        setInputMode,
        cameraActive,
        cameraError,
        modelsReady,
        modelsLoading,
        modelLoadError,
        startCamera,
        stopCamera,
        settings,
        updateSettings,
        simConfig,
        updateSimConfig,
        classifierMode,
        setClassifierMode,
        dataset,
        addGestureSamples,
        clearCustomSamples,
        trainAndEvaluate,
        evaluationReport,
      }}
    >
      {/* Hidden persistent video element so switching views keeps camera stream uninterrupted */}
      <video ref={videoRef} playsInline muted className="hidden" />
      {children}
    </VisionContext.Provider>
  );
};

export function useVision(): VisionContextValue {
  const ctx = useContext(VisionContext);
  if (!ctx) throw new Error('useVision must be used inside VisionProvider');
  return ctx;
}
