export interface Point2D {
  x: number;
  y: number;
}

export interface Point3D {
  x: number;
  y: number;
  z: number;
}

export type ExpressionLabel =
  | 'Neutral'
  | 'Happy'
  | 'Surprised'
  | 'Angry'
  | 'Sad'
  | 'Fearful'
  | 'Disgusted';

export interface ExpressionPrediction {
  label: ExpressionLabel;
  confidence: number;
  scores: Record<ExpressionLabel, number>;
  timestamp: number;
}

export interface HeadPose {
  pitch: number; // degrees, + down / - up
  yaw: number;   // degrees, + right / - left
  roll: number;  // degrees
}

export interface FaceMetrics {
  detected: boolean;
  confidence: number;
  landmarks: Point3D[];
  leftEAR: number;
  rightEAR: number;
  avgEAR: number;
  isBlinking: boolean;
  blinkCount: number;
  mouthOpenRatio: number;
  headPose: HeadPose;
  expression: ExpressionPrediction;
  topBlendshapes: Array<{ name: string; score: number }>;
}

export type FingerName = 'thumb' | 'index' | 'middle' | 'ring' | 'pinky';

export type FingerStates = Record<FingerName, boolean>;

export type GestureClassifierMode = 'rules' | 'knn' | 'mlp';

export type BuiltinGesture =
  | 'Open Palm'
  | 'Closed Fist'
  | 'Pointing Index'
  | 'Pinch'
  | 'Peace Sign'
  | 'Thumbs Up'
  | 'Thumbs Down'
  | 'OK Sign'
  | 'Three Fingers'
  | 'Four Fingers'
  | 'Unknown';

export type DynamicMotion =
  | 'Idle'
  | 'Swipe Left'
  | 'Swipe Right'
  | 'Swipe Up'
  | 'Swipe Down';

export interface TrackedHand {
  id: number;
  handedness: 'Left' | 'Right';
  handednessScore: number;
  landmarks: Point3D[];
  fingerStates: FingerStates;
  extendedCount: number;
  pinchDistance: number;
  isPinching: boolean;
  indexTip: Point2D;
  thumbTip: Point2D;
  palmCenter: Point2D;
  features: number[]; // 28-D engineered feature vector
  gesture: string;
  gestureConfidence: number;
  motion: DynamicMotion;
  velocity: Point2D;
}

export type GeometricShapeLabel =
  | 'Circle'
  | 'Ellipse'
  | 'Triangle'
  | 'Square'
  | 'Rectangle'
  | 'Pentagon'
  | 'Hexagon'
  | 'Star'
  | 'Arrow'
  | 'Line'
  | 'Freeform';

export interface ShapeRecognitionResult {
  geometricShape: GeometricShapeLabel;
  confidence: number;
  customMatchName: string | null;
  customMatchConfidence: number | null;
  metrics: {
    vertices: number;
    circularity: number;
    aspectRatio: number;
    solidity: number;
    perimeter: number;
    area: number;
  };
  normalizedPoints: Point2D[];
}

export interface SavedCustomShape {
  id: string;
  customName: string;
  baseShape: GeometricShapeLabel;
  normalizedPoints: Point2D[];
  aspectRatio: number;
  circularity: number;
  vertices: number;
  createdAt: number;
  matchCount: number;
}

export interface Stroke {
  id: string;
  points: Point2D[];
  color: string;
  width: number;
  isEraser?: boolean;
}

export interface GestureSample {
  id: string;
  label: string;
  features: number[];
  source: 'seeded' | 'webcam';
  createdAt: number;
}

export interface ModelEvaluationReport {
  modelType: 'knn' | 'mlp';
  accuracy: number;
  macroPrecision: number;
  macroRecall: number;
  macroF1: number;
  labels: string[];
  confusionMatrix: number[][];
  lossHistory: number[];
  accuracyHistory: number[];
  trainCount: number;
  testCount: number;
  trainedAt: number;
}

export type YouTubeAction =
  | 'play_pause'
  | 'scroll_down'
  | 'scroll_up'
  | 'volume_up'
  | 'volume_down'
  | 'mute_toggle'
  | 'seek_forward'
  | 'seek_backward'
  | 'fullscreen_toggle';

export interface InteractionLogEntry {
  id: string;
  timestamp: number;
  module: 'Face' | 'Gesture' | 'AirDraw' | 'YouTube' | 'Keyboard' | 'MLLab';
  event: string;
  detail: string;
}

export type AppTab = 'overview' | 'air-studio' | 'youtube' | 'virtual-search' | 'ml-lab';
export type InputSourceMode = 'webcam' | 'simulation';
