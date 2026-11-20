import type { FaceLandmarkerResult } from '@mediapipe/tasks-vision';
import {
  ExpressionLabel,
  ExpressionPrediction,
  FaceMetrics,
  HeadPose,
  Point3D,
} from '../types/vision';

const EXPRESSION_LABELS: ExpressionLabel[] = [
  'Neutral',
  'Happy',
  'Surprised',
  'Angry',
  'Sad',
  'Fearful',
  'Disgusted',
];

function dist2D(a: Point3D, b: Point3D): number {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

function computeEAR(landmarks: Point3D[], indices: number[]): number {
  const [p1, p2, p3, p4, p5, p6] = indices.map((i) => landmarks[i]);
  if (!p1 || !p2 || !p3 || !p4 || !p5 || !p6) return 0.3;
  const v1 = dist2D(p2, p6);
  const v2 = dist2D(p3, p5);
  const h = dist2D(p1, p4);
  if (h < 1e-5) return 0.3;
  return (v1 + v2) / (2.0 * h);
}

function computeMouthRatio(landmarks: Point3D[]): number {
  const top = landmarks[13];
  const bottom = landmarks[14];
  const left = landmarks[61];
  const right = landmarks[291];
  if (!top || !bottom || !left || !right) return 0;
  const width = dist2D(left, right);
  if (width < 1e-5) return 0;
  return dist2D(top, bottom) / width;
}

function extractHeadPose(matrixData?: number[]): HeadPose {
  if (!matrixData || matrixData.length < 16) {
    return { pitch: 0, yaw: 0, roll: 0 };
  }
  // MediaPipe 4x4 transformation matrix (column-major or row-major rotation submatrix)
  const r00 = matrixData[0];
  const r10 = matrixData[4];
  const r20 = matrixData[8];
  const r21 = matrixData[9];
  const r22 = matrixData[10];

  const sy = Math.sqrt(r00 * r00 + r10 * r10);
  const singular = sy < 1e-6;

  let pitch = 0;
  let yaw = 0;
  let roll = 0;

  if (!singular) {
    pitch = Math.atan2(r21, r22);
    yaw = Math.atan2(-r20, sy);
    roll = Math.atan2(r10, r00);
  } else {
    pitch = Math.atan2(-matrixData[6], matrixData[5]);
    yaw = Math.atan2(-r20, sy);
    roll = 0;
  }

  const radToDeg = 180 / Math.PI;
  return {
    pitch: Number((pitch * radToDeg).toFixed(1)),
    yaw: Number((yaw * radToDeg).toFixed(1)),
    roll: Number((roll * radToDeg).toFixed(1)),
  };
}

export class FaceExpressionAnalyzer {
  private smoothedScores: Record<ExpressionLabel, number> = {
    Neutral: 1,
    Happy: 0,
    Surprised: 0,
    Angry: 0,
    Sad: 0,
    Fearful: 0,
    Disgusted: 0,
  };
  private blinkCount = 0;
  private wasBlinking = false;
  private lastPredictionTs = 0;
  private cachedExpression: ExpressionPrediction | null = null;

  reset() {
    this.blinkCount = 0;
    this.wasBlinking = false;
    this.cachedExpression = null;
  }

  analyze(
    result: FaceLandmarkerResult | null,
    predictionIntervalMs = 80,
    smoothingFactor = 0.28
  ): FaceMetrics {
    if (!result || !result.faceLandmarks || result.faceLandmarks.length === 0) {
      return {
        detected: false,
        confidence: 0,
        landmarks: [],
        leftEAR: 0,
        rightEAR: 0,
        avgEAR: 0,
        isBlinking: false,
        blinkCount: this.blinkCount,
        mouthOpenRatio: 0,
        headPose: { pitch: 0, yaw: 0, roll: 0 },
        expression: this.cachedExpression ?? {
          label: 'Neutral',
          confidence: 0,
          scores: { ...this.smoothedScores },
          timestamp: Date.now(),
        },
        topBlendshapes: [],
      };
    }

    const landmarks = result.faceLandmarks[0];
    const leftEAR = computeEAR(landmarks, [33, 160, 158, 133, 153, 144]);
    const rightEAR = computeEAR(landmarks, [362, 385, 387, 263, 373, 380]);
    const avgEAR = (leftEAR + rightEAR) / 2;

    const isBlinking = avgEAR < 0.21;
    if (isBlinking && !this.wasBlinking) {
      this.blinkCount += 1;
    }
    this.wasBlinking = isBlinking;

    const mouthOpenRatio = computeMouthRatio(landmarks);
    const matrixData = result.facialTransformationMatrixes?.[0]?.data;
    const headPose = extractHeadPose(matrixData);

    const blendshapeCats = result.faceBlendshapes?.[0]?.categories ?? [];
    const bs: Record<string, number> = {};
    for (const c of blendshapeCats) {
      bs[c.categoryName] = c.score;
    }

    const topBlendshapes = [...blendshapeCats]
      .filter((c) => c.categoryName !== '_neutral')
      .sort((a, b) => b.score - a.score)
      .slice(0, 6)
      .map((c) => ({ name: c.categoryName, score: c.score }));

    const now = performance.now();
    if (!this.cachedExpression || now - this.lastPredictionTs >= predictionIntervalMs) {
      this.lastPredictionTs = now;
      const raw = this.classifyFromBlendshapesAndGeometry(bs, mouthOpenRatio, avgEAR);

      // Normalize + temporal EMA smoothing
      const totalRaw = Object.values(raw).reduce((s, v) => s + v, 0) || 1;
      for (const label of EXPRESSION_LABELS) {
        const norm = raw[label] / totalRaw;
        this.smoothedScores[label] =
          this.smoothedScores[label] * (1 - smoothingFactor) + norm * smoothingFactor;
      }

      let bestLabel: ExpressionLabel = 'Neutral';
      let bestScore = -1;
      for (const label of EXPRESSION_LABELS) {
        if (this.smoothedScores[label] > bestScore) {
          bestScore = this.smoothedScores[label];
          bestLabel = label;
        }
      }

      this.cachedExpression = {
        label: bestLabel,
        confidence: Math.min(0.99, Math.max(0.15, bestScore)),
        scores: { ...this.smoothedScores },
        timestamp: Date.now(),
      };
    }

    return {
      detected: true,
      confidence: 0.96,
      landmarks,
      leftEAR: Number(leftEAR.toFixed(3)),
      rightEAR: Number(rightEAR.toFixed(3)),
      avgEAR: Number(avgEAR.toFixed(3)),
      isBlinking,
      blinkCount: this.blinkCount,
      mouthOpenRatio: Number(mouthOpenRatio.toFixed(3)),
      headPose,
      expression: this.cachedExpression,
      topBlendshapes,
    };
  }

  private classifyFromBlendshapesAndGeometry(
    bs: Record<string, number>,
    mouthOpenRatio: number,
    avgEAR: number
  ): Record<ExpressionLabel, number> {
    const smile = ((bs['mouthSmileLeft'] ?? 0) + (bs['mouthSmileRight'] ?? 0)) / 2;
    const cheekSquint = ((bs['cheekSquintLeft'] ?? 0) + (bs['cheekSquintRight'] ?? 0)) / 2;
    const jawOpen = Math.max(bs['jawOpen'] ?? 0, Math.min(1, mouthOpenRatio * 1.4));
    const eyeWide = ((bs['eyeWideLeft'] ?? 0) + (bs['eyeWideRight'] ?? 0)) / 2;
    const browInnerUp = bs['browInnerUp'] ?? 0;
    const browOuterUp = ((bs['browOuterUpLeft'] ?? 0) + (bs['browOuterUpRight'] ?? 0)) / 2;
    const browDown = ((bs['browDownLeft'] ?? 0) + (bs['browDownRight'] ?? 0)) / 2;
    const frown = ((bs['mouthFrownLeft'] ?? 0) + (bs['mouthFrownRight'] ?? 0)) / 2;
    const noseSneer = ((bs['noseSneerLeft'] ?? 0) + (bs['noseSneerRight'] ?? 0)) / 2;
    const mouthUpperUp = ((bs['mouthUpperUpLeft'] ?? 0) + (bs['mouthUpperUpRight'] ?? 0)) / 2;
    const mouthStretch = ((bs['mouthStretchLeft'] ?? 0) + (bs['mouthStretchRight'] ?? 0)) / 2;
    const mouthPress = ((bs['mouthPressLeft'] ?? 0) + (bs['mouthPressRight'] ?? 0)) / 2;

    const happy = Math.max(0, smile * 1.65 + cheekSquint * 0.55 - frown * 0.6);
    const surprised = Math.max(0, eyeWide * 1.15 + jawOpen * 0.95 + browOuterUp * 0.65 - browDown * 0.5);
    const angry = Math.max(0, browDown * 1.55 + mouthPress * 0.6 + noseSneer * 0.35 - smile * 0.8);
    const sad = Math.max(0, frown * 1.35 + browInnerUp * 0.85 - smile * 0.8 - jawOpen * 0.3);
    const fearful = Math.max(0, eyeWide * 0.9 + browInnerUp * 0.8 + mouthStretch * 0.9 + jawOpen * 0.4 - smile * 0.7);
    const disgusted = Math.max(0, noseSneer * 1.6 + mouthUpperUp * 0.85 + browDown * 0.35 - smile * 0.7);

    const maxExpr = Math.max(happy, surprised, angry, sad, fearful, disgusted);
    const neutral = Math.max(0.18, 0.72 - maxExpr * 0.95 + (avgEAR > 0.22 && maxExpr < 0.25 ? 0.25 : 0));

    return {
      Neutral: neutral,
      Happy: happy,
      Surprised: surprised,
      Angry: angry,
      Sad: sad,
      Fearful: fearful,
      Disgusted: disgusted,
    };
  }
}
