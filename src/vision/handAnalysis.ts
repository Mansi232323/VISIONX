import type { HandLandmarkerResult } from '@mediapipe/tasks-vision';
import {
  DynamicMotion,
  FingerStates,
  Point2D,
  Point3D,
  TrackedHand,
} from '../types/vision';
import { LandmarkArraySmoother, Point2DSmoother } from './smoothing';

function dist3D(a: Point3D, b: Point3D): number {
  return Math.hypot(a.x - b.x, a.y - b.y, (a.z ?? 0) - (b.z ?? 0));
}

function dist2D(a: Point3D | Point2D, b: Point3D | Point2D): number {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

function angleBetween(a: Point3D, b: Point3D, c: Point3D): number {
  const v1x = a.x - b.x;
  const v1y = a.y - b.y;
  const v1z = (a.z ?? 0) - (b.z ?? 0);
  const v2x = c.x - b.x;
  const v2y = c.y - b.y;
  const v2z = (c.z ?? 0) - (b.z ?? 0);
  const m1 = Math.hypot(v1x, v1y, v1z);
  const m2 = Math.hypot(v2x, v2y, v2z);
  if (m1 < 1e-6 || m2 < 1e-6) return 180;
  const dot = (v1x * v2x + v1y * v2y + v1z * v2z) / (m1 * m2);
  return (Math.acos(Math.min(1, Math.max(-1, dot))) * 180) / Math.PI;
}

export function detectFingerStates(lm: Point3D[]): FingerStates {
  const wrist = lm[0];
  const palmScale = Math.max(0.05, dist3D(wrist, lm[9]));

  // Thumb: compare tip-to-pinky-MCP vs IP-to-pinky-MCP & thumb angle
  const thumbAngle = angleBetween(lm[2], lm[3], lm[4]);
  const thumbOut = dist2D(lm[4], lm[17]) > dist2D(lm[3], lm[17]) * 1.12 && thumbAngle > 142;

  // Non-thumb fingers: tip further from wrist than PIP, plus PIP angle check
  const isExtended = (mcp: number, pip: number, dip: number, tip: number) => {
    const dTip = dist2D(lm[tip], wrist);
    const dPip = dist2D(lm[pip], wrist);
    const dMcp = dist2D(lm[mcp], wrist);
    const pipAngle = angleBetween(lm[mcp], lm[pip], lm[dip]);
    return dTip > dPip + palmScale * 0.08 && dTip > dMcp && pipAngle > 135;
  };

  return {
    thumb: thumbOut,
    index: isExtended(5, 6, 7, 8),
    middle: isExtended(9, 10, 11, 12),
    ring: isExtended(13, 14, 15, 16),
    pinky: isExtended(17, 18, 19, 20),
  };
}

/**
 * Extracts a 28-dimensional translation-invariant, scale-normalized feature vector
 * combining joint bend angles, wrist-to-fingertip ratios, pinch gaps, finger spreads,
 * and palm orientation for KNN & MLP Neural Network gesture classification.
 */
export function extractHandFeatures(lm: Point3D[]): number[] {
  const wrist = lm[0];
  const middleMcp = lm[9];
  const palmScale = Math.max(0.04, dist3D(wrist, middleMcp));

  // 1. Five normalized wrist-to-tip distances (5)
  const tipIndices = [4, 8, 12, 16, 20];
  const tipRatios = tipIndices.map((idx) => Math.min(3, dist3D(wrist, lm[idx]) / palmScale) / 3);

  // 2. Five PIP/IP joint bend angles normalized to [0, 1] (5)
  const jointTriples: [number, number, number][] = [
    [2, 3, 4],
    [5, 6, 8],
    [9, 10, 12],
    [13, 14, 16],
    [17, 18, 20],
  ];
  const jointAngles = jointTriples.map(([a, b, c]) => angleBetween(lm[a], lm[b], lm[c]) / 180);

  // 3. Four thumb-to-other-fingertip normalized distances (4)
  const thumbPinches = [8, 12, 16, 20].map((idx) => Math.min(2.5, dist3D(lm[4], lm[idx]) / palmScale) / 2.5);

  // 4. Three adjacent fingertip spread ratios (3)
  const spreads = [
    dist3D(lm[8], lm[12]),
    dist3D(lm[12], lm[16]),
    dist3D(lm[16], lm[20]),
  ].map((d) => Math.min(2, d / palmScale) / 2);

  // 5. Five vertical fingertip offsets relative to MCP (captures Thumbs Up vs Thumbs Down) (5)
  const mcpIndices = [2, 5, 9, 13, 17];
  const verticalOffsets = tipIndices.map((tipIdx, i) => {
    const dy = (lm[mcpIndices[i]].y - lm[tipIdx].y) / palmScale;
    return Math.max(-1, Math.min(1, dy / 1.5)) * 0.5 + 0.5;
  });

  // 6. Five horizontal fingertip offsets relative to wrist (captures handedness-agnostic spread) (5)
  const horizontalOffsets = tipIndices.map((tipIdx) => {
    const dx = (lm[tipIdx].x - wrist.x) / palmScale;
    return Math.max(-1, Math.min(1, dx / 1.5)) * 0.5 + 0.5;
  });

  // 7. Palm orientation pitch/tilt proxy (1)
  const palmVerticalTilt = Math.max(-1, Math.min(1, (wrist.y - middleMcp.y) / palmScale)) * 0.5 + 0.5;

  return [
    ...tipRatios,
    ...jointAngles,
    ...thumbPinches,
    ...spreads,
    ...verticalOffsets,
    ...horizontalOffsets,
    palmVerticalTilt,
  ];
}

export function classifyGestureRules(
  lm: Point3D[],
  fingers: FingerStates,
  pinchDistance: number
): { gesture: string; confidence: number } {
  const wrist = lm[0];
  const palmScale = Math.max(0.04, dist2D(wrist, lm[9]));
  const normPinch = pinchDistance / palmScale;

  const { thumb, index, middle, ring, pinky } = fingers;
  const count = [thumb, index, middle, ring, pinky].filter(Boolean).length;

  // OK Sign: thumb & index tips touching, middle/ring/pinky extended
  if (normPinch < 0.36 && middle && ring && pinky) {
    return { gesture: 'OK Sign', confidence: 0.94 };
  }

  // Pinch: thumb & index close, other fingers curled or relaxed
  if (normPinch < 0.34 && !middle && !ring) {
    return { gesture: 'Pinch', confidence: 0.93 };
  }

  // Open Palm: all 5 extended
  if (index && middle && ring && pinky && thumb) {
    return { gesture: 'Open Palm', confidence: 0.96 };
  }

  // Four Fingers: 4 non-thumb fingers extended, thumb tucked
  if (!thumb && index && middle && ring && pinky) {
    return { gesture: 'Four Fingers', confidence: 0.92 };
  }

  // Three Fingers: index, middle, ring extended
  if (index && middle && ring && !pinky) {
    return { gesture: 'Three Fingers', confidence: 0.91 };
  }

  // Peace Sign: index & middle extended
  if (index && middle && !ring && !pinky) {
    return { gesture: 'Peace Sign', confidence: 0.94 };
  }

  // Pointing Index: only index extended
  if (index && !middle && !ring && !pinky) {
    return { gesture: 'Pointing Index', confidence: 0.95 };
  }

  // Thumbs Up / Thumbs Down: thumb extended, 4 fingers curled
  if (thumb && !index && !middle && !ring && !pinky) {
    const thumbDy = lm[4].y - lm[2].y;
    if (thumbDy < -palmScale * 0.25) {
      return { gesture: 'Thumbs Up', confidence: 0.93 };
    }
    if (thumbDy > palmScale * 0.25) {
      return { gesture: 'Thumbs Down', confidence: 0.92 };
    }
  }

  // Closed Fist: 0 fingers extended (and thumb curled inward)
  if (count === 0) {
    return { gesture: 'Closed Fist', confidence: 0.94 };
  }

  return { gesture: 'Unknown', confidence: 0.45 };
}

export class HandTrackingProcessor {
  private landmarkSmoothers = [new LandmarkArraySmoother(0.55), new LandmarkArraySmoother(0.55)];
  private indexSmoothers = [new Point2DSmoother(0.45), new Point2DSmoother(0.45)];
  private prevPalmCenters: Array<Point2D | null> = [null, null];
  private motionCooldowns = [0, 0];

  reset() {
    this.landmarkSmoothers.forEach((s) => s.reset());
    this.indexSmoothers.forEach((s) => s.reset());
    this.prevPalmCenters = [null, null];
    this.motionCooldowns = [0, 0];
  }

  process(
    result: HandLandmarkerResult | null,
    classifierFn?: (features: number[], landmarks: Point3D[], fingers: FingerStates, pinchDist: number) => {
      gesture: string;
      confidence: number;
    }
  ): TrackedHand[] {
    if (!result || !result.landmarks || result.landmarks.length === 0) {
      return [];
    }

    const now = performance.now();
    const hands: TrackedHand[] = [];

    for (let i = 0; i < Math.min(2, result.landmarks.length); i++) {
      const rawLm = result.landmarks[i];
      const smoothedLm = this.landmarkSmoothers[i].update(rawLm);
      const handednessCat = result.handedness?.[i]?.[0];
      const rawHandedness = handednessCat?.categoryName === 'Left' ? 'Left' : 'Right';
      const handednessScore = handednessCat?.score ?? 0.9;

      const fingerStates = detectFingerStates(smoothedLm);
      const extendedCount = Object.values(fingerStates).filter(Boolean).length;

      const indexTip = this.indexSmoothers[i].update({
        x: smoothedLm[8].x,
        y: smoothedLm[8].y,
      });
      const thumbTip: Point2D = { x: smoothedLm[4].x, y: smoothedLm[4].y };
      const palmCenter: Point2D = {
        x: (smoothedLm[0].x + smoothedLm[5].x + smoothedLm[9].x + smoothedLm[17].x) / 4,
        y: (smoothedLm[0].y + smoothedLm[5].y + smoothedLm[9].y + smoothedLm[17].y) / 4,
      };

      const pinchDistance = dist2D(smoothedLm[4], smoothedLm[8]);
      const palmScale = Math.max(0.04, dist2D(smoothedLm[0], smoothedLm[9]));
      const isPinching = pinchDistance / palmScale < 0.34;

      const features = extractHandFeatures(smoothedLm);
      const classified = classifierFn
        ? classifierFn(features, smoothedLm, fingerStates, pinchDistance)
        : classifyGestureRules(smoothedLm, fingerStates, pinchDistance);

      // Velocity & dynamic swipe estimation
      const prevCenter = this.prevPalmCenters[i];
      const velocity: Point2D = prevCenter
        ? { x: palmCenter.x - prevCenter.x, y: palmCenter.y - prevCenter.y }
        : { x: 0, y: 0 };
      this.prevPalmCenters[i] = palmCenter;

      let motion: DynamicMotion = 'Idle';
      if (now > this.motionCooldowns[i]) {
        const speed = Math.hypot(velocity.x, velocity.y);
        if (speed > 0.036) {
          if (Math.abs(velocity.x) > Math.abs(velocity.y)) {
            motion = velocity.x > 0 ? 'Swipe Right' : 'Swipe Left';
          } else {
            motion = velocity.y > 0 ? 'Swipe Down' : 'Swipe Up';
          }
          this.motionCooldowns[i] = now + 450;
        }
      }

      hands.push({
        id: i,
        handedness: rawHandedness,
        handednessScore,
        landmarks: smoothedLm,
        fingerStates,
        extendedCount,
        pinchDistance,
        isPinching,
        indexTip,
        thumbTip,
        palmCenter,
        features,
        gesture: classified.gesture,
        gestureConfidence: classified.confidence,
        motion,
        velocity,
      });
    }

    return hands;
  }
}
