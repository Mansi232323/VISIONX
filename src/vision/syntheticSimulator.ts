import type { FaceLandmarkerResult, HandLandmarkerResult } from '@mediapipe/tasks-vision';
import { ExpressionLabel, Point2D, Point3D } from '../types/vision';

export type SimShapeTrajectory = 'none' | 'circle' | 'triangle' | 'square' | 'star' | 'arrow';

export interface SimulationConfig {
  expression: ExpressionLabel;
  gesture: string;
  shapeTrajectory: SimShapeTrajectory;
  pointerTarget: Point2D | null;
  triggerPinch: boolean;
}

/**
 * Generates mathematically consistent 21-point 3D hand landmarks for any core gesture,
 * plus smooth air-drawing fingertip motion or virtual keyboard cursor targeting.
 */
export function generateSyntheticHandResult(
  config: SimulationConfig,
  tSec: number
): HandLandmarkerResult {
  // Determine index fingertip target position in normalized [0..1] coords
  let cx = 0.55 + Math.sin(tSec * 1.1) * 0.04;
  let cy = 0.62 + Math.cos(tSec * 0.9) * 0.03;

  if (config.shapeTrajectory !== 'none') {
    const phase = (tSec * 0.55) % 1;
    const angle = phase * Math.PI * 2;
    if (config.shapeTrajectory === 'circle') {
      cx = 0.5 + Math.cos(angle) * 0.19;
      cy = 0.48 + Math.sin(angle) * 0.22;
    } else if (config.shapeTrajectory === 'square') {
      const side = Math.floor(phase * 4);
      const u = (phase * 4) % 1;
      const pts: Point2D[] = [
        { x: 0.35, y: 0.32 },
        { x: 0.65, y: 0.32 },
        { x: 0.65, y: 0.68 },
        { x: 0.35, y: 0.68 },
      ];
      const a = pts[side % 4];
      const b = pts[(side + 1) % 4];
      cx = a.x + (b.x - a.x) * u;
      cy = a.y + (b.y - a.y) * u;
    } else if (config.shapeTrajectory === 'triangle') {
      const side = Math.floor(phase * 3);
      const u = (phase * 3) % 1;
      const pts: Point2D[] = [
        { x: 0.5, y: 0.26 },
        { x: 0.68, y: 0.68 },
        { x: 0.32, y: 0.68 },
      ];
      const a = pts[side % 3];
      const b = pts[(side + 1) % 3];
      cx = a.x + (b.x - a.x) * u;
      cy = a.y + (b.y - a.y) * u;
    } else if (config.shapeTrajectory === 'star') {
      const verts: Point2D[] = [];
      for (let i = 0; i < 10; i++) {
        const r = i % 2 === 0 ? 0.22 : 0.095;
        const th = (i * Math.PI) / 5 - Math.PI / 2;
        verts.push({ x: 0.5 + Math.cos(th) * r, y: 0.5 + Math.sin(th) * r * 1.1 });
      }
      const seg = Math.floor(phase * 10);
      const u = (phase * 10) % 1;
      const a = verts[seg % 10];
      const b = verts[(seg + 1) % 10];
      cx = a.x + (b.x - a.x) * u;
      cy = a.y + (b.y - a.y) * u;
    } else if (config.shapeTrajectory === 'arrow') {
      const pts: Point2D[] = [
        { x: 0.30, y: 0.52 },
        { x: 0.68, y: 0.52 },
        { x: 0.57, y: 0.38 },
        { x: 0.68, y: 0.52 },
        { x: 0.57, y: 0.66 },
      ];
      const seg = Math.min(3, Math.floor(phase * 4));
      const u = Math.min(1, (phase * 4) % 1);
      cx = pts[seg].x + (pts[seg + 1].x - pts[seg].x) * u;
      cy = pts[seg].y + (pts[seg + 1].y - pts[seg].y) * u;
    }
  } else if (config.pointerTarget) {
    cx = config.pointerTarget.x;
    cy = config.pointerTarget.y;
  }

  // Build 21 landmarks anchored so index tip (8) sits accurately at (cx, cy) when pointing/drawing
  const scale = 0.16;
  const wrist: Point3D = { x: cx - 0.03, y: cy + scale * 1.45, z: 0 };

  const g = config.triggerPinch ? 'Pinch' : config.gesture;
  const ext = {
    thumb: g === 'Open Palm' || g === 'Thumbs Up' || g === 'Thumbs Down',
    index:
      g === 'Open Palm' ||
      g === 'Pointing Index' ||
      g === 'Peace Sign' ||
      g === 'Three Fingers' ||
      g === 'Four Fingers' ||
      config.shapeTrajectory !== 'none',
    middle:
      g === 'Open Palm' ||
      g === 'Peace Sign' ||
      g === 'Three Fingers' ||
      g === 'Four Fingers' ||
      g === 'OK Sign',
    ring:
      g === 'Open Palm' ||
      g === 'Three Fingers' ||
      g === 'Four Fingers' ||
      g === 'OK Sign',
    pinky: g === 'Open Palm' || g === 'Four Fingers' || g === 'OK Sign',
  };

  const makeFinger = (
    mcpX: number,
    mcpY: number,
    extended: boolean,
    spreadX: number
  ): [Point3D, Point3D, Point3D, Point3D] => {
    const mcp: Point3D = { x: mcpX, y: mcpY, z: 0 };
    if (extended) {
      return [
        mcp,
        { x: mcpX + spreadX * 0.3, y: mcpY - scale * 0.32, z: -0.01 },
        { x: mcpX + spreadX * 0.6, y: mcpY - scale * 0.58, z: -0.02 },
        { x: mcpX + spreadX * 0.85, y: mcpY - scale * 0.85, z: -0.03 },
      ];
    }
    return [
      mcp,
      { x: mcpX + spreadX * 0.15, y: mcpY - scale * 0.18, z: -0.02 },
      { x: mcpX + spreadX * 0.05, y: mcpY + scale * 0.02, z: -0.04 },
      { x: mcpX, y: mcpY + scale * 0.16, z: -0.03 },
    ];
  };

  const indexChain = makeFinger(wrist.x - 0.04, wrist.y - scale * 0.72, ext.index, -0.015);
  const middleChain = makeFinger(wrist.x, wrist.y - scale * 0.76, ext.middle, 0.0);
  const ringChain = makeFinger(wrist.x + 0.04, wrist.y - scale * 0.73, ext.ring, 0.015);
  const pinkyChain = makeFinger(wrist.x + 0.075, wrist.y - scale * 0.66, ext.pinky, 0.03);

  // Override index tip exact position when pointing or drawing
  if (ext.index) {
    indexChain[3] = { x: cx, y: cy, z: -0.03 };
    indexChain[2] = { x: (indexChain[1].x + cx) / 2, y: (indexChain[1].y + cy) / 2, z: -0.02 };
  }

  let thumbChain: [Point3D, Point3D, Point3D, Point3D];
  if (g === 'Thumbs Up') {
    thumbChain = [
      { x: wrist.x - 0.05, y: wrist.y - scale * 0.2, z: 0 },
      { x: wrist.x - 0.08, y: wrist.y - scale * 0.55, z: -0.01 },
      { x: wrist.x - 0.09, y: wrist.y - scale * 0.9, z: -0.02 },
      { x: wrist.x - 0.09, y: wrist.y - scale * 1.25, z: -0.03 },
    ];
  } else if (g === 'Thumbs Down') {
    thumbChain = [
      { x: wrist.x - 0.05, y: wrist.y - scale * 0.1, z: 0 },
      { x: wrist.x - 0.08, y: wrist.y + scale * 0.25, z: -0.01 },
      { x: wrist.x - 0.09, y: wrist.y + scale * 0.58, z: -0.02 },
      { x: wrist.x - 0.09, y: wrist.y + scale * 0.92, z: -0.03 },
    ];
  } else if (g === 'Pinch' || g === 'OK Sign') {
    const tipX = g === 'Pinch' ? cx : indexChain[1].x - 0.01;
    const tipY = g === 'Pinch' ? cy : indexChain[1].y - 0.02;
    indexChain[2] = { x: tipX + 0.01, y: tipY + 0.01, z: -0.02 };
    indexChain[3] = { x: tipX, y: tipY, z: -0.03 };
    thumbChain = [
      { x: wrist.x - 0.05, y: wrist.y - scale * 0.2, z: 0 },
      { x: wrist.x - 0.07, y: wrist.y - scale * 0.45, z: -0.01 },
      { x: (wrist.x - 0.07 + tipX) / 2, y: (wrist.y - scale * 0.45 + tipY) / 2, z: -0.02 },
      { x: tipX + 0.006, y: tipY + 0.005, z: -0.03 },
    ];
  } else if (ext.thumb) {
    thumbChain = [
      { x: wrist.x - 0.05, y: wrist.y - scale * 0.18, z: 0 },
      { x: wrist.x - 0.09, y: wrist.y - scale * 0.36, z: -0.01 },
      { x: wrist.x - 0.13, y: wrist.y - scale * 0.55, z: -0.02 },
      { x: wrist.x - 0.16, y: wrist.y - scale * 0.72, z: -0.03 },
    ];
  } else {
    thumbChain = [
      { x: wrist.x - 0.04, y: wrist.y - scale * 0.18, z: 0 },
      { x: wrist.x - 0.05, y: wrist.y - scale * 0.36, z: -0.02 },
      { x: wrist.x - 0.02, y: wrist.y - scale * 0.44, z: -0.03 },
      { x: wrist.x + 0.02, y: wrist.y - scale * 0.46, z: -0.04 },
    ];
  }

  const landmarks: Point3D[] = [
    wrist,
    ...thumbChain,
    ...indexChain,
    ...middleChain,
    ...ringChain,
    ...pinkyChain,
  ];

  const withVis = landmarks.map((p) => ({ ...p, visibility: 1 }));

  return {
    landmarks: [withVis],
    worldLandmarks: [withVis],
    handednesses: [[{ score: 0.98, index: 0, categoryName: 'Right', displayName: 'Right' }]],
    handedness: [[{ score: 0.98, index: 0, categoryName: 'Right', displayName: 'Right' }]],
  };
}

/**
 * Generates realistic synthetic 478-point face landmark subset + ARKit blendshape categories
 * corresponding to the selected expression pattern.
 */
export function generateSyntheticFaceResult(
  expression: ExpressionLabel,
  tSec: number
): FaceLandmarkerResult {
  const lm: Point3D[] = Array.from({ length: 478 }, () => ({ x: 0.25, y: 0.32, z: 0 }));

  const fx = 0.22 + Math.sin(tSec * 0.7) * 0.008;
  const fy = 0.32 + Math.cos(tSec * 0.5) * 0.006;

  // Periodic natural blink every ~4s
  const blinkPhase = tSec % 4.2;
  const eyeOpen = blinkPhase > 4.0 ? 0.003 : 0.016;
  const mouthOpen = expression === 'Surprised' ? 0.032 : expression === 'Fearful' ? 0.02 : 0.005;

  // Populate Face Oval (key contour landmarks)
  const ovalIndices = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152,
    148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
  ];
  ovalIndices.forEach((idx, k) => {
    const a = (k / ovalIndices.length) * Math.PI * 2 - Math.PI / 2;
    lm[idx] = { x: fx + Math.cos(a) * 0.09, y: fy + Math.sin(a) * 0.15, z: 0 };
  });

  // Left eye EAR points: [33, 160, 158, 133, 153, 144]
  lm[33] = { x: fx - 0.052, y: fy - 0.035, z: -0.02 };
  lm[133] = { x: fx - 0.018, y: fy - 0.035, z: -0.02 };
  lm[160] = { x: fx - 0.041, y: fy - 0.035 - eyeOpen, z: -0.02 };
  lm[158] = { x: fx - 0.029, y: fy - 0.035 - eyeOpen, z: -0.02 };
  lm[144] = { x: fx - 0.041, y: fy - 0.035 + eyeOpen, z: -0.02 };
  lm[153] = { x: fx - 0.029, y: fy - 0.035 + eyeOpen, z: -0.02 };

  // Right eye EAR points: [362, 385, 387, 263, 373, 380]
  lm[362] = { x: fx + 0.018, y: fy - 0.035, z: -0.02 };
  lm[263] = { x: fx + 0.052, y: fy - 0.035, z: -0.02 };
  lm[385] = { x: fx + 0.029, y: fy - 0.035 - eyeOpen, z: -0.02 };
  lm[387] = { x: fx + 0.041, y: fy - 0.035 - eyeOpen, z: -0.02 };
  lm[380] = { x: fx + 0.029, y: fy - 0.035 + eyeOpen, z: -0.02 };
  lm[373] = { x: fx + 0.041, y: fy - 0.035 + eyeOpen, z: -0.02 };

  // Eyebrows
  const browLift = expression === 'Surprised' || expression === 'Fearful' ? -0.014 : expression === 'Angry' ? 0.01 : 0;
  [70, 63, 105, 66, 107].forEach((idx, k) => {
    lm[idx] = { x: fx - 0.055 + k * 0.01, y: fy - 0.065 + browLift, z: -0.02 };
  });
  [336, 296, 334, 293, 300].forEach((idx, k) => {
    lm[idx] = { x: fx + 0.015 + k * 0.01, y: fy - 0.065 + browLift, z: -0.02 };
  });

  // Mouth [61, 13, 291, 14]
  const smileCurve = expression === 'Happy' ? -0.012 : expression === 'Sad' ? 0.01 : 0;
  lm[61] = { x: fx - 0.035, y: fy + 0.058 + smileCurve, z: -0.02 };
  lm[291] = { x: fx + 0.035, y: fy + 0.058 + smileCurve, z: -0.02 };
  lm[13] = { x: fx, y: fy + 0.056 - mouthOpen * 0.4, z: -0.03 };
  lm[14] = { x: fx, y: fy + 0.062 + mouthOpen, z: -0.03 };

  const bsMap: Record<string, number> = {
    mouthSmileLeft: expression === 'Happy' ? 0.85 : 0.04,
    mouthSmileRight: expression === 'Happy' ? 0.84 : 0.04,
    cheekSquintLeft: expression === 'Happy' ? 0.55 : 0.02,
    cheekSquintRight: expression === 'Happy' ? 0.54 : 0.02,
    eyeWideLeft: expression === 'Surprised' ? 0.82 : expression === 'Fearful' ? 0.72 : 0.05,
    eyeWideRight: expression === 'Surprised' ? 0.82 : expression === 'Fearful' ? 0.72 : 0.05,
    jawOpen: expression === 'Surprised' ? 0.78 : expression === 'Fearful' ? 0.42 : 0.05,
    browOuterUpLeft: expression === 'Surprised' ? 0.74 : 0.05,
    browOuterUpRight: expression === 'Surprised' ? 0.74 : 0.05,
    browDownLeft: expression === 'Angry' ? 0.82 : 0.04,
    browDownRight: expression === 'Angry' ? 0.82 : 0.04,
    mouthPressLeft: expression === 'Angry' ? 0.58 : 0.03,
    mouthPressRight: expression === 'Angry' ? 0.58 : 0.03,
    mouthFrownLeft: expression === 'Sad' ? 0.76 : 0.03,
    mouthFrownRight: expression === 'Sad' ? 0.76 : 0.03,
    browInnerUp: expression === 'Sad' ? 0.68 : expression === 'Fearful' ? 0.75 : 0.04,
    mouthStretchLeft: expression === 'Fearful' ? 0.72 : 0.03,
    mouthStretchRight: expression === 'Fearful' ? 0.72 : 0.03,
    noseSneerLeft: expression === 'Disgusted' ? 0.84 : 0.02,
    noseSneerRight: expression === 'Disgusted' ? 0.82 : 0.02,
    mouthUpperUpLeft: expression === 'Disgusted' ? 0.65 : 0.02,
    mouthUpperUpRight: expression === 'Disgusted' ? 0.65 : 0.02,
  };

  const categories = Object.entries(bsMap).map(([categoryName, score], index) => ({
    categoryName,
    displayName: categoryName,
    score,
    index,
  }));

  return {
    faceLandmarks: [lm.map((p) => ({ ...p, visibility: 1 }))],
    faceBlendshapes: [{ categories, headIndex: 0, headName: 'blendshapes' }],
    facialTransformationMatrixes: [
      {
        rows: 4,
        columns: 4,
        data: [
          1, 0, Math.sin(tSec * 0.5) * 0.08, 0,
          0, 1, Math.cos(tSec * 0.6) * 0.05, 0,
          -Math.sin(tSec * 0.5) * 0.08, -Math.cos(tSec * 0.6) * 0.05, 1, 0,
          0, 0, 0, 1,
        ],
      },
    ],
  };
}
