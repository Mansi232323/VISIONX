import { FaceMetrics, Point2D, Stroke, TrackedHand } from '../types/vision';

const HAND_CONNECTIONS: Array<[number, number]> = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16],
  [13, 17], [17, 18], [18, 19], [19, 20],
  [0, 17],
];

const FACE_OVAL_INDICES = [
  10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152,
  148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109, 10,
];

const LEFT_EYE = [33, 160, 158, 133, 153, 144, 33];
const RIGHT_EYE = [362, 385, 387, 263, 373, 380, 362];
const LEFT_BROW = [70, 63, 105, 66, 107];
const RIGHT_BROW = [336, 296, 334, 293, 300];
const LIPS_OUTER = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185, 61];

export interface OverlayRenderOptions {
  width: number;
  height: number;
  mirror: boolean;
  showFaceMesh: boolean;
  showHandSkeleton: boolean;
  face: FaceMetrics;
  hands: TrackedHand[];
  strokes?: Stroke[];
  activeStroke?: Stroke | null;
  cursorPoint?: Point2D | null;
  isDrawing?: boolean;
}

function drawPathIndices(
  ctx: CanvasRenderingContext2D,
  pts: Array<{ x: number; y: number }>,
  indices: number[],
  w: number,
  h: number,
  mirror: boolean
) {
  ctx.beginPath();
  let started = false;
  for (const idx of indices) {
    const p = pts[idx];
    if (!p) continue;
    const x = (mirror ? 1 - p.x : p.x) * w;
    const y = p.y * h;
    if (!started) {
      ctx.moveTo(x, y);
      started = true;
    } else {
      ctx.lineTo(x, y);
    }
  }
  ctx.stroke();
}

export function renderVisionOverlay(
  ctx: CanvasRenderingContext2D,
  opts: OverlayRenderOptions
) {
  const { width, height, mirror, showFaceMesh, showHandSkeleton, face, hands } = opts;

  ctx.clearRect(0, 0, width, height);

  // 1. Air-drawing overlay strokes (if passed)
  if (opts.strokes || opts.activeStroke) {
    const allStrokes = [...(opts.strokes ?? []), ...(opts.activeStroke ? [opts.activeStroke] : [])];
    for (const s of allStrokes) {
      if (s.points.length < 2) continue;
      ctx.save();
      ctx.lineCap = 'round';
      ctx.lineJoin = 'round';
      ctx.lineWidth = s.width;
      ctx.strokeStyle = s.isEraser ? 'rgba(15, 23, 42, 0.85)' : s.color;
      ctx.beginPath();
      ctx.moveTo(s.points[0].x * width, s.points[0].y * height);
      for (let i = 1; i < s.points.length; i++) {
        const prev = s.points[i - 1];
        const curr = s.points[i];
        const midX = ((prev.x + curr.x) / 2) * width;
        const midY = ((prev.y + curr.y) / 2) * height;
        ctx.quadraticCurveTo(prev.x * width, prev.y * height, midX, midY);
      }
      ctx.stroke();
      ctx.restore();
    }
  }

  // 2. Face landmarks & expression bounding HUD
  if (showFaceMesh && face.detected && face.landmarks.length > 0) {
    const lm = face.landmarks;
    ctx.save();

    // Subtle landmark point cloud
    ctx.fillStyle = 'rgba(56, 189, 248, 0.35)';
    for (let i = 0; i < lm.length; i += 3) {
      const p = lm[i];
      const x = (mirror ? 1 - p.x : p.x) * width;
      const y = p.y * height;
      ctx.fillRect(x - 1, y - 1, 2, 2);
    }

    // Precision facial contours
    ctx.lineWidth = 1.4;
    ctx.strokeStyle = 'rgba(6, 182, 212, 0.72)';
    drawPathIndices(ctx, lm, FACE_OVAL_INDICES, width, height, mirror);

    ctx.strokeStyle = face.isBlinking ? 'rgba(245, 158, 11, 0.95)' : 'rgba(34, 211, 238, 0.9)';
    drawPathIndices(ctx, lm, LEFT_EYE, width, height, mirror);
    drawPathIndices(ctx, lm, RIGHT_EYE, width, height, mirror);

    ctx.strokeStyle = 'rgba(148, 163, 184, 0.85)';
    drawPathIndices(ctx, lm, LEFT_BROW, width, height, mirror);
    drawPathIndices(ctx, lm, RIGHT_BROW, width, height, mirror);

    ctx.strokeStyle = 'rgba(52, 211, 153, 0.85)';
    drawPathIndices(ctx, lm, LIPS_OUTER, width, height, mirror);

    // Compact expression label near forehead
    const forehead = lm[10];
    if (forehead) {
      const fx = (mirror ? 1 - forehead.x : forehead.x) * width;
      const fy = Math.max(28, forehead.y * height - 18);
      const labelText = `${face.expression.label} · ${(face.expression.confidence * 100).toFixed(0)}%`;
      ctx.font = '600 11px "JetBrains Mono", monospace';
      const tw = ctx.measureText(labelText).width + 16;
      ctx.fillStyle = 'rgba(9, 12, 18, 0.82)';
      ctx.strokeStyle = 'rgba(6, 182, 212, 0.55)';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.roundRect(fx - tw / 2, fy - 14, tw, 22, 4);
      ctx.fill();
      ctx.stroke();
      ctx.fillStyle = '#E2E8F0';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(labelText, fx, fy - 3);
    }

    ctx.restore();
  }

  // 3. Hand skeletons (21 landmarks per hand)
  if (showHandSkeleton && hands.length > 0) {
    for (const hand of hands) {
      const lm = hand.landmarks;
      ctx.save();

      // Connections
      ctx.lineWidth = 2;
      ctx.strokeStyle = hand.handedness === 'Left' ? 'rgba(245, 158, 11, 0.85)' : 'rgba(6, 182, 212, 0.9)';
      for (const [a, b] of HAND_CONNECTIONS) {
        const pa = lm[a];
        const pb = lm[b];
        if (!pa || !pb) continue;
        ctx.beginPath();
        ctx.moveTo((mirror ? 1 - pa.x : pa.x) * width, pa.y * height);
        ctx.lineTo((mirror ? 1 - pb.x : pb.x) * width, pb.y * height);
        ctx.stroke();
      }

      // Joints & Fingertip rings
      const tips = new Set([4, 8, 12, 16, 20]);
      for (let i = 0; i < lm.length; i++) {
        const p = lm[i];
        const x = (mirror ? 1 - p.x : p.x) * width;
        const y = p.y * height;
        ctx.beginPath();
        ctx.arc(x, y, tips.has(i) ? 4.5 : 2.8, 0, Math.PI * 2);
        ctx.fillStyle = i === 8 ? '#22D3EE' : tips.has(i) ? '#F8FAFC' : '#94A3B8';
        ctx.fill();
      }

      // Pinch tether visualization
      if (hand.isPinching) {
        const tx = (mirror ? 1 - hand.thumbTip.x : hand.thumbTip.x) * width;
        const ty = hand.thumbTip.y * height;
        const ix = (mirror ? 1 - hand.indexTip.x : hand.indexTip.x) * width;
        const iy = hand.indexTip.y * height;
        ctx.strokeStyle = '#10B981';
        ctx.lineWidth = 2.5;
        ctx.beginPath();
        ctx.moveTo(tx, ty);
        ctx.lineTo(ix, iy);
        ctx.stroke();
      }

      // Gesture label near wrist
      const wrist = lm[0];
      if (wrist) {
        const wx = (mirror ? 1 - wrist.x : wrist.x) * width;
        const wy = Math.min(height - 18, wrist.y * height + 24);
        const text = `${hand.handedness}: ${hand.gesture} · ${(hand.gestureConfidence * 100).toFixed(0)}%`;
        ctx.font = '600 11px "JetBrains Mono", monospace';
        const tw = ctx.measureText(text).width + 16;
        ctx.fillStyle = 'rgba(9, 12, 18, 0.85)';
        ctx.strokeStyle = hand.handedness === 'Left' ? 'rgba(245, 158, 11, 0.6)' : 'rgba(6, 182, 212, 0.6)';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.roundRect(wx - tw / 2, wy - 11, tw, 22, 4);
        ctx.fill();
        ctx.stroke();
        ctx.fillStyle = '#F8FAFC';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(text, wx, wy);
      }

      ctx.restore();
    }
  }

  // 4. Precision fingertip cursor crosshair
  if (opts.cursorPoint) {
    const cx = opts.cursorPoint.x * width;
    const cy = opts.cursorPoint.y * height;
    ctx.save();
    ctx.strokeStyle = opts.isDrawing ? '#10B981' : '#22D3EE';
    ctx.lineWidth = 1.6;
    ctx.beginPath();
    ctx.arc(cx, cy, opts.isDrawing ? 10 : 13, 0, Math.PI * 2);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(cx - 17, cy);
    ctx.lineTo(cx - 6, cy);
    ctx.moveTo(cx + 6, cy);
    ctx.lineTo(cx + 17, cy);
    ctx.moveTo(cx, cy - 17);
    ctx.lineTo(cx, cy - 6);
    ctx.moveTo(cx, cy + 6);
    ctx.lineTo(cx, cy + 17);
    ctx.stroke();
    ctx.restore();
  }
}
