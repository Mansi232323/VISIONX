import {
  GeometricShapeLabel,
  Point2D,
  SavedCustomShape,
  ShapeRecognitionResult,
} from '../types/vision';

const STORAGE_KEY = 'visionx_custom_shapes_v1';

function pathLength(pts: Point2D[]): number {
  let d = 0;
  for (let i = 1; i < pts.length; i++) {
    d += Math.hypot(pts[i].x - pts[i - 1].x, pts[i].y - pts[i - 1].y);
  }
  return d;
}

export function resamplePoints(pts: Point2D[], n = 64): Point2D[] {
  if (pts.length === 0) return [];
  if (pts.length === 1) return Array.from({ length: n }, () => ({ ...pts[0] }));

  const totalLen = pathLength(pts);
  if (totalLen < 1e-5) return Array.from({ length: n }, () => ({ ...pts[0] }));

  const interval = totalLen / (n - 1);
  let accum = 0;
  const src = pts.map((p) => ({ ...p }));
  const resampled: Point2D[] = [{ ...src[0] }];

  for (let i = 1; i < src.length; i++) {
    let d = Math.hypot(src[i].x - src[i - 1].x, src[i].y - src[i - 1].y);
    while (accum + d >= interval && resampled.length < n) {
      const t = (interval - accum) / d;
      const nx = src[i - 1].x + t * (src[i].x - src[i - 1].x);
      const ny = src[i - 1].y + t * (src[i].y - src[i - 1].y);
      const newPt = { x: nx, y: ny };
      resampled.push(newPt);
      src.splice(i, 0, newPt);
      accum = 0;
      d = Math.hypot(src[i + 1].x - newPt.x, src[i + 1].y - newPt.y);
      i++;
    }
    accum += d;
  }

  while (resampled.length < n) {
    resampled.push({ ...src[src.length - 1] });
  }
  return resampled.slice(0, n);
}

export function normalizePointCloud(pts: Point2D[], n = 64): Point2D[] {
  const resampled = resamplePoints(pts, n);
  if (resampled.length === 0) return [];

  let cx = 0;
  let cy = 0;
  for (const p of resampled) {
    cx += p.x;
    cy += p.y;
  }
  cx /= resampled.length;
  cy /= resampled.length;

  let maxSpan = 1e-5;
  const centered = resampled.map((p) => ({ x: p.x - cx, y: p.y - cy }));
  for (const p of centered) {
    maxSpan = Math.max(maxSpan, Math.abs(p.x), Math.abs(p.y));
  }

  return centered.map((p) => ({
    x: p.x / maxSpan,
    y: p.y / maxSpan,
  }));
}

function perpendicularDistance(pt: Point2D, lineStart: Point2D, lineEnd: Point2D): number {
  const dx = lineEnd.x - lineStart.x;
  const dy = lineEnd.y - lineStart.y;
  const mag = Math.hypot(dx, dy);
  if (mag < 1e-6) return Math.hypot(pt.x - lineStart.x, pt.y - lineStart.y);
  return Math.abs(dx * (lineStart.y - pt.y) - (lineStart.x - pt.x) * dy) / mag;
}

function douglasPeucker(pts: Point2D[], epsilon: number): Point2D[] {
  if (pts.length < 3) return pts;
  let maxDist = 0;
  let index = 0;
  const end = pts.length - 1;
  for (let i = 1; i < end; i++) {
    const d = perpendicularDistance(pts[i], pts[0], pts[end]);
    if (d > maxDist) {
      maxDist = d;
      index = i;
    }
  }
  if (maxDist > epsilon) {
    const left = douglasPeucker(pts.slice(0, index + 1), epsilon);
    const right = douglasPeucker(pts.slice(index), epsilon);
    return [...left.slice(0, -1), ...right];
  }
  return [pts[0], pts[end]];
}

function polygonArea(pts: Point2D[]): number {
  let area = 0;
  for (let i = 0; i < pts.length; i++) {
    const j = (i + 1) % pts.length;
    area += pts[i].x * pts[j].y - pts[j].x * pts[i].y;
  }
  return Math.abs(area) / 2;
}

function convexHull(points: Point2D[]): Point2D[] {
  if (points.length < 3) return points;
  const sorted = [...points].sort((a, b) => (a.x === b.x ? a.y - b.y : a.x - b.x));
  const cross = (o: Point2D, a: Point2D, b: Point2D) =>
    (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x);

  const lower: Point2D[] = [];
  for (const p of sorted) {
    while (lower.length >= 2 && cross(lower[lower.length - 2], lower[lower.length - 1], p) <= 0) {
      lower.pop();
    }
    lower.push(p);
  }
  const upper: Point2D[] = [];
  for (let i = sorted.length - 1; i >= 0; i--) {
    const p = sorted[i];
    while (upper.length >= 2 && cross(upper[upper.length - 2], upper[upper.length - 1], p) <= 0) {
      upper.pop();
    }
    upper.push(p);
  }
  lower.pop();
  upper.pop();
  return [...lower, ...upper];
}

function cloudDistance(a: Point2D[], b: Point2D[]): number {
  if (a.length !== b.length || a.length === 0) return Infinity;
  // Forward and reverse direction matching for stroke-direction invariance
  let forward = 0;
  let reverse = 0;
  const n = a.length;
  for (let i = 0; i < n; i++) {
    forward += Math.hypot(a[i].x - b[i].x, a[i].y - b[i].y);
    reverse += Math.hypot(a[i].x - b[n - 1 - i].x, a[i].y - b[n - 1 - i].y);
  }
  return Math.min(forward, reverse) / n;
}

function countRadialPeaks(pts: Point2D[]): number {
  let cx = 0;
  let cy = 0;
  for (const p of pts) {
    cx += p.x;
    cy += p.y;
  }
  cx /= pts.length;
  cy /= pts.length;

  const radii = pts.map((p) => Math.hypot(p.x - cx, p.y - cy));
  const meanR = radii.reduce((s, r) => s + r, 0) / radii.length;
  let peaks = 0;
  for (let i = 2; i < radii.length - 2; i++) {
    if (
      radii[i] > meanR * 1.15 &&
      radii[i] > radii[i - 1] &&
      radii[i] >= radii[i + 1] &&
      radii[i] > radii[i - 2] &&
      radii[i] >= radii[i + 2]
    ) {
      peaks++;
      i += 3;
    }
  }
  return peaks;
}

export function recognizeAirShape(
  rawPoints: Point2D[],
  customLibrary: SavedCustomShape[] = []
): ShapeRecognitionResult | null {
  if (rawPoints.length < 8) return null;

  const normalizedPoints = normalizePointCloud(rawPoints, 64);
  const resampled = resamplePoints(rawPoints, 64);

  let minX = Infinity;
  let maxX = -Infinity;
  let minY = Infinity;
  let maxY = -Infinity;
  for (const p of resampled) {
    minX = Math.min(minX, p.x);
    maxX = Math.max(maxX, p.x);
    minY = Math.min(minY, p.y);
    maxY = Math.max(maxY, p.y);
  }
  const width = Math.max(1e-5, maxX - minX);
  const height = Math.max(1e-5, maxY - minY);
  const diag = Math.hypot(width, height);
  const aspectRatio = width / height;

  const endToEnd = Math.hypot(
    resampled[0].x - resampled[resampled.length - 1].x,
    resampled[0].y - resampled[resampled.length - 1].y
  );
  const isClosed = endToEnd < diag * 0.36;

  const perimeter = pathLength(resampled);
  const hull = convexHull(resampled);
  const hullArea = Math.max(1e-6, polygonArea(hull));
  const area = Math.max(1e-6, polygonArea(resampled));
  const solidity = Math.min(1, area / hullArea);
  const circularity = Math.min(1.25, (4 * Math.PI * area) / Math.max(1e-6, perimeter * perimeter));

  const simplified = douglasPeucker(resampled, diag * 0.055);
  let vertices = simplified.length;
  if (isClosed && vertices > 2) {
    const first = simplified[0];
    const last = simplified[simplified.length - 1];
    if (Math.hypot(first.x - last.x, first.y - last.y) < diag * 0.16) {
      vertices -= 1;
    }
  }

  let geometricShape: GeometricShapeLabel = 'Freeform';
  let confidence = 0.65;

  // 1. Line detection (straight open stroke)
  if (!isClosed && endToEnd > perimeter * 0.86) {
    geometricShape = 'Line';
    confidence = Math.min(0.97, endToEnd / Math.max(1e-5, perimeter));
  } else if (!isClosed && vertices >= 3 && vertices <= 5 && endToEnd > diag * 0.45) {
    // Open arrow heuristic: shaft + tip angle
    geometricShape = 'Arrow';
    confidence = 0.78;
  } else {
    const radialPeaks = countRadialPeaks(resampled);

    if (radialPeaks >= 4 && radialPeaks <= 6 && solidity < 0.72) {
      geometricShape = 'Star';
      confidence = 0.89;
    } else if (circularity > 0.76 && vertices >= 6) {
      if (aspectRatio >= 0.78 && aspectRatio <= 1.28) {
        geometricShape = 'Circle';
        confidence = Math.min(0.98, 0.72 + circularity * 0.25);
      } else {
        geometricShape = 'Ellipse';
        confidence = Math.min(0.95, 0.7 + circularity * 0.22);
      }
    } else if (vertices === 3) {
      geometricShape = 'Triangle';
      confidence = 0.91;
    } else if (vertices === 4) {
      if (aspectRatio >= 0.8 && aspectRatio <= 1.25) {
        geometricShape = 'Square';
        confidence = 0.9;
      } else {
        geometricShape = 'Rectangle';
        confidence = 0.89;
      }
    } else if (vertices === 5) {
      geometricShape = 'Pentagon';
      confidence = 0.86;
    } else if (vertices === 6) {
      geometricShape = 'Hexagon';
      confidence = 0.84;
    } else if (circularity > 0.64) {
      geometricShape = aspectRatio >= 0.78 && aspectRatio <= 1.28 ? 'Circle' : 'Ellipse';
      confidence = 0.79;
    } else {
      geometricShape = 'Freeform';
      confidence = 0.7;
    }
  }

  // Match against Custom Shape Library
  let customMatchName: string | null = null;
  let customMatchConfidence: number | null = null;
  let bestDist = Infinity;

  for (const saved of customLibrary) {
    const dist = cloudDistance(normalizedPoints, saved.normalizedPoints);
    const aspectDiff = Math.abs(aspectRatio - saved.aspectRatio);
    const combinedScore = dist + aspectDiff * 0.12;
    if (combinedScore < bestDist) {
      bestDist = combinedScore;
      customMatchName = saved.customName;
    }
  }

  // Threshold for reliable custom shape recall
  if (customMatchName && bestDist < 0.42) {
    customMatchConfidence = Math.max(0.55, Math.min(0.99, 1 - bestDist * 1.15));
  } else {
    customMatchName = null;
    customMatchConfidence = null;
  }

  return {
    geometricShape,
    confidence: Number(confidence.toFixed(3)),
    customMatchName,
    customMatchConfidence: customMatchConfidence ? Number(customMatchConfidence.toFixed(3)) : null,
    metrics: {
      vertices,
      circularity: Number(circularity.toFixed(3)),
      aspectRatio: Number(aspectRatio.toFixed(2)),
      solidity: Number(solidity.toFixed(3)),
      perimeter: Number(perimeter.toFixed(3)),
      area: Number(area.toFixed(4)),
    },
    normalizedPoints,
  };
}

export function loadCustomShapes(): SavedCustomShape[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

export function saveCustomShapes(shapes: SavedCustomShape[]): void {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(shapes));
  } catch {
    // ignore storage quota failures
  }
}
