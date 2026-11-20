import { Point2D, Point3D } from '../types/vision';

export class Point2DSmoother {
  private prev: Point2D | null = null;
  private alpha: number;

  constructor(alpha = 0.42) {
    this.alpha = alpha;
  }

  setAlpha(alpha: number) {
    this.alpha = Math.min(1, Math.max(0.05, alpha));
  }

  update(pt: Point2D): Point2D {
    if (!this.prev) {
      this.prev = { ...pt };
      return { ...pt };
    }
    const dx = pt.x - this.prev.x;
    const dy = pt.y - this.prev.y;
    const speed = Math.hypot(dx, dy);
    // Adaptive alpha: smoother when still, snappier during fast strokes
    const adaptiveAlpha = Math.min(0.88, Math.max(this.alpha, this.alpha + speed * 2.2));
    const next: Point2D = {
      x: this.prev.x + adaptiveAlpha * dx,
      y: this.prev.y + adaptiveAlpha * dy,
    };
    this.prev = next;
    return next;
  }

  reset() {
    this.prev = null;
  }
}

export class LandmarkArraySmoother {
  private prev: Point3D[] | null = null;
  private alpha: number;

  constructor(alpha = 0.55) {
    this.alpha = alpha;
  }

  update(landmarks: Point3D[]): Point3D[] {
    if (!this.prev || this.prev.length !== landmarks.length) {
      this.prev = landmarks.map((p) => ({ ...p }));
      return this.prev;
    }
    const next = landmarks.map((p, i) => {
      const prevPt = this.prev![i];
      return {
        x: prevPt.x + this.alpha * (p.x - prevPt.x),
        y: prevPt.y + this.alpha * (p.y - prevPt.y),
        z: prevPt.z + this.alpha * (p.z - prevPt.z),
      };
    });
    this.prev = next;
    return next;
  }

  reset() {
    this.prev = null;
  }
}

export class ScalarEMASmoother {
  private prev: number | null = null;
  private alpha: number;

  constructor(alpha = 0.25) {
    this.alpha = alpha;
  }

  update(val: number): number {
    if (this.prev === null) {
      this.prev = val;
      return val;
    }
    this.prev = this.prev + this.alpha * (val - this.prev);
    return this.prev;
  }

  reset() {
    this.prev = null;
  }
}
