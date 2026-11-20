import { GestureSample, ModelEvaluationReport } from '../types/vision';

export const CORE_GESTURE_LABELS = [
  'Open Palm',
  'Closed Fist',
  'Pointing Index',
  'Pinch',
  'Peace Sign',
  'Thumbs Up',
  'Thumbs Down',
  'OK Sign',
  'Three Fingers',
  'Four Fingers',
] as const;

/**
 * Generates clean kinematic prototype feature vectors (28-D) for the 10 core gestures
 * so the KNN and MLP neural network models can be trained, evaluated, and tested out-of-the-box,
 * and augmented with live webcam samples recorded by the user.
 */
function buildPrototypeVector(label: string): number[] {
  // Base vector structure (28 dims):
  // [0..4]   wrist-to-tip ratios (thumb, index, middle, ring, pinky)
  // [5..9]   joint angles (0..1)
  // [10..13] thumb-to-other-tip pinch dists
  // [14..16] adjacent fingertip spreads
  // [17..21] vertical tip offsets (0.5 = level, >0.65 = up, <0.35 = down)
  // [22..26] horizontal tip offsets
  // [27]     palm tilt
  const base = new Array(28).fill(0.5);

  const setFingers = (ext: [boolean, boolean, boolean, boolean, boolean]) => {
    for (let i = 0; i < 5; i++) {
      base[i] = ext[i] ? 0.68 : 0.30;       // tip ratio
      base[5 + i] = ext[i] ? 0.92 : 0.42;   // joint angle
      base[17 + i] = ext[i] ? 0.80 : 0.38;  // vertical offset
    }
  };

  switch (label) {
    case 'Open Palm':
      setFingers([true, true, true, true, true]);
      base[10] = 0.58; base[11] = 0.72; base[12] = 0.80; base[13] = 0.86;
      base[14] = 0.34; base[15] = 0.32; base[16] = 0.32;
      base[27] = 0.85;
      break;
    case 'Closed Fist':
      setFingers([false, false, false, false, false]);
      base[10] = 0.22; base[11] = 0.24; base[12] = 0.28; base[13] = 0.32;
      base[14] = 0.14; base[15] = 0.14; base[16] = 0.14;
      base[27] = 0.80;
      break;
    case 'Pointing Index':
      setFingers([false, true, false, false, false]);
      base[10] = 0.55; base[11] = 0.28; base[12] = 0.28; base[13] = 0.32;
      base[14] = 0.55; base[15] = 0.14; base[16] = 0.14;
      base[27] = 0.84;
      break;
    case 'Pinch':
      setFingers([false, false, false, false, false]);
      base[1] = 0.50; base[6] = 0.68;
      base[10] = 0.06; // thumb-index pinch near zero
      base[11] = 0.28; base[12] = 0.30; base[13] = 0.34;
      base[27] = 0.80;
      break;
    case 'Peace Sign':
      setFingers([false, true, true, false, false]);
      base[10] = 0.52; base[11] = 0.58; base[12] = 0.26; base[13] = 0.30;
      base[14] = 0.36; base[15] = 0.52; base[16] = 0.14;
      base[27] = 0.85;
      break;
    case 'Thumbs Up':
      setFingers([true, false, false, false, false]);
      base[17] = 0.90; // thumb pointing strongly upward
      base[10] = 0.52; base[11] = 0.50; base[12] = 0.52; base[13] = 0.56;
      base[27] = 0.55;
      break;
    case 'Thumbs Down':
      setFingers([true, false, false, false, false]);
      base[17] = 0.10; // thumb pointing strongly downward
      base[10] = 0.52; base[11] = 0.50; base[12] = 0.52; base[13] = 0.56;
      base[27] = 0.45;
      break;
    case 'OK Sign':
      setFingers([false, false, true, true, true]);
      base[1] = 0.48; base[6] = 0.64;
      base[10] = 0.06; // thumb-index touching
      base[11] = 0.62; base[12] = 0.72; base[13] = 0.78;
      base[15] = 0.30; base[16] = 0.30;
      base[27] = 0.85;
      break;
    case 'Three Fingers':
      setFingers([false, true, true, true, false]);
      base[10] = 0.52; base[11] = 0.62; base[12] = 0.66; base[13] = 0.28;
      base[14] = 0.30; base[15] = 0.30; base[16] = 0.52;
      base[27] = 0.85;
      break;
    case 'Four Fingers':
      setFingers([false, true, true, true, true]);
      base[10] = 0.48; base[11] = 0.58; base[12] = 0.65; base[13] = 0.72;
      base[14] = 0.28; base[15] = 0.28; base[16] = 0.28;
      base[27] = 0.85;
      break;
  }
  return base;
}

export function generateSeededDataset(samplesPerClass = 16): GestureSample[] {
  const dataset: GestureSample[] = [];
  let seed = 42;
  const rand = () => {
    seed = (seed * 1664525 + 1013904223) % 4294967296;
    return seed / 4294967296;
  };

  for (const label of CORE_GESTURE_LABELS) {
    const proto = buildPrototypeVector(label);
    for (let i = 0; i < samplesPerClass; i++) {
      const noisy = proto.map((v) =>
        Math.max(0, Math.min(1, v + (rand() - 0.5) * 0.09))
      );
      dataset.push({
        id: `seed-${label}-${i}`,
        label,
        features: noisy,
        source: 'seeded',
        createdAt: Date.now(),
      });
    }
  }
  return dataset;
}

export class KNNClassifier {
  private samples: GestureSample[] = [];
  private k: number;

  constructor(k = 5) {
    this.k = k;
  }

  fit(samples: GestureSample[]) {
    this.samples = samples;
  }

  predict(features: number[]): { gesture: string; confidence: number } {
    if (this.samples.length === 0) {
      return { gesture: 'Unknown', confidence: 0 };
    }

    const distances: Array<{ label: string; dist: number }> = [];
    for (const s of this.samples) {
      let sumSq = 0;
      for (let i = 0; i < features.length; i++) {
        const d = features[i] - (s.features[i] ?? 0);
        sumSq += d * d;
      }
      distances.push({ label: s.label, dist: Math.sqrt(sumSq) });
    }

    distances.sort((a, b) => a.dist - b.dist);
    const topK = distances.slice(0, Math.min(this.k, distances.length));

    const weights: Record<string, number> = {};
    let totalWeight = 0;
    for (const item of topK) {
      const w = 1 / (item.dist + 1e-3);
      weights[item.label] = (weights[item.label] ?? 0) + w;
      totalWeight += w;
    }

    let bestLabel = 'Unknown';
    let bestWeight = -1;
    for (const [label, w] of Object.entries(weights)) {
      if (w > bestWeight) {
        bestWeight = w;
        bestLabel = label;
      }
    }

    const minDist = topK[0]?.dist ?? 1;
    const distPenalty = Math.exp(-minDist * 0.7);
    const voteShare = totalWeight > 0 ? bestWeight / totalWeight : 0;

    return {
      gesture: bestLabel,
      confidence: Number(Math.min(0.99, voteShare * distPenalty).toFixed(3)),
    };
  }
}

/**
 * Pure TypeScript 3-Layer Multi-Layer Perceptron (28 -> 32 ReLU -> 24 ReLU -> K Softmax)
 * trained in-browser with SGD + Momentum and Cross-Entropy Loss.
 */
export class MLPGestureNetwork {
  private inputSize = 28;
  private h1Size = 32;
  private h2Size = 24;
  private labels: string[] = [];

  private w1: Float64Array = new Float64Array(0);
  private b1: Float64Array = new Float64Array(0);
  private w2: Float64Array = new Float64Array(0);
  private b2: Float64Array = new Float64Array(0);
  private w3: Float64Array = new Float64Array(0);
  private b3: Float64Array = new Float64Array(0);

  private isTrained = false;

  private initWeights(numClasses: number) {
    const randNorm = (scale: number) => ((Math.random() * 2 - 1) + (Math.random() * 2 - 1)) * scale;

    this.w1 = new Float64Array(this.inputSize * this.h1Size);
    this.b1 = new Float64Array(this.h1Size);
    const s1 = Math.sqrt(2 / this.inputSize);
    for (let i = 0; i < this.w1.length; i++) this.w1[i] = randNorm(s1 * 0.6);

    this.w2 = new Float64Array(this.h1Size * this.h2Size);
    this.b2 = new Float64Array(this.h2Size);
    const s2 = Math.sqrt(2 / this.h1Size);
    for (let i = 0; i < this.w2.length; i++) this.w2[i] = randNorm(s2 * 0.6);

    this.w3 = new Float64Array(this.h2Size * numClasses);
    this.b3 = new Float64Array(numClasses);
    const s3 = Math.sqrt(2 / this.h2Size);
    for (let i = 0; i < this.w3.length; i++) this.w3[i] = randNorm(s3 * 0.6);
  }

  private forward(x: number[]) {
    const numClasses = this.labels.length;

    const z1 = new Float64Array(this.h1Size);
    const a1 = new Float64Array(this.h1Size);
    for (let j = 0; j < this.h1Size; j++) {
      let sum = this.b1[j];
      for (let i = 0; i < this.inputSize; i++) {
        sum += (x[i] ?? 0) * this.w1[i * this.h1Size + j];
      }
      z1[j] = sum;
      a1[j] = sum > 0 ? sum : 0;
    }

    const z2 = new Float64Array(this.h2Size);
    const a2 = new Float64Array(this.h2Size);
    for (let j = 0; j < this.h2Size; j++) {
      let sum = this.b2[j];
      for (let i = 0; i < this.h1Size; i++) {
        sum += a1[i] * this.w2[i * this.h2Size + j];
      }
      z2[j] = sum;
      a2[j] = sum > 0 ? sum : 0;
    }

    const logits = new Float64Array(numClasses);
    let maxLogit = -Infinity;
    for (let k = 0; k < numClasses; k++) {
      let sum = this.b3[k];
      for (let i = 0; i < this.h2Size; i++) {
        sum += a2[i] * this.w3[i * numClasses + k];
      }
      logits[k] = sum;
      if (sum > maxLogit) maxLogit = sum;
    }

    const probs = new Float64Array(numClasses);
    let expSum = 0;
    for (let k = 0; k < numClasses; k++) {
      const e = Math.exp(logits[k] - maxLogit);
      probs[k] = e;
      expSum += e;
    }
    for (let k = 0; k < numClasses; k++) {
      probs[k] /= expSum || 1;
    }

    return { z1, a1, z2, a2, probs };
  }

  train(
    trainSamples: GestureSample[],
    epochs = 65,
    lr = 0.045
  ): { lossHistory: number[]; accuracyHistory: number[] } {
    this.labels = Array.from(new Set(trainSamples.map((s) => s.label))).sort();
    const numClasses = this.labels.length;
    if (numClasses < 2 || trainSamples.length === 0) {
      this.isTrained = false;
      return { lossHistory: [], accuracyHistory: [] };
    }

    const labelToIdx = new Map<string, number>();
    this.labels.forEach((l, idx) => labelToIdx.set(l, idx));

    this.initWeights(numClasses);

    const vw1 = new Float64Array(this.w1.length);
    const vb1 = new Float64Array(this.b1.length);
    const vw2 = new Float64Array(this.w2.length);
    const vb2 = new Float64Array(this.b2.length);
    const vw3 = new Float64Array(this.w3.length);
    const vb3 = new Float64Array(this.b3.length);
    const momentum = 0.85;

    const lossHistory: number[] = [];
    const accuracyHistory: number[] = [];

    for (let epoch = 0; epoch < epochs; epoch++) {
      // Shuffle indices
      const indices = Array.from({ length: trainSamples.length }, (_, i) => i);
      for (let i = indices.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [indices[i], indices[j]] = [indices[j], indices[i]];
      }

      let epochLoss = 0;
      let correct = 0;

      for (const idx of indices) {
        const sample = trainSamples[idx];
        const targetIdx = labelToIdx.get(sample.label)!;
        const { z1, a1, z2, a2, probs } = this.forward(sample.features);

        epochLoss += -Math.log(Math.max(1e-7, probs[targetIdx]));
        let predIdx = 0;
        for (let k = 1; k < numClasses; k++) {
          if (probs[k] > probs[predIdx]) predIdx = k;
        }
        if (predIdx === targetIdx) correct++;

        // Backpropagation
        const dLogits = new Float64Array(numClasses);
        for (let k = 0; k < numClasses; k++) {
          dLogits[k] = probs[k] - (k === targetIdx ? 1 : 0);
        }

        const dA2 = new Float64Array(this.h2Size);
        for (let i = 0; i < this.h2Size; i++) {
          let err = 0;
          for (let k = 0; k < numClasses; k++) {
            const wIdx = i * numClasses + k;
            err += dLogits[k] * this.w3[wIdx];
            const grad = a2[i] * dLogits[k];
            vw3[wIdx] = momentum * vw3[wIdx] - lr * grad;
            this.w3[wIdx] += vw3[wIdx];
          }
          dA2[i] = z2[i] > 0 ? err : 0;
        }
        for (let k = 0; k < numClasses; k++) {
          vb3[k] = momentum * vb3[k] - lr * dLogits[k];
          this.b3[k] += vb3[k];
        }

        const dA1 = new Float64Array(this.h1Size);
        for (let i = 0; i < this.h1Size; i++) {
          let err = 0;
          for (let j = 0; j < this.h2Size; j++) {
            const wIdx = i * this.h2Size + j;
            err += dA2[j] * this.w2[wIdx];
            const grad = a1[i] * dA2[j];
            vw2[wIdx] = momentum * vw2[wIdx] - lr * grad;
            this.w2[wIdx] += vw2[wIdx];
          }
          dA1[i] = z1[i] > 0 ? err : 0;
        }
        for (let j = 0; j < this.h2Size; j++) {
          vb2[j] = momentum * vb2[j] - lr * dA2[j];
          this.b2[j] += vb2[j];
        }

        for (let i = 0; i < this.inputSize; i++) {
          const xi = sample.features[i] ?? 0;
          for (let j = 0; j < this.h1Size; j++) {
            const wIdx = i * this.h1Size + j;
            const grad = xi * dA1[j];
            vw1[wIdx] = momentum * vw1[wIdx] - lr * grad;
            this.w1[wIdx] += vw1[wIdx];
          }
        }
        for (let j = 0; j < this.h1Size; j++) {
          vb1[j] = momentum * vb1[j] - lr * dA1[j];
          this.b1[j] += vb1[j];
        }
      }

      lossHistory.push(Number((epochLoss / trainSamples.length).toFixed(4)));
      accuracyHistory.push(Number((correct / trainSamples.length).toFixed(4)));
    }

    this.isTrained = true;
    return { lossHistory, accuracyHistory };
  }

  ready(): boolean {
    return this.isTrained && this.labels.length > 0;
  }

  predict(features: number[]): { gesture: string; confidence: number } {
    if (!this.isTrained || this.labels.length === 0) {
      return { gesture: 'Unknown', confidence: 0 };
    }
    const { probs } = this.forward(features);
    let bestIdx = 0;
    for (let k = 1; k < probs.length; k++) {
      if (probs[k] > probs[bestIdx]) bestIdx = k;
    }
    return {
      gesture: this.labels[bestIdx],
      confidence: Number(probs[bestIdx].toFixed(3)),
    };
  }
}

export function evaluateClassifier(
  samples: GestureSample[],
  modelType: 'knn' | 'mlp'
): { report: ModelEvaluationReport; knn: KNNClassifier; mlp: MLPGestureNetwork } {
  const labels = Array.from(new Set(samples.map((s) => s.label))).sort();
  const labelToIdx = new Map<string, number>();
  labels.forEach((l, i) => labelToIdx.set(l, i));

  // Stratified 75/25 train/test split
  const train: GestureSample[] = [];
  const test: GestureSample[] = [];
  for (const label of labels) {
    const classSamples = samples.filter((s) => s.label === label);
    const splitIdx = Math.max(1, Math.floor(classSamples.length * 0.75));
    train.push(...classSamples.slice(0, splitIdx));
    test.push(...classSamples.slice(splitIdx));
  }
  const evalSet = test.length > 0 ? test : train;

  const knn = new KNNClassifier(5);
  knn.fit(train);

  const mlp = new MLPGestureNetwork();
  const { lossHistory, accuracyHistory } = mlp.train(train, 60, 0.04);

  // Refit full KNN for live usage after evaluation
  const kCount = labels.length;
  const cm = Array.from({ length: kCount }, () => new Array(kCount).fill(0));
  let correct = 0;

  for (const item of evalSet) {
    const pred = modelType === 'mlp' ? mlp.predict(item.features) : knn.predict(item.features);
    const actualIdx = labelToIdx.get(item.label) ?? 0;
    const predIdx = labelToIdx.get(pred.gesture) ?? 0;
    cm[actualIdx][predIdx] += 1;
    if (actualIdx === predIdx) correct++;
  }

  let precisionSum = 0;
  let recallSum = 0;
  let f1Sum = 0;
  for (let c = 0; c < kCount; c++) {
    const tp = cm[c][c];
    let fp = 0;
    let fn = 0;
    for (let r = 0; r < kCount; r++) {
      if (r !== c) {
        fp += cm[r][c];
        fn += cm[c][r];
      }
    }
    const prec = tp + fp > 0 ? tp / (tp + fp) : 0;
    const rec = tp + fn > 0 ? tp / (tp + fn) : 0;
    const f1 = prec + rec > 0 ? (2 * prec * rec) / (prec + rec) : 0;
    precisionSum += prec;
    recallSum += rec;
    f1Sum += f1;
  }

  // Fit models on full dataset for live inference
  knn.fit(samples);
  if (modelType === 'mlp') {
    mlp.train(samples, 60, 0.04);
  }

  return {
    report: {
      modelType,
      accuracy: evalSet.length > 0 ? Number((correct / evalSet.length).toFixed(4)) : 0,
      macroPrecision: kCount > 0 ? Number((precisionSum / kCount).toFixed(4)) : 0,
      macroRecall: kCount > 0 ? Number((recallSum / kCount).toFixed(4)) : 0,
      macroF1: kCount > 0 ? Number((f1Sum / kCount).toFixed(4)) : 0,
      labels,
      confusionMatrix: cm,
      lossHistory,
      accuracyHistory,
      trainCount: train.length,
      testCount: evalSet.length,
      trainedAt: Date.now(),
    },
    knn,
    mlp,
  };
}
