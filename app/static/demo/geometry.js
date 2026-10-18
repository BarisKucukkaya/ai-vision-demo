// Pure geometry helpers (no DOM). Boxes are [x1, y1, x2, y2] in original pixels.

export function iou(a, b) {
  const ix = Math.max(0, Math.min(a[2], b[2]) - Math.max(a[0], b[0]));
  const iy = Math.max(0, Math.min(a[3], b[3]) - Math.max(a[1], b[1]));
  const inter = ix * iy;
  const union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter;
  return union > 0 ? inter / union : 0;
}

export function filterByThreshold(detections, threshold) {
  return detections.filter((d) => d.confidence >= threshold);
}

// Greedy one-to-one matching: walk A by descending score, take the best still-free B with the same
// label and IoU >= minIou. Returns index lists into the given arrays.
export function matchGreedy(a, b, minIou = 0.5) {
  const order = a.map((_, i) => i).sort((i, j) => a[j].confidence - a[i].confidence);
  const usedB = new Set();
  const pairs = [];
  const usedA = new Set();
  for (const i of order) {
    let best = -1;
    let bestIou = minIou;
    for (let j = 0; j < b.length; j++) {
      if (usedB.has(j) || b[j].label !== a[i].label) continue;
      const v = iou(a[i].box, b[j].box);
      if (v >= bestIou) {
        best = j;
        bestIou = v;
      }
    }
    if (best >= 0) {
      usedB.add(best);
      usedA.add(i);
      pairs.push({ a: i, b: best, iou: bestIou });
    }
  }
  return {
    pairs,
    onlyA: a.map((_, i) => i).filter((i) => !usedA.has(i)),
    onlyB: b.map((_, j) => j).filter((j) => !usedB.has(j)),
  };
}

export function agreement(a, b, minIou = 0.5) {
  const m = matchGreedy(a, b, minIou);
  return { matched: m.pairs.length, onlyA: m.onlyA.length, onlyB: m.onlyB.length };
}

export const SWEEP_THRESHOLDS = Array.from({ length: 19 }, (_, i) => Math.round((0.05 + i * 0.05) * 100) / 100);

// Number of boxes with score >= t for every threshold t.
export function sweepCounts(detections, thresholds = SWEEP_THRESHOLDS) {
  return thresholds.map((t) => detections.reduce((n, d) => n + (d.confidence >= t - 1e-9 ? 1 : 0), 0));
}

// COCO size convention by box area in original pixels: small < 32^2, medium 32^2..96^2 (inclusive), large > 96^2.
export function sizeBucket(box) {
  const area = (box[2] - box[0]) * (box[3] - box[1]);
  if (area < 32 * 32) return "small";
  if (area <= 96 * 96) return "medium";
  return "large";
}

export function bucketCounts(detections) {
  const counts = { small: 0, medium: 0, large: 0 };
  for (const d of detections) counts[sizeBucket(d.box)] += 1;
  return counts;
}

// Boxes in `without` (NMS off) that have no same-label counterpart in `kept` (IoU >= minIou): those NMS removed.
export function removedByNms(kept, without, minIou = 0.9) {
  const m = matchGreedy(without, kept, minIou);
  return m.onlyA.map((i) => without[i]);
}

export function classCoverage(detections) {
  let persons = 0;
  let cars = 0;
  let other = 0;
  for (const d of detections) {
    if (d.label === "person") persons += 1;
    else if (d.label === "car") cars += 1;
    else other += 1;
  }
  return { persons, cars, other };
}

export function median(values) {
  if (!values.length) return null;
  const s = [...values].sort((x, y) => x - y);
  const mid = s.length >> 1;
  return s.length % 2 ? s[mid] : (s[mid - 1] + s[mid]) / 2;
}
