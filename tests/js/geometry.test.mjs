import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import {
  iou, matchGreedy, agreement, sweepCounts, SWEEP_THRESHOLDS, sizeBucket, bucketCounts, removedByNms,
  classCoverage, median, filterByThreshold,
} from "../../app/static/demo/geometry.js";
import { tokenizePython, renderCode } from "../../app/static/demo/codepanel.js";

const close = (a, b) => assert.ok(Math.abs(a - b) < 1e-9, `${a} != ${b}`);
const det = (label, confidence, box) => ({ label, confidence, box });

test("iou matches hand-computed values", () => {
  close(iou([0, 0, 10, 10], [20, 20, 30, 30]), 0);
  close(iou([0, 0, 10, 10], [0, 0, 10, 10]), 1);
  close(iou([0, 0, 10, 10], [5, 0, 15, 10]), 1 / 3);
  close(iou([0, 0, 0, 0], [0, 0, 0, 0]), 0);
});

test("iou agrees with the Python implementation on the shared fixture", () => {
  const cases = JSON.parse(readFileSync(new URL("../fixtures/iou_cases.json", import.meta.url)));
  for (const c of cases) close(iou(c.a, c.b), c.iou);
});

test("greedy one-to-one matching and agreement counts", () => {
  const a = [det("person", 0.9, [0, 0, 10, 10]), det("person", 0.8, [0, 0, 10, 10]), det("car", 0.7, [50, 50, 90, 90]),
             det("person", 0.6, [200, 200, 210, 210])];
  const b = [det("person", 0.95, [0, 1, 10, 11]), det("car", 0.5, [52, 50, 92, 90]), det("dog", 0.4, [200, 200, 210, 210])];
  const m = matchGreedy(a, b);
  assert.equal(m.pairs.length, 2);
  assert.deepEqual(m.pairs.map((p) => [p.a, p.b]).sort(), [[0, 0], [2, 1]]);
  assert.deepEqual(m.onlyA.sort(), [1, 3]); // second person has no free partner, last person has a different label in B
  assert.deepEqual(m.onlyB, [2]);
  assert.deepEqual(agreement(a, b), { matched: 2, onlyA: 2, onlyB: 1 });
});

test("threshold sweep counts", () => {
  const dets = [0.05, 0.3, 0.3, 0.5, 0.95].map((c) => det("person", c, [0, 0, 5, 5]));
  const counts = sweepCounts(dets);
  assert.equal(SWEEP_THRESHOLDS.length, 19);
  assert.equal(counts[0], 5); // t = 0.05
  assert.equal(counts[SWEEP_THRESHOLDS.indexOf(0.3)], 4);
  assert.equal(counts[SWEEP_THRESHOLDS.indexOf(0.5)], 2);
  assert.equal(counts.at(-1), 1); // t = 0.95
  assert.equal(filterByThreshold(dets, 0.5).length, 2);
});

test("size buckets at the COCO boundaries", () => {
  const sq = (n) => [0, 0, n, n];
  assert.equal(sizeBucket(sq(31)), "small");
  assert.equal(sizeBucket(sq(32)), "medium");
  assert.equal(sizeBucket(sq(96)), "medium");
  assert.equal(sizeBucket(sq(97)), "large");
  assert.deepEqual(bucketCounts([det("a", 1, sq(31)), det("a", 1, sq(32)), det("a", 1, sq(97))]), { small: 1, medium: 1, large: 1 });
});

test("removedByNms finds boxes missing after NMS", () => {
  const kept = [det("car", 0.9, [0, 0, 10, 10])];
  const without = [det("car", 0.9, [0, 0, 10, 10]), det("car", 0.8, [1, 0, 11, 10])];
  assert.equal(removedByNms(kept, without).length, 1);
});

test("class coverage and median", () => {
  assert.deepEqual(classCoverage([det("person", 1, []), det("car", 1, []), det("dog", 1, [])]), { persons: 1, cars: 1, other: 1 });
  assert.equal(median([3, 1, 2]), 2);
  assert.equal(median([4, 1, 2, 3]), 2.5);
  assert.equal(median([]), null);
});

test("python tokenizer", () => {
  const t = tokenizePython('x = foo("a#b", 1.5)  # note');
  const types = t.map((x) => x.type + ":" + x.text);
  assert.ok(types.includes('string:"a#b"'));
  assert.ok(types.includes("number:1.5"));
  assert.ok(types.includes("comment:# note"));
  assert.deepEqual(tokenizePython("def f(self):").filter((x) => x.type === "keyword").map((x) => x.text), ["def"]);
  assert.deepEqual(tokenizePython("return None").filter((x) => x.type === "keyword").map((x) => x.text), ["return", "None"]);
  assert.equal(tokenizePython("''' doc '''")[0].type, "string");
});

test("renderCode puts numbered badges on annotated lines with real line numbers", () => {
  const ex = { id: "demo", start_line: 40, code: "a = 1\nb = cv2.cvtColor(x)\nc = 3" };
  const { html, notes } = renderCode(ex, { "demo:cvtColor": "RGB to BGR", "other:a": "ignored" });
  assert.deepEqual(notes, ["RGB to BGR"]);
  assert.match(html, /<span class="ln">41<\/span>/);
  assert.match(html, /class="badge"[^>]*>1</);
});
