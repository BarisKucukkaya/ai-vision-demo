// The six live-analysis cards. Everything is computed in the browser from the stored floor-0.05 results.
import {
  SWEEP_THRESHOLDS, agreement, bucketCounts, classCoverage, filterByThreshold, removedByNms, sweepCounts,
} from "./geometry.js";
import { createStage, el } from "./stage.js";

export const FW_NAMES = { pytorch: "PyTorch", tensorflow: "TensorFlow", opencv: "OpenCV" };
export const ALG_VAR = { hog_svm: "--alg-hog", faster_rcnn: "--alg-frcnn", retinanet: "--alg-retina" };
const DASH = { pytorch: "", tensorflow: "7 4", opencv: "2 3" };

const SHORT = { hog_svm: "HOG", faster_rcnn: "Faster R-CNN", retinanet: "RetinaNet" };
const FW_SHORT = { pytorch: "PT", tensorflow: "TF", opencv: "" };
export function shortLabel(entry) {
  return entry.alg === "hog_svm" ? SHORT.hog_svm : `${SHORT[entry.alg]} ${FW_SHORT[entry.fw]}`;
}

export function comboLabel(entry, algos) {
  const name = algos[entry.alg].name;
  return entry.alg === "hog_svm" ? name : `${name} · ${FW_NAMES[entry.fw]}`;
}

const SVGNS = "http://www.w3.org/2000/svg";
function svg(tag, attrs = {}, text) {
  const node = document.createElementNS(SVGNS, tag);
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
  if (text != null) node.textContent = text;
  return node;
}

function card(title, ...body) {
  return el("section", { class: "acard" }, el("h3", { text: title }), ...body);
}

const empty = (msg = "Press Detect to fill this card.") => el("p", { class: "cap", text: msg });

function atThreshold(entry, t) {
  return filterByThreshold(entry.result.detections, t);
}

// 1. Threshold sweep ------------------------------------------------------------------------------
function sweepCard(ctx) {
  const c = card("Threshold sweep");
  if (!ctx.entries.length) return c.append(empty()), c;
  const W = 400, H = 240, L = 34, R = 104, T = 10, B = 32;
  const series = ctx.entries.map((e) => ({ e, counts: sweepCounts(e.result.detections) }));
  const yMax = Math.max(1, ...series.flatMap((s) => s.counts));
  const x = (t) => L + ((t - 0.05) / 0.9) * (W - L - R);
  const y = (n) => T + (1 - n / yMax) * (H - T - B);
  const root = svg("svg", { viewBox: `0 0 ${W} ${H}`, tabindex: "0", role: "img",
    "aria-label": "Number of boxes at each threshold, one line per algorithm. Use left and right arrow keys to read values." });
  for (const v of [0, Math.round(yMax / 2), yMax]) {
    root.append(svg("line", { x1: L, x2: W - R, y1: y(v), y2: y(v), class: "axis" }), svg("text", { x: L - 5, y: y(v) + 4, "text-anchor": "end", class: "muted" }, v));
  }
  for (const t of [0.05, 0.25, 0.5, 0.75, 0.95]) root.append(svg("text", { x: x(t), y: H - 14, "text-anchor": t > 0.9 ? "end" : "middle", class: "muted" }, t.toFixed(2)));
  root.append(svg("text", { x: (L + W - R) / 2, y: H - 1, "text-anchor": "middle", class: "muted" }, "threshold"));
  const labels = [];
  for (const s of series) {
    const d = s.counts.map((n, i) => `${i ? "L" : "M"}${x(SWEEP_THRESHOLDS[i]).toFixed(1)},${y(n).toFixed(1)}`).join(" ");
    root.append(svg("path", { d, fill: "none", "stroke-width": 2.5, "stroke-dasharray": DASH[s.e.fw], style: `stroke:var(${ALG_VAR[s.e.alg]})` }));
    labels.push({ y: y(s.counts.at(-1)), text: shortLabel(s.e), alg: s.e.alg });
  }
  labels.sort((a, b) => a.y - b.y);
  for (let i = 1; i < labels.length; i++) labels[i].y = Math.max(labels[i].y, labels[i - 1].y + 16);
  for (const l of labels) root.append(svg("text", { x: W - R + 10, y: l.y + 4, style: `fill:var(${ALG_VAR[l.alg]});font-weight:700` }, l.text));
  const marker = svg("line", { y1: T, y2: H - B, "stroke-width": 1.5, "stroke-dasharray": "3 3", style: "stroke:var(--text)" });
  const cursor = svg("line", { y1: T, y2: H - B, "stroke-width": 1, style: "stroke:var(--muted)", visibility: "hidden" });
  root.append(marker, cursor);
  marker.setAttribute("x1", x(ctx.threshold)); marker.setAttribute("x2", x(ctx.threshold));
  const readout = el("div", { class: "readout num", "aria-live": "polite" });
  const show = (idx) => {
    const t = SWEEP_THRESHOLDS[idx];
    cursor.setAttribute("x1", x(t)); cursor.setAttribute("x2", x(t)); cursor.setAttribute("visibility", "visible");
    readout.textContent = `threshold ${t.toFixed(2)}: ` + series.map((s) => `${shortLabel(s.e)} ${s.counts[idx]}`).join(" · ");
  };
  const nearest = (clientX) => {
    const r = root.getBoundingClientRect();
    const t = 0.05 + (((clientX - r.left) / r.width) * W - L) / (W - L - R) * 0.9;
    return Math.min(18, Math.max(0, Math.round((t - 0.05) / 0.05)));
  };
  let idx = Math.round((ctx.threshold - 0.05) / 0.05);
  root.addEventListener("pointermove", (ev) => { idx = nearest(ev.clientX); show(idx); });
  root.addEventListener("focus", () => show(idx));
  root.addEventListener("keydown", (ev) => {
    if (ev.key === "ArrowLeft" || ev.key === "ArrowRight") {
      idx = Math.min(18, Math.max(0, idx + (ev.key === "ArrowRight" ? 1 : -1)));
      show(idx);
      ev.preventDefault();
    }
  });
  root.addEventListener("pointerleave", () => cursor.setAttribute("visibility", "hidden"));
  readout.textContent = "Hover or focus the chart to read values.";
  c.append(root, readout, el("p", { class: "cap", text: "This shows how each algorithm's output reacts to the threshold. It is not accuracy: scores of different algorithms are not on the same scale and we have no ground truth." }));
  return c;
}

// A labelled bar: name | track with bar | number. Never colour alone: the number is always printed.
function barLine(cls, label, n, max, extraStyle = "") {
  return el("div", { class: "bline" }, el("span", { class: "bl", text: label }),
    el("div", { class: "btrack" }, el("div", { class: "bar " + cls, style: `width:${Math.max(n ? 2 : 0, (n / max) * 100)}%;${extraStyle}` })),
    el("span", { class: "bv num", text: n }));
}

// 2. Class coverage -------------------------------------------------------------------------------
function coverageCard(ctx) {
  const c = card("Class coverage");
  if (!ctx.entries.length) return c.append(empty()), c;
  const rows = ctx.entries.map((e) => ({ e, cov: classCoverage(atThreshold(e, ctx.threshold)) }));
  const max = Math.max(1, ...rows.map((r) => Math.max(r.cov.persons, r.cov.cars, r.cov.other)));
  for (const { e, cov } of rows) {
    c.append(el("div", { class: "brow", style: `--alg:var(${ALG_VAR[e.alg]})` },
      el("div", { class: "bhead" }, el("strong", { text: comboLabel(e, ctx.algos) }), el("span", { class: "cap", text: `can detect ${e.result.model_info?.classes_supported ?? "?"} classes at all` })),
      barLine("p", "persons", cov.persons, max), barLine("c", "cars", cov.cars, max), barLine("o", "other", cov.other, max)));
  }
  c.append(el("p", { class: "cap", text: "Counts at the current threshold. The class count comes from the model, not from text." }));
  return c;
}

// 3. Size buckets ---------------------------------------------------------------------------------
function sizeCard(ctx) {
  const c = card("Size buckets");
  if (!ctx.entries.length) return c.append(empty()), c;
  const rows = ctx.entries.map((e) => ({ e, b: bucketCounts(atThreshold(e, ctx.threshold)) }));
  const max = Math.max(1, ...rows.flatMap((r) => Object.values(r.b)));
  for (const { e, b } of rows) {
    c.append(el("div", { class: "brow", style: `--alg:var(${ALG_VAR[e.alg]})` },
      el("div", { class: "bhead" }, el("strong", { text: comboLabel(e, ctx.algos) })),
      barLine("", "small", b.small, max, "opacity:0.45"), barLine("", "medium", b.medium, max, "opacity:0.7"), barLine("", "large", b.large, max)));
  }
  c.append(el("p", { class: "cap", text: "By box area in original pixels: small < 32², medium 32²–96², large > 96² (the COCO convention). Shows where an algorithm starts missing small objects." }));
  return c;
}

// 4. Agreement matrix -----------------------------------------------------------------------------
function agreementCard(ctx) {
  const c = card("Agreement matrix");
  c.classList.add("wide");
  if (ctx.entries.length < 2) return c.append(empty(ctx.entries.length ? "Needs at least two active results." : undefined)), c;
  const tbody = el("tbody");
  for (let i = 0; i < ctx.entries.length; i++) {
    for (let j = i + 1; j < ctx.entries.length; j++) {
      const A = ctx.entries[i], B = ctx.entries[j];
      const a = agreement(atThreshold(A, ctx.threshold), atThreshold(B, ctx.threshold));
      const same = A.alg === "retinanet" && B.alg === "retinanet";
      tbody.append(el("tr", { class: same ? "hl" : "" },
        el("td", { text: comboLabel(A, ctx.algos) + (same ? "  (same algorithm, two frameworks)" : "") }),
        el("td", { text: comboLabel(B, ctx.algos) }),
        el("td", { class: "r num", text: a.matched }), el("td", { class: "r num", text: a.onlyA }), el("td", { class: "r num", text: a.onlyB })));
    }
  }
  c.append(el("div", { class: "table-wrap" }, el("table", {}, el("thead", {}, el("tr", {}, el("th", { text: "A" }), el("th", { text: "B" }), el("th", { class: "r", text: "matched" }), el("th", { class: "r", text: "only A" }), el("th", { class: "r", text: "only B" }))), tbody)),
    el("p", { class: "cap", text: "Matched = same label and IoU ≥ 0.5, greedy one-to-one by score, at the current threshold." }));
  return c;
}

// 5. NMS effect -----------------------------------------------------------------------------------
function nmsCard(ctx) {
  const c = card("NMS effect");
  const { nms } = ctx;
  const button = el("button", { type: "button", class: "ghost", onclick: nms.onRun }, nms.state === "running" ? "Running…" : "Compare with NMS off");
  if (!ctx.entries.length || ctx.mode !== "live" || nms.state === "running") button.disabled = true;
  c.append(button);
  if (ctx.mode !== "live") c.append(el("p", { class: "cap", text: "Needs a live server: disabled in Recorded mode." }));
  else if (!ctx.entries.length) c.append(empty());
  if (nms.state === "error") c.append(el("p", { class: "error", text: nms.error }));
  const rows = ctx.entries.filter((e) => nms.results[e.combo]);
  if (rows.length) {
    const tbody = el("tbody");
    for (const e of rows) {
      const on = atThreshold(e, ctx.threshold);
      const off = atThreshold({ result: nms.results[e.combo] }, ctx.threshold);
      tbody.append(el("tr", {}, el("td", { text: comboLabel(e, ctx.algos) }), el("td", { class: "r num", text: on.length }), el("td", { class: "r num", text: off.length }), el("td", { class: "r num", text: removedByNms(on, off).length })));
    }
    c.append(el("div", { class: "table-wrap" }, el("table", {}, el("thead", {}, el("tr", {}, el("th", { text: "algorithm" }), el("th", { class: "r", text: `NMS ${ctx.nmsIou.toFixed(2)}` }), el("th", { class: "r", text: "NMS off" }), el("th", { class: "r", text: "removed" }))), tbody)));
    const sel = rows.find((e) => e.combo === nms.selected) || rows[0];
    const picker = el("select", { "aria-label": "Show removed boxes for", onchange: (ev) => nms.onSelect(ev.target.value) });
    for (const e of rows) picker.append(el("option", { value: e.combo, text: comboLabel(e, ctx.algos), ...(e.combo === sel.combo ? { selected: "" } : {}) }));
    const on = atThreshold(sel, ctx.threshold);
    const off = atThreshold({ result: nms.results[sel.combo] }, ctx.threshold);
    const removed = removedByNms(on, off);
    const stage = createStage(ctx.photo);
    stage.el.classList.add("mini");
    stage.setItems([
      ...on.map((d) => ({ ...d, text: `kept ${d.label} ${d.confidence.toFixed(2)}` })),
      ...removed.map((d) => ({ ...d, color: "#d9480f", dashed: true, text: `removed ${d.label} ${d.confidence.toFixed(2)}` })),
    ]);
    c.append(el("label", { class: "cap" }, "Removed boxes for ", picker), stage.el);
  }
  c.append(el("p", { class: "cap", text: "One extra request per active algorithm with NMS off. Counts are limited by each model's maximum detections per image." }));
  return c;
}

// 6. Cost -----------------------------------------------------------------------------------------
function costCard(ctx) {
  const c = card("Cost");
  c.classList.add("wide");
  if (!ctx.entries.length) return c.append(empty()), c;
  const tbody = el("tbody");
  const fmt = (n) => (n == null ? "—" : n >= 1e6 ? `${(n / 1e6).toFixed(2)} M` : n.toLocaleString());
  for (const e of ctx.entries) {
    const r = e.result, rep = ctx.repeat[e.combo], bench = ctx.benchmark?.results?.[e.combo];
    tbody.append(el("tr", {},
      el("td", { text: comboLabel(e, ctx.algos) }),
      el("td", { class: "r num", text: `${r.inference_ms.toFixed(0)} ms` }),
      el("td", { class: "r num", text: rep ? `${rep.median.toFixed(0)} ms (${rep.min.toFixed(0)}–${rep.max.toFixed(0)})` : bench ? `${bench.median} ms (p95 ${bench.p95})` : "—" }),
      el("td", { class: "r num", text: fmt(r.model_info?.parameters) }),
      el("td", { class: "r num", text: r.model_info?.model_size_mb != null ? `${r.model_info.model_size_mb} MB` : "—" })));
  }
  c.append(el("div", { class: "table-wrap" }, el("table", {}, el("thead", {}, el("tr", {}, ...["algorithm", "this request", "median (min–max)", "parameters", "weights (float32)"].map((t, i) => el("th", { class: i ? "r" : "", text: t })))), tbody)));
  const dev = [...new Set(ctx.entries.map((e) => e.result.device))].join(", ");
  c.append(el("p", { class: "cap", text: `One sample, not a benchmark. Device: ${dev}.${ctx.benchmark ? ` Benchmark data: ${ctx.benchmark.machine?.os ?? ""}.` : ""} Median appears after Repeat ×5.` }));
  return c;
}

export function renderAnalysis(container, ctx) {
  container.replaceChildren(sweepCard(ctx), coverageCard(ctx), sizeCard(ctx), nmsCard(ctx), agreementCard(ctx), costCard(ctx));
}
