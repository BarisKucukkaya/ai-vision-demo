// State, API calls and rendering of the controls, result views and side-by-side table.
import { filterByThreshold, matchGreedy, classCoverage, median } from "./geometry.js";
import { renderAnalysis, comboLabel, FW_NAMES, ALG_VAR } from "./analysis.js";
import { createStage, cssVar, el } from "./stage.js";
import { initCodePanel, copyText } from "./codepanel.js";

const $ = (s) => document.querySelector(s);
const ORDER = ["hog_svm", "faster_rcnn", "retinanet"];
const FLOOR = 0.05;
const SUBTITLE = { hog_svm: "classic machine learning", faster_rcnn: "deep, two-stage", retinanet: "deep, one-stage" };
const INPUT_EXPECTED = { opencv: "BGR uint8, HWC", pytorch: "RGB, CHW float 0–1", tensorflow: "RGB, HWC 0–255 (letterboxed by us)" };
const LIBRARY = { opencv: "OpenCV (cv2.HOGDescriptor)", pytorch: "PyTorch · torchvision", tensorflow: "TensorFlow · KerasHub" };
const GREY_REASON = "KerasHub has no Faster R-CNN";

const state = {
  framework: "pytorch",
  chips: { hog_svm: true, faster_rcnn: true, retinanet: true },
  photo: null, // {name, blob, url, width, height}
  samples: [],
  health: null,
  serverUp: null,
  mode: "live",
  modeAuto: false,
  results: {}, // combo -> {ok, ...prediction} at the floor threshold
  source: null, // live | recorded | example
  recordedMeta: null,
  threshold: 0.5,
  view: "cards",
  knobs: { nms_iou: 0.5, hog_stride: 8, hog_scale: 1.05, hit_threshold: 0 },
  stale: false,
  busy: false,
  pending: new Set(), // combos whose request is still running
  message: null,
  repeat: {},
  nms: { state: "idle", results: {}, selected: null, error: null },
  sel: {},
  openRaw: new Set(),
  pickA: null,
  pickB: null,
  overlayX: 50,
};
let algos = {};
let glossary = {};
let benchmark = null;
const recordedCache = new Map();
let codePanel = null;
let healthTimer = null;
let dataFiles = { files: [], recorded: [] }; // optional data files the server reports as present

// ------------------------------------------------------------------------------------------ utilities
async function getJSON(url) {
  try {
    const r = await fetch(url);
    return r.ok ? await r.json() : null;
  } catch {
    return null;
  }
}

function store(key, value) {
  try { localStorage.setItem(key, value); } catch { /* storage may be blocked */ }
}

const fixed = (n, d = 2) => Number(n).toFixed(d);

function entries() {
  const out = [];
  const fw = state.framework;
  for (const alg of ORDER) {
    if (!state.chips[alg]) continue;
    if (alg === "hog_svm") out.push({ combo: "hog_svm/opencv", alg, fw: "opencv" });
    if (alg === "faster_rcnn") {
      if (fw !== "tensorflow") out.push({ combo: "faster_rcnn/pytorch", alg, fw: "pytorch" });
      if (fw !== "pytorch") out.push({ combo: "faster_rcnn/tensorflow", alg, fw: "tensorflow", grey: true, reason: GREY_REASON });
    }
    if (alg === "retinanet") {
      if (fw !== "tensorflow") out.push({ combo: "retinanet/pytorch", alg, fw: "pytorch" });
      if (fw !== "pytorch") out.push({ combo: "retinanet/tensorflow", alg, fw: "tensorflow" });
    }
  }
  return out;
}

const realEntries = () => entries().filter((e) => !e.grey);
const activeResults = () => realEntries().filter((e) => state.results[e.combo]?.ok).map((e) => ({ ...e, result: state.results[e.combo] }));
const dets = (e) => filterByThreshold(e.result.detections, state.threshold);

function imageDims() {
  const first = Object.values(state.results).find((r) => r?.ok);
  if (first) return first.image;
  return state.photo ? { width: state.photo.width, height: state.photo.height } : { width: 4, height: 3 };
}

function stageOpts(result) {
  const dims = result?.image ?? imageDims();
  return { src: state.photo?.url ?? null, width: dims.width, height: dims.height };
}

function queryString(extra = {}) {
  const k = state.knobs;
  return new URLSearchParams({
    threshold: String(FLOOR), nms_iou: String(k.nms_iou), hog_stride: String(k.hog_stride),
    hog_scale: String(k.hog_scale), hit_threshold: String(k.hit_threshold), ...extra,
  });
}

// ------------------------------------------------------------------------------------------ health
function backendState(info) {
  if (!info) return "unavailable";
  if (info.error) return "unavailable";
  return info.loaded ? "ready" : "loading";
}

async function pollHealth() {
  clearTimeout(healthTimer);
  let loading = false;
  try {
    const r = await fetch("/health");
    state.health = await r.json();
    const wasDown = state.serverUp === false;
    state.serverUp = true;
    loading = Object.values(state.health.backends).some((b) => backendState(b) === "loading");
    if (wasDown && state.modeAuto) await setMode("live", true);
  } catch {
    state.serverUp = false;
    if (state.mode === "live") await setMode("recorded", true);
  }
  renderPills();
  renderChips();
  renderBanners();
  renderResults();
  healthTimer = setTimeout(pollHealth, loading ? 2000 : 15000);
}

function renderPills() {
  const ul = $("#pills");
  ul.replaceChildren();
  if (state.serverUp === false) {
    ul.append(el("li", { class: "pill" }, el("span", { class: "dot unavailable" }), "Server unreachable"));
    return;
  }
  for (const [combo, info] of Object.entries(state.health?.backends ?? {})) {
    const [alg, fw] = combo.split("/");
    const s = backendState(info);
    const name = alg === "hog_svm" ? "HOG + SVM" : `${algos[alg]?.name ?? alg} · ${FW_NAMES[fw]}`;
    const label = `${name}${info.device && s === "ready" ? ` · ${info.device}` : ""}`;
    ul.append(el("li", { class: "pill", title: s === "unavailable" ? info.error ?? "unavailable" : s }, el("span", { class: `dot ${s}` }), `${label} (${s})`));
  }
}

function renderChips() {
  const box = $("#chips");
  box.replaceChildren();
  for (const alg of ORDER) {
    const list = entries().filter((e) => e.alg === alg);
    let s = "ready";
    let reason = "";
    for (const e of list) {
      if (e.grey) continue;
      const st = state.serverUp === false ? "unavailable" : backendState(state.health?.backends?.[e.combo]);
      if (st === "unavailable") { s = "unavailable"; reason = state.health?.backends?.[e.combo]?.error ?? "not reachable"; }
      else if (st === "loading" && s !== "unavailable") s = "loading";
    }
    if (state.framework === "tensorflow" && alg === "faster_rcnn") { s = "unavailable"; reason = GREY_REASON; }
    const chip = el("button", { type: "button", class: "chip", "aria-pressed": String(state.chips[alg]), style: `--chip:var(${ALG_VAR[alg]})`, title: reason || s,
      onclick: () => { state.chips[alg] = !state.chips[alg]; renderAll(); } },
      el("span", { class: "t" }, el("span", { class: `dot ${s}` }), algos[alg]?.name ?? alg),
      el("span", { class: "s", text: algos[alg]?.family ?? SUBTITLE[alg] }));
    box.append(chip);
  }
}

// ------------------------------------------------------------------------------------------ banners
function renderBanners() {
  const box = $("#banners");
  box.replaceChildren();
  const add = (cls, text) => box.append(el("div", { class: `banner ${cls}`, role: cls === "warn" ? "alert" : "status", text }));
  if (state.message) add("warn", state.message);
  if (state.mode === "recorded") {
    if (state.source === "example") add("rec", "MOCK DATA: committed example recording (fake detections from the mock adapters, no real inference).");
    else if (state.source === "recorded") add("rec", `RECORDED: results saved ${state.recordedMeta?.recorded_at ?? ""} on ${state.recordedMeta?.machine ?? "another machine"}. Not live.`);
    else add("rec", "RECORDED mode: no recorded data found for this photo.");
    if (state.serverUp === false) add("rec", "Server not reachable: showing recorded data. The code panel uses its snapshot.");
  }
  const shown = Object.values(state.results).filter((r) => r?.ok);
  if (shown.some((r) => r.mock)) add("warn", "Test model: these results come from mock or random-weight models. Boxes are meaningless.");
}

// ------------------------------------------------------------------------------------------ controls
function setRadio(container, attr, value) {
  container.querySelectorAll("[role=radio]").forEach((b) => b.setAttribute("aria-checked", String(b.dataset[attr] === value)));
}

function renderControls() {
  setRadio($("#fw-switch"), "fw", state.framework);
  setRadio($("#mode-switch"), "mode", state.mode);
  setRadio($("#view-switch"), "view", state.view);
  $("#threshold-out").textContent = fixed(state.threshold);
  $("#stale").hidden = !state.stale;
  $("#btn-detect").disabled = state.busy;
  $("#btn-repeat").disabled = state.busy || state.mode !== "live";
  $("#btn-detect").textContent = state.busy ? "Working…" : state.mode === "recorded" ? "Load recorded" : "Detect";
  const nmsDisabled = state.mode === "live" && activeResults().length > 0 && activeResults().every((e) => e.result.settings?.nms_applied === false);
  $("#k-nms").disabled = nmsDisabled;
  $("#k-nms").closest(".knob").classList.toggle("disabled", nmsDisabled);
  $("#k-nms").title = nmsDisabled ? "The active backends could not apply this setting." : "";
}

function renderThumbs() {
  $("#photo-name").textContent = state.photo && !state.photo.virtual ? `Using: ${state.photo.name}` : "";
  const box = $("#thumbs");
  box.replaceChildren();
  for (const name of state.samples) {
    box.append(el("button", { type: "button", class: "thumb", title: name, "aria-label": `Photo ${name}`, "aria-pressed": String(state.photo?.name === name), onclick: () => selectSample(name) },
      el("img", { src: `/samples/${encodeURIComponent(name)}`, alt: "" })));
  }
}

async function measure(url) {
  return new Promise((resolve) => {
    const img = new Image();
    img.onload = () => resolve({ width: img.naturalWidth, height: img.naturalHeight });
    img.onerror = () => resolve({ width: 4, height: 3 });
    img.src = url;
  });
}

async function setPhoto(photo) {
  state.photo = photo;
  state.results = {}; state.repeat = {}; state.sel = {};
  state.nms = { state: "idle", results: {}, selected: null, error: null };
  state.stale = false; state.message = null; state.source = null; state.recordedMeta = null;
  if (state.mode === "recorded") await loadRecorded();
  renderAll();
}

async function selectSample(name, auto = false) {
  try {
    const r = await fetch(`/samples/${encodeURIComponent(name)}`);
    if (!r.ok) throw new Error("not found");
    const blob = await r.blob();
    const url = URL.createObjectURL(blob);
    const dims = await measure(url);
    if (auto && state.photo && !state.photo.virtual) return; // the user picked a photo while this one was loading
    await setPhoto({ name, blob, url, ...dims });
  } catch {
    state.message = `Could not load ${name} from the server.`;
    renderAll();
  }
}

async function useFile(file) {
  if (!file || !file.type.startsWith("image/")) {
    state.message = "That file is not an image.";
    return renderAll();
  }
  // The photo is only used in this browser session: nothing is stored on the server.
  const name = file.name && file.name !== "image.png" ? file.name : `pasted-${new Date().toISOString().replace(/[-:T]/g, "").slice(0, 14)}.png`;
  const url = URL.createObjectURL(file);
  const dims = await measure(url);
  await setPhoto({ name, blob: file, url, ...dims });
}

// ------------------------------------------------------------------------------------------ detect
async function postCompare(extra, only = null) {
  const combos = (only ?? realEntries().map((e) => e.combo)).join(",");
  const fd = new FormData();
  fd.append("file", state.photo.blob, state.photo.name);
  const r = await fetch(`/compare?${queryString({ combos, ...extra })}`, { method: "POST", body: fd });
  if (!r.ok) {
    let detail = r.statusText;
    try { detail = (await r.json()).detail ?? detail; } catch { /* keep status text */ }
    throw Object.assign(new Error(detail), { status: r.status });
  }
  return r.json();
}

async function detect() {
  if (state.busy) return;
  if (state.mode === "recorded") {
    await loadRecorded();
    return renderAll();
  }
  if (!state.photo || state.photo.virtual) {
    state.message = "Choose a photo first (thumbnail, file picker, drop or paste).";
    return renderAll();
  }
  if (!realEntries().length) {
    state.message = "Switch on at least one algorithm.";
    return renderAll();
  }
  state.busy = true; state.message = null;
  const combos = realEntries().map((e) => e.combo);
  state.results = {}; state.repeat = {}; state.sel = {}; state.source = "live"; state.stale = false;
  state.nms = { state: "idle", results: {}, selected: null, error: null };
  state.pending = new Set(combos);
  renderAll();
  // One request per algorithm, in sequence: each card fills as soon as its result arrives, and the
  // timings stay honest because nothing runs concurrently.
  for (const combo of combos) {
    try {
      const data = await postCompare({}, [combo]);
      state.results[combo] = data.results[combo];
    } catch (err) {
      if (err.status === undefined) {
        state.pending.clear();
        state.message = "Cannot reach the server. Switching to recorded data.";
        await setMode("recorded", true);
        break;
      }
      state.results[combo] = { ok: false, error: `The server refused the request: ${err.message}` };
    }
    state.pending.delete(combo);
    renderResults();
  }
  state.pending.clear();
  state.busy = false;
  renderAll();
}

async function repeat5() {
  if (state.busy || state.mode !== "live" || !state.photo) return;
  state.busy = true; state.message = null;
  const times = {};
  const combos = realEntries().map((e) => e.combo);
  try {
    for (let i = 1; i <= 5; i++) {
      for (const combo of combos) {
        $("#btn-repeat").textContent = `Run ${i}/5 · ${combo.split("/")[0].replace("_", " ")}…`;
        const data = await postCompare({}, [combo]);
        const r = data.results[combo];
        if (r.ok) (times[combo] ??= []).push(r.inference_ms);
        state.results[combo] = r;
      }
      renderResults();
    }
    state.source = "live";
    state.repeat = Object.fromEntries(Object.entries(times).map(([c, t]) => [c, { median: median(t), min: Math.min(...t), max: Math.max(...t), n: t.length }]));
    state.stale = false;
  } catch (err) {
    state.message = `Repeat stopped: ${err.message}`;
  }
  $("#btn-repeat").textContent = "Repeat ×5";
  state.busy = false;
  renderAll();
}

async function runNmsOff() {
  if (state.mode !== "live" || !state.photo) return;
  state.nms = { ...state.nms, state: "running", error: null };
  renderAnalysisStrip();
  try {
    const data = await postCompare({ nms_iou: "1.0" });
    state.nms = { state: "done", results: Object.fromEntries(Object.entries(data.results).filter(([, r]) => r.ok)), selected: state.nms.selected, error: null };
  } catch (err) {
    state.nms = { ...state.nms, state: "error", error: `NMS-off request failed: ${err.message}` };
  }
  renderAnalysisStrip();
}

// ------------------------------------------------------------------------------------------ recorded
async function fetchRecorded(stem) {
  if (recordedCache.has(stem)) return recordedCache.get(stem);
  const data = dataFiles.recorded.includes(stem) ? await getJSON(`/static/demo/data/recorded/${encodeURIComponent(stem)}.json`) : null;
  recordedCache.set(stem, data);
  return data;
}

async function loadRecorded() {
  const own = state.photo && !state.photo.virtual ? state.photo : null; // a blank stand-in photo is not the user's
  const stem = own ? own.name.replace(/\.[^.]+$/, "") : null;
  let rec = stem ? await fetchRecorded(stem) : null;
  if (!rec) rec = await fetchRecorded("_example");
  if (!rec) {
    state.results = {}; state.source = null;
    return;
  }
  state.results = Object.fromEntries(Object.entries(rec.results).map(([combo, r]) => {
    const [algorithm, framework] = combo.split("/");
    return [combo, { ok: true, algorithm, framework, via: "recorded", image: rec.image, ...r }];
  }));
  state.source = rec.image.name === "_example" ? "example" : "recorded";
  state.recordedMeta = { recorded_at: rec.recorded_at, machine: rec.machine };
  state.repeat = {};
  state.nms = { state: "idle", results: {}, selected: null, error: null };
  state.stale = false;
  // Without a photo of their own, show the boxes on a blank canvas of the recorded size.
  if (!own) state.photo = { name: rec.image.name, blob: null, url: null, width: rec.image.width, height: rec.image.height, virtual: true };
}

async function setMode(mode, auto = false) {
  if (mode === state.mode && !auto) return;
  state.mode = mode; state.modeAuto = auto;
  state.results = {}; state.source = null; state.repeat = {}; state.sel = {};
  state.nms = { state: "idle", results: {}, selected: null, error: null };
  if (mode === "recorded") await loadRecorded();
  if (!auto) renderAll();
}

// ------------------------------------------------------------------------------------------ result views
const fwBadge = (fw) => el("span", { class: `fwbadge ${fw}`, text: FW_NAMES[fw] });

function curlFor(e) {
  const k = state.knobs;
  const q = `algorithm=${e.alg}&framework=${e.fw}&threshold=${FLOOR}&nms_iou=${k.nms_iou}` + (e.alg === "hog_svm" ? `&hog_stride=${k.hog_stride}&hog_scale=${k.hog_scale}&hit_threshold=${k.hit_threshold}` : "");
  return `curl -F "file=@${state.photo?.name ?? "photo.jpg"}" "${location.origin}/predict?${q}"`;
}

function cardFor(e) {
  const algo = algos[e.alg];
  const style = `--alg:var(${ALG_VAR[e.alg]})`;
  if (e.grey) {
    return el("article", { class: "card grey", style }, el("div", { class: "card-head" }, el("h3", { text: algo.name }), fwBadge("tensorflow")),
      el("p", { class: "sub", text: algo.family }), el("p", { class: "placeholder" }, el("strong", { text: "Not available on this side:" }), ` ${e.reason}.`, el("br"), "Frameworks ship different sets of algorithms."));
  }
  const r = state.results[e.combo];
  const head = el("div", { class: "card-head" }, el("h3", { text: algo.name }), fwBadge(e.fw));
  const sub = el("p", { class: "sub", text: e.alg === "hog_svm" ? `${algo.family} · OpenCV · same in every framework` : algo.family });
  if (!r && state.pending.has(e.combo)) {
    return el("article", { class: "card", style, "aria-busy": "true" }, head, sub, el("p", { class: "placeholder" }, el("span", { class: "spinner" }), " Running…"));
  }
  if (!r) return el("article", { class: "card", style }, head, sub, el("p", { class: "placeholder", text: state.mode === "live" ? "Press Detect." : "No recorded data for this algorithm." }));
  if (!r.ok) return el("article", { class: "card", style }, head, sub, el("p", { class: "error", role: "alert", text: r.error }));
  const list = dets({ result: r });
  const stage = createStage(stageOpts(r));
  const draw = () => stage.setItems(list.map((d, i) => ({ ...d, highlight: state.sel[e.combo] === i })));
  draw();
  const rep = state.repeat[e.combo];
  const ms = el("div", { class: "bigms num" }, `${r.inference_ms.toFixed(0)} `, el("small", { text: "ms" }),
    rep ? el("small", { text: `  median ${rep.median.toFixed(0)} (${rep.min.toFixed(0)}–${rep.max.toFixed(0)}), client-side timing, one machine` }) : null);
  const tbody = el("tbody");
  list.forEach((d, i) => {
    const row = el("tr", { tabindex: "0", "aria-label": `${d.label} ${d.confidence}`, class: state.sel[e.combo] === i ? "sel" : "" },
      el("td", { text: d.label }), el("td", { class: "r num", text: fixed(d.confidence) }), el("td", { class: "num", text: d.box.map((v) => Math.round(v)).join(", ") }));
    const pick = () => { state.sel[e.combo] = state.sel[e.combo] === i ? null : i; tbody.querySelectorAll("tr").forEach((tr, n) => tr.classList.toggle("sel", state.sel[e.combo] === n)); draw(); };
    row.addEventListener("click", pick);
    row.addEventListener("keydown", (ev) => { if (ev.key === "Enter" || ev.key === " ") { ev.preventDefault(); pick(); } });
    tbody.append(row);
  });
  const json = JSON.stringify(r, null, 1);
  const curl = curlFor(e);
  const raw = el("details", { class: "rawjson" }, el("summary", { text: "Raw JSON and curl" }),
    el("div", {}, el("button", { type: "button", class: "ghost", onclick: (ev) => flash(ev.target, copyText(json)) }, "Copy JSON"), " ",
      el("button", { type: "button", class: "ghost", onclick: (ev) => flash(ev.target, copyText(curl)) }, "Copy curl")),
    el("code", { class: "curl", text: curl }), el("pre", { text: json }));
  if (state.openRaw.has(e.combo)) raw.open = true;
  raw.addEventListener("toggle", () => (raw.open ? state.openRaw.add(e.combo) : state.openRaw.delete(e.combo)));
  const notes = [];
  if (r.settings?.nms_applied === false) notes.push(el("p", { class: "cap hint", text: "NMS IoU could not be applied for this backend." }));
  return el("article", { class: "card", style }, head, sub, ms, el("div", { class: "modelname", text: `${r.model} · ${r.device}${r.via && r.via !== "local" ? ` · via ${r.via}` : ""}` }),
    stage.el, ...notes, el("div", { class: "scrollbox" }, el("table", {}, el("thead", {}, el("tr", {}, el("th", { text: "label" }), el("th", { class: "r", text: "confidence" }), el("th", { text: "box x1,y1,x2,y2" }))), tbody)),
    el("p", { class: "hint", text: `${list.length} boxes at threshold ${fixed(state.threshold)} · ${r.score_meaning}` }), raw);
}

async function flash(btn, promise) {
  await promise;
  const old = btn.textContent;
  btn.textContent = "Copied";
  setTimeout(() => (btn.textContent = old), 1200);
}

function pairPicker(list) {
  if (list.length < 2) return null;
  if (!list.some((e) => e.combo === state.pickA)) state.pickA = list[0].combo;
  if (!list.some((e) => e.combo === state.pickB) || state.pickB === state.pickA) state.pickB = list.find((e) => e.combo !== state.pickA).combo;
  const select = (id, value, label) => {
    const s = el("select", { id, onchange: (ev) => { state[id === "pickA" ? "pickA" : "pickB"] = ev.target.value; renderResults(); } });
    for (const e of list) s.append(el("option", { value: e.combo, text: comboLabel(e, algos), ...(e.combo === value ? { selected: "" } : {}) }));
    return el("label", {}, label + " ", s);
  };
  return el("div", { class: "pickrow" }, select("pickA", state.pickA, "A (left)"), select("pickB", state.pickB, "B (right)"));
}

function overlayView(list) {
  const picker = pairPicker(list);
  if (!picker) return el("p", { class: "placeholder", text: "The overlay slider needs two results. Press Detect with at least two algorithms active (or choose Both frameworks)." });
  const A = list.find((e) => e.combo === state.pickA), B = list.find((e) => e.combo === state.pickB);
  const sa = createStage(stageOpts(A.result)), sb = createStage(stageOpts(B.result));
  sa.setItems(dets(A)); sb.setItems(dets(B));
  const top = el("div", { class: "top" }, sb.el);
  const handle = el("div", { class: "handle", tabindex: "0", role: "slider", "aria-label": "Divider between A (left) and B (right)", "aria-valuemin": "0", "aria-valuemax": "100" });
  const set = (v) => {
    state.overlayX = Math.min(100, Math.max(0, v));
    top.style.clipPath = `inset(0 0 0 ${state.overlayX}%)`;
    handle.style.left = `${state.overlayX}%`;
    handle.setAttribute("aria-valuenow", String(Math.round(state.overlayX)));
  };
  const wrap = el("div", { class: "compare" }, sa.el, top, handle, el("span", { class: "tagA", text: `A: ${comboLabel(A, algos)}` }), el("span", { class: "tagB", text: `B: ${comboLabel(B, algos)}` }));
  const move = (ev) => { const r = wrap.getBoundingClientRect(); set(((ev.clientX - r.left) / r.width) * 100); };
  const stop = () => { wrap.onpointermove = null; };
  wrap.addEventListener("pointerdown", (ev) => {
    ev.preventDefault(); // no text/image selection while dragging
    getSelection()?.removeAllRanges();
    wrap.setPointerCapture(ev.pointerId);
    move(ev);
    wrap.onpointermove = move;
    handle.focus({ preventScroll: true });
  });
  wrap.addEventListener("pointerup", stop);
  wrap.addEventListener("pointercancel", stop);
  wrap.addEventListener("lostpointercapture", stop);
  wrap.addEventListener("dragstart", (ev) => ev.preventDefault());
  wrap.addEventListener("selectstart", (ev) => ev.preventDefault());
  handle.addEventListener("keydown", (ev) => {
    const step = ev.shiftKey ? 10 : 2;
    if (ev.key === "ArrowLeft") { set(state.overlayX - step); ev.preventDefault(); }
    if (ev.key === "ArrowRight") { set(state.overlayX + step); ev.preventDefault(); }
    if (ev.key === "Home") { set(0); ev.preventDefault(); }
    if (ev.key === "End") { set(100); ev.preventDefault(); }
  });
  set(state.overlayX);
  return el("div", {}, picker, wrap, el("p", { class: "hint", text: "Drag the divider or focus it and use the arrow keys (Shift for bigger steps)." }));
}

function diffView(list) {
  const picker = pairPicker(list);
  if (!picker) return el("p", { class: "placeholder", text: "The diff needs two results. Press Detect with at least two algorithms active (or choose Both frameworks)." });
  const A = list.find((e) => e.combo === state.pickA), B = list.find((e) => e.combo === state.pickB);
  const a = dets(A), b = dets(B);
  const m = matchGreedy(a, b, 0.5);
  const ca = cssVar("--diff-a") || "#2b64ad", cb = cssVar("--diff-b") || "#a85a00";
  const stage = createStage(stageOpts(A.result));
  stage.setItems([
    ...m.pairs.map((p) => ({ ...a[p.a], color: "#6b7280", text: `= ${a[p.a].label} ${fixed(a[p.a].confidence)}` })),
    ...m.onlyA.map((i) => ({ ...a[i], color: ca, text: `A ${a[i].label} ${fixed(a[i].confidence)}` })),
    ...m.onlyB.map((j) => ({ ...b[j], color: cb, dashed: true, text: `B ${b[j].label} ${fixed(b[j].confidence)}` })),
  ]);
  const legend = el("div", { class: "legend num" },
    el("span", {}, el("i", { class: "swatch", style: "border-color:#6b7280" }), `agreement (same label, IoU ≥ 0.5): ${m.pairs.length}`),
    el("span", {}, el("i", { class: "swatch", style: `border-color:${ca}` }), `only in A (${comboLabel(A, algos)}): ${m.onlyA.length}`),
    el("span", {}, el("i", { class: "swatch", style: `border-color:${cb};border-style:dashed` }), `only in B (${comboLabel(B, algos)}): ${m.onlyB.length}`));
  return el("div", {}, picker, stage.el, legend);
}

function renderResults() {
  const area = $("#stage-area");
  const list = activeResults();
  let view;
  if (state.view === "overlay") view = overlayView(list);
  else if (state.view === "diff") view = diffView(list);
  else view = el("div", { class: "cards" }, ...entries().map(cardFor));
  area.replaceChildren(view);
  renderControls();
  renderAnalysisStrip();
  renderTable();
  codePanel?.refreshTabs();
}

function renderAnalysisStrip() {
  renderAnalysis($("#analysis"), {
    entries: activeResults(), threshold: state.threshold, algos, mode: state.mode, nmsIou: state.knobs.nms_iou,
    photo: stageOpts(Object.values(state.results).find((r) => r?.ok)),
    repeat: state.repeat, benchmark,
    nms: { ...state.nms, onRun: runNmsOff, onSelect: (c) => { state.nms.selected = c; renderAnalysisStrip(); } },
  });
}

// ------------------------------------------------------------------------------------------ side-by-side table
function renderTable() {
  const cols = entries();
  const wrap = $("#table-wrap");
  const withR = (e) => (e.grey ? null : state.results[e.combo]?.ok ? { ...e, result: state.results[e.combo] } : null);
  const cell = (e, fn) => {
    if (e.grey) return el("td", { class: "grey", text: "—" });
    const x = withR(e);
    return x ? el("td", { class: "num", text: fn(x) }) : el("td", { class: "grey", text: "—" });
  };
  const text = (fn) => (e) => el("td", { text: fn(e) });
  const rows = [
    ["Family", text((e) => algos[e.alg].family)],
    ["Features", text((e) => algos[e.alg].features)],
    ["How it searches", text((e) => algos[e.alg].how_it_searches)],
    ["Classes", (e) => cell(e, (x) => {
      const cov = classCoverage(dets(x));
      return `can detect ${x.result.model_info?.classes_supported ?? "?"} classes · persons ${cov.persons}, cars ${cov.cars}`;
    })],
    ["Speed", (e) => cell(e, (x) => `${x.result.inference_ms.toFixed(0)} ms` + (state.repeat[x.combo] ? ` · median ${state.repeat[x.combo].median.toFixed(0)} ms` : ""))],
    ["Accuracy", (e) => (e.grey ? el("td", { class: "grey", text: "—" }) : el("td", { text: "not measured here" }))],
    ["Detections ≥ threshold", (e) => cell(e, (x) => String(dets(x).length))],
    ["Framework / library", text((e) => LIBRARY[e.fw])],
    ["Parameters", (e) => cell(e, (x) => (x.result.model_info?.parameters != null ? x.result.model_info.parameters.toLocaleString() : "—"))],
    ["Weights size (float32)", (e) => cell(e, (x) => (x.result.model_info?.model_size_mb != null ? `${x.result.model_info.model_size_mb} MB` : "—"))],
    ["Score meaning", (e) => cell(e, (x) => x.result.score_meaning)],
    ["Input the model expects", text((e) => INPUT_EXPECTED[e.fw])],
    ["Settings used", (e) => cell(e, (x) => {
      const s = x.result.settings ?? {};
      return `NMS IoU ${s.nms_iou ?? "?"}${s.nms_applied === false ? " (not applied)" : ""}` + (e.alg === "hog_svm" ? `, stride ${s.hog_stride}, scale ${s.hog_scale}, hit ${s.hit_threshold}` : "");
    })],
  ];
  const thead = el("thead", {}, el("tr", {}, el("th", { text: "" }), ...cols.map((e) => el("th", { class: "alg", style: `--alg:var(${ALG_VAR[e.alg]})` }, algos[e.alg].name, " ", fwBadge(e.fw)))));
  const tbody = el("tbody", {}, ...rows.map(([name, fn]) => el("tr", {}, el("th", { scope: "row", text: name }), ...cols.map(fn))));
  wrap.replaceChildren(el("table", {}, thead, tbody));
}

// ------------------------------------------------------------------------------------------ misc UI
function renderTerms() {
  const box = $("#terms");
  box.replaceChildren();
  for (const [term, text] of Object.entries(glossary)) {
    box.append(el("span", { class: "term", tabindex: "0" }, term, el("span", { class: "tip", role: "tooltip", text })));
  }
}

function renderAll() {
  renderControls();
  renderThumbs();
  renderChips();
  renderBanners();
  renderResults();
}

function activeTabs() {
  const set = new Set();
  const r = activeResults();
  if (r.some((e) => e.alg === "hog_svm")) set.add("hog");
  if (r.some((e) => e.alg === "faster_rcnn")) set.add("frcnn");
  if (r.some((e) => e.alg === "retinanet")) set.add("retina");
  if (r.some((e) => e.combo === "retinanet/pytorch") && r.some((e) => e.combo === "retinanet/tensorflow")) set.add("framework");
  return set;
}

function toggleTheme() {
  const dark = document.documentElement.dataset.theme === "dark" || (!document.documentElement.dataset.theme && matchMedia("(prefers-color-scheme: dark)").matches);
  const next = dark ? "light" : "dark";
  document.documentElement.dataset.theme = next;
  store("theme", next);
  renderResults();
}

function toggleFullscreen() {
  const region = $("#results");
  if (document.fullscreenElement) document.exitFullscreen();
  else if (region.requestFullscreen) region.requestFullscreen().catch(() => region.classList.toggle("fullscreen"));
  else region.classList.toggle("fullscreen");
}

function setCodeCollapsed(collapsed) {
  document.documentElement.classList.toggle("code-collapsed", collapsed);
  const btn = $("#btn-code");
  btn.setAttribute("aria-expanded", String(!collapsed));
  btn.querySelector(".btn-label").textContent = collapsed ? "Expand" : "Collapse";
  store("codeCollapsed", collapsed ? "1" : "0");
}

function markStale() {
  if (state.mode === "live" && Object.keys(state.results).length) state.stale = true;
  renderControls();
}

function bind() {
  $("#fw-switch").addEventListener("click", (ev) => {
    const b = ev.target.closest("[data-fw]");
    if (b) { state.framework = b.dataset.fw; renderAll(); }
  });
  $("#mode-switch").addEventListener("click", async (ev) => {
    const b = ev.target.closest("[data-mode]");
    if (b) await setMode(b.dataset.mode);
  });
  $("#view-switch").addEventListener("click", (ev) => {
    const b = ev.target.closest("[data-view]");
    if (b) { state.view = b.dataset.view; renderAll(); }
  });
  $("#threshold").addEventListener("input", (ev) => { state.threshold = Number(ev.target.value); renderResults(); });
  $("#k-nms").addEventListener("input", (ev) => { state.knobs.nms_iou = Number(ev.target.value); $("#k-nms-out").textContent = fixed(state.knobs.nms_iou); markStale(); });
  $("#k-stride").addEventListener("change", (ev) => { state.knobs.hog_stride = Number(ev.target.value); markStale(); });
  $("#k-scale").addEventListener("change", (ev) => { state.knobs.hog_scale = Number(ev.target.value); markStale(); });
  $("#k-hit").addEventListener("input", (ev) => { state.knobs.hit_threshold = Number(ev.target.value); $("#k-hit-out").textContent = fixed(state.knobs.hit_threshold); markStale(); });
  $("#btn-detect").addEventListener("click", detect);
  $("#btn-repeat").addEventListener("click", repeat5);
  $("#btn-theme").addEventListener("click", toggleTheme);
  $("#btn-code").addEventListener("click", () => setCodeCollapsed(!document.documentElement.classList.contains("code-collapsed")));
  $("#btn-full").addEventListener("click", toggleFullscreen);
  $("#btn-bigger").addEventListener("click", (ev) => {
    const on = document.documentElement.classList.toggle("bigger");
    ev.target.setAttribute("aria-pressed", String(on));
    store("bigger", on ? "1" : "0");
    renderResults();
  });
  $("#file").addEventListener("change", (ev) => useFile(ev.target.files[0]));
  window.addEventListener("dragover", (ev) => { ev.preventDefault(); document.body.classList.add("dragging"); });
  window.addEventListener("dragleave", (ev) => { if (!ev.relatedTarget) document.body.classList.remove("dragging"); });
  window.addEventListener("drop", (ev) => { ev.preventDefault(); document.body.classList.remove("dragging"); useFile(ev.dataTransfer?.files?.[0]); });
  window.addEventListener("paste", (ev) => {
    const file = [...(ev.clipboardData?.files ?? [])].find((f) => f.type.startsWith("image/"));
    if (file) useFile(file);
  });
  document.addEventListener("keydown", (ev) => {
    if (ev.ctrlKey || ev.metaKey || ev.altKey) return;
    if (ev.key === "Escape") {
      document.querySelectorAll("details[open]").forEach((d) => d.removeAttribute("open"));
      if (document.fullscreenElement) document.exitFullscreen();
      return;
    }
    // range sliders and text-like fields keep their keys; a file picker or checkbox that merely has focus does not
    if (ev.target.closest("input:not([type=file]):not([type=checkbox]), select, textarea, [contenteditable], [role=slider]")) return;
    const k = ev.key.toLowerCase();
    if (k === "d") detect();
    else if (k === "t") toggleTheme();
    else if (k === "f") toggleFullscreen();
    else if (k === "c") setCodeCollapsed(!document.documentElement.classList.contains("code-collapsed"));
  });
  let resizeTimer;
  matchMedia("(prefers-color-scheme: dark)").addEventListener?.("change", () => { clearTimeout(resizeTimer); resizeTimer = setTimeout(renderResults, 50); });
}

async function init() {
  try { if (localStorage.getItem("bigger") === "1") { document.documentElement.classList.add("bigger"); $("#btn-bigger").setAttribute("aria-pressed", "true"); } } catch { /* ignore */ }
  dataFiles = (await getJSON("/api/data")) ?? dataFiles;
  const [a, g, b] = await Promise.all([
    getJSON("/static/demo/data/algorithms.json"), getJSON("/static/demo/data/glossary.json"),
    dataFiles.files.includes("benchmark.json") ? getJSON("/static/demo/data/benchmark.json") : null,
  ]);
  algos = a?.algorithms ?? {
    hog_svm: { name: "HOG + SVM", family: "Classic machine learning" }, faster_rcnn: { name: "Faster R-CNN", family: "Deep learning, two-stage" },
    retinanet: { name: "RetinaNet", family: "Deep learning, one-stage" },
  };
  glossary = g ?? {};
  benchmark = b;
  codePanel = initCodePanel({ tabsEl: $("#tabs"), viewEl: $("#codeview"), fileBarEl: $("#filebar"), available: dataFiles.files }, activeTabs);
  bind();
  setCodeCollapsed(document.documentElement.classList.contains("code-collapsed")); // sync button with the saved state
  renderTerms();
  codePanel.init();
  const list = await getJSON("/samples");
  state.samples = list?.samples ?? [];
  // Warm the recorded-data cache while the server is still up, so the fallback works if it goes away.
  for (const name of [...state.samples.map((n) => n.replace(/\.[^.]+$/, "")), "_example"]) fetchRecorded(name);
  renderAll();
  await pollHealth();
  if (state.mode === "live" && state.samples.length && !state.photo) await selectSample(state.samples[0], true);
  else if (state.mode === "recorded") { await loadRecorded(); renderAll(); }
}

init();
