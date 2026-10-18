// Code panel: tabs, a small hand-written Python tokenizer, line numbers, annotation badges, copy button.

const KEYWORDS = new Set([
  "False", "None", "True", "and", "as", "assert", "async", "await", "break", "class", "continue", "def", "del",
  "elif", "else", "except", "finally", "for", "from", "global", "if", "import", "in", "is", "lambda", "nonlocal",
  "not", "or", "pass", "raise", "return", "try", "while", "with", "yield",
]);

// Returns [{type, text}] with type in keyword | string | comment | number | plain.
export function tokenizePython(line) {
  const tokens = [];
  let i = 0;
  const push = (type, text) => {
    if (!text) return;
    const last = tokens[tokens.length - 1];
    if (last && last.type === type && type === "plain") last.text += text;
    else tokens.push({ type, text });
  };
  while (i < line.length) {
    const c = line[i];
    if (c === "#") {
      push("comment", line.slice(i));
      break;
    }
    if (c === '"' || c === "'") {
      const triple = line.startsWith(c.repeat(3), i);
      const quote = triple ? c.repeat(3) : c;
      let j = i + quote.length;
      while (j < line.length && !line.startsWith(quote, j)) j += line[j] === "\\" ? 2 : 1;
      j = Math.min(line.length, j + quote.length);
      push("string", line.slice(i, j));
      i = j;
      continue;
    }
    if (/[0-9]/.test(c) || (c === "." && /[0-9]/.test(line[i + 1] || ""))) {
      const m = /^(0x[0-9a-fA-F_]+|[0-9][0-9_]*\.?[0-9_]*(e[+-]?[0-9]+)?|\.[0-9]+)/.exec(line.slice(i));
      push("number", m[0]);
      i += m[0].length;
      continue;
    }
    if (/[A-Za-z_]/.test(c)) {
      const m = /^[A-Za-z_][A-Za-z0-9_]*/.exec(line.slice(i));
      push(KEYWORDS.has(m[0]) ? "keyword" : "plain", m[0]);
      i += m[0].length;
      continue;
    }
    push("plain", c);
    i += 1;
  }
  return tokens;
}

const esc = (s) => s.replace(/[&<>"]/g, (ch) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[ch]);

export function highlightLine(line) {
  return tokenizePython(line)
    .map((t) => (t.type === "plain" ? esc(t.text) : `<span class="tok-${t.type}">${esc(t.text)}</span>`))
    .join("");
}

// annotations: {"<excerpt id>:<text that appears in the line>": "short explanation"}
export function renderCode(excerpt, annotations = {}) {
  const lines = excerpt.code.split("\n");
  const notes = [];
  const marks = new Map();
  for (const [key, text] of Object.entries(annotations)) {
    const sep = key.indexOf(":");
    if (key.slice(0, sep) !== excerpt.id) continue;
    const needle = key.slice(sep + 1);
    const at = lines.findIndex((l, idx) => l.includes(needle) && !marks.has(idx));
    if (at >= 0) marks.set(at, text);
  }
  let gap = false; // after a "# ..." separator the real file line numbers are no longer contiguous
  const html = lines
    .map((line, idx) => {
      if (line === "# ...") gap = true;
      const lead = line.length - line.trimStart().length; // indent becomes padding so wrapped lines keep a hanging indent
      let badge = "";
      if (marks.has(idx)) {
        notes.push(marks.get(idx));
        badge = `<button class="badge" type="button" aria-label="Note ${notes.length}: ${esc(marks.get(idx))}" title="${esc(marks.get(idx))}">${notes.length}</button>`;
      }
      return `<div class="cl"><span class="gut">${badge}</span><span class="ln">${gap ? "" : excerpt.start_line + idx}</span><code style="padding-left:${lead + 4}ch;text-indent:-4ch">${highlightLine(line.slice(lead)) || " "}</code></div>`;
    })
    .join("");
  return { html, notes };
}

// ---------------------------------------------------------------------------------------------------
// Panel UI

const TABS = [
  { id: "hog", title: "HOG + SVM", ids: ["hog_predict"], intro: "Hand-made features: a fixed window slides over the image, edge-direction histograms, a linear SVM decides person or not." },
  { id: "frcnn", title: "Faster R-CNN", ids: ["fasterrcnn_pytorch"], intro: "Two stages (region proposals, then per-region classification). In our adapter the difference is one constructor call, because the library implements the algorithm." },
  { id: "retina", title: "RetinaNet", ids: ["retinanet_pytorch"], intro: "One dense pass, about 120k anchors scored at once; the model does resizing and NMS internally." },
  { id: "nms", title: "NMS in 15 lines", ids: ["nms_reference"], intro: "The same idea ends every detector; the deep ones do it inside the model." },
  { id: "post", title: "Shared post-processing", ids: ["postprocess_build"], intro: "Same threshold, clipping, naming and sorting for every algorithm." },
  { id: "contract", title: "Adapter contract", ids: ["backend_contract"], intro: "Every algorithm/framework implements _load and _predict; the rest of the system never imports torch, tensorflow or cv2." },
  { id: "framework", title: "Framework check", ids: ["retinanet_pytorch", "retinanet_tensorflow"], titles: ["PyTorch (torchvision)", "TensorFlow (KerasHub)"], intro: "The same algorithm in two frameworks, side by side." },
  { id: "training", title: "Training: loop vs fit", ids: ["train_pytorch", "train_keras"], titles: ["PyTorch: explicit loop", "Keras: compile + fit"], intro: "The same small CNN trained both ways (synthetic data; inference is what the demo is about)." },
];

async function getJSON(url) {
  try {
    const r = await fetch(url);
    return r.ok ? await r.json() : null;
  } catch {
    return null;
  }
}

function fallbackCopy(text) {
  const ta = document.createElement("textarea");
  ta.value = text;
  document.body.append(ta);
  ta.select();
  try { document.execCommand("copy"); } catch { /* ignore */ }
  ta.remove();
}

export function copyText(text) {
  if (navigator.clipboard?.writeText) return navigator.clipboard.writeText(text).catch(() => fallbackCopy(text));
  return Promise.resolve(fallbackCopy(text));
}

export function initCodePanel({ tabsEl, viewEl, fileBarEl, available = [] }, getActive) {
  const cache = new Map();
  let snapshot;
  let annotations = {};
  let frameworkCheck = { callouts: [] };
  let lines = null;
  let files = null;
  let current = TABS[0].id;

  async function excerpt(id) {
    if (cache.has(id)) return cache.get(id);
    let data = await getJSON(`/api/code/${id}`);
    let fromSnapshot = false;
    if (!data) {
      snapshot ??= (await getJSON("/static/demo/data/code_excerpts.json")) ?? {};  // always shipped
      data = snapshot[id] ?? null;
      fromSnapshot = !!data;
    }
    const value = data ? { ...data, fromSnapshot } : null;
    if (value) cache.set(id, value);
    return value;
  }

  function codeBox(ex, notesMap) {
    const { html, notes } = renderCode(ex, notesMap);
    const wrap = document.createElement("div");
    wrap.className = "codewrap";
    const box = document.createElement("div");
    box.className = "codebox";
    box.tabIndex = 0;
    box.setAttribute("aria-label", `Code from ${ex.file}`);
    box.innerHTML = html;
    const copy = document.createElement("button");
    copy.type = "button";
    copy.className = "ghost copybtn";
    copy.textContent = "Copy";
    copy.addEventListener("click", async () => {
      await copyText(ex.code);
      copy.textContent = "Copied";
      setTimeout(() => (copy.textContent = "Copy"), 1200);
    });
    wrap.append(copy, box);
    return { wrap, box, notes };
  }

  function noteList(notes) {
    const ol = document.createElement("ol");
    ol.className = "notes";
    for (const n of notes) {
      const li = document.createElement("li");
      li.textContent = n;
      ol.append(li);
    }
    return ol;
  }

  function fileChips(exs) {
    fileBarEl.replaceChildren();
    for (const ex of exs) {
      if (!ex) continue;
      const chip = document.createElement("span");
      chip.className = "chipfile";
      chip.textContent = `${ex.file}:${ex.start_line}${ex.fromSnapshot ? " (snapshot)" : ""}`;
      fileBarEl.append(chip);
    }
    if (files) {
      const sum = (list) => list.reduce((n, f) => n + f.lines, 0);
      const total = sum(files.shared) + sum(files.framework);
      const bar = document.createElement("span");
      bar.className = "sharebar";
      bar.title = `Shared: ${files.shared.map((f) => f.file).join(", ")}. Framework-specific: ${files.framework.map((f) => f.file).join(", ")}.`;
      const a = document.createElement("span");
      a.className = "shared";
      a.style.width = `${(sum(files.shared) / total) * 100}%`;
      const b = document.createElement("span");
      b.className = "fw";
      b.style.width = `${(sum(files.framework) / total) * 100}%`;
      bar.append(a, b);
      const label = document.createElement("span");
      label.textContent = `shared ${sum(files.shared)} lines | framework-specific ${sum(files.framework)} lines`;
      label.title = bar.title;
      fileBarEl.append(bar, label);
    }
  }

  async function show(id) {
    current = id;
    const tab = TABS.find((t) => t.id === id);
    renderTabs();
    viewEl.replaceChildren();
    const exs = await Promise.all(tab.ids.map(excerpt));
    if (current !== id) return;
    fileChips(exs);
    const intro = document.createElement("p");
    intro.className = "codeintro";
    intro.textContent = tab.intro;
    viewEl.append(intro);
    if (exs.some((e) => !e)) {
      const err = document.createElement("p");
      err.className = "error";
      err.textContent = "Code is not available (server unreachable and no snapshot found).";
      viewEl.append(err);
      return;
    }
    if (tab.ids.length === 1) {
      const { wrap, notes } = codeBox(exs[0], annotations);
      viewEl.append(wrap, noteList(notes));
      return;
    }
    const callouts = id === "framework" ? frameworkCheck.callouts : null;
    const panes = document.createElement("div");
    panes.className = "panes two";
    const boxes = [];
    exs.forEach((ex, i) => {
      const map = {};
      if (callouts) callouts.forEach((c, n) => { map[c[i === 0 ? "left" : "right"]] = `${n + 1}. ${c.title}`; });
      const pane = document.createElement("div");
      pane.className = "pane";
      const h = document.createElement("h4");
      h.textContent = tab.titles[i];
      const { wrap, box, notes } = codeBox(ex, callouts ? map : annotations);
      pane.append(h, wrap);
      if (!callouts) pane.append(noteList(notes));
      panes.append(pane);
      boxes.push(box);
    });
    let syncing = false;
    boxes.forEach((box, i) => box.addEventListener("scroll", () => {
      if (syncing) return;
      syncing = true;
      const other = boxes[1 - i];
      const ratio = box.scrollTop / Math.max(1, box.scrollHeight - box.clientHeight);
      other.scrollTop = ratio * (other.scrollHeight - other.clientHeight);
      requestAnimationFrame(() => { syncing = false; });
    }));
    viewEl.append(panes);
    if (callouts) {
      const ol = document.createElement("ol");
      ol.className = "notes";
      for (const c of callouts) {
        const li = document.createElement("li");
        const strong = document.createElement("strong");
        strong.textContent = `${c.title}: `;
        li.append(strong, document.createTextNode(c.text));
        ol.append(li);
      }
      viewEl.append(ol);
    }
    if (id === "training" && lines) {
      const stat = document.createElement("p");
      stat.className = "codestat num";
      stat.textContent = `Training code lines (non-blank, non-comment): PyTorch ${lines.train_pytorch}, Keras ${lines.train_keras}.`;
      viewEl.append(stat);
    }
  }

  function renderTabs() {
    const active = getActive();
    tabsEl.replaceChildren();
    for (const t of TABS) {
      const b = document.createElement("button");
      b.type = "button";
      b.className = "tab" + (active.has(t.id) ? " active" : "");
      b.setAttribute("role", "tab");
      b.setAttribute("aria-selected", String(t.id === current));
      b.textContent = t.title;
      if (active.has(t.id)) b.title = "Used by an active result";
      b.addEventListener("click", () => show(t.id));
      tabsEl.append(b);
    }
  }

  return {
    async init() {
      const [a, f, l, c] = await Promise.all([
        getJSON("/static/demo/data/code_annotations.json"),
        getJSON("/static/demo/data/framework_check.json"),
        available.includes("code_lines.json") ? getJSON("/static/demo/data/code_lines.json") : null,
        available.includes("code_files.json") ? getJSON("/static/demo/data/code_files.json") : null,
      ]);
      annotations = a ?? {};
      frameworkCheck = f ?? { callouts: [] };
      lines = l;
      files = c;
      await show(current);
    },
    refreshTabs: renderTabs,
  };
}
