export function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") node.className = v;
    else if (k === "text") node.textContent = v;
    else if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
    else node.setAttribute(k, v);
  }
  node.append(...children.flat().filter((c) => c != null));
  return node;
}

// A photo with a canvas overlay. Boxes are given in original pixels and scaled to the displayed size.

const EXTRA = ["#7048e8", "#0b7285", "#e8590c", "#5c940d", "#c2255c", "#364fc7", "#2b8a3e", "#9c36b5", "#a61e4d", "#087f5b"];
const FIXED = { person: "#d6336c", car: "#1c6fd1" };

export function colorFor(label) {
  if (FIXED[label]) return FIXED[label];
  let h = 0;
  for (const ch of label) h = (h * 31 + ch.charCodeAt(0)) >>> 0;
  return EXTRA[h % EXTRA.length];
}

export function cssVar(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

function textOn(hex) {
  const h = hex.replace("#", "");
  const full = h.length === 3 ? [...h].map((c) => c + c).join("") : h;
  const [r, g, b] = [0, 2, 4].map((i) => parseInt(full.slice(i, i + 2), 16));
  return 0.299 * r + 0.587 * g + 0.114 * b > 150 ? "#111" : "#fff";
}

// items: [{label, confidence, box, color?, text?, dashed?, highlight?}]
export function createStage({ src, width, height, note }) {
  const root = document.createElement("div");
  root.className = "stage";
  let img = null;
  if (src) {
    img = document.createElement("img");
    img.alt = "Photo with detected objects";
    img.src = src;
    img.draggable = false;
    root.append(img);
  } else {
    const blank = document.createElement("div");
    blank.className = "blank";
    blank.style.aspectRatio = `${width} / ${height}`;
    blank.textContent = note || "Photo not available offline (boxes only)";
    root.append(blank);
  }
  const canvas = document.createElement("canvas");
  root.append(canvas);
  let items = [];

  function draw() {
    const cw = root.clientWidth;
    const ch = root.clientHeight;
    if (!cw || !ch) return;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.round(cw * dpr);
    canvas.height = Math.round(ch * dpr);
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, cw, ch);
    const k = cw / width;
    const lw = Math.max(2, cw / 280);
    const fs = Math.max(11, Math.min(18, cw / 38));
    ctx.font = `600 ${fs}px system-ui, sans-serif`;
    ctx.textBaseline = "top";
    for (const it of items) {
      const [x1, y1, x2, y2] = it.box.map((v) => v * k);
      const color = it.color || colorFor(it.label);
      ctx.lineWidth = it.highlight ? lw * 2.2 : lw;
      ctx.strokeStyle = color;
      ctx.setLineDash(it.dashed ? [lw * 3, lw * 2] : []);
      ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
      ctx.setLineDash([]);
      const text = it.text ?? `${it.label} ${it.confidence.toFixed(2)}`;
      const tw = ctx.measureText(text).width + 8;
      const th = fs + 4;
      const ty = y1 - th >= 0 ? y1 - th : y1;
      const tx = Math.min(x1, Math.max(0, cw - tw));
      ctx.fillStyle = color;
      ctx.fillRect(tx, ty, tw, th);
      ctx.fillStyle = textOn(color);
      ctx.fillText(text, tx + 4, ty + 2);
    }
  }

  const observer = new ResizeObserver(draw);
  observer.observe(root);
  if (img) img.addEventListener("load", draw);

  return {
    el: root,
    setItems(next) {
      items = next;
      draw();
    },
    redraw: draw,
    destroy() {
      observer.disconnect();
    },
  };
}
