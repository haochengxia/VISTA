// CacheLab project page: charts drawn as plain SVG from window.CACHELAB (site/data.js).
// Logs store the change in misses vs S3-FIFO (negative = fewer). Charts show the reduction
// (positive = fewer misses), so longer bars and higher lines are better everywhere.
"use strict";
(() => {
  const D = window.CACHELAB;
  const NS = "http://www.w3.org/2000/svg";
  const MINUS = "−";
  const red = (v) => (v === 0 ? 0 : -v);
  const num = (v, d = 2) => (v < 0 ? MINUS : "") + Math.abs(v).toFixed(d);
  const pct = (v, d = 2) => num(v, d) + "%";
  const fewer = (r) => (r >= 0 ? `${r.toFixed(2)}% fewer` : `${(-r).toFixed(2)}% more`);

  // ---------- DOM helpers ----------
  function svgEl(tag, attrs = {}, parent, text) {
    const e = document.createElementNS(NS, tag);
    for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
    if (text !== undefined) e.textContent = text;
    if (parent) parent.appendChild(e);
    return e;
  }
  function htmlEl(tag, attrs = {}, parent, text) {
    const e = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
    if (text !== undefined) e.textContent = text;
    if (parent) parent.appendChild(e);
    return e;
  }
  const ctx = document.createElement("canvas").getContext("2d");
  function textWidth(s, size = 13, weight = 400) {
    ctx.font = `${weight} ${size}px ${getComputedStyle(document.body).fontFamily}`;
    return ctx.measureText(s).width;
  }
  function niceTicks(lo, hi, count = 5) {
    const span = hi - lo || 1;
    const raw = span / Math.max(1, count);
    const mag = 10 ** Math.floor(Math.log10(raw));
    const err = raw / mag;
    const step = (err >= 7.5 ? 10 : err >= 3.5 ? 5 : err >= 1.5 ? 2 : 1) * mag;
    const start = Math.floor(lo / step + 1e-9) * step;
    const end = Math.ceil(hi / step - 1e-9) * step;
    const ticks = [];
    for (let t = start; t <= end + step / 2; t += step) ticks.push(+t.toFixed(10));
    return { ticks, lo: start, hi: end, decimals: step < 1 ? (step < 0.1 ? 2 : 1) : 0 };
  }
  // A bar from a to b along one axis, rounded (4px) only at the data end.
  function barPath(horizontal, base, end, c0, thick) {
    const len = Math.abs(end - base);
    const r = Math.min(4, len, thick / 2);
    const dir = end >= base ? 1 : -1;
    if (horizontal) {
      const y0 = c0, y1 = c0 + thick, e = end, b = base;
      return `M${b},${y0} H${e - dir * r} Q${e},${y0} ${e},${y0 + r} V${y1 - r} Q${e},${y1} ${e - dir * r},${y1} H${b} Z`;
    }
    const x0 = c0, x1 = c0 + thick, e = end, b = base; // vertical: smaller y is up
    return `M${x0},${b} V${e - dir * r} Q${x0},${e} ${x0 + r},${e} H${x1 - r} Q${x1},${e} ${x1},${e - dir * r} V${b} Z`;
  }

  // ---------- tooltip (one per page, content via textContent) ----------
  const tip = htmlEl("div", { class: "tip", role: "tooltip", id: "tip", hidden: "" }, document.body);
  function fillTip(content) {
    tip.replaceChildren();
    htmlEl("div", { class: "tip-title" }, tip, content.title);
    for (const r of content.rows) {
      const row = htmlEl("div", { class: "tip-row" }, tip);
      if (r.cls) htmlEl("span", { class: `sw bg-${r.cls}` }, row);
      htmlEl("span", { class: "tip-label" }, row, r.label);
      htmlEl("span", { class: "tip-value" }, row, r.value);
    }
    if (content.note) htmlEl("div", { class: "tip-note" }, tip, content.note);
    tip.hidden = false;
  }
  function placeTip(x, y) {
    const pad = 14, w = tip.offsetWidth, h = tip.offsetHeight;
    let left = x + pad, top = y + pad;
    if (left + w > innerWidth - 8) left = x - w - pad;
    if (top + h > innerHeight - 8) top = y - h - pad;
    tip.style.transform = `translate(${Math.max(8, left)}px, ${Math.max(8, top)}px)`;
  }
  const hideTip = () => (tip.hidden = true);
  // Hover and keyboard focus both show the tooltip; the hit target is larger than the mark.
  function bindTip(target, content, { onEnter, onLeave } = {}) {
    target.setAttribute("tabindex", "0");
    target.addEventListener("pointerenter", (e) => { onEnter?.(); fillTip(content()); placeTip(e.clientX, e.clientY); });
    target.addEventListener("pointermove", (e) => placeTip(e.clientX, e.clientY));
    target.addEventListener("pointerleave", () => { onLeave?.(); hideTip(); });
    target.addEventListener("focus", () => {
      onEnter?.(); fillTip(content());
      const b = target.getBoundingClientRect();
      placeTip(b.left + b.width / 2, b.top + b.height / 2);
    });
    target.addEventListener("blur", () => { onLeave?.(); hideTip(); });
  }

  // ---------- legend and table ----------
  function legend(host, items) {
    host.replaceChildren();
    const box = htmlEl("div", { class: "legend" }, host);
    for (const it of items) {
      const item = htmlEl("span", { class: "legend-item" }, box);
      htmlEl("span", { class: `sw ${it.line ? "sw-line " : ""}bg-${it.cls}` }, item);
      htmlEl("span", {}, item, it.label);
    }
  }
  function dataTable(host, caption, head, body, { plain = false, rowClass } = {}) {
    host.replaceChildren();
    let d = host;
    if (!plain) {
      d = htmlEl("details", { class: "table-view" }, host);
      htmlEl("summary", {}, d, "Data table");
    }
    const wrap = htmlEl("div", { class: "table-wrap" }, d);
    const t = htmlEl("table", {}, wrap);
    htmlEl("caption", {}, t, caption);
    const hr = htmlEl("tr", {}, htmlEl("thead", {}, t));
    head.forEach((h) => htmlEl("th", { scope: "col" }, hr, h));
    const tb = htmlEl("tbody", {}, t);
    body.forEach((row, i) => {
      const tr = htmlEl("tr", rowClass?.(i) ? { class: rowClass(i) } : {}, tb);
      row.forEach((c, j) => htmlEl(j ? "td" : "th", j ? {} : { scope: "row" }, tr, String(c)));
    });
  }

  // ---------- a figure: legend slot, plot (redrawn on resize), table slot ----------
  function mount(id, draw) {
    const fig = document.getElementById(id);
    const plot = fig.querySelector(".plot");
    let lastWidth = 0;
    const render = () => {
      const w = Math.floor(plot.clientWidth);
      if (!w || w === lastWidth) return;
      lastWidth = w;
      plot.replaceChildren();
      draw(plot, w);
    };
    new ResizeObserver(render).observe(plot);
    render();
    return { fig, redraw: () => { lastWidth = 0; render(); } };
  }

  // ---------- horizontal grouped bars: categories are rows ----------
  function hbars(plot, width, o) {
    const ns = o.series.length;
    const barH = o.barH ?? (ns > 1 ? 14 : 18);
    const labelW = Math.max(...o.cats.map((c) => textWidth(c.label, 13))) + 14;
    const above = labelW > width * 0.42; // narrow screens: labels go above the bars
    const ref = o.refs?.length ? 22 : 6;
    const m = { l: above ? 4 : labelW, r: 64, t: ref, b: 34 };
    const group = ns * barH + (ns - 1) * 2;
    const band = group + 14 + (above ? 20 : 0);
    const h = m.t + o.cats.length * band + m.b;
    const vals = o.cats.flatMap((c) => o.series.map((s) => o.get(c, s)));
    const nt = niceTicks(Math.min(0, ...vals, o.min ?? 0), Math.max(0, ...vals, o.max ?? 0), Math.max(2, Math.floor((width - m.l - m.r) / 80)));
    const x = (v) => m.l + ((v - nt.lo) / (nt.hi - nt.lo)) * (width - m.l - m.r);
    const svg = svgEl("svg", { width, height: h, role: "img", "aria-label": o.aria }, plot);
    for (const t of nt.ticks) {
      svgEl("line", { x1: x(t), x2: x(t), y1: m.t - 4, y2: h - m.b, class: t === 0 ? "baseline" : "grid" }, svg);
      svgEl("text", { x: x(t), y: h - m.b + 16, class: "tick", "text-anchor": "middle" }, svg, pct(t, nt.decimals));
    }
    if (o.axis) svgEl("text", { x: x(nt.lo), y: h - 4, class: "axis-title" }, svg, o.axis);
    for (const rf of o.refs ?? []) {
      svgEl("line", { x1: x(rf.value), x2: x(rf.value), y1: m.t - 4, y2: h - m.b, class: `ref ${rf.cls ? "ref-" + rf.cls : ""}` }, svg);
      svgEl("text", { x: x(rf.value), y: m.t - 9, class: "ref-label", "text-anchor": "middle" }, svg, rf.label);
    }
    o.cats.forEach((c, i) => {
      const top = m.t + i * band;
      const g = svgEl("g", { class: "grp" }, svg);
      svgEl("rect", { x: 0, y: top, width, height: band, class: "band-wash", rx: 4 }, g);
      const barTop = top + 7 + (above ? 20 : 0);
      if (above) svgEl("text", { x: m.l, y: top + 18, class: "cat" }, g, c.label);
      else svgEl("text", { x: m.l - 10, y: barTop + group / 2, class: "cat", "text-anchor": "end", "dominant-baseline": "central" }, g, c.label);
      o.series.forEach((s, j) => {
        const v = o.get(c, s);
        const y = barTop + j * (barH + 2);
        const cls = o.clsOf ? o.clsOf(c, s) : s.cls;
        svgEl("path", { d: barPath(true, x(0), x(v), y, barH), class: `bar fill-${cls}` }, g);
        if (o.label?.(c, s)) {
          const pos = v >= 0;
          svgEl("text", { x: x(v) + (pos ? 6 : -6), y: y + barH / 2, class: `vlabel ${o.labelDim?.(c, s) ? "dim" : ""}`, "text-anchor": pos ? "start" : "end", "dominant-baseline": "central" }, g, pct(v));
        }
      });
      const hit = svgEl("rect", { x: 0, y: top, width, height: band, class: "hit", "aria-label": c.label }, g);
      bindTip(hit, () => o.tip(c), {
        onEnter: () => { svg.classList.add("active"); g.classList.add("on"); },
        onLeave: () => { svg.classList.remove("active"); g.classList.remove("on"); },
      });
    });
  }

  // ---------- vertical grouped bars: categories along x ----------
  function vbars(plot, width, o) {
    const ns = o.series.length;
    const h = o.height ?? 270;
    const m = { l: 46, r: 8, t: 22, b: o.subOf ? 46 : 30 };
    const vals = o.cats.flatMap((c) => o.series.map((s) => o.get(c, s)));
    const nt = niceTicks(Math.min(0, ...vals), Math.max(0, ...vals), 5);
    const y = (v) => m.t + (1 - (v - nt.lo) / (nt.hi - nt.lo)) * (h - m.t - m.b);
    const svg = svgEl("svg", { width, height: h, role: "img", "aria-label": o.aria }, plot);
    for (const t of nt.ticks) {
      svgEl("line", { x1: m.l, x2: width - m.r, y1: y(t), y2: y(t), class: t === 0 ? "baseline" : "grid" }, svg);
      svgEl("text", { x: m.l - 8, y: y(t), class: "tick", "text-anchor": "end", "dominant-baseline": "central" }, svg, pct(t, nt.decimals));
    }
    const band = (width - m.l - m.r) / o.cats.length;
    const bw = Math.max(8, Math.min(24, (band * 0.6 - (ns - 1) * 2) / ns));
    const gw = ns * bw + (ns - 1) * 2;
    o.cats.forEach((c, i) => {
      const cx = m.l + band * (i + 0.5);
      const g = svgEl("g", { class: "grp" }, svg);
      svgEl("rect", { x: cx - band / 2 + 2, y: m.t - 18, width: band - 4, height: h - m.t - m.b + 18, class: "band-wash", rx: 4 }, g);
      o.series.forEach((s, j) => {
        const v = o.get(c, s);
        const x0 = cx - gw / 2 + j * (bw + 2);
        svgEl("path", { d: barPath(false, y(0), y(v), x0, bw), class: `bar fill-${s.cls}` }, g);
        if (o.label?.(c, s)) {
          const pos = v >= 0;
          svgEl("text", { x: x0 + bw / 2, y: y(v) + (pos ? -6 : 14), class: "vlabel", "text-anchor": "middle" }, g, num(v, 1));
        }
      });
      svgEl("text", { x: cx, y: h - m.b + 18, class: "cat", "text-anchor": "middle" }, g, c.label);
      if (o.subOf) svgEl("text", { x: cx, y: h - m.b + 35, class: "cat-sub", "text-anchor": "middle" }, g, o.subOf(c));
      const hit = svgEl("rect", { x: cx - band / 2, y: m.t - 18, width: band, height: h - m.t + 18, class: "hit", "aria-label": c.label }, g);
      bindTip(hit, () => o.tip(c), {
        onEnter: () => { svg.classList.add("active"); g.classList.add("on"); },
        onLeave: () => { svg.classList.remove("active"); g.classList.remove("on"); },
      });
    });
  }

  // ---------- lines with markers, crosshair tooltip, direct end labels ----------
  function lines(plot, width, o) {
    const h = o.height ?? 280;
    const endW = o.endLabels ? Math.max(...o.series.map((s) => textWidth(s.label, 12))) + 16 : 20;
    const m = { l: 50, r: endW, t: 22, b: 40 };
    const vals = o.series.flatMap((s) => s.values).concat((o.refs ?? []).map((r) => r.value));
    const nt = niceTicks(Math.min(o.min ?? Infinity, ...vals), Math.max(o.max ?? -Infinity, ...vals), 5);
    const xv = o.xs.map((v) => (o.xLog ? Math.log2(v) : v));
    const x0 = Math.min(...xv), x1 = Math.max(...xv);
    const pad = o.xPad ?? 0;
    const x = (i) => m.l + pad + ((xv[i] - x0) / (x1 - x0)) * (width - m.l - m.r - 2 * pad);
    const y = (v) => m.t + (1 - (v - nt.lo) / (nt.hi - nt.lo)) * (h - m.t - m.b);
    const svg = svgEl("svg", { width, height: h, role: "img", "aria-label": o.aria }, plot);
    for (const t of nt.ticks) {
      svgEl("line", { x1: m.l, x2: width - m.r + 8, y1: y(t), y2: y(t), class: t === 0 ? "baseline" : "grid" }, svg);
      svgEl("text", { x: m.l - 8, y: y(t), class: "tick", "text-anchor": "end", "dominant-baseline": "central" }, svg, pct(t, nt.decimals));
    }
    o.xs.forEach((_, i) => svgEl("text", { x: x(i), y: h - m.b + 18, class: "tick", "text-anchor": "middle" }, svg, o.xTick(i)));
    if (o.xTitle) svgEl("text", { x: (m.l + width - m.r) / 2, y: h - 4, class: "axis-title", "text-anchor": "middle" }, svg, o.xTitle);
    for (const rf of o.refs ?? []) {
      svgEl("line", { x1: m.l, x2: width - m.r + 8, y1: y(rf.value), y2: y(rf.value), class: `ref ${rf.cls ? "ref-" + rf.cls : ""}` }, svg);
      svgEl("text", { x: rf.right ? width - m.r + 8 : m.l + 6, y: y(rf.value) + (rf.below ? 14 : -6), class: "ref-label", "text-anchor": rf.right ? "end" : "start" }, svg, rf.label);
    }
    const markers = o.series.map(() => []);
    o.series.forEach((s, k) => {
      const d = s.values.map((v, i) => `${i ? "L" : "M"}${x(i)},${y(v)}`).join(" ");
      svgEl("path", { d, class: `ln stroke-${s.cls}` }, svg);
      s.values.forEach((v, i) => markers[k].push(svgEl("circle", { cx: x(i), cy: y(v), r: 4, class: `marker fill-${s.cls}` }, svg)));
    });
    for (const a of o.annotate ?? []) {
      const s = o.series[a.series ?? 0];
      svgEl("text", { x: x(a.i), y: y(s.values[a.i]) + (a.below ? 20 : -12), class: "annot", "text-anchor": "middle" }, svg, a.text);
    }
    if (o.endLabels) {
      const last = o.xs.length - 1;
      const items = o.series.map((s) => ({ s, y: y(s.values[last]) })).sort((a, b) => a.y - b.y);
      for (let i = 1; i < items.length; i++) items[i].y = Math.max(items[i].y, items[i - 1].y + 15);
      for (const it of items) svgEl("text", { x: x(last) + 10, y: it.y, class: "end-label", "dominant-baseline": "central" }, svg, it.s.label);
    }
    // crosshair: one hit column per x value
    const cross = svgEl("line", { y1: m.t - 6, y2: h - m.b, class: "cross", visibility: "hidden" }, svg);
    o.xs.forEach((_, i) => {
      const left = i === 0 ? m.l : (x(i - 1) + x(i)) / 2;
      const right = i === o.xs.length - 1 ? width - m.r + 8 : (x(i) + x(i + 1)) / 2;
      const hit = svgEl("rect", { x: left, y: m.t - 6, width: right - left, height: h - m.t - m.b + 6, class: "hit", "aria-label": o.xTick(i) }, svg);
      bindTip(hit, () => o.tip(i), {
        onEnter: () => {
          cross.setAttribute("x1", x(i)); cross.setAttribute("x2", x(i)); cross.setAttribute("visibility", "visible");
          markers.forEach((ms) => ms[i].setAttribute("r", 5.5));
        },
        onLeave: () => { cross.setAttribute("visibility", "hidden"); markers.forEach((ms) => ms[i].setAttribute("r", 4)); },
      });
    });
  }

  // =================== page content ===================
  const ho = D.heldout;
  const v1 = ho.find((r) => r.role === "ours");
  const l3 = ho.find((r) => r.role === "rival");

  // Stat tiles
  const fill = (k, s) => document.querySelectorAll(`[data-fill="${k}"]`).forEach((e) => (e.textContent = s));
  fill("v1-mean", `${red(v1.mean).toFixed(2)}%`);
  fill("3l-mean", `${red(l3.mean).toFixed(2)}%`);
  fill("v1-median", `${red(v1.median).toFixed(2)}%`);
  fill("3l-median", `${red(l3.median).toFixed(2)}%`);
  fill("v1-worse", `${v1.worse} of ${v1.cells}`);
  fill("3l-worse", String(l3.worse));
  fill("traces-won", `${D.tracesWon.won} of ${D.tracesWon.traces}`);
  fill("traces-bar", String(D.tracesWon.bar));
  fill("cells", String(D.heldoutCells));

  // Fig 1: held-out, mean and median as two panels on one shared scale
  const roleCls = { ours: "c1", rival: "c2", variant: "cn", ideal: "cn" };
  const hoMax = Math.max(...ho.map((r) => red(r.mean)));
  const hoTip = (r) => ({
    title: r.name,
    rows: [
      { label: "mean", value: fewer(red(r.mean)) },
      { label: "median", value: fewer(red(r.median)) },
      { label: "cells better than 3L-Cache", value: r.role === "rival" ? "–" : `${r.beats} of ${r.cells}` },
      { label: "cells worse than S3-FIFO", value: `${r.worse} of ${r.cells}` },
    ],
    note: r.role === "ideal" ? "Upper bound: every candidate runs at full size, switching is free." : `Selector memory: ${r.memory}`,
  });
  for (const [id, key] of [["fig-heldout-mean", "mean"], ["fig-heldout-median", "median"]]) {
    mount(id, (plot, w) =>
      hbars(plot, w, {
        cats: ho.map((r) => ({ ...r, label: r.name })),
        series: [{ key }],
        get: (r) => red(r[key]),
        clsOf: (r) => roleCls[r.role],
        max: hoMax,
        label: () => true,
        labelDim: (r) => r.role === "variant" || r.role === "ideal",
        tip: hoTip,
        aria: `Held-out ${key} reduction in misses vs S3-FIFO by policy`,
      }),
    );
  }
  legend(document.querySelector("#fig-heldout .legend-slot"), [
    { cls: "c1", label: "v1" },
    { cls: "c2", label: "3L-Cache" },
    { cls: "cn", label: "other selector configurations" },
  ]);
  dataTable(
    document.querySelector("#fig-heldout .table-slot"),
    `Held-out traces, ${D.heldoutCells} cells, last 80% of each trace. Fewer misses than S3-FIFO (negative = more).`,
    ["policy", "mean", "median", "cells better than 3L-Cache", "cells worse than S3-FIFO", "selector memory"],
    ho.map((r) => [r.name, pct(red(r.mean)), pct(red(r.median)), r.role === "rival" ? "–" : `${r.beats}/${r.cells}`, r.worse, r.memory]),
    { rowClass: (i) => (ho[i].role === "ours" ? "ours" : "") },
  );

  // Fig 2: by cache size, held-out / train toggle
  const sizeSeries = [{ key: "v1", label: "v1", cls: "c1" }, { key: "3L", label: "3L-Cache", cls: "c2" }];
  let sizeView = "heldout";
  const sizeFig = mount("fig-size", (plot, w) =>
    vbars(plot, w, {
      cats: D.bySize[sizeView].map((r) => ({ ...r, label: r.size })),
      series: sizeSeries,
      get: (r, s) => red(r[s.key]),
      subOf: (r) => `${r.cells} cells`,
      label: () => true,
      tip: (r) => ({
        title: `Cache ${r.size} of the trace's objects`,
        rows: sizeSeries.map((s) => ({ cls: s.cls, label: s.label, value: fewer(red(r[s.key])) })),
        note: `${r.cells} cells, ${sizeView === "heldout" ? "held-out" : "train, 5-fold CV by trace"}`,
      }),
      aria: "Fewer misses than S3-FIFO by cache size, v1 and 3L-Cache",
    }),
  );
  legend(document.querySelector("#fig-size .legend-slot"), sizeSeries);
  const sizeTable = () =>
    dataTable(
      document.querySelector("#fig-size .table-slot"),
      `${sizeView === "heldout" ? "Held-out" : "Train-long, 5-fold cross-validation by trace"}. Mean fewer misses than S3-FIFO.`,
      ["cache size (% of objects)", "cells", "v1", "3L-Cache"],
      D.bySize[sizeView].map((r) => [r.size, r.cells, pct(red(r.v1)), pct(red(r["3L"]))]),
    );
  sizeTable();
  document.querySelectorAll("#fig-size .controls button").forEach((b) =>
    b.addEventListener("click", () => {
      sizeView = b.dataset.view;
      document.querySelectorAll("#fig-size .controls button").forEach((x) => x.setAttribute("aria-pressed", String(x === b)));
      sizeFig.redraw();
      sizeTable();
    }),
  );

  // Fig 3: choosing is the bottleneck (train, every design run at full size)
  const ch = D.choosing.filter((r) => !r.name.includes("live cache"));
  mount("fig-choose", (plot, w) =>
    hbars(plot, w, {
      cats: ch.map((r) => ({ ...r, label: r.name })),
      series: [{ key: "mean" }],
      get: (r) => red(r.mean),
      clsOf: (r) => (r.name.includes("online") ? "c1" : "cn"),
      label: () => true,
      refs: [{ value: red(D.train3L), label: "3L-Cache", cls: "c2" }],
      tip: (r) => ({
        title: r.name,
        rows: [
          { label: "mean", value: fewer(red(r.mean)) },
          { label: "median", value: fewer(red(r.median)) },
          { label: "cells better than 3L-Cache", value: `${r.beats} of ${r.cells}` },
        ],
      }),
      aria: "Fewer misses than S3-FIFO by how the design is chosen",
    }),
  );
  dataTable(
    document.querySelector("#fig-choose .table-slot"),
    "Train-long, 100 cells, 426 designs. Mean fewer misses than S3-FIFO.",
    ["how the design is chosen", "mean", "median", "cells better than 3L-Cache"],
    ch.map((r) => [r.name, pct(red(r.mean)), pct(red(r.median)), `${r.beats}/${r.cells}`]).concat([["3L-Cache", pct(red(D.train3L)), "", ""]]),
  );

  // Fig 4: online vs static, held-out
  const osSeries = [{ key: "online", label: "re-chosen online", cls: "c1" }, { key: "static", label: "chosen once at 20%", cls: "cn" }];
  mount("fig-online", (plot, w) =>
    vbars(plot, w, {
      cats: D.onlineStatic.map((r) => ({ ...r, label: r.set })),
      series: osSeries,
      get: (r, s) => red(r[s.key]),
      label: () => true,
      height: 240,
      tip: (r) => ({ title: `Candidates: ${r.set}`, rows: osSeries.map((s) => ({ cls: s.cls, label: s.label, value: fewer(red(r[s.key])) })) }),
      aria: "Online versus static choice on held-out traces",
    }),
  );
  legend(document.querySelector("#fig-online .legend-slot"), osSeries);
  dataTable(
    document.querySelector("#fig-online .table-slot"),
    "Held-out, live cache with miniature simulations. Mean fewer misses than S3-FIFO.",
    ["candidates", "re-chosen online", "chosen once"],
    D.onlineStatic.map((r) => [r.set, pct(red(r.online)), pct(red(r.static))]),
  );

  // Fig 5: portfolio size (train, 5-fold CV)
  const pf = D.portfolio;
  const k32 = pf.findIndex((r) => r.k === 32);
  mount("fig-portfolio", (plot, w) =>
    lines(plot, w, {
      xs: pf.map((r) => r.k),
      xLog: true,
      xPad: 8,
      xTick: (i) => String(pf[i].k),
      xTitle: "designs in the portfolio (log scale)",
      series: [{ label: "greedy portfolio", cls: "c1", values: pf.map((r) => red(r.mean)) }],
      min: 0,
      refs: [
        { value: red(D.portfolioAll), label: "all 423 designs" },
        { value: red(D.train3L), label: "3L-Cache", cls: "c2" },
      ],
      annotate: [{ i: k32, text: "v1", below: true }],
      tip: (i) => ({
        title: `${pf[i].k} designs`,
        rows: [
          { cls: "c1", label: "mean", value: fewer(red(pf[i].mean)) },
          { label: "median", value: fewer(red(pf[i].median)) },
        ],
        note: "Train-long, 5-fold CV by trace, idealized switching.",
      }),
      aria: "Fewer misses than S3-FIFO versus number of designs in the portfolio",
    }),
  );
  dataTable(
    document.querySelector("#fig-portfolio .table-slot"),
    `Train-long, 5-fold CV by trace, idealized online choice. All 423 designs: ${pct(red(D.portfolioAll))}; 3L-Cache: ${pct(red(D.train3L))}.`,
    ["designs", "mean", "median"],
    pf.map((r) => [r.k, pct(red(r.mean)), pct(red(r.median))]),
  );

  // Fig 6: takeover point and hysteresis (train, live cache)
  const tkSeries = [{ key: "split", label: "take over at 20%", cls: "c1" }, { key: "start", label: "select from window 4", cls: "cn" }];
  mount("fig-takeover", (plot, w) =>
    hbars(plot, w, {
      cats: D.takeover.map((r) => ({ ...r, label: r.policy })),
      series: tkSeries,
      get: (r, s) => red(r[s.key]),
      label: (r, s) => s.key === "split",
      tip: (r) => ({ title: r.policy, rows: tkSeries.map((s) => ({ cls: s.cls, label: s.label, value: fewer(red(r[s.key])) })) }),
      aria: "Fewer misses than S3-FIFO by takeover point and switching margin",
    }),
  );
  legend(document.querySelector("#fig-takeover .legend-slot"), tkSeries);
  dataTable(
    document.querySelector("#fig-takeover .table-slot"),
    "Train-long, 100 cells, live cache. Mean fewer misses than S3-FIFO.",
    ["selector", "take over at 20%", "select from window 4"],
    D.takeover.map((r) => [r.policy, pct(red(r.split)), pct(red(r.start))]),
  );

  // Fig 7: ghost depth (ordinal ramp by cache size class)
  const gh = D.ghost;
  const ghSeries = gh.classes.map((c, i) => ({ label: `cache ${c.size}`, cls: `o${i + 1}`, values: c.change.map(red) }));
  mount("fig-ghost", (plot, w) =>
    lines(plot, w, {
      xs: gh.depths,
      xLog: true,
      xPad: 8,
      xTick: (i) => `${gh.depths[i]}×`,
      xTitle: "ghost size, in multiples of the cache size (log scale)",
      series: ghSeries,
      endLabels: true,
      max: 0.5,
      tip: (i) => ({
        title: `Ghost of ${gh.depths[i]}× the cache`,
        rows: ghSeries.map((s) => ({ cls: s.cls, label: s.label, value: fewer(s.values[i]) })),
        note: "Best fixed design per cell, mean over its size class.",
      }),
      aria: "Fewer misses than S3-FIFO versus ghost size, by cache size class",
    }),
  );
  legend(document.querySelector("#fig-ghost .legend-slot"), ghSeries.map((s) => ({ ...s, line: true })));
  dataTable(
    document.querySelector("#fig-ghost .table-slot"),
    "Train-long, best fixed design per cell. Fewer misses than S3-FIFO (negative = more).",
    ["cache size class"].concat(gh.depths.map((d) => `ghost ${d}×`)),
    ghSeries.map((s) => [s.label].concat(s.values.map((v) => pct(v)))),
  );

  // Fig 8: board progress (static objective)
  const bd = D.boards;
  mount("fig-boards", (plot, w) =>
    lines(plot, w, {
      xs: bd.map((_, i) => i),
      xPad: 24,
      xTick: (i) => bd[i].stage,
      series: [{ label: "pool", cls: "c1", values: bd.map((r) => red(r.mean)) }],
      min: 2,
      height: 240,
      refs: [{ value: red(D.train3L), label: "3L-Cache", cls: "c2", right: true }],
      tip: (i) => ({ title: bd[i].stage, rows: [{ cls: "c1", label: "pool, static pick", value: fewer(red(bd[i].mean)) }] }),
      aria: "Board pool score by architect generation",
    }),
  );
  dataTable(
    document.querySelector("#fig-boards .table-slot"),
    "Train-long, pool scored as a per-workload static pick after the takeover. Mean fewer misses than S3-FIFO.",
    ["board", "mean"],
    bd.map((r) => [r.stage, pct(red(r.mean))]),
  );

  // CPU table (always visible: four rows)
  dataTable(
    document.getElementById("cpu-table"),
    "One train-long trace (15.4M requests), single thread. Run time relative to S3-FIFO in the same simulator.",
    ["cache size (objects)", "3L-Cache / S3-FIFO (libCacheSim)", "v1 / S3-FIFO (vqsim)"],
    D.cpu.map((r) => [r.capacity.toLocaleString("en-US"), `${r["3L"].toFixed(1)}×`, `${r.v1.toFixed(1)}×`]),
    { plain: true },
  );

  // Theme toggle: auto -> light -> dark; ?theme=dark|light also works (for screenshots and links)
  const root = document.documentElement;
  const btn = document.getElementById("theme-toggle");
  const modes = ["auto", "light", "dark"];
  const apply = (mode) => {
    if (mode === "auto") root.removeAttribute("data-theme");
    else root.setAttribute("data-theme", mode);
    btn.textContent = `Theme: ${mode}`;
  };
  let mode = new URLSearchParams(location.search).get("theme") || localStorage.getItem("cachelab-theme") || "auto";
  if (!modes.includes(mode)) mode = "auto";
  apply(mode);
  btn.addEventListener("click", () => {
    mode = modes[(modes.indexOf(mode) + 1) % modes.length];
    localStorage.setItem("cachelab-theme", mode);
    apply(mode);
  });
})();
