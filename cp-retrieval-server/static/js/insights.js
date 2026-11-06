/* Result insights: topic-group cards + 2D SVG scatter. All dynamic text uses textContent. */
(function () {
  "use strict";

  var SVG_NS = "http://www.w3.org/2000/svg";
  var I = window.CPRET_I18N || {};
  var W = 480, H = 340, PAD = 26;

  function $(id) { return document.getElementById(id); }
  function el(tag, cls, text) {
    var n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text !== undefined && text !== null) n.textContent = text;
    return n;
  }
  function svgEl(tag, attrs) {
    var n = document.createElementNS(SVG_NS, tag);
    if (attrs) Object.keys(attrs).forEach(function (k) { n.setAttribute(k, attrs[k]); });
    return n;
  }
  function color(c) { return "var(--cl-" + ((c || 0) % 8) + ")"; }
  function fmt(x) { return Number(x).toFixed(4); }

  var state = { ins: null, params: null, opts: {} };

  function setHighlight(cid) {
    var svg = $("scatter");
    if (!svg) return;
    var active = state.params && state.params.cluster;
    var target = (cid === null || cid === undefined) ? active : cid;
    var pts = svg.querySelectorAll(".pt");
    if (target === null || target === undefined) {
      svg.classList.remove("dim");
      pts.forEach(function (p) { p.classList.remove("hl"); });
      return;
    }
    svg.classList.add("dim");
    pts.forEach(function (p) {
      p.classList.toggle("hl", Number(p.getAttribute("data-c")) === Number(target));
    });
  }

  function renderClusters(clusters, active) {
    var list = $("cluster-list");
    list.textContent = "";
    list.classList.toggle("has-active", active !== null && active !== undefined);
    if (!clusters || !clusters.length) {
      list.appendChild(el("div", "empty-hint", I.clusters_empty || ""));
      return;
    }
    var maxScore = Math.max.apply(null, clusters.map(function (c) { return c.max_score; })) || 1;
    var minMean = Math.min.apply(null, clusters.map(function (c) { return c.mean_score; }));
    clusters.forEach(function (c, i) {
      var card = el("button", "cluster-card");
      card.type = "button";
      card.style.setProperty("--cc", color(c.id));
      card.style.animationDelay = (i * 50) + "ms";
      card.setAttribute("data-cluster", String(c.id));
      card.setAttribute("aria-pressed", String(active === c.id));
      if (active === c.id) card.classList.add("active");

      var top = el("div", "cluster-top");
      var label = el("span", "cluster-label", c.label || "—");
      label.title = c.label || "";
      top.appendChild(label);
      top.appendChild(el("span", "cluster-size", c.size + " " + (I.problems_unit || "")));
      card.appendChild(top);

      var meta = el("div", "cluster-meta");
      meta.appendChild(el("span", "", I.avg_sim || "avg"));
      var meter = el("span", "cluster-meter");
      var fill = el("span");
      // Emphasise differences between groups: map [minMean*0.9, maxScore] -> [8%, 100%].
      var lo = minMean * 0.9, frac = (c.mean_score - lo) / Math.max(1e-6, maxScore - lo);
      fill.style.width = Math.max(8, Math.min(100, frac * 100)).toFixed(1) + "%";
      meter.appendChild(fill);
      meta.appendChild(meter);
      meta.appendChild(el("span", "num", fmt(c.mean_score)));
      meta.appendChild(el("span", "", "· " + (I.max_sim || "max")));
      meta.appendChild(el("span", "num", fmt(c.max_score)));
      card.appendChild(meta);

      var ex = (c.top_titles || []).filter(function (t) { return t && t !== c.label; }).slice(0, 2);
      if (ex.length) {
        var ul = el("ul", "cluster-examples");
        ex.forEach(function (t) { ul.appendChild(el("li", "", t)); });
        card.appendChild(ul);
      }

      card.addEventListener("click", function () {
        if (state.opts.onCluster) state.opts.onCluster(active === c.id ? null : c.id);
      });
      card.addEventListener("mouseenter", function () { setHighlight(c.id); });
      card.addEventListener("mouseleave", function () { setHighlight(null); });
      card.addEventListener("focus", function () { setHighlight(c.id); });
      card.addEventListener("blur", function () { setHighlight(null); });
      list.appendChild(card);
    });
  }

  function sx(x) { return PAD + (x + 1) / 2 * (W - 2 * PAD); }
  function sy(y) { return H - PAD - (y + 1) / 2 * (H - 2 * PAD); }

  function starPath(cx, cy, R, r) {
    var d = "";
    for (var i = 0; i < 10; i++) {
      var rad = (i % 2 === 0) ? R : r;
      var a = -Math.PI / 2 + i * Math.PI / 5;
      d += (i ? "L" : "M") + (cx + rad * Math.cos(a)).toFixed(1) + " " + (cy + rad * Math.sin(a)).toFixed(1);
    }
    return d + "Z";
  }

  function renderScatter(ins) {
    var svg = $("scatter");
    while (svg.firstChild) svg.removeChild(svg.firstChild);
    svg.setAttribute("viewBox", "0 0 " + W + " " + H);

    var grid = svgEl("g", { "class": "grid" });
    for (var g = 1; g < 4; g++) {
      grid.appendChild(svgEl("line", { x1: PAD + g * (W - 2 * PAD) / 4, x2: PAD + g * (W - 2 * PAD) / 4, y1: PAD / 2, y2: H - PAD / 2 }));
      grid.appendChild(svgEl("line", { y1: PAD + g * (H - 2 * PAD) / 4, y2: PAD + g * (H - 2 * PAD) / 4, x1: PAD / 2, x2: W - PAD / 2 }));
    }
    svg.appendChild(grid);

    var qp = ins.query_point;
    if (qp) {
      var qx = sx(qp.x), qy = sy(qp.y);
      svg.appendChild(svgEl("circle", { "class": "q-ring", cx: qx, cy: qy, r: 34 }));
      svg.appendChild(svgEl("circle", { "class": "q-ring", cx: qx, cy: qy, r: 70 }));
    }

    var pts = (ins.points || []).slice();
    var smin = Infinity, smax = -Infinity;
    pts.forEach(function (p) { smin = Math.min(smin, p.score); smax = Math.max(smax, p.score); });
    var span = Math.max(1e-6, smax - smin);
    pts.sort(function (a, b) { return a.score - b.score; }); // high scores on top
    var hasClusters = ins.clusters && ins.clusters.length > 0;
    var layer = svgEl("g", {});
    pts.forEach(function (p, i) {
      var r = 3.6 + 5.2 * (p.score - smin) / span;
      var c = svgEl("circle", {
        "class": "pt",
        cx: sx(p.x).toFixed(1),
        cy: sy(p.y).toFixed(1),
        r: r.toFixed(2),
        "data-pid": String(p.pid),
        "data-c": String(hasClusters ? p.cluster : 0),
        tabindex: "0",
        role: "link"
      });
      c.setAttribute("aria-label", (p.title || "") + " " + fmt(p.score));
      c.style.fill = color(hasClusters ? p.cluster : 0);
      c.style.animationDelay = (i * 5) + "ms";
      c.__data = p;
      layer.appendChild(c);
    });
    svg.appendChild(layer);

    if (qp) {
      var x0 = sx(qp.x), y0 = sy(qp.y);
      svg.appendChild(svgEl("path", { "class": "q-star", d: starPath(x0, y0, 10, 4.4) }));
      var lbl = svgEl("text", { "class": "q-label", x: x0 + 13, y: y0 + 4 });
      lbl.textContent = I.query_point || "Query";
      svg.appendChild(lbl);
    }
    setHighlight(null);
  }

  function clusterById(id) {
    var cs = (state.ins && state.ins.clusters) || [];
    for (var i = 0; i < cs.length; i++) if (cs[i].id === id) return cs[i];
    return null;
  }

  function showTip(evt, p) {
    var tip = $("scatter-tip"), wrap = $("scatter-wrap");
    tip.textContent = "";
    tip.appendChild(el("div", "tt-title", p.title || "—"));
    var row = el("div", "tt-row");
    row.appendChild(el("span", "", "score"));
    row.appendChild(el("span", "num", fmt(p.score)));
    if (p.dups) row.appendChild(el("span", "", "· +" + p.dups));
    tip.appendChild(row);
    var c = clusterById(p.cluster);
    if (c) {
      var r2 = el("div", "tt-row");
      var dot = el("span", "tt-dot");
      dot.style.background = color(c.id);
      r2.appendChild(dot);
      r2.appendChild(el("span", "", c.label));
      tip.appendChild(r2);
    }
    var box = wrap.getBoundingClientRect();
    var x = evt.clientX - box.left + 14, y = evt.clientY - box.top + 14;
    tip.classList.add("show");
    var tw = tip.offsetWidth, th = tip.offsetHeight;
    if (x + tw > box.width) x = Math.max(0, evt.clientX - box.left - tw - 14);
    if (y + th > box.height) y = Math.max(0, evt.clientY - box.top - th - 14);
    tip.style.left = x + "px";
    tip.style.top = y + "px";
  }

  function hideTip() { var t = $("scatter-tip"); if (t) t.classList.remove("show"); }

  function bindScatterEvents() {
    var svg = $("scatter");
    if (!svg || svg.__bound) return;
    svg.__bound = true;
    svg.addEventListener("mousemove", function (e) {
      var t = e.target;
      if (t && t.classList && t.classList.contains("pt") && t.__data) showTip(e, t.__data);
      else hideTip();
    });
    svg.addEventListener("mouseleave", hideTip);
    function open(t) {
      if (!t || !t.__data || !state.opts.linkFor) return;
      window.open(state.opts.linkFor(t.__data.pid), "_blank", "noopener");
    }
    svg.addEventListener("click", function (e) {
      if (e.target && e.target.classList && e.target.classList.contains("pt")) open(e.target);
    });
    svg.addEventListener("keydown", function (e) {
      if ((e.key === "Enter" || e.key === " ") && e.target.classList.contains("pt")) {
        e.preventDefault();
        open(e.target);
      }
    });
  }

  function render(ins, params, opts) {
    state.ins = ins;
    state.params = params || {};
    state.opts = opts || state.opts || {};
    var sec = $("insights");
    if (!sec) return;
    if (!ins || !ins.points || ins.points.length < 3) {
      sec.classList.add("is-hidden");
      return;
    }
    sec.classList.remove("is-hidden");
    var hint = $("insights-hint");
    if (hint) hint.textContent = (I.insights_hint || "").replace("{k}", String(ins.k));
    var active = state.params.cluster;
    if (active === undefined) active = null;
    renderClusters(ins.clusters, active);
    renderScatter(ins);
    bindScatterEvents();
  }

  function hide() {
    var sec = $("insights");
    if (sec) sec.classList.add("is-hidden");
  }

  window.CPRetInsights = { render: render, hide: hide };
})();
