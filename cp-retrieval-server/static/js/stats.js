/* Daily search stats chart: native SVG (area + line + EMA), hover crosshair, range switch. */
(function () {
  "use strict";

  var SVG_NS = "http://www.w3.org/2000/svg";
  var I = window.CPRET_I18N || {};
  var EMA_WEIGHT = 0.85;
  var HEIGHT = 340;
  var M = { t: 16, r: 18, b: 34, l: 52 };

  var box = document.getElementById("chart");
  var tip = document.getElementById("chart-tip");
  var dataNode = document.getElementById("stats-data");
  if (!box || !dataNode) return;

  var raw = JSON.parse(dataNode.textContent || "{}");
  var allDates = raw.dates || [];
  var allCounts = (raw.counts || []).map(Number);

  // TensorBoard-like smoothing (same as before: weight 0.85), computed on the full series.
  function smooth(data, weight) {
    if (!data.length) return [];
    var last = data[0], out = [last];
    for (var i = 1; i < data.length; i++) {
      var sm = last * weight + data[i] * (1 - weight);
      out.push(sm);
      last = sm;
    }
    return out;
  }
  var allEma = smooth(allCounts, EMA_WEIGHT);
  var allTimes = allDates.map(function (d) { return Date.parse(d + "T00:00:00"); });

  var range = 0;

  function svgEl(tag, attrs) {
    var n = document.createElementNS(SVG_NS, tag);
    if (attrs) Object.keys(attrs).forEach(function (k) { n.setAttribute(k, attrs[k]); });
    return n;
  }

  function niceStep(span, ticks) {
    var raw = span / Math.max(1, ticks);
    var mag = Math.pow(10, Math.floor(Math.log10(raw || 1)));
    var norm = raw / mag;
    var step = norm < 1.5 ? 1 : norm < 3 ? 2 : norm < 7 ? 5 : 10;
    return Math.max(1, step * mag);
  }

  function pad2(n) { return n < 10 ? "0" + n : String(n); }
  function fmtDate(t, long) {
    var d = new Date(t);
    return long ? d.getFullYear() + "-" + pad2(d.getMonth() + 1) : pad2(d.getMonth() + 1) + "-" + pad2(d.getDate());
  }

  function slice() {
    var n = allDates.length;
    var start = range > 0 ? Math.max(0, n - range) : 0;
    return {
      dates: allDates.slice(start),
      times: allTimes.slice(start),
      counts: allCounts.slice(start),
      ema: allEma.slice(start)
    };
  }

  function clearSvg() {
    var old = box.querySelector("svg");
    if (old) box.removeChild(old);
    var empty = box.querySelector(".empty-hint");
    if (empty) box.removeChild(empty);
  }

  function draw(animate) {
    clearSvg();
    var d = slice();
    if (!d.dates.length) {
      var e = document.createElement("div");
      e.className = "empty-hint";
      e.textContent = I.no_data || "No data";
      box.appendChild(e);
      return;
    }
    var width = Math.max(320, box.clientWidth);
    var iw = width - M.l - M.r, ih = HEIGHT - M.t - M.b;

    var t0 = d.times[0], t1 = d.times[d.times.length - 1];
    if (t1 === t0) { t0 -= 86400000; t1 += 86400000; }
    var ymaxRaw = Math.max.apply(null, d.counts.concat(d.ema).concat([1]));
    var ystep = niceStep(ymaxRaw, 5);
    var ymax = Math.ceil(ymaxRaw / ystep) * ystep;

    function X(t) { return M.l + (t - t0) / (t1 - t0) * iw; }
    function Y(v) { return M.t + ih - v / ymax * ih; }

    var svg = svgEl("svg", { viewBox: "0 0 " + width + " " + HEIGHT, preserveAspectRatio: "none", role: "img" });
    svg.style.height = HEIGHT + "px";

    var defs = svgEl("defs");
    var grad = svgEl("linearGradient", { id: "areaGrad", x1: "0", y1: "0", x2: "0", y2: "1" });
    grad.appendChild(svgEl("stop", { offset: "0%", "stop-color": "#0061cc", "stop-opacity": "0.30" }));
    grad.appendChild(svgEl("stop", { offset: "100%", "stop-color": "#0061cc", "stop-opacity": "0.02" }));
    defs.appendChild(grad);
    svg.appendChild(defs);

    // Y grid + labels
    var axis = svgEl("g", { "class": "axis" });
    for (var v = 0; v <= ymax + 1e-9; v += ystep) {
      var y = Y(v);
      axis.appendChild(svgEl("line", { "class": "gridline", x1: M.l, x2: width - M.r, y1: y, y2: y }));
      var tl = svgEl("text", { x: M.l - 8, y: y + 4, "text-anchor": "end" });
      tl.textContent = v >= 1000 ? (v / 1000).toFixed(v % 1000 ? 1 : 0) + "k" : String(v);
      axis.appendChild(tl);
    }
    // X labels
    var spanDays = (t1 - t0) / 86400000;
    var longFmt = spanDays > 200;
    var nTicks = Math.max(2, Math.floor(iw / 90));
    var n = d.times.length;
    for (var k = 0; k < nTicks; k++) {
      var idx = Math.round(k * (n - 1) / Math.max(1, nTicks - 1));
      var tx = X(d.times[idx]);
      var xl = svgEl("text", { x: tx, y: HEIGHT - 10, "text-anchor": k === 0 ? "start" : (k === nTicks - 1 ? "end" : "middle") });
      xl.textContent = fmtDate(d.times[idx], longFmt);
      axis.appendChild(xl);
      if (n === 1) break;
    }
    svg.appendChild(axis);

    // Paths
    var line = "", area = "", ema = "";
    for (var i = 0; i < n; i++) {
      var px = X(d.times[i]).toFixed(1);
      line += (i ? "L" : "M") + px + " " + Y(d.counts[i]).toFixed(1);
      ema += (i ? "L" : "M") + px + " " + Y(d.ema[i]).toFixed(1);
    }
    area = line + "L" + X(d.times[n - 1]).toFixed(1) + " " + Y(0) + "L" + X(d.times[0]).toFixed(1) + " " + Y(0) + "Z";
    svg.appendChild(svgEl("path", { "class": "area", d: area }));
    var linePath = svgEl("path", { "class": "line", d: line });
    svg.appendChild(linePath);
    var emaPath = svgEl("path", { "class": "ema", d: ema });
    svg.appendChild(emaPath);
    if (n === 1) {
      svg.appendChild(svgEl("circle", { cx: X(d.times[0]), cy: Y(d.counts[0]), r: 4, fill: "#0061cc" }));
    }

    // Hover layer
    var cross = svgEl("line", { "class": "cross", y1: M.t, y2: M.t + ih, visibility: "hidden" });
    var dot1 = svgEl("circle", { "class": "hover-dot", r: 4.5, stroke: "#0061cc", visibility: "hidden" });
    var dot2 = svgEl("circle", { "class": "hover-dot", r: 4, stroke: "#eab308", visibility: "hidden" });
    svg.appendChild(cross);
    svg.appendChild(dot1);
    svg.appendChild(dot2);
    var overlay = svgEl("rect", { x: M.l, y: M.t, width: iw, height: ih, fill: "transparent" });
    svg.appendChild(overlay);

    function nearest(t) {
      var lo = 0, hi = n - 1;
      while (lo < hi) {
        var mid = (lo + hi) >> 1;
        if (d.times[mid] < t) lo = mid + 1; else hi = mid;
      }
      if (lo > 0 && Math.abs(d.times[lo - 1] - t) < Math.abs(d.times[lo] - t)) lo -= 1;
      return lo;
    }

    function onMove(evt) {
      var rect = svg.getBoundingClientRect();
      var sxScale = width / rect.width;
      var mx = (evt.clientX - rect.left) * sxScale;
      var t = t0 + (mx - M.l) / iw * (t1 - t0);
      var j = nearest(t);
      var cx = X(d.times[j]);
      [cross, dot1, dot2].forEach(function (nd) { nd.setAttribute("visibility", "visible"); });
      cross.setAttribute("x1", cx);
      cross.setAttribute("x2", cx);
      dot1.setAttribute("cx", cx);
      dot1.setAttribute("cy", Y(d.counts[j]));
      dot2.setAttribute("cx", cx);
      dot2.setAttribute("cy", Y(d.ema[j]));

      tip.textContent = "";
      var tt = document.createElement("div");
      tt.className = "tt-title";
      tt.textContent = d.dates[j];
      tip.appendChild(tt);
      [[I.search_count || "Count", d.counts[j], "#0061cc", 0], [I.moving_average || "EMA", d.ema[j], "#eab308", 1]].forEach(function (r) {
        var row = document.createElement("div");
        row.className = "tt-row";
        var dt = document.createElement("span");
        dt.className = "tt-dot";
        dt.style.background = r[2];
        var lab = document.createElement("span");
        lab.textContent = r[0];
        var num = document.createElement("span");
        num.className = "num";
        num.textContent = r[3] ? r[1].toFixed(1) : String(r[1]);
        row.appendChild(dt);
        row.appendChild(lab);
        row.appendChild(num);
        tip.appendChild(row);
      });
      tip.classList.add("show");
      var bx = box.getBoundingClientRect();
      var left = (cx / sxScale) + 14;
      if (left + tip.offsetWidth > bx.width) left = (cx / sxScale) - tip.offsetWidth - 14;
      tip.style.left = Math.max(0, left) + "px";
      tip.style.top = (M.t + 8) + "px";
    }
    function onLeave() {
      [cross, dot1, dot2].forEach(function (nd) { nd.setAttribute("visibility", "hidden"); });
      tip.classList.remove("show");
    }
    overlay.addEventListener("mousemove", onMove);
    overlay.addEventListener("touchstart", function (e) { if (e.touches[0]) onMove(e.touches[0]); }, { passive: true });
    overlay.addEventListener("touchmove", function (e) { if (e.touches[0]) onMove(e.touches[0]); }, { passive: true });
    overlay.addEventListener("mouseleave", onLeave);

    box.insertBefore(svg, tip);

    if (animate && linePath.getTotalLength) {
      var len = Math.ceil(linePath.getTotalLength());
      linePath.style.strokeDasharray = len;
      linePath.style.setProperty("--len", len);
      linePath.classList.add("draw");
      linePath.addEventListener("animationend", function () { linePath.style.strokeDasharray = ""; }, { once: true });
    }
  }

  // Range switch
  var seg = document.getElementById("range-seg");
  if (seg) {
    seg.addEventListener("click", function (e) {
      var b = e.target.closest("button[data-range]");
      if (!b) return;
      range = parseInt(b.getAttribute("data-range"), 10) || 0;
      seg.querySelectorAll("button").forEach(function (x) { x.classList.toggle("active", x === b); });
      draw(true);
    });
  }

  draw(true);

  var lastW = box.clientWidth, raf = 0;
  function onResize() {
    if (Math.abs(box.clientWidth - lastW) < 2) return;
    lastW = box.clientWidth;
    cancelAnimationFrame(raf);
    raf = requestAnimationFrame(function () { draw(false); });
  }
  if (typeof ResizeObserver !== "undefined") new ResizeObserver(onResize).observe(box);
  else window.addEventListener("resize", onResize);
})();
