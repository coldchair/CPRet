/* Shared: GitHub star badge (cached 6h) + language switch keeping current URL. */
(function () {
  "use strict";

  var SVG_NS = "http://www.w3.org/2000/svg";
  var GITHUB_REPO = "coldchair/CPRet";
  var CACHE_DURATION_HOURS = 6;
  var CACHE_KEY = "github_stars_data";
  var box = document.getElementById("github-stars-container");

  function starIcon() {
    var svg = document.createElementNS(SVG_NS, "svg");
    svg.setAttribute("width", "13");
    svg.setAttribute("height", "13");
    svg.setAttribute("viewBox", "0 0 16 16");
    svg.setAttribute("fill", "currentColor");
    svg.setAttribute("aria-hidden", "true");
    var p = document.createElementNS(SVG_NS, "path");
    p.setAttribute("d", "M3.612 15.443c-.386.198-.824-.149-.746-.592l.83-4.73L.173 6.765c-.329-.314-.158-.888.283-.95l4.898-.696L7.538.792c.197-.39.73-.39.927 0l2.184 4.327 4.898.696c.441.062.612.636.282.95l-3.522 3.356.83 4.73c.078.443-.36.79-.746.592L8 13.187l-4.389 2.256z");
    svg.appendChild(p);
    return svg;
  }

  function displayStars(count) {
    if (!box || count === null || count === undefined || !isFinite(Number(count))) return;
    box.textContent = "";
    box.title = "GitHub Stars";
    box.appendChild(starIcon());
    box.appendChild(document.createTextNode(Number(count).toLocaleString()));
  }

  function readCache() {
    try {
      var raw = localStorage.getItem(CACHE_KEY);
      return raw ? JSON.parse(raw) : null;
    } catch (_) {
      return null;
    }
  }

  if (box) {
    var cached = readCache();
    var fresh = cached && typeof cached.timestamp === "number" &&
      (Date.now() - cached.timestamp) / 3600000 < CACHE_DURATION_HOURS;
    if (fresh) {
      displayStars(cached.stars);
    } else {
      var ctrl = typeof AbortController !== "undefined" ? new AbortController() : null;
      var to = ctrl ? setTimeout(function () { ctrl.abort(); }, 6000) : null;
      fetch("https://api.github.com/repos/" + GITHUB_REPO, ctrl ? { signal: ctrl.signal } : {})
        .then(function (r) {
          if (!r.ok) throw new Error("Network response was not ok");
          return r.json();
        })
        .then(function (data) {
          displayStars(data.stargazers_count);
          try {
            localStorage.setItem(CACHE_KEY, JSON.stringify({ stars: data.stargazers_count, timestamp: Date.now() }));
          } catch (_) { /* storage may be unavailable */ }
        })
        .catch(function () {
          if (cached) displayStars(cached.stars);
        })
        .finally(function () { if (to) clearTimeout(to); });
    }
  }

  // Language switch: use the live URL (in-page search updates it via history API).
  document.querySelectorAll(".lang-link").forEach(function (a) {
    a.addEventListener("click", function () {
      var u = new URL(window.location.href);
      u.searchParams.delete("q");
      u.searchParams.delete("count");
      u.searchParams.set("lang", a.getAttribute("data-lang"));
      a.setAttribute("href", u.pathname + u.search);
    });
  });
})();
