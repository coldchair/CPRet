/* In-page search: progress/ETA, pagination, OJ filter, view toggles, cluster filter. */
(function () {
  "use strict";

  function $(id) { return document.getElementById(id); }
  var dataNode = $("page-data");
  if (!dataNode) return;
  var PAGE = JSON.parse(dataNode.textContent || "{}");
  var I = window.CPRET_I18N || {};
  var TOKEN_LABEL = I.token_count_label || "Token";
  var VIEW_KEYS = { mmr: 1, dedup: 1, cluster: 1 };

  var form = $("search-form");
  var box = $("query-box");
  var resultsRoot = $("search-results");
  var submitBtn = $("search-submit");
  var inMmr = $("param-mmr");
  var inDedup = $("param-dedup");
  var inCluster = $("param-cluster");

  var searchElapsedTimer = null;
  var searchLoading = false;
  var searchSeq = 0;
  var activeCtrl = null;

  // ---------- helpers ----------
  function ojBoxes() { return Array.prototype.slice.call(document.querySelectorAll(".oj-checkbox")); }

  function buildSearchParams(page) {
    var params = new URLSearchParams();
    var boxes = ojBoxes();
    var allChecked = boxes.length > 0 && boxes.every(function (b) { return b.checked; });
    new FormData(form).forEach(function (v, k) {
      if (VIEW_KEYS[k] && v === "") return;
      if (k === "oj" && allChecked) return; // server default = all OJs
      params.append(k, v);
    });
    params.set("page", String(page));
    return params;
  }

  function syncBrowserUrl(params, push) {
    var url = PAGE.indexUrl + (params.toString() ? "?" + params.toString() : "");
    if (push) history.pushState({ cpret: 1 }, "", url);
    else history.replaceState({ cpret: 1 }, "", url);
  }

  function formatSeconds(ms) {
    if (ms === null || ms === undefined || Number.isNaN(ms)) return "--";
    return (ms / 1000).toFixed(2);
  }

  function setSearchSubmitDisabled(disabled) {
    if (!submitBtn) return;
    submitBtn.disabled = disabled;
    submitBtn.classList.toggle("is-loading", disabled);
  }

  // ---------- progress ----------
  function showSearchLoading() {
    var wrapper = $("search-progress-wrapper");
    var bar = $("search-progress-bar");
    var elapsedNode = $("search-elapsed-seconds");
    var estimatedNode = $("search-estimated-seconds");
    if (!wrapper || !bar || !elapsedNode || !estimatedNode) return;

    wrapper.classList.remove("is-hidden");
    wrapper.classList.add("is-running");
    bar.setAttribute("aria-busy", "true");
    bar.removeAttribute("aria-valuenow");
    elapsedNode.textContent = "0.00";
    estimatedNode.textContent = "--";

    var begin = Date.now();
    if (searchElapsedTimer) clearInterval(searchElapsedTimer);
    searchElapsedTimer = setInterval(function () {
      elapsedNode.textContent = ((Date.now() - begin) / 1000).toFixed(2);
    }, 100);
  }

  function finishSearchLoading(meta) {
    var wrapper = $("search-progress-wrapper");
    var bar = $("search-progress-bar");
    var elapsedNode = $("search-elapsed-seconds");
    var estimatedNode = $("search-estimated-seconds");
    var tokenNode = $("search-token-count");
    if (searchElapsedTimer) {
      clearInterval(searchElapsedTimer);
      searchElapsedTimer = null;
    }
    if (!wrapper || !bar || !elapsedNode || !estimatedNode) return;

    wrapper.classList.remove("is-running");
    bar.setAttribute("aria-busy", "false");
    bar.setAttribute("aria-valuenow", "100");

    if (meta) {
      elapsedNode.textContent = formatSeconds(meta.elapsed);
      estimatedNode.textContent = formatSeconds(meta.estimated_ms);
      if (tokenNode && meta.query_token_count !== undefined) {
        tokenNode.textContent = TOKEN_LABEL + ": " + meta.query_token_count;
      }
    }
  }

  function hideSearchProgress() {
    var wrapper = $("search-progress-wrapper");
    if (searchElapsedTimer) {
      clearInterval(searchElapsedTimer);
      searchElapsedTimer = null;
    }
    if (wrapper) wrapper.classList.add("is-hidden");
  }

  function prefetchEta(query) {
    return fetch(PAGE.etaUrl, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ q: query })
    })
      .then(function (resp) { return resp.ok ? resp.json() : null; })
      .then(function (data) {
        if (!data) return;
        var tokenNode = $("search-token-count");
        var estimatedNode = $("search-estimated-seconds");
        if (tokenNode && data.token_count !== undefined) tokenNode.textContent = TOKEN_LABEL + ": " + data.token_count;
        if (estimatedNode && searchLoading) estimatedNode.textContent = formatSeconds(data.estimated_ms);
      })
      .catch(function () { /* optional hint only */ });
  }

  // ---------- insights bridge ----------
  function problemLink(pid) {
    var base = String(PAGE.problemUrl || "/p/0").replace(/0$/, String(pid));
    var p = new URLSearchParams();
    p.set("lang", PAGE.lang || "zh");
    var q = (box.value || "").trim();
    if (q) p.set("q", q);
    return base + "?" + p.toString();
  }

  function renderInsights(ins, params) {
    if (!window.CPRetInsights) return;
    window.CPRetInsights.render(ins, params, {
      onCluster: function (cid) {
        inCluster.value = (cid === null || cid === undefined) ? "" : String(cid);
        runSearch(1, { count: false, push: false });
      },
      linkFor: problemLink
    });
  }

  function applyParams(params) {
    if (!params) return;
    inMmr.value = params.mmr ? "1" : "";
    inDedup.value = params.dedup ? "" : "0";
    inCluster.value = (params.cluster === null || params.cluster === undefined) ? "" : String(params.cluster);
  }

  // ---------- main search ----------
  function runSearch(page, opts) {
    opts = opts || {};
    var count = opts.count !== false;
    var push = opts.push !== false;
    var query = (box.value || "").trim();
    if (!query || searchLoading) return;

    var params = buildSearchParams(page);
    var reqParams = new URLSearchParams(params);
    if (!count) reqParams.set("count", "0");

    searchLoading = true;
    setSearchSubmitDisabled(true);
    showSearchLoading();
    if (count) prefetchEta(query);

    fetch(PAGE.searchApi + "?" + reqParams.toString(), { headers: { "Accept": "application/json" } })
      .then(function (resp) {
        if (!resp.ok) throw new Error("search failed");
        return resp.json();
      })
      .then(function (data) {
        if (resultsRoot) resultsRoot.innerHTML = data.html || "";
        applyParams(data.params);
        renderInsights(data.insights, data.params);
        finishSearchLoading(data);
        syncBrowserUrl(buildSearchParams(page), push);
        if (opts.scroll && resultsRoot) resultsRoot.scrollIntoView({ behavior: "smooth", block: "start" });
      })
      .catch(function () {
        hideSearchProgress();
        window.location.href = PAGE.indexUrl + "?" + params.toString();
      })
      .finally(function () {
        searchLoading = false;
        setSearchSubmitDisabled(false);
      });
  }

  // ---------- OJ filter ----------
  function updateOjCount() {
    var node = $("oj-count");
    if (!node) return;
    var boxes = ojBoxes();
    var n = boxes.filter(function (b) { return b.checked; }).length;
    node.textContent = (I.selected_count || "{n}/{total}").replace("{n}", n).replace("{total}", boxes.length);
  }
  function setAllOJs(checked) {
    ojBoxes().forEach(function (b) { b.checked = checked; });
    updateOjCount();
  }

  // ---------- events ----------
  if (form) {
    form.addEventListener("submit", function (e) {
      e.preventDefault();
      inCluster.value = ""; // clusters change with a new query
      runSearch(1);
    });
  }

  if (box) {
    box.addEventListener("keydown", function (e) {
      if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
        e.preventDefault();
        if (form.requestSubmit) form.requestSubmit();
        else form.dispatchEvent(new Event("submit", { cancelable: true }));
      }
    });
  }

  var tplBtn = $("fill-template");
  if (tplBtn) {
    tplBtn.addEventListener("click", function () {
      box.value = (PAGE.templates || {})[PAGE.lang] || "";
      box.focus();
    });
  }
  var allBtn = $("oj-all"), noneBtn = $("oj-none");
  if (allBtn) allBtn.addEventListener("click", function () { setAllOJs(true); });
  if (noneBtn) noneBtn.addEventListener("click", function () { setAllOJs(false); });
  ojBoxes().forEach(function (b) { b.addEventListener("change", updateOjCount); });
  updateOjCount();

  if (resultsRoot) {
    resultsRoot.addEventListener("click", function (e) {
      var clear = e.target.closest("[data-clear-cluster]");
      if (clear) {
        e.preventDefault();
        inCluster.value = "";
        runSearch(1, { count: false, push: false });
        return;
      }
      var link = e.target.closest("a.page-link[data-page]");
      if (!link) return;
      var item = link.closest(".page-item");
      if (item && item.classList.contains("disabled")) return;
      e.preventDefault();
      var page = parseInt(link.getAttribute("data-page"), 10);
      if (!Number.isFinite(page) || page < 1) return;
      runSearch(page, { count: false, scroll: true });
    });

    resultsRoot.addEventListener("change", function (e) {
      var t = e.target;
      if (!t || !t.matches("input[data-toggle]")) return;
      var which = t.getAttribute("data-toggle");
      if (which === "mmr") inMmr.value = t.checked ? "1" : "";
      if (which === "dedup") {
        inDedup.value = t.checked ? "" : "0";
        inCluster.value = ""; // cluster ids may change
      }
      runSearch(1, { count: false, push: false });
    });
  }

  window.addEventListener("popstate", function (e) {
    if (e.state && e.state.cpret) window.location.reload();
  });

  // ---------- SSR initial state ----------
  if (PAGE.hasQuery) {
    finishSearchLoading(PAGE.meta);
    var wrapper = $("search-progress-wrapper");
    if (wrapper) wrapper.classList.remove("is-hidden");
    renderInsights(PAGE.insights, PAGE.params);
  }
  history.replaceState({ cpret: 1 }, "", window.location.href);
})();
