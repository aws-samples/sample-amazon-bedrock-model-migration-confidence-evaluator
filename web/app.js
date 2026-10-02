// ModelShift SPA — vanilla JS, no build step. Wires to /api/v1.
const API = "/api/v1";
const $ = (sel, el = document) => el.querySelector(sel);
const BOOL_ATTRS = new Set(["disabled", "checked", "selected", "readonly", "required", "hidden"]);
// Empty a node without touching innerHTML (avoids the HTML-sink anti-pattern).
const clear = (el) => { if (el) el.replaceChildren(); return el; };
// Materialize a STATIC, author-controlled HTML string into DOM nodes using the
// non-executing DOMParser instead of innerHTML. <script> and inline event
// handlers are stripped so this can never be an XSS sink even if the input ever
// became dynamic. Returns a DocumentFragment ready to .append().
const frag = (htmlStr) => {
  const doc = new DOMParser().parseFromString(String(htmlStr), "text/html");
  doc.querySelectorAll("script").forEach((n) => n.remove());
  doc.querySelectorAll("*").forEach((n) => {
    [...n.attributes].forEach((a) => { if (/^on/i.test(a.name)) n.removeAttribute(a.name); });
  });
  const f = document.createDocumentFragment();
  f.append(...doc.body.childNodes);
  return f;
};
const h = (tag, attrs = {}, ...kids) => {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") e.className = v;
    else if (k === "html") e.append(frag(v));  // static markup only; parsed safely, never innerHTML
    else if (k.startsWith("on")) e.addEventListener(k.slice(2), v);
    else if (BOOL_ATTRS.has(k)) { if (v) e.setAttribute(k, ""); }  // presence = true; falsy = omit
    else if (v !== null && v !== undefined) e.setAttribute(k, v);
  }
  for (const kid of kids.flat()) if (kid != null) e.append(kid.nodeType ? kid : document.createTextNode(kid));
  return e;
};
const fmtPct = (n) => `${Math.round(n)}%`;
const money = (n) => `$${(n ?? 0).toFixed(4)}`;
const LABELS = ["compatible", "compatible_with_drift", "partial", "incompatible"];
const LABEL_TEXT = { compatible: "Compatible", compatible_with_drift: "Compatible w/ drift", partial: "Partial", incompatible: "Incompatible" };
const BAND_TEXT = { safe_drop_in: "Safe drop-in", drop_in_with_prompt_tuning: "Drop-in with prompt tuning", needs_work: "Needs work", not_a_drop_in: "Not a drop-in" };
const ENDPOINT_TEXT = { "bedrock-runtime": "Bedrock Runtime", "bedrock-mantle": "Bedrock Mantle", "litellm": "LiteLLM proxy" };

// The six compatibility dimensions (D6 tool-call only appears for tool-call goldens).
const DIMENSIONS = [
  { key: "d1_semantic", code: "D1", name: "Semantic similarity", desc: "Does the candidate mean the same thing as the original answer?" },
  { key: "d2_format", code: "D2", name: "Format / schema", desc: "Same structure (JSON validity + keys, list vs prose) so downstream parsing won't break.", hard: true },
  { key: "d3_factual", code: "D3", name: "Factual agreement", desc: "Concrete facts match — numbers, phone numbers, emails, dates.", hard: true },
  { key: "d4_verbosity", code: "D4", name: "Verbosity", desc: "Response length is comparable — not materially longer or shorter." },
  { key: "d5_instruction", code: "D5", name: "Instruction-following", desc: "Honors the system/instructions the original also had to follow." },
  { key: "d6_tool_call", code: "D6", name: "Tool-call match", desc: "Same function name + argument schema (tool-call responses only).", hard: true },
];
const JUDGE_REASON_TEXT = {
  equivalent: "Judge found the candidate interchangeable with the original.",
  missing_content: "Candidate dropped content the original included.",
  factual_divergence: "Candidate and original disagree on a concrete fact.",
  format_change: "Candidate changed the output format/structure.",
  added_verbosity: "Candidate is materially more verbose than the original.",
  instruction_violation: "Candidate did not follow the instructions.",
};
const LABEL_EXPLAIN = {
  compatible: "Drop-in interchangeable with the original response.",
  compatible_with_drift: "Same meaning, but stylistically different (length/format drift).",
  partial: "Usable but would need review before relying on it.",
  incompatible: "Would break the downstream consumer — not a safe drop-in.",
};

async function api(path, opts = {}) {
  const r = await fetch(API + path, opts);
  if (!r.ok) {
    let d; try { d = await r.json(); } catch { d = {}; }
    let msg = d?.detail?.error?.message;
    if (!msg && typeof d?.detail === "string") msg = d.detail;
    if (!msg && d?.detail) msg = JSON.stringify(d.detail);
    throw new Error(msg || `${r.status} ${r.statusText}`);
  }
  return r.status === 204 ? null : r.json();
}

// ---- state ---------------------------------------------------------------
let MODELS = null;   // /models payload
let SETTINGS = null; // /settings payload (thresholds, prices)
const state = { view: "runs", draft: null };

// ---- router --------------------------------------------------------------
function go(view, ...args) { state.view = view; render(view, ...args); document.querySelectorAll("#nav a").forEach(a => a.classList.toggle("active", a.dataset.view === view)); }
document.querySelectorAll("#nav a").forEach(a => a.addEventListener("click", () => go(a.dataset.view)));

async function boot() {
  try {
    MODELS = await api("/models"); SETTINGS = await api("/settings");
    let ver = {};
    try { ver = await api("/version"); } catch {}
    const status = $("#apiStatus");
    clear(status);
    status.append(
      h("div", {}, "API connected"),
      h("div", { title: `source ${ver.source_hash || "?"} · mtime ${ver.source_mtime || "?"} · started ${ver.started_at || "?"}` },
        "build ", h("code", {}, ver.build_id || "?")),
      h("div", { style: "font-size:10px;opacity:.7" }, "up since " + (ver.started_at ? ver.started_at.replace("T", " ").replace("+00:00", "Z") : "?")));
  } catch (e) { $("#apiStatus").textContent = "API offline"; }
  go("runs");
}

function topbar(title) {
  const acct = "bedrock · us-east-1";
  return h("div", { class: "topbar" }, h("h2", {}, title), h("span", { class: "env" }, acct));
}

// ---- views ---------------------------------------------------------------
async function render(view, arg) {
  const main = $("#main");
  clear(main);
  if (view === "runs") return renderRuns(main);
  if (view === "new") return renderNewRun(main);
  if (view === "run") return renderRunDetail(main, arg);
  if (view === "settings") return renderSettings(main);
  if (view === "guide") return renderGuide(main);
}

async function renderRuns(main) {
  main.append(topbar("Runs"));
  let runs = [];
  try { runs = await api("/runs"); } catch {}
  if (!runs.length) {
    main.append(h("div", { class: "card" },
      h("h3", {}, "No runs yet"),
      h("p", { class: "muted" }, "Start by loading a LiteLLM log file or pointing at an S3 location."),
      h("button", { class: "btn", onclick: () => go("new") }, "New Run")));
    return;
  }
  const rows = runs.map(r => {
    const delBtn = h("button", { class: "btn ghost", style: "padding:2px 8px;font-size:12px", title: "Delete run" }, "Delete");
    delBtn.addEventListener("click", async (ev) => {
      ev.stopPropagation();
      if (!confirm(`Delete run "${r.name || r.run_id}"? This cannot be undone.`)) return;
      try { await api(`/runs/${r.run_id}`, { method: "DELETE" }); go("runs"); }
      catch (e) { alert("Delete failed: " + (e && e.message ? e.message : "error")); }
    });
    return h("tr", { onclick: () => go("run", r.run_id) },
      h("td", { class: "mono" }, r.run_id),
      h("td", {}, r.name),
      h("td", {}, r.status),
      h("td", {}, r.verdict_summary?.length
        ? h("span", {}, `${r.verdict_summary[0].model} · ${fmtPct(r.verdict_summary[0].confidence)}`)
        : h("span", { class: "muted" }, "—")),
      h("td", {}, delBtn));
  });
  main.append(h("div", { class: "card" },
    h("div", { class: "row spread" }, h("h3", {}, "All runs"), h("button", { class: "btn", onclick: () => go("new") }, "New Run")),
    h("table", {}, h("thead", {}, h("tr", {}, ...["Run", "Name", "Status", "Top candidate", ""].map(t => h("th", {}, t)))),
      h("tbody", {}, ...rows))));
}

// New Run: Source -> Config wizard
function renderNewRun(main) {
  main.append(topbar("New Run"));
  state.draft = state.draft || { name: "", step: "source", runId: null, ingestion: null, candidates: [], callPath: "bedrock", endpoint: "bedrock_runtime", effort: "medium" };
  const d = state.draft;
  if (d.step === "source") return newRunSource(main, d);
  if (d.step === "config") return newRunConfig(main, d);
}

function newRunSource(main, d) {
  const nameInput = h("input", {
    type: "text", value: d.name, placeholder: "e.g. member-services chatbot",
    oninput: e => {
      d.name = e.target.value;
      // If the run already exists (file dropped first), push the name to the backend.
      if (d.runId) {
        clearTimeout(d._nameTimer);
        d._nameTimer = setTimeout(() => {
          api(`/runs/${d.runId}`, { method: "PATCH", headers: { "content-type": "application/json" },
            body: JSON.stringify({ name: d.name }) }).catch(() => {});
        }, 400);
      }
    },
  });
  const s3Input = h("input", { type: "text", placeholder: "s3://bucket/prefix" });
  const fileInput = h("input", { type: "file", style: "display:none", onchange: e => uploadThenIngest(main, d, e.target.files[0]) });
  const dz = h("div", { class: "dropzone", onclick: () => fileInput.click() },
    "Drop a LiteLLM log file, or click to browse", fileInput);
  const preview = h("div", { id: "ingPreview" });
  main.append(h("div", { class: "card" },
    h("h3", {}, "1 · Source"),
    h("label", { class: "field" }, h("span", {}, "Run name"), nameInput),
    h("div", { class: "grid cols-2" },
      h("div", {}, h("div", { class: "muted", style: "margin-bottom:8px" }, "Upload file"), dz),
      h("div", {}, h("div", { class: "muted", style: "margin-bottom:8px" }, "or S3 location"),
        s3Input, h("button", { class: "btn secondary", style: "margin-top:8px", onclick: () => s3Ingest(main, d, s3Input.value) }, "Validate & ingest"))),
    preview));
}

async function ensureRun(d) {
  if (d.runId) return d.runId;
  const r = await api("/runs", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ name: d.name || "untitled" }) });
  d.runId = r.run_id; return d.runId;
}

async function uploadThenIngest(main, d, file) {
  const pv = $("#ingPreview"); clear(pv);
  const status = h("div", { class: "muted" }, `Uploading ${file.name} (${(file.size / 1024).toFixed(1)} KB)…`);
  pv.append(status, h("div", { class: "skeleton", style: "margin-top:8px" }));
  try {
    await ensureRun(d);
    const fd = new FormData(); fd.append("file", file);
    const up = await api("/uploads", { method: "POST", body: fd });
    status.textContent = "Parsing & normalizing log rows…";
    const ing = await api(`/runs/${d.runId}/ingest`, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ upload_id: up.upload_id }) });
    d.ingestion = ing.ingestion; showIngestion(d);
  } catch (e) {
    clear(pv);
    pv.append(h("div", { class: "banner err" }, "Ingestion failed: " + (e && e.message ? e.message : "unknown error")));
  }
}

async function s3Ingest(main, d, uri) {
  const pv = $("#ingPreview"); clear(pv); pv.append(h("div", { class: "skeleton" }));
  try {
    await ensureRun(d);
    const ing = await api(`/runs/${d.runId}/ingest`, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ s3_uri: uri }) });
    d.ingestion = ing.ingestion; showIngestion(d);
  } catch (e) { clear(pv); pv.append(h("div", { class: "banner err" }, "S3 ingestion failed: " + e.message)); }
}

function showIngestion(d) {
  const ing = d.ingestion, pv = $("#ingPreview"); clear(pv);
  const dropTotal = Object.values(ing.dropped || {}).reduce((a, b) => a + b, 0);
  const seg = h("div", {});
  const dm = (ing.detected_models || []).map(m => `${m.model} (${m.rows})`).join(", ") || "—";
  const bar = h("div", { class: "distbar", style: "margin:8px 0" },
    h("span", { class: "compatible", style: `width:${100 * ing.evaluable / Math.max(ing.found, 1)}%` }),
    h("span", { class: "incompatible", style: `width:${100 * dropTotal / Math.max(ing.found, 1)}%` }));
  const lowSignal = (ing.redaction_rate > (SETTINGS?.high_redaction_threshold ?? 0.5)
    || ing.evaluable < (SETTINGS?.small_sample_threshold ?? 10));
  const children = [
    h("div", { class: "row spread" }, h("strong", {}, "Ingestion preview"), h("span", { class: "muted mono" }, `${ing.found} rows`)),
    bar,
    h("div", { class: "muted" }, `${ing.evaluable} evaluable · ${dropTotal} dropped`),
    h("div", { class: "muted", style: "margin-top:6px" }, "Detected legacy models: " + dm),
    (Object.keys(ing.teams || {}).length || Object.keys(ing.users || {}).length)
      ? h("div", { class: "muted", style: "margin-top:4px" },
          (Object.keys(ing.teams || {}).length ? `Teams: ${Object.keys(ing.teams).length}` : "") +
          (Object.keys(ing.users || {}).length ? `${Object.keys(ing.teams || {}).length ? " · " : ""}Users: ${Object.keys(ing.users).length}` : "") +
          " — you can target specific ones in the next step")
      : null,
    lowSignal
      ? h("div", { class: "banner warn", style: "margin-top:10px" }, `Low-signal corpus (${ing.evaluable} evaluable) — results will carry a confidence caveat, but the run will still proceed.`)
      : null,
    h("button", { class: "btn", style: "margin-top:14px", disabled: ing.evaluable === 0, onclick: () => { d.step = "config"; renderNewRun($("#main")); } }, "Continue to configuration"),
  ].filter(Boolean);
  seg.append(...children);
  pv.append(h("div", { class: "card" }, seg));
}

function newRunConfig(main, d) {
  const legacy = d.ingestion?.detected_models?.[0]?.model;
  const entry = (MODELS?.matrix || []).find(e => normModel(e.legacy_model) === normModel(legacy));
  const allModels = Object.keys(MODELS?.models || {});
  // Matrix candidates first (pre-selected), then every other catalog model
  // (incl. Claude) as available extras for a cross-family comparison.
  const primary = entry ? entry.candidates.slice() : allModels;
  const extras = allModels.filter(m => !primary.includes(m));
  const cands = [...primary, ...extras];
  const mapped = !!entry;
  if (!d.candidates.length) {
    d.candidates = entry ? entry.candidates.slice() : (allModels.length ? [allModels[0]] : []);
  }
  const cardsWrap = h("div", { class: "grid cols-3" });
  cands.forEach(m => {
    const fam = MODELS?.models?.[m]?.family;
    const card = h("div", { class: "candcard" + (d.candidates.includes(m) ? " sel" : "") },
      h("div", { class: "row spread" }, h("strong", {}, m),
        h("span", {}, MODELS?.models?.[m]?.like_for_like ? h("span", { class: "badge" }, "peer") : null,
          (fam && fam !== "openai") ? h("span", { class: "badge", style: "margin-left:4px" }, fam) : null)),
      h("div", { class: "muted", style: "font-size:12px;margin-top:4px" },
        m === entry?.default_start ? "recommended start"
          : (fam && fam !== "openai" ? "cross-family baseline" : "candidate")));
    card.addEventListener("click", () => {
      const i = d.candidates.indexOf(m);
      if (i >= 0) d.candidates.splice(i, 1); else d.candidates.push(m);
      card.classList.toggle("sel");
      updatePlan(d);
    });
    cardsWrap.append(card);
  });

  // --- Dynamic LiteLLM discovery: aliases the configured proxy actually serves.
  // Shown only when the endpoint is LiteLLM. Discovered aliases NOT in the catalog
  // are added to d.candidates as raw names (run through the proxy as-is; scored
  // normally, cost skipped when un-priced). Catalog aliases are skipped (already above).
  const discoveredWrap = h("div", { style: "margin-top:14px" });
  async function loadDiscovered() {
    clear(discoveredWrap);
    const litellm = d.endpoint === "litellm";
    // On the LiteLLM path the catalog (Bedrock) cards don't apply — hide them and
    // let the proxy's discovered models BE the candidate list. Restore otherwise.
    cardsWrap.style.display = litellm ? "none" : "";
    const catHead = document.getElementById("catalogCandHeading");
    if (catHead) catHead.style.display = litellm ? "none" : "";
    if (!litellm) return;
    const head = h("div", { class: "row spread", style: "margin-bottom:6px" },
      h("div", { style: "font-weight:600" }, "Candidate models (from your LiteLLM proxy)"),
      h("button", { class: "btn ghost", onclick: () => loadDiscovered() }, "↻ Refresh"));
    const status = h("div", { class: "muted", style: "font-size:12px;margin-bottom:8px" }, "Querying proxy…");
    discoveredWrap.append(head, status);
    let res;
    try { res = await api("/litellm/models"); }
    catch (e) { status.textContent = "Could not query the proxy: " + (e && e.message ? e.message : "error"); return; }
    const models = res.models || [];
    if (!models.length) {
      status.textContent = res.reason
        ? `No models from ${res.base || "the proxy"} — ${res.reason}`
        : `No models registered on ${res.base || "the proxy"}.`;
      return;
    }
    // Drop any lingering catalog (Bedrock) selections that aren't served by the proxy,
    // so the plan reflects only what's actually selectable on this path.
    const served = new Set(models);
    d.candidates = d.candidates.filter(m => served.has(m));
    // Split: registered model-group aliases (what an admin configured) vs LiteLLM's
    // automatic bedrock/* passthrough (often 100s) — show aliases as the primary
    // picker, tuck the passthrough behind a toggle so it doesn't flood the screen.
    const primary = models.filter(m => !m.startsWith("bedrock/"));
    const passthrough = models.filter(m => m.startsWith("bedrock/"));
    const primaryList = primary.length ? primary : models;  // if all are bedrock/, show them
    status.textContent = `${primary.length} model group(s) from ${res.base}`
      + (passthrough.length ? ` (+${passthrough.length} bedrock/* passthrough)` : "")
      + ". Pick any to evaluate through the proxy (cost shown only if priced).";

    function makeCard(alias) {
      const card = h("div", { class: "candcard" + (d.candidates.includes(alias) ? " sel" : "") },
        h("div", { class: "row spread" }, h("strong", {}, alias),
          h("span", { class: "badge" }, "proxy")),
        h("div", { class: "muted", style: "font-size:12px;margin-top:4px" }, "LiteLLM-discovered candidate"));
      card.addEventListener("click", () => {
        const i = d.candidates.indexOf(alias);
        if (i >= 0) d.candidates.splice(i, 1); else d.candidates.push(alias);
        card.classList.toggle("sel");
        updatePlan(d);
      });
      return card;
    }

    const grid = h("div", { class: "grid cols-3" });
    primaryList.forEach(a => grid.append(makeCard(a)));
    discoveredWrap.append(grid);

    // Collapsible bedrock/* passthrough (only when there ARE registered aliases,
    // otherwise they're already shown above).
    if (primary.length && passthrough.length) {
      const moreGrid = h("div", { class: "grid cols-3", style: "margin-top:10px" });
      passthrough.forEach(a => moreGrid.append(makeCard(a)));
      discoveredWrap.append(h("details", { style: "margin-top:10px" },
        h("summary", { style: "cursor:pointer;font-size:12px;color:var(--muted)" },
          `Show all ${passthrough.length} bedrock/* passthrough models`),
        moreGrid));
    }
  }

  const effortSeg = h("div", { class: "seg" }, ...["minimal", "low", "medium", "high"].map(x =>
    h("button", { class: x === d.effort ? "active" : "", onclick: e => { d.effort = x; effortSeg.querySelectorAll("button").forEach(b => b.classList.toggle("active", b.textContent === x)); updatePlan(d); } }, x)));
  const ENDPOINTS = [
    ["bedrock_runtime", "Bedrock Runtime", "us.openai.* · /openai/v1 · Luna/Terra/Sol"],
    ["bedrock_mantle", "Bedrock Mantle", "openai.* · /openai/v1 · required for GPT-5.4"],
    ["litellm", "LiteLLM proxy", "your proxy · /v1/chat/completions"],
  ];
  // If a mantle-only model (GPT-5.4) is selected, force Mantle and lock the others.
  const hasMantleOnly = d.candidates.some(m => MODELS?.models?.[m] && MODELS.models[m].runtime_supported === false);
  if (hasMantleOnly) d.endpoint = "bedrock_mantle";
  const pathSeg = h("div", { class: "seg" }, ...ENDPOINTS.map(([v, t, hint]) =>
    h("button", {
      class: v === d.endpoint ? "active" : "",
      title: hasMantleOnly && v !== "bedrock_mantle" ? "Disabled — a mantle-only model (GPT-5.4) is selected" : hint,
      disabled: hasMantleOnly && v !== "bedrock_mantle",
      onclick: () => { d.endpoint = v; pathSeg.querySelectorAll("button").forEach((b) => b.classList.toggle("active", b.textContent === t)); loadDiscovered(); updatePlan(d); },
    }, t)));
  const mantleModels = d.candidates.filter(m => MODELS?.models?.[m] && MODELS.models[m].runtime_supported === false);
  const endpointHint = h("div", { class: "muted", style: "font-size:11px;margin-top:4px" },
    hasMantleOnly
      ? `Locked to Bedrock Mantle — ${mantleModels.join(", ")} ${mantleModels.length > 1 ? "are" : "is"} Mantle-only (needs a Bedrock API key).`
      : (ENDPOINTS.find(e => e[0] === d.endpoint) || [])[2]);
  const planBox = h("div", { id: "planBox", class: "muted mono" }, "—");

  // Limit records: run a smaller subset instead of the whole log.
  const evalTotal = (d.ingestion && d.ingestion.evaluable) || 0;
  d.samplingMode = d.samplingMode || "all";
  d.samplingN = d.samplingN || Math.min(50, evalTotal || 50);
  const modeSel = h("select", { style: "width:auto" },
    h("option", { value: "all" }, "All records"),
    h("option", { value: "first_n" }, "First N"),
    h("option", { value: "random_n" }, "Random N"));
  modeSel.value = d.samplingMode;
  const nInput = h("input", { type: "text", value: String(d.samplingN), style: "width:90px",
    oninput: e => { d.samplingN = parseInt(e.target.value, 10) || 0; updatePlan(d); } });
  const nWrap = h("span", { style: d.samplingMode === "all" ? "display:none" : "" }, nInput,
    h("span", { class: "muted", style: "margin-left:6px;font-size:12px" }, evalTotal ? `of ${evalTotal}` : ""));
  modeSel.addEventListener("change", () => {
    d.samplingMode = modeSel.value;
    nWrap.style.display = d.samplingMode === "all" ? "none" : "";
    updatePlan(d);
  });
  const limitField = h("label", { class: "field", style: "margin-top:18px" },
    h("span", {}, "Limit records (optional)"),
    h("div", { class: "muted", style: "font-size:11px;margin-bottom:6px" },
      "Evaluate a smaller subset instead of every record — cheaper/faster for a quick check. Applied after any team/user filter."),
    h("div", { class: "row", style: "gap:10px;align-items:center" }, modeSel, nWrap));

  // Evaluation steering: custom guidance/skill injected into the LLM judge's prompt.
  const steeringArea = h("textarea", {
    rows: "5", placeholder: "Optional. e.g. 'Treat any change to a dosage number as Incompatible. A shorter answer that keeps all facts is Compatible. Ignore differences in greeting style.'",
    style: "width:100%;font:inherit;color:var(--text);background:var(--bg);border:1px solid var(--border);border-radius:var(--radius-sm);padding:8px;resize:vertical",
    oninput: e => d.judgeSteering = e.target.value,
  });
  steeringArea.value = d.judgeSteering || "";
  const steerFile = h("input", { type: "file", accept: ".md,.txt,.text", style: "display:none",
    onchange: async e => {
      const f = e.target.files[0]; if (!f) return;
      const text = await f.text();
      d.judgeSteering = (d.judgeSteering ? d.judgeSteering + "\n\n" : "") + text;
      steeringArea.value = d.judgeSteering;
    } });
  // Team / user targeting: filter a shared log down to a specific app/use case.
  const ing = d.ingestion || {};
  const teamEntries = Object.entries(ing.teams || {});
  const userEntries = Object.entries(ing.users || {});
  d.filterTeams = d.filterTeams || [];
  d.filterUsers = d.filterUsers || [];
  function checkGroup(entries, selArr, label) {
    if (!entries.length) return null;
    const boxes = entries.map(([name, count]) => {
      const cb = h("input", { type: "checkbox" });
      cb.checked = selArr.includes(name);
      cb.addEventListener("change", () => {
        const i = selArr.indexOf(name);
        if (cb.checked && i < 0) selArr.push(name);
        else if (!cb.checked && i >= 0) selArr.splice(i, 1);
        updatePlan(d);
      });
      return h("label", { class: "row", style: "gap:6px;align-items:center;font-size:13px" },
        cb, h("span", {}, name), h("span", { class: "muted mono", style: "font-size:11px" }, `(${count})`));
    });
    return h("div", { style: "margin-bottom:8px" },
      h("div", { class: "muted", style: "font-size:12px;margin-bottom:4px" }, label),
      h("div", { class: "row", style: "gap:14px;flex-wrap:wrap" }, ...boxes));
  }
  const teamGroup = checkGroup(teamEntries, d.filterTeams, "Teams");
  const userGroup = checkGroup(userEntries, d.filterUsers, "Users");
  const targetingField = (teamGroup || userGroup)
    ? h("label", { class: "field", style: "margin-top:18px" },
        h("span", {}, "Target specific teams / users (optional)"),
        h("div", { class: "muted", style: "font-size:11px;margin-bottom:6px" },
          "This log has multiple teams/users. Select some to evaluate only their requests (target one app / use case). Leave all unchecked to evaluate everything."),
        teamGroup, userGroup)
    : null;

  const steeringField = h("label", { class: "field", style: "margin-top:18px" },
    h("span", {}, "Evaluation steering / skill for the LLM judge (optional)"),
    h("div", { class: "muted", style: "font-size:11px;margin-bottom:6px" },
      "Extra instructions the judge follows when scoring compatibility — paste a rubric/skill or upload a .md/.txt file."),
    steeringArea,
    h("div", { class: "row", style: "gap:8px;margin-top:6px" },
      h("button", { class: "btn ghost", onclick: () => steerFile.click() }, "Upload skill/steering file"),
      h("button", { class: "btn ghost", onclick: () => { d.judgeSteering = ""; steeringArea.value = ""; } }, "Clear"),
      steerFile));

  main.append(h("div", { class: "card" },
    h("h3", {}, "2 · Configuration"),
    h("div", { id: "catalogCandHeading", class: "muted", style: "margin-bottom:8px" }, "Candidate models" + (
      mapped ? ` (pre-filled from matrix for ${legacy})`
             : ` — “${legacy || "unknown"}” isn’t in the migration matrix; pick any candidate to evaluate against`)),
    cardsWrap,
    discoveredWrap,
    h("div", { class: "grid cols-2", style: "margin-top:18px" },
      h("label", { class: "field" }, h("span", {}, "Reasoning effort (start at medium to baseline)"), effortSeg),
      h("label", { class: "field" }, h("span", {}, "Endpoint"), pathSeg, endpointHint)),
    targetingField,
    limitField,
    steeringField,
    h("div", { class: "row spread", style: "margin-top:8px" },
      h("div", {}, h("span", { class: "muted" }, "Estimated: "), planBox),
      h("div", { class: "row" },
        h("button", { class: "btn ghost", onclick: () => { d.step = "source"; renderNewRun($("#main")); } }, "Back"),
        h("button", { class: "btn secondary", onclick: () => launch(d, true) }, "Dry run (no spend)"),
        h("button", { class: "btn", onclick: () => launch(d, false) }, "Review & launch")))));
  updatePlan(d);
  loadDiscovered();
}

async function updatePlan(d) {
  const box = $("#planBox"); if (!box) return;
  if (!d.candidates.length) { box.textContent = "select a candidate"; return; }
  try {
    const plan = await api(`/runs/${d.runId}/plan`, { method: "POST", headers: { "content-type": "application/json" },
      body: JSON.stringify(cfgBody(d)) });
    box.textContent = `${money(plan.est_cost)} · ${plan.estimated_calls} calls`;
  } catch (e) { box.textContent = e.message; }
}

function cfgBody(d) {
  const callPath = d.endpoint === "litellm" ? "litellm" : "bedrock";
  const bedrockEndpoint = d.endpoint === "bedrock_mantle" ? "mantle" : "runtime";
  return { candidates: d.candidates.map(m => ({ model: m, reasoning_effort: d.effort })),
           name: d.name || null,
           call_path: callPath, bedrock_endpoint: bedrockEndpoint,
           sampling_mode: d.samplingMode || "all",
           sampling_n: d.samplingN || 200,
           judge_steering: d.judgeSteering || null,
           filter_teams: (d.filterTeams && d.filterTeams.length) ? d.filterTeams : null,
           filter_users: (d.filterUsers && d.filterUsers.length) ? d.filterUsers : null };
}

async function launch(d, dry) {
  const planBox = $("#planBox");
  try {
    if (planBox) planBox.textContent = "launching…";
    await api(`/runs/${d.runId}/launch`, { method: "POST", headers: { "content-type": "application/json" },
      body: JSON.stringify({ ...cfgBody(d), dry_run: dry }) });
    const rid = d.runId; state.draft = null; go("run", rid);
  } catch (e) {
    const box = $("#planBox");
    if (box) box.textContent = "launch failed: " + (e && e.message ? e.message : "unknown error");
  }
}

// Run detail: progress -> results/samples/remediations
async function renderRunDetail(main, runId) {
  const info = await api(`/runs/${runId}`);
  main.append(topbar(info.name || "Run"));
  main.append(h("div", { class: "muted mono", style: "font-size:11px;margin:-8px 0 12px" }, runId));
  const tabWrap = h("div", { class: "card" });
  main.append(tabWrap);
  if (["done"].includes(info.status)) return renderResults(tabWrap, runId);
  if (["failed"].includes(info.status)) {
    tabWrap.append(h("div", { class: "banner err" }, "Run failed: " + (info.error || "unknown")));
    // Pull full detail (traceback + per-call errors) from /results.
    try {
      const res = await api(`/runs/${runId}/results`);
      tabWrap.append(errorDetailPanel(res));
    } catch (e) { /* detail is best-effort */ }
    return;
  }
  // progress
  renderProgress(tabWrap, runId);
}

// Reusable panel: full run-level traceback + every distinct per-call error.
function errorDetailPanel(res) {
  const wrap = h("div", { style: "margin-top:12px" });
  const errs = (res && res.sample_errors) || [];
  if (errs.length) {
    wrap.append(h("div", { style: "font-weight:600;margin-bottom:4px" }, `Per-call errors (${errs.length})`));
    const list = h("div", { class: "logpane" });
    errs.forEach(e => list.append(h("div", { class: "logline err" },
      h("span", { class: "mono" }, e.model || "?"), "  —  ",
      h("span", {}, e.error))));
    wrap.append(list);
  }
  const trace = res && res.error_detail;
  if (trace) {
    const pre = h("pre", { style: "white-space:pre-wrap;overflow:auto;max-height:340px;background:var(--bg);border:1px solid var(--border);border-radius:var(--radius-sm);padding:10px;font-size:12px" }, trace);
    const copy = h("button", { class: "btn ghost", style: "margin-bottom:6px",
      onclick: () => { navigator.clipboard && navigator.clipboard.writeText(trace); } }, "Copy stack trace");
    wrap.append(h("details", { style: "margin-top:10px" },
      h("summary", { style: "cursor:pointer;font-weight:600" }, "Full stack trace"),
      h("div", { style: "margin-top:8px" }, copy, pre)));
  }
  if (!errs.length && !trace) {
    wrap.append(h("div", { class: "muted", style: "font-size:12px" }, "No further detail was captured."));
  }
  return wrap;
}

function renderProgress(el, runId) {
  clear(el);
  const log = h("div", { class: "logpane", id: "runLog" });
  const errChip = h("span", { class: "pill incompatible", style: "display:none" }, "0 errors");
  const errToggle = h("label", { class: "row", style: "gap:6px;font-size:12px;cursor:pointer" });
  const errOnly = h("input", { type: "checkbox" });
  errToggle.append(errOnly, h("span", { class: "muted" }, "errors only"));
  errOnly.addEventListener("change", () => log.classList.toggle("errors-only", errOnly.checked));
  el.append(
    h("div", { class: "row spread" },
      h("div", { class: "row", style: "gap:10px;align-items:center" }, h("h3", { style: "margin:0" }, "Running"), errChip),
      h("div", { class: "row", style: "gap:12px;align-items:center" }, errToggle,
        h("button", { class: "btn ghost", onclick: () => cancelRun(runId) }, "Cancel"))),
    h("div", { class: "muted", id: "progText" }, "Connecting to run…"),
    h("div", { class: "progress", style: "margin:10px 0" }, h("span", { id: "progBar", style: "width:0%" })),
    h("div", { class: "muted", style: "font-size:11px;margin-bottom:4px" }, "Live log"),
    log);
  let errCount = 0;

  const seen = new Set();
  const addLog = (line, cls) => {
    const row = h("div", { class: "logrow" + (cls ? " " + cls : "") }, line);
    log.append(row); log.scrollTop = log.scrollHeight;
  };
  const onEvent = (type, d) => {
    const key = type + JSON.stringify(d);
    if (seen.has(key)) return; seen.add(key);
    if (type === "plan") { addLog(`▶ plan: ${d.estimated_calls} calls · est $${(d.est_cost || 0).toFixed(4)}`); }
    else if (type === "progress") {
      $("#progText").textContent = `${d.done}/${d.total} · ${d.model} · ${d.sample_id} (${d.status})`;
      $("#progBar").style.width = `${100 * d.done / Math.max(d.total, 1)}%`;
      const isErr = d.status === "error";
      let line = `[${d.done}/${d.total}] ${d.model} · ${d.sample_id} → ${d.status}`;
      if (isErr && d.error) line += `  —  ${d.error}`;
      addLog(line, isErr ? "err" : "");
      if (isErr) { errCount++; errChip.textContent = `${errCount} error${errCount === 1 ? "" : "s"}`; errChip.style.display = ""; }
    } else if (type === "done") { addLog("✓ done", "ok"); finish(); }
    else if (type === "cancelled") { addLog("✗ cancelled", "err"); finish(); }
    else if (type === "error") { addLog("✗ error: " + (d.error || "unknown"), "err"); finish(); }
  };

  let finished = false;
  const finish = () => {
    if (finished) return; finished = true;
    if (es) es.close(); if (poll) clearInterval(poll);
    setTimeout(() => renderResults(el, runId), 400);
  };

  // Primary: SSE stream
  let es = null;
  try {
    es = new EventSource(`${API}/runs/${runId}/stream`);
    ["plan", "progress", "done", "cancelled", "error"].forEach(t =>
      es.addEventListener(t, ev => { try { onEvent(t, JSON.parse(ev.data)); } catch {} }));
    es.addEventListener("end", finish);
    es.onerror = () => { /* fall back to polling below */ };
  } catch { es = null; }

  // Fallback: poll run status + replay events (covers proxies that buffer SSE)
  const poll = setInterval(async () => {
    try {
      const info = await api(`/runs/${runId}`);
      if (["done", "failed", "cancelled"].includes(info.status)) finish();
    } catch {}
  }, 1500);
}

async function cancelRun(runId) {
  try { await api(`/runs/${runId}/cancel`, { method: "POST" }); } catch {}
}

function goldenCard(src) {
  const ms = (x) => (x == null ? "—" : `${Math.round(x)} ms`);
  const models = (src.detected_models || []).map(m => `${m.model} (${m.rows})`).join(", ") || "—";
  const sources = (src.sources || []).map(s => s.replace(/^upload:/, "")).join(", ") || "—";
  const dropTotal = Object.values(src.dropped || {}).reduce((a, b) => a + b, 0);
  const lat = src.original_latency_ms || {};
  const stat = (label, val) => h("div", {},
    h("div", { class: "value tnum", style: "font-size:22px" }, val),
    h("div", { class: "label" }, label));
  const latLine = lat.count
    ? `avg ${ms(lat.avg)} · p50 ${ms(lat.p50)} · p95 ${ms(lat.p95)} (${lat.count} timed)`
    : "not recorded in source logs";
  return h("div", { class: "card", style: "margin-bottom:14px;border-left:3px solid var(--accent)" },
    h("div", { class: "row spread" },
      h("h3", { style: "margin:0" }, "Golden / existing baseline"),
      h("span", { class: "badge" }, "source of truth")),
    h("div", { class: "muted", style: "font-size:12px;margin:6px 0" },
      "The existing production logs replayed against each candidate. Candidate responses are scored against these."),
    h("div", { class: "grid cols-3", style: "margin-top:8px" },
      stat("Evaluations", String(src.evaluable ?? 0)),
      stat("Rows found", String(src.found ?? 0)),
      stat("Dropped", String(dropTotal))),
    h("table", { style: "margin-top:12px" }, h("tbody", {},
      h("tr", {}, h("td", { class: "muted", style: "width:180px" }, "Source"), h("td", { class: "mono" }, sources)),
      h("tr", {}, h("td", { class: "muted" }, "Existing model(s)"), h("td", {}, models)),
      h("tr", {}, h("td", { class: "muted" }, "Original latency"), h("td", { class: "tnum" }, latLine)),
      h("tr", {}, h("td", { class: "muted" }, "Avg tokens (prompt/total)"),
        h("td", { class: "tnum" }, `${src.avg_prompt_tokens ?? "—"} / ${src.avg_total_tokens ?? "—"}`)),
      h("tr", {}, h("td", { class: "muted" }, "Redaction rate"), h("td", { class: "tnum" }, `${Math.round((src.redaction_rate || 0) * 100)}%`)))));
}

async function renderResults(el, runId) {
  clear(el);
  const res = await api(`/runs/${runId}/results`);
  const scoredTotal = (res.candidates || []).reduce((a, c) => a + (c.scored || 0), 0);
  const erroredTotal = (res.candidates || []).reduce((a, c) => a + (c.errored || 0), 0);

  // Run failed, or every candidate call errored — explain why, don't show empty verdicts.
  if (res.status === "failed" || res.error || (scoredTotal === 0 && erroredTotal > 0)) {
    const why = res.error || res.sample_error || "All candidate calls failed.";
    el.append(
      h("h3", {}, "Run failed"),
      h("div", { class: "banner err" }, why),
      h("p", { class: "muted", style: "margin-top:10px" },
        "Fix the cause above and start a new run. Tip: GPT-5.4 is Bedrock Mantle-only — either pick a "
        + "runtime model (GPT-5.6 Luna / Terra / Sol) or start the server with a Bedrock API key for the Mantle endpoint."),
      h("button", { class: "btn", style: "margin-top:12px", onclick: () => go("new") }, "Start a new run"),
      h("a", { class: "btn ghost", style: "margin-top:12px;margin-left:8px",
               href: `${API}/runs/${runId}/report.pdf`, target: "_blank", rel: "noopener" },
        "⬇ Download PDF report"));
    el.append(errorDetailPanel(res));
    return;
  }

  if (!res.candidates.length) {
    el.append(h("h3", {}, "Results"), h("p", { class: "muted" }, "Dry run — no candidate calls were made. Launch a live run to see verdicts."));
    return;
  }
  if (erroredTotal > 0) {
    el.append(h("div", { class: "banner warn", style: "margin-bottom:14px" },
      `${erroredTotal} candidate call(s) errored and were excluded` + (res.sample_error ? `: ${res.sample_error}` : ".")));
    el.append(errorDetailPanel(res));
  }
  // Results header + PDF download (opens the attachment endpoint in a new tab so
  // the browser handles the download; endpoint 409s until the run is done).
  el.append(h("div", { class: "row spread", style: "margin-bottom:12px" },
    h("h3", { style: "margin:0" }, "Results"),
    h("div", { class: "row", style: "gap:8px" },
      h("a", { class: "btn ghost", href: `${API}/runs/${runId}/report.pdf?theme=light`,
               target: "_blank", rel: "noopener", title: "Light theme, better for printing" }, "Print-friendly PDF"),
      h("a", { class: "btn", href: `${API}/runs/${runId}/report.pdf`,
               target: "_blank", rel: "noopener" }, "⬇ Download PDF report"))));
  (res.caveats || []).forEach(c => el.append(h("div", { class: "banner warn", style: "margin-bottom:14px" }, c)));
  // Golden / existing baseline (what was ingested)
  try {
    const src = await api(`/runs/${runId}/source`);
    el.append(goldenCard(src));
  } catch (e) { /* older run without source summary */ }
  // per-candidate verdict cards
  const cardsWrap = h("div", { class: "grid cols-" + Math.min(res.candidates.length, 3) });
  res.candidates.forEach(v => {
    const dist = v.distribution || {};
    const total = Object.values(dist).reduce((a, b) => a + b, 0) || 1;
    const bar = h("div", { class: "distbar", style: "margin:12px 0" },
      ...LABELS.map(l => h("span", { class: l, style: `width:${100 * (dist[l] || 0) / total}%` })));
    const card = h("div", { class: "candcard", title: "Click to view this model's evaluations" },
      h("div", { class: "row spread" }, h("strong", {}, v.model), h("span", { class: "band " + v.verdict_band }, BAND_TEXT[v.verdict_band])),
      v.endpoint ? h("div", { class: "muted mono", style: "margin-top:4px;font-size:11px" }, "via " + (ENDPOINT_TEXT[v.endpoint] || v.endpoint)) : null,
      h("div", { class: "metric", style: "margin-top:10px" }, h("div", { class: "value tnum" }, fmtPct(v.migration_confidence)), h("div", { class: "label" }, "Migration Confidence")),
      bar,
      h("div", { class: "muted mono", style: "font-size:12px" }, LABELS.map(l => `${(dist[l] || 0)} ${LABEL_TEXT[l]}`).join(" · ")),
      h("div", { class: "muted", style: "margin-top:8px;font-size:12px" }, `hard-break ${fmtPct(v.hard_break_rate * 100)} · ${money(v.cost_total)} · ${v.latency_avg_ms == null ? "—" : Math.round(v.latency_avg_ms) + " ms avg"}`),
      dimBars(v.dimension_averages),
      h("div", { class: "muted", style: "margin-top:8px;font-size:11px;color:var(--accent)" }, "View evaluations →"));
    card.addEventListener("click", () => openCandidate(el, runId, v.model));
    cardsWrap.append(card);
  });
  el.append(h("h3", {}, "Results"), cardsWrap);

  // side-by-side comparison table
  const ms = (x) => (x == null ? "—" : `${Math.round(x)} ms`);
  el.append(h("h3", { style: "margin-top:20px" }, "Side-by-side"),
    h("table", {}, h("thead", {}, h("tr", {}, ...["Candidate", "Confidence", "Hard-break", "Cost", "Latency avg", "Latency p95", "Verdict"].map(t => h("th", {}, t)))),
      h("tbody", {}, ...res.candidates.map(v => {
        const tr = h("tr", { title: "Click to view this model's evaluations" },
          h("td", { class: "mono" }, v.model), h("td", { class: "tnum" }, fmtPct(v.migration_confidence)),
          h("td", { class: "tnum" }, fmtPct(v.hard_break_rate * 100)), h("td", { class: "tnum" }, money(v.cost_total)),
          h("td", { class: "tnum" }, ms(v.latency_avg_ms)), h("td", { class: "tnum" }, ms(v.latency_p95_ms)),
          h("td", {}, h("span", { class: "band " + v.verdict_band }, BAND_TEXT[v.verdict_band])));
        tr.addEventListener("click", () => openCandidate(el, runId, v.model));
        return tr;
      }))));

  // D1–D6 × candidate breakdown: why one model beats another, without drilling in.
  const breakdown = dimBreakdown(res.candidates);
  if (breakdown) {
    el.append(h("div", { class: "row spread", style: "margin-top:20px" },
      h("h3", { style: "margin:0" }, "Dimension breakdown"),
      h("span", { class: "muted", style: "font-size:11px" }, "average score per dimension · 0.00–1.00 · best per row marked ▲")),
      breakdown);
  }

  // candidate selector + drill-down actions
  const models = res.candidates.map(v => v.model);
  const sel = h("select", { id: "candSelect", style: "width:auto;min-width:160px" },
    ...models.map(m => h("option", { value: m }, m)));
  const cur = () => sel.value || models[0];
  const tabs = h("div", { class: "row", style: "margin-top:20px;gap:8px;flex-wrap:wrap;align-items:center" },
    h("span", { class: "muted", style: "font-size:12px" }, "Candidate:"), sel,
    h("button", { class: "btn secondary", onclick: () => loadSamples(el, runId, cur()) }, "View evaluations"),
    h("button", { class: "btn secondary", onclick: () => loadRemediations(el, runId, cur()) }, "See remediations"));
  el.append(tabs, h("div", { id: "drill" }));
}

// Select a candidate and open its evaluations drill-down (shared by the candidate
// cards and the Side-by-side table rows).
function openCandidate(el, runId, model) {
  const sel = document.getElementById("candSelect");
  if (sel) sel.value = model;
  loadSamples(el, runId, model);
  const drill = document.getElementById("drill");
  if (drill) drill.scrollIntoView({ behavior: "smooth", block: "start" });
}

async function loadSamples(el, runId, model) {
  const drill = $("#drill"); clear(drill); drill.append(h("div", { class: "skeleton" }));
  const data = await api(`/runs/${runId}/samples?candidate=${encodeURIComponent(model)}&limit=200`);
  clear(drill);
  const detail = h("div", { id: "sampleDetail" });
  const rows = data.items.map(p => {
    const tr = h("tr", { title: "Click for full request/response detail" },
      h("td", { class: "mono" }, p.sample_id),
      h("td", {}, h("span", { class: "pill " + labelKey(p) }, LABEL_TEXT[labelKey(p)])),
      h("td", { class: "tnum" }, (p.d1_semantic?.score ?? 0).toFixed(2)),
      h("td", { class: "tnum" }, (p.d2_format?.score ?? 0).toFixed(2)),
      h("td", { class: "tnum" }, (p.d3_factual?.score ?? 0).toFixed(2)),
      h("td", { class: "tnum" }, (p.d4_verbosity?.score ?? 0).toFixed(2)),
      h("td", { class: "muted" }, p.judge_reason || (p.hard_break ? "hard-break" : "—")));
    tr.addEventListener("click", () => showSampleDetail(detail, runId, model, p.sample_id));
    return tr;
  });
  drill.append(h("h3", {}, `Evaluations · ${model} (${data.total})`),
    h("div", { class: "muted", style: "font-size:12px;margin-bottom:6px" }, "Every request evaluated for this candidate. Click a row to see the original vs candidate request/response, models and latency."),
    h("table", {}, h("thead", {}, h("tr", {}, ...["Request", "Label", "D1", "D2", "D3", "D4", "Reason"].map(t => h("th", {}, t)))),
      h("tbody", {}, ...rows)),
    detail);
}

async function showSampleDetail(host, runId, model, sampleId) {
  clear(host);
  host.append(h("div", { class: "skeleton", style: "margin-top:14px" }));
  let d;
  try {
    d = await api(`/runs/${runId}/samples/${encodeURIComponent(sampleId)}?candidate=${encodeURIComponent(model)}`);
  } catch (e) {
    clear(host); host.append(h("div", { class: "banner err" }, "Could not load sample: " + (e && e.message ? e.message : "error"))); return;
  }
  const o = d.original, c = d.candidate_result, sc = d.score || {};
  const ms = (x) => (x == null ? "—" : `${Math.round(x)} ms`);
  const respText = (r) => {
    if (!r) return "(no response)";
    if (r.tool_calls && r.tool_calls.length) return "tool_call → " + JSON.stringify(r.tool_calls, null, 2);
    return r.text != null ? r.text : "(empty)";
  };
  const msgsText = (msgs, instr) =>
    (instr ? `[system/instructions]\n${instr}\n\n` : "") +
    (msgs || []).map(m => `[${m.role}]\n${m.content}`).join("\n\n");

  const pane = (title, sub, bodyText) => h("div", {},
    h("div", { class: "row spread" }, h("strong", {}, title), h("span", { class: "muted mono", style: "font-size:11px" }, sub)),
    h("div", { class: "diff-pane", style: "margin-top:6px;max-height:280px;overflow:auto" }, bodyText));

  clear(host);
  host.append(h("div", { class: "card", style: "margin-top:14px" },
    h("div", { class: "row spread" },
      h("h3", { style: "margin:0" }, `Sample ${sampleId}`),
      h("span", { class: "pill " + labelKey(sc) }, LABEL_TEXT[labelKey(sc)] || "—")),
    // request (shared — the candidate replays the same request)
    h("div", { style: "margin-top:12px" },
      pane("Request (replayed to both)", `original model: ${o.model || "—"}`,
        msgsText(o.messages, o.instructions))),
    // side-by-side responses, with word-level difference highlighting
    responseCompare(o, c, respText, ms),
    // latency comparison line
    h("div", { class: "muted", style: "margin-top:10px;font-size:12px" },
      `Latency — original ${ms(o.latency_ms)} vs candidate ${c ? ms(c.latency_ms) : "—"}` +
      (o.latency_ms && c && c.latency_ms ? ` (${c.latency_ms <= o.latency_ms ? "faster" : "slower"} by ${Math.abs(c.latency_ms - o.latency_ms)} ms)` : "")),
    // detailed compatibility scores
    scoresBlock(sc)));
  host.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

function scoreColor(v) {
  if (v >= 0.85) return "var(--ok)";
  if (v >= 0.6) return "var(--partial)";
  return "var(--bad)";
}

// ---- response diff ---------------------------------------------------------
// If both sides are JSON, pretty-print them with sorted keys so the same key lands
// on the same line on both sides; a changed value then shows as one red/green pair.
function canonicalJson(text) {
  const t = (text || "").trim();
  if (!t || !"{[".includes(t[0])) return null;
  try {
    const sortKeys = (v) => Array.isArray(v) ? v.map(sortKeys)
      : (v && typeof v === "object")
        ? Object.fromEntries(Object.keys(v).sort().map(k => [k, sortKeys(v[k])])) : v;
    return JSON.stringify(sortKeys(JSON.parse(t)), null, 2);
  } catch (e) { return null; }
}

// Words, whitespace runs and punctuation are separate tokens, so whitespace-only
// changes don't hide real ones and "42," vs "41," diffs as just the number.
function diffTokenize(s) { return (s || "").match(/\s+|[A-Za-z0-9_]+|[^\sA-Za-z0-9_]/g) || []; }

const DIFF_MAX_TOKENS = 2500;  // LCS is O(n·m); beyond this we skip highlighting

// Returns [{op: "eq"|"del"|"ins", text}] or null if too large to diff.
function diffWords(a, b) {
  const A = diffTokenize(a), B = diffTokenize(b);
  // trim common prefix/suffix first (cheap, and usually most of the text)
  let pre = 0;
  while (pre < A.length && pre < B.length && A[pre] === B[pre]) pre++;
  let suf = 0;
  while (suf < A.length - pre && suf < B.length - pre && A[A.length - 1 - suf] === B[B.length - 1 - suf]) suf++;
  const a2 = A.slice(pre, A.length - suf), b2 = B.slice(pre, B.length - suf);
  if (a2.length > DIFF_MAX_TOKENS || b2.length > DIFF_MAX_TOKENS) return null;
  const n = a2.length, m = b2.length;
  const L = Array.from({ length: n + 1 }, () => new Uint16Array(m + 1));
  for (let i = n - 1; i >= 0; i--)
    for (let j = m - 1; j >= 0; j--)
      L[i][j] = a2[i] === b2[j] ? L[i + 1][j + 1] + 1 : Math.max(L[i + 1][j], L[i][j + 1]);
  const out = [];
  const push = (op, text) => {
    const last = out[out.length - 1];
    if (last && last.op === op) last.text += text; else out.push({ op, text });
  };
  if (pre) push("eq", A.slice(0, pre).join(""));
  let i = 0, j = 0;
  while (i < n && j < m) {
    if (a2[i] === b2[j]) { push("eq", a2[i]); i++; j++; }
    else if (L[i + 1][j] >= L[i][j + 1]) { push("del", a2[i]); i++; }
    else { push("ins", b2[j]); j++; }
  }
  while (i < n) push("del", a2[i++]);
  while (j < m) push("ins", b2[j++]);
  if (suf) push("eq", A.slice(A.length - suf).join(""));
  // whitespace-only changes are noise: keep them on their own side but don't highlight
  return out.map(p => (p.op !== "eq" && !p.text.trim()) ? { ...p, ws: true } : p);
}

// Render one side of the diff: original shows deletions, candidate shows insertions.
function diffSide(parts, side) {
  const keep = side === "orig" ? "del" : "ins";
  return parts.filter(p => p.op === "eq" || p.op === keep)
    .map(p => (p.op === "eq" || p.ws) ? p.text : h("span", { class: "d-" + p.op }, p.text));
}

function responseCompare(o, c, respText, ms) {
  const origText = respText(o.response);
  const candOk = c && c.status !== "error";
  const candText = !c ? "—" : (c.status === "error" ? "ERROR: " + (c.error || "unknown") : respText(c.response));
  const origSub = `${o.model || "—"} · ${ms(o.latency_ms)}${o.usage && o.usage.total_tokens ? " · " + o.usage.total_tokens + " tok" : ""}`;
  const candSub = c
    ? `${c.model}${c.reasoning_effort ? " · effort " + c.reasoning_effort : ""} · ${ms(c.latency_ms)}${c.usage && c.usage.total_tokens ? " · " + c.usage.total_tokens + " tok" : ""}${c.cost_estimate ? " · " + money(c.cost_estimate) : ""}`
    : "(no candidate result)";

  // Normalize JSON on both sides only when BOTH parse, so the layout is comparable.
  const jo = canonicalJson(origText), jc = candOk ? canonicalJson(candText) : null;
  const isJson = jo != null && jc != null;
  const A = isJson ? jo : origText, B = isJson ? jc : candText;
  const parts = candOk ? diffWords(A, B) : null;

  const origPane = h("div", { class: "diff-pane", style: "margin-top:6px;max-height:320px;overflow:auto" });
  const candPane = h("div", { class: "diff-pane", style: "margin-top:6px;max-height:320px;overflow:auto" });
  const stats = h("span", { class: "muted", style: "font-size:11px" });
  const toggleId = "diffToggle" + (++responseCompare.seq);
  const toggle = h("input", { type: "checkbox", id: toggleId });
  toggle.checked = parts != null;
  toggle.disabled = parts == null;

  const paint = () => {
    clear(origPane); clear(candPane);
    if (toggle.checked && parts) {
      origPane.append(...diffSide(parts, "orig"));
      candPane.append(...diffSide(parts, "cand"));
    } else {
      origPane.append(isJson && toggle.checked ? A : origText);
      candPane.append(isJson && toggle.checked ? B : candText);
    }
  };
  toggle.addEventListener("change", paint);
  paint();

  if (parts) {
    const words = (op) => parts.filter(p => p.op === op).reduce((n, p) => n + (p.text.match(/[A-Za-z0-9_]+/g) || []).length, 0);
    const same = parts.every(p => p.op === "eq" || p.ws);
    stats.textContent = same ? "identical" : `−${words("del")} / +${words("ins")} words${isJson ? " · JSON keys sorted & aligned" : ""}`;
  } else if (!candOk) {
    stats.textContent = "no candidate response to compare";
  } else {
    stats.textContent = "too long to highlight — showing plain text";
  }

  const head = (title, sub) => h("div", { class: "row spread" },
    h("strong", {}, title), h("span", { class: "muted mono", style: "font-size:11px" }, sub));
  return h("div", { style: "margin-top:12px" },
    h("div", { class: "row spread", style: "margin-bottom:6px;flex-wrap:wrap" },
      h("label", { class: "row", style: "gap:6px;font-size:12px;cursor:pointer", for: toggleId },
        toggle, "Highlight differences"),
      h("span", { class: "row", style: "gap:10px;font-size:11px" },
        h("span", { class: "d-del" }, "removed from original"),
        h("span", { class: "d-ins" }, "added by candidate"),
        stats)),
    h("div", { class: "grid cols-2" },
      h("div", {}, head("Original response", origSub), origPane),
      h("div", {}, head("Candidate response", candSub), candPane)));
}
responseCompare.seq = 0;

// Small horizontal score bar (0..1), colored by the same thresholds as everywhere else.
function miniBar(v, width = "100%") {
  const pctW = Math.max(0, Math.min(1, v || 0)) * 100;
  return h("div", { class: "minibar", style: `width:${width}` },
    h("span", { style: `width:${pctW}%;background:${scoreColor(v)}` }));
}

// Per-dimension score bars for a Results card (dimension_averages keyed d1_semantic…).
function dimBars(avgs) {
  const rows = DIMENSIONS.filter(d => typeof (avgs || {})[d.key] === "number").map(d => {
    const v = avgs[d.key];
    return h("div", { class: "dimrow", title: `${d.code} ${d.name} — ${d.desc}` },
      h("span", { class: "dimname" }, h("b", {}, d.code), " " + d.name),
      miniBar(v),
      h("span", { class: "tnum dimval", style: `color:${scoreColor(v)}` }, v.toFixed(2)));
  });
  if (!rows.length) return null;
  return h("div", { class: "dimbars" }, ...rows);
}

// D1–D6 × candidate table; highlights the best candidate per dimension.
function dimBreakdown(cands) {
  const dims = DIMENSIONS.filter(d => cands.some(v => typeof (v.dimension_averages || {})[d.key] === "number"));
  if (!dims.length || !cands.length) return null;
  const head = h("tr", {}, h("th", {}, "Dimension"), ...cands.map(v => h("th", { class: "mono" }, v.model)));
  const body = dims.map(d => {
    const vals = cands.map(v => (v.dimension_averages || {})[d.key]);
    const nums = vals.filter(x => typeof x === "number");
    const best = nums.length > 1 ? Math.max(...nums) : null;
    const spread = nums.length > 1 && Math.max(...nums) - Math.min(...nums) > 0.001;
    return h("tr", { class: "static" },
      h("td", { title: d.desc },
        h("b", {}, d.code), " " + d.name,
        d.hard ? h("span", { class: "hardtag", title: "A failure here caps the verdict at Incompatible" }, "hard-break") : null),
      ...vals.map(x => typeof x !== "number"
        ? h("td", { class: "muted" }, "—")
        : h("td", {}, h("div", { class: "dimcell" },
            miniBar(x),
            h("span", { class: "tnum dimval", style: `color:${scoreColor(x)}` }, x.toFixed(2)),
            (spread && x === best) ? h("span", { class: "bestmark", title: "Best on this dimension" }, "▲") : null))));
  });
  return h("table", { class: "breakdown" }, h("thead", {}, head), h("tbody", {}, ...body));
}

function scoresBlock(sc) {
  const label = labelKey(sc);
  const wrap = h("div", { style: "margin-top:16px" });
  wrap.append(
    h("div", { class: "row spread", style: "margin-bottom:8px" },
      h("strong", {}, "Compatibility scores"),
      h("span", { class: "muted", style: "font-size:11px" }, "0.00–1.00 · higher = more compatible")),
    // overall label + what it means
    h("div", { class: "row", style: "gap:8px;align-items:center;margin-bottom:6px" },
      h("span", { class: "pill " + label }, LABEL_TEXT[label] || "—"),
      h("span", { class: "muted", style: "font-size:12px" }, LABEL_EXPLAIN[label] || "")),
    sc.hard_break
      ? h("div", { class: "banner err", style: "margin:6px 0;font-size:12px" },
          "Hard break — a broken format, diverged fact, or mismatched tool call caps this at Incompatible regardless of the other scores.")
      : null);

  const rows = DIMENSIONS.map(dim => {
    const d = sc[dim.key];
    if (!d) return null;  // D6 absent for text responses
    const v = d.score ?? 0;
    const bar = h("div", { style: "height:6px;border-radius:999px;background:var(--surface-2);width:90px;overflow:hidden" },
      h("span", { style: `display:block;height:100%;width:${Math.round(v * 100)}%;background:${scoreColor(v)}` }));
    return h("tr", {},
      h("td", { style: "white-space:nowrap" },
        h("strong", {}, dim.code), " ", dim.name,
        dim.hard ? h("span", { class: "muted", title: "A failure here is a hard break", style: "font-size:10px" }, " ⚠") : null),
      h("td", { class: "tnum", style: "width:52px" }, v.toFixed(2)),
      h("td", { style: "width:100px" }, bar),
      h("td", { class: "muted", style: "font-size:12px" }, d.reason || dim.desc));
  }).filter(Boolean);

  wrap.append(h("table", { style: "margin-top:6px" },
    h("thead", {}, h("tr", {}, ...["Dimension", "Score", "", "Detail"].map(t => h("th", {}, t)))),
    h("tbody", {}, ...rows)));

  // judge verdict explained
  if (sc.judge_reason || sc.judge_rationale) {
    wrap.append(h("div", { class: "muted", style: "margin-top:8px;font-size:12px" },
      h("strong", {}, "Judge verdict: "), h("code", {}, sc.judge_reason || "—"), " — ",
      JUDGE_REASON_TEXT[sc.judge_reason] || "categorical judge outcome."));
    if (sc.judge_rationale) {
      wrap.append(h("div", { class: "diff-pane", style: "margin-top:6px;font-family:var(--sans);font-size:12px;white-space:pre-wrap" },
        h("span", { class: "muted" }, "Judge rationale: "), sc.judge_rationale));
    }
  }
  wrap.append(scoresLegend());
  return wrap;
}

function scoresLegend() {
  const details = document.createElement("details");
  details.style.marginTop = "12px";
  details.style.fontSize = "12px";
  const summary = document.createElement("summary");
  summary.textContent = "How to read these scores";
  summary.style.cursor = "pointer";
  summary.className = "muted";
  details.append(summary);

  const box = h("div", { style: "margin-top:10px" });

  // Labels
  box.append(h("div", { style: "font-weight:600;margin-bottom:4px" }, "Overall label"));
  LABELS.forEach(l => box.append(h("div", { class: "row", style: "gap:8px;align-items:center;margin-bottom:3px" },
    h("span", { class: "pill " + l, style: "min-width:120px" }, LABEL_TEXT[l]),
    h("span", { class: "muted" }, LABEL_EXPLAIN[l]))));

  // Dimensions
  box.append(h("div", { style: "font-weight:600;margin:10px 0 4px" }, "Dimensions (0.00–1.00, higher = more compatible)"));
  DIMENSIONS.forEach(d => box.append(h("div", { class: "muted", style: "margin-bottom:2px" },
    h("strong", {}, d.code + " " + d.name), (d.hard ? " ⚠ " : " — "), d.desc)));

  // Score colors
  box.append(h("div", { style: "font-weight:600;margin:10px 0 4px" }, "Score bar colors"),
    h("div", { class: "row", style: "gap:14px;flex-wrap:wrap" },
      h("span", {}, h("span", { style: "display:inline-block;width:10px;height:10px;border-radius:2px;background:var(--ok);margin-right:5px" }), "≥ 0.85 strong"),
      h("span", {}, h("span", { style: "display:inline-block;width:10px;height:10px;border-radius:2px;background:var(--partial);margin-right:5px" }), "0.60–0.84 moderate"),
      h("span", {}, h("span", { style: "display:inline-block;width:10px;height:10px;border-radius:2px;background:var(--bad);margin-right:5px" }), "< 0.60 weak")));

  // Hard break + judge reasons
  box.append(h("div", { class: "muted", style: "margin:10px 0" },
    h("strong", {}, "⚠ Hard break"), " — a failure on D2 (format), D3 (facts), or D6 (tool call) caps the verdict at Incompatible regardless of the other scores."));
  box.append(h("div", { style: "font-weight:600;margin:10px 0 4px" }, "Judge reasons"));
  Object.entries(JUDGE_REASON_TEXT).forEach(([k, v]) => box.append(h("div", { class: "muted", style: "margin-bottom:2px" },
    h("code", {}, k), " — ", v)));

  details.append(box);
  return details;
}

async function loadRemediations(el, runId, model) {
  const drill = $("#drill"); clear(drill); drill.append(h("div", { class: "skeleton" }));
  let rems;
  try { rems = await api(`/runs/${runId}/remediations?candidate=${encodeURIComponent(model)}`); }
  catch (e) { clear(drill); drill.append(h("div", { class: "banner err" }, "Could not load remediations: " + e.message)); return; }
  clear(drill);
  if (!rems.length) { drill.append(h("h3", {}, "Remediations"), h("p", { class: "muted" }, "No remediation clusters — nothing incompatible to fix.")); return; }
  drill.append(
    h("h3", {}, `Remediations · ${model}`),
    h("div", { class: "muted", style: "font-size:12px;margin-bottom:10px" },
      "Each card groups this candidate's failing evaluations by the judge's reason. Edit the suggested change, "
      + "check the estimate, then Apply & re-test: the cluster is re-run with the change, plus a sample of "
      + "Compatible evaluations to catch regressions. The run's own results are never overwritten."),
    ...rems.map(r => remediationCard(runId, r)));
}

const EFFORTS = ["minimal", "low", "medium", "high"];
const REM_TYPE_TEXT = { prompt_edit: "Prompt instruction", format_schema: "Format instruction", reasoning_effort: "Reasoning effort" };

function remediationCard(runId, rem) {
  const base = `/runs/${runId}/remediations/${rem.remediation_id}`;
  const isInstr = (t) => t === "prompt_edit" || t === "format_schema";
  // editable draft of the change (sent with plan + launch; launch saves it)
  const draft = {
    type: rem.change.type,
    instr: isInstr(rem.change.type) ? (rem.change.after || "") : "",
    placement: rem.change.placement || "append",
    effort: rem.change.type === "reasoning_effort" ? (rem.change.after || "low") : "low",
    instrType: isInstr(rem.change.type) ? rem.change.type : "prompt_edit",
    regAll: false, regN: 20,
  };
  const body = () => {
    const b = { type: draft.type, regression_sample: draft.regAll ? -1 : Math.max(0, parseInt(draft.regN, 10) || 0) };
    if (draft.type === "reasoning_effort") b.after = draft.effort;
    else { b.after = draft.instr; b.placement = draft.placement; }
    return b;
  };
  const post = (path, b) => api(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(b || {}) });

  // --- change editor
  const fromEffort = rem.change.type === "reasoning_effort" ? (rem.change.before || "medium") : null;
  const instrArea = h("textarea", { rows: 4, class: "remtext", "aria-label": "Instruction added to the system prompt" });
  instrArea.value = draft.instr;
  instrArea.addEventListener("input", () => { draft.instr = instrArea.value; scheduleEstimate(); });
  const placeSel = h("select", { style: "width:auto", "aria-label": "Where to add the instruction" },
    h("option", { value: "append" }, "Append to system prompt"), h("option", { value: "prepend" }, "Prepend to system prompt"));
  placeSel.value = draft.placement;
  placeSel.addEventListener("change", () => { draft.placement = placeSel.value; scheduleEstimate(); });
  const effortSel = h("select", { style: "width:auto", "aria-label": "Reasoning effort to test" },
    ...EFFORTS.map(e => h("option", { value: e }, e)));
  effortSel.value = draft.effort;
  effortSel.addEventListener("change", () => { draft.effort = effortSel.value; scheduleEstimate(); });
  const instrBox = h("div", {}, instrArea,
    h("div", { class: "row", style: "margin-top:6px;gap:8px" }, placeSel,
      h("span", { class: "muted", style: "font-size:11px" }, "Added to every request's own system prompt in this cluster.")));
  const effortBox = h("div", { class: "row", style: "gap:8px" },
    h("span", { class: "muted", style: "font-size:12px" }, `Reasoning effort${fromEffort ? " " + fromEffort : ""} →`), effortSel,
    h("span", { class: "muted", style: "font-size:11px" }, "No prompt change."));
  const typeSeg = h("div", { class: "seg" });
  const setType = (t) => {
    draft.type = t === "instruction" ? draft.instrType : "reasoning_effort";
    typeSeg.querySelectorAll("button").forEach(b => b.classList.toggle("active",
      (b.dataset.t === "effort") === (draft.type === "reasoning_effort")));
    instrBox.style.display = draft.type === "reasoning_effort" ? "none" : "";
    effortBox.style.display = draft.type === "reasoning_effort" ? "" : "none";
    scheduleEstimate();
  };
  typeSeg.append(
    h("button", { "data-t": "instruction", onclick: () => setType("instruction") }, "Instruction"),
    h("button", { "data-t": "effort", onclick: () => setType("effort") }, "Reasoning effort"));

  // --- regression sample
  const regN = h("input", { type: "number", min: "0", value: String(draft.regN), style: "width:70px", "aria-label": "Compatible evaluations to re-check" });
  regN.addEventListener("input", () => { draft.regN = regN.value; scheduleEstimate(); });
  const regAll = h("input", { type: "checkbox", "aria-label": "Re-check all Compatible evaluations" });
  regAll.addEventListener("change", () => { draft.regAll = regAll.checked; regN.disabled = regAll.checked; scheduleEstimate(); });
  const regRow = h("div", { class: "row", style: "gap:8px;font-size:12px;flex-wrap:wrap" },
    h("span", { class: "muted" }, "Regression check: re-run"), regN,
    h("span", { class: "muted" }, "Compatible evaluations"),
    h("label", { class: "row", style: "gap:4px;cursor:pointer" }, regAll, "all"));

  // --- estimate + actions
  const estLine = h("div", { class: "muted", style: "font-size:12px" }, "Estimating…");
  const msg = h("div");
  const applyBtn = h("button", { class: "btn" }, "Apply & re-test");
  const cancelBtn = h("button", { class: "btn ghost", style: "display:none" }, "Cancel re-test");
  const copyBtn = h("button", { class: "btn secondary" }, "Copy patch");
  const progWrap = h("div", { style: "display:none;margin-top:10px" });
  const resultHost = h("div");
  let estTimer = null, lastPlan = null;
  function scheduleEstimate() { clearTimeout(estTimer); estTimer = setTimeout(estimate, 350); }
  async function estimate() {
    try {
      const p = await post(base + "/retest/plan", body());
      lastPlan = p;
      estLine.textContent = `${p.calls} call${p.calls === 1 ? "" : "s"} (${p.target_calls} in this cluster + ${p.regression_calls} regression check`
        + `${p.regression_calls === 1 ? "" : "s"} of ${p.compatible_available} Compatible) · ~${money(p.est_cost)} · ~${fmtDuration(p.est_seconds)}`;
      applyBtn.disabled = false; clear(msg);
    } catch (e) {
      lastPlan = null; estLine.textContent = ""; applyBtn.disabled = true;
      clear(msg); msg.append(h("div", { class: "banner err", style: "margin-top:8px" }, e.message));
    }
  }

  let polling = null;
  function showProgress(res) {
    progWrap.style.display = "";
    clear(progWrap);
    const pct = res.total ? Math.round(100 * res.done / res.total) : 0;
    progWrap.append(
      h("div", { class: "row spread", style: "font-size:12px;margin-bottom:4px" },
        h("span", {}, `Re-testing… ${res.done} / ${res.total}`), h("span", { class: "muted tnum" }, `${pct}% · ${money(res.cost)} so far`)),
      h("div", { class: "progress" }, h("span", { style: `width:${pct}%` })));
  }
  function setRunning(on) {
    applyBtn.disabled = on; cancelBtn.style.display = on ? "" : "none";
    [instrArea, placeSel, effortSel, regN, regAll].forEach(x => { x.disabled = on || (x === regN && draft.regAll); });
    typeSeg.querySelectorAll("button").forEach(b => { b.disabled = on; });
  }
  async function poll() {
    let cur;
    try { cur = await api(base); } catch (e) { return; }
    const res = cur.result;
    if (res && res.status === "running") { showProgress(res); polling = setTimeout(poll, 1000); return; }
    polling = null; setRunning(false); progWrap.style.display = "none";
    rem = cur;
    renderResult();
  }
  applyBtn.addEventListener("click", async () => {
    clear(msg);
    if (lastPlan && lastPlan.calls > 50 &&
        !confirm(`This re-test makes ${lastPlan.calls} model calls (~${money(lastPlan.est_cost)}). Continue?`)) return;
    try {
      await post(base + "/retest", body());
      setRunning(true); showProgress({ done: 0, total: lastPlan ? lastPlan.calls : 0, cost: 0 });
      clear(resultHost);
      poll();
    } catch (e) {
      msg.append(h("div", { class: "banner warn", style: "margin-top:8px" }, e.message));
    }
  });
  cancelBtn.addEventListener("click", async () => {
    try { await post(base + "/retest/cancel"); } catch (e) { /* finished meanwhile */ }
  });
  copyBtn.addEventListener("click", () => copyText(remediationPatch(rem, draft), copyBtn));

  function renderResult() {
    clear(resultHost);
    if (rem.result) resultHost.append(retestResultPanel(runId, rem));
  }

  const card = h("div", { class: "card remcard" },
    h("div", { class: "row spread" },
      h("strong", {}, rem.target.reason || "cluster"),
      h("span", { class: "muted mono", style: "font-size:11px" },
        `${rem.evidence_count} evaluation${rem.evidence_count === 1 ? "" : "s"} · ${REM_TYPE_TEXT[rem.change.type] || rem.change.type}`)),
    JUDGE_REASON_TEXT[rem.target.reason] ? h("div", { class: "muted", style: "font-size:12px;margin-top:2px" }, JUDGE_REASON_TEXT[rem.target.reason]) : null,
    h("div", { style: "margin:6px 0 10px;font-size:13px" }, rem.expected_effect || ""),
    h("div", { class: "row", style: "margin-bottom:8px" }, h("span", { class: "muted", style: "font-size:12px" }, "Change:"), typeSeg),
    instrBox, effortBox,
    h("div", { style: "margin-top:10px" }, regRow),
    h("div", { class: "row spread", style: "margin-top:12px;flex-wrap:wrap;gap:8px" },
      estLine, h("div", { class: "row", style: "gap:8px" }, copyBtn, cancelBtn, applyBtn)),
    msg, progWrap, resultHost);

  setType(rem.change.type === "reasoning_effort" ? "effort" : "instruction");
  renderResult();
  if (rem.result && rem.result.status === "running") { setRunning(true); showProgress(rem.result); poll(); }
  return card;
}

function fmtDuration(s) {
  if (s == null) return "—";
  if (s < 60) return `${Math.round(s)} s`;
  const m = Math.floor(s / 60), r = Math.round(s % 60);
  return r ? `${m} min ${r} s` : `${m} min`;
}

function copyText(text, btn) {
  const done = () => { const t = btn.textContent; btn.textContent = "Copied"; setTimeout(() => { btn.textContent = t; }, 1500); };
  if (navigator.clipboard && window.isSecureContext) { navigator.clipboard.writeText(text).then(done, () => fallback()); return; }
  fallback();
  function fallback() {
    const ta = h("textarea", { style: "position:fixed;top:-1000px" }); ta.value = text;
    document.body.append(ta); ta.select();
    try { document.execCommand("copy"); done(); } catch (e) { window.prompt("Copy the patch:", text); }
    ta.remove();
  }
}

// Text the customer applies in their own app / gateway config.
function remediationPatch(rem, draft) {
  const r = rem.result && rem.result.status === "done" ? rem.result : null;
  const validated = r && r.change ? r.change : null;
  const type = validated ? validated.type : draft.type;
  const lines = [
    `# ModelShift remediation — candidate ${rem.candidate}`,
    `# cluster: ${rem.target.reason || "cluster"} (${rem.evidence_count} evaluation${rem.evidence_count === 1 ? "" : "s"}) · run ${rem.run_id}`,
  ];
  if (r) {
    lines.push(`# re-test: cluster confidence ${fmtPct(r.confidence_before)} -> ${fmtPct(r.confidence_after)}; `
      + `overall ${fmtPct(r.confidence_run_before)} -> ${fmtPct(r.confidence_run_after)} (projected); `
      + `${r.moved.to_compatible || 0} fixed, ${r.moved.regressions || 0} regressions in ${r.regression_checked} re-checked`);
  } else {
    lines.push("# NOT re-tested yet — run Apply & re-test before rolling this out.");
  }
  if (type === "reasoning_effort") {
    const eff = validated ? validated.after : draft.effort;
    lines.push("", `# Set the request parameter for ${rem.candidate}:`, `reasoning_effort: ${eff}`,
      "", "# LiteLLM proxy config (config.yaml), if the parameter is set at the gateway:",
      "model_list:", `  - model_name: ${rem.candidate}`, "    litellm_params:", "      # ...existing params...",
      `      reasoning_effort: ${eff}`);
  } else {
    const text = validated ? validated.after : draft.instr;
    const place = (validated ? validated.placement : draft.placement) || "append";
    lines.push("", `# ${place === "prepend" ? "Prepend to the start of" : "Append to the end of"} the system prompt `
      + "for the requests in this workload:", "", text);
  }
  return lines.join("\n");
}

function retestResultPanel(runId, rem) {
  const r = rem.result;
  const wrap = h("div", { class: "retest", style: "margin-top:12px" });
  if (r.status !== "done") {
    const kind = r.status === "cancelled" ? "warn" : "err";
    wrap.append(h("div", { class: "banner " + kind }, `Re-test ${r.status}` + (r.error ? `: ${r.error}` : "")
      + (r.done ? ` (${r.done} of ${r.total} calls completed)` : "")));
    return wrap;
  }
  const delta = (a, b) => {
    if (a == null || b == null) return h("span", { class: "muted" }, "—");
    const d = b - a, col = d > 0.05 ? "var(--ok)" : d < -0.05 ? "var(--bad)" : "var(--muted)";
    return h("span", { class: "tnum" }, `${fmtPct(a)} → `, h("b", { style: `color:${col}` }, fmtPct(b)),
      h("span", { style: `color:${col};font-size:11px;margin-left:4px` }, `(${d >= 0 ? "+" : ""}${Math.round(d)})`));
  };
  const m = r.moved || {};
  const stat = (label, val, color) => h("div", {},
    h("div", { class: "tnum", style: `font-size:20px;font-weight:700${color ? ";color:" + color : ""}` }, String(val)),
    h("div", { class: "muted", style: "font-size:11px" }, label));
  const appliedLine = r.change
    ? (r.change.type === "reasoning_effort" ? `reasoning effort ${r.change.before || ""} → ${r.change.after}`
      : `${r.change.placement || "append"}: "${(r.change.after || "").slice(0, 140)}${(r.change.after || "").length > 140 ? "…" : ""}"`)
    : "";
  wrap.append(
    h("div", { class: "row spread", style: "margin-bottom:8px" },
      h("strong", {}, "Re-test result"),
      h("span", { class: "muted", style: "font-size:11px" }, `${r.finished_at ? new Date(r.finished_at).toLocaleString() : ""} · ${money(r.cost)}`)),
    h("div", { class: "muted mono", style: "font-size:11px;margin-bottom:8px" }, "applied " + appliedLine),
    h("table", { class: "kv" }, h("tbody", {},
      h("tr", { class: "static" }, h("td", { class: "muted" }, "This cluster"), h("td", {}, delta(r.confidence_before, r.confidence_after))),
      h("tr", { class: "static" }, h("td", { class: "muted" }, `${rem.candidate} overall (projected)`),
        h("td", {}, delta(r.confidence_run_before, r.confidence_run_after))))),
    h("div", { class: "grid cols-4", style: "margin:10px 0" },
      stat("fixed", m.to_compatible || 0, (m.to_compatible ? "var(--ok)" : "")),
      stat("regressions", m.regressions || 0, (m.regressions ? "var(--bad)" : "")),
      stat("drifted", m.drifted || 0, (m.drifted ? "var(--drift)" : "")),
      stat("errors", m.errors || 0, (m.errors ? "var(--bad)" : ""))),
    h("div", { class: "muted", style: "font-size:11px;margin-bottom:8px" },
      r.regression_checked
        ? `Regressions observed on ${r.regression_checked} re-checked Compatible evaluation${r.regression_checked === 1 ? "" : "s"}. `
          + "Model output varies between calls, so a small number can be noise — re-check more evaluations before trusting a small difference."
        : "No Compatible evaluations were re-checked, so regressions were not measured."));

  const rows = (r.items || []).slice().sort((a, b) => (a.role === b.role ? 0 : a.role === "target" ? -1 : 1));
  const detail = h("div");
  const tbody = h("tbody", {}, ...rows.map(it => {
    const changed = it.after_label && it.after_label !== it.before_label;
    const tr = h("tr", { tabindex: "0", title: "Show the old vs new candidate response" },
      h("td", { class: "mono" }, it.sample_id),
      h("td", { class: "muted", style: "font-size:11px" }, it.role === "regression_check" ? "regression check" : "cluster"),
      h("td", {}, it.before_label ? h("span", { class: "pill " + it.before_label }, LABEL_TEXT[it.before_label]) : "—"),
      h("td", {}, it.status === "error" ? h("span", { class: "pill incompatible" }, "error")
        : h("span", { class: "pill " + it.after_label }, LABEL_TEXT[it.after_label] || "—")),
      h("td", {}, outcomeTag(it)),
      h("td", { class: "tnum muted" }, it.latency_ms == null ? "—" : `${Math.round(it.latency_ms)} ms`));
    if (changed) tr.classList.add("changed");
    const open = () => retestItemDetail(detail, runId, rem.candidate, it);
    tr.addEventListener("click", open);
    tr.addEventListener("keydown", e => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); open(); } });
    return tr;
  }));
  wrap.append(
    h("table", {}, h("thead", {}, h("tr", {}, ...["Evaluation", "Role", "Before", "After", "Outcome", "Latency"].map(t => h("th", {}, t)))), tbody),
    detail);
  return wrap;
}

function outcomeTag(it) {
  if (it.status === "error") return h("span", { style: "color:var(--bad);font-size:12px" }, "call failed");
  const rank = { compatible: 0, compatible_with_drift: 1, partial: 2, incompatible: 3 };
  const a = rank[it.before_label], b = rank[it.after_label];
  if (a == null || b == null) return h("span", { class: "muted" }, "—");
  if (b < a) return h("span", { style: "color:var(--ok);font-size:12px;font-weight:600" }, b === 0 ? "fixed" : "improved");
  if (b > a) return h("span", { style: "color:var(--bad);font-size:12px;font-weight:600" },
    it.role === "regression_check" && b >= 2 ? "regression" : "worse");
  return h("span", { class: "muted", style: "font-size:12px" }, "unchanged");
}

async function retestItemDetail(host, runId, model, it) {
  clear(host);
  host.append(h("div", { class: "skeleton", style: "margin-top:10px" }));
  let d = null;
  try { d = await api(`/runs/${runId}/samples/${encodeURIComponent(it.sample_id)}?candidate=${encodeURIComponent(model)}`); }
  catch (e) { /* fall back to showing just the new response */ }
  const respText = (r) => {
    if (!r) return "(no response)";
    if (r.tool_calls && r.tool_calls.length) return "tool_call → " + JSON.stringify(r.tool_calls, null, 2);
    return r.text != null ? r.text : "(empty)";
  };
  const oldC = d && d.candidate_result ? d.candidate_result : null;
  const ms = (x) => (x == null ? "—" : `${Math.round(x)} ms`);
  // reuse the evaluation diff: "original" = the candidate's response in the run, "candidate" = after the change
  const before = { model: `${model} (in run)`, latency_ms: oldC ? oldC.latency_ms : null, usage: oldC ? oldC.usage : null,
    response: oldC ? (oldC.status === "error" ? { text: "ERROR: " + (oldC.error || "") } : oldC.response) : { text: "(not available)" } };
  const after = { model: `${model} (after change)`, reasoning_effort: "", latency_ms: it.latency_ms, usage: null,
    status: it.status, error: it.error, response: { text: it.response } };
  const cmp = responseCompare(before, after, respText, ms);
  // relabel the two panes for this context
  const titles = cmp.querySelectorAll ? cmp.querySelectorAll("strong") : [];
  if (titles.length >= 2) { titles[0].textContent = "Before the change (run)"; titles[1].textContent = "After the change"; }
  clear(host);
  host.append(h("div", { class: "card", style: "margin-top:12px" },
    h("div", { class: "row spread" }, h("h3", { style: "margin:0" }, `Evaluation ${it.sample_id}`),
      h("div", { class: "row", style: "gap:8px" }, outcomeTag(it),
        h("button", { class: "btn ghost", onclick: () => { const box = h("div"); host.append(box); showSampleDetail(box, runId, model, it.sample_id); } },
          "Open full evaluation"))),
    cmp,
    it.judge_reason ? h("div", { class: "muted", style: "font-size:12px;margin-top:8px" },
      `Judge after the change: ${it.judge_reason}` + (JUDGE_REASON_TEXT[it.judge_reason] ? ` — ${JUDGE_REASON_TEXT[it.judge_reason]}` : "")) : null));
}

function labelKey(p) { return (p.overridden && p.override_label) ? p.override_label : p.label; }
function normModel(m) { return (m || "").toLowerCase().replace(/^(azure\/|openai\/|bedrock\/converse\/|bedrock\/|us\.openai\.|litellm_proxy\/)/, ""); }

async function renderSettings(main) {
  main.append(topbar("Settings"));
  const s = await api("/settings");
  const draft = { price_table: {} };

  // Defaults form
  const effortSeg = h("div", { class: "seg" }, ...["minimal", "low", "medium", "high"].map(x =>
    h("button", { class: x === s.default_reasoning_effort ? "active" : "",
      onclick: () => { draft.default_reasoning_effort = x; effortSeg.querySelectorAll("button").forEach(b => b.classList.toggle("active", b.textContent === x)); } }, x)));
  const judgeInput = h("input", { type: "text", value: s.judge_model, oninput: e => draft.judge_model = e.target.value });
  const parInput = h("input", { type: "text", value: String(s.parallelism), oninput: e => draft.parallelism = parseInt(e.target.value, 10) });
  const smallInput = h("input", { type: "text", value: String(s.small_sample_threshold), oninput: e => draft.small_sample_threshold = parseInt(e.target.value, 10) });
  const redInput = h("input", { type: "text", value: String(s.high_redaction_threshold), oninput: e => draft.high_redaction_threshold = parseFloat(e.target.value) });

  // LiteLLM proxy connection (used when a run's endpoint is "LiteLLM").
  const llBaseInput = h("input", { type: "text", value: s.litellm_base || "",
    placeholder: "http://127.0.0.1:4000",
    oninput: e => draft.litellm_base = e.target.value });
  const llKeyInput = h("input", { type: "password", autocomplete: "new-password",
    placeholder: s.litellm_key_set ? "•••••••• (key set — leave blank to keep)" : "sk-… (proxy master or virtual key)",
    oninput: e => draft.litellm_key = e.target.value });
  // LiteLLM judge model: a dropdown populated from the proxy's discovered models
  // (falls back to a free-text input if the proxy is unreachable so the saved
  // value stays editable). This is the judge used ONLY when a run routes through
  // LiteLLM; the direct-Bedrock judge above (judge_model) is separate.
  const current = s.judge_litellm_model || "";
  const llJudgeSelect = h("select", { style: "width:auto;min-width:240px",
    onchange: e => draft.judge_litellm_model = e.target.value });
  // seed with the current value so it's shown before discovery returns
  llJudgeSelect.append(h("option", { value: current }, current || "(none)"));
  const llJudgeText = h("input", { type: "text", value: current, placeholder: "claude-sonnet-4.5",
    style: "display:none",
    oninput: e => draft.judge_litellm_model = e.target.value });
  const llJudgeHint = h("div", { class: "muted", style: "font-size:11px;margin-top:4px" }, "Loading proxy models…");
  (async () => {
    let res;
    try { res = await api("/litellm/models"); } catch (e) { res = null; }
    const models = (res && res.models) || [];
    if (!models.length) {
      // Proxy unreachable / empty -> free text so the value is still editable.
      llJudgeSelect.style.display = "none";
      llJudgeText.style.display = "";
      llJudgeHint.textContent = res && res.reason
        ? `Proxy models unavailable (${res.reason}); enter the alias manually.`
        : "Proxy models unavailable; enter the alias manually.";
      return;
    }
    clear(llJudgeSelect);
    const opts = models.includes(current) || !current ? models : [current, ...models];
    opts.forEach(m => {
      const o = h("option", { value: m }, m);
      if (m === current) o.selected = true;
      llJudgeSelect.append(o);
    });
    llJudgeHint.textContent = `${models.length} model group(s) from ${res.base}.`;
  })();
  const llJudgeInput = h("div", {}, llJudgeSelect, llJudgeText, llJudgeHint);

  // Editable price table
  const priceRows = Object.entries(s.price_table).map(([m, p]) => {
    draft.price_table[m] = { input_per_1k: p.input_per_1k, output_per_1k: p.output_per_1k };
    const inIn = h("input", { type: "text", value: String(p.input_per_1k), style: "width:120px",
      oninput: e => draft.price_table[m].input_per_1k = parseFloat(e.target.value) });
    const outIn = h("input", { type: "text", value: String(p.output_per_1k), style: "width:120px",
      oninput: e => draft.price_table[m].output_per_1k = parseFloat(e.target.value) });
    return h("tr", {}, h("td", { class: "mono" }, m), h("td", {}, inIn), h("td", {}, outIn));
  });

  const saveMsg = h("span", { class: "muted", style: "margin-left:12px" }, "");
  const save = async () => {
    saveMsg.textContent = "Saving…";
    try {
      const body = { ...draft };
      // only send changed scalar fields that are valid numbers
      ["parallelism", "small_sample_threshold"].forEach(k => { if (Number.isNaN(body[k])) delete body[k]; });
      if (Number.isNaN(body.high_redaction_threshold)) delete body.high_redaction_threshold;
      SETTINGS = await api("/settings", { method: "PUT", headers: { "content-type": "application/json" }, body: JSON.stringify(body) });
      saveMsg.textContent = "Saved ✓";
      setTimeout(() => { saveMsg.textContent = ""; }, 2500);
    } catch (e) { saveMsg.textContent = "Save failed: " + (e && e.message ? e.message : "error"); }
  };

  main.append(
    h("div", { class: "card" },
      h("h3", {}, "Run defaults"),
      h("div", { class: "grid cols-2" },
        h("label", { class: "field" }, h("span", {}, "Default reasoning effort"), effortSeg),
        h("label", { class: "field" }, h("span", {}, "Concurrency (parallel calls)"), parInput)),
      h("label", { class: "field" }, h("span", {}, "Judge model — direct Bedrock (inference-profile id; used when endpoint is Bedrock)"), judgeInput),
      h("div", { class: "grid cols-2" },
        h("label", { class: "field" }, h("span", {}, "Small-sample caveat threshold (evaluable rows)"), smallInput),
        h("label", { class: "field" }, h("span", {}, "High-redaction caveat threshold (0–1)"), redInput))),
    h("div", { class: "card" },
      h("h3", {}, "LiteLLM proxy"),
      h("div", { class: "muted", style: "margin:-4px 0 10px" },
        "Used when a run's endpoint is set to LiteLLM — calls are proxied to Bedrock through this gateway instead of hitting Bedrock directly. Changes apply immediately (no restart)."),
      h("label", { class: "field" }, h("span", {}, "Proxy base URL"), llBaseInput),
      h("label", { class: "field" }, h("span", {}, s.litellm_key_set ? "API key (a key is set)" : "API key"), llKeyInput),
      h("label", { class: "field" }, h("span", {}, "Judge model — LiteLLM path (pick from your proxy's models; used when endpoint is LiteLLM)"), llJudgeInput)),
    h("div", { class: "card" },
      h("h3", {}, "Price table (USD per 1K tokens)"),
      h("table", {}, h("thead", {}, h("tr", {}, ...["Model", "Input /1K", "Output /1K"].map(t => h("th", {}, t)))),
        h("tbody", {}, ...priceRows))),
    h("div", { class: "row", style: "margin-top:16px" },
      h("button", { class: "btn", onclick: save }, "Save settings"), saveMsg));
}

boot();
