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
  { key: "d7_consistency", code: "D7", name: "Consistency", desc: "How alike the model's answers are when the same request is replayed (replays only). Doesn't change any answer's label or Migration Confidence." },
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
  // Deep link: ?run=<run_id> opens that run directly (shareable; ids are validated by the API).
  const linkRun = new URLSearchParams(location.search).get("run");
  if (linkRun && /^[A-Za-z0-9_-]{1,64}$/.test(linkRun)) { go("run", linkRun); return; }
  const linkView = new URLSearchParams(location.search).get("view");
  if (["settings", "guide", "runs"].includes(linkView)) { go(linkView); return; }
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

// ---- Runs list (Option A): summary tiles, search + status filters, sort, and per-run rows
// with the Best overall pick and a confidence strip of every candidate. Live runs show progress.
const runsUi = { filter: "all", q: "", sort: "newest" };
const RUN_STATUS = {
  done: ["ok", "✓ Done"], failed: ["bad", "✗ Failed"], cancelled: ["mu", "Cancelled"],
  ingested: ["run", "● Ready to configure"], created: ["mu", "Created"], ingesting: ["run", "● Loading logs"],
  planning: ["run", "● Running"], replaying: ["run", "● Running"], scoring: ["run", "● Running"], aggregating: ["run", "● Finishing"],
};
const LIVE = new Set(["planning", "replaying", "scoring", "aggregating", "ingesting"]);
const runGroup = (r) => r.status === "done" ? "done" : r.status === "failed" ? "failed"
  : LIVE.has(r.status) ? "running" : "notrun";
const runWhen = (ts) => {
  if (!ts) return ["—", ""];
  const d = new Date(ts * 1000), now = new Date();
  const t = d.toLocaleTimeString([], { hour: "numeric", minute: "2-digit" });
  const sameDay = (a, b) => a.toDateString() === b.toDateString();
  const yest = new Date(now); yest.setDate(now.getDate() - 1);
  const label = sameDay(d, now) ? `Today ${t}` : sameDay(d, yest) ? `Yesterday ${t}`
    : `${d.toLocaleDateString([], { month: "short", day: "numeric", ...(d.getFullYear() !== now.getFullYear() ? { year: "numeric" } : {}) })} ${t}`;
  const mins = Math.max(0, Math.round((now - d) / 60000));
  const ago = mins < 1 ? "just now" : mins < 60 ? `${mins} min ago` : mins < 1440 ? `${Math.round(mins / 60)} h ago`
    : `${Math.round(mins / 1440)} day${Math.round(mins / 1440) === 1 ? "" : "s"} ago`;
  return [label, ago];
};
const bestConf = (r) => Math.max(-1, ...(r.verdict_summary || []).map(v => v.confidence ?? -1));

// Every candidate's confidence on one 0–100 line; recommended one ringed; gate tick marked.
function confStrip(r) {
  const vs = r.verdict_summary || [];
  if (!vs.length) return h("span", { class: "muted" }, "—");
  const gate = (SETTINGS && SETTINGS.recommendation && SETTINGS.recommendation.min_confidence) ?? 65;
  const desc = vs.map(v => `${v.model} ${Math.round(v.confidence)}% (${BAND_TEXT[v.band] || v.band})`).join(", ");
  return h("div", {},
    h("div", { class: "cstrip", role: "img", title: desc,
      "aria-label": `Candidate confidence: ${desc}. Quality gate at ${gate}.` + (r.best_overall ? ` Recommended: ${r.best_overall}.` : "") },
      h("span", { class: "gate", style: `left:${gate}%` }),
      ...[...vs].sort((a, b) => (a.model === r.best_overall) - (b.model === r.best_overall)).map(v =>
        h("i", { class: v.model === r.best_overall ? "best" : "",
          style: `left:${Math.max(0, Math.min(100, v.confidence || 0))}%;background:${BAND_COLOR[v.band] || "var(--muted)"}` }))),
    h("div", { class: "sub" }, `${vs.length} candidate${vs.length === 1 ? "" : "s"}`));
}

function runRecCell(r) {
  if (r.status === "failed") return h("div", { class: "errline", title: r.error || "" }, r.error || "Run failed");
  if (r.progress) {
    const p = r.progress, pct = p.total ? 100 * p.done / p.total : 0;
    let left = "";
    if (p.done > 0 && p.started_ts && p.last_ts && p.last_ts > p.started_ts && p.total > p.done)
      left = ` · ~${fmtDur((p.total - p.done) * (p.last_ts - p.started_ts) / p.done)} left`;
    return h("div", {},
      h("div", { class: "rbar", role: "progressbar", "aria-valuemin": "0", "aria-valuemax": String(p.total || 0),
        "aria-valuenow": String(p.done), "aria-label": `${p.done} of ${p.total} calls done` },
        h("span", { style: `width:${pct}%` })),
      h("div", { class: "sub" }, p.stage === "aggregating" ? "computing verdicts…" : `${p.done} / ${p.total || "?"} calls${left}`,
        p.inflight ? ` · ${p.inflight} in flight` : "",
        p.throttled ? h("span", { style: "color:var(--partial)" }, " · throttled") : ""));
  }
  if (r.status === "ingested") return h("div", { class: "row", style: "gap:8px" },
    h("span", { class: "sub" }, "Logs loaded, not run yet"),
    h("button", { class: "btn ghost sm", onclick: (e) => { e.stopPropagation(); continueSetup(r.run_id); } }, "Continue setup →"));
  if (r.status === "cancelled") return h("span", { class: "sub" }, "Stopped before results");
  if (r.status !== "done") return h("span", { class: "muted" }, "—");
  if (r.none_eligible || !r.best_overall) return h("div", {},
    h("span", { class: "band needs_work" }, "No candidate ready"),
    h("div", { class: "sub" }, "none passed the quality gate"));
  const b = (r.verdict_summary || []).find(v => v.model === r.best_overall) || {};
  return h("div", {},
    h("div", {}, h("span", { class: "mono" }, r.best_overall), " ", h("span", { class: "muted tnum" }, fmtPct(b.confidence ?? 0))),
    h("div", { class: "row", style: "gap:6px;margin-top:2px" },
      b.band ? h("span", { class: "band " + b.band }, BAND_TEXT[b.band]) : null, h("span", { class: "sub" }, "Best overall")));
}

async function continueSetup(runId) {
  try {
    const info = await api(`/runs/${runId}`);
    state.draft = { name: info.name || "", step: "config", runId, ingestion: info.ingestion, candidates: [],
                    callPath: "bedrock", endpoint: "bedrock_runtime", effort: "medium" };
    go("new");
  } catch (e) { alert("Could not open this run: " + (e && e.message ? e.message : "error")); }
}

function runMenu(r) {
  const del = async (e) => {
    e.stopPropagation();
    if (!confirm(`Delete run "${r.name || r.run_id}"? This cannot be undone.`)) return;
    try { await api(`/runs/${r.run_id}`, { method: "DELETE" }); go("runs"); }
    catch (err) { alert("Delete failed: " + (err && err.message ? err.message : "error")); }
  };
  const menu = h("details", { class: "rowmenu", onclick: (e) => e.stopPropagation() },
    h("summary", { "aria-label": `Actions for ${r.name || r.run_id}`, title: "Actions" }, "⋯"),
    h("div", { class: "menu", role: "menu" },
      h("button", { role: "menuitem", onclick: (e) => { e.stopPropagation(); r.status === "ingested" ? continueSetup(r.run_id) : go("run", r.run_id); } },
        r.status === "done" ? "Open results" : r.status === "ingested" ? "Continue setup" : "Open"),
      r.status === "done" ? h("a", { role: "menuitem", href: `${API}/runs/${r.run_id}/report.pdf`, target: "_blank", rel: "noopener" }, "Download PDF") : null,
      LIVE.has(r.status) ? null : h("button", { role: "menuitem", class: "danger", onclick: del }, "Delete")));
  return menu;
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
  const card = h("div", { class: "card runs" });
  main.append(card);
  // Server-wide parallel calls, shown when two or more runs are live (they share the cap).
  const serverStrip = h("div", { class: "rp-now", style: "display:none;margin:0 0 10px" });
  if (runs.filter(r => r.progress).length >= 2) {
    api("/concurrency").then(cs => {
      const waiting = (cs.runs || []).filter(x => x.waiting).length;
      serverStrip.replaceChildren(h("b", {}, `${cs.server_inflight} of ${cs.server_cap}`),
        ` calls in flight across ${(cs.runs || []).length} runs`,
        waiting ? h("span", { style: "color:var(--partial)" }, ` · ${waiting} waiting for capacity`) : "",
        h("span", { class: "muted" }, " · limit set in Settings"));
      serverStrip.style.display = "";
    }).catch(() => {});
  }

  const count = (g) => runs.filter(r => runGroup(r) === g).length;
  const done = runs.filter(r => r.status === "done");
  const live = runs.filter(r => r.progress);
  const spend = runs.reduce((a, r) => a + (r.spend || 0), 0);
  const oldest = Math.min(...runs.map(r => r.created_at || Infinity));
  const tile = (l, v, s) => h("div", { class: "rp-kpi" }, h("div", { class: "l" }, l), h("div", { class: "v" }, v), h("div", { class: "s" }, s));
  const liveDone = live.reduce((a, r) => a + (r.progress.done || 0), 0), liveTotal = live.reduce((a, r) => a + (r.progress.total || 0), 0);
  const tiles = h("div", { class: "rp-kpis" },
    tile("Runs", String(runs.length), isFinite(oldest) ? `since ${new Date(oldest * 1000).toLocaleDateString([], { month: "short", day: "numeric" })}` : ""),
    tile("Completed", String(done.length), `${done.filter(r => r.none_eligible).length} with no candidate ready`),
    tile("Running", String(live.length), live.length ? `${liveTotal ? Math.round(100 * liveDone / liveTotal) : 0}% of calls done` : "none right now"),
    tile("Failed", String(count("failed")), count("failed") ? "open a run to see why" : "no failures"),
    tile("Total spend", fmtUsd(spend), runs.some(r => r.spend_unknown) ? "some prices unknown" : "list price, no caching"));

  const search = h("input", { type: "search", class: "runsearch", placeholder: "Search name, run ID or model…",
    "aria-label": "Search runs", value: runsUi.q });
  const segBtns = (opts, key, cb) => h("div", { class: "seg", role: "group" }, ...opts.map(([k, t]) =>
    h("button", { class: runsUi[key] === k ? "active" : "", "aria-pressed": runsUi[key] === k ? "true" : "false",
      onclick: () => { runsUi[key] = k; cb(); } }, t)));
  const filters = () => segBtns([["all", ["All", h("span", { class: "n" }, String(runs.length))]],
    ["done", ["Done", h("span", { class: "n" }, String(count("done")))]],
    ["running", ["Running", h("span", { class: "n" }, String(count("running")))]],
    ["failed", ["Failed", h("span", { class: "n" }, String(count("failed")))]],
    ["notrun", ["Not run", h("span", { class: "n" }, String(count("notrun")))]]], "filter", draw);
  const sorts = () => segBtns([["newest", "Newest"], ["name", "Name"], ["confidence", "Best confidence"]], "sort", draw);
  const filterBox = h("span", {}), sortBox = h("span", {});
  const tbody = h("tbody");
  const empty = h("div", { class: "muted", style: "display:none;padding:14px 4px" }, "No runs match.");
  const heads = [["Run"], ["Status"], ["When"], ["Evals", 1], ["Recommendation"], ["Candidates (confidence 0–100)"], ["Spend", 1], [""]];

  function draw() {
    filterBox.replaceChildren(filters()); sortBox.replaceChildren(sorts());
    const q = runsUi.q.trim().toLowerCase();
    let rows = runs.filter(r => runsUi.filter === "all" || runGroup(r) === runsUi.filter)
      .filter(r => !q || [r.name, r.run_id, r.best_overall, ...(r.verdict_summary || []).map(v => v.model)]
        .some(x => x && String(x).toLowerCase().includes(q)));
    rows.sort(runsUi.sort === "name" ? (a, b) => (a.name || "").localeCompare(b.name || "")
      : runsUi.sort === "confidence" ? (a, b) => bestConf(b) - bestConf(a)
      : (a, b) => (b.created_at || 0) - (a.created_at || 0));
    tbody.replaceChildren(...rows.map(r => {
      const [whenTxt, ago] = runWhen(r.created_at);
      const [cls, txt] = RUN_STATUS[r.status] || ["mu", r.status];
      const open = () => (r.status === "ingested" ? continueSetup(r.run_id) : go("run", r.run_id));
      return h("tr", { tabindex: "0", title: "Open this run", onclick: open,
        onkeydown: (e) => { if (e.key === "Enter") { e.preventDefault(); open(); } } },
        h("td", {}, h("div", { style: "font-weight:600" }, r.name || "untitled"), h("div", { class: "sub mono" }, r.run_id)),
        h("td", {}, h("span", { class: "rst " + cls }, txt)),
        h("td", {}, h("div", {}, whenTxt), h("div", { class: "sub" }, ago)),
        h("td", { class: "num tnum" }, String(r.evaluations ?? "—"),
          (r.replays || 1) > 1 ? h("span", { class: "chip", style: "margin-left:6px", title: `Each request sent ${r.replays} times` }, `×${r.replays}`) : null),
        h("td", {}, runRecCell(r)),
        h("td", {}, r.progress ? h("span", { class: "sub" }, "results when done") : confStrip(r)),
        h("td", { class: "num tnum" }, r.spend == null ? h("span", { class: "muted" }, "—")
          : [fmtUsd(r.spend), r.progress ? h("div", { class: "sub" }, "so far") : null, r.spend_unknown ? h("div", { class: "sub" }, "+ unknown") : null]),
        h("td", { class: "num" }, runMenu(r)));
    }));
    empty.style.display = rows.length ? "none" : "";
  }
  search.addEventListener("input", () => { runsUi.q = search.value; draw(); });

  card.append(
    h("div", { class: "row spread" }, h("h3", { style: "margin:0" }, "All runs"), h("button", { class: "btn", onclick: () => go("new") }, "+ New Run")),
    tiles,
    serverStrip,
    h("div", { class: "row spread", style: "flex-wrap:wrap;gap:10px;margin-bottom:6px" },
      h("div", { class: "row", style: "gap:10px;flex-wrap:wrap" }, search, filterBox),
      h("div", { class: "row", style: "gap:6px" }, h("span", { class: "muted", style: "font-size:12px" }, "Sort"), sortBox)),
    h("div", { class: "table-scroll" }, h("table", { class: "runs-table" },
      h("thead", {}, h("tr", {}, ...heads.map(([t, num]) => h("th", { scope: "col", class: num ? "num" : "" }, t)))), tbody)),
    empty,
    h("div", { class: "legend", style: "margin-top:10px" },
      ...[["safe_drop_in"], ["drop_in_with_prompt_tuning"], ["needs_work"], ["not_a_drop_in"]].map(([b]) =>
        h("span", {}, h("i", { class: "dot", style: `background:${BAND_COLOR[b]}` }), BAND_TEXT[b])),
      h("span", {}, "ringed dot = recommended · red tick = quality gate")));
  draw();

  // While anything is running, refresh the list every 3 s (stops once you leave the page;
  // waits while you're typing in search or have a row menu open, so nothing is lost).
  if (live.length || runs.some(r => r.status === "ingesting")) {
    const tick = () => {
      if (!document.body.contains(card) || state.view !== "runs") return;
      if (document.activeElement === search || card.querySelector("details.rowmenu[open]")) { setTimeout(tick, 3000); return; }
      clear(main); renderRuns(main);
    };
    setTimeout(tick, 3000);
  }
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
  const fileInput = h("input", { type: "file", accept: ".json,.jsonl,.ndjson,.log,.gz,.zip", style: "display:none", onchange: e => uploadThenIngest(main, d, e.target.files[0]) });
  const dz = h("div", { class: "dropzone", onclick: () => fileInput.click() },
    "Drop a LiteLLM log file — or a .zip of log folders, consolidated into one run — or click to browse", fileInput);
  const preview = h("div", { id: "ingPreview" });
  main.append(h("div", { class: "card" },
    h("h3", {}, "1 · Source"),
    h("label", { class: "field" }, h("span", {}, "Run name"), nameInput),
    h("div", { class: "grid cols-2" },
      h("div", {}, h("div", { class: "muted", style: "margin-bottom:8px" }, "Upload file"), dz),
      h("div", {}, h("div", { class: "muted", style: "margin-bottom:8px" }, "or S3 location"),
        s3Input, h("button", { class: "btn secondary", style: "margin-top:8px", onclick: () => s3Ingest(main, d, s3Input.value) }, "Validate & ingest"))),
    manualSourceBlock(main, d),
    preview));
}

// "Enter prompts manually": no log file — the user types one or more golden
// request/response pairs that become the dataset. Posts manual_prompts to /ingest;
// everything downstream (normalize, score, judge, aggregate) is unchanged.
function manualSourceBlock(main, d) {
  d.manualRows = d.manualRows && d.manualRows.length ? d.manualRows : [{ prompt: "", golden: "", instructions: "" }];
  const list = h("div", {});
  const taStyle = "width:100%;font:inherit;color:var(--text);background:var(--bg);border:1px solid var(--border);border-radius:var(--radius-sm);padding:8px;resize:vertical";

  function renderRows() {
    clear(list);
    d.manualRows.forEach((row, i) => {
      const promptTa = h("textarea", { rows: "3", placeholder: "Prompt sent to the model (the request)", style: taStyle,
        oninput: e => { row.prompt = e.target.value; } });
      promptTa.value = row.prompt || "";
      const goldenTa = h("textarea", { rows: "3", placeholder: "Response your current model gave (the golden answer)", style: taStyle,
        oninput: e => { row.golden = e.target.value; } });
      goldenTa.value = row.golden || "";
      const instrTa = h("textarea", { rows: "2", placeholder: "Optional system / developer instructions", style: taStyle,
        oninput: e => { row.instructions = e.target.value; } });
      instrTa.value = row.instructions || "";
      const header = h("div", { class: "row spread", style: "margin-bottom:6px" },
        h("span", { class: "muted mono", style: "font-size:12px" }, `Prompt ${i + 1}`),
        d.manualRows.length > 1
          ? h("button", { class: "btn ghost", style: "padding:2px 8px", onclick: () => { d.manualRows.splice(i, 1); renderRows(); } }, "Remove")
          : null);
      list.append(h("div", { class: "g-mini", style: "margin-bottom:10px;padding:10px" },
        header,
        h("label", { class: "field" }, h("span", {}, "Input prompt"), promptTa),
        h("label", { class: "field", style: "margin-top:8px" }, h("span", {}, "Golden response"), goldenTa),
        h("label", { class: "field", style: "margin-top:8px" }, h("span", {}, "Instructions (optional)"), instrTa)));
    });
  }
  renderRows();

  return h("details", { class: "card", style: "margin-top:12px;background:transparent" },
    h("summary", { style: "cursor:pointer;font-weight:600" }, "or enter prompts manually (no log file)"),
    h("div", { class: "muted", style: "margin:8px 0" },
      "Type one or more golden request/response pairs. Each becomes an evaluation that is replayed against the candidate models — exactly like a logged request."),
    list,
    h("div", { class: "row", style: "gap:8px;margin-top:4px" },
      h("button", { class: "btn secondary", onclick: () => { d.manualRows.push({ prompt: "", golden: "", instructions: "" }); renderRows(); } }, "+ Add another prompt"),
      h("button", { class: "btn", onclick: () => manualIngest(main, d) }, "Use these prompts")));
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
    status.textContent = /\.zip$/i.test(file.name)
      ? "Expanding zip and normalizing every log file into one run…"
      : "Parsing & normalizing log rows…";
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

async function manualIngest(main, d) {
  const rows = (d.manualRows || [])
    .map(r => ({ prompt: (r.prompt || "").trim(), golden: (r.golden || "").trim(), instructions: (r.instructions || "").trim() }))
    .filter(r => r.prompt && r.golden);
  const pv = $("#ingPreview"); clear(pv);
  if (!rows.length) {
    pv.append(h("div", { class: "banner err" }, "Enter at least one prompt with a golden response."));
    return;
  }
  pv.append(h("div", { class: "skeleton" }));
  try {
    await ensureRun(d);
    const ing = await api(`/runs/${d.runId}/ingest`, { method: "POST", headers: { "content-type": "application/json" },
      body: JSON.stringify({ manual_prompts: rows }) });
    d.ingestion = ing.ingestion; showIngestion(d);
  } catch (e) { clear(pv); pv.append(h("div", { class: "banner err" }, "Ingestion failed: " + (e && e.message ? e.message : "unknown error"))); }
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

// ---- Candidate picker (candidate-picker spec, Option A) ----------------------
// Left: "Your candidates", each with its own reasoning efforts (each effort = one run).
// Right: the endpoint's live model list, searchable, suggested first, providers grouped.
const PICK_ENDPOINT_TEXT = { bedrock_runtime: "Bedrock Runtime", bedrock_mantle: "Bedrock Mantle", litellm: "LiteLLM proxy" };
const AVAIL = {};   // endpoint -> /models/available response (per page load)

function defaultEffortFor(info) {
  const ok = (info && info.efforts) || [];
  if (!info || info.effort_applies === false) return ok[0] || "medium";
  const def = (MODELS && MODELS.default_reasoning_effort) || "medium";
  return ok.includes(def) ? def : (ok.includes("medium") ? "medium" : (ok[0] || def));
}

function candidatePicker(d, legacy, onChange) {
  const tray = h("div", { class: "pk-tray" });
  const browse = h("div", { class: "pk-browse" });
  const status = h("div", { class: "pk-live" });
  const ui = { query: "", provider: "all", allProviders: false, list: null };
  const info = (name) => ((ui.list && ui.list.models) || []).find(m => m.name === name);
  const probing = new Set();
  const probeErr = {};

  const changed = () => { d.candidates = d.picks.map(p => p.model); drawTray(); drawBrowse(); onChange(); };

  function addPick(name) {
    if (d.picks.some(p => p.model === name)) return;
    const m = info(name);
    d.picks.push({ model: name, efforts: [defaultEffortFor(m)] });
    if (m && m.effort_applies !== false && !m.efforts_checked) probe(name);
    changed();
  }
  const removePick = (name) => { d.picks = d.picks.filter(p => p.model !== name); changed(); };

  async function probe(name) {
    probing.add(name); delete probeErr[name]; drawTray();
    try {
      const r = await api("/models/probe-efforts", { method: "POST", headers: { "content-type": "application/json" },
        body: JSON.stringify({ name }) });
      const m = info(name);
      if (m) { m.efforts = r.efforts; m.efforts_checked = !!r.efforts; }
      if (!r.efforts) probeErr[name] = Object.values(r.errors || {})[0] || "no effort was accepted";
      const p = d.picks.find(x => x.model === name);
      if (p && r.efforts) {
        p.efforts = p.efforts.filter(e => r.efforts.includes(e));
        if (!p.efforts.length) p.efforts = [defaultEffortFor(m)];
      }
    } catch (e) { probeErr[name] = e && e.message ? e.message : "check failed"; }
    probing.delete(name); changed();
  }

  function tags(m) {
    if (!m) return [];
    return [m.suggested ? h("span", { class: "pk-tag sug" }, "suggested") : null,
      m.like_for_like ? h("span", { class: "pk-tag peer" }, "like-for-like") : null,
      (m.catalog && m.provider !== "OpenAI") ? h("span", { class: "pk-tag" }, "cross-family") : null,
      m.price ? null : h("span", { class: "pk-tag warn", title: "No list price configured: cost is left out of ranking" }, "no price")]
      .filter(Boolean);
  }
  const priceText = (m) => m && m.price ? `$${m.price.input_per_1m.toFixed(2)} / $${m.price.output_per_1m.toFixed(2)} per 1M` : "no price";

  function effortChips(p) {
    const m = info(p.model);
    if (m && m.effort_applies === false) {
      return h("div", { class: "pk-eff" }, h("span", { class: "pk-chip on" }, "default"),
        h("span", { class: "muted", style: "font-size:11px" }, "reasoning effort doesn't apply on this path"));
    }
    if (probing.has(p.model)) return h("div", { class: "pk-eff muted", style: "font-size:11px" }, "Checking which efforts this model accepts…");
    const ok = (m && m.efforts) || null;
    if (!ok) {
      return h("div", { class: "pk-eff" },
        h("span", { class: "pk-chip on" }, p.efforts[0] || "medium"),
        h("span", { class: "muted", style: "font-size:11px" },
          probeErr[p.model] ? `Couldn't check efforts (${probeErr[p.model].slice(0, 90)}). ` : "Efforts not checked yet. "),
        h("button", { class: "linkbtn", onclick: () => probe(p.model) }, "Check efforts"));
    }
    return h("div", { class: "pk-eff", role: "group", "aria-label": `Reasoning efforts for ${p.model}` }, ...ok.map(e => {
      const on = p.efforts.includes(e);
      return h("button", { class: "pk-chip" + (on ? " on" : ""), "aria-pressed": on ? "true" : "false",
        title: on ? `Remove ${p.model}@${e}` : `Also run ${p.model}@${e}`,
        onclick: () => {
          if (on && p.efforts.length === 1) return;       // keep at least one effort
          p.efforts = on ? p.efforts.filter(x => x !== e) : ok.filter(x => x === e || p.efforts.includes(x));
          changed();
        } }, e);
    }));
  }

  function drawTray() {
    clear(tray);
    const runs = d.picks.reduce((a, p) => a + p.efforts.length, 0);
    tray.append(h("div", { class: "row spread", style: "margin-bottom:4px" },
      h("b", {}, `Your candidates · ${d.picks.length} model${d.picks.length === 1 ? "" : "s"} → ${runs} run${runs === 1 ? "" : "s"}`),
      h("span", { class: "muted", style: "font-size:11px" }, "click an effort to add or remove it")));
    if (!d.picks.length) {
      tray.append(h("div", { class: "muted", style: "padding:18px 6px" }, "No candidates yet. Add models from the list on the right."));
      return;
    }
    d.picks.forEach(p => {
      const m = info(p.model);
      tray.append(h("div", { class: "pk-trow" },
        h("div", {}, h("b", { class: "mono" }, p.model), " ", ...tags(m),
          h("div", { class: "pk-meta" }, `${(m && m.provider) || "—"} · ${priceText(m)}`)),
        h("button", { class: "pk-x", title: `Remove ${p.model}`, "aria-label": `Remove ${p.model}`, onclick: () => removePick(p.model) }, "×"),
        h("div", { style: "grid-column:1/3" }, effortChips(p))));
    });
    tray.append(h("div", { class: "pk-runs mono" }, d.picks.flatMap(p => p.efforts.map(e =>
      (info(p.model) || {}).effort_applies === false ? p.model : `${p.model}@${e}`)).join(" · ")));
  }

  function drawBrowse() {
    clear(browse);
    const list = (ui.list && ui.list.models) || [];
    const q = ui.query.trim().toLowerCase();
    const match = (m) => !q || [m.name, m.display, m.provider, m.model_id].some(x => (x || "").toLowerCase().includes(q));
    const counts = {};
    list.forEach(m => { counts[m.provider] = (counts[m.provider] || 0) + 1; });
    const provs = Object.entries(counts).sort((a, b) => b[1] - a[1]);
    const shown = ui.allProviders ? provs : provs.slice(0, 4);
    const search = h("input", { type: "search", class: "pk-search", placeholder: `Search ${list.length} models…`,
      value: ui.query, "aria-label": "Search models",
      oninput: (e) => { ui.query = e.target.value; const pos = e.target.selectionStart; drawBrowse();
        const s = browse.querySelector(".pk-search"); s.focus(); s.setSelectionRange(pos, pos); } });
    const chip = (key, label, n) => h("button", { class: "pk-chip" + (ui.provider === key ? " on" : ""),
      onclick: () => { ui.provider = key; drawBrowse(); } }, label, h("span", { class: "n" }, ` ${n}`));
    browse.append(search, h("div", { class: "pk-provs" }, chip("all", "All", list.length),
      ...shown.map(([p, n]) => chip(p, p, n)),
      provs.length > 4 ? h("button", { class: "pk-chip", onclick: () => { ui.allProviders = !ui.allProviders; drawBrowse(); } },
        ui.allProviders ? "fewer" : `+${provs.length - 4} more`) : null));

    const visible = list.filter(m => match(m) && (ui.provider === "all" || m.provider === ui.provider));
    const row = (m) => {
      const on = d.picks.some(p => p.model === m.name);
      const r = h("div", { class: "pk-lrow", role: "button", tabindex: "0", "aria-pressed": on ? "true" : "false",
        title: on ? "Remove from candidates" : "Add to candidates" },
        h("span", { class: "pk-plus" + (on ? " done" : "") }, on ? "✓" : "+"),
        h("div", { class: "grow" }, h("b", { class: "mono" }, m.name), " ", ...tags(m).filter(t => !t.classList.contains("sug")),
          h("div", { class: "pk-meta" }, (m.catalog ? "built-in" : m.model_id)
            + " · " + (m.price ? "priced" : "no price")
            + (m.effort_applies === false ? " · effort n/a" : (m.efforts_checked ? " · efforts known" : " · efforts checked on add")))));
      const toggle = () => (on ? removePick(m.name) : addPick(m.name));
      r.addEventListener("click", toggle);
      r.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); toggle(); } });
      return r;
    };
    const body = h("div", { class: "pk-list" });
    const sug = visible.filter(m => m.suggested);
    if (sug.length) body.append(h("div", { class: "pk-group" }, h("span", {}, `Suggested for ${legacy || "your model"}`), h("span", {}, String(sug.length))), ...sug.map(row));
    const rest = visible.filter(m => !m.suggested);
    const groups = {};
    rest.forEach(m => { (groups[m.provider] = groups[m.provider] || []).push(m); });
    const order = Object.keys(groups).sort((a, b) => (groups[b].some(m => m.catalog) - groups[a].some(m => m.catalog)) || groups[b].length - groups[a].length);
    order.forEach((p, i) => {
      // Expanded: while searching, when filtered to one provider, or the first (built-in) group.
      const open = !!q || ui.provider !== "all" || i === 0;
      const det = h("details", { class: "pk-det" },
        h("summary", { class: "pk-group" }, h("span", {}, p), h("span", {}, String(groups[p].length))), ...groups[p].map(row));
      if (open) det.open = true;
      body.append(det);
    });
    if (!visible.length) body.append(h("div", { class: "muted", style: "padding:12px 4px" },
      list.length ? "No model matches your search." : "No models to show."));
    browse.append(body);
  }

  async function load(refresh = false) {
    const ep = d.endpoint || "bedrock_runtime";
    clear(status); status.append(h("span", { class: "muted" }, `Listing models from ${PICK_ENDPOINT_TEXT[ep]}…`));
    let res = (!refresh && AVAIL[ep]) || null;
    if (!res) {
      try { res = await api(`/models/available?endpoint=${encodeURIComponent(ep)}&legacy=${encodeURIComponent(legacy || "")}${refresh ? "&refresh=true" : ""}`); }
      catch (e) { res = { models: [], reason: e && e.message ? e.message : "request failed", count: 0 }; }
      if (!res.reason) AVAIL[ep] = res;
    }
    ui.list = res;
    // Keep only picks this endpoint can run (e.g. switching to LiteLLM drops unserved models).
    const names = new Set((res.models || []).map(m => m.name));
    d.picks = d.picks.filter(p => names.has(p.model));
    if (!d.picks.length && !d.pickedOnce) {
      d.picks = (res.models || []).filter(m => m.suggested).map(m => ({ model: m.name, efforts: [defaultEffortFor(m)] }));
    }
    d.pickedOnce = true;
    clear(status);
    status.append(h("i", { class: res.reason ? "bad" : "" }),
      res.reason ? `${PICK_ENDPOINT_TEXT[ep]}: ${res.reason}. Showing built-in models only. `
        : `Listed live from ${PICK_ENDPOINT_TEXT[ep]}${res.region && ep !== "litellm" ? " · " + res.region : ""} · ${res.listed_count} models · `,
      h("button", { class: "linkbtn", onclick: () => load(true) }, "refresh"));
    changed();
  }

  return { tray, browse, status, load };
}

function newRunConfig(main, d) {
  const legacy = d.ingestion?.detected_models?.[0]?.model;
  const entry = (MODELS?.matrix || []).find(e => normModel(e.legacy_model) === normModel(legacy));
  const mapped = !!entry;
  // Candidates + per-model efforts. Migrates state from the older global-effort form.
  if (!Array.isArray(d.picks)) {
    const effs = (Array.isArray(d.efforts) && d.efforts.length) ? d.efforts : [d.effort || "medium"];
    d.picks = (d.candidates || []).map(m => ({ model: m, efforts: candidateEfforts(m, effs).length ? candidateEfforts(m, effs) : [effortsFor(m)[0]] }));
    if (d.picks.length) d.pickedOnce = true;
  }
  const picker = candidatePicker(d, legacy, () => updatePlan(d));
  const ENDPOINTS = [
    ["bedrock_runtime", "Bedrock Runtime", "us.openai.* · /openai/v1 · Luna/Terra/Sol/GPT-5.4"],
    ["bedrock_mantle", "Bedrock Mantle", "openai.* and other providers · /openai/v1 and /v1 · signed with your AWS credentials"],
    ["litellm", "LiteLLM proxy", "your proxy · /v1/chat/completions"],
  ];
  // If a mantle-only model is selected, force Mantle and lock the others.
  const hasMantleOnly = d.candidates.some(m => MODELS?.models?.[m] && MODELS.models[m].runtime_supported === false);
  if (hasMantleOnly) d.endpoint = "bedrock_mantle";
  const pathSeg = h("div", { class: "seg" }, ...ENDPOINTS.map(([v, t, hint]) =>
    h("button", {
      class: v === d.endpoint ? "active" : "",
      title: hasMantleOnly && v !== "bedrock_mantle" ? "Disabled — a mantle-only model is selected" : hint,
      disabled: hasMantleOnly && v !== "bedrock_mantle",
      onclick: () => { d.endpoint = v; pathSeg.querySelectorAll("button").forEach((b) => b.classList.toggle("active", b.textContent === t)); picker.load(); updatePlan(d); },
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
    h("div", { class: "row spread", style: "align-items:flex-end;gap:14px;flex-wrap:wrap" },
      h("label", { class: "field", style: "margin:0" }, h("span", {}, "Call models through"), pathSeg, endpointHint),
      picker.status),
    h("div", { class: "muted", style: "margin:12px 0 8px" }, "Candidate models" + (
      mapped ? ` (suggested for ${legacy} are pre-selected)`
             : ` — “${legacy || "unknown"}” isn’t in the migration matrix; add any models to compare`)),
    h("div", { class: "pk-grid" }, picker.tray, picker.browse),
    replayField(d),
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
  picker.load();
}

// Which of the selected efforts each model actually runs. Unsupported ones are skipped (Bedrock
// would reject every call with a 400); a model whose call path ignores effort (Claude) runs once.
function candidateEfforts(model, selected) {
  const ok = effortsFor(model);
  if (!effortApplies(model)) return [ok[0]];
  const keep = selected.filter(e => ok.includes(e));
  return keep.length ? keep : [];
}

function effortSkipNote(models, selected) {
  const notes = [];
  (models || []).forEach(m => {
    const ok = effortsFor(m);
    if (!effortApplies(m)) {
      if (selected.length > 1 || selected[0] !== ok[0]) notes.push(`${m}: reasoning effort isn't applied on this model, so it runs once`);
      return;
    }
    const skipped = selected.filter(e => !ok.includes(e));
    if (skipped.length === selected.length) notes.push(`${m}: none of the selected efforts are supported — it will not run (supports ${ok.join(", ")})`);
    else if (skipped.length) notes.push(`${m}: skipping ${skipped.join(", ")} (supports ${ok.join(", ")})`);
  });
  return notes.join(" · ");
}

async function updatePlan(d) {
  const box = $("#planBox"); if (!box) return;
  if (!(d.picks || []).length) { box.textContent = "add a candidate"; return; }
  try {
    const plan = await api(`/runs/${d.runId}/plan`, { method: "POST", headers: { "content-type": "application/json" },
      body: JSON.stringify(cfgBody(d)) });
    if ((plan.replays || 1) > 1) {
      box.replaceChildren(
        `${plan.calls_without_replays} → ${plan.estimated_calls} calls · est ${money(plan.est_cost_without_replays)} → ${money(plan.est_cost)}`,
        h("span", { class: "muted" }, ` (${plan.requests} request${plan.requests === 1 ? "" : "s"} × ${plan.per_candidate.length} candidate${plan.per_candidate.length === 1 ? "" : "s"} × ${plan.replays} replays)`));
    } else {
      box.textContent = `${money(plan.est_cost)} · ${plan.estimated_calls} calls`;
    }
    if (plan.est_seconds != null) {
      const n = plan.parallelism || 1;
      box.append(` · ~${fmtDur(plan.est_seconds)} at ${n === 1 ? "one call at a time" : `${n} parallel calls`} `,
        h("a", { href: "?view=settings", class: "muted", style: "font-size:11px",
          title: "Parallel calls are set in Settings → Run defaults",
          onclick: (e) => { e.preventDefault(); go("settings"); } }, "(change in Settings)"));
    }
    const warn = document.getElementById("replayWarn");
    if (warn) {
      warn.style.display = plan.over_ceiling ? "" : "none";
      warn.textContent = plan.over_ceiling
        ? `${plan.estimated_calls} calls is above the ${plan.call_ceiling}-call limit set in Settings. Consider fewer replays, or sample fewer requests (Limit, above).`
        : "";
    }
  } catch (e) { box.textContent = e.message; }
}

// Configuration step: replay each request N times to measure consistency (D7).
function replayField(d) {
  if (d.replaysOn === undefined) d.replaysOn = false;
  const def = (SETTINGS && SETTINGS.recommendation && SETTINGS.recommendation.default_replays) || 3;
  d.replayN = d.replayN || def;
  const nSeg = h("div", { class: "seg", role: "group", "aria-label": "Times to send each request" }, ...[2, 3, 4, 5].map(n =>
    h("button", { class: n === d.replayN ? "active" : "", "aria-pressed": n === d.replayN ? "true" : "false",
      onclick: () => {
        d.replayN = n;
        nSeg.querySelectorAll("button").forEach(b => { const on = Number(b.textContent) === n; b.classList.toggle("active", on); b.setAttribute("aria-pressed", on ? "true" : "false"); });
        updatePlan(d);
      } }, String(n))));
  const nWrap = h("div", { class: "row", style: `gap:8px;${d.replaysOn ? "" : "display:none"}` },
    h("span", { class: "muted", style: "font-size:12px" }, "Send each request"), nSeg, h("span", { class: "muted", style: "font-size:12px" }, "times"));
  const toggle = h("input", { type: "checkbox", id: "replayToggle", "aria-describedby": "replayHelp" });
  toggle.checked = !!d.replaysOn;
  toggle.addEventListener("change", () => { d.replaysOn = toggle.checked; nWrap.style.display = d.replaysOn ? "" : "none"; updatePlan(d); });
  return h("div", { class: "field", style: "margin-top:4px" },
    h("div", { class: "row", style: "gap:16px;flex-wrap:wrap;align-items:center" },
      h("label", { class: "row", style: "gap:8px;cursor:pointer", for: "replayToggle" }, toggle, h("b", { style: "font-size:13px" }, "Replay requests")),
      nWrap),
    h("div", { id: "replayHelp", class: "muted", style: "font-size:11px;margin-top:4px" },
      "Sends the same request several times to see whether the model answers consistently. Every answer is scored normally; "
      + "a separate D7 Consistency score compares the answers. Multiplies calls and cost."),
    h("div", { id: "replayWarn", class: "banner warn", style: "display:none;margin-top:6px;font-size:12px" }));
}

function cfgBody(d) {
  const callPath = d.endpoint === "litellm" ? "litellm" : "bedrock";
  const bedrockEndpoint = d.endpoint === "bedrock_mantle" ? "mantle" : "runtime";
  return { candidates: (d.picks || []).flatMap(p => p.efforts.map(e => ({ model: p.model, reasoning_effort: e }))),
           name: d.name || null,
           call_path: callPath, bedrock_endpoint: bedrockEndpoint,
           sampling_mode: d.samplingMode || "all",
           sampling_n: d.samplingN || 200,
           judge_steering: d.judgeSteering || null,
           filter_teams: (d.filterTeams && d.filterTeams.length) ? d.filterTeams : null,
           filter_users: (d.filterUsers && d.filterUsers.length) ? d.filterUsers : null,
           replays: d.replaysOn ? (d.replayN || 3) : 1 };
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

// ---- Run progress: stages, headline numbers, one row per candidate; a row expands to
// that candidate's call matrix (one square per request, click a square for its details).
const fmtDur = (sec) => {
  if (sec == null || !isFinite(sec)) return "—";
  sec = Math.max(0, Math.round(sec));
  if (sec < 60) return `${sec}s`;
  if (sec < 3600) return `${Math.floor(sec / 60)}m ${String(sec % 60).padStart(2, "0")}s`;
  return `${Math.floor(sec / 3600)}h ${String(Math.floor(sec % 3600 / 60)).padStart(2, "0")}m`;
};
const fmtLat = (ms) => (ms == null ? "—" : ms < 1000 ? `${Math.round(ms)} ms` : `${(ms / 1000).toFixed(1)} s`);
const fmtTok = (n) => (n == null ? "—" : n >= 1000 ? `${(n / 1000).toFixed(1)}k` : String(n));
const fmtUsd = (n) => (n == null ? "—" : `$${n.toFixed(n < 1 ? 4 : 2)}`);
const pctl = (arr, p) => {
  if (!arr.length) return null;
  const s = [...arr].sort((a, b) => a - b);
  return s[Math.min(s.length - 1, Math.max(0, Math.ceil(p * s.length) - 1))];
};
const mean = (arr) => (arr.length ? arr.reduce((a, b) => a + b, 0) / arr.length : null);
// How each call is drawn: [matrix square class, plain-words label].
const CELL = {
  ok: ["ok", "ok"], rt: ["rt", "ok after retry"], ca: ["ca", "from cache"],
  er: ["er", "failed"], rn: ["rn", "in flight"], bo: ["bo", "backing off"], pend: ["", "pending"],
};
const cellKind = (cell) => !cell ? "pend" : cell.status === "error" ? "er"
  : cell.from_cache ? "ca" : (cell.attempts || 1) > 1 ? "rt" : "ok";
const retryWhy = (r) => (r === "throttled" ? "a throttle" : r === "timeout" ? "a timeout" : r === "error" ? "an error" : r || "an error");

function renderProgress(el, runId) {
  clear(el);
  const S = {
    planTs: null, lastTs: null, lastLocal: 0, total: 0, done: 0, estCost: null, stage: "plan",
    cands: new Map(), retries: 0, errors: 0, cache: 0, spend: 0, spendUnknown: false,
    doneTs: [], open: new Set(), sel: null,
    // Parallel calls: every in-flight call, the limit, per-endpoint effective limits, throttles.
    inflight: new Map(), par: 1, eff: {}, waiting: false,
    throttles: 0, throttleRetries: 0, throttleFails: 0, thrByEp: {}, thrLast: {}, thrCands: {},
    series: [], thrTimes: [], changes: [],
  };
  const cand = (label) => {
    let c = S.cands.get(label);
    if (!c) {
      c = { label, calls: 0, est: null, priceKnown: true, cells: [], done: 0, ok: 0, failed: 0,
            cache: 0, retried: 0, lat: [], cost: 0, costUnknown: false, outTok: 0, throttles: 0 };
      S.cands.set(label, c);
    }
    return c;
  };
  // With replays, a candidate's calls are laid out request by request, N replays each:
  // position = request index × N + replay.
  const R = () => S.replays || 1;
  const posOf = (index, replay) => index * R() + (replay || 0);
  const fkey = (model, index, replay) => `${model}|${index}|${replay || 0}`;
  const flightAt = (c, i) => S.inflight.get(fkey(c.label, Math.floor(i / R()), i % R()));
  const inflightOf = (c) => [...S.inflight.values()].filter(f => f.model === c.label);
  const isNow = (c, i) => (i === undefined ? inflightOf(c).length > 0 : !!flightAt(c, i));
  const effMin = () => { const v = Object.values(S.eff); return v.length ? Math.min(S.par, ...v) : S.par; };
  const slowed = () => effMin() < S.par;
  const backingOff = (f) => !!f && f.backoffUntil != null && serverNow() != null && serverNow() < f.backoffUntil;
  const EP_TEXT = { "bedrock-runtime": "Bedrock runtime", "bedrock-mantle": "Bedrock Mantle",
                    "bedrock-converse": "Bedrock Converse", litellm: "LiteLLM proxy", default: "Model endpoint" };
  const epText = (ep) => EP_TEXT[ep] || ep;

  // --- static frame
  const log = h("div", { class: "logpane", id: "runLog" });
  const errChip = h("span", { class: "rp-st err", style: "display:none" });
  const retryChip = h("span", { class: "rp-st warn", style: "display:none" });
  const errOnly = h("input", { type: "checkbox" });
  errOnly.addEventListener("change", () => log.classList.toggle("errors-only", errOnly.checked));
  const logWrap = h("div", { style: "display:none;margin-top:14px" },
    h("div", { class: "row spread", style: "margin-bottom:4px" },
      h("span", { class: "muted", style: "font-size:11px" }, "Live log"),
      h("label", { class: "row", style: "gap:6px;font-size:12px;cursor:pointer" }, errOnly,
        h("span", { class: "muted" }, "errors only"))),
    log);
  const logBtn = h("button", { class: "btn ghost", "aria-expanded": "false",
    onclick: () => {
      const show = logWrap.style.display === "none";
      logWrap.style.display = show ? "" : "none";
      logBtn.textContent = show ? "Hide log" : "Show log";
      logBtn.setAttribute("aria-expanded", show ? "true" : "false");
      if (show) log.scrollTop = log.scrollHeight;
    } }, "Show log");
  const stepsEl = h("div", { class: "rp-steps" });
  const kpiEl = h("div", { class: "rp-kpis" });
  const tableEl = h("div", { style: "overflow-x:auto" });
  const nowEl = h("div", { class: "rp-now", "aria-live": "polite" }, "Connecting to run…");
  const thrChip = h("span", { class: "rp-st warn", style: "display:none" });
  const bannerEl = h("div", { class: "banner warn", role: "status", style: "display:none;margin:10px 0" });
  const timelineEl = h("div", { class: "rp-timeline" });
  const flightEl = h("div", {});
  const epEl = h("div", {});
  const sideEl = h("div", { class: "rp-side" }, flightEl, epEl);
  el.append(
    h("div", { class: "row spread" },
      h("div", { class: "row", style: "gap:10px;align-items:center" },
        h("h3", { style: "margin:0" }, "Running"), thrChip, errChip, retryChip),
      h("div", { class: "row", style: "gap:8px;align-items:center" }, logBtn,
        h("button", { class: "btn ghost", onclick: () => cancelRun(runId) }, "Cancel"))),
    bannerEl, stepsEl, timelineEl, kpiEl, tableEl, sideEl, nowEl,
    h("div", { class: "muted", style: "font-size:11px;margin-top:8px" },
      "Click a candidate to see each of its calls. Latency counts the successful attempt only; cached calls are excluded. Cost uses public list prices and assumes no prompt caching."),
    logWrap);

  // Server-clock "now": last event timestamp plus local time passed since it arrived.
  const serverNow = () => (S.lastTs == null ? null : S.lastTs + (Date.now() - S.lastLocal) / 1000);

  const renderSteps = () => {
    const pct = S.total ? Math.round(100 * S.done / S.total) : 0;
    const order = ["plan", "replay", "aggregate", "done"];
    const state = (key) => {
      const i = order.indexOf(key), cur = order.indexOf(S.stage);
      return i < cur ? "done" : i === cur ? "cur" : "pend";
    };
    const step = (key, title, sub) => {
      const s = h("div", { class: `rp-step ${state(key)}` }, h("b", {}, title), h("span", { class: "tnum" }, sub));
      if (key === "replay" && state(key) === "cur") s.style.setProperty("--pct", `${pct}%`);
      return s;
    };
    stepsEl.replaceChildren(
      step("plan", "Plan", S.planTs ? `${S.total} calls · est ${fmtUsd(S.estCost)}` : "preparing…"),
      step("replay", "Replay + score", `${S.done} / ${S.total || "?"} calls (${pct}%)`
        + (R() > 1 && S.stage === "replay" ? ` · pass ${(S.pass || 0) + 1} of ${R()}` : "")),
      step("aggregate", "Aggregate", "verdicts & recommendation"),
      step("done", "Done", "results open automatically"));
  };

  const pace = () => {  // calls per second over the last 20 completions
    const t = S.doneTs.slice(-20);
    if (t.length < 2 || t[t.length - 1] <= t[0]) return null;
    return (t.length - 1) / (t[t.length - 1] - t[0]);
  };

  const renderKpis = () => {
    const now = serverNow();
    const elapsed = S.planTs != null && now != null ? now - S.planTs : null;
    const rate = pace();
    const left = S.stage === "aggregate" ? "finishing…"
      : rate && S.total ? `~${fmtDur((S.total - S.done) / rate)}` : "—";
    const kpi = (l, v, s, amber) => h("div", { class: "rp-kpi" + (amber ? " amber" : "") }, h("div", { class: "l" }, l),
      h("div", { class: "v" }, ...v), h("div", { class: "s" }, s));
    const live = S.stage === "replay";
    const nIn = S.inflight.size, remaining = S.total ? S.total - S.done : null;
    const parSub = !live ? `limit ${S.par}` : S.waiting && !nIn ? "waiting for capacity"
      : slowed() ? "slowed by throttling" : S.par === 1 ? "one call at a time"
      : (remaining != null && nIn >= remaining) ? "last calls finishing"
      : nIn >= effMin() ? "at the configured limit" : "below the limit";
    kpiEl.replaceChildren(
      kpi("Calls", [String(S.done), h("small", {}, ` / ${S.total || "?"}`)], `${S.done - S.errors} ok · ${S.errors} failed`),
      kpi("Parallel calls", [String(live ? S.inflight.size : 0), h("small", {}, ` of ${live ? effMin() : S.par}`)],
        slowed() && live ? `${parSub} (set to ${S.par})` : parSub, live && (slowed() || S.waiting)),
      kpi("Throttles", [String(S.throttles)],
        S.throttles ? `${S.throttleRetries} retried · ${S.throttleFails} failed` : "none so far", S.throttles > 0),
      kpi("Elapsed", [fmtDur(elapsed)], S.planTs ? `started ${new Date(S.planTs * 1000).toLocaleTimeString()}` : ""),
      kpi("Time left", [left], !rate ? "needs a few calls first" : slowed() && live ? "(slowed by throttling)" : "at current pace",
        slowed() && live),
      kpi("Pace", [rate ? String(Math.round(rate * 60)) : "—", h("small", {}, " /min")], "last 20 calls"),
      kpi("Spend so far", [fmtUsd(S.spend)],
        (S.estCost != null ? `of ~${fmtUsd(S.estCost)} estimated` : "") + (S.spendUnknown ? " · some prices unknown" : "")),
      kpi("Cache hits", [String(S.cache)], "reused, not re-billed"),
      kpi("Retries", [String(S.retries)], "throttles and transient errors"));
  };

  // Concurrency timeline: calls in flight (area), current limit (step line), configured limit
  // (dashed), throttles (red ticks), slow-downs / recoveries (triangles).
  const renderTimeline = () => {
    const pts = S.series;
    if (!S.planTs || pts.length < 2) { timelineEl.replaceChildren(); return; }
    const W = 1000, H = 80, L = 34, Rr = W - 6, T = 12, B = H - 16;
    const tMax = Math.max(pts[pts.length - 1].t, 1), yMax = Math.max(S.par, 1);
    const X = (t) => L + (t / tMax) * (Rr - L), Y = (v) => B - (v / yMax) * (B - T);
    const s = sv("svg", { viewBox: `0 0 ${W} ${H}`, width: "100%", height: String(H), role: "img",
      "aria-label": `Parallel calls over time. Limit ${S.par}. ${S.throttles} throttles.`
        + (S.changes.length ? ` ${S.changes.filter(c => c.reason === "throttle").length} slow-downs.` : "") });
    const kids = [sv("text", { x: 0, y: 10, fill: "var(--muted)", "font-size": 10 }, "Parallel calls")];
    for (let v = 1; v <= yMax; v++) {
      if (yMax > 6 && v % Math.ceil(yMax / 4)) continue;
      kids.push(sv("line", { x1: L, x2: Rr, y1: Y(v), y2: Y(v), stroke: "var(--border)", "stroke-width": 0.5 }),
        sv("text", { x: L - 6, y: Y(v) + 3, "text-anchor": "end", fill: "var(--muted)", "font-size": 9 }, String(v)));
    }
    kids.push(sv("path", { d: `M${X(pts[0].t)},${B} ` + pts.map(p => `L${X(p.t)},${Y(p.inflight)}`).join(" ") + ` L${X(pts[pts.length - 1].t)},${B} Z`,
      fill: "rgba(79,124,255,.25)", stroke: "var(--accent)", "stroke-width": 1 }));
    kids.push(sv("path", { d: pts.map((p, i) => `${i ? "L" : "M"}${X(p.t)},${Y(p.eff)}${i < pts.length - 1 ? ` L${X(pts[i + 1].t)},${Y(p.eff)}` : ""}`).join(" "),
      fill: "none", stroke: "var(--partial)", "stroke-width": 2 }));
    kids.push(sv("line", { x1: L, x2: Rr, y1: Y(S.par), y2: Y(S.par), stroke: "var(--muted)", "stroke-dasharray": "4 4" }));
    S.thrTimes.forEach(t => kids.push(sv("line", { x1: X(t), x2: X(t), y1: B, y2: B - 8, stroke: "var(--bad)", "stroke-width": 2 })));
    S.changes.forEach(c => {
      const col = c.reason === "throttle" ? "var(--partial)" : "var(--ok)";
      const tri = sv("path", { d: `M${X(c.t) - 4},${T - 4} L${X(c.t) + 4},${T - 4} L${X(c.t)},${T + 2} Z`, fill: col });
      tri.append(sv("title", {}, `${fmtDur(c.t)}: ${c.reason === "throttle" ? "slowed" : "recovered"} to ${c.eff}`));
      kids.push(tri);
    });
    kids.push(sv("text", { x: L, y: H - 2, fill: "var(--muted)", "font-size": 9 }, "0:00"),
      sv("text", { x: Rr, y: H - 2, "text-anchor": "end", fill: "var(--muted)", "font-size": 9 },
        `${fmtDur(tMax)} · area = in flight · amber = current limit · dashed = your setting (${S.par}) · red = throttles`));
    s.append(...kids.filter(Boolean));
    timelineEl.replaceChildren(s);
  };

  const renderFlight = () => {
    const live = S.stage === "replay";
    const rows = [...S.inflight.values()].sort((a, b) => a.ts - b.ts);
    const now = serverNow();
    const lim = effMin();
    const head = h("h4", { class: "rp-h4" }, live ? `In flight (${rows.length} of ${lim})` : "In flight");
    if (!rows.length) {
      flightEl.replaceChildren(head, h("div", { class: "muted", style: "font-size:12px" },
        !live ? "—" : S.waiting ? "Waiting for capacity: other runs are using the server-wide limit." : "No call in flight right now."));
    } else {
      const tr = rows.slice(0, 8).map(f => {
        const bo = backingOff(f);
        const what = bo ? `retrying after ${retryWhy(f.reason)} · attempt ${f.attempt} of ${f.max} · next try in ${Math.max(0, Math.ceil(f.backoffUntil - now))} s`
          : f.attempt > 1 ? `attempt ${f.attempt}${f.max ? ` of ${f.max}` : ""}` : "attempt 1";
        return h("tr", { class: bo ? "boff" : "" },
          h("td", { class: "mono" }, f.model),
          h("td", { class: "mono" }, f.sample_id + (R() > 1 ? ` · replay ${(f.replay || 0) + 1}` : "")),
          h("td", {}, what),
          h("td", { class: "num tnum" }, fmtDur(now != null ? now - f.ts : 0)));
      });
      // Native replaceChildren prints null as the text "null", so drop the optional child first.
      flightEl.replaceChildren(...[head, h("table", { class: "rp-flight" }, h("tbody", {}, ...tr)),
        rows.length > 8 ? h("div", { class: "muted", style: "font-size:11px" }, `+${rows.length - 8} more`) : null]
        .filter(Boolean));
    }
    // Throttling by endpoint: only once Bedrock (or the proxy) has pushed back.
    const eps = [...new Set([...Object.keys(S.thrByEp), ...Object.keys(S.eff)])];
    if (!S.throttles || !eps.length) { epEl.replaceChildren(); return; }
    const ago = (t) => (t == null || now == null ? "—" : `${fmtDur(now - t)} ago`);
    epEl.replaceChildren(h("h4", { class: "rp-h4" }, "Throttling by endpoint"),
      h("table", { class: "rp-flight" },
        h("thead", {}, h("tr", {}, h("th", {}, "Endpoint"), h("th", { class: "num" }, "Limit"),
          h("th", { class: "num" }, "Throttles"), h("th", {}, "Last"))),
        h("tbody", {}, ...eps.map(ep => {
          const e = S.eff[ep] ?? S.par;
          return h("tr", {},
            h("td", {}, epText(ep), " ", h("span", { class: "muted" }, `(${[...(S.thrCands[ep] || [])].join(", ") || "—"})`)),
            h("td", { class: "num tnum", style: e < S.par ? "color:var(--partial)" : "" }, `${e} / ${S.par}`),
            h("td", { class: "num tnum" }, String(S.thrByEp[ep] || 0)),
            h("td", {}, ago(S.thrLast[ep])));
        }))));
  };

  const renderBanner = () => {
    const live = S.stage === "replay";
    if (live && slowed()) {
      const eps = Object.entries(S.eff).filter(([, v]) => v < S.par).map(([k]) => epText(k));
      bannerEl.replaceChildren("⚠ Running slower than configured: ", h("b", {}, `${eps.join(", ") || "Bedrock"} is throttling`),
        `, so ModelShift is sending ${effMin()} call${effMin() === 1 ? "" : "s"} at a time instead of ${S.par}. Calls are retried automatically; no answers are lost.`);
      bannerEl.style.display = "";
    } else bannerEl.style.display = "none";
  };

  const sample = () => {   // one timeline point per event (deduped), ≤ 600 points
    if (!S.planTs || S.lastTs == null) return;
    const p = { t: Math.max(0, S.lastTs - S.planTs), inflight: S.inflight.size, eff: effMin() };
    const last = S.series[S.series.length - 1];
    if (last && last.inflight === p.inflight && last.eff === p.eff) return;
    S.series.push(p);
    if (S.series.length > 600) S.series = S.series.filter((_, i) => i % 2 === 0 || i === S.series.length - 1);
  };

  const candStatus = (c) => {
    const k = inflightOf(c).length;
    if (k) return h("span", { class: "rp-st run" }, h("span", { class: "rp-dot rp-pulse" }),
      S.par > 1 ? `Running · ${k} in flight` : "Running");
    if (c.calls && c.done >= c.calls) return c.failed
      ? h("span", { class: "rp-st warn" }, `✓ Complete · ${c.failed} failed`)
      : h("span", { class: "rp-st done" }, "✓ Complete");
    return h("span", { class: "rp-st pend" }, c.done ? (S.stage === "replay" ? "Queued" : "Paused") : "Pending");
  };

  const laneBar = (c) => {
    const n = Math.max(c.calls, c.done, 1);
    const fl = inflightOf(c), boN = fl.filter(backingOff).length;
    const seg = (cls, k, label) => (k ? h("span", { class: cls, style: `width:${100 * k / n}%`, title: `${k} ${label}` }) : null);
    return h("div", { class: "rp-bar", role: "img",
      "aria-label": `${c.done} of ${c.calls} calls done: ${c.ok} ok, ${c.cache} from cache, ${c.failed} failed, ${fl.length} in flight` },
      seg("rp-ok", c.ok - c.cache - c.retried, "ok"), seg("rp-rt", c.retried, "ok after retry"),
      seg("rp-ca", c.cache, "from cache"), seg("rp-er", c.failed, "failed"),
      seg("rp-rn rp-pulse", fl.length - boN, "in flight"), seg("rp-bo rp-pulse", boN, "backing off"));
  };

  const cellDetail = (c, i) => {
    const cell = c.cells[i];
    const item = (l, v) => h("div", {}, h("span", { class: "l" }, l), v);
    const box = h("div", { class: "rp-detail" });
    const repItem = R() > 1 ? item("Replay", `${(i % R()) + 1} of ${R()}`) : null;
    const fl = flightAt(c, i);
    if (fl) {
      const bo = backingOff(fl);
      box.append(item("Call", h("span", { class: "mono" }, `${c.label} · ${fl.sample_id}`)),
        ...(repItem ? [repItem] : []),
        item("Status", h("span", { style: `color:var(--${bo ? "partial" : "accent"})` }, bo ? "backing off" : "in flight")),
        item("Attempt", fl.max ? `${fl.attempt} of ${fl.max}` : String(fl.attempt)),
        item("So far", fmtDur(serverNow() - fl.ts)));
      if (fl.reason) box.append(h("div", { class: "err" }, `Retrying after ${retryWhy(fl.reason)}: ${fl.error || ""}`));
      return box;
    }
    if (!cell) {
      box.append(item("Call", h("span", { class: "mono" }, `${c.label} · request ${Math.floor(i / R()) + 1}`)),
        ...(repItem ? [repItem] : []), item("Status", "not started yet"));
      return box;
    }
    const k = cellKind(cell);
    const colour = { er: "bad", ca: "drift", rt: "partial", ok: "ok" }[k];
    box.append(
      item("Call", h("span", { class: "mono" }, `${c.label} · ${cell.sample_id}`)),
      ...(repItem ? [repItem] : []),
      item("Status", h("span", { style: `color:var(--${colour})` }, CELL[k][1])),
      item("Attempts", String(cell.attempts || 1)),
      item(cell.status === "error" ? "Failed after" : "Latency", cell.from_cache ? "cached" : fmtLat(cell.latency_ms)),
      item("Tokens in / out", `${fmtTok(cell.input_tokens)} / ${fmtTok(cell.output_tokens)}`),
      item("Reasoning tokens", fmtTok(cell.reasoning_tokens)),
      item("Cost", cell.cost != null ? fmtUsd(cell.cost) : cell.status === "error" ? "$0" : "unknown"));
    if (cell.error) box.append(h("div", { class: "err" }, cell.error));
    return box;
  };

  const matrix = (c) => {
    const n = Math.max(c.calls, c.done);
    const cells = h("div", { class: "rp-cells" + (R() > 1 ? " grouped" : ""), role: "group", "aria-label": `Calls for ${c.label}` });
    let grp = null;
    for (let i = 0; i < n; i++) {
      const cell = c.cells[i];
      const fl = flightAt(c, i);
      const k = fl ? (backingOff(fl) ? "bo" : "rn") : cellKind(cell);
      const sel = !!S.sel && S.sel.label === c.label && S.sel.index === i;
      const rep = R() > 1 ? ` (replay ${(i % R()) + 1} of ${R()})` : "";
      const label = `${cell ? cell.sample_id : fl ? fl.sample_id : `request ${Math.floor(i / R()) + 1}`}${rep}: ${CELL[k][1]}` +
        (cell && cell.latency_ms != null && !cell.from_cache ? `, ${fmtLat(cell.latency_ms)}` : "");
      const btn = h("button", { class: `rp-c ${CELL[k][0]}${sel ? " sel" : ""}`, title: label, "aria-label": label,
        "aria-pressed": sel ? "true" : "false", "data-cell": `${c.label}|${i}`,
        onclick: (e) => { e.stopPropagation(); S.sel = sel ? null : { label: c.label, index: i }; render(true); } });
      if (R() > 1) {   // one column per request, its replays stacked
        if (i % R() === 0) { grp = h("span", { class: "rp-grp" }); cells.append(grp); }
        grp.append(btn);
      } else cells.append(btn);
    }
    const avg = mean(c.lat);
    const lat = avg != null ? `avg ${fmtLat(avg)} · p95 ${fmtLat(pctl(c.lat, 0.95))}` : "no timed calls yet";
    const legend = h("div", { class: "rp-legend" },
      ...["ok", "rt", "ca", "er", "rn", "bo", "pend"].map(k => h("span", {}, h("i", { class: k === "pend" ? "" : `rp-${k}` }), CELL[k][1])));
    return h("div", {},
      h("div", { style: "margin-bottom:6px;font-size:12px" }, h("b", {}, c.label), h("span", { class: "muted" },
        `  ·  ${c.done}/${c.calls || c.done} calls · ${c.ok} ok · ${c.cache} from cache · ${c.retried} retried · ${c.failed} failed · ${lat}`)),
      legend, cells,
      S.sel && S.sel.label === c.label ? cellDetail(c, S.sel.index)
        : h("div", { class: "muted", style: "font-size:11px;margin-top:8px" }, "Click a square for that call's details."));
  };

  const renderTable = () => {
    // Rebuilt on every update: remember which row/square had keyboard focus and restore it.
    const f = document.activeElement;
    const focusSel = f && tableEl.contains(f)
      ? (f.dataset.lane ? `[data-lane="${CSS.escape(f.dataset.lane)}"]`
        : f.dataset.cell ? `[data-cell="${CSS.escape(f.dataset.cell)}"]` : null) : null;
    const cols = [["Candidate"], ["Status"], ["Progress"], ["Calls", 1], ["Failed", 1], ["Throttles", 1],
      ["Avg latency", 1], ["p95", 1], ["Out tokens", 1], ["Cost", 1]];
    const body = h("tbody");
    for (const c of S.cands.values()) {
      const open = S.open.has(c.label);
      const toggle = () => { if (open) S.open.delete(c.label); else S.open.add(c.label); render(true); };
      const pending = !c.done;
      body.append(h("tr", {
        class: "rp-row" + (isNow(c) ? " cur" : ""), tabindex: "0", role: "button",
        "aria-expanded": open ? "true" : "false", "data-lane": c.label,
        "aria-label": `${c.label}: ${open ? "hide" : "show"} each call`,
        onclick: toggle, onkeydown: (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); toggle(); } },
      },
        h("td", { class: "mono", style: "white-space:nowrap" }, h("span", { class: "chev", "aria-hidden": "true" }, "▸"), " ", c.label),
        h("td", {}, candStatus(c)),
        h("td", { style: "width:24%" }, laneBar(c)),
        h("td", { class: "num" }, `${c.done}/${c.calls || "?"}`),
        h("td", { class: "num", style: c.failed ? "color:var(--bad)" : "" }, pending ? "—" : String(c.failed)),
        h("td", { class: "num", style: c.throttles ? "color:var(--partial)" : "" }, String(c.throttles)),
        h("td", { class: "num" }, fmtLat(mean(c.lat))),
        h("td", { class: "num" }, fmtLat(pctl(c.lat, 0.95))),
        h("td", { class: "num" }, pending ? "—" : fmtTok(c.outTok)),
        h("td", { class: "num" }, pending ? (c.est != null && c.priceKnown ? `~${fmtUsd(c.est)} est` : "—")
          : fmtUsd(c.cost) + (c.costUnknown ? " +?" : ""))));
      if (open) body.append(h("tr", { class: "rp-mx" }, h("td", { colspan: String(cols.length) }, matrix(c))));
    }
    tableEl.replaceChildren(h("table", { class: "rp-table" },
      h("thead", {}, h("tr", {}, ...cols.map(([t, num]) => h("th", { class: num ? "num" : "" }, t)))), body));
    if (focusSel) { const t = tableEl.querySelector(focusSel); if (t) t.focus(); }
  };

  const renderNow = () => {
    nowEl.style.display = "";
    if (S.stage === "aggregate") nowEl.textContent = "All calls finished. Computing verdicts and the recommendation…";
    else if (S.stage === "done") nowEl.textContent = "Done. Opening results…";
    else if (S.inflight.size) nowEl.style.display = "none";
    else if (S.planTs) nowEl.textContent = S.done ? "Between calls…" : "Waiting for the first call…";
  };

  let timer = null, lastRender = 0;
  const render = (force) => {
    const t = Date.now();
    if (!force && t - lastRender < 250) {  // coalesce bursts (e.g. replayed history on open)
      if (!timer) timer = setTimeout(() => { timer = null; render(true); }, 250);
      return;
    }
    lastRender = t;
    renderSteps(); renderBanner(); renderTimeline(); renderKpis(); renderTable(); renderFlight(); renderNow();
    thrChip.style.display = S.throttles ? "" : "none";
    thrChip.textContent = `${S.throttles} throttle${S.throttles === 1 ? "" : "s"}` + (slowed() && S.stage === "replay" ? ` · slowed to ${effMin()} of ${S.par}` : "");
    errChip.style.display = S.errors ? "" : "none";
    errChip.textContent = `${S.errors} error${S.errors === 1 ? "" : "s"}`;
    retryChip.style.display = S.retries ? "" : "none";
    retryChip.textContent = `${S.retries} retr${S.retries === 1 ? "y" : "ies"}`;
  };
  const tick = setInterval(() => render(true), 1000);  // keeps "Elapsed" and "so far" ticking

  const seen = new Set();
  const addLog = (line, cls) => {
    const row = h("div", { class: "logrow" + (cls ? " " + cls : "") }, line);
    log.append(row); log.scrollTop = log.scrollHeight;
  };
  const onEvent = (type, d) => {
    const key = type + JSON.stringify(d);
    if (seen.has(key)) return; seen.add(key);
    if (d.ts != null) { S.lastTs = d.ts; S.lastLocal = Date.now(); }
    if (type === "plan") {
      S.planTs = d.ts ?? Date.now() / 1000;
      S.total = d.estimated_calls || 0; S.estCost = d.est_cost ?? null; S.stage = "replay";
      S.replays = d.replays || 1;
      S.par = d.parallelism || S.par;
      (d.per_candidate || []).forEach(p => {
        const c = cand(p.model); c.calls = p.calls; c.est = p.est_cost; c.priceKnown = p.price_known !== false;
      });
      addLog(`▶ plan: ${d.estimated_calls} calls · est $${(d.est_cost || 0).toFixed(4)}`
        + (S.replays > 1 ? ` · each request sent ${S.replays} times` : ""));
    } else if (type === "pass") {
      S.pass = d.replay || 0; S.replays = d.replays || S.replays || 1;
      addLog(`▶ pass ${S.pass + 1} of ${S.replays}`);
    } else if (type === "stage") {
      S.stage = d.stage === "aggregate" ? "aggregate" : "replay";
      if (d.replays) S.replays = d.replays;
      if (d.parallelism) S.par = d.parallelism;
      if (d.stage === "aggregate") S.inflight.clear();
    } else if (type === "call_start") {
      cand(d.model);
      if (d.replays) S.replays = d.replays;
      S.inflight.set(fkey(d.model, d.index, d.replay), { model: d.model, sample_id: d.sample_id, index: d.index,
        replay: d.replay || 0, ts: d.ts ?? serverNow(), attempt: 1, max: null, reason: null, error: null,
        backoffUntil: null, endpoint: d.endpoint });
    } else if (type === "retry") {
      S.retries++;
      const f = S.inflight.get(fkey(d.model, d.index, d.replay));
      if (f) Object.assign(f, { attempt: d.attempt, max: d.max_attempts, reason: d.reason, error: d.error,
        backoffUntil: d.ts != null && d.wait_s ? d.ts + d.wait_s : null });
      if (d.reason === "throttled") {
        const ep = d.endpoint || (f && f.endpoint) || "default";
        S.throttles++; S.throttleRetries++;
        S.thrByEp[ep] = (S.thrByEp[ep] || 0) + 1;
        if (d.ts != null) S.thrLast[ep] = d.ts;
        (S.thrCands[ep] = S.thrCands[ep] || new Set()).add(d.model);
        cand(d.model).throttles++;
        if (S.planTs && d.ts != null) S.thrTimes.push(d.ts - S.planTs);
      }
      addLog(`↻ ${d.model} · ${d.sample_id} → ${retryWhy(d.reason)}, retrying (attempt ${d.attempt}/${d.max_attempts}`
        + (d.wait_s ? `, waiting ${d.wait_s} s` : "") + `)  —  ${d.error || ""}`, "err");
    } else if (type === "concurrency") {
      S.eff[d.endpoint] = d.effective;
      if (d.limit) S.par = d.limit;
      const at = d.ts != null ? new Date(d.ts * 1000).toLocaleTimeString() : "";
      if (S.planTs && d.ts != null) S.changes.push({ t: d.ts - S.planTs, reason: d.reason, eff: d.effective });
      addLog(d.reason === "throttle"
        ? `⚠ ${at} ${epText(d.endpoint)} throttled · slowed to ${d.effective} parallel call${d.effective === 1 ? "" : "s"}`
        : `✓ ${at} ${epText(d.endpoint)} recovered · back to ${d.effective} parallel call${d.effective === 1 ? "" : "s"}`,
        d.reason === "throttle" ? "err" : "ok");
    } else if (type === "waiting_capacity") {
      S.waiting = !!d.run_waiting;
      if (S.waiting) addLog(`… waiting for capacity: ${d.server_inflight} of ${d.server_cap} calls in flight across all runs`);
    } else if (type === "progress") {
      const c = cand(d.model);
      const i = d.index != null ? posOf(d.index, d.replay) : c.done;  // older servers send no index
      if (typeof d.throttle_failures === "number" && d.throttle_failures > S.throttleFails) {
        const ep = d.endpoint || "default";   // final attempt was throttled too: count it
        S.throttleFails = d.throttle_failures;
        S.thrByEp[ep] = (S.thrByEp[ep] || 0) + 1;
        (S.thrCands[ep] = S.thrCands[ep] || new Set()).add(d.model);
        c.throttles++;
      }
      if (typeof d.throttles === "number") S.throttles = Math.max(S.throttles, d.throttles);
      c.cells[i] = d; c.done++;
      if (c.done > c.calls) c.calls = c.done;
      if (d.status === "error") { c.failed++; S.errors++; }
      else {
        c.ok++;
        if (d.from_cache) { c.cache++; S.cache++; }
        else {
          if ((d.attempts || 1) > 1) c.retried++;
          if (d.latency_ms != null) c.lat.push(d.latency_ms);
        }
        c.outTok += d.output_tokens || 0;
      }
      if (d.cost != null) { c.cost += d.cost; S.spend += d.cost; }
      else if (d.status !== "error" && "cost" in d) { c.costUnknown = true; S.spendUnknown = true; }
      S.done = typeof d.done === "number" ? d.done : S.done + 1;
      S.total = d.total || S.total;
      if (d.ts != null) S.doneTs.push(d.ts);
      if (d.index != null) S.inflight.delete(fkey(d.model, d.index, d.replay));
      else for (const [k, f] of S.inflight) if (f.model === d.model) { S.inflight.delete(k); break; }
      const isErr = d.status === "error";
      let line = `[${d.done}/${d.total}] ${d.model} · ${d.sample_id} → ${d.status}`;
      if (!isErr && d.latency_ms != null) line += d.from_cache ? " (cache)" : ` · ${fmtLat(d.latency_ms)}`;
      if (isErr && d.error) line += `  —  ${d.error}`;
      addLog(line, isErr ? "err" : "");
    } else if (type === "done") { S.stage = "done"; S.inflight.clear(); addLog("✓ done", "ok"); finish(); }
    else if (type === "cancelled") { S.inflight.clear(); addLog("✗ cancelled", "err"); finish(); }
    else if (type === "error") { S.inflight.clear(); addLog("✗ error: " + (d.error || "unknown"), "err"); finish(); }
    sample();
    render();
  };
  render(true);

  let finished = false;
  const finish = () => {
    if (finished) return; finished = true;
    clearInterval(tick); if (timer) clearTimeout(timer);
    render(true);
    if (es) es.close(); if (poll) clearInterval(poll);
    setTimeout(() => renderResults(el, runId), 400);
  };

  // Primary: SSE stream
  let es = null;
  try {
    es = new EventSource(`${API}/runs/${runId}/stream`);
    ["plan", "pass", "stage", "call_start", "retry", "concurrency", "waiting_capacity", "progress", "done", "cancelled", "error"].forEach(t =>
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

// ---- Multi-objective recommendation (mockup Option A) ---------------------------------- //
const PRESETS = [["balanced", "Balanced"], ["quality_first", "Quality"], ["cost_first", "Cost"],
                 ["latency_first", "Latency"], ["custom", "Custom"]];
const PROFILE_TEXT = { best_overall: "Best overall", best_quality: "Best quality",
                       most_economical: "Most economical", lowest_latency: "Lowest latency†" };
const PROFILE_CHIP = { best_overall: "Best overall", best_quality: "Best quality",
                       most_economical: "Most economical", lowest_latency: "Lowest latency" };
// Cost: unknown price is never shown as $0.
const costPer1k = (v) => v.cost_known === false ? "Price not configured"
  : (v.cost_per_1k_requests == null ? "—" : `$${v.cost_per_1k_requests.toFixed(2)}`);
const costTotal = (v) => v.cost_known === false ? "price not configured" : money(v.cost_total);
const msTok = (v) => v.ms_per_output_token_p50 == null ? "—"
  : `${v.ms_per_output_token_p50.toFixed(1)}${v.latency_low_confidence ? " (few calls)" : ""}`;
const pctSigned = (x) => x == null ? "—" : `${x > 0 ? "+" : ""}${Math.round(x)}%`;
const picksFor = (rec, model) => (rec && !rec.none_eligible ? rec.profiles : [])
  .filter(p => p.model === model).map(p => p.kind);

function disclaimerNote(text) {
  // One line by default; "Why?" expands the full disclaimer.
  return h("details", { class: "info", style: "margin-top:14px" },
    h("summary", {}, "† Latency numbers are indicative only — measured from this evaluation's machine and network, not production. ",
      h("u", {}, "Why?")),
    h("div", { style: "margin-top:6px" }, text || ""));
}

async function setWeights(runId, body) {
  return api(`/runs/${runId}/weights`, { method: "PUT", headers: { "content-type": "application/json" },
    body: JSON.stringify(body) });
}

function weightsControl(res, runId, onChanged) {
  const cur = (res.weights && res.weights.preset) || "balanced";
  const box = h("div", {});
  const seg = h("div", { class: "seg", role: "group", "aria-label": "Optimise for" }, ...PRESETS.map(([k, t]) =>
    h("button", { class: k === cur ? "active" : "", "aria-pressed": k === cur ? "true" : "false",
      onclick: async () => {
        if (k === "custom") { custom.style.display = custom.style.display === "none" ? "" : "none"; return; }
        try { await setWeights(runId, { preset: k }); onChanged(); }
        catch (e) { err.textContent = "Could not change weights: " + (e && e.message ? e.message : "error"); }
      } }, t)));
  const w = res.weights || { quality: 0.5, cost: 0.3, latency: 0.2 };
  const vals = { quality: Math.round(w.quality * 100), cost: Math.round(w.cost * 100), latency: Math.round(w.latency * 100) };
  const outs = {};
  const slider = (axis, label, min) => {
    outs[axis] = h("span", { class: "tnum" }, `${vals[axis]}`);
    return [h("span", { class: "muted" }, label),
      h("input", { type: "range", min: String(min), max: "100", value: String(vals[axis]), "aria-label": `${label} weight`,
        oninput: e => { vals[axis] = parseInt(e.target.value, 10); outs[axis].textContent = String(vals[axis]); } }),
      outs[axis]];
  };
  const custom = h("div", { style: cur === "custom" ? "" : "display:none" },
    h("div", { class: "custom-weights" }, ...slider("quality", "Quality", 20), ...slider("cost", "Cost", 0),
      ...slider("latency", "Latency", 0)),
    h("div", { class: "row", style: "gap:8px;margin-top:8px" },
      h("button", { class: "btn secondary", onclick: async () => {
        try { await setWeights(runId, { preset: "custom", ...vals }); onChanged(); }
        catch (e) { err.textContent = "Could not change weights: " + (e && e.message ? e.message : "error"); }
      } }, "Apply custom weights"),
      h("span", { class: "muted", style: "font-size:11px" }, "Relative weights; normalised to 100%. Quality never drops below 20%.")));
  const err = h("div", { class: "muted", style: "font-size:12px;color:var(--bad)" });
  box.append(h("div", { class: "row", style: "gap:8px" }, h("span", { class: "muted", style: "font-size:12px",
    title: "Changes how much quality, cost and latency count toward Best overall. Re-ranks instantly; no models are re-run." }, "Optimise for"), seg), custom, err);
  return box;
}

// "Why this pick": short data-driven lines from the recommendation (falls back to the old
// one-line reason for runs whose server didn't send explanations). `extra` = [[label, lines]]
// for the other picks the hero also holds, shown in an expander.
function whyBox(lines, fallback, extra) {
  if (!lines || !lines.length) return h("div", { class: "muted", style: "font-size:11px;margin-top:4px" }, fallback || "");
  return h("div", { class: "why-box" },
    h("div", { class: "lbl" }, "Why this pick"),
    h("div", { class: "why-lines" }, ...lines.map(l => h("div", {}, l))),
    (extra || []).length ? h("details", { class: "why-more" },
      h("summary", {}, "Why it's also " + extra.map(([lab]) => lab.toLowerCase()).join(" and ")),
      ...extra.map(([lab, ls]) => h("div", { class: "why-lines" }, h("b", {}, lab + ": "), ls.join(" ")))) : null);
}

function recommendationPanel(res, runId, onChanged) {
  const rec = res.recommendation;
  if (!rec) return null;
  const byModel = Object.fromEntries((res.candidates || []).map(v => [v.model, v]));
  const card = h("div", { class: "card", style: "margin-bottom:16px" },
    h("div", { class: "rec-head" }, h("h3", {}, "Recommendation"), weightsControl(res, runId, onChanged)));

  const ineligibleLine = () => {
    const items = Object.entries(rec.ineligible || {});
    if (!items.length) return null;
    return h("div", { class: "muted", style: "font-size:12px;margin-top:10px" },
      h("span", { title: "Only models that pass the quality gate (enough confidence, few hard breaks, few errors) can be recommended. Adjust the gate in Settings → Recommendation." },
        "Not recommended (failed the quality gate): "),
      ...items.flatMap(([m, rs], i) => [i ? " · " : "", h("span", { class: "mono" }, m), ` (${rs.join(", ") || "not eligible"})`]));
  };

  const EX = rec.explanations || {};
  if (rec.none_eligible) {
    card.append(...[h("div", { class: "banner warn" },
      h("b", {}, "No candidate is ready to recommend. "),
      (EX.none || []).length
        ? [h("div", { class: "why-lines", style: "margin-top:6px" }, ...EX.none.map(l => h("div", {}, l))),
           h("div", { style: "margin-top:4px" }, "See its remediations below, or adjust the quality gate in Settings.")]
        : rec.closest_by_fit ? ["Closest by fit: ", h("span", { class: "mono" }, rec.closest_by_fit),
          ". See its remediations below, or adjust the quality gate in Settings."] : ""),
      ineligibleLine(), disclaimerNote(rec.latency_disclaimer)].filter(Boolean));
    return card;
  }

  const profile = (k) => (rec.profiles || []).find(p => p.kind === k) || {};
  const axes = Object.fromEntries((rec.axes || []).map(a => [a.model, a]));
  const bo = profile("best_overall");
  const v = byModel[bo.model] || {};
  const ax = axes[bo.model] || {};
  // Option A: hero with confidence ring + score composition; picks the hero also holds become
  // chips; only genuinely different alternatives get their own card.
  const w = rec.weights || res.weights || { quality: 0.5, cost: 0.3, latency: 0.2 };
  const others = (rec.profiles || []).filter(p => p.kind !== "best_overall");
  const also = others.filter(p => p.model && p.model === bo.model);
  const diff = others.filter(p => !p.model || p.model !== bo.model);
  const scored = v.scored || 0;
  const hardN = Math.round((v.hard_break_rate || 0) * scored);
  const segs = [["Quality", ax.quality_score, w.quality, "var(--ok)"], ["Cost", ax.cost, w.cost, "var(--accent)"],
                ["Latency", ax.latency, w.latency, "var(--drift)"]]
    .map(([n, val, wt, col]) => ({ n, col, max: 100 * (wt || 0), pts: val == null ? null : val * (wt || 0) }));
  const comp = h("div", { class: "score-comp", role: "img",
    "aria-label": "Score made of " + segs.map(s => `${s.n.toLowerCase()} ${s.pts == null ? "not included" : s.pts.toFixed(1)} of ${s.max.toFixed(0)}`).join(", ") },
    ...segs.filter(s => s.pts).map(s => h("span", { style: `width:${s.pts}%;background:${s.col}`, title: `${s.n} ${s.pts.toFixed(1)} of ${s.max.toFixed(0)}` })));
  const stat = (k, val, unit, sub) => h("div", { class: "hstat" }, h("div", { class: "lbl" }, k),
    h("div", { class: "v tnum" }, val, unit ? h("small", {}, " " + unit) : null), h("div", { class: "sub" }, sub));
  const hero = h("div", { class: "pcard hero" },
    h("div", { class: "lbl" }, "Best overall"),
    h("div", { class: "hero-top" },
      h("div", { class: "ringwrap" }, confRing(v.migration_confidence, BAND_COLOR[v.verdict_band] || "var(--ok)", 112),
        ciText(v) ? h("div", { class: "ci-line", title: CI_TITLE }, ciText(v)) : null),
      h("div", { style: "flex:1;min-width:0" },
        h("div", { class: "pick" }, bo.model || "—"),
        h("div", { class: "row", style: "gap:6px;flex-wrap:wrap;margin:6px 0 12px" },
          v.verdict_band ? h("span", { class: "band " + v.verdict_band }, BAND_TEXT[v.verdict_band]) : null,
          ...also.map(p => h("span", { class: "chip" }, "also " + PROFILE_CHIP[p.kind]))),
        ax.composite != null ? [
          h("div", { class: "row spread", style: "font-size:12px" },
            h("span", { class: "muted", title: "Weighted score out of 100 combining quality, cost and latency with the weights chosen above. Only comparable within this run." },
              "Overall score ", h("b", { style: "color:var(--text)" }, ax.composite.toFixed(1)), " / 100"),
            h("span", { class: "muted" }, "what it's made of")),
          comp,
          h("div", { class: "legend" }, ...segs.map(s => h("span", {}, h("i", { style: `background:${s.col}` }),
            `${s.n} ${s.pts == null ? "not included" : s.pts.toFixed(1) + " / " + s.max.toFixed(0)}${s.n === "Latency" ? "†" : ""}`)))] : null)),
    h("div", { class: "hstats" },
      stat("Cost / 1K requests", costPer1k(v), null,
        v.cost_vs_current_pct == null ? "list price, no caching" : `${pctSigned(v.cost_vs_current_pct)} vs current`),
      stat("Speed†", v.ms_per_output_token_p50 == null ? "—" : v.ms_per_output_token_p50.toFixed(1), "ms/token",
        v.latency_p95_ms == null ? "" : `p95 ${fmtLat(v.latency_p95_ms)} per request`),
      stat("Hard breaks", String(hardN), `of ${scored}`, `${fmtPct((v.hard_break_rate || 0) * 100)} of answers`),
      v.consistency != null
        ? stat("Consistency (D7)", v.consistency.toFixed(2), null,
            `${v.unstable_requests || 0} unstable of ${v.consistency_measured || 0} requests × ${v.replays}`
            + (v.consistency_low_confidence ? " · few requests" : ""))
        : null),
    h("div", { class: "muted", style: "font-size:11px;margin-top:12px" },
      "Quality = how closely answers match your current model (Migration Confidence) · Cost = list price per 1,000 requests · Speed = milliseconds per word-piece (token) generated, lower is faster"),
    whyBox(EX.best_overall, bo.reason,
      also.filter(p => (EX[p.kind] || []).length).map(p => [PROFILE_CHIP[p.kind], EX[p.kind]])));

  const altCard = (p) => {
    const av = byModel[p.model] || {};
    const lines = p.model ? (EX[p.kind] || []) : [];
    return h("div", { class: "pcard" },
      h("div", { class: "lbl" }, PROFILE_TEXT[p.kind]),
      h("div", { class: "row spread", style: "gap:8px" },
        h("div", { class: "pick" }, p.model || "No candidate qualifies"),
        av.verdict_band ? h("span", { class: "band " + av.verdict_band }, BAND_TEXT[av.verdict_band]) : null),
      lines.length
        ? h("div", { class: "why why-lines" }, ...lines.map(l => h("div", {}, l)))
        : h("div", { class: "why" + (p.model ? "" : " none") }, p.model ? (p.tradeoff || p.reason || "") : (p.reason || "")));
  };
  const inelItems = Object.entries(rec.ineligible || {});
  const gateBox = inelItems.length ? h("div", { class: "pcard gate" },
    h("div", { class: "lbl", title: "Only models that pass the quality gate (enough confidence, few hard breaks, few errors) can be recommended. Adjust the gate in Settings → Recommendation." },
      "Didn't pass the quality gate"),
    ...inelItems.map(([m, rs]) => h("div", { style: "font-size:12px;margin-top:6px" },
      h("span", { class: "mono" }, m), h("span", { class: "muted" }, " · " + (rs.join("; ") || "not eligible"))))) : null;
  card.append(h("div", { class: "rec-grid" }, hero,
    h("div", { class: "rec-side" }, ...diff.map(altCard), gateBox,
      !diff.length && !gateBox ? h("div", { class: "pcard" }, h("div", { class: "why none" },
        "Best overall is also the best on quality, cost and latency.")) : null)));
  (rec.notes || []).forEach(n => card.append(h("div", { class: "muted", style: "font-size:12px;margin-top:8px" }, n)));
  card.append(disclaimerNote(rec.latency_disclaimer));
  return card;
}

const BAND_COLOR = { safe_drop_in: "var(--ok)", drop_in_with_prompt_tuning: "var(--drift)",
                     needs_work: "var(--partial)", not_a_drop_in: "var(--bad)" };
const SVGNS = "http://www.w3.org/2000/svg";
const sv = (tag, attrs = {}, text) => {
  const e = document.createElementNS(SVGNS, tag);
  for (const [k, val] of Object.entries(attrs)) if (val != null) e.setAttribute(k, val);
  if (text != null) e.textContent = text;
  return e;
};

// Confidence (0-100) as a ring, coloured by verdict band.
// 95% bootstrap interval of a candidate's Migration Confidence, e.g. "95% CI 79–97" (null if not computed).
function ciText(v, prefix = "95% CI ") {
  return v && v.confidence_ci_low != null && v.confidence_ci_high != null
    ? `${prefix}${Math.round(v.confidence_ci_low)}–${Math.round(v.confidence_ci_high)}` : null;
}
const CI_TITLE = "95% confidence interval from a bootstrap over the requests: the range the score would likely "
  + "land in with a different sample of requests like these. Fewer requests = a wider range.";

function confRing(conf, color, size) {
  const r = size / 2 - 8, c = 2 * Math.PI * r, mid = size / 2, val = Math.max(0, Math.min(100, conf || 0));
  const s = sv("svg", { width: size, height: size, viewBox: `0 0 ${size} ${size}`, role: "img",
    "aria-label": `Migration Confidence ${val.toFixed(0)}%`, class: "conf-ring" });
  s.append(sv("circle", { cx: mid, cy: mid, r, fill: "none", stroke: "var(--surface)", "stroke-width": 10 }),
    sv("circle", { cx: mid, cy: mid, r, fill: "none", stroke: color, "stroke-width": 10, "stroke-linecap": "round",
      "stroke-dasharray": `${c * val / 100} ${c}`, transform: `rotate(-90 ${mid} ${mid})` }),
    sv("text", { x: mid, y: mid + 2, "text-anchor": "middle", fill: "var(--text)", "font-size": size * 0.21, "font-weight": 700 }, `${Math.round(val)}%`),
    sv("text", { x: mid, y: mid + size * 0.17, "text-anchor": "middle", fill: "var(--muted)", "font-size": 10 }, "confidence"));
  return s;
}

// Per-request latency as a range: p50 dot, average tick, bar out to p95 (shared scale).
function latencyRange(v, maxMs) {
  const { latency_p50_ms: p50, latency_avg_ms: avg, latency_p95_ms: p95 } = v;
  if (p50 == null || p95 == null || !maxMs) return h("span", { class: "muted" }, "—");
  const W = 150, X = (ms) => 6 + Math.min(ms, maxMs) / maxMs * (W - 12);
  const s = sv("svg", { width: W, height: 34, viewBox: `0 0 ${W} 34`, class: "lat-range", role: "img",
    "aria-label": `Latency per request: median ${fmtLat(p50)}, average ${fmtLat(avg)}, p95 ${fmtLat(p95)}` });
  s.append(...[sv("title", {}, `p50 ${fmtLat(p50)} · avg ${fmtLat(avg)} · p95 ${fmtLat(p95)}`),
    sv("line", { x1: 6, x2: W - 6, y1: 17, y2: 17, stroke: "var(--border)" }),
    sv("line", { x1: X(p50), x2: X(p95), y1: 17, y2: 17, stroke: "var(--drift)", "stroke-width": 5, "stroke-linecap": "round", "stroke-opacity": 0.45 }),
    sv("circle", { cx: X(p50), cy: 17, r: 4.5, fill: "var(--drift)" }),
    avg != null ? sv("line", { x1: X(avg), x2: X(avg), y1: 9, y2: 25, stroke: "var(--text)", "stroke-width": 1.5 }) : null,
    sv("text", { x: X(p50), y: 7, "text-anchor": "middle", fill: "var(--muted)", "font-size": 9 }, fmtLat(p50)),
    sv("text", { x: Math.min(X(p95), W - 2), y: 33, "text-anchor": "end", fill: "var(--muted)", "font-size": 9 }, fmtLat(p95))]
    .filter(Boolean));
  return s;
}

// Marks the best value in a column/row. A trophy, not an arrow: an arrow reads as
// "this metric went up", which is not what it means.
const BEST_ICON = "🏆";

// A number with a small comparison bar; best value in the column marked with BEST_ICON.
function barCell(text, frac, color, best) {
  return h("td", { class: "tnum barcell" }, h("div", { class: "bc" },
    h("span", {}, text, best ? h("span", { class: "bestmark", title: "Best in this column" }, BEST_ICON) : null),
    frac == null ? null : h("div", { class: "axbar" },
      h("span", { style: `width:${Math.max(3, Math.min(100, 100 * frac))}%;background:${color}` }))));
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
        "Fix the cause above and start a new run."),
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
  if (res.exec_summary) el.append(h("div", { class: "muted", style: "font-size:12px;margin:-6px 0 12px" },
    res.exec_summary + ((res.parallelism || 1) > 1 ? ". Latency was measured with parallel calls." : ".")));
  (res.caveats || []).forEach(c => el.append(h("div", { class: "banner warn", style: "margin-bottom:14px" }, c)));
  // Recommendation (all things considered) — above the per-candidate detail.
  const recPanel = recommendationPanel(res, runId, () => renderResults(el, runId));
  if (recPanel) el.append(recPanel);
  const rec = res.recommendation;
  const axesBy = Object.fromEntries(((rec && rec.axes) || []).map(a => [a.model, a]));
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
    const picks = picksFor(rec, v.model);
    const elig = v.eligible === undefined ? null
      : (v.eligible ? h("div", { class: "elig ok" }, "Passed the quality gate")
                    : h("div", { class: "elig no", title: "A hard break is an answer that changes a critical fact (a dose, an ID, an amount) or breaks the required format." },
                        "Not recommended: " + ((v.ineligible_reasons || []).join(", ") || "failed the quality gate")));
    const card = h("div", { class: "candcard", title: "Click to view this model's evaluations" },
      h("div", { class: "row spread" }, h("strong", {}, v.model), h("span", { class: "band " + v.verdict_band }, BAND_TEXT[v.verdict_band])),
      v.endpoint ? h("div", { class: "muted mono", style: "margin-top:4px;font-size:11px" }, "via " + (ENDPOINT_TEXT[v.endpoint] || v.endpoint)) : null,
      picks.length ? h("div", { class: "row", style: "gap:6px;margin-top:6px;flex-wrap:wrap" },
        ...picks.map(k => h("span", { class: "chip pick" }, PROFILE_CHIP[k]))) : null,
      elig,
      h("div", { class: "metric", style: "margin-top:10px" }, h("div", { class: "value tnum" }, fmtPct(v.migration_confidence)), h("div", { class: "label" }, "Migration Confidence"),
        ciText(v) ? h("div", { class: "ci-line", title: CI_TITLE }, ciText(v)) : null),
      bar,
      h("div", { class: "muted mono", style: "font-size:12px" }, LABELS.map(l => `${(dist[l] || 0)} ${LABEL_TEXT[l]}`).join(" · ")),
      (v.replays || 1) > 1 ? h("div", { class: "muted", style: "font-size:11px;margin-top:2px" },
        `${v.scored} answers · ${v.requests_scored} request${v.requests_scored === 1 ? "" : "s"} × ${v.replays} replays`) : null,
      h("div", { class: "muted", style: "margin-top:8px;font-size:12px" }, `hard-break ${fmtPct(v.hard_break_rate * 100)} · ${costTotal(v)} · ${v.latency_avg_ms == null ? "—" : Math.round(v.latency_avg_ms) + " ms avg†"}`),
      dimBars(v.dimension_averages),
      h("div", { class: "muted", style: "margin-top:8px;font-size:11px;color:var(--accent)" }, "View evaluations →"));
    card.addEventListener("click", () => openCandidate(el, runId, v.model));
    cardsWrap.append(card);
  });
  el.append(h("h3", {}, "Results"), cardsWrap);

  // side-by-side comparison table: inline comparison bars, best value marked, latency as a range plot
  const cands = res.candidates;
  const nums = (f) => cands.map(f).filter(x => typeof x === "number" && isFinite(x));
  // frac for the bar: higher-is-better scales by max; lower-is-better by min/value.
  const scale = (f, dir) => {
    const vs = nums(f); if (!vs.length) return () => [null, false];
    const mx = Math.max(...vs), mn = Math.min(...vs), spread = mx - mn > 1e-9 && vs.length > 1;
    return (v) => {
      const x = f(v);
      if (typeof x !== "number" || !isFinite(x)) return [null, false];
      const frac = dir > 0 ? (mx ? x / mx : 0) : (x > 0 ? mn / x : 1);
      return [frac, spread && x === (dir > 0 ? mx : mn)];
    };
  };
  const cConf = scale(v => v.migration_confidence, 1), cHb = scale(v => v.hard_break_rate, -1),
        cCost = scale(v => (v.cost_known === false ? null : v.cost_per_1k_requests), -1),
        cTok = scale(v => v.ms_per_output_token_p50, -1),
        cScore = scale(v => (axesBy[v.model] || {}).composite, 1);
  const hasD7 = cands.some(v => v.consistency != null);
  const cD7 = scale(v => v.consistency, 1);
  const maxP95 = Math.max(0, ...nums(v => v.latency_p95_ms));
  const heads = [["Candidate"], ["Verdict"], ["Confidence", 1], ["Hard-break", 1], ...(hasD7 ? [["Consistency (D7)", 1]] : []),
                 ["Cost / 1K req", 1], ["vs current", 1],
                 ["Answer length", 1], ["ms / token†", 1], ["Latency per request†"], ["Overall score", 1], ["Picks"]];
  el.append(h("h3", { style: "margin-top:20px" }, "Side-by-side"),
    h("div", { class: "table-scroll" },
    h("table", { class: "sbs" }, h("thead", {}, h("tr", {}, ...heads.map(([t, num]) => h("th", { scope: "col", class: num ? "num" : "" }, t)))),
      h("tbody", {}, ...cands.map(v => {
        const a = axesBy[v.model];
        const picks = picksFor(rec, v.model);
        const out = v.eligible === false;
        const [fc, bc] = cConf(v), [fh, bh] = cHb(v), [fco, bco] = cCost(v), [ft, bt] = cTok(v), [fs, bs] = cScore(v);
        const tr = h("tr", { class: out ? "inel" : "", title: "Click to view this model's evaluations" },
          h("td", { class: "mono" }, v.model),
          h("td", {}, h("span", { class: "band " + v.verdict_band }, BAND_TEXT[v.verdict_band]),
            v.unstable ? h("span", { class: "band unstable", style: "margin-left:4px",
              title: "Average consistency (D7) is below the instability threshold: answers to the same request vary a lot." }, "Unstable") : null),
          (() => {
            const td = barCell(fmtPct(v.migration_confidence), fc, "var(--ok)", bc);
            const ci = ciText(v, "");
            const tied = (rec && rec.quality_ties || []).includes(v.model);
            if (ci) td.append(h("div", { class: "ci-line", style: "text-align:right", title: CI_TITLE },
              ci, tied ? h("span", { class: "tiechip", title: "Statistically tied with the best-quality candidate: the data can't separate them on quality" }, "≈ tied") : null));
            return td;
          })(),
          barCell(fmtPct(v.hard_break_rate * 100), fh, "var(--bad)", bh),
          ...(hasD7 ? [(() => {
            const [fd, bd] = cD7(v);
            const td = barCell(v.consistency == null ? "—" : v.consistency.toFixed(2), fd, "var(--accent)", bd);
            if (v.consistency != null) td.append(h("div", { class: "muted", style: "font-size:11px;text-align:right" },
              `${v.unstable_requests || 0} unstable` + (v.flip_rate ? ` · ${fmtPct(v.flip_rate * 100)} flip` : "")));
            return td;
          })()] : []),
          barCell(costPer1k(v), fco, "var(--accent)", bco),
          h("td", { class: "tnum num" }, pctSigned(v.cost_vs_current_pct)),
          h("td", { class: "tnum num", title: "Answer length vs your current model: 1.5× means 50% longer (and costlier)" },
            v.output_token_ratio == null ? "—" : `${v.output_token_ratio.toFixed(2)}×`),
          barCell(msTok(v), ft, "var(--drift)", bt),
          h("td", {}, latencyRange(v, maxP95)),
          barCell(a && a.composite != null ? a.composite.toFixed(1) : "—", fs, "var(--ok)", bs),
          h("td", {}, picks.length
            ? h("div", { class: "row", style: "gap:4px;flex-wrap:nowrap" }, ...picks.map(k => h("span", { class: "chip" + (k === "best_overall" ? " pick" : "") }, PROFILE_CHIP[k])))
            : out ? h("span", { class: "muted", style: "font-size:12px" }, "Not eligible: " + ((v.ineligible_reasons || []).join("; ") || "failed the quality gate"))
            : h("span", { class: "muted" }, "—")));
        tr.addEventListener("click", () => openCandidate(el, runId, v.model));
        return tr;
      })))),
    h("div", { class: "legend", style: "margin-top:8px" },
      h("span", {}, "Latency per request: "),
      h("span", {}, h("i", { class: "dot", style: "background:var(--drift)" }), "median (p50)"),
      h("span", {}, h("i", { class: "tick" }), "average"),
      h("span", {}, h("i", { style: "background:rgba(34,211,238,.45);width:18px" }), "out to p95 (slowest 5%)")),
    h("div", { class: "muted", style: "font-size:11px;margin-top:6px" },
      "Bars compare candidates within each column; 🏆 marks the best. The range under Confidence is its 95% interval; "
      + "≈ tied means it can't be separated from the best on quality. Dimmed rows didn't pass the quality gate. "
      + "Cost is list price at current public Bedrock rates, assuming no prompt caching; vs current compares the same requests against today's model, when its price is known. "
      + "† Latency is indicative only (see the note above). Overall score is the weighted recommendation score; only models that passed the gate get one."));

  // D1–D6 × candidate breakdown: why one model beats another, without drilling in.
  const breakdown = dimBreakdown(res.candidates, rec);
  if (breakdown) {
    el.append(h("div", { class: "row spread", style: "margin-top:20px" },
      h("h3", { style: "margin:0" }, "Dimension breakdown"),
      h("span", { class: "muted", style: "font-size:11px" }, "average score per dimension · 0.00–1.00 · green ≥ 0.85 · amber 0.60–0.85 · red < 0.60 · best per row 🏆"
        + (rec && !rec.none_eligible ? " · tinted columns are the recommended picks (outlined = Best overall)" : ""))),
      breakdown);
  }
  if (res.candidates.some(v => v.consistency != null)) el.append(leastConsistentCard(el, runId, res.candidates));

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

async function loadSamples(el, runId, model, opts = {}) {
  const drill = $("#drill"); clear(drill); drill.append(h("div", { class: "skeleton" }));
  const data = await api(`/runs/${runId}/samples?candidate=${encodeURIComponent(model)}&limit=500`
    + (opts.unstable ? "&unstable=true" : ""));
  clear(drill);
  const detail = h("div", { id: "sampleDetail" });
  const reps = Math.max(1, ...data.items.map(p => p.replays || 1));
  const rows = data.items.map(p => {
    const c = p.consistency;
    const tr = h("tr", { title: "Click for full request/response detail" },
      h("td", { class: "mono" }, p.sample_id),
      reps > 1 ? h("td", { class: "tnum" }, `${(p.replay_index || 0) + 1} / ${reps}`) : null,
      h("td", {}, h("span", { class: "pill " + labelKey(p) }, LABEL_TEXT[labelKey(p)])),
      h("td", { class: "tnum" }, (p.d1_semantic?.score ?? 0).toFixed(2)),
      h("td", { class: "tnum" }, (p.d2_format?.score ?? 0).toFixed(2)),
      h("td", { class: "tnum" }, (p.d3_factual?.score ?? 0).toFixed(2)),
      h("td", { class: "tnum" }, (p.d4_verbosity?.score ?? 0).toFixed(2)),
      reps > 1 ? h("td", { class: "tnum" }, c && c.d7 != null
        ? [h("span", { style: `color:${scoreColor(c.d7)}` }, c.d7.toFixed(2)), c.unstable ? h("span", { class: "band unstable", style: "margin-left:4px" }, "Unstable") : null]
        : "—") : null,
      h("td", { class: "muted" }, p.judge_reason || (p.hard_break ? "hard-break" : "—")));
    tr.addEventListener("click", () => showSampleDetail(detail, runId, model, p.sample_id, p.replay_index || 0));
    return tr;
  });
  const filter = reps > 1 || opts.unstable ? h("label", { class: "row", style: "gap:6px;font-size:12px;cursor:pointer" },
    h("input", { type: "checkbox", checked: !!opts.unstable,
      onchange: (e) => loadSamples(el, runId, model, { unstable: e.target.checked }) }),
    "Unstable requests only") : null;
  const heads = ["Request", ...(reps > 1 ? ["Replay"] : []), "Label", "D1", "D2", "D3", "D4", ...(reps > 1 ? ["D7"] : []), "Reason"];
  drill.append(h("div", { class: "row spread" }, h("h3", {}, `Evaluations · ${model} (${data.total}${reps > 1 ? " answers" : ""})`), filter),
    h("div", { class: "muted", style: "font-size:12px;margin-bottom:6px" },
      reps > 1 ? `Each request was sent ${reps} times; every answer is listed and scored. D7 is the request's consistency across its answers. Click a row to see it, with tabs for the other replays.`
               : "Every request evaluated for this candidate. Click a row to see the original vs candidate request/response, models and latency."),
    h("table", {}, h("thead", {}, h("tr", {}, ...heads.map(t => h("th", {}, t)))),
      h("tbody", {}, ...rows)),
    detail);
}

async function showSampleDetail(host, runId, model, sampleId, replay) {
  clear(host);
  host.append(h("div", { class: "skeleton", style: "margin-top:14px" }));
  let d;
  try {
    d = await api(`/runs/${runId}/samples/${encodeURIComponent(sampleId)}?candidate=${encodeURIComponent(model)}`
      + (replay != null ? `&replay=${replay}` : ""));
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
    (d.replays || 1) > 1 ? replayTabs(host, runId, model, sampleId, d, respText, ms) : null,
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

// ---- Replays: tabs per answer, the request's D7, and a diff of the two most different answers.
const LABEL_DOT = { compatible: "var(--ok)", compatible_with_drift: "var(--drift)", partial: "var(--partial)", incompatible: "var(--bad)" };
function labelDots(labels) {
  return h("span", { class: "ldots", role: "img", "aria-label": "Labels per replay: " + labels.map(l => LABEL_TEXT[l] || l).join(", ") },
    ...labels.map(l => h("i", { style: `background:${LABEL_DOT[l] || "var(--muted)"}`, title: LABEL_TEXT[l] || l })));
}

function replayTabs(host, runId, model, sampleId, d, respText, ms) {
  const cons = d.consistency || {};
  const tabs = h("div", { class: "seg", role: "tablist", "aria-label": "Replays of this request" },
    ...(d.answers || []).map(a => h("button", {
      role: "tab", "aria-selected": a.replay_index === d.replay ? "true" : "false",
      class: a.replay_index === d.replay ? "active" : "",
      onclick: () => showSampleDetail(host, runId, model, sampleId, a.replay_index) },
      h("i", { class: "tabdot", style: `background:${a.status === "error" ? "var(--bad)" : (LABEL_DOT[a.label] || "var(--muted)")}` }),
      `Replay ${a.replay_index + 1}`)));
  const box = h("div", { class: "replaybox" },
    h("div", { class: "row spread", style: "flex-wrap:wrap;gap:8px" },
      h("div", { class: "row", style: "gap:10px;flex-wrap:wrap" }, h("span", { class: "muted", style: "font-size:12px" },
        `Sent ${d.replays} times`), tabs),
      cons.d7 != null
        ? h("div", { class: "row", style: "gap:8px;font-size:12px" },
            h("span", {}, "Consistency (D7) ", h("b", { class: "tnum", style: `color:${scoreColor(cons.d7)}` }, cons.d7.toFixed(2))),
            h("span", { class: "muted" }, `answers ${(cons.answer_agreement ?? 0).toFixed(2)} · verdicts ${(cons.verdict_agreement ?? 0).toFixed(2)}`),
            cons.unstable ? h("span", { class: "band unstable" }, "Unstable") : null,
            cons.flip ? h("span", { class: "band not_a_drop_in", title: "Some answers hard-break and some don't" }, "Flips") : null)
        : h("span", { class: "muted", style: "font-size:12px" }, "Consistency not measured (fewer than 2 successful answers)")));
  const pair = cons.most_different_pair;
  const byRep = Object.fromEntries((d.answers || []).map(a => [a.replay_index, a]));
  if (pair && byRep[pair[0]] && byRep[pair[1]]) {
    const A = byRep[pair[0]], B = byRep[pair[1]];
    const asSide = (a) => ({ model: `replay ${a.replay_index + 1} · ${LABEL_TEXT[a.label] || a.status}`, status: a.status, error: a.error,
      latency_ms: a.latency_ms, usage: {}, cost_estimate: a.cost_estimate, response: { text: a.text, tool_calls: a.tool_calls } });
    box.append(h("details", { class: "cmpreplays" },
      h("summary", {}, `Compare the two most different answers (replay ${pair[0] + 1} vs ${pair[1] + 1})`),
      responseCompare({ ...asSide(A), response: { text: A.text, tool_calls: A.tool_calls } }, asSide(B), respText, ms,
        [`Replay ${pair[0] + 1}`, `Replay ${pair[1] + 1}`])));
  }
  return box;
}

// Results page: requests whose answers varied most across replays.
function leastConsistentCard(el, runId, cands) {
  const measured = cands.filter(v => v.consistency != null);
  const sel = h("select", { style: "width:auto", "aria-label": "Candidate" }, ...measured.map(v => h("option", { value: v.model }, v.model)));
  const body = h("div", {});
  const load = async () => {
    body.replaceChildren(h("div", { class: "skeleton" }));
    let data;
    try { data = await api(`/runs/${runId}/candidates/${encodeURIComponent(sel.value)}/consistency?limit=10`); }
    catch (e) { body.replaceChildren(h("div", { class: "muted" }, "Could not load: " + (e && e.message ? e.message : "error"))); return; }
    if (!data.items.length) { body.replaceChildren(h("div", { class: "muted", style: "font-size:12px" }, "No requests with two or more successful answers.")); return; }
    body.replaceChildren(h("table", { class: "lc-table" },
      h("thead", {}, h("tr", {}, ...["Request", "D7", "Labels per replay", "", ""].map(t => h("th", { scope: "col" }, t)))),
      h("tbody", {}, ...data.items.map(it => {
        const tr = h("tr", { tabindex: "0", title: "Open this request's answers" },
          h("td", {}, h("div", { class: "mono" }, it.sample_id), h("div", { class: "muted lc-prev" }, it.prompt_preview || "")),
          h("td", { class: "tnum" }, h("b", { style: `color:${scoreColor(it.d7)}` }, it.d7.toFixed(2))),
          h("td", {}, labelDots(it.labels)),
          h("td", {}, it.flip ? h("span", { class: "band not_a_drop_in", title: "Some answers hard-break and some don't" }, "Flips") : null),
          h("td", {}, it.unstable ? h("span", { class: "band unstable" }, "Unstable") : null));
        const open = () => {
          const s = document.getElementById("candSelect"); if (s) s.value = sel.value;
          loadSamples(el, runId, sel.value).then(() => {
            const det = document.getElementById("sampleDetail");
            if (det) showSampleDetail(det, runId, sel.value, it.sample_id, (it.most_different_pair || [0])[0]);
          });
        };
        tr.addEventListener("click", open);
        tr.addEventListener("keydown", (e) => { if (e.key === "Enter") { e.preventDefault(); open(); } });
        return tr;
      }))));
  };
  sel.addEventListener("change", load);
  const card = h("div", { style: "margin-top:20px" },
    h("div", { class: "row spread" }, h("h3", { style: "margin:0" }, "Least consistent requests"),
      h("div", { class: "row", style: "gap:8px" }, h("span", { class: "muted", style: "font-size:12px" }, "Candidate"), sel)),
    h("div", { class: "muted", style: "font-size:11px;margin:4px 0 8px" },
      "Requests whose answers varied most when replayed (lowest D7 first). Each dot is one replay's label. Click a row to compare its answers."),
    body);
  load();
  return card;
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

function responseCompare(o, c, respText, ms, labels) {
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
        h("span", { class: "d-del" }, labels ? `only in ${labels[0]}` : "removed from original"),
        h("span", { class: "d-ins" }, labels ? `only in ${labels[1]}` : "added by candidate"),
        stats)),
    h("div", { class: "grid cols-2" },
      h("div", {}, head(labels ? labels[0] : "Original response", origSub), origPane),
      h("div", {}, head(labels ? labels[1] : "Candidate response", candSub), candPane)));
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

// D1–D6 × candidate table; highlights the best candidate per dimension, and labels the
// columns of the models the recommendation picked (Best overall / quality / economical / latency).
function dimBreakdown(cands, rec) {
  const dims = DIMENSIONS.filter(d => cands.some(v => typeof (v.dimension_averages || {})[d.key] === "number"));
  if (!dims.length || !cands.length) return null;
  const order = ["best_overall", "best_quality", "most_economical", "lowest_latency"];
  const picks = cands.map(v => picksFor(rec, v.model).sort((a, b) => order.indexOf(a) - order.indexOf(b)));
  const colCls = (i) => !picks[i].length ? "" : (picks[i].includes("best_overall") ? "pickcol bo" : "pickcol");
  const head = h("tr", {}, h("th", {}, "Dimension"), ...cands.map((v, i) => h("th", { class: ("mono " + colCls(i)).trim() },
    v.model,
    picks[i].length ? h("div", { class: "colpicks" },
      ...picks[i].map(k => h("span", { class: "chip" + (k === "best_overall" ? " pick" : ""), title: PROFILE_TEXT[k] }, PROFILE_CHIP[k]))) : null)));
  const body = dims.map((d, ri) => {
    const vals = cands.map(v => (v.dimension_averages || {})[d.key]);
    const nums = vals.filter(x => typeof x === "number");
    const best = nums.length > 1 ? Math.max(...nums) : null;
    const spread = nums.length > 1 && Math.max(...nums) - Math.min(...nums) > 0.001;
    const last = ri === dims.length - 1 ? " last" : "";
    return h("tr", { class: "static" },
      h("td", { title: d.desc },
        h("b", {}, d.code), " " + d.name,
        d.hard ? h("span", { class: "hardtag", title: "A failure here caps the verdict at Incompatible" }, "hard-break") : null),
      ...vals.map((x, i) => typeof x !== "number"
        ? h("td", { class: ("muted heat " + colCls(i) + (colCls(i) ? last : "")).trim() }, "—")
        : h("td", { class: ("heat " + colCls(i) + (colCls(i) ? last : "")).trim() }, h("div", { class: "heatcell tnum", style: `background:${heatBg(x)}`,
            title: `${d.name}: ${x.toFixed(2)}` + ((spread && x === best) ? " (best)" : "") },
            x.toFixed(2), (spread && x === best) ? h("span", { class: "bestmark", title: "Best on this dimension" }, BEST_ICON) : null))));
  });
  return h("table", { class: "breakdown heatmap" }, h("thead", {}, head), h("tbody", {}, ...body));
}

// Heatmap tile colour: same thresholds as scoreColor (≥0.85 good, ≥0.60 watch, else poor),
// stronger tint the further into the band.
function heatBg(v) {
  if (v >= 0.85) return `rgba(52,211,153,${(0.22 + 0.5 * Math.min(1, (v - 0.85) / 0.15)).toFixed(2)})`;
  if (v >= 0.6) return `rgba(245,181,68,${(0.22 + 0.4 * Math.min(1, (v - 0.6) / 0.25)).toFixed(2)})`;
  return "rgba(244,104,95,.55)";
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
      h("span", { class: "muted", style: "font-size:12px" }, LABEL_EXPLAIN[label] || "")));
  // Native append prints null as the text "null": add the hard-break banner only when there is one.
  if (sc.hard_break) {
    wrap.append(h("div", { class: "banner err", style: "margin:6px 0;font-size:12px" },
      "Hard break — a broken format, diverged fact, or mismatched tool call caps this at Incompatible regardless of the other scores."));
  }

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

const EFFORTS = ["none", "minimal", "low", "medium", "high", "xhigh"];
// Efforts a model accepts (from /models; probed live against Bedrock). Unknown models: all.
const baseModel = (m) => String(m || "").split("@")[0];
const effortsFor = (m) => (MODELS?.models?.[baseModel(m)]?.reasoning_efforts) || EFFORTS;
const effortApplies = (m) => MODELS?.models?.[baseModel(m)]?.effort_applies !== false;
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
  const remEfforts = effortsFor(rem.candidate);
  if (!remEfforts.includes(draft.effort)) draft.effort = remEfforts.includes("low") ? "low" : remEfforts[0];
  const effortSel = h("select", { style: "width:auto", "aria-label": "Reasoning effort to test" },
    ...remEfforts.map(e => h("option", { value: e }, e)));
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
      "model_list:", `  - model_name: ${String(rem.candidate || "").split("@")[0]}`, "    litellm_params:", "      # ...existing params...",
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
  const effortSeg = h("div", { class: "seg" }, ...["none", "low", "medium", "high", "xhigh"].map(x =>
    h("button", { class: x === s.default_reasoning_effort ? "active" : "",
      onclick: () => { draft.default_reasoning_effort = x; effortSeg.querySelectorAll("button").forEach(b => b.classList.toggle("active", b.textContent === x)); } }, x)));
  const judgeInput = h("input", { type: "text", value: s.judge_model, oninput: e => draft.judge_model = e.target.value });
  const cc = s.concurrency || {};
  const parInput = h("input", { type: "number", min: "1", max: String(cc.max_parallelism || 16), step: "1",
    value: String(s.parallelism), style: "width:90px", "aria-describedby": "parHelp",
    oninput: e => draft.parallelism = parseInt(e.target.value, 10) });
  const adaptBox = h("input", { type: "checkbox", id: "adaptConc", checked: cc.adaptive_concurrency !== false,
    onchange: e => { draft.adaptive_concurrency = e.target.checked; } });
  const capInput = h("input", { type: "number", min: "1", max: "64", step: "1", value: String(cc.max_inflight_calls ?? 16),
    style: "width:90px", oninput: e => draft.max_inflight_calls = parseInt(e.target.value, 10) });
  const ffInput = h("input", { type: "number", min: "5", max: "100", step: "5", placeholder: "off", style: "width:90px",
    value: cc.fail_fast_error_share == null ? "" : String(Math.round(cc.fail_fast_error_share * 100)),
    oninput: e => { const n = parseFloat(e.target.value); draft.fail_fast_error_share = Number.isFinite(n) ? n / 100 : null; } });
  const concurrencyBlock = h("div", { class: "field", style: "margin-top:4px" },
    h("div", { style: "font-weight:600;margin-bottom:6px" }, "Parallel calls"),
    h("div", { class: "grid cols-2" },
      h("label", { class: "field" }, h("span", {}, `Parallel calls per run (1–${cc.max_parallelism || 16})`), parInput,
        h("div", { id: "parHelp", class: "muted", style: "font-size:11px;margin-top:4px" },
          "How many model calls are sent at the same time. Higher is faster but more likely to hit Bedrock throttling; ModelShift slows down automatically if it does. Copied into each run when it starts.")),
      h("label", { class: "field" }, h("span", {}, "Max calls in flight across all runs"), capInput,
        h("div", { class: "muted", style: "font-size:11px;margin-top:4px" }, "A server-wide cap: when several runs are live they share it."))),
    h("div", { class: "grid cols-2" },
      h("label", { class: "row", style: "gap:8px;cursor:pointer", for: "adaptConc" }, adaptBox,
        h("span", {}, "Slow down automatically when throttled, and speed back up")),
      h("label", { class: "field" }, h("span", {}, "Stop a run early when this % of the last 20 calls fail with the same error (blank = off)"), ffInput)));
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

  // Recommendation (quality gate, tolerance, latency metric, preset weights).
  const rs = s.recommendation || {};
  const recDraft = {};
  const numIn = (key, scale, width = "90px") => h("input", { type: "number", step: "any",
    value: rs[key] == null ? "" : String(+(rs[key] * scale).toFixed(4)), style: `width:${width}`,
    oninput: e => { const n = parseFloat(e.target.value); if (Number.isFinite(n)) recDraft[key] = n / scale; else delete recDraft[key]; } });
  // Optional gate: leaving it blank switches the gate off (saved as null).
  const optIn = (key, scale, placeholder) => h("input", { type: "number", step: "any", placeholder,
    value: rs[key] == null ? "" : String(+(rs[key] * scale).toFixed(4)), style: "width:90px",
    oninput: e => { const n = parseFloat(e.target.value); recDraft[key] = Number.isFinite(n) ? n / scale : null; } });
  const latSeg = h("div", { class: "seg" }, ...[["per_token", "ms per output token"], ["p50", "median end-to-end"]].map(([k, t]) =>
    h("button", { class: (rs.latency_metric || "per_token") === k ? "active" : "",
      onclick: () => { recDraft.latency_metric = k; latSeg.querySelectorAll("button").forEach((b, i) => b.classList.toggle("active", i === (k === "p50" ? 1 : 0))); } }, t)));
  const tieBox = h("input", { type: "checkbox", id: "statTies", checked: rs.statistical_ties !== false,
    onchange: e => { recDraft.statistical_ties = e.target.checked; } });
  const presetRows = Object.entries(rs.weight_presets || {}).map(([name, w]) => {
    const cell = (axis) => h("input", { type: "number", step: "0.05", min: "0", value: String(w[axis]), style: "width:80px",
      oninput: e => {
        const n = parseFloat(e.target.value);
        recDraft.weight_presets = recDraft.weight_presets || JSON.parse(JSON.stringify(rs.weight_presets));
        if (Number.isFinite(n)) recDraft.weight_presets[name][axis] = n;
      } });
    const label = (PRESETS.find(p => p[0] === name) || [name, name])[1];
    return h("tr", {}, h("td", {}, label), h("td", {}, cell("quality")), h("td", {}, cell("cost")), h("td", {}, cell("latency")));
  });
  const missingPrice = h("div", { class: "muted", style: "font-size:12px;margin-top:8px" });
  (async () => {
    try {
      const m = await api("/models");
      const missing = Object.keys(m.models || {}).filter(k => !(k in (s.price_table || {})));
      missingPrice.textContent = missing.length
        ? `No price configured for: ${missing.join(", ")}. Their cost shows as "Price not configured" and they are left out of cost ranking — add a row below.`
        : "Every catalog model has a price.";
    } catch (e) { missingPrice.textContent = ""; }
  })();

  const save = async () => {
    saveMsg.textContent = "Saving…";
    try {
      const body = { ...draft };
      if (Object.keys(recDraft).length) body.recommendation = recDraft;
      // only send changed scalar fields that are valid numbers
      ["parallelism", "small_sample_threshold", "max_inflight_calls"].forEach(k => { if (Number.isNaN(body[k])) delete body[k]; });
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
        h("label", { class: "field" }, h("span", {}, "Default reasoning effort"), effortSeg)),
      concurrencyBlock,
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
      h("h3", {}, "Recommendation"),
      h("div", { class: "muted", style: "margin:-4px 0 10px" },
        "Only candidates that pass every quality gate can be recommended. Changes apply to past runs too (results are re-ranked, no model calls)."),
      h("div", { class: "grid cols-2" },
        h("label", { class: "field" }, h("span", {}, "Minimum Migration Confidence (0–100)"), numIn("min_confidence", 1)),
        h("label", { class: "field" }, h("span", {}, "Maximum hard-break rate (%)"), numIn("max_hard_break_rate", 100))),
      h("div", { class: "grid cols-2" },
        h("label", { class: "field" }, h("span", {}, "Maximum error rate (%)"), numIn("max_error_rate", 100)),
        h("label", { class: "field" }, h("span", {}, "Quality window for the economical / latency / overall picks (confidence points below the best)"), numIn("quality_tolerance", 1))),
      h("div", { class: "grid cols-2" },
        h("label", { class: "field" }, h("span", {}, "Minimum timed calls before latency counts"), numIn("min_latency_samples", 1)),
        h("label", { class: "field" }, h("span", {}, "Latency measure used for ranking"), latSeg)),
      h("label", { class: "row", style: "gap:8px;align-items:flex-start;margin:6px 0 4px", for: "statTies" }, tieBox,
        h("span", {}, "Treat statistically tied quality as equal",
          h("div", { class: "muted", style: "font-size:12px" },
            "When a candidate's confidence can't be separated from the best (95% bootstrap interval of the difference "
            + "includes 0), it gets the full quality score, so cost and latency decide between them. Applies inside the quality window only."))),
      h("div", { class: "muted", style: "margin:10px 0 6px;font-weight:600" }, "Replays and consistency (D7)"),
      h("div", { class: "grid cols-2" },
        h("label", { class: "field" }, h("span", {}, "Default times to send each request when replays are on (2–5)"), numIn("default_replays", 1)),
        h("label", { class: "field" }, h("span", {}, "Warn before launch above this many calls"), numIn("replay_call_ceiling", 1))),
      h("div", { class: "grid cols-2" },
        h("label", { class: "field" }, h("span", {}, "Unstable below this consistency (0–1)"), numIn("instability_threshold", 1)),
        h("label", { class: "field" }, h("span", {}, "Gate: minimum consistency (0–1, blank = off)"), optIn("min_consistency", 1, "off"))),
      h("div", { class: "grid cols-2" },
        h("label", { class: "field" }, h("span", {}, "Gate: maximum flip rate (%, blank = off) — requests where some answers hard-break and some don't"), optIn("max_flip_rate", 100, "off")),
        h("div", { class: "muted", style: "font-size:11px;align-self:center" },
          "Consistency never changes Migration Confidence or labels; these gates only decide whether a model can be recommended.")),
      h("div", { class: "muted", style: "margin:6px 0" }, "Preset weights (relative; normalised to 100%, quality never below 20%)"),
      h("table", {}, h("thead", {}, h("tr", {}, ...["Preset", "Quality", "Cost", "Latency"].map(t => h("th", { scope: "col" }, t)))),
        h("tbody", {}, ...presetRows))),
    h("div", { class: "card" },
      h("h3", {}, "Price table (USD per 1K tokens)"),
      h("div", { class: "muted", style: "margin:-4px 0 6px;font-size:12px" },
        "Public Amazon Bedrock list prices (Standard tier, short context). Reasoning tokens bill at the output rate; prompt caching is not modelled."),
      missingPrice,
      h("table", {}, h("thead", {}, h("tr", {}, ...["Model", "Input /1K", "Output /1K"].map(t => h("th", {}, t)))),
        h("tbody", {}, ...priceRows))),
    h("div", { class: "row", style: "margin-top:16px" },
      h("button", { class: "btn", onclick: save }, "Save settings"), saveMsg));
}

boot();
