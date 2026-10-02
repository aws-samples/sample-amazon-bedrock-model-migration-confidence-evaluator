// ModelShift — Guide view (landing page for new users).
// Interactive pipeline of a run (click / arrow keys / auto tour), a "Start here"
// checklist remembered in this browser, key terms, and tips. Uses app.js helpers
// (h, go, api) and the design tokens in styles.css. Static copy only — no user data
// is ever inserted as HTML.

const GUIDE_ICON = {
  logs: '<path d="M5 4h10l4 4v12H5z"/><path d="M15 4v4h4M8 12h8M8 16h6"/>',
  filter: '<path d="M4 5h16l-6 8v5l-4 2v-7z"/>',
  config: '<circle cx="12" cy="12" r="3"/><path d="M12 2v3M12 19v3M4.2 4.2l2.1 2.1M17.7 17.7l2.1 2.1M2 12h3M19 12h3M4.2 19.8l2.1-2.1M17.7 6.3l2.1-2.1"/>',
  run: '<path d="M7 5l12 7-12 7z"/>',
  score: '<path d="M4 20V10M10 20V4M16 20v-7M22 20H2"/>',
  compare: '<rect x="3" y="4" width="7" height="16" rx="1.5"/><rect x="14" y="4" width="7" height="16" rx="1.5"/>',
  fix: '<path d="M14 6l4 4-9 9H5v-4z"/><path d="M13 7l4 4"/>',
  share: '<path d="M6 3h9l4 4v14H6z"/><path d="M9 13h6M9 17h4"/>',
};

// Small static preview snippets (HTML we author; never user content).
const gBar = (label, v) => {
  const col = v >= 0.85 ? "var(--ok)" : v >= 0.6 ? "var(--partial)" : "var(--bad)";
  return `<div class="g-barrow"><span>${label}</span><div class="minibar"><span style="width:${v * 100}%;background:${col}"></span></div>`
    + `<b style="color:${col}">${v.toFixed(2)}</b></div>`;
};

const GUIDE_STEPS = [
  { k: "logs", t: "Bring your logs", where: "new",
    why: "Upload a LiteLLM log export or point at an S3 prefix. ModelShift turns every logged request into an evaluation it can replay, and tells you which rows it had to skip.",
    pts: ["<b>Upload</b> a JSON / JSONL export, or enter an <b>S3 URI</b>",
      "See rows <b>found</b>, <b>evaluable</b> and <b>dropped</b> — with the reason (redacted, tool calls, empty)",
      "It detects the <b>model you use today</b> and the <b>teams</b> and <b>users</b> in the logs"],
    decide: "Is this the traffic you want to migrate?", go: "New Run → 1 · Source",
    pv: `<div class="g-cap">Ingestion summary</div><div class="g-stats"><div><b>1,248</b><span>rows found</span></div>
      <div><b style="color:var(--ok)">1,102</b><span>evaluable</span></div><div><b style="color:var(--partial)">146</b><span>dropped</span></div></div>
      <div class="g-mini row spread" style="margin-top:12px"><span class="mono">gpt-4.1</span><span class="muted mono">current model · 1,102 rows</span></div>` },
  { k: "filter", t: "Narrow the scope", where: "new",
    why: "Shared logs often mix many apps. Keep only the teams or users you're migrating, and cap the number of records for a quick, cheap first pass.",
    pts: ["Filter by <b>team</b> and <b>user</b>", "<b>Limit records</b> to try a sample first",
      "A <b>low-signal</b> warning appears when too few evaluations are left to trust"],
    decide: "The whole workload, or one team first?", go: "New Run → 1 · Source",
    pv: `<div class="g-cap">Filters</div><div class="row" style="flex-wrap:wrap;gap:6px"><span class="pill compatible">claims-triage</span>
      <span class="pill compatible">pharmacy</span><span class="g-chip">member-chat</span></div>
      <div class="g-mini" style="margin-top:12px"><div class="muted" style="font-size:12px">Limit records</div>
      <div class="row spread" style="margin-top:4px"><span class="mono">200</span><span class="muted mono">of 640 matching</span></div></div>` },
  { k: "config", t: "Choose candidates", where: "new",
    why: "Pick where calls go and which newer models to try. The models we recommend for the one you're replacing are pre-selected.",
    pts: ["Endpoint: <b>Bedrock Runtime</b>, <b>Bedrock Mantle</b>, or your <b>LiteLLM proxy</b> (its models are listed automatically)",
      "Candidates, including a <b>cross-family</b> baseline, and <b>reasoning effort</b>",
      "Optional <b>judge steering</b>: tell the AI judge what matters for your use case",
      "Check the <b>number of calls and estimated cost</b> before you launch"],
    decide: "Which models, and what must the judge be strict about?", go: "New Run → 2 · Configuration",
    pv: `<div class="g-cap">Candidates</div>
      <div class="g-mini row spread g-sel"><b>gpt-5.6-luna</b><span class="muted" style="font-size:11px">recommended start</span></div>
      <div class="g-mini row spread" style="margin-top:6px"><b>gpt-5.6-terra</b><span class="muted" style="font-size:11px">candidate</span></div>
      <div class="g-mini row spread" style="margin-top:6px"><b>claude-sonnet-4.5</b><span class="muted" style="font-size:11px">cross-family</span></div>
      <div class="muted mono" style="margin-top:10px">≈ 600 calls · est. $1.84</div>` },
  { k: "run", t: "Replay", where: "run",
    why: "Every request is sent to every candidate exactly as your app sent it. Progress is live; if a call fails you see the full error, and failed calls are left out of the scores rather than guessed.",
    pts: ["<b>Live progress</b> per candidate", "Show <b>errors only</b> to spot access or quota problems fast",
      "<b>Cancel</b> any time; the full error and stack trace are kept"],
    decide: "Is anything failing for every request (access, quota)?", go: "Run details → Progress",
    pv: `<div class="g-cap">Live log</div><div class="g-log"><div>✓ gpt-5.6-luna · req_0412 · 812 ms</div><div>✓ gpt-5.6-terra · req_0412 · 1.2 s</div>
      <div style="color:var(--bad)">✗ gpt-5.4 · req_0413 · 403 access denied</div><div>✓ gpt-5.6-luna · req_0413 · 640 ms</div></div>
      <div class="progress" style="margin-top:12px"><span class="g-grow" style="width:64%"></span></div>` },
  { k: "score", t: "Score every answer", where: "run",
    why: "Each candidate answer is compared with the answer your current model gave, on six checks. An AI judge explains its verdict in plain words.",
    pts: ["<b>D1–D6</b>: meaning, format, facts, length, instructions, tool calls",
      "A <b>hard break</b> — broken format, a wrong fact, or a mismatched tool call — makes the answer Incompatible",
      "Each answer is labelled <b>Compatible</b>, <b>Compatible w/ drift</b>, <b>Partial</b> or <b>Incompatible</b>"],
    decide: "Do the judge's reasons match what your users care about?", go: "Results → View evaluations",
    pv: `<div class="g-cap">One evaluation</div>${gBar("D1 Meaning", 0.92)}${gBar("D2 Format", 1)}${gBar("D3 Facts", 0.55)}${gBar("D4 Length", 0.78)}
      <span class="pill incompatible" style="margin-top:8px">Incompatible · hard break</span>` },
  { k: "compare", t: "Compare candidates", where: "run",
    why: "Each candidate gets a Migration Confidence score and a verdict. From there you can open any evaluation to see exactly how the two answers differ.",
    pts: ["<b>Result cards</b>: confidence, the label mix, and per-check bars", "<b>Side-by-side</b>: confidence, hard breaks, cost, latency",
      "<b>Dimension breakdown</b>: which model wins on which check",
      "<b>Evaluations</b>: original vs candidate, with the differences highlighted"],
    decide: "Which candidate — and what's holding it back?", go: "Run details → Results",
    pv: `<div class="g-cap">Results</div><div class="row" style="gap:8px;align-items:stretch">
      <div class="g-mini" style="flex:1;border-top:3px solid var(--ok)"><b style="font-size:12px">gpt-5.6-terra</b><div class="g-big">88%</div><span class="band safe_drop_in" style="margin:0">Safe drop-in</span></div>
      <div class="g-mini" style="flex:1;border-top:3px solid var(--drift)"><b style="font-size:12px">gpt-5.6-luna</b><div class="g-big">74%</div><span class="band drop_in_with_prompt_tuning" style="margin:0">Prompt tuning</span></div></div>
      <div class="g-diff mono">Your copay is <span class="d-del">$15</span> <span class="d-ins">$10</span> for a 90-day supply.</div>` },
  { k: "fix", t: "Fix and re-test", where: "run",
    why: "Failures are grouped by cause, each with a ready-made fix. Edit it, apply it and re-test — the re-test also re-runs answers that were already fine, so a fix can't quietly break them.",
    pts: ["A suggested <b>prompt instruction</b> or <b>reasoning-effort</b> change per group", "<b>Estimate</b> calls, cost and time first",
      "<b>Regression check</b> on evaluations that were already Compatible", "<b>Copy patch</b> to apply in your app or LiteLLM config"],
    decide: "Does the fix help without breaking anything else?", go: "Results → See remediations",
    pv: `<div class="g-cap">Re-test result</div><div class="g-stats"><div><span>this group</span><b>40% → <em style="color:var(--ok)">90%</em></b></div>
      <div><span>overall (projected)</span><b>74% → <em style="color:var(--ok)">82%</em></b></div></div>
      <div class="muted" style="font-size:12px;margin-top:10px">5 fixed · 0 regressions in 20 re-checked</div>` },
  { k: "share", t: "Share the evidence", where: "run",
    why: "Download a report for the people who sign off: verdicts, scores, fixes and an appendix of real requests and responses.",
    pts: ["<b>PDF report</b> in the app's look, or a <b>print-friendly</b> light version",
      "Appendix with the first 50 evaluations per candidate and the judge's reasoning",
      "Every number traces back to a real request you can open in the app"],
    decide: "Who needs to approve the migration?", go: "Results → Download PDF report",
    pv: `<div class="g-cap">Report</div><div class="g-pages"><i></i><i></i><i class="light"></i></div>` },
];

const GUIDE_CHECKLIST = [
  { t: "Upload your logs", d: "Drag in a LiteLLM log export (or enter an S3 URI) and check the found / evaluable / dropped counts.", where: "new" },
  { t: "Pick a slice", d: "Choose the teams or users to migrate first, and set Limit records to about 200 for a fast first pass.", where: "new" },
  { t: "Choose candidates", d: "Keep the recommended models, pick the endpoint, and add judge steering if exact facts matter.", where: "new" },
  { t: "Launch and read the results", d: "Compare Migration Confidence, the Side-by-side table and the Dimension breakdown; open a few evaluations.", where: "run" },
  { t: "Fix, re-test, share", d: "Open See remediations, re-test a fix with a regression check, then download the PDF.", where: "run" },
];

const GUIDE_TERMS = [
  ["Migration Confidence", "0–100. How much of this workload the candidate handles as well as your current model. Drives the verdict."],
  ["Verdict", "Safe drop-in · Drop-in with prompt tuning · Needs work · Not a drop-in."],
  ["Evaluation", "One logged request, replayed to a candidate and compared with the original answer."],
  ["Labels", "Compatible · Compatible w/ drift (same meaning, different style) · Partial · Incompatible."],
  ["Hard break", "A broken format, a different fact, or a mismatched tool call. The answer is Incompatible no matter how good the rest is."],
  ["Judge steering", "Your own instructions to the AI judge — e.g. \"amounts must match exactly; tone doesn't matter\"."],
];

const GUIDE_TIPS = [
  ["Start small", "Run 200 records first. It takes minutes and costs cents; scale up once the setup looks right."],
  ["Steer the judge", "If exact amounts, IDs or dates matter, say so in judge steering — it changes what counts as a failure."],
  ["Keep a baseline", "Include a cross-family model (e.g. Claude) to see whether a problem is the candidate or the prompt."],
  ["Read the failures", "Sort out errors first (errors only), then open a few Incompatible evaluations before trusting the score."],
  ["Re-check widely", "Before rolling out a fix, re-test with \"all\" Compatible evaluations in the regression check."],
  ["Tune in Settings", "Judge models, prices, thresholds and your LiteLLM proxy address live under Settings."],
];

const GUIDE_DONE_KEY = "modelshift.guide.checklist";

function guideChecklistState() {
  try { return new Set(JSON.parse(localStorage.getItem(GUIDE_DONE_KEY) || "[]")); } catch (e) { return new Set(); }
}

async function guideLatestDoneRun() {
  try {
    const rs = await api("/runs");
    const list = Array.isArray(rs) ? rs : (rs.runs || []);
    const done = list.filter(r => r.status === "done" && !String(r.run_id).startsWith("run_uidemo"));
    return (done[done.length - 1] || null);
  } catch (e) { return null; }
}

function guideSvg(path) {
  const s = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  s.setAttribute("viewBox", "0 0 24 24"); s.setAttribute("aria-hidden", "true");
  // Static author-defined path markup — parse in the SVG namespace and adopt the
  // nodes instead of assigning innerHTML.
  const doc = new DOMParser().parseFromString(
    `<svg xmlns="http://www.w3.org/2000/svg">${path}</svg>`, "image/svg+xml");
  [...doc.documentElement.childNodes].forEach((n) => s.append(s.ownerDocument.importNode(n, true)));
  return s;
}

async function renderGuide(main) {
  main.append(topbar("Guide"));
  const root = h("div", { class: "guide" });
  main.append(root);
  const alive = () => document.body.contains(root);
  const latest = await guideLatestDoneRun();
  const goTo = (where) => {
    if (where === "new") return go("new");
    if (where === "run" && latest) return go("run", latest.run_id);
    return go("runs");
  };
  const goLabel = (where) => (where === "run" ? (latest ? "opens your latest finished run" : "opens your runs") : "");

  // ---- hero
  const playBtn = h("button", { class: "btn" }, "▶ Play the 2-minute tour");
  root.append(h("section", { class: "g-hero" },
    h("div", { class: "g-tag mono" }, "Getting started"),
    h("h1", {}, "Find out which Bedrock model can replace your GPT model — ", h("em", {}, "before"), " you switch."),
    h("p", { class: "g-lede" }, "ModelShift replays your real production traffic against newer models, scores every answer against what "
      + "your current model said, and tells you which candidate is a safe drop-in, what breaks, and how to fix it."),
    h("div", { class: "row", style: "gap:10px" }, playBtn,
      h("button", { class: "btn secondary", onclick: () => go("new") }, "Start a new run"),
      latest ? h("button", { class: "btn ghost", onclick: () => go("run", latest.run_id) }, "Open my latest run") : null)));

  // ---- pipeline
  const track = h("div", { class: "g-track", role: "tablist", "aria-label": "Steps of a run" });
  const packet = h("span", { class: "g-packet", "aria-hidden": "true" });
  const line = h("div", { class: "g-line", "aria-hidden": "true" }, h("span", { class: "g-line-fill" }));
  const detail = h("div", { class: "g-detail", role: "tabpanel", "aria-live": "polite" });
  const dots = h("div", { class: "g-dots", "aria-hidden": "true" });
  const prevBtn = h("button", { class: "btn ghost" }, "← Back");
  const nextBtn = h("button", { class: "btn" }, "Next →");
  const nodes = GUIDE_STEPS.map((s, i) => {
    const b = h("button", { class: "g-node", role: "tab", id: `g-step-${i}`, "aria-label": `Step ${i + 1}: ${s.t}`, tabindex: "-1" },
      h("span", { class: "g-num" }, String(i + 1)), h("span", { class: "g-dot" }, guideSvg(GUIDE_ICON[s.k])), h("span", { class: "g-lbl" }, s.t));
    b.addEventListener("click", () => { stopTour(); show(i, true); });
    track.append(b);
    dots.append(h("i"));
    return b;
  });
  track.append(line, packet);
  track.addEventListener("keydown", (e) => {
    const k = { ArrowRight: 1, ArrowDown: 1, ArrowLeft: -1, ArrowUp: -1 }[e.key];
    if (k) { e.preventDefault(); stopTour(); show(cur + k, true); }
    if (e.key === "Home") { e.preventDefault(); stopTour(); show(0, true); }
    if (e.key === "End") { e.preventDefault(); stopTour(); show(GUIDE_STEPS.length - 1, true); }
  });
  root.append(h("section", { class: "card g-pipe" },
    h("div", { class: "row spread", style: "margin-bottom:14px" }, h("h3", { style: "margin:0" }, "How a run works"),
      h("span", { class: "muted", style: "font-size:12px" }, "Click a step, or use the arrow keys")),
    track, detail,
    h("div", { class: "row spread", style: "margin-top:16px" }, dots, h("div", { class: "row", style: "gap:8px" }, prevBtn, nextBtn))));

  let cur = 0, tour = null;
  function placePacket() {
    const n = nodes[cur], tr = track.getBoundingClientRect(), r = n.querySelector(".g-dot").getBoundingClientRect();
    packet.style.left = (r.left - tr.left + r.width / 2 - 5) + "px";
    const first = nodes[0].querySelector(".g-dot").getBoundingClientRect();
    const last = nodes[nodes.length - 1].querySelector(".g-dot").getBoundingClientRect();
    line.style.left = (first.left - tr.left + first.width / 2) + "px";
    line.style.width = (last.left - first.left) + "px";
    line.firstChild.style.width = (100 * cur / (GUIDE_STEPS.length - 1)) + "%";
  }
  function show(i, focus) {
    cur = (i + GUIDE_STEPS.length) % GUIDE_STEPS.length;
    const s = GUIDE_STEPS[cur];
    nodes.forEach((n, j) => {
      n.classList.toggle("active", j === cur); n.classList.toggle("done", j < cur);
      n.setAttribute("aria-selected", j === cur ? "true" : "false"); n.tabIndex = j === cur ? 0 : -1;
    });
    [...dots.children].forEach((d, j) => d.classList.toggle("on", j === cur));
    detail.setAttribute("aria-labelledby", `g-step-${cur}`);
    clear(detail);
    const pts = h("ul", { class: "g-pts" });
    s.pts.forEach(p => { const li = h("li"); li.append(frag(p)); pts.append(li); });  // static authored copy
    const pv = h("div", { class: "g-preview" }); pv.append(frag(s.pv));               // static authored copy
    const hint = goLabel(s.where);
    detail.append(
      h("div", { class: "g-anim" },
        h("div", { class: "muted mono", style: "font-size:11px" }, `STEP ${cur + 1} OF ${GUIDE_STEPS.length}`),
        h("h2", {}, s.t), h("p", { class: "g-why" }, s.why), pts,
        h("div", { class: "g-decide" }, "You decide: ", h("b", {}, s.decide)),
        h("button", { class: "g-goto", onclick: () => goTo(s.where) }, "Take me there →",
          h("span", { class: "muted mono" }, s.go + (hint ? ` · ${hint}` : "")))),
      h("div", { class: "g-anim", style: "animation-delay:.06s" }, pv));
    placePacket();
    requestAnimationFrame(placePacket);  // again after layout settles (fonts, scrollbars)
    if (focus) nodes[cur].focus({ preventScroll: true });
  }
  function stopTour() { if (tour) { clearInterval(tour); tour = null; } playBtn.textContent = "▶ Play the 2-minute tour"; }
  playBtn.addEventListener("click", () => {
    if (tour) return stopTour();
    show(0); playBtn.textContent = "❚❚ Pause the tour";
    root.querySelector(".g-pipe").scrollIntoView({ behavior: "smooth", block: "start" });
    tour = setInterval(() => {
      if (!alive()) return stopTour();
      if (cur === GUIDE_STEPS.length - 1) stopTour(); else show(cur + 1);
    }, 7000);
  });
  prevBtn.addEventListener("click", () => { stopTour(); show(cur - 1); });
  nextBtn.addEventListener("click", () => { stopTour(); show(cur + 1); });
  const onResize = () => { if (!alive()) return window.removeEventListener("resize", onResize); placePacket(); };
  window.addEventListener("resize", onResize);

  // ---- start here (checklist) + key terms
  const done = guideChecklistState();
  const meter = h("span");
  const sub = h("div", { class: "muted", style: "font-size:12px;margin-bottom:10px" });
  const save = () => { try { localStorage.setItem(GUIDE_DONE_KEY, JSON.stringify([...done])); } catch (e) { /* private mode */ } };
  const updateMeter = () => {
    meter.style.width = (100 * done.size / GUIDE_CHECKLIST.length) + "%";
    sub.textContent = done.size === GUIDE_CHECKLIST.length ? "All done — you've been through the full loop."
      : `${done.size} of ${GUIDE_CHECKLIST.length} done · tick each step as you finish it (saved in this browser).`;
  };
  const items = GUIDE_CHECKLIST.map((c, i) => {
    const box = h("input", { type: "checkbox", id: `g-chk-${i}`, "aria-describedby": `g-chk-d-${i}` });
    box.checked = done.has(i);
    const row = h("div", { class: "g-item" + (box.checked ? " done" : "") },
      box,
      h("div", { style: "flex:1" },
        h("label", { for: `g-chk-${i}` }, `${i + 1}. ${c.t}`),
        h("div", { class: "muted", id: `g-chk-d-${i}`, style: "font-size:12px" }, c.d)),
      h("button", { class: "btn ghost g-small", onclick: () => goTo(c.where), "aria-label": `Go to: ${c.t}` }, "Go"));
    box.addEventListener("change", () => {
      box.checked ? done.add(i) : done.delete(i);
      row.classList.toggle("done", box.checked); save(); updateMeter();
    });
    return row;
  });
  updateMeter();
  const terms = h("dl", { class: "g-terms" });
  GUIDE_TERMS.forEach(([k, v]) => terms.append(h("dt", {}, k), h("dd", {}, v)));
  root.append(h("div", { class: "grid cols-2 g-two" },
    h("section", { class: "card" }, h("h3", {}, "Start here: your first run"), sub,
      h("div", { class: "g-meter" }, meter), ...items),
    h("section", { class: "card" }, h("h3", {}, "Key terms"), terms)));

  // ---- tips
  root.append(h("section", { style: "margin-top:16px" },
    h("h3", {}, "Get the most out of ModelShift"),
    h("div", { class: "grid cols-3" }, ...GUIDE_TIPS.map(([t, d]) => h("div", { class: "card g-tip" }, h("b", {}, t), h("span", {}, d))))));

  show(0);
}
