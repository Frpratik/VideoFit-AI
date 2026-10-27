"use strict";
/* VideoFit AI front end. Plain JavaScript, no build step.
   Every piece of dynamic text is inserted with textContent, never innerHTML. */

const $ = (id) => document.getElementById(id);
const state = { session: null, videoId: null, duration: 0, hasTimestamps: false, report: null, loadedKey: "", busy: false };

/* ---------- tiny DOM helper ---------- */
function h(tag, props = {}, ...kids) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(props)) {
    if (v === undefined || v === null || v === false) continue;
    if (k === "class") node.className = v;
    else if (k === "text") node.textContent = v;
    else if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
    else node.setAttribute(k, v === true ? "" : v);
  }
  for (const kid of kids.flat()) if (kid !== null && kid !== undefined) node.append(kid);
  return node;
}
const fmt = (sec) => {
  sec = Math.max(0, Math.floor(sec || 0));
  const hh = Math.floor(sec / 3600), mm = Math.floor((sec % 3600) / 60), ss = sec % 60;
  return (hh ? `${hh}:${String(mm).padStart(2, "0")}` : `${mm}`) + `:${String(ss).padStart(2, "0")}`;
};
const safeYouTube = (url) => typeof url === "string" && url.startsWith("https://www.youtube.com/");
const VIDEO_ID = /^[A-Za-z0-9_-]{11}$/;

function toast(message) {
  const t = $("toast");
  t.textContent = message; t.hidden = false;
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => (t.hidden = true), 6000);
}

async function api(path, body) {
  let res;
  try {
    res = await fetch(path, body ? { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) } : undefined);
  } catch (e) { throw new Error("Cannot reach the backend. Is the server running?"); }
  let data = null;
  try { data = await res.json(); } catch (e) { /* non-JSON error */ }
  if (!res.ok) {
    const detail = data && data.detail;
    throw new Error(typeof detail === "string" ? detail : Array.isArray(detail) ? "Please check your input." : `Request failed (${res.status}).`);
  }
  return data;
}

/* ---------- navigation ---------- */
function showView(name) {
  for (const v of ["load", "dash", "chat"]) $("view-" + v).hidden = v !== name;
  document.querySelectorAll(".tab").forEach((b) => b.classList.toggle("active", b.dataset.view === name));
  if (name === "chat") $("question").focus();
}
document.querySelectorAll(".tab").forEach((b) => b.addEventListener("click", () => showView(b.dataset.view)));
function enableTabs() { document.querySelectorAll(".tab").forEach((b) => (b.disabled = false)); }

/* ---------- health ---------- */
(async function health() {
  try {
    const d = await api("/api/health");
    $("health").replaceChildren(h("i", { class: "dot " + (d.llm_key_set ? "ok" : "bad") }),
      h("span", { class: "txt", text: d.llm_key_set ? `${d.llm_model}` : "GROQ_API_KEY missing" }));
  } catch (e) {
    $("health").replaceChildren(h("i", { class: "dot bad" }), h("span", { class: "txt", text: "backend offline" }));
  }
})();

/* ---------- load form ---------- */
$("demo-btn").addEventListener("click", async () => {
  try {
    const d = await api("/api/sample");
    $("url").value = d.url; $("transcript").value = d.transcript;
    $("goal").value ||= "Learn neural networks from scratch as a beginner";
    $("demo-note").textContent = d.note; $("demo-note").hidden = false;
  } catch (e) { toast(e.message); }
});

function setStatus(text, { error = false, working = false } = {}) {
  const s = $("load-status");
  s.className = "status" + (error ? " err" : "");
  s.replaceChildren(text);
  if (working) s.append(h("div", { class: "bar" }));
}
function setBusy(busy) {
  state.busy = busy;
  $("analyze-btn").disabled = busy; $("chat-only-btn").disabled = busy;
}

async function ensureLoaded() {
  const url = $("url").value.trim(), transcript = $("transcript").value;
  if (!url) throw new Error("Please enter the YouTube URL.");
  if (!transcript.trim()) throw new Error("Please paste the transcript.");
  const key = url + "\n" + transcript;
  if (state.session && key === state.loadedKey) return null;
  const d = await api("/api/video", { url, transcript });
  Object.assign(state, { session: d.session_id, videoId: d.video_id, duration: d.duration_sec, hasTimestamps: d.has_timestamps, loadedKey: key, report: null });
  $("thread").replaceChildren(); renderSuggestions();
  setPlayer(null);
  enableTabs();
  return d;
}

function loadSummary(d) {
  const scripts = Object.entries(d.scripts).map(([k, v]) => `${k} ${v}`).join(", ");
  const bits = [`${d.segments} segments`, d.format.replace("_", " "), d.has_timestamps ? `${fmt(d.duration_sec)} long` : "no timestamps", scripts];
  return bits.join(" · ") + (d.warnings.length ? "  ⚠ " + d.warnings.join(" ") : "");
}

$("chat-only-btn").addEventListener("click", async () => {
  setBusy(true); setStatus("Indexing the transcript…", { working: true });
  try {
    const d = await ensureLoaded();
    setStatus(d ? "Loaded: " + loadSummary(d) : "Already loaded.");
    $("chat-language").value = $("language").value;
    showView("chat");
  } catch (e) { setStatus(e.message, { error: true }); }
  setBusy(false);
});

$("load-form").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const goal = $("goal").value.trim();
  if (!goal) return setStatus("Please describe what you want to learn.", { error: true });
  setBusy(true);
  const started = Date.now();
  const tick = setInterval(() => setStatus(`Analyzing… ${Math.round((Date.now() - started) / 1000)}s. This usually takes 30–90 seconds (about a dozen model calls).`, { working: true }), 1000);
  setStatus("Indexing the transcript…", { working: true });
  try {
    const d = await ensureLoaded();
    const topics = $("topics").value.split("\n").map((t) => t.trim()).filter(Boolean);
    const res = await api("/api/analyze", {
      session_id: state.session, goal, level: $("level").value, language: $("language").value,
      topics, comments: $("comments").value,
    });
    clearInterval(tick);
    state.report = res.report; state.duration = res.duration_sec;
    $("chat-language").value = $("language").value;
    renderDashboard(res.report);
    renderSuggestions();
    setStatus(d ? "Done. " + loadSummary(d) : "Done.");
    showView("dash");
  } catch (e) { setStatus(e.message, { error: true }); }
  clearInterval(tick);
  setBusy(false);
});

/* ---------- dashboard ---------- */
const COVER_LABEL = { covered: "Covered", partial: "Partial", not_detected: "Not detected" };
const COVER_COLOR = { covered: "var(--covered)", partial: "var(--partial)", not_detected: "var(--missing)" };
const DEPTH_PIPS = { none: 0, mention: 1, explained: 2, in_depth: 3 };

function timecode(c) {
  if (!c.timestamp) return h("span", { class: "tc", text: "no time" });
  const attrs = safeYouTube(c.url) ? { href: c.url, target: "_blank", rel: "noopener noreferrer", title: c.quote } : { title: c.quote };
  return h(safeYouTube(c.url) ? "a" : "span", { class: "tc", ...attrs, text: "▶ " + c.timestamp });
}

function ring(score) {
  const R = 80, C = 2 * Math.PI * R;
  const ns = "http://www.w3.org/2000/svg";
  const svg = document.createElementNS(ns, "svg");
  svg.setAttribute("width", 190); svg.setAttribute("height", 190); svg.setAttribute("viewBox", "0 0 190 190");
  for (const [stroke, frac] of [["#262626", 1], ["#ff0000", score / 100]]) {
    const c = document.createElementNS(ns, "circle");
    for (const [k, v] of Object.entries({ cx: 95, cy: 95, r: R, fill: "none", stroke, "stroke-width": 12, "stroke-linecap": "round", "stroke-dasharray": `${C * frac} ${C}` })) c.setAttribute(k, v);
    svg.append(c);
  }
  return h("div", { class: "ring", role: "img", "aria-label": `Overall learning suitability score ${score} out of 100` },
    svg, h("div", { class: "num" }, h("b", { text: Math.round(score) }), h("span", { text: "out of 100" })));
}

function reel(topics, duration) {
  if (!state.hasTimestamps || !duration) return h("p", { class: "hint", text: "This transcript has no timestamps, so there is no timeline." });
  const strip = h("div", { class: "reel" });
  topics.forEach((t, i) => t.citations.forEach((c) => {
    if (c.start_sec === null || c.start_sec === undefined) return;
    strip.append(h("button", {
      class: "mark", type: "button", style: `left:${Math.min(99.5, (c.start_sec / duration) * 100)}%;background:${COVER_COLOR[t.coverage]}`,
      title: `${t.topic.name} · ${c.timestamp}`, "aria-label": `${t.topic.name} at ${c.timestamp}`,
      onclick: () => { const el = $("topic-" + i); el.scrollIntoView({ behavior: "smooth", block: "center" }); el.classList.remove("flash"); void el.offsetWidth; el.classList.add("flash"); },
    }));
  }));
  return h("div", {}, strip,
    h("div", { class: "axis" }, h("span", { text: "0:00" }), h("span", { text: fmt(duration / 2) }), h("span", { text: fmt(duration) })),
    h("div", { class: "legend" }, ...["covered", "partial"].map((k) => h("span", {}, h("i", { style: `background:${COVER_COLOR[k]}` }), COVER_LABEL[k] + " evidence"))));
}

function topicCard(t, i) {
  const pips = h("span", { class: "pips", title: "Explanation depth: " + t.depth }, ...[1, 2, 3].map((n) => h("i", { class: n <= DEPTH_PIPS[t.depth] ? "on" : "" })));
  const status = t.assessed ? t.coverage : "not_detected";
  return h("article", { class: "topic " + status, id: "topic-" + i },
    h("header", {}, h("h3", { text: t.topic.name }), h("span", { class: "chip " + status, text: t.assessed ? COVER_LABEL[t.coverage] : "Not assessed",
      title: status === "not_detected" ? "Not found in the supplied transcript. It may still be taught elsewhere in the video." : "" })),
    h("div", { class: "tmeta" }, h("span", { class: "chip " + t.topic.importance, text: t.topic.importance }), h("span", {}, "Depth", pips),
      t.has_example ? h("span", { text: "● has example" }) : null),
    h("p", { class: "reason", text: t.reason }),
    h("div", {}, t.citations.map(timecode)));
}

function list(items, empty) {
  return items.length ? h("ul", {}, items.map((x) => h("li", { text: x }))) : h("p", { class: "hint", text: empty });
}

function feedbackGroup(title, items) {
  if (!items.length) return null;
  return h("div", {}, h("h3", { text: title }), h("ul", {}, items.map((it) =>
    h("li", {}, it.text, it.example_comment ? h("blockquote", { text: it.example_comment }) : null))));
}

function renderDashboard(r) {
  const root = $("dash");
  const counts = { covered: 0, partial: 0, not_detected: 0 };
  r.topics.forEach((t) => t.assessed && counts[t.coverage]++);
  root.replaceChildren(...[
    r.warnings.length ? h("div", { class: "banner", role: "status" }, r.warnings.join(" ")) : null,
    h("section", { class: "panel hero" }, ring(r.overall_score),
      h("dl", { class: "meta" },
        h("dt", { text: "Your goal" }), h("dd", { text: r.goal }),
        h("dt", { text: "Level" }), h("dd", { text: r.level }),
        h("dt", { text: "Topics" }), h("dd", { text: `${counts.covered} covered · ${counts.partial} partial · ${counts.not_detected} not detected` }),
        h("dt", { text: "About this score" }), h("dd", { class: "hint", text: "A weighted heuristic (40/25/15/10/10), not a validated measurement. Read the evidence below." }))),
    h("section", { class: "section panel" }, h("h2", { text: "Score breakdown" }),
      r.criteria.map((c) => h("div", { class: "crit" },
        h("div", { class: "name" }, c.label, h("small", { text: `weight ${Math.round(c.weight * 100)}%` })),
        h("div", { class: "track", role: "progressbar", "aria-valuenow": Math.round(c.score), "aria-valuemin": 0, "aria-valuemax": 100 }, h("div", { class: "fill", style: `width:${c.score}%` })),
        h("div", { class: "val", text: Math.round(c.score) }),
        h("div", { class: "why", text: c.explanation })))),
    h("section", { class: "section panel" }, h("h2", {}, "Evidence timeline ", h("small", { text: "where the video covers your topics" })), reel(r.topics, state.duration)),
    h("section", { class: "section" },
      h("h2", {}, "Topic coverage ",
        h("button", { type: "button", class: "link", title: "Auto-generated checklists differ a little from run to run. Reusing this list makes the next score comparable.",
          text: "Reuse this checklist", onclick: reuseChecklist })),
      h("div", { class: "topics" }, r.topics.map(topicCard))),
    h("section", { class: "section three" },
      h("div", { class: "panel" }, h("h2", { text: "Strengths" }), list(r.strengths, "None reported.")),
      h("div", { class: "panel" }, h("h2", { text: "Limitations" }), list(r.limitations, "None reported.")),
      h("div", { class: "panel" }, h("h2", { text: "Recommended next topics" }), list(r.recommended_next_topics, "None reported."))),
    r.comment_insights ? h("section", { class: "section panel" },
      h("h2", {}, "Audience comments ", h("small", { text: `${r.comment_insights.comments_analyzed} comments · not part of the score` })),
      h("div", { class: "feedback" },
        feedbackGroup("Recurring questions", r.comment_insights.recurring_questions),
        feedbackGroup("Praise", r.comment_insights.praise),
        feedbackGroup("Difficulties", r.comment_insights.difficulties),
        feedbackGroup("Complaints", r.comment_insights.complaints))) : null,
    h("section", { class: "section panel" }, h("h2", { text: "How to read this report" }), h("ul", { class: "caveats" }, r.caveats.map((c) => h("li", { text: c }))))].filter(Boolean));
  $("dash-empty").hidden = true; root.hidden = false;
}

function reuseChecklist() {
  if (!state.report) return;
  $("topics").value = state.report.topics.map((t) => t.topic.name).join("\n");
  $("topics").closest("details").open = true;
  showView("load");
  toast("Checklist copied to the Load page. Press Analyze video to re-run with the same topics.");
}

/* ---------- chat ---------- */
const PLAYER_PLACEHOLDER = $("player").cloneNode(true).childNodes;
function setPlayer(startSec) {
  const box = $("player");
  if (startSec === null || startSec === undefined || !VIDEO_ID.test(state.videoId || "")) {
    box.replaceChildren(...[...PLAYER_PLACEHOLDER].map((n) => n.cloneNode(true)));
    return;
  }
  const src = `https://www.youtube.com/embed/${state.videoId}?start=${Math.floor(startSec)}&autoplay=1&rel=0`;
  box.replaceChildren(h("iframe", { src, title: "YouTube video player", allow: "autoplay; encrypted-media; picture-in-picture", allowfullscreen: true, referrerpolicy: "strict-origin-when-cross-origin" }));
}

function renderSuggestions() {
  const topics = state.report ? state.report.topics.slice(0, 3).map((t) => `Where did the instructor discuss ${t.topic.name}?`) : [];
  const base = state.report ? ["Which important topics are missing?"] : ["What is the main idea of this video?", "Does the instructor give any examples?"];
  $("suggest").replaceChildren(...[...topics, ...base].map((q) => h("button", { type: "button", text: q, onclick: () => ask(q) })));
}

function citationRow(c) {
  return h("div", { class: "cite" },
    c.timestamp && state.hasTimestamps ? h("button", { type: "button", class: "tc", text: "▶ " + c.timestamp, onclick: () => setPlayer(c.start_sec) }) : h("span", { class: "tc", text: "no time" }),
    h("div", {}, h("q", { text: c.quote }), safeYouTube(c.url) ? h("a", { class: "ext", href: c.url, target: "_blank", rel: "noopener noreferrer", text: "↗", title: "Open on YouTube" }) : null));
}

function botMessage(res) {
  const a = res.answer;
  return h("div", { class: "msg bot" },
    h("div", { class: "label", text: a.evidence_found ? "From the video" : "Not found in the transcript" }),
    ...a.answer_from_transcript.split(/\n+/).map((p) => h("p", { text: p })),
    a.general_knowledge_note ? h("div", { class: "outside" }, h("div", { class: "label out", text: "Outside the video (general knowledge)" }), a.general_knowledge_note) : null,
    a.citations.length ? h("div", { class: "cites" }, a.citations.map(citationRow)) : null,
    ...a.warnings.map((w) => h("div", { class: "warn", text: "⚠ " + w })),
    res.understood_as ? h("div", { class: "understood", text: "Searched as: " + res.understood_as }) : null);
}

async function ask(question) {
  question = (question || "").trim();
  if (!question || state.busy || !state.session) return;
  state.busy = true; $("send-btn").disabled = true; $("question").value = "";
  const thread = $("thread");
  thread.append(h("div", { class: "msg user", text: question }));
  const typing = h("div", { class: "msg bot", "aria-label": "Thinking" }, h("div", { class: "dots" }, h("span"), h("span"), h("span")));
  thread.append(typing); thread.scrollTop = thread.scrollHeight;
  try {
    const res = await api("/api/chat", { session_id: state.session, question, language: $("chat-language").value });
    typing.replaceWith(botMessage(res));
  } catch (e) { typing.remove(); toast(e.message); }
  thread.scrollTop = thread.scrollHeight;
  state.busy = false; $("send-btn").disabled = false; $("question").focus();
}
$("chat-form").addEventListener("submit", (ev) => { ev.preventDefault(); ask($("question").value); });
$("reset-btn").addEventListener("click", async () => {
  if (!state.session) return;
  try { await api("/api/chat/reset", { session_id: state.session }); $("thread").replaceChildren(); } catch (e) { toast(e.message); }
});
