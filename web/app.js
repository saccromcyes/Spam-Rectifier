"use strict";
const $ = (s) => document.querySelector(s);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const pct = (x) => `${Math.round(x * 100)}%`;
const KIND = { naive_bayes: "Word counts, written from scratch", embeddings: "Sentence meaning + logistic regression", llm: "Zero-shot language model" };

function view(v) {
  $("#v-try").hidden = v !== "try";
  $("#v-benchmark").hidden = v !== "benchmark";
  document.querySelectorAll("nav a[data-v]").forEach((a) => a.toggleAttribute("aria-current", a.dataset.v === v));
  if (v === "benchmark") loadBench();
}
document.querySelectorAll("nav a[data-v]").forEach((a) => a.addEventListener("click", (e) => {
  e.preventDefault(); history.replaceState(null, "", a.dataset.v === "try" ? "#" : "#benchmark"); view(a.dataset.v);
}));

// Highlight each word by the log-odds Naive Bayes assigns it (spam positive, ham negative).
function highlight(text, evidence) {
  const weights = new Map(evidence.map((e) => [e.token, e.log_odds]));
  return text.split(/(\s+)/).map((part) => {
    const key = part.toLowerCase().replace(/^[^a-z0-9]+|[^a-z0-9]+$/g, "");
    const w = weights.get(key);
    if (w === undefined || Math.abs(w) < 0.7) return esc(part);
    const alpha = Math.min(0.9, Math.abs(w) / 5);
    const color = w > 0 ? `color-mix(in srgb, var(--spam) ${Math.round(alpha * 55)}%, transparent)` : `color-mix(in srgb, var(--ham) ${Math.round(alpha * 55)}%, transparent)`;
    return `<mark style="background:${color}" title="${w > 0 ? "Points to spam" : "Points to a normal message"} (${w.toFixed(1)})">${esc(part)}</mark>`;
  }).join("");
}

function card(r) {
  if (r.error) return `<article class="model"><div class="model-head"><span class="model-name">${esc(r.name)}</span></div><p class="why">${esc(r.error)}</p></article>`;
  let why = "";
  if (r.key === "naive_bayes") {
    const top = [...r.evidence].sort((a, b) => Math.abs(b.log_odds) - Math.abs(a.log_odds)).slice(0, 4);
    why = `<p class="why">Strongest words: ${top.map((t) => `“${esc(t.token)}” ${t.log_odds > 0 ? "→ spam" : "→ normal"}`).join(", ")}.</p>`;
  } else if (r.key === "embeddings") {
    why = `<p class="why">Most similar training messages:</p><ul class="neigh">${r.neighbours.map((n) => `<li class="${n.label}">${esc(n.text.slice(0, 110))}${n.text.length > 110 ? "…" : ""} <span class="muted">(${n.label})</span></li>`).join("")}</ul>`;
  } else if (r.key === "llm") {
    why = `<p class="why">${esc(r.reason)}</p>`;
  }
  return `<article class="model">
    <div class="model-head"><span class="model-name">${esc(r.name)}</span><span class="label ${r.label}">${r.label === "spam" ? "Spam" : "Not spam"}</span></div>
    <div class="model-kind">${esc(KIND[r.key] || "")}</div>
    <div class="prob" role="img" aria-label="Spam probability ${pct(r.spam_probability)}"><b style="width:${pct(r.spam_probability)}"></b></div>
    <div class="prob-cap"><span>Spam probability</span><strong>${pct(r.spam_probability)}</strong></div>
    ${why}
    <div class="lat">${r.latency_ms < 1 ? "under 1 ms" : r.latency_ms < 1000 ? `${Math.round(r.latency_ms)} ms` : `${(r.latency_ms / 1000).toFixed(1)} s`}${r.tokens ? `, ${r.tokens} tokens` : ""}</div>
  </article>`;
}

async function check(text) {
  $("#send").disabled = true; $("#error").hidden = true;
  try {
    const res = await fetch("/api/classify", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ text }) });
    const d = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(Array.isArray(d.detail) ? "Messages need between 1 and 2,000 characters." : d.detail || `Request failed (${res.status}).`);
    $("#out").hidden = false;
    const v = $("#verdict");
    v.className = `verdict ${d.consensus}`;
    v.innerHTML = `${d.consensus === "spam" ? "Likely spam" : "Looks like a normal message"}<small>${d.agreement} models agree.</small>`;
    $("#bubble").innerHTML = highlight(text, d.results.find((r) => r.key === "naive_bayes").evidence);
    $("#models").innerHTML = d.results.map(card).join("");
  } catch (e) {
    $("#error").textContent = e.message; $("#error").hidden = false;
  } finally { $("#send").disabled = false; }
}
$("#form").addEventListener("submit", (e) => { e.preventDefault(); check($("#msg").value.trim()); });
$("#samples").addEventListener("click", (e) => { const b = e.target.closest("button"); if (b) { $("#msg").value = b.textContent; check(b.textContent); } });

let benchLoaded = false;
async function loadBench() {
  if (benchLoaded) return;
  const el = $("#bench");
  const d = await (await fetch("/api/benchmark")).json().catch(() => ({ available: false }));
  if (!d.available) { el.innerHTML = `<p class="muted">The benchmark appears here after the first automated run.</p>`; return; }
  benchLoaded = true;
  const best = d.models.reduce((a, b) => (b.metrics.f1 > a.metrics.f1 ? b : a));
  el.innerHTML = `<p class="muted">Trained on ${d.train_size.toLocaleString()} messages, tested on ${d.test_size.toLocaleString()} (${d.test_spam} spam). ${esc(d.dataset)}. Run ${new Date(d.run_at).toLocaleDateString(undefined, { dateStyle: "medium" })}.</p>
  <div style="overflow-x:auto"><table><thead><tr><th>Model</th><th class="n">F1</th><th class="n">Precision</th><th class="n">Recall</th><th class="n">Real messages flagged</th><th class="n">Time per message</th></tr></thead><tbody>${
    d.models.map((m) => `<tr class="${m === best ? "best" : ""}"><td>${esc(m.label)}${m.metrics.n !== d.test_size ? ` <span class="muted">(${m.metrics.n} sampled)</span>` : ""}</td><td class="n">${m.metrics.f1.toFixed(3)}</td><td class="n">${m.metrics.precision.toFixed(3)}</td><td class="n">${m.metrics.recall.toFixed(3)}</td><td class="n">${(m.metrics.false_positive_rate * 100).toFixed(1)}%</td><td class="n">${m.latency_ms == null ? "—" : m.latency_ms < 1 ? "<1 ms" : `${Math.round(m.latency_ms)} ms`}${m.tokens_per_message ? `<br><span class="muted">${m.tokens_per_message} tokens</span>` : ""}</td></tr>`).join("")
  }</tbody></table></div>
  <h2>Where the models disagree</h2>
  <p class="muted">Test messages with split verdicts are the most instructive: each model's blind spot shows up here.</p>
  <ul class="dis">${d.disagreements.map((x) => `<li>${esc(x.text)}<div class="votes">Actually ${x.truth}. ${Object.entries(x.votes).map(([k, v]) => `${k.replace("_", " ")}: ${v}`).join(", ")}</div></li>`).join("")}</ul>`;
}
view(location.hash === "#benchmark" ? "benchmark" : "try");
