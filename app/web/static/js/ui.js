/** Fact knowledge layer UI.
 *
 * The conflicts view is the important one: it puts two facts side by side with
 * the verdict, the rule that produced it, and the evidence each side rests on.
 */
import { api } from "./api.js";
const $ = (selector) => document.querySelector(selector);
const escapeHtml = (value) => String(value ?? "").replace(/[&<>"']/g, (ch) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[ch]);
const VERDICT_LABEL = {
    corroborates: "corroborates",
    contradicts: "contradicts",
    reconciled: "reconciled",
};
function period(fact) {
    if (fact.period_start && fact.period_end) {
        return `${fact.period_start} → ${fact.period_end}`;
    }
    return fact.period_raw || "period unstated";
}
function qualifiers(fact) {
    const parts = [fact.basis, fact.variant, fact.modality !== "actual" ? fact.modality : null]
        .filter(Boolean)
        .map((q) => `<span class="qual">${escapeHtml(q)}</span>`);
    return parts.length ? parts.join("") : '<span class="qual muted">unqualified</span>';
}
function factCard(fact) {
    if (!fact)
        return '<div class="fact-card muted">missing</div>';
    return `
    <div class="fact-card">
      <div class="fact-value">${escapeHtml(fact.value_raw)}
        <span class="unit">${escapeHtml(fact.unit_raw ?? "")}</span></div>
      <div class="fact-metric">${escapeHtml(fact.metric ?? "")}</div>
      <div class="fact-quals">${qualifiers(fact)}</div>
      <div class="fact-period">${escapeHtml(period(fact))}</div>
      <blockquote class="evidence">${escapeHtml(fact.quote)}</blockquote>
      <div class="fact-source">
        ${escapeHtml(fact.publisher || fact.doc_title || fact.filename)}
        · p${fact.page_no}
        <a href="/api/documents/${fact.doc_id}/pdf#page=${fact.page_no}"
           target="_blank" rel="noopener">open pdf</a>
      </div>
    </div>`;
}
function relationCard(relation) {
    const inferred = relation.qualifier_inferred
        ? '<span class="badge-inferred" title="a qualifier was missing and treated as a match">inferred qualifier</span>'
        : "";
    return `
    <article class="relation ${relation.verdict}">
      <header>
        <span class="verdict ${relation.verdict}">${VERDICT_LABEL[relation.verdict]}</span>
        <code class="rule">${escapeHtml(relation.rule)}</code>
        ${inferred}
      </header>
      <p class="explanation">${escapeHtml(relation.explanation)}</p>
      <div class="pair">${factCard(relation.a)}${factCard(relation.b)}</div>
    </article>`;
}
// --- views -----------------------------------------------------------------
async function renderOverview() {
    const [stats, documents] = await Promise.all([api.stats(), api.documents()]);
    const v = stats.verdicts;
    $("#overview").innerHTML = `
    <div class="tiles">
      ${tile("documents", stats.documents)}
      ${tile("pages", stats.pages)}
      ${tile("facts", stats.facts)}
      ${tile("relations", stats.relations)}
      ${tile("corroborates", v.corroborates ?? 0, "corroborates")}
      ${tile("contradicts", v.contradicts ?? 0, "contradicts")}
      ${tile("reconciled", v.reconciled ?? 0, "reconciled")}
      ${tile("grounding rejected", `${(stats.grounding_rejection_rate * 100).toFixed(1)}%`)}
    </div>
    <h3>Documents</h3>
    <table class="grid">
      <thead><tr><th>Title</th><th>Publisher</th><th>Type</th><th>Published</th>
        <th class="num">Pages</th><th class="num">Facts</th></tr></thead>
      <tbody>${documents
        .map((d) => `<tr>
            <td>${escapeHtml(d.title ?? d.filename)}</td>
            <td>${escapeHtml(d.publisher ?? "")}</td>
            <td>${escapeHtml(d.doc_type ?? "")}</td>
            <td>${escapeHtml(d.published_date ?? "")}</td>
            <td class="num">${d.n_pages}</td>
            <td class="num">${d.n_facts}</td></tr>`)
        .join("")}</tbody>
    </table>
    ${stats.failures.length
        ? `<h3>Rejections</h3>
           <p class="muted">Every rejected candidate is recorded rather than silently dropped.</p>
           <table class="grid"><thead><tr><th>Stage</th><th>Reason</th><th class="num">Count</th></tr></thead>
           <tbody>${stats.failures
            .map((f) => `<tr><td>${escapeHtml(f.stage)}</td><td>${escapeHtml(f.reason)}</td><td class="num">${f.n}</td></tr>`)
            .join("")}</tbody></table>`
        : ""}`;
}
const tile = (label, value, cls = "") => `
  <div class="tile ${cls}"><div class="tile-value">${value}</div>
  <div class="tile-label">${escapeHtml(label)}</div></div>`;
async function renderConflicts(verdict) {
    const relations = await api.conflicts(verdict);
    $("#conflicts-list").innerHTML = relations.length
        ? relations.map(relationCard).join("")
        : '<p class="muted">No relations yet. Ingest a document first.</p>';
}
async function renderFacts(query = "") {
    const { total, facts } = await api.facts(query ? { q: query } : {});
    $("#facts-count").textContent = `${facts.length} of ${total}`;
    $("#facts-list").innerHTML = facts.length
        ? `<table class="grid">
        <thead><tr><th>Metric</th><th class="num">Value</th><th>Qualifiers</th>
          <th>Period</th><th>Source</th></tr></thead>
        <tbody>${facts
            .map((f) => `<tr>
              <td>${escapeHtml(f.metric ?? "")}</td>
              <td class="num mono">${escapeHtml(f.value_raw)}
                <span class="unit">${escapeHtml(f.unit_raw ?? "")}</span></td>
              <td>${qualifiers(f)}</td>
              <td class="mono small">${escapeHtml(period(f))}</td>
              <td class="small">${escapeHtml(f.publisher || f.filename)} p${f.page_no}</td>
            </tr>`)
            .join("")}</tbody></table>`
        : '<p class="muted">No facts match.</p>';
}
async function renderClusters() {
    const clusters = await api.clusters();
    $("#clusters-list").innerHTML = clusters.length
        ? `<table class="grid">
        <thead><tr><th>Entity</th><th>Metric</th><th>Period</th>
          <th class="num">Facts</th><th class="num">Documents</th></tr></thead>
        <tbody>${clusters
            .map((c) => `<tr>
              <td>${escapeHtml(c.entity ?? "")}</td>
              <td>${escapeHtml(c.metric ?? "")}</td>
              <td class="mono small">${escapeHtml(c.period_start ?? "")} → ${escapeHtml(c.period_end ?? "")}</td>
              <td class="num">${c.n_facts}</td>
              <td class="num">${c.n_docs}</td></tr>`)
            .join("")}</tbody></table>`
        : '<p class="muted">No clusters with more than one fact yet.</p>';
}
// --- upload ----------------------------------------------------------------
async function handleUpload(file) {
    const status = $("#upload-status");
    status.textContent = `uploading ${file.name}…`;
    try {
        const job = await api.upload(file);
        await pollJob(job.id, status);
    }
    catch (error) {
        status.textContent = `failed: ${error.message}`;
    }
}
async function pollJob(id, status) {
    for (;;) {
        const job = await api.job(id);
        if (job.status === "done") {
            const r = job.report;
            status.textContent =
                `done — ${r.n_stored} facts stored, ${r.n_relations} relations, ` +
                    `${((Number(r.grounding_rejection_rate) || 0) * 100).toFixed(1)}% grounding rejected`;
            await Promise.all([renderOverview(), renderConflicts(), renderFacts(), renderClusters()]);
            return;
        }
        if (job.status === "failed") {
            status.textContent = `failed: ${job.detail}`;
            return;
        }
        status.textContent = `${job.status}… (extraction is rate limited, this takes a few minutes)`;
        await new Promise((resolve) => setTimeout(resolve, 2000));
    }
}
// --- wiring ----------------------------------------------------------------
function showTab(name) {
    document.querySelectorAll(".panel").forEach((panel) => {
        panel.hidden = panel.dataset.panel !== name;
    });
    document.querySelectorAll(".tab").forEach((tab) => {
        tab.classList.toggle("active", tab.dataset.tab === name);
    });
}
function init() {
    document.querySelectorAll(".tab").forEach((tab) => {
        tab.addEventListener("click", () => showTab(tab.dataset.tab));
    });
    $("#file-input").addEventListener("change", (event) => {
        const input = event.target;
        if (input.files?.[0])
            void handleUpload(input.files[0]);
    });
    $("#verdict-filter").addEventListener("change", (event) => {
        void renderConflicts(event.target.value || undefined);
    });
    let debounce;
    $("#fact-search").addEventListener("input", (event) => {
        window.clearTimeout(debounce);
        const value = event.target.value;
        debounce = window.setTimeout(() => void renderFacts(value), 250);
    });
    showTab("overview");
    void renderOverview();
    void renderConflicts();
    void renderFacts();
    void renderClusters();
}
document.addEventListener("DOMContentLoaded", init);
