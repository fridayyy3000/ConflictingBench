(function () {
  "use strict";

  const STORAGE_KEY = "docuinsight_corpus";
  const ALLOWED_EXTENSIONS = [".pdf", ".docx", ".txt", ".md"];
  const PDFJS_URL = "https://cdnjs.cloudflare.com/ajax/libs/pdf.js/4.10.38/pdf.min.mjs";
  const PDFJS_WORKER_URL = "https://cdnjs.cloudflare.com/ajax/libs/pdf.js/4.10.38/pdf.worker.min.mjs";
  const DECISION_LABELS = {
    answered: "Answered",
    answered_with_dissent: "Answered · dissent flagged",
    unresolved_conflict: "Unresolved conflict",
    needs_information: "Needs information",
    insufficient_evidence: "Insufficient evidence",
  };
  const RULE_LABELS = {
    governance: "Governance",
    corroboration: "Corroboration",
    consistency: "Consistency",
    tie: "Tie · not resolved",
    single_source: "Single source",
    none: "No commitment",
  };
  const contextFields = {
    relevant_date: "context-date",
    location: "context-location",
    employee_category: "context-category",
    circumstances: "context-circumstances",
  };

  const el = {};
  [
    "error-banner", "new-collection-btn", "pipeline-chip", "create-collection-card", "create-collection-form",
    "collection-name-input", "workspace", "collection-name-display", "collection-meta",
    "document-count", "dropzone", "browse-btn", "browse-folder-btn", "file-input", "folder-input", "load-example-btn",
    "folder-source-form", "folder-path-input", "folder-source-btn", "folder-source-hint",
    "upload-status", "documents-empty", "document-tree-wrap", "document-tree", "question-input", "resolve-btn",
    "resolve-hint", "submitted-context", "results", "decision-badge", "export-btn", "result-answer",
    "result-confidence", "short-answers", "result-explanation", "missing-information", "missing-list",
    "clarification-question", "resolution-panel", "resolution-rule", "resolution-summary", "corroboration-strip",
    "citation-list", "result-footer", "governing-evidence", "competing-evidence",
    "no-governing-evidence", "no-competing-evidence", "technical-evidence",
    "loading-overlay", "loading-message", "reader", "reader-slots", "reader-close", "compare-btn",
  ].forEach((id) => {
    el[id.replace(/-([a-z])/g, (_, letter) => letter.toUpperCase())] = document.getElementById(id);
  });
  const workspaceGrid = document.querySelector(".workspace-grid");
  const slotTemplate = document.getElementById("reader-slot-template");

  let state = { corpusId: null, corpusName: null, expiresAt: null, documents: [] };
  let options = { status: [], docType: [] };
  let lastResult = null;
  let lastSubmittedContext = null;
  let reader = { primary: null, secondary: null, split: false };
  const documentTextCache = new Map();
  let pdfjsPromise = null;

  // ---------------------------------------------------------------- helpers
  function showError(message) {
    el.errorBanner.textContent = message;
    el.errorBanner.classList.remove("hidden");
    window.clearTimeout(showError.timer);
    showError.timer = window.setTimeout(() => el.errorBanner.classList.add("hidden"), 8000);
  }

  function showLoading(message) {
    el.loadingMessage.textContent = message || "Working…";
    el.loadingOverlay.classList.remove("hidden");
  }

  function hideLoading() {
    el.loadingOverlay.classList.add("hidden");
  }

  async function apiFetch(path, options) {
    let response;
    try {
      response = await fetch(path, options);
    } catch (error) {
      throw new Error("Could not reach the server. Check your connection and try again.");
    }
    const contentType = response.headers.get("content-type") || "";
    if (!response.ok) {
      let detail = null;
      try { detail = (await response.json()).detail; } catch (error) { detail = null; }
      throw new Error(typeof detail === "string" ? detail : `Request failed (HTTP ${response.status})`);
    }
    if (contentType.includes("application/json")) return response.json();
    return response;
  }

  function corpusPath(suffix) {
    return `/demo/corpora/${encodeURIComponent(state.corpusId)}${suffix || ""}`;
  }

  function persistState() {
    sessionStorage.setItem(STORAGE_KEY, JSON.stringify({
      corpusId: state.corpusId, corpusName: state.corpusName, expiresAt: state.expiresAt,
    }));
  }

  function clearState() {
    state = { corpusId: null, corpusName: null, expiresAt: null, documents: [] };
    sessionStorage.removeItem(STORAGE_KEY);
    documentTextCache.clear();
  }

  function labelize(value) {
    return String(value || "unknown").replace(/_/g, " ");
  }

  function formatScore(value) {
    return typeof value === "number" ? value.toFixed(3) : "n/a";
  }

  function effectiveStatus(doc) {
    return doc.status_override || doc.inferred_status || "unknown";
  }

  function effectiveType(doc) {
    return doc.doc_type_override || doc.inferred_doc_type || "unknown";
  }

  // Mirror of pipeline/evidence_locator.py: normalize text with an offset map so
  // a verified quote can be located inside text the server did not see (PDF
  // text layers, Word HTML).
  const UNIFY = { "‘": "'", "’": "'", "‚": "'", "‛": "'", "“": '"', "”": '"', "„": '"', "‟": '"', "–": "-", "—": "-", "−": "-", " ": " " };
  function normalizeWithMap(text) {
    const out = [];
    const offsets = [];
    let pendingSpace = false;
    for (let index = 0; index < text.length; index += 1) {
      let char = UNIFY[text[index]] || text[index];
      if (/\s/.test(char)) { pendingSpace = out.length > 0; continue; }
      if (pendingSpace) { out.push(" "); offsets.push(index); pendingSpace = false; }
      const folded = char.toLowerCase();
      for (const piece of folded) { out.push(piece); offsets.push(index); }
    }
    return { text: out.join(""), offsets };
  }

  function locateQuote(quote, text) {
    if (!quote || !text) return [];
    const { text: normalized, offsets } = normalizeWithMap(text);
    const segments = quote.split(/…|\.\.\./).map((part) => normalizeWithMap(part).text.replace(/^[\s"'.,;:]+|[\s"'.,;:]+$/g, "")).filter(Boolean);
    const spans = [];
    let position = 0;
    for (const segment of segments) {
      let start = normalized.indexOf(segment, position);
      if (start < 0) start = normalized.indexOf(segment);
      if (start < 0) return [];
      const end = start + segment.length;
      spans.push([offsets[start], offsets[end - 1] + 1]);
      position = end;
    }
    return spans;
  }

  // ---------------------------------------------------------- pipeline chip
  async function loadPipelineInfo() {
    try {
      const health = await apiFetch("/health");
      options = { status: health.status_options || [], docType: health.doc_type_options || [] };
      const p = health.pipeline || {};
      const model = (p.generation_model || "").split("/").pop();
      el.pipelineChip.textContent = `${model} · ${p.retrieval === "hybrid" ? "BM25 + dense" : "dense"} retrieval · ${p.policy === "answer_and_flag" ? "answer and flag" : "refuse on conflict"}`;
      el.pipelineChip.title = `Provider ${p.provider}; embeddings ${p.embedding_model}`;
      const connector = health.folder_connector || {};
      el.folderSourceForm.classList.toggle("hidden", connector.enabled === false);
      if (connector.enabled !== false) {
        const roots = connector.roots || [];
        el.folderSourceHint.textContent = roots.length
          ? `PDF, Word, text and Markdown files under ${roots.join(", ")}. Hidden files and links are skipped.`
          : "PDF, Word, text and Markdown files. Hidden files and links are skipped.";
      }
    } catch (error) {
      el.pipelineChip.textContent = "";
    }
  }

  // ---------------------------------------------------------- collection UI
  function renderCollection() {
    const hasCollection = Boolean(state.corpusId);
    el.createCollectionCard.classList.toggle("hidden", hasCollection);
    el.workspace.classList.toggle("hidden", !hasCollection);
    el.newCollectionBtn.classList.toggle("hidden", !hasCollection);
    if (!hasCollection) return;

    el.collectionNameDisplay.textContent = state.corpusName || "Collection";
    el.collectionMeta.textContent = state.expiresAt
      ? `Temporary collection · expires ${new Date(state.expiresAt).toLocaleString()}`
      : "Collection";
    el.documentCount.textContent = `${state.documents.length} ${state.documents.length === 1 ? "document" : "documents"}`;
    renderDocumentTree();
    el.documentsEmpty.classList.toggle("hidden", state.documents.length > 0);
    el.documentTreeWrap.classList.toggle("hidden", state.documents.length === 0);
    updateResolveAvailability();
  }

  function renderDocumentTree() {
    el.documentTree.replaceChildren();
    const groups = new Map();
    state.documents.forEach((doc) => {
      const path = doc.relative_path || doc.filename || doc.original_filename || "Document";
      const parts = path.split("/");
      const folder = parts.length > 1 ? parts.slice(0, -1).join("/") : "";
      if (!groups.has(folder)) groups.set(folder, []);
      groups.get(folder).push({ doc, name: parts[parts.length - 1] });
    });
    const folders = Array.from(groups.keys()).sort((a, b) => (a === "" ? -1 : b === "" ? 1 : a.localeCompare(b)));
    folders.forEach((folder) => {
      if (folder) {
        const heading = document.createElement("p");
        heading.className = "tree-folder";
        heading.textContent = folder;
        el.documentTree.appendChild(heading);
      }
      groups.get(folder).sort((a, b) => a.name.localeCompare(b.name)).forEach(({ doc, name }) => {
        el.documentTree.appendChild(renderTreeRow(doc, name, Boolean(folder)));
      });
    });
  }

  function makeSelect(values, current, inferred, title) {
    const select = document.createElement("select");
    select.className = "tree-select";
    select.title = title;
    const inferredOption = document.createElement("option");
    inferredOption.value = "";
    inferredOption.textContent = `Inferred: ${labelize(inferred)}`;
    select.appendChild(inferredOption);
    const list = values.length ? values : [inferred];
    list.forEach((value) => {
      const option = document.createElement("option");
      option.value = value;
      option.textContent = labelize(value);
      select.appendChild(option);
    });
    select.value = current || "";
    select.classList.toggle("is-override", Boolean(current));
    return select;
  }

  function renderTreeRow(doc, name, nested) {
    const row = document.createElement("div");
    row.className = `tree-row${nested ? " tree-row--nested" : ""}`;
    row.dataset.documentId = doc.document_id;

    const nameCell = document.createElement("div");
    nameCell.className = "tree-row__name";
    const type = document.createElement("span");
    type.className = "type-tag";
    type.textContent = (doc.file_type || "").replace(".", "") || "doc";
    const label = document.createElement("strong");
    label.textContent = name;
    const badge = document.createElement("span");
    const status = effectiveStatus(doc);
    badge.className = `status-badge status-badge--${status}`;
    badge.textContent = labelize(status) + (doc.status_override ? " · set" : "");
    nameCell.append(type, label, badge);

    const controls = document.createElement("div");
    controls.className = "tree-row__controls";
    const statusSelect = makeSelect(options.status, doc.status_override, doc.inferred_status, "Document status");
    statusSelect.addEventListener("change", () => updateOverride(doc, { status_override: statusSelect.value || null }));
    const typeSelect = makeSelect(options.docType, doc.doc_type_override, doc.inferred_doc_type, "Document type");
    typeSelect.addEventListener("change", () => updateOverride(doc, { doc_type_override: typeSelect.value || null }));
    const remove = document.createElement("button");
    remove.type = "button";
    remove.className = "tree-row__delete";
    remove.title = "Remove document";
    remove.setAttribute("aria-label", `Remove ${name}`);
    remove.textContent = "×";
    remove.addEventListener("click", () => removeDocument(doc));
    controls.append(statusSelect, typeSelect, remove);
    row.append(nameCell, controls);
    return row;
  }

  async function updateOverride(doc, fields) {
    try {
      const updated = await apiFetch(corpusPath(`/documents/${encodeURIComponent(doc.document_id)}`), {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(fields),
      });
      Object.assign(doc, updated);
      renderCollection();
      const what = "status_override" in fields ? "Status" : "Type";
      showUploadStatus(`${what} updated for ${updated.relative_path}. Ask again to see the effect.`, "success");
      markResultStale("Document metadata changed since this result.");
    } catch (error) {
      showError(error.message);
      renderCollection();
    }
  }

  async function removeDocument(doc) {
    if (!window.confirm(`Remove ${doc.relative_path || doc.filename} from this collection?`)) return;
    try {
      await apiFetch(corpusPath(`/documents/${encodeURIComponent(doc.document_id)}`), { method: "DELETE" });
      state.documents = state.documents.filter((d) => d.document_id !== doc.document_id);
      documentTextCache.delete(doc.document_id);
      renderCollection();
      markResultStale("A document was removed since this result.");
    } catch (error) {
      showError(error.message);
    }
  }

  function markResultStale(message) {
    if (!lastResult) return;
    el.submittedContext.textContent = `${message} Rerun to review the updated case.`;
    el.submittedContext.classList.remove("hidden");
    el.submittedContext.classList.add("is-stale");
  }

  function updateResolveAvailability() {
    const hasDocuments = state.documents.length > 0;
    el.resolveBtn.disabled = !hasDocuments;
    el.resolveHint.classList.toggle("hidden", hasDocuments);
  }

  function readContext() {
    const context = {};
    Object.entries(contextFields).forEach(([key, id]) => {
      const value = document.getElementById(id).value.trim();
      if (value) context[key] = value;
    });
    return context;
  }

  function contextSummary(context) {
    const labels = { relevant_date: "Date", location: "Location", employee_category: "Employee category", circumstances: "Circumstances" };
    const parts = Object.entries(context || {}).map(([key, value]) => `${labels[key] || key}: ${value}`);
    return parts.length ? parts.join(" · ") : "No optional case context";
  }

  function updateContextFreshness() {
    if (!lastSubmittedContext) return;
    const changed = JSON.stringify(readContext()) !== JSON.stringify(lastSubmittedContext);
    if (el.submittedContext.classList.contains("is-stale") && !changed) return;
    el.submittedContext.classList.toggle("is-stale", changed);
    el.submittedContext.textContent = changed
      ? `Case context changed since this result. Rerun to review the updated case. Current: ${contextSummary(readContext())}`
      : `Submitted case: ${contextSummary(lastSubmittedContext)}`;
  }

  // -------------------------------------------------------------- uploads
  function showUploadStatus(message, kind) {
    const status = el.uploadStatus;
    status.textContent = message;
    status.classList.remove("hidden", "upload-status--success", "upload-status--error");
    status.classList.add(kind === "error" ? "upload-status--error" : "upload-status--success");
  }

  async function uploadFiles(fileList, pathsByFile) {
    const files = Array.from(fileList || []);
    if (!files.length) return;
    const accepted = [];
    const rejected = [];
    files.forEach((file) => {
      const supported = ALLOWED_EXTENSIONS.some((extension) => file.name.toLowerCase().endsWith(extension));
      (supported ? accepted : rejected).push(file);
    });
    if (rejected.length) showUploadStatus(`Skipped ${rejected.length} unsupported file(s): ${rejected.slice(0, 5).map((file) => file.name).join(", ")}${rejected.length > 5 ? "…" : ""}`, "error");
    if (!accepted.length) return;

    const formData = new FormData();
    accepted.forEach((file) => {
      formData.append("files", file, file.name);
      const relative = (pathsByFile && pathsByFile.get(file)) || file.webkitRelativePath || file.name;
      formData.append("paths", relative);
    });
    showLoading(`Uploading ${accepted.length} document${accepted.length === 1 ? "" : "s"}…`);
    try {
      const response = await apiFetch(corpusPath("/documents"), { method: "POST", body: formData });
      (response.uploaded || []).forEach((doc) => state.documents.push(doc));
      renderCollection();
      showUploadStatus(`Uploaded ${response.uploaded.length} file(s). Total: ${response.document_count}.`, "success");
    } catch (error) {
      showUploadStatus(error.message, "error");
    } finally {
      hideLoading();
    }
  }

  async function filesFromDrop(dataTransfer) {
    // Folder drops expose directory entries; walk them so relative paths survive.
    const items = Array.from(dataTransfer.items || []);
    const entries = items.map((item) => (item.webkitGetAsEntry ? item.webkitGetAsEntry() : null)).filter(Boolean);
    if (!entries.length || !entries.some((entry) => entry.isDirectory)) {
      return { files: Array.from(dataTransfer.files || []), paths: null };
    }
    const files = [];
    const paths = new Map();
    async function walk(entry, prefix) {
      if (entry.isFile) {
        const file = await new Promise((resolve, reject) => entry.file(resolve, reject));
        files.push(file);
        paths.set(file, `${prefix}${entry.name}`);
        return;
      }
      const dirReader = entry.createReader();
      let batch;
      do {
        batch = await new Promise((resolve, reject) => dirReader.readEntries(resolve, reject));
        for (const child of batch) await walk(child, `${prefix}${entry.name}/`);
      } while (batch.length);
    }
    for (const entry of entries) await walk(entry, "");
    return { files, paths };
  }

  async function indexServerFolder(event) {
    event.preventDefault();
    const path = el.folderPathInput.value.trim();
    if (!path) { el.folderPathInput.focus(); return; }
    showLoading(`Indexing ${path}…`);
    el.folderSourceBtn.disabled = true;
    try {
      const response = await apiFetch(corpusPath("/sources/folder"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ path }),
      });
      (response.uploaded || []).forEach((doc) => state.documents.push(doc));
      renderCollection();
      const skipped = response.skipped || [];
      const label = response.source && response.source.label ? response.source.label : path;
      let message = `Indexed ${response.uploaded.length} file(s) from ${label}. Total: ${response.document_count}.`;
      if (skipped.length) {
        message += ` Skipped ${skipped.length}: ${skipped.slice(0, 3).map((item) => `${item.path} (${item.reason})`).join("; ")}${skipped.length > 3 ? "…" : ""}`;
      }
      showUploadStatus(message, skipped.length ? "error" : "success");
      el.folderPathInput.value = "";
    } catch (error) {
      showUploadStatus(error.message, "error");
    } finally {
      el.folderSourceBtn.disabled = false;
      hideLoading();
    }
  }

  async function loadExampleDataset() {
    showLoading("Loading the example documents…");
    try {
      const response = await apiFetch(corpusPath("/load_example"), { method: "POST" });
      (response.uploaded || []).forEach((doc) => state.documents.push(doc));
      renderCollection();
      showUploadStatus(`Loaded ${response.uploaded.length} example document(s).`, "success");
    } catch (error) {
      showUploadStatus(error.message, "error");
    } finally {
      hideLoading();
    }
  }

  // ---------------------------------------------------------------- result
  function formatLocation(source) {
    if (source.page_start !== null && source.page_start !== undefined) {
      const pages = source.page_end && source.page_end !== source.page_start
        ? `pp. ${source.page_start}–${source.page_end}` : `p. ${source.page_start}`;
      return source.section ? `${pages} · ${source.section}` : pages;
    }
    return source.section ? source.section : `Chunk ${Number(source.chunk_index || 0) + 1}`;
  }

  function roleOf(citation) {
    if (citation && citation.role) return citation.role;
    if (!citation) return "context";
    return ["conflicting", "superseded", "different_scope"].includes(citation.treatment) ? "dissent" : (citation.treatment === "uncertain" ? "context" : "support");
  }

  function evidenceTarget(source, citation) {
    return {
      documentId: citation ? citation.document_id : source.document_id,
      filename: citation ? citation.filename : (source.original_filename || source.filename),
      chunkId: citation ? citation.chunk_id : source.chunk_id,
      quote: citation ? citation.quote : (source.supporting_sentence || null),
      documentSpans: citation ? citation.document_spans : [],
      passage: source.passage || (citation && citation.passage) || "",
      pageStart: (citation && citation.page_start) || source.page_start || null,
      section: (citation && citation.section) || source.section || null,
      role: roleOf(citation),
      status: source.status || null,
    };
  }

  function addEvidenceCard(container, source, citation, label) {
    const card = document.createElement("article");
    const role = roleOf(citation);
    card.className = `evidence-card evidence-card--${role}`;
    const treatment = citation ? citation.treatment : "retrieved";
    if (treatment === "uncertain") card.classList.add("evidence-card--uncertain");

    const head = document.createElement("div");
    head.className = "evidence-card__head";
    const filename = document.createElement("p");
    filename.className = "evidence-card__name";
    filename.textContent = source.original_filename || source.filename || (citation && citation.filename) || "Evidence";
    const pill = document.createElement("span");
    pill.className = `treatment-pill treatment-pill--${treatment.replace(/[^a-z_]/g, "")}`;
    pill.textContent = label || labelize(treatment);
    head.append(filename, pill);

    const location = document.createElement("p");
    location.className = "evidence-card__location";
    location.textContent = formatLocation(source);
    card.append(head, location);
    const excerptLabel = document.createElement("p");
    excerptLabel.className = "evidence-card__excerpt-label";
    excerptLabel.textContent = citation ? "Verified quote" : "Retrieved passage · not cited";
    const excerpt = document.createElement("blockquote");
    excerpt.className = "evidence-card__quote";
    const passage = citation ? citation.quote : (source.passage || "");
    excerpt.textContent = citation || passage.length <= 420 ? passage : `${passage.slice(0, 420).trimEnd()}…`;
    card.append(excerptLabel, excerpt);
    if (citation && citation.explanation) {
      const explanation = document.createElement("p");
      explanation.className = "evidence-card__explanation";
      explanation.textContent = citation.explanation;
      card.appendChild(explanation);
    }
    const actions = document.createElement("div");
    actions.className = "evidence-card__actions";
    const show = document.createElement("button");
    show.type = "button";
    show.className = "evidence-card__open";
    show.textContent = "Show in document";
    show.addEventListener("click", () => openInReader(evidenceTarget(source, citation), "text"));
    const original = document.createElement("button");
    original.type = "button";
    original.className = "evidence-card__open";
    original.textContent = "Open original";
    original.addEventListener("click", () => openInReader(evidenceTarget(source, citation), "original"));
    actions.append(show, original);
    card.appendChild(actions);
    container.appendChild(card);
  }

  function addCitationButton(citation, source) {
    const button = document.createElement("button");
    button.type = "button";
    const role = roleOf(citation);
    button.className = `citation-link citation-link--${role}`;
    button.dataset.chunkId = citation.chunk_id;
    button.textContent = `${citation.filename || source.filename} · ${formatLocation(source)}`;
    button.setAttribute("aria-label", `Show citation from ${citation.filename || source.filename} in the document`);
    button.addEventListener("click", () => openInReader(evidenceTarget(source, citation), "text"));
    el.citationList.appendChild(button);
  }

  function renderResolution(result) {
    const resolution = result.resolution;
    const show = Boolean(resolution && (resolution.corroboration.length || resolution.rule !== "none"));
    el.resolutionPanel.classList.toggle("hidden", !show);
    if (!show) return;
    el.resolutionRule.className = `rule-pill rule-pill--${resolution.rule}`;
    el.resolutionRule.textContent = RULE_LABELS[resolution.rule] || labelize(resolution.rule);
    el.resolutionSummary.textContent = resolution.summary || "";
    el.corroborationStrip.replaceChildren();
    resolution.corroboration.forEach((entry) => {
      const chip = document.createElement("button");
      chip.type = "button";
      chip.className = `corroboration-chip ${entry.committed ? "corroboration-chip--committed" : "corroboration-chip--dissent"}`;
      const value = document.createElement("strong");
      value.textContent = entry.value;
      const count = document.createElement("span");
      count.className = "corroboration-chip__count";
      count.textContent = `${entry.count} ${entry.count === 1 ? "document" : "documents"}`;
      const docs = document.createElement("span");
      docs.className = "corroboration-chip__docs";
      docs.textContent = entry.documents.map((name) => name.split("/").pop()).join(", ");
      chip.title = `${entry.committed ? "Committed value" : "Competing value"} · ${entry.documents.join(", ")}`;
      chip.append(value, count, docs);
      chip.addEventListener("click", () => {
        const citation = (result.citations || []).find((c) => entry.document_ids.includes(c.document_id));
        const source = citation ? sourceFor(result, citation) : null;
        if (citation) openInReader(evidenceTarget(source, citation), "text");
        else {
          const target = (result.top_sources || []).find((s) => entry.document_ids.includes(s.document_id));
          if (target) openInReader(evidenceTarget(target, null), "text");
        }
      });
      el.corroborationStrip.appendChild(chip);
    });
  }

  function sourceFor(result, citation) {
    return (result.top_sources || []).find((s) => s.chunk_id === citation.chunk_id) || {
      chunk_id: citation.chunk_id, document_id: citation.document_id, filename: citation.filename,
      original_filename: citation.filename, page_start: citation.page_start, page_end: citation.page_end,
      section: citation.section, passage: citation.passage,
    };
  }

  function renderTechnicalDetails(sources) {
    el.technicalEvidence.replaceChildren();
    sources.forEach((source) => {
      const row = document.createElement("div");
      row.className = "technical-row";
      [
        source.original_filename || source.filename,
        `Status: ${labelize(source.status)}`,
        `Type: ${labelize(source.document_type)}`,
        `Authority: ${formatScore(source.authority_score)}`,
        `Scope: ${formatScore(source.scope_score)}`,
        `Semantic: ${formatScore(source.semantic_score)}`,
        `Final rank: ${formatScore(source.final_score)}`,
      ].forEach((value, index) => {
        const cell = document.createElement(index === 0 ? "strong" : "span");
        cell.textContent = value;
        row.appendChild(cell);
      });
      el.technicalEvidence.appendChild(row);
    });
  }

  function renderResult(result, fixtureMode) {
    lastResult = result;
    el.results.classList.remove("hidden");
    const decision = result.decision_label || result.decision || (result.conflict_detected ? "unresolved_conflict" : "answered");
    el.decisionBadge.className = `decision-badge decision-badge--${decision}`;
    el.decisionBadge.textContent = DECISION_LABELS[decision] || labelize(decision);
    el.resultAnswer.textContent = result.answer || "The available evidence does not support a safe answer.";
    el.resultConfidence.textContent = result.confidence ? `Model-reported confidence: ${result.confidence}` : "";
    el.shortAnswers.replaceChildren();
    (result.short_answers || []).forEach((value) => {
      const chip = document.createElement("span");
      chip.className = "short-answer";
      chip.textContent = value;
      el.shortAnswers.appendChild(chip);
    });
    el.resultExplanation.textContent = result.explanation || result.reason || "";

    const missing = Array.isArray(result.missing_information) ? result.missing_information : [];
    el.missingInformation.classList.toggle("hidden", missing.length === 0 && !result.clarification_question);
    el.missingList.replaceChildren();
    missing.forEach((item) => {
      const li = document.createElement("li");
      li.textContent = item;
      el.missingList.appendChild(li);
    });
    el.clarificationQuestion.textContent = result.clarification_question || "";
    renderResolution(result);

    const sources = Array.isArray(result.top_sources) ? result.top_sources : [];
    const citations = Array.isArray(result.citations) ? result.citations : [];
    el.citationList.replaceChildren();
    el.governingEvidence.replaceChildren();
    el.competingEvidence.replaceChildren();

    const citedIds = new Set();
    citations.forEach((citation) => {
      const source = sourceFor(result, citation);
      citedIds.add(citation.chunk_id);
      addCitationButton(citation, source);
      if (roleOf(citation) === "support") addEvidenceCard(el.governingEvidence, source, citation);
      else addEvidenceCard(el.competingEvidence, source, citation);
    });
    sources.forEach((source) => {
      if (source.chunk_id && citedIds.has(source.chunk_id)) return;
      addEvidenceCard(el.competingEvidence, source, null, "retrieved context");
    });
    el.noGoverningEvidence.classList.toggle("hidden", el.governingEvidence.children.length > 0);
    el.noCompetingEvidence.classList.toggle("hidden", el.competingEvidence.children.length > 0);
    renderTechnicalDetails(sources);

    const p = result.pipeline || {};
    const footer = [];
    if (result.model) footer.push(`Model ${result.model}`);
    if (p.retrieval) footer.push(`${p.retrieval === "hybrid" ? "BM25 + dense" : "dense"} retrieval`);
    if (p.policy) footer.push(p.policy === "answer_and_flag" ? "answer-and-flag policy" : "refuse-on-conflict policy");
    if (typeof result.latency_s === "number") footer.push(`${result.latency_s.toFixed(1)} s`);
    el.resultFooter.textContent = footer.join(" · ");

    if (fixtureMode) {
      const tag = document.createElement("p");
      tag.className = "fixture-banner";
      tag.textContent = "LOCAL UI FIXTURE — illustrative state only; no API request was made.";
      el.results.prepend(tag);
    }

    // Open the reader on the committed answer's source, with the dissent beside it.
    const support = citations.find((c) => roleOf(c) === "support");
    const dissent = citations.find((c) => roleOf(c) === "dissent");
    if (support) {
      openInReader(evidenceTarget(sourceFor(result, support), support), "text", { slot: "primary", quiet: true });
      if (dissent) {
        reader.split = true;
        openInReader(evidenceTarget(sourceFor(result, dissent), dissent), "text", { slot: "secondary", quiet: true });
      }
    } else if (dissent) {
      openInReader(evidenceTarget(sourceFor(result, dissent), dissent), "text", { slot: "primary", quiet: true });
    }
    el.compareBtn.classList.toggle("hidden", !(support && dissent));
    updateCompareButton();
  }

  function exportRecord() {
    if (!lastResult) return;
    const record = {
      exported_at: new Date().toISOString(),
      collection: { id: state.corpusId, name: state.corpusName },
      question: lastResult.question,
      case_context: lastSubmittedContext || {},
      decision: lastResult.decision_label || lastResult.decision,
      answer: lastResult.answer,
      short_answers: lastResult.short_answers,
      confidence: lastResult.confidence,
      explanation: lastResult.explanation,
      resolution: lastResult.resolution,
      citations: (lastResult.citations || []).map((c) => ({
        document: c.filename, document_id: c.document_id, page: c.page_start, section: c.section,
        treatment: c.treatment, role: c.role, quote: c.quote, explanation: c.explanation,
      })),
      documents: state.documents.map((d) => ({
        path: d.relative_path, type: d.file_type, status: effectiveStatus(d), status_set_by_reviewer: Boolean(d.status_override),
        document_type: effectiveType(d),
      })),
      pipeline: lastResult.pipeline, model: lastResult.model, latency_s: lastResult.latency_s,
    };
    const blob = new Blob([JSON.stringify(record, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `docresolve-record-${new Date().toISOString().slice(0, 19).replace(/[:T]/g, "-")}.json`;
    document.body.appendChild(link);
    link.click();
    link.remove();
    window.setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  // ---------------------------------------------------------------- reader
  async function fetchDocumentText(documentId) {
    if (documentTextCache.has(documentId)) return documentTextCache.get(documentId);
    const data = await apiFetch(corpusPath(`/documents/${encodeURIComponent(documentId)}/text`));
    documentTextCache.set(documentId, data);
    return data;
  }

  function ensureSlot(name) {
    if (reader[name] && reader[name].root.isConnected) return reader[name];
    const fragment = slotTemplate.content.cloneNode(true);
    const root = fragment.querySelector(".reader-slot");
    const slot = {
      root,
      name: root.querySelector(".reader-slot__name"),
      meta: root.querySelector(".reader-slot__meta"),
      tabs: Array.from(root.querySelectorAll(".tab")),
      textView: root.querySelector('.reader-view[data-view="text"]'),
      originalView: root.querySelector('.reader-view[data-view="original"]'),
      originalContent: root.querySelector(".original-content"),
      pdfToolbar: root.querySelector(".pdf-toolbar"),
      pdfLabel: root.querySelector(".pdf-page-label"),
      pdfPrev: root.querySelector(".pdf-prev"),
      pdfNext: root.querySelector(".pdf-next"),
      pdfDownload: root.querySelector(".pdf-download"),
      target: null,
      view: "text",
      pdf: null,
    };
    slot.tabs.forEach((tab) => tab.addEventListener("click", () => switchView(slot, tab.dataset.view)));
    slot.pdfPrev.addEventListener("click", () => slot.pdf && renderPdfPage(slot, slot.pdf.page - 1));
    slot.pdfNext.addEventListener("click", () => slot.pdf && renderPdfPage(slot, slot.pdf.page + 1));
    reader[name] = slot;
    if (name === "primary") el.readerSlots.prepend(root);
    else el.readerSlots.appendChild(root);
    return slot;
  }

  function dropSlot(name) {
    if (reader[name]) {
      reader[name].root.remove();
      reader[name] = null;
    }
  }

  function showReader() {
    el.reader.classList.remove("hidden");
    workspaceGrid.classList.add("has-reader");
  }

  function hideReader() {
    el.reader.classList.add("hidden");
    workspaceGrid.classList.remove("has-reader");
    dropSlot("primary");
    dropSlot("secondary");
    reader.split = false;
    document.querySelectorAll(".citation-link.is-active").forEach((node) => node.classList.remove("is-active"));
  }

  function updateCompareButton() {
    el.readerSlots.classList.toggle("is-split", Boolean(reader.split && reader.secondary));
    el.compareBtn.textContent = reader.split ? "Single document" : "Compare side by side";
  }

  async function openInReader(target, view, settings) {
    settings = settings || {};
    showReader();
    let slotName = settings.slot;
    if (!slotName) {
      // Dissent goes to the second slot while comparing; everything else replaces the primary.
      slotName = reader.split && target.role === "dissent" ? "secondary" : "primary";
      if (reader.split && target.role !== "dissent" && reader.secondary && reader.secondary.target && reader.secondary.target.role !== "dissent") slotName = "primary";
    }
    const slot = ensureSlot(slotName);
    slot.target = target;
    slot.root.classList.toggle("reader-slot--dissent", target.role === "dissent");
    slot.name.textContent = target.filename || "Document";
    slot.meta.textContent = [target.pageStart ? `p. ${target.pageStart}` : null, target.section, target.status && target.status !== "unknown" ? `status: ${labelize(target.status)}` : null].filter(Boolean).join(" · ");
    document.querySelectorAll(".citation-link").forEach((node) => node.classList.toggle("is-active", node.dataset.chunkId === target.chunkId));
    updateCompareButton();
    await switchView(slot, view || "text");
    if (!settings.quiet) slot.root.scrollIntoView({ block: "nearest", behavior: "smooth" });
  }

  async function switchView(slot, view) {
    slot.view = view;
    slot.tabs.forEach((tab) => tab.classList.toggle("is-active", tab.dataset.view === view));
    slot.textView.classList.toggle("hidden", view !== "text");
    slot.originalView.classList.toggle("hidden", view !== "original");
    try {
      if (view === "text") await renderTextView(slot);
      else await renderOriginalView(slot);
    } catch (error) {
      const note = document.createElement("p");
      note.className = "reader-note";
      note.textContent = error.message;
      (view === "text" ? slot.textView : slot.originalContent).replaceChildren(note);
    }
  }

  function buildHighlightedText(text, spans, role, chunks) {
    // spans: [[start, end]] in document coordinates; chunks: [{start, chunk_index}]
    const boundaries = new Set([0, text.length]);
    const marks = spans.filter((s) => Array.isArray(s) && s[0] >= 0 && s[1] <= text.length && s[1] > s[0]);
    marks.forEach(([start, end]) => { boundaries.add(start); boundaries.add(end); });
    const chunkStarts = new Map();
    (chunks || []).forEach((chunk) => {
      if (typeof chunk.start === "number" && chunk.start > 0) { boundaries.add(chunk.start); chunkStarts.set(chunk.start, chunk); }
    });
    const points = Array.from(boundaries).sort((a, b) => a - b);
    const fragment = document.createDocumentFragment();
    let firstMark = null;
    for (let index = 0; index < points.length - 1; index += 1) {
      const start = points[index];
      const end = points[index + 1];
      if (chunkStarts.has(start)) {
        const rule = document.createElement("span");
        rule.className = "chunk-start";
        rule.dataset.label = `chunk ${chunkStarts.get(start).chunk_index + 1}`;
        fragment.appendChild(rule);
      }
      const piece = text.slice(start, end);
      const inMark = marks.some(([s, e]) => start >= s && end <= e);
      if (inMark) {
        const mark = document.createElement("mark");
        mark.className = `hl hl--${role}`;
        mark.textContent = piece;
        fragment.appendChild(mark);
        if (!firstMark) firstMark = mark;
      } else {
        fragment.appendChild(document.createTextNode(piece));
      }
    }
    return { fragment, firstMark };
  }

  async function renderTextView(slot) {
    const target = slot.target;
    slot.textView.replaceChildren(Object.assign(document.createElement("p"), { className: "reader-note", textContent: "Loading document…" }));
    let data = null;
    try {
      data = await fetchDocumentText(target.documentId);
    } catch (error) {
      data = null;
    }
    let text = data ? data.text : target.passage;
    let spans = (data && target.documentSpans && target.documentSpans.length) ? target.documentSpans : [];
    if (!spans.length && target.quote && text) spans = locateQuote(target.quote, text);
    if (!data) {
      const note = document.createElement("p");
      note.className = "reader-note";
      note.textContent = "Full document unavailable; showing the retrieved passage.";
      slot.textView.replaceChildren(note);
    } else {
      slot.textView.replaceChildren();
    }
    const { fragment, firstMark } = buildHighlightedText(text || "", spans, target.role, data ? data.chunks : []);
    slot.textView.appendChild(fragment);
    if (firstMark) firstMark.scrollIntoView({ block: "center" });
    else if (!spans.length && target.quote) {
      const note = document.createElement("p");
      note.className = "reader-note";
      note.textContent = "The cited sentence could not be located in the document text; it is shown in the evidence card.";
      slot.textView.prepend(note);
    }
  }

  function loadPdfJs() {
    if (!pdfjsPromise) {
      pdfjsPromise = import(PDFJS_URL).then((pdfjs) => {
        pdfjs.GlobalWorkerOptions.workerSrc = PDFJS_WORKER_URL;
        return pdfjs;
      });
    }
    return pdfjsPromise;
  }

  async function renderOriginalView(slot) {
    const target = slot.target;
    const doc = state.documents.find((d) => d.document_id === target.documentId);
    const fileType = doc ? doc.file_type : "";
    const url = corpusPath(`/documents/${encodeURIComponent(target.documentId)}/original`);
    slot.pdfToolbar.classList.add("hidden");
    slot.originalContent.replaceChildren(Object.assign(document.createElement("p"), { className: "reader-note", textContent: "Loading original…" }));
    if (!doc || doc.has_original === false) {
      throw new Error("The original file was not kept for this document.");
    }
    if (fileType === ".pdf") {
      const pdfjs = await loadPdfJs();
      const pdf = await pdfjs.getDocument({ url }).promise;
      slot.pdf = { pdf, pdfjs, page: Math.min(Math.max(target.pageStart || 1, 1), pdf.numPages) };
      slot.pdfToolbar.classList.remove("hidden");
      slot.pdfDownload.href = url;
      await renderPdfPage(slot, slot.pdf.page);
      return;
    }
    if (fileType === ".docx") {
      const response = await apiFetch(`${url}?format=html`);
      const html = await response.text();
      const parsed = new DOMParser().parseFromString(html, "text/html");
      parsed.querySelectorAll("script, iframe, object, embed, link, style").forEach((node) => node.remove());
      parsed.querySelectorAll("*").forEach((node) => Array.from(node.attributes).forEach((attr) => {
        if (/^on/i.test(attr.name) || (attr.name === "href" && /^\s*javascript:/i.test(attr.value))) node.removeAttribute(attr.name);
      }));
      const container = document.createElement("div");
      container.className = "docx-content";
      container.append(...Array.from(parsed.body.childNodes));
      slot.originalContent.replaceChildren(container);
      highlightInDom(container, target.quote, target.role);
      return;
    }
    const response = await apiFetch(url);
    const text = await response.text();
    const spans = target.quote ? locateQuote(target.quote, text) : [];
    const pre = document.createElement("div");
    pre.className = "reader-view--text";
    pre.appendChild(buildHighlightedText(text, spans, target.role, []).fragment);
    slot.originalContent.replaceChildren(pre);
    const mark = pre.querySelector("mark");
    if (mark) mark.scrollIntoView({ block: "center" });
  }

  function highlightInDom(container, quote, role) {
    if (!quote) return;
    const walker = document.createTreeWalker(container, NodeFilter.SHOW_TEXT);
    const nodes = [];
    let full = "";
    let node;
    while ((node = walker.nextNode())) {
      nodes.push({ node, start: full.length, end: full.length + node.nodeValue.length });
      full += node.nodeValue;
    }
    const spans = locateQuote(quote, full);
    if (!spans.length) return;
    let firstMark = null;
    spans.reverse().forEach(([start, end]) => {
      // Wrap each text node's intersection with the span.
      nodes.slice().reverse().forEach((entry) => {
        const s = Math.max(start, entry.start);
        const e = Math.min(end, entry.end);
        if (s >= e || !entry.node.isConnected) return;
        const range = document.createRange();
        range.setStart(entry.node, s - entry.start);
        range.setEnd(entry.node, e - entry.start);
        const mark = document.createElement("mark");
        mark.className = `hl hl--${role}`;
        range.surroundContents(mark);
        firstMark = mark;
      });
    });
    if (firstMark) firstMark.scrollIntoView({ block: "center" });
  }

  async function renderPdfPage(slot, pageNumber) {
    const { pdf, pdfjs } = slot.pdf;
    pageNumber = Math.min(Math.max(pageNumber, 1), pdf.numPages);
    slot.pdf.page = pageNumber;
    slot.pdfLabel.textContent = `Page ${pageNumber} of ${pdf.numPages}`;
    slot.pdfPrev.disabled = pageNumber <= 1;
    slot.pdfNext.disabled = pageNumber >= pdf.numPages;
    const page = await pdf.getPage(pageNumber);
    const width = Math.max(320, slot.originalContent.clientWidth - 4);
    const baseViewport = page.getViewport({ scale: 1 });
    const scale = width / baseViewport.width;
    const viewport = page.getViewport({ scale });
    const ratio = window.devicePixelRatio || 1;

    const wrapper = document.createElement("div");
    wrapper.className = "pdf-page";
    wrapper.style.width = `${viewport.width}px`;
    wrapper.style.height = `${viewport.height}px`;
    const canvas = document.createElement("canvas");
    canvas.width = Math.floor(viewport.width * ratio);
    canvas.height = Math.floor(viewport.height * ratio);
    canvas.style.width = `${viewport.width}px`;
    canvas.style.height = `${viewport.height}px`;
    const context = canvas.getContext("2d");
    context.scale(ratio, ratio);
    const textLayer = document.createElement("div");
    textLayer.className = "pdf-text-layer";
    wrapper.append(canvas, textLayer);
    slot.originalContent.replaceChildren(wrapper);

    await page.render({ canvasContext: context, viewport }).promise;
    const textContent = await page.getTextContent();
    const items = textContent.items.filter((item) => item.str !== undefined);
    // Position every text item, then mark the ones covered by the quote.
    let pageText = "";
    const ranges = [];
    items.forEach((item) => {
      const start = pageText.length;
      pageText += item.str + (item.hasEOL ? "\n" : " ");
      ranges.push([start, start + item.str.length]);
    });
    const spans = slot.target.quote ? locateQuote(slot.target.quote, pageText) : [];
    let firstHit = null;
    items.forEach((item, index) => {
      const transform = pdfjs.Util.transform(viewport.transform, item.transform);
      const fontHeight = Math.hypot(transform[2], transform[3]);
      const span = document.createElement("span");
      span.textContent = item.str;
      span.style.left = `${transform[4]}px`;
      span.style.top = `${transform[5] - fontHeight}px`;
      span.style.fontSize = `${fontHeight}px`;
      span.style.fontFamily = "sans-serif";
      if (item.width) {
        span.style.width = `${item.width * scale}px`;
      }
      const [itemStart, itemEnd] = ranges[index];
      const hit = spans.some(([s, e]) => itemStart < e && itemEnd > s);
      if (hit) {
        span.classList.add(`hl--${slot.target.role}`);
        if (!firstHit) firstHit = span;
      }
      textLayer.appendChild(span);
    });
    // Fit each span's rendered width to the PDF's measured width.
    Array.from(textLayer.children).forEach((span) => {
      const targetWidth = parseFloat(span.style.width);
      if (!targetWidth) return;
      const actual = span.getBoundingClientRect().width;
      if (actual > 0) span.style.transform = `scaleX(${targetWidth / actual})`;
    });
    if (firstHit) firstHit.scrollIntoView({ block: "center" });
    else if (slot.target.quote && !spans.length) {
      const note = document.createElement("p");
      note.className = "reader-note";
      note.textContent = `The cited sentence was not found on page ${pageNumber}; the citation points at this page.`;
      slot.originalContent.prepend(note);
    }
  }

  // ---------------------------------------------------------------- flows
  async function createCollection(name) {
    const created = await apiFetch("/demo/corpora", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ name }),
    });
    state = { corpusId: created.corpus_id, corpusName: created.name, expiresAt: created.expires_at, documents: [] };
    persistState();
    renderCollection();
  }

  async function askQuestion() {
    const question = el.questionInput.value.trim();
    if (!question) {
      showError("Enter a question before reviewing evidence.");
      el.questionInput.focus();
      return;
    }
    const context = readContext();
    lastSubmittedContext = context;
    el.submittedContext.textContent = `Submitted case: ${contextSummary(context)}`;
    el.submittedContext.classList.remove("hidden", "is-stale");
    showLoading("Retrieving evidence and reviewing the case…");
    el.resolveBtn.disabled = true;
    hideReader();
    try {
      const result = await apiFetch(corpusPath("/query"), {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question, context, top_k: 8 }),
      });
      renderResult(result, false);
      el.results.scrollIntoView({ block: "start", behavior: "smooth" });
    } catch (error) {
      showError(error.message);
    } finally {
      hideLoading();
      updateResolveAvailability();
      updateContextFreshness();
    }
  }

  async function startNewCollection() {
    if (state.corpusId) {
      try {
        await apiFetch(corpusPath(), { method: "DELETE" });
      } catch (error) {
        showError(`Could not delete the previous collection: ${error.message}`);
      }
    }
    clearState();
    lastResult = null;
    el.questionInput.value = "";
    Object.values(contextFields).forEach((id) => { document.getElementById(id).value = ""; });
    lastSubmittedContext = null;
    el.submittedContext.classList.add("hidden");
    el.uploadStatus.classList.add("hidden");
    el.results.classList.add("hidden");
    hideReader();
    renderCollection();
  }

  async function restoreSession() {
    const raw = sessionStorage.getItem(STORAGE_KEY);
    if (!raw) return;
    let saved;
    try { saved = JSON.parse(raw); } catch (error) { sessionStorage.removeItem(STORAGE_KEY); return; }
    if (!saved || !saved.corpusId) return;
    try {
      const [metadata, docsResponse] = await Promise.all([
        apiFetch(`/demo/corpora/${encodeURIComponent(saved.corpusId)}`),
        apiFetch(`/demo/corpora/${encodeURIComponent(saved.corpusId)}/documents`),
      ]);
      state = { corpusId: metadata.corpus_id, corpusName: metadata.name, expiresAt: metadata.expires_at, documents: docsResponse.documents || [] };
      renderCollection();
    } catch (error) {
      clearState();
    }
  }

  function renderLocalFixture(name) {
    const source = {
      chunk_id: "fixture:chunk:1", document_id: "fixture-a", original_filename: "LOCAL FIXTURE — Regional policy.md", filename: "LOCAL FIXTURE — Regional policy.md",
      passage: "Employees in Region North may claim 30 days, except during leave.", section: "Eligibility", status: "active",
      document_type: "policy", semantic_score: .81, authority_score: .5, scope_score: .73, final_score: .68,
    };
    const rival = { ...source, chunk_id: "fixture:chunk:2", document_id: "fixture-b", filename: "LOCAL FIXTURE — Employee handbook.md", original_filename: "LOCAL FIXTURE — Employee handbook.md", passage: "Employees in Region North may claim 45 days, except during leave.", status: "unknown", final_score: .66 };
    const cite = (s, treatment, role) => ({ chunk_id: s.chunk_id, document_id: s.document_id, filename: s.filename, quote: s.passage, passage: s.passage, treatment, role, explanation: "Fixture citation.", passage_spans: [[0, s.passage.length]], document_spans: [] });
    const fixtures = {
      answered: { decision: "answered", decision_label: "answered", answer: "The stated limit is 30 days, except during leave.", short_answers: ["30 days"], explanation: "The cited passage states the limit.", citations: [cite(source, "governing", "support")], top_sources: [source], confidence: "high", resolution: { rule: "single_source", summary: "Answered from one source.", committed_values: ["30 days"], corroboration: [] } },
      answered_with_dissent: { decision: "answered", decision_label: "answered_with_dissent", answer: "The limit is 30 days; the handbook says 45 days but is not the governing policy.", short_answers: ["30 days"], explanation: "The regional policy is active; the handbook disagrees.", citations: [cite(source, "governing", "support"), cite(rival, "conflicting", "dissent")], top_sources: [source, rival], confidence: "medium", resolution: { rule: "governance", summary: "An active document outranks the dissenting one.", committed_values: ["30 days"], corroboration: [{ value: "30 days", documents: [source.filename], document_ids: [source.document_id], count: 1, committed: true }, { value: "45 days", documents: [rival.filename], document_ids: [rival.document_id], count: 1, committed: false }] } },
      unresolved_conflict: { decision: "unresolved_conflict", decision_label: "unresolved_conflict", answer: "Two applicable passages disagree. The documents do not establish which governs.", short_answers: [], explanation: "No documented precedence resolves these statements.", citations: [cite(source, "conflicting", "dissent"), cite(rival, "conflicting", "dissent")], top_sources: [source, rival], confidence: "medium", resolution: { rule: "tie", summary: "Competing values are equally supported.", committed_values: [], corroboration: [] } },
      insufficient_evidence: { decision: "insufficient_evidence", decision_label: "insufficient_evidence", answer: "The collection does not contain enough information to answer this question.", short_answers: [], explanation: "No retrieved passage supports a conclusion.", citations: [], top_sources: [], confidence: "low", resolution: { rule: "none", summary: "No value was committed.", committed_values: [], corroboration: [] } },
    };
    if (!fixtures[name]) return false;
    document.title = `DocResolve — Local UI fixture: ${name}`;
    renderResult({ ...fixtures[name], pipeline: { retrieval: "hybrid", policy: "answer_and_flag" }, model: "fixture", latency_s: 0 }, true);
    return true;
  }

  // ---------------------------------------------------------------- wiring
  el.createCollectionForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    const name = el.collectionNameInput.value.trim();
    if (!name) return;
    showLoading("Creating collection…");
    try {
      await createCollection(name);
      el.collectionNameInput.value = "";
    } catch (error) {
      showError(error.message);
    } finally {
      hideLoading();
    }
  });
  el.browseBtn.addEventListener("click", () => el.fileInput.click());
  el.browseFolderBtn.addEventListener("click", () => el.folderInput.click());
  el.fileInput.addEventListener("change", (event) => { uploadFiles(event.target.files); event.target.value = ""; });
  el.folderInput.addEventListener("change", (event) => { uploadFiles(event.target.files); event.target.value = ""; });
  el.loadExampleBtn.addEventListener("click", loadExampleDataset);
  el.folderSourceForm.addEventListener("submit", indexServerFolder);
  ["dragover", "dragenter"].forEach((eventName) => el.dropzone.addEventListener(eventName, (event) => {
    event.preventDefault();
    el.dropzone.classList.add("dropzone--active");
  }));
  ["dragleave", "dragend"].forEach((eventName) => el.dropzone.addEventListener(eventName, () => el.dropzone.classList.remove("dropzone--active")));
  el.dropzone.addEventListener("drop", async (event) => {
    event.preventDefault();
    el.dropzone.classList.remove("dropzone--active");
    try {
      const { files, paths } = await filesFromDrop(event.dataTransfer);
      uploadFiles(files, paths);
    } catch (error) {
      showUploadStatus(`Could not read the dropped folder: ${error.message}`, "error");
    }
  });
  el.resolveBtn.addEventListener("click", askQuestion);
  el.questionInput.addEventListener("keydown", (event) => {
    if ((event.metaKey || event.ctrlKey) && event.key === "Enter" && !el.resolveBtn.disabled) askQuestion();
  });
  el.newCollectionBtn.addEventListener("click", startNewCollection);
  el.exportBtn.addEventListener("click", exportRecord);
  el.readerClose.addEventListener("click", hideReader);
  el.compareBtn.addEventListener("click", () => {
    reader.split = !reader.split;
    if (!reader.split) dropSlot("secondary");
    else if (lastResult) {
      const dissent = (lastResult.citations || []).find((c) => roleOf(c) === "dissent");
      if (dissent) openInReader(evidenceTarget(sourceFor(lastResult, dissent), dissent), "text", { slot: "secondary", quiet: true });
    }
    updateCompareButton();
  });
  Object.values(contextFields).forEach((id) => document.getElementById(id).addEventListener("input", updateContextFreshness));

  loadPipelineInfo();
  const fixture = new URLSearchParams(window.location.search).get("fixture");
  if (!renderLocalFixture(fixture)) restoreSession();
  else {
    el.createCollectionCard.classList.add("hidden");
    el.workspace.classList.remove("hidden");
    el.results.classList.remove("hidden");
  }
})();
