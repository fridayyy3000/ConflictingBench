(function () {
  "use strict";

  const STORAGE_KEY = "docuinsight_corpus";
  const ALLOWED_EXTENSIONS = [".pdf", ".docx", ".txt", ".md"];
  const contextFields = {
    relevant_date: "context-date",
    location: "context-location",
    employee_category: "context-category",
    circumstances: "context-circumstances",
  };
  const el = {};
  [
    "error-banner", "new-collection-btn", "create-collection-card", "create-collection-form",
    "collection-name-input", "workspace", "collection-name-display", "collection-meta",
    "document-count", "dropzone", "browse-btn", "file-input", "load-example-btn",
    "upload-status", "documents-empty", "document-list", "question-input", "resolve-btn",
    "resolve-hint", "submitted-context", "results", "decision-badge", "result-answer",
    "result-confidence", "result-explanation", "missing-information", "missing-list",
    "clarification-question", "citation-list", "governing-evidence", "competing-evidence",
    "no-governing-evidence", "no-competing-evidence", "technical-evidence",
    "loading-overlay", "loading-message", "evidence-dialog", "dialog-title",
    "dialog-location", "dialog-passage", "close-evidence",
  ].forEach((id) => {
    el[id.replace(/-([a-z])/g, (_, letter) => letter.toUpperCase())] = document.getElementById(id);
  });

  let state = { corpusId: null, corpusName: null, expiresAt: null, documents: [] };
  let displayedEvidence = new Map();
  let lastSubmittedContext = null;

  function showError(message) {
    el.errorBanner.textContent = message;
    el.errorBanner.classList.remove("hidden");
    window.clearTimeout(showError.timer);
    showError.timer = window.setTimeout(() => el.errorBanner.classList.add("hidden"), 7000);
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

    const text = await response.text();
    let body = null;
    if (text) {
      try {
        body = JSON.parse(text);
      } catch (error) {
        body = null;
      }
    }
    if (!response.ok) {
      const detail = body && body.detail;
      throw new Error(typeof detail === "string" ? detail : `Request failed (HTTP ${response.status})`);
    }
    return body;
  }

  function persistState() {
    sessionStorage.setItem(STORAGE_KEY, JSON.stringify({
      corpusId: state.corpusId,
      corpusName: state.corpusName,
      expiresAt: state.expiresAt,
    }));
  }

  function clearState() {
    state = { corpusId: null, corpusName: null, expiresAt: null, documents: [] };
    sessionStorage.removeItem(STORAGE_KEY);
  }

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
    el.documentList.replaceChildren();
    state.documents.forEach((doc) => {
      const item = document.createElement("li");
      const icon = document.createElement("span");
      icon.className = "doc-icon";
      icon.setAttribute("aria-hidden", "true");
      icon.textContent = "✓";
      const label = document.createElement("span");
      label.textContent = doc.original_filename || doc.filename || "Document";
      item.append(icon, label);
      el.documentList.appendChild(item);
    });
    el.documentsEmpty.classList.toggle("hidden", state.documents.length > 0);
    el.documentList.classList.toggle("hidden", state.documents.length === 0);
    updateResolveAvailability();
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
    const labels = {
      relevant_date: "Date",
      location: "Location",
      employee_category: "Employee category",
      circumstances: "Circumstances",
    };
    const parts = Object.entries(context || {}).map(([key, value]) => `${labels[key] || key}: ${value}`);
    return parts.length ? parts.join(" · ") : "No optional case context";
  }

  function updateContextFreshness() {
    if (!lastSubmittedContext) return;
    const changed = JSON.stringify(readContext()) !== JSON.stringify(lastSubmittedContext);
    el.submittedContext.classList.toggle("is-stale", changed);
    el.submittedContext.textContent = changed
      ? `Case context changed since this result. Rerun to review the updated case. Current: ${contextSummary(readContext())}`
      : `Submitted case: ${contextSummary(lastSubmittedContext)}`;
  }

  function formatLocation(source) {
    if (source.page_start !== null && source.page_start !== undefined) {
      const pages = source.page_end && source.page_end !== source.page_start
        ? `pp. ${source.page_start}–${source.page_end}`
        : `p. ${source.page_start}`;
      return source.section ? `${pages} · ${source.section}` : pages;
    }
    return source.section ? source.section : `Chunk ${Number(source.chunk_index || 0) + 1}`;
  }

  function addEvidenceCard(container, source, citation, label) {
    const card = document.createElement("article");
    card.className = "evidence-card";
    const treatment = citation ? citation.treatment : "related context";
    if (treatment === "conflicting") card.classList.add("evidence-card--conflicting");
    if (treatment === "uncertain") card.classList.add("evidence-card--uncertain");

    const head = document.createElement("div");
    head.className = "evidence-card__head";
    const filename = document.createElement("p");
    filename.className = "evidence-card__name";
    filename.textContent = source.original_filename || source.filename || citation.filename || "Evidence";
    const pill = document.createElement("span");
    pill.className = `treatment-pill treatment-pill--${treatment.replace(/[^a-z_]/g, "")}`;
    pill.textContent = label || treatment.replace(/_/g, " ");
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
    excerpt.textContent = citation || passage.length <= 420
      ? passage
      : `${passage.slice(0, 420).trimEnd()}…`;
    card.append(excerptLabel, excerpt);
    if (citation && citation.explanation) {
      const explanation = document.createElement("p");
      explanation.className = "evidence-card__explanation";
      explanation.textContent = citation.explanation;
      card.appendChild(explanation);
    }
    const open = document.createElement("button");
    open.type = "button";
    open.className = "evidence-card__open";
    open.textContent = citation ? "Open passage and verified quote" : "Open retrieved passage";
    open.addEventListener("click", () => openEvidence(source, citation));
    card.appendChild(open);
    container.appendChild(card);
  }

  function addCitationButton(citation, source) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "citation-link";
    button.textContent = `${citation.filename || source.filename} · ${formatLocation(source)}`;
    button.setAttribute("aria-label", `Open citation from ${citation.filename || source.filename}, ${formatLocation(source)}`);
    button.addEventListener("click", () => openEvidence(source, citation));
    el.citationList.appendChild(button);
  }

  function escapeRegex(value) {
    return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  }

  function openEvidence(source, citation) {
    const filename = source.original_filename || source.filename || "Evidence";
    el.dialogTitle.textContent = filename;
    el.dialogLocation.textContent = formatLocation(source);
    el.dialogPassage.replaceChildren();
    const passage = source.passage || "";
    const quote = citation && citation.quote;
    if (quote) {
      const expression = new RegExp(escapeRegex(quote.trim()).replace(/\s+/g, "\\s+"));
      const match = expression.exec(passage);
      if (match) {
        el.dialogPassage.append(
          document.createTextNode(passage.slice(0, match.index)),
          Object.assign(document.createElement("mark"), { textContent: match[0] }),
          document.createTextNode(passage.slice(match.index + match[0].length))
        );
      } else {
        el.dialogPassage.textContent = passage;
      }
    } else {
      el.dialogPassage.textContent = passage;
    }
    if (!passage) {
      el.dialogPassage.textContent = citation ? citation.quote : "Full passage is not available in this response.";
    }
    if (typeof el.evidenceDialog.showModal === "function") el.evidenceDialog.showModal();
    else el.evidenceDialog.setAttribute("open", "");
  }

  function renderTechnicalDetails(sources) {
    el.technicalEvidence.replaceChildren();
    sources.forEach((source) => {
      const row = document.createElement("div");
      row.className = "technical-row";
      [
        source.original_filename || source.filename,
        `Inferred status: ${source.status || "unknown"}`,
        `Inferred type: ${source.document_type || "unknown"}`,
        `Authority signal: ${formatScore(source.authority_score)}`,
        `Scope similarity: ${formatScore(source.scope_score)}`,
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
    el.results.classList.remove("hidden");
    const decision = result.decision || (result.conflict_detected ? "unresolved_conflict" : "answered");
    el.decisionBadge.className = `decision-badge decision-badge--${decision}`;
    el.decisionBadge.textContent = decision.replace(/_/g, " ");
    el.resultAnswer.textContent = result.answer || "The available evidence does not support a safe answer.";
    el.resultConfidence.textContent = result.confidence
      ? `Model-reported confidence: ${result.confidence}`
      : "Model-reported confidence: not provided";
    el.resultExplanation.textContent = result.explanation || result.reason || "No decision explanation was provided.";

    const missing = Array.isArray(result.missing_information) ? result.missing_information : [];
    el.missingInformation.classList.toggle("hidden", missing.length === 0 && !result.clarification_question);
    el.missingList.replaceChildren();
    missing.forEach((item) => {
      const li = document.createElement("li");
      li.textContent = item;
      el.missingList.appendChild(li);
    });
    el.clarificationQuestion.textContent = result.clarification_question || "";

    const sources = Array.isArray(result.top_sources) ? result.top_sources : [];
    const citations = Array.isArray(result.citations) ? result.citations : [];
    displayedEvidence = new Map(sources.map((source) => [source.chunk_id, source]));
    el.citationList.replaceChildren();
    el.governingEvidence.replaceChildren();
    el.competingEvidence.replaceChildren();

    const citedIds = new Set();
    citations.forEach((citation) => {
      const source = displayedEvidence.get(citation.chunk_id) || {
        chunk_id: citation.chunk_id,
        document_id: citation.document_id,
        filename: citation.filename,
        original_filename: citation.filename,
        page_start: citation.page_start,
        page_end: citation.page_end,
        section: citation.section,
        passage: citation.passage,
      };
      citedIds.add(citation.chunk_id);
      addCitationButton(citation, source);
      if (["conflicting", "superseded", "different_scope", "uncertain"].includes(citation.treatment)) {
        addEvidenceCard(el.competingEvidence, source, citation);
      } else {
        addEvidenceCard(el.governingEvidence, source, citation);
      }
    });
    sources.forEach((source) => {
      if (source.chunk_id && citedIds.has(source.chunk_id)) return;
      addEvidenceCard(el.competingEvidence, source, null, "retrieved context");
    });

    el.noGoverningEvidence.classList.toggle("hidden", el.governingEvidence.children.length > 0);
    el.noCompetingEvidence.classList.toggle("hidden", el.competingEvidence.children.length > 0);
    renderTechnicalDetails(sources);

    if (fixtureMode) {
      const tag = document.createElement("p");
      tag.className = "fixture-banner";
      tag.textContent = "LOCAL UI FIXTURE — illustrative state only; no API request was made.";
      el.results.prepend(tag);
    }
  }

  function formatScore(value) {
    return typeof value === "number" ? value.toFixed(3) : "not available";
  }

  async function createCollection(name) {
    const created = await apiFetch("/demo/corpora", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name }),
    });
    state = {
      corpusId: created.corpus_id,
      corpusName: created.name,
      expiresAt: created.expires_at,
      documents: [],
    };
    persistState();
    renderCollection();
  }

  function showUploadStatus(message, kind) {
    const status = el.uploadStatus;
    status.textContent = message;
    status.classList.remove("hidden", "upload-status--success", "upload-status--error");
    status.classList.add(kind === "error" ? "upload-status--error" : "upload-status--success");
  }

  async function uploadFiles(fileList) {
    const files = Array.from(fileList || []);
    if (!files.length) return;
    const accepted = [];
    const rejected = [];
    files.forEach((file) => {
      const supported = ALLOWED_EXTENSIONS.some((extension) => file.name.toLowerCase().endsWith(extension));
      (supported ? accepted : rejected).push(file);
    });
    if (rejected.length) showUploadStatus(`Skipped unsupported file(s): ${rejected.map((file) => file.name).join(", ")}`, "error");
    if (!accepted.length) return;

    const formData = new FormData();
    accepted.forEach((file) => formData.append("files", file));
    showLoading("Uploading documents…");
    try {
      const response = await apiFetch(`/demo/corpora/${encodeURIComponent(state.corpusId)}/documents`, {
        method: "POST",
        body: formData,
      });
      (response.uploaded || []).forEach((doc) => state.documents.push(doc));
      renderCollection();
      showUploadStatus(`Uploaded ${response.uploaded.length} file(s). Total: ${response.document_count}.`, "success");
    } catch (error) {
      showUploadStatus(error.message, "error");
    } finally {
      hideLoading();
    }
  }

  async function loadExampleDataset() {
    showLoading("Loading the example documents…");
    try {
      const response = await apiFetch(`/demo/corpora/${encodeURIComponent(state.corpusId)}/load_example`, { method: "POST" });
      (response.uploaded || []).forEach((doc) => state.documents.push(doc));
      renderCollection();
      showUploadStatus(`Loaded ${response.uploaded.length} example document(s).`, "success");
    } catch (error) {
      showUploadStatus(error.message, "error");
    } finally {
      hideLoading();
    }
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
    try {
      const result = await apiFetch(`/demo/corpora/${encodeURIComponent(state.corpusId)}/query`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question, context, top_k: 8 }),
      });
      renderResult(result, false);
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
        await apiFetch(`/demo/corpora/${encodeURIComponent(state.corpusId)}`, { method: "DELETE" });
      } catch (error) {
        showError(`Could not delete the previous collection: ${error.message}`);
      }
    }
    clearState();
    el.questionInput.value = "";
    document.getElementById("context-date").value = "";
    document.getElementById("context-location").value = "";
    document.getElementById("context-category").value = "";
    document.getElementById("context-circumstances").value = "";
    lastSubmittedContext = null;
    el.submittedContext.classList.add("hidden");
    el.uploadStatus.classList.add("hidden");
    el.results.classList.add("hidden");
    renderCollection();
  }

  async function restoreSession() {
    const raw = sessionStorage.getItem(STORAGE_KEY);
    if (!raw) return;
    let saved;
    try {
      saved = JSON.parse(raw);
    } catch (error) {
      sessionStorage.removeItem(STORAGE_KEY);
      return;
    }
    if (!saved || !saved.corpusId) return;
    try {
      const [metadata, docsResponse] = await Promise.all([
        apiFetch(`/demo/corpora/${encodeURIComponent(saved.corpusId)}`),
        apiFetch(`/demo/corpora/${encodeURIComponent(saved.corpusId)}/documents`),
      ]);
      state = {
        corpusId: metadata.corpus_id,
        corpusName: metadata.name,
        expiresAt: metadata.expires_at,
        documents: docsResponse.documents || [],
      };
      renderCollection();
    } catch (error) {
      clearState();
    }
  }

  function renderLocalFixture(name) {
    const citation = {
      chunk_id: "fixture:chunk:1",
      filename: "LOCAL FIXTURE — Regional policy.md",
      quote: "Employees in Region North may claim 30 days, except during leave.",
      passage: "Employees in Region North may claim 30 days, except during leave.",
      treatment: "conflicting",
      explanation: "This fixture demonstrates unresolved competing terms.",
    };
    const source = {
      chunk_id: citation.chunk_id,
      document_id: "fixture",
      original_filename: citation.filename,
      filename: citation.filename,
      passage: "Employees in Region North may claim 30 days, except during leave.",
      section: "Eligibility",
      status: "unknown",
      document_type: "unknown",
      semantic_score: .81,
      authority_score: .5,
      scope_score: .73,
      final_score: .68,
    };
    const rivalCitation = {
      chunk_id: "fixture:chunk:2",
      document_id: "fixture-rival",
      filename: "LOCAL FIXTURE — Employee handbook.md",
      quote: "Employees in Region North may claim 45 days, except during leave.",
      passage: "Employees in Region North may claim 45 days, except during leave.",
      treatment: "conflicting",
      explanation: "This equally applicable passage states a different limit.",
    };
    const rivalSource = {
      ...source,
      chunk_id: rivalCitation.chunk_id,
      document_id: rivalCitation.document_id,
      filename: rivalCitation.filename,
      original_filename: rivalCitation.filename,
      passage: rivalCitation.passage,
      final_score: .66,
    };
    const fixtures = {
      answered: {
        decision: "answered",
        answer: "The stated limit is 30 days, except during leave.",
        explanation: "The cited passage states the limit and preserves its exception.",
        citations: [{ ...citation, treatment: "governing", explanation: "Directly states the limit and exception." }],
        top_sources: [source],
        confidence: "high",
      },
      unresolved_conflict: {
        decision: "unresolved_conflict",
        answer: "Two applicable passages disagree. The documents provided do not establish which term governs.",
        explanation: "No documented precedence resolves these equally applicable statements.",
        citations: [citation, rivalCitation],
        top_sources: [source, rivalSource],
        confidence: "medium",
      },
      needs_information: {
        decision: "needs_information",
        answer: "The rule depends on the employee’s region and category.",
        explanation: "A relevant rule exists, but the case details needed to apply it are missing.",
        missing_information: ["Employee location", "Employee category"],
        clarification_question: "Which region and employee category apply?",
        citations: [{ ...citation, treatment: "governing" }],
        top_sources: [source],
        confidence: "low",
      },
      insufficient_evidence: {
        decision: "insufficient_evidence",
        answer: "The collection does not contain enough information to answer this question.",
        explanation: "No retrieved passage supports a reliable conclusion.",
        citations: [],
        top_sources: [],
        confidence: "low",
      },
    };
    if (!fixtures[name]) return false;
    document.title = `DocuInsight — Local UI fixture: ${name}`;
    renderResult(fixtures[name], true);
    return true;
  }

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
  el.fileInput.addEventListener("change", (event) => {
    uploadFiles(event.target.files);
    event.target.value = "";
  });
  el.loadExampleBtn.addEventListener("click", loadExampleDataset);
  ["dragover", "dragenter"].forEach((eventName) => el.dropzone.addEventListener(eventName, (event) => {
    event.preventDefault();
    el.dropzone.classList.add("dropzone--active");
  }));
  ["dragleave", "dragend"].forEach((eventName) => el.dropzone.addEventListener(eventName, () => {
    el.dropzone.classList.remove("dropzone--active");
  }));
  el.dropzone.addEventListener("drop", (event) => {
    event.preventDefault();
    el.dropzone.classList.remove("dropzone--active");
    uploadFiles(event.dataTransfer.files);
  });
  el.resolveBtn.addEventListener("click", askQuestion);
  el.newCollectionBtn.addEventListener("click", startNewCollection);
  Object.values(contextFields).forEach((id) => document.getElementById(id).addEventListener("input", updateContextFreshness));
  el.closeEvidence.addEventListener("click", () => el.evidenceDialog.close());
  el.evidenceDialog.addEventListener("click", (event) => {
    if (event.target === el.evidenceDialog) el.evidenceDialog.close();
  });

  const fixture = new URLSearchParams(window.location.search).get("fixture");
  if (!renderLocalFixture(fixture)) restoreSession();
  else {
    el.createCollectionCard.classList.add("hidden");
    el.workspace.classList.remove("hidden");
    el.results.classList.remove("hidden");
  }
})();
