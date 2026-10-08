const API = "http://localhost:8000";

/* ── Module-level state ── */
let selectedFile = null;

/* ── Helpers ── */
function getTime() {
  return new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

function scrollToBottom() {
  const msgs = document.getElementById("messages");
  msgs.scrollTop = msgs.scrollHeight;
}

function appendMessage(text, role) {
  const msgs = document.getElementById("messages");

  const wrapper = document.createElement("div");
  wrapper.className = `message ${role}`;

  const avatar = document.createElement("div");
  avatar.className = "avatar";
  avatar.textContent = role === "bot" ? "🤖" : "🧑";

  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.innerHTML = `<p>${text}</p><span class="timestamp">${getTime()}</span>`;

  wrapper.appendChild(avatar);
  wrapper.appendChild(bubble);
  msgs.appendChild(wrapper);
  scrollToBottom();
  return wrapper;
}

function showTyping() {
  const msgs = document.getElementById("messages");

  const wrapper = document.createElement("div");
  wrapper.className = "message bot";
  wrapper.id = "typing";

  const avatar = document.createElement("div");
  avatar.className = "avatar";
  avatar.textContent = "🤖";

  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.innerHTML = `
    <div class="typing-indicator">
      <span></span><span></span><span></span>
    </div>`;

  wrapper.appendChild(avatar);
  wrapper.appendChild(bubble);
  msgs.appendChild(wrapper);
  scrollToBottom();
}

function removeTyping() {
  const t = document.getElementById("typing");
  if (t) t.remove();
}

/* ── Initialize Knowledgebase ── */
async function initKnowledgebase() {
  const btn    = document.getElementById("initBtn");
  const icon   = document.getElementById("initBtnIcon");
  const label  = document.getElementById("initBtnText");
  const status = document.getElementById("initStatus");

  btn.disabled = true;
  icon.textContent  = "⏳";
  label.textContent = "Initializing...";
  status.textContent = "Building knowledgebase, please wait…";
  status.className   = "init-status";

  try {
    const res  = await fetch(`${API}/create-knowledgebase`, { method: "POST" });
    const data = await res.json();

    if (res.ok) {
      icon.textContent   = "✅";
      label.textContent  = "Knowledgebase Ready";
      status.textContent = data.message;
      status.className   = "init-status success";
      appendMessage("Knowledgebase is ready! Ask me anything 🎉", "bot");
    } else {
      throw new Error(data.detail || "Unknown error");
    }
  } catch (err) {
    icon.textContent   = "⚡";
    label.textContent  = "Retry Initialization";
    status.textContent = "❌ " + err.message;
    status.className   = "init-status error";
    btn.disabled = false;
  }
}

/* ── Multimodal: file selection & validation ── */
const ALLOWED_EXTENSIONS = [".png", ".jpg", ".jpeg", ".webp", ".pdf"];
const MAX_FILE_BYTES = 10 * 1024 * 1024; // 10 MB

function handleFileSelect(event) {
  const file = event.target.files && event.target.files[0];
  if (!file) return;

  // Validate extension
  const lower = file.name.toLowerCase();
  const validExt = ALLOWED_EXTENSIONS.some(function(ext) {
    return lower.endsWith(ext);
  });
  if (!validExt) {
    appendMessage(
      "⚠️ Unsupported file type. Please attach a PNG, JPG, JPEG, WEBP, or PDF file.",
      "bot"
    );
    event.target.value = "";
    return;
  }

  // Validate size
  if (file.size > MAX_FILE_BYTES) {
    appendMessage(
      "⚠️ File is too large. Maximum allowed size is 10 MB.",
      "bot"
    );
    event.target.value = "";
    return;
  }

  selectedFile = file;

  // Show file pill (safe: textContent only)
  const pill     = document.getElementById("filePill");
  const pillName = document.getElementById("filePillName");
  pillName.textContent = file.name;
  pill.style.display = "inline-flex";
}

function clearAttachment() {
  selectedFile = null;
  const pill  = document.getElementById("filePill");
  const input = document.getElementById("evidenceFileInput");
  pill.style.display = "none";
  input.value = "";
}

/* ── Send a question ── */
async function sendMessage() {
  const input    = document.getElementById("questionInput");
  const question = input.value.trim();
  if (!question && !selectedFile) return;

  const displayText = question || "(evidence upload)";
  input.value = "";
  appendMessage(displayText, "user");
  showTyping();

  // ── Branch: multimodal ──────────────────────────────────────────────────
  if (selectedFile) {
    const fileToSend = selectedFile;
    clearAttachment();

    try {
      const form = new FormData();
      form.append("file", fileToSend);
      if (question) form.append("message", question);
      form.append("processing_mode", "auto");

      const res  = await fetch(`${API}/multimodal/analyze`, {
        method: "POST",
        body: form,
      });
      const data = await res.json();
      removeTyping();

      if (!res.ok) {
        const safeMsg = typeof data.detail === "string"
          ? data.detail.replace(/[<>]/g, "")
          : "Multimodal analysis failed.";
        appendMessage("⚠️ " + safeMsg, "bot");
        return;
      }

      if (data.status === "queued" && data.job_id) {
        // Async path — show status message and start polling
        const botWrapper = appendMessage("📋 Evidence received. Analyzing in background…", "bot");
        const statusEl   = botWrapper.querySelector("p");
        pollJobStatus(data.job_id, statusEl, botWrapper);
      } else {
        // Sync path — result is already in the response
        const resultData = data.result || data;
        const botWrapper = appendBotBubbleEmpty();
        renderEvidenceCard(resultData, botWrapper.querySelector(".bubble"));
        scrollToBottom();
      }
    } catch (err) {
      removeTyping();
      appendMessage("❌ Could not reach the server. Make sure the backend is running on port 8000.", "bot");
    }
    return;
  }

  // ── Branch: standard /ask (unchanged) ──────────────────────────────────
  try {
    const res  = await fetch(`${API}/ask`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });
    const data = await res.json();
    removeTyping();

    if (res.ok) {
      appendMessage(data.answer, "bot");
    } else {
      const msg = data.detail || "Something went wrong.";
      appendMessage(
        msg.includes("Knowledgebase not found")
          ? "⚠️ Please click <strong>Initialize Knowledgebase</strong> first before asking questions."
          : "⚠️ " + msg,
        "bot"
      );
    }
  } catch (err) {
    removeTyping();
    appendMessage("❌ Could not reach the server. Make sure the backend is running on port 8000.", "bot");
  }
}

/* ── Append an empty bot bubble (returns wrapper) ── */
function appendBotBubbleEmpty() {
  const msgs = document.getElementById("messages");

  const wrapper = document.createElement("div");
  wrapper.className = "message bot";

  const avatar = document.createElement("div");
  avatar.className = "avatar";
  avatar.textContent = "🤖";

  const bubble = document.createElement("div");
  bubble.className = "bubble";

  wrapper.appendChild(avatar);
  wrapper.appendChild(bubble);
  msgs.appendChild(wrapper);
  scrollToBottom();
  return wrapper;
}

/* ── Poll job status (async multimodal) ── */
async function pollJobStatus(jobId, statusEl, wrapper) {
  const MAX_ATTEMPTS  = 30;
  const INTERVAL_MS   = 1200;
  let   attempts      = 0;

  const tick = async function() {
    attempts++;
    try {
      const res  = await fetch(`${API}/multimodal/jobs/${encodeURIComponent(jobId)}`);
      const data = await res.json();

      if (!res.ok) {
        if (statusEl) statusEl.textContent = "⚠️ Could not retrieve job status.";
        return;
      }

      const status = data.status || "";

      if (status === "completed") {
        const resultData = (data.result) || data;
        const bubble = wrapper.querySelector(".bubble");
        if (bubble) {
          const oldP = bubble.querySelector("p");
          if (oldP) oldP.remove();
          renderEvidenceCard(resultData, bubble);
          scrollToBottom();
        }
        return;
      }

      if (status === "failed" || status === "error") {
        if (statusEl) statusEl.textContent = "⚠️ Analysis failed. Please try again.";
        return;
      }

      // Still in progress
      const statusMessages = {
        queued:     "📋 Queued for analysis…",
        processing: "🔍 Analyzing document…",
        retrying:   "🔄 Retrying analysis…",
      };
      const displayStatus = statusMessages[status] || "⏳ Processing…";
      if (statusEl) statusEl.textContent = displayStatus;

      if (attempts < MAX_ATTEMPTS) {
        setTimeout(tick, INTERVAL_MS);
      } else {
        if (statusEl) statusEl.textContent = "⏰ Analysis is taking longer than expected. Check back later.";
      }
    } catch (err) {
      if (statusEl) statusEl.textContent = "❌ Connection error while checking job status.";
    }
  };

  setTimeout(tick, INTERVAL_MS);
}

/* ── Render evidence card (safe DOM API — no innerHTML for user/backend data) ── */
function renderEvidenceCard(data, container) {
  const card = document.createElement("div");
  card.className = "evidence-card";

  // ── Header ──────────────────────────────────────────────────────────────
  const header = document.createElement("div");
  header.className = "evidence-header";

  const headerIcon = document.createElement("span");
  headerIcon.textContent = "📄";
  header.appendChild(headerIcon);

  const headerTitle = document.createElement("span");
  headerTitle.textContent = "Evidence Analysis";
  header.appendChild(headerTitle);

  const quality = data.evidence_quality || data.extraction_quality || (data.extraction && data.extraction.quality) || "";
  if (quality) {
    const badge = document.createElement("span");
    const qLower = String(quality).toLowerCase();
    badge.className = "evidence-quality " + (
      qLower === "good" ? "good" : qLower === "acceptable" ? "acceptable" : "poor"
    );
    badge.textContent = quality.toUpperCase();
    header.appendChild(badge);
  }

  card.appendChild(header);

  // ── Document / File Info ─────────────────────────────────────────────────
  const docFilename = data.original_filename || (data.metadata && data.metadata.original_filename);
  const docType = data.document_type || "document";
  if (docFilename) {
    const fileInfo = document.createElement("div");
    fileInfo.className = "evidence-status-text";
    fileInfo.textContent = `File: ${docFilename} (${docType.toUpperCase()})`;
    card.appendChild(fileInfo);
  }

  // ── User / Status Message ────────────────────────────────────────────────
  const userMsg = data.user_message || (data.comparison && data.comparison.clarification_message);
  if (userMsg) {
    const msgEl = document.createElement("div");
    msgEl.className = "evidence-status-text";
    msgEl.style.color = "var(--text)";
    msgEl.style.fontStyle = "normal";
    msgEl.style.padding = "8px 12px";
    msgEl.style.background = "var(--surface)";
    msgEl.style.borderRadius = "8px";
    msgEl.style.border = "1px solid var(--border)";
    msgEl.style.lineHeight = "1.5";
    msgEl.textContent = userMsg;
    card.appendChild(msgEl);
  }

  // ── Quality Notes ────────────────────────────────────────────────────────
  if (data.quality_notes && Array.isArray(data.quality_notes) && data.quality_notes.length > 0) {
    const notesDiv = document.createElement("div");
    notesDiv.className = "evidence-status-text";
    notesDiv.style.color = "var(--muted)";
    notesDiv.style.fontSize = "0.75rem";
    notesDiv.textContent = "ℹ️ " + data.quality_notes.join(" | ");
    card.appendChild(notesDiv);
  }

  // ── Extracted fields ─────────────────────────────────────────────────────
  const extracted = data.extracted_fields ||
    (data.extraction && data.extraction.fields) ||
    (data.extraction && data.extraction.extracted_fields) ||
    null;

  if (extracted && typeof extracted === "object") {
    const FIELD_LABELS = {
      order_id: "Order ID", invoice_number: "Invoice #", date: "Date",
      amount: "Amount", currency: "Currency", product_name: "Product",
      product_code: "Product Code", quantity: "Quantity",
      error_code: "Error Code", tracking_number: "Tracking #",
      delivery_status: "Delivery Status", payment_method: "Payment Method",
    };
    const fieldKeys = Object.keys(extracted).filter(function(k) {
      const v = extracted[k];
      if (v === null || v === undefined || v === "") return false;
      if (typeof v === "object" && (v.value === null || v.value === undefined || v.value === "")) return false;
      return true;
    });

    if (fieldKeys.length > 0) {
      const secTitle = document.createElement("div");
      secTitle.className = "evidence-section-title";
      secTitle.textContent = "Extracted Fields";
      card.appendChild(secTitle);

      const grid = document.createElement("div");
      grid.className = "evidence-grid";

      fieldKeys.forEach(function(k) {
        const keyEl = document.createElement("span");
        keyEl.className = "ev-key";
        keyEl.textContent = (FIELD_LABELS[k] || k) + ":";

        const valEl = document.createElement("span");
        let val = extracted[k];
        if (val && typeof val === "object" && val.value !== undefined) {
          val = val.value;
        }

        if (val === null || val === undefined || val === "") {
          valEl.className = "ev-empty";
          valEl.textContent = "Not detected";
        } else {
          valEl.className = "ev-val";
          valEl.textContent = String(val);
        }

        grid.appendChild(keyEl);
        grid.appendChild(valEl);
      });
      card.appendChild(grid);
    } else {
      const noFields = document.createElement("div");
      noFields.className = "evidence-status-text";
      noFields.textContent = "No structured fields (Order ID, Amount, Date) detected in this file.";
      card.appendChild(noFields);
    }
  }

  // ── Comparison results ───────────────────────────────────────────────────
  const comparison = data.comparison_results ||
    (data.comparison && data.comparison.results) ||
    null;

  const compArray = Array.isArray(comparison) ? comparison :
    (comparison && typeof comparison === "object" ? Object.values(comparison) : null);

  if (compArray && compArray.length > 0) {
    const secTitle = document.createElement("div");
    secTitle.className = "evidence-section-title";
    secTitle.textContent = "Comparison Results";
    card.appendChild(secTitle);

    compArray.forEach(function(item) {
      if (!item) return;
      const row = document.createElement("div");
      row.className = "comparison-row";

      const badge = document.createElement("span");
      const result = String(item.result || item.status || "missing").toLowerCase();
      badge.className = "cmp-badge " + (
        result === "match" ? "match" :
        result === "conflict" ? "conflict" : "missing"
      );
      badge.textContent = result.toUpperCase();

      const fieldLabel = document.createElement("span");
      fieldLabel.textContent = String(item.field || item.field_name || "field");

      row.appendChild(badge);
      row.appendChild(fieldLabel);
      card.appendChild(row);
    });
  }

  // ── Conflict details ─────────────────────────────────────────────────────
  const conflicts = data.conflicts ||
    (data.comparison && data.comparison.conflicts) ||
    null;

  const conflictArray = Array.isArray(conflicts) ? conflicts :
    (conflicts && typeof conflicts === "object" ? Object.values(conflicts) : null);

  if (conflictArray && conflictArray.length > 0) {
    const secTitle = document.createElement("div");
    secTitle.className = "evidence-section-title";
    secTitle.textContent = "Conflicts Detected";
    card.appendChild(secTitle);

    conflictArray.forEach(function(c) {
      if (!c) return;
      const alert = document.createElement("div");
      alert.className = "conflict-alert";

      const alertTitle = document.createElement("div");
      alertTitle.className = "conflict-alert-title";
      alertTitle.textContent = "⚠️ Mismatch";
      alert.appendChild(alertTitle);

      const fieldName = document.createElement("div");
      fieldName.className = "conflict-field-name";
      fieldName.textContent = String(c.field || c.field_name || "");
      alert.appendChild(fieldName);

      if (c.customer_value !== undefined || c.document_value !== undefined) {
        const vals = document.createElement("div");
        vals.className = "conflict-values";

        const custLabel = document.createElement("span");
        custLabel.className = "conflict-val-label";
        custLabel.textContent = "Your message:";
        const custVal = document.createElement("span");
        custVal.className = "conflict-val-data";
        custVal.textContent = String(c.customer_value !== undefined ? c.customer_value : "—");

        const docLabel = document.createElement("span");
        docLabel.className = "conflict-val-label";
        docLabel.textContent = "Document:";
        const docVal = document.createElement("span");
        docVal.className = "conflict-val-data";
        docVal.textContent = String(c.document_value !== undefined ? c.document_value : "—");

        vals.appendChild(custLabel);
        vals.appendChild(custVal);
        vals.appendChild(docLabel);
        vals.appendChild(docVal);
        alert.appendChild(vals);
      }

      if (c.clarification_message || c.message) {
        const clarify = document.createElement("div");
        clarify.className = "conflict-clarify";
        clarify.textContent = String(c.clarification_message || c.message);
        alert.appendChild(clarify);
      }

      card.appendChild(alert);
    });
  }

  // ── Security notice ──────────────────────────────────────────────────────
  const secNotice = data.security_notice ||
    (data.security && data.security.notice) ||
    null;
  if (secNotice) {
    const notice = document.createElement("div");
    notice.className = "security-notice";

    const icon = document.createElement("span");
    icon.textContent = "🔒";
    const text = document.createElement("span");
    text.textContent = String(secNotice);

    notice.appendChild(icon);
    notice.appendChild(text);
    card.appendChild(notice);
  }

  // ── Retention notice ─────────────────────────────────────────────────────
  const retentionHours = data.retention_hours ||
    (data.retention && data.retention.hours) ||
    null;
  if (retentionHours) {
    const notice = document.createElement("div");
    notice.className = "retention-notice";

    const icon = document.createElement("span");
    icon.textContent = "🗂️";
    const text = document.createElement("span");
    text.textContent = "File will be automatically deleted after " +
      String(retentionHours) + " hours.";

    notice.appendChild(icon);
    notice.appendChild(text);
    card.appendChild(notice);
  }

  // ── Timestamp ────────────────────────────────────────────────────────────
  const ts = document.createElement("span");
  ts.className = "timestamp";
  ts.textContent = getTime();
  card.appendChild(ts);

  container.appendChild(card);
}

/* ── Sample question click ── */
function fillQuestion(el) {
  const input = document.getElementById("questionInput");
  input.value = el.textContent;
  input.focus();
}