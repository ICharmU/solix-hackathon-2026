/**
 * Solix FactLens - Lean Client State & Fact Transaction Store
 * Zero external frameworks. Fast, accessible, permissive.
 * No emojis.
 */

(function () {
  'use strict';

  // Application State
  const state = {
    sessionId: 'solix_demo_test',
    summary: {},
    facts: [],
    selectedCardIndex: 0,
    currentTimestamp: 0,
    maxDuration: 56,
    codeCache: {},
  };

  // DOM Elements
  const el = {
    sessionSelect: document.getElementById('session-select'),
    btnExportJson: document.getElementById('btn-export-json'),
    btnDownloadAll: document.getElementById('btn-download-all'),
    
    // KPIs
    kpiDuration: document.getElementById('kpi-duration'),
    kpiFramesCount: document.getElementById('kpi-frames-count'),
    kpiFrictionPct: document.getElementById('kpi-friction-pct'),
    kpiFrictionDesc: document.getElementById('kpi-friction-desc'),
    kpiStepsCount: document.getElementById('kpi-steps-count'),
    kpiAppsCount: document.getElementById('kpi-apps-count'),
    kpiAppsList: document.getElementById('kpi-apps-list'),
    kpiAutomationsCount: document.getElementById('kpi-automations-count'),

    // Omnibar & Chips
    omnibarInput: document.getElementById('omnibar-input'),
    chips: document.querySelectorAll('.chip-btn'),
    factCount: document.getElementById('fact-count'),
    factStreamContainer: document.getElementById('fact-stream-container'),

    // Inspector
    inspectorTimestampBadge: document.getElementById('inspector-timestamp-badge'),
    inspectorImg: document.getElementById('inspector-img'),
    keyframeFallback: document.getElementById('keyframe-fallback'),
    inspectorApp: document.getElementById('inspector-app'),
    inspectorWindow: document.getElementById('inspector-window'),
    inspectorAction: document.getElementById('inspector-action'),
    scriptFilename: document.getElementById('script-filename'),
    scriptTargetBadge: document.getElementById('script-target-badge'),
    inspectorCode: document.getElementById('inspector-code'),
    btnCopyCode: document.getElementById('btn-copy-code'),

    // Scrubber
    scrubberTime: document.getElementById('scrubber-time'),
    scrubberTask: document.getElementById('scrubber-task'),
    scrubberRange: document.getElementById('scrubber-range'),
    scrubberMaxLabel: document.getElementById('scrubber-max-label'),

    // Toast
    toast: document.getElementById('toast'),
  };

  // --------------------------------------------------------------------------
  // API Fetch Helpers
  // --------------------------------------------------------------------------
  async function fetchJson(endpoint) {
    try {
      const res = await fetch(endpoint);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      return await res.json();
    } catch (err) {
      console.warn(`Fetch error for ${endpoint}:`, err);
      return null;
    }
  }

  function showToast(message) {
    if (!el.toast) return;
    el.toast.textContent = message;
    el.toast.classList.add('show');
    setTimeout(() => {
      el.toast.classList.remove('show');
    }, 2200);
  }

  function formatTime(seconds) {
    const s = Math.max(0, Math.floor(seconds));
    const m = Math.floor(s / 60);
    const rem = s % 60;
    return `${String(m).padStart(2, '0')}:${String(rem).padStart(2, '0')}`;
  }

  // --------------------------------------------------------------------------
  // Initialization & Session Loading
  // --------------------------------------------------------------------------
  async function init() {
    setupEventListeners();
    await loadSessions();
    await loadSessionData(state.sessionId);
  }

  async function loadSessions() {
    const data = await fetchJson('/api/sessions');
    if (data && data.sessions && data.sessions.length > 0) {
      el.sessionSelect.innerHTML = '';
      data.sessions.forEach(sess => {
        const opt = document.createElement('option');
        opt.value = sess;
        opt.textContent = sess;
        el.sessionSelect.appendChild(opt);
      });
      state.sessionId = data.sessions[0];
      el.sessionSelect.value = state.sessionId;
    }
  }

  async function loadSessionData(sessionId) {
    state.sessionId = sessionId;
    
    // 1. Fetch Summary
    const summary = await fetchJson(`/api/sessions/${sessionId}/summary`);
    if (summary) {
      state.summary = summary;
      updateKpis(summary);
    }

    // 2. Query initial facts (Overview)
    await executeQuery('');

    // 3. Set scrubber to 0
    setTimelinePosition(0);
  }

  function updateKpis(summary) {
    const duration = summary.duration || '0s';
    const frames = summary.total_frames || 0;
    const frictionPct = summary.friction_percentage || 0;
    const frictionSec = summary.total_friction_seconds || 0;
    const automations = summary.total_automations || 0;
    const steps = summary.total_steps || 0;
    const apps = summary.active_applications || [];

    state.maxDuration = summary.duration_seconds || 56;

    el.kpiDuration.textContent = duration;
    el.kpiFramesCount.textContent = `${frames} Frames`;
    el.kpiFrictionPct.textContent = `${frictionPct}%`;
    el.kpiFrictionDesc.textContent = `${frictionSec}s lost to context switching & overhead`;
    el.kpiStepsCount.textContent = `${steps} Steps`;
    el.kpiAppsCount.textContent = `${apps.length} Apps`;
    el.kpiAppsList.textContent = apps.join(', ') || 'Various applications';
    el.kpiAutomationsCount.textContent = `${automations} Scripts`;

    // Update scrubber bounds
    el.scrubberRange.max = Math.max(0, state.maxDuration - 1);
    el.scrubberMaxLabel.textContent = `${Math.round(state.maxDuration)}s (End)`;
  }

  // --------------------------------------------------------------------------
  // Fact Query Execution & Rendering
  // --------------------------------------------------------------------------
  async function executeQuery(queryText) {
    const cleanQuery = (queryText || '').trim();
    const data = await fetchJson(`/api/sessions/${state.sessionId}/query?q=${encodeURIComponent(cleanQuery)}`);
    
    const results = (data && data.results) ? data.results : [];
    state.facts = results;
    renderFactCards(results);
  }

  function renderFactCards(cards) {
    el.factStreamContainer.innerHTML = '';
    el.factCount.textContent = `${cards.length} ${cards.length === 1 ? 'Fact' : 'Facts'}`;

    if (!cards || cards.length === 0) {
      const emptyCard = document.createElement('div');
      emptyCard.className = 'fact-card';
      emptyCard.innerHTML = `
        <div class="fact-card-top">
          <span class="fact-card-title">No Direct Fact Found</span>
          <span class="fact-badge badge-neutral">0 Results</span>
        </div>
        <div class="fact-detail">Try a broader query such as "friction", "automation", "steps", "OBS", or a timestamp like "25s".</div>
      `;
      el.factStreamContainer.appendChild(emptyCard);
      return;
    }

    cards.forEach((card, index) => {
      const cardEl = document.createElement('article');
      cardEl.className = `fact-card ${index === state.selectedCardIndex ? 'selected' : ''}`;
      cardEl.setAttribute('tabindex', '0');
      cardEl.setAttribute('role', 'button');
      cardEl.setAttribute('aria-label', `${card.title}. ${card.metric_headline || ''}`);

      let badgeClass = 'badge-neutral';
      if (card.card_type === 'friction') badgeClass = 'badge-amber';
      else if (card.card_type === 'automation') badgeClass = 'badge-purple';
      else if (card.card_type === 'step') badgeClass = 'badge-cyan';
      else if (card.card_type === 'time_allocation') badgeClass = 'badge-emerald';

      const timestampText = card.evidence_timestamp !== undefined 
        ? formatTime(card.evidence_timestamp) 
        : '00:00';

      cardEl.innerHTML = `
        <div class="fact-card-top">
          <h3 class="fact-card-title">${escapeHtml(card.title || 'Fact')}</h3>
          <span class="fact-badge ${badgeClass}">${escapeHtml(card.badge || 'Verified')}</span>
        </div>
        ${card.metric_headline ? `<div class="fact-headline">${escapeHtml(card.metric_headline)}</div>` : ''}
        <p class="fact-detail">${escapeHtml(card.detail || '')}</p>
        <div class="fact-card-bottom">
          <span class="fact-evidence-tag">
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
              <circle cx="12" cy="12" r="10"></circle>
              <polyline points="12 6 12 12 16 14"></polyline>
            </svg>
            Cited Evidence: ${timestampText} (${card.evidence_timestamp || 0}s)
          </span>
          <div class="fact-actions">
            <button class="action-btn btn-copy-fact" data-idx="${index}" title="Copy fact summary to clipboard">
              Copy Fact
            </button>
            <button class="action-btn primary btn-inspect-fact" data-idx="${index}" title="Inspect cited keyframe and details">
              Inspect Proof
            </button>
          </div>
        </div>
      `;

      cardEl.addEventListener('click', (e) => {
        if (e.target.closest('.btn-copy-fact')) return;
        selectFactCard(index);
      });

      cardEl.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault();
          selectFactCard(index);
        }
      });

      const copyBtn = cardEl.querySelector('.btn-copy-fact');
      if (copyBtn) {
        copyBtn.addEventListener('click', (e) => {
          e.stopPropagation();
          copyFactToClipboard(card);
        });
      }

      el.factStreamContainer.appendChild(cardEl);
    });

    // Auto-select first card
    if (cards.length > 0) {
      selectFactCard(0);
    }
  }

  function selectFactCard(index) {
    state.selectedCardIndex = index;
    const cards = el.factStreamContainer.querySelectorAll('.fact-card');
    cards.forEach((c, i) => {
      c.classList.toggle('selected', i === index);
    });

    const card = state.facts[index];
    if (!card) return;

    // Synchronize Inspector Evidence
    const ts = card.evidence_timestamp !== undefined ? card.evidence_timestamp : 0;
    setTimelinePosition(ts);

    // If card has code snippet, show it
    if (card.code_snippet) {
      el.scriptFilename.textContent = card.script_filename || 'automation.py';
      el.scriptTargetBadge.textContent = card.badge || 'Python Script';
      el.inspectorCode.textContent = card.code_snippet;
    } else if (card.card_type === 'automation' && card.raw_data && card.raw_data.code) {
      el.scriptFilename.textContent = card.raw_data.script_filename || 'automation.py';
      el.scriptTargetBadge.textContent = card.raw_data.automation_type || 'Python Script';
      el.inspectorCode.textContent = card.raw_data.code;
    }
  }

  function copyFactToClipboard(card) {
    const text = `FACT: ${card.title}\n${card.metric_headline ? card.metric_headline + '\n' : ''}${card.detail}\nEvidence: ${formatTime(card.evidence_timestamp || 0)} (${card.evidence_timestamp || 0}s)`;
    navigator.clipboard.writeText(text).then(() => {
      showToast('Fact copied to clipboard');
    });
  }

  // --------------------------------------------------------------------------
  // Timeline Scrubber & Frame Inspector
  // --------------------------------------------------------------------------
  async function setTimelinePosition(timestamp) {
    state.currentTimestamp = timestamp;
    el.scrubberRange.value = timestamp;

    const formatted = formatTime(timestamp);
    const maxFormatted = formatTime(state.maxDuration);
    el.scrubberTime.textContent = `${formatted} / ${maxFormatted}`;
    el.inspectorTimestampBadge.textContent = formatted;

    // Load keyframe image
    const frameIndex = timestamp + 1;
    const frameFile = `frame_${String(frameIndex).padStart(6, '0')}.jpg`;
    const frameUrl = `/frames/${state.sessionId}/${frameFile}`;

    el.inspectorImg.onload = () => {
      el.inspectorImg.style.display = 'block';
      el.keyframeFallback.style.display = 'none';
    };
    el.inspectorImg.onerror = () => {
      el.inspectorImg.style.display = 'none';
      el.keyframeFallback.style.display = 'block';
    };
    el.inspectorImg.src = frameUrl;

    // Fetch slot metadata
    const slot = await fetchJson(`/api/sessions/${state.sessionId}/slot/${timestamp}`);
    if (slot) {
      el.inspectorApp.textContent = slot.active_application || 'Unknown';
      el.inspectorWindow.textContent = slot.window_title || 'Untitled';
      el.inspectorWindow.title = slot.window_title || '';
      el.inspectorAction.textContent = slot.observed_interaction || 'Idle';
      el.inspectorAction.title = slot.observed_interaction || '';

      el.scrubberTask.textContent = `At ${formatted} (${slot.active_application}): ${slot.observed_interaction || 'Active'}`;
    }
  }

  // --------------------------------------------------------------------------
  // Event Listeners & Hotkeys
  // --------------------------------------------------------------------------
  function setupEventListeners() {
    // Session selector change
    el.sessionSelect.addEventListener('change', (e) => {
      loadSessionData(e.target.value);
    });

    // Omnibar input
    let debounceTimer;
    el.omnibarInput.addEventListener('input', (e) => {
      clearTimeout(debounceTimer);
      debounceTimer = setTimeout(() => {
        executeQuery(e.target.value);
        // Clear active chip styling when typing custom query
        el.chips.forEach(c => c.classList.remove('active'));
      }, 150);
    });

    // Quick-Fact Chips
    el.chips.forEach(chip => {
      chip.addEventListener('click', () => {
        el.chips.forEach(c => c.classList.remove('active'));
        chip.classList.add('active');
        const q = chip.getAttribute('data-query');
        el.omnibarInput.value = q;
        executeQuery(q);
      });
    });

    // Scrubber input
    el.scrubberRange.addEventListener('input', (e) => {
      setTimelinePosition(parseInt(e.target.value, 10));
    });

    // Copy Code button
    el.btnCopyCode.addEventListener('click', () => {
      const code = el.inspectorCode.textContent;
      navigator.clipboard.writeText(code).then(() => {
        showToast('Code copied to clipboard');
      });
    });

    // Export Facts as JSON
    el.btnExportJson.addEventListener('click', async () => {
      const data = await fetchJson(`/api/sessions/${state.sessionId}/facts`);
      if (data) {
        const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `${state.sessionId}_facts_export.json`;
        a.click();
        URL.revokeObjectURL(url);
        showToast('Facts JSON exported');
      }
    });

    // Download Automation Package
    el.btnDownloadAll.addEventListener('click', () => {
      const code = el.inspectorCode.textContent;
      const filename = el.scriptFilename.textContent || 'automation_package.py';
      const blob = new Blob([code], { type: 'text/x-python' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = filename;
      a.click();
      URL.revokeObjectURL(url);
      showToast(`Downloaded ${filename}`);
    });

    // Keyboard Shortcuts (Hotkeys)
    window.addEventListener('keydown', (e) => {
      const isInputActive = document.activeElement === el.omnibarInput;

      // Focus search: '/' or Ctrl+K
      if ((e.key === '/' && !isInputActive) || (e.ctrlKey && e.key.toLowerCase() === 'k')) {
        e.preventDefault();
        el.omnibarInput.focus();
        el.omnibarInput.select();
        return;
      }

      // Escape: Clear and blur
      if (e.key === 'Escape') {
        el.omnibarInput.value = '';
        el.omnibarInput.blur();
        el.chips.forEach(c => c.classList.remove('active'));
        if (el.chips[0]) el.chips[0].classList.add('active');
        executeQuery('');
        return;
      }

      // If typing in input, don't trigger other hotkeys
      if (isInputActive) return;

      // Arrow keys for timeline scrubber
      if (e.key === 'ArrowLeft') {
        e.preventDefault();
        const next = Math.max(0, state.currentTimestamp - 1);
        setTimelinePosition(next);
      } else if (e.key === 'ArrowRight') {
        e.preventDefault();
        const next = Math.min(state.maxDuration - 1, state.currentTimestamp + 1);
        setTimelinePosition(next);
      }

      // Number keys 1-4 for quick chips
      if (e.key >= '1' && e.key <= '4') {
        const targetChip = Array.from(el.chips).find(c => c.getAttribute('data-key') === e.key);
        if (targetChip) {
          targetChip.click();
        }
      }
    });
  }

  function escapeHtml(str) {
    if (typeof str !== 'string') return '';
    return str
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#039;');
  }

  // Run on DOM ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();

