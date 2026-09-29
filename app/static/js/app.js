/* ==========================================================================
   Persian Speed ALPR & Radar Control Dashboard - Client Script
   app/static/js/app.js
   ========================================================================== */

(() => {
  'use strict';

  const state = {
    videoLoaded: false,
    currentVideoFile: null,
    isStreaming: false,
    currentAngle: 0,
    ws: null,
    wsReconnectTimer: null,
    // Calibration State
    calibImg: new Image(),
    calibPoints: [],       // [{x, y}] in natural image coordinates
    customGateLine: [],    // [{x, y}] 2 points for custom trigger shot line
    gateOption: 'default', // 'default' (P4-P3) or 'custom' (G1-G2)
    dragTarget: null,      // { type: 'road'|'gate', index: number }
    cachedViolations: new Map()
  };

  const PERSIAN_DIGITS = ['۰', '۱', '۲', '۳', '۴', '۵', '۶', '۷', '۸', '۹'];
  const toPersianDigits = (str) => String(str).replace(/[0-9]/g, (d) => PERSIAN_DIGITS[parseInt(d, 10)]);

  function formatImgUrl(path) {
    if (!path) return '/static/placeholder.png';
    let clean = String(path).replace(/\\/g, '/');
    if (!clean.startsWith('/') && !clean.startsWith('http') && !clean.startsWith('data:')) {
      clean = '/' + clean;
    }
    return clean;
  }

  const dom = {
    wsStatus: document.getElementById('wsStatus'),
    navTabs: document.querySelectorAll('.nav-tab'),
    tabViolationBadge: document.getElementById('tabViolationBadge'),
    statTotalViolations: document.getElementById('statTotalViolations'),
    statMaxSpeed: document.getElementById('statMaxSpeed'),
    statAvgSpeed: document.getElementById('statAvgSpeed'),
    inputSpeedLimit: document.getElementById('inputSpeedLimit'),
    btnSetLimit: document.getElementById('btnSetLimit'),
    dropZone: document.getElementById('dropZone'),
    videoFileInput: document.getElementById('videoFileInput'),
    uploadProgressContainer: document.getElementById('uploadProgressContainer'),
    uploadProgressBar: document.getElementById('uploadProgressBar'),
    currentAngleBadge: document.getElementById('currentAngleBadge'),
    anglePills: document.querySelectorAll('.angle-btn'),
    liveStreamFeed: document.getElementById('liveStreamFeed'),
    streamOverlayPlaceholder: document.getElementById('streamOverlayPlaceholder'),
    videoMetaText: document.getElementById('videoMetaText'),
    btnStartStream: document.getElementById('btnStartStream'),
    btnStopStream: document.getElementById('btnStopStream'),
    violationsList: document.getElementById('violationsList'),
    emptyViolationsState: document.getElementById('emptyViolationsState'),
    violationCountBadge: document.getElementById('violationCountBadge'),
    dbTableBody: document.getElementById('dbTableBody'),
    filterPlateInput: document.getElementById('filterPlateInput'),
    btnRefreshDbTable: document.getElementById('btnRefreshDbTable'),
    tableRecordCount: document.getElementById('tableRecordCount'),
    // Studio Calibration
    btnOpenCalibration: document.getElementById('btnOpenCalibration'),
    calibrationModal: document.getElementById('calibrationModal'),
    btnCloseCalibration: document.getElementById('btnCloseCalibration'),
    calibrationCanvas: document.getElementById('calibrationCanvas'),
    calibRoadWidth: document.getElementById('calibRoadWidth'),
    calibRoadLength: document.getElementById('calibRoadLength'),
    btnResetAllCalib: document.getElementById('btnResetAllCalib'),
    btnSaveCalibration: document.getElementById('btnSaveCalibration'),
    optCardDefault: document.getElementById('optCardDefault'),
    optCardCustom: document.getElementById('optCardCustom'),
    radioDefaultGate: document.getElementById('radioDefaultGate'),
    radioCustomGate: document.getElementById('radioCustomGate'),
    calibStatusGuide: document.getElementById('calibStatusGuide'),
    // Inspection Modal
    inspectModal: document.getElementById('inspectModal'),
    btnCloseInspect: document.getElementById('btnCloseInspect'),
    btnCloseInspectBtn: document.getElementById('btnCloseInspectBtn'),
    inspectRecordTitle: document.getElementById('inspectRecordTitle'),
    inspectVehImg: document.getElementById('inspectVehImg'),
    inspectPlateImg: document.getElementById('inspectPlateImg'),
    inspectPlateBadge: document.getElementById('inspectPlateBadge'),
    inspectSpeed: document.getElementById('inspectSpeed'),
    inspectLimit: document.getElementById('inspectLimit'),
    inspectExcess: document.getElementById('inspectExcess'),
    inspectAccuracy: document.getElementById('inspectAccuracy')
  };

  document.addEventListener('DOMContentLoaded', () => {
    initTabs();
    initInspectModal();
    initWebSocket();
    initUploadListeners();
    initRotationControls();
    initStreamControls();
    initCalibrationStudio();
    fetchStats();
    fetchViolations();
    fetchConfig();
  });

  // ==========================================================================
  // Navigation Tabs
  // ==========================================================================
  function initTabs() {
    dom.navTabs.forEach(tab => {
      tab.addEventListener('click', () => {
        const targetTabId = tab.getAttribute('data-tab');
        dom.navTabs.forEach(t => t.classList.remove('active'));
        document.querySelectorAll('.tab-content').forEach(c => {
          c.classList.remove('active');
          c.style.display = 'none';
        });

        tab.classList.add('active');
        const targetEl = document.getElementById(targetTabId);
        if (targetEl) {
          targetEl.classList.add('active');
          targetEl.style.display = (targetTabId === 'tabLiveMonitor') ? 'grid' : 'flex';
        }
        if (targetTabId === 'tabViolationsDb') {
          populateFullDatabaseTable();
        }
      });
    });

    if (dom.btnRefreshDbTable) {
      dom.btnRefreshDbTable.addEventListener('click', populateFullDatabaseTable);
    }

    if (dom.filterPlateInput) {
      dom.filterPlateInput.addEventListener('input', (e) => {
        const query = e.target.value.toLowerCase();
        const rows = dom.dbTableBody.querySelectorAll('tr');
        rows.forEach(r => {
          r.style.display = r.textContent.toLowerCase().includes(query) ? '' : 'none';
        });
      });
    }
  }

  // ==========================================================================
  // Inspection Modal
  // ==========================================================================
  function initInspectModal() {
    const hideModal = () => { if (dom.inspectModal) dom.inspectModal.style.display = 'none'; };
    if (dom.btnCloseInspect) dom.btnCloseInspect.addEventListener('click', hideModal);
    if (dom.btnCloseInspectBtn) dom.btnCloseInspectBtn.addEventListener('click', hideModal);
    if (dom.inspectModal) {
      dom.inspectModal.addEventListener('click', (e) => {
        if (e.target === dom.inspectModal) hideModal();
      });
    }
  }

  function openInspectModal(v) {
    if (!v) return;

    dom.inspectRecordTitle.textContent = `Infraction #${v.id || v.track_id}`;
    dom.inspectVehImg.src = formatImgUrl(v.vehicle_image_url || v.vehicle_image_path);
    dom.inspectPlateImg.src = formatImgUrl(v.plate_image_url || v.plate_image_path);

    dom.inspectPlateBadge.innerHTML = renderPlateHTML(v.plate_text);

    const speed = parseFloat(v.speed_kmh) || 0.0;
    const limit = parseFloat(v.speed_limit) || 60.0;
    dom.inspectSpeed.textContent = `${speed.toFixed(1)} km/h`;
    dom.inspectLimit.textContent = `${limit.toFixed(0)} km/h`;
    dom.inspectExcess.textContent = `+${Math.max(0.0, speed - limit).toFixed(1)} km/h`;
    dom.inspectAccuracy.textContent = `${Math.round((v.ocr_confidence || 0) * 100)}%`;

    dom.inspectModal.style.display = 'flex';
  }

  // ==========================================================================
  // WebSocket Live Events
  // ==========================================================================
  function initWebSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws/live`;
    if (state.ws) { try { state.ws.close(); } catch (_) {} }

    state.ws = new WebSocket(wsUrl);
    state.ws.onopen = () => {
      if (dom.wsStatus) {
        dom.wsStatus.className = 'connection-status connected';
        dom.wsStatus.querySelector('.status-text').textContent = 'Live GPU Online';
      }
      if (state.wsReconnectTimer) {
        clearInterval(state.wsReconnectTimer);
        state.wsReconnectTimer = null;
      }
    };
    state.ws.onmessage = (event) => {
      try {
        const payload = JSON.parse(event.data);
        if (payload.type === 'NEW_VIOLATION') {
          handleIncomingViolation(payload.data);
          fetchStats();
        }
      } catch (err) {
        console.error(err);
      }
    };
    state.ws.onclose = () => {
      if (dom.wsStatus) {
        dom.wsStatus.className = 'connection-status disconnected';
        dom.wsStatus.querySelector('.status-text').textContent = 'Disconnected';
      }
      if (!state.wsReconnectTimer) state.wsReconnectTimer = setInterval(initWebSocket, 4000);
    };
  }

  // ==========================================================================
  // Video Upload
  // ==========================================================================
  function initUploadListeners() {
    const { dropZone, videoFileInput } = dom;
    if (!dropZone || !videoFileInput) return;

    ['dragenter', 'dragover'].forEach(name => {
      dropZone.addEventListener(name, (e) => { e.preventDefault(); dropZone.style.borderColor = 'var(--accent)'; });
    });
    ['dragleave', 'drop'].forEach(name => {
      dropZone.addEventListener(name, (e) => { e.preventDefault(); dropZone.style.borderColor = 'var(--border)'; });
    });

    dropZone.addEventListener('drop', (e) => {
      if (e.dataTransfer.files.length > 0) uploadVideo(e.dataTransfer.files[0]);
    });
    videoFileInput.addEventListener('change', (e) => {
      if (e.target.files.length > 0) uploadVideo(e.target.files[0]);
    });
  }

  function uploadVideo(file) {
    if (!file) return;
    const fd = new FormData();
    fd.append('file', file);

    dom.uploadProgressContainer.style.display = 'block';
    dom.uploadProgressBar.style.width = '0%';
    dom.videoMetaText.textContent = `Uploading ${file.name}...`;

    const xhr = new XMLHttpRequest();
    xhr.open('POST', '/api/upload', true);
    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable) dom.uploadProgressBar.style.width = `${Math.round((e.loaded / e.total) * 100)}%`;
    };
    xhr.onload = () => {
      dom.uploadProgressContainer.style.display = 'none';
      if (xhr.status >= 200 && xhr.status < 300) {
        const resp = JSON.parse(xhr.responseText);
        state.videoLoaded = true;
        state.currentVideoFile = resp.filename;
        dom.videoMetaText.textContent = `${resp.filename} | ${resp.fps} FPS`;
        dom.btnStartStream.disabled = false;
        dom.streamOverlayPlaceholder.style.display = 'none';
        dom.liveStreamFeed.src = `/api/calibration/reference-frame?t=${Date.now()}`;
        loadCalibrationReference();
      } else {
        alert('Upload failed: ' + xhr.status);
      }
    };
    xhr.send(fd);
  }

  // ==========================================================================
  // Video Rotation
  // ==========================================================================
  function initRotationControls() {
    dom.anglePills.forEach(pill => {
      pill.addEventListener('click', async () => {
        const angle = parseInt(pill.getAttribute('data-angle'), 10);
        state.currentAngle = angle;
        dom.anglePills.forEach(p => p.classList.remove('active'));
        pill.classList.add('active');
        if (dom.currentAngleBadge) dom.currentAngleBadge.textContent = `${angle}°`;

        try {
          await fetch('/api/video/rotate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ angle: angle })
          });
          const bust = Date.now();
          if (state.isStreaming) {
            dom.liveStreamFeed.src = `/api/stream/video?t=${bust}`;
          } else if (state.videoLoaded) {
            dom.liveStreamFeed.src = `/api/calibration/reference-frame?t=${bust}`;
          }

          state.calibPoints = [];
          state.customGateLine = [];
          loadCalibrationReference();
        } catch (err) {
          console.error(err);
        }
      });
    });
  }

  // ==========================================================================
  // Stream Controls
  // ==========================================================================
  function initStreamControls() {
    dom.btnStartStream.addEventListener('click', () => {
      if (!state.videoLoaded) return;
      state.isStreaming = true;
      dom.btnStartStream.disabled = true;
      dom.btnStopStream.disabled = false;
      dom.streamOverlayPlaceholder.style.display = 'none';
      dom.liveStreamFeed.src = `/api/stream/video?t=${Date.now()}`;
    });

    dom.btnStopStream.addEventListener('click', async () => {
      try { await fetch('/api/stream/stop', { method: 'POST' }); } catch (_) {}
      state.isStreaming = false;
      dom.btnStartStream.disabled = false;
      dom.btnStopStream.disabled = true;
      dom.liveStreamFeed.src = `/api/calibration/reference-frame?t=${Date.now()}`;
    });

    dom.btnSetLimit.addEventListener('click', async () => {
      const val = parseFloat(dom.inputSpeedLimit.value);
      if (val > 0) {
        await fetch('/api/config/threshold', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ speed_kmh: val })
        });
      }
    });
  }

  // ==========================================================================
  // Clean License Plate Formatter: ۵۷ - ۴۲۵ ص - ۶۷ IR (No Cartoon Styling)
  // ==========================================================================
  function parseIranianPlate(rawText) {
    if (!rawText || rawText === 'UNKNOWN') {
      return { part1: '--', letter: '?', part2: '---', prov: '--', raw: 'UNKNOWN' };
    }

    let s = String(rawText)
      .replace(/[۰٠]/g, '0').replace(/[۱١]/g, '1').replace(/[۲٢]/g, '2')
      .replace(/[۳٣]/g, '3').replace(/[۴٤]/g, '4').replace(/[۵٥]/g, '5')
      .replace(/[۶٦]/g, '6').replace(/[۷٧]/g, '7').replace(/[۸٨]/g, '8')
      .replace(/[۹٩]/g, '9');

    s = s.replace(/[\u200B-\u200F\u202A-\u202E\uFEFF\[\]\(\)\-_]/g, ' ').trim();

    const digits = s.match(/\d+/g) || [];
    const letters = s.match(/[^\d\s]+/g) || [];
    const validLetters = letters.filter(l => l !== 'ایران' && l !== 'IR' && l !== 'ir' && l !== 'iran');
    const letter = validLetters.length > 0 ? validLetters[0] : '';

    let p1 = '--', p2 = '---', prov = '--';

    if (digits.length >= 3) {
      p1 = digits[0].slice(0, 2);
      p2 = digits[1].slice(0, 3);
      prov = digits[2].slice(0, 2);
    } else if (digits.length === 1 && digits[0].length >= 7) {
      const d = digits[0];
      p1 = d.slice(0, 2);
      p2 = d.slice(2, 5);
      prov = d.slice(5, 7);
    } else if (digits.length === 2) {
      p1 = digits[0].slice(0, 2);
      p2 = digits[1].slice(0, 3);
      prov = digits[1].length >= 5 ? digits[1].slice(3, 5) : '--';
    }

    return { part1: p1, letter: letter, part2: p2, prov: prov, raw: rawText };
  }

  function renderPlateHTML(rawText) {
    const p = parseIranianPlate(rawText);

    if (!p.letter && p.part1 === '--') {
      return `<div class="clean-plate-badge single-text">${toPersianDigits(rawText)}</div>`;
    }

    // Clean, crisp presentation: [ ۵۷ ] - [ ۴۲۵ ص ] - [ ۶۷ IR ]
    return `
      <div class="clean-plate-badge" title="${p.raw}" dir="ltr">
        <span class="plate-seg plate-seg-num">${toPersianDigits(p.part1)}</span>
        <span class="plate-divider">-</span>
        <span class="plate-seg">
          <span class="plate-seg-num">${toPersianDigits(p.part2)}</span>
          <span class="plate-seg-char">${p.letter}</span>
        </span>
        <span class="plate-divider">-</span>
        <span class="plate-seg plate-seg-prov">
          <span class="plate-seg-num">${toPersianDigits(p.prov)}</span>
          <span class="plate-ir-tag">IR</span>
        </span>
      </div>
    `;
  }

  // ==========================================================================
  // Live Violations Feed
  // ==========================================================================
  function handleIncomingViolation(v) {
    if (dom.emptyViolationsState) dom.emptyViolationsState.style.display = 'none';

    const key = v.id || v.track_id;
    state.cachedViolations.set(String(key), v);

    const card = document.createElement('div');
    card.className = 'violation-card-item';

    const vehPath = formatImgUrl(v.vehicle_image_url || v.vehicle_image_path);
    const platePath = formatImgUrl(v.plate_image_url || v.plate_image_path);

    card.innerHTML = `
      <div class="violation-card-top">
        <div class="v-veh-crop-wrap">
          <img src="${vehPath}" class="v-crop-veh" onerror="this.src='/static/placeholder.png';">
        </div>
        <div class="v-plate-info">
          ${renderPlateHTML(v.plate_text)}
          <div class="v-crop-plate-wrap">
            <img src="${platePath}" class="v-crop-plate" onerror="this.src='/static/placeholder.png';">
          </div>
        </div>
      </div>
      <div class="violation-card-meta">
        <div class="meta-item">
          <span class="meta-lbl">SPEED</span>
          <span class="meta-val text-red">${parseFloat(v.speed_kmh).toFixed(1)} km/h</span>
        </div>
        <div class="meta-item">
          <span class="meta-lbl">LIMIT</span>
          <span class="meta-val">${parseFloat(v.speed_limit || 60).toFixed(0)} km/h</span>
        </div>
        <div class="meta-item">
          <span class="meta-lbl">TIME</span>
          <span class="meta-val">${v.video_time || '--:--'}</span>
        </div>
        <div class="meta-item">
          <button class="btn-inspect" data-key="${key}">Inspect</button>
        </div>
      </div>
    `;

    card.querySelector('.btn-inspect').addEventListener('click', (e) => {
      const k = e.currentTarget.getAttribute('data-key');
      openInspectModal(state.cachedViolations.get(k));
    });

    dom.violationsList.insertBefore(card, dom.violationsList.firstChild);

    const currentCount = dom.violationsList.querySelectorAll('.violation-card-item').length;
    if (dom.violationCountBadge) dom.violationCountBadge.textContent = currentCount;
    if (dom.tabViolationBadge) dom.tabViolationBadge.textContent = currentCount;
  }

  async function fetchViolations() {
    try {
      const res = await fetch('/api/violations?limit=50');
      if (!res.ok) return;
      const data = await res.json();
      const list = data.violations || [];
      dom.violationsList.innerHTML = '';

      if (list.length === 0) {
        if (dom.emptyViolationsState) {
          dom.violationsList.appendChild(dom.emptyViolationsState);
          dom.emptyViolationsState.style.display = 'block';
        }
        if (dom.violationCountBadge) dom.violationCountBadge.textContent = '0';
        if (dom.tabViolationBadge) dom.tabViolationBadge.textContent = '0';
        return;
      }

      if (dom.violationCountBadge) dom.violationCountBadge.textContent = list.length;
      if (dom.tabViolationBadge) dom.tabViolationBadge.textContent = list.length;
      list.slice().reverse().forEach(handleIncomingViolation);
    } catch (err) {
      console.error(err);
    }
  }

  // ==========================================================================
  // Full Violations Table (Tab 2)
  // ==========================================================================
  async function populateFullDatabaseTable() {
    if (!dom.dbTableBody) return;

    try {
      const res = await fetch('/api/violations?limit=250');
      if (!res.ok) return;
      const data = await res.json();
      const records = data.violations || [];

      dom.dbTableBody.innerHTML = '';
      if (dom.tableRecordCount) dom.tableRecordCount.textContent = `${records.length} records registered`;
      if (dom.tabViolationBadge) dom.tabViolationBadge.textContent = records.length;

      if (records.length === 0) {
        dom.dbTableBody.innerHTML = `<tr><td colspan="8" style="text-align: center; color: var(--text-muted); padding: 30px;">No violation records found in database.</td></tr>`;
        return;
      }

      records.forEach(r => {
        const key = r.id || r.track_id;
        state.cachedViolations.set(String(key), r);

        const tr = document.createElement('tr');
        const vehPath = formatImgUrl(r.vehicle_image_url || r.vehicle_image_path);
        const platePath = formatImgUrl(r.plate_image_url || r.plate_image_path);

        tr.innerHTML = `
          <td><strong>#${key}</strong></td>
          <td><img src="${vehPath}" class="table-thumb-veh" onerror="this.src='/static/placeholder.png';"></td>
          <td>${renderPlateHTML(r.plate_text)}</td>
          <td><img src="${platePath}" class="table-thumb-plate" onerror="this.src='/static/placeholder.png';"></td>
          <td><strong class="text-red">${parseFloat(r.speed_kmh).toFixed(1)} km/h</strong></td>
          <td>${parseFloat(r.speed_limit || 60).toFixed(0)} km/h</td>
          <td>${r.video_time || r.timestamp || '--:--'}</td>
          <td><button class="btn-inspect" data-key="${key}">Inspect</button></td>
        `;

        tr.querySelector('.btn-inspect').addEventListener('click', (e) => {
          const k = e.currentTarget.getAttribute('data-key');
          openInspectModal(state.cachedViolations.get(k));
        });

        dom.dbTableBody.appendChild(tr);
      });
    } catch (err) {
      console.error(err);
    }
  }

  // ==========================================================================
  // Metrics & Config
  // ==========================================================================
  async function fetchStats() {
    try {
      const res = await fetch('/api/stats');
      if (!res.ok) return;
      const stats = await res.json();
      if (dom.statTotalViolations) dom.statTotalViolations.textContent = stats.total_violations || 0;
      if (dom.statMaxSpeed) dom.statMaxSpeed.innerHTML = `${(stats.max_speed || 0).toFixed(1)} <small>km/h</small>`;
      if (dom.statAvgSpeed) dom.statAvgSpeed.innerHTML = `${(stats.avg_speed || 0).toFixed(1)} <small>km/h</small>`;
    } catch (_) {}
  }

  async function fetchConfig() {
    try {
      const res = await fetch('/api/config');
      if (!res.ok) return;
      const conf = await res.json();
      if (conf.thresholds && conf.thresholds.speed_kmh) {
        dom.inputSpeedLimit.value = conf.thresholds.speed_kmh;
      }
    } catch (_) {}
  }

  // ==========================================================================
  // Interactive Calibration Studio (Drag & Drop + 2 Explicit Gate Options)
  // ==========================================================================
  function getAccurateMousePos(e) {
    const canvas = dom.calibrationCanvas;
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;

    return {
      x: Math.round((e.clientX - rect.left) * scaleX),
      y: Math.round((e.clientY - rect.top) * scaleY)
    };
  }

  function findNearbyPoint(pos) {
    const canvas = dom.calibrationCanvas;
    const hitRadius = Math.max(30, Math.round(canvas.width * 0.035));

    // Check custom gate line first if active
    if (state.gateOption === 'custom') {
      for (let i = 0; i < state.customGateLine.length; i++) {
        const p = state.customGateLine[i];
        if (Math.hypot(p.x - pos.x, p.y - pos.y) <= hitRadius) {
          return { type: 'gate', index: i };
        }
      }
    }

    // Check road points
    for (let i = 0; i < state.calibPoints.length; i++) {
      const p = state.calibPoints[i];
      if (Math.hypot(p.x - pos.x, p.y - pos.y) <= hitRadius) {
        return { type: 'road', index: i };
      }
    }

    return null;
  }

  function initCalibrationStudio() {
    const { btnOpenCalibration, btnCloseCalibration, calibrationModal, calibrationCanvas } = dom;

    btnOpenCalibration.addEventListener('click', () => {
      calibrationModal.style.display = 'flex';
      loadCalibrationReference();
      loadCurrentCalibrationData();
    });

    btnCloseCalibration.addEventListener('click', () => {
      calibrationModal.style.display = 'none';
    });

    calibrationModal.addEventListener('click', (e) => {
      if (e.target === calibrationModal) calibrationModal.style.display = 'none';
    });

    // 2 Explicit Gate Mode Options
    dom.optCardDefault.addEventListener('click', () => {
      state.gateOption = 'default';
      dom.radioDefaultGate.checked = true;
      dom.optCardDefault.classList.add('selected');
      dom.optCardCustom.classList.remove('selected');
      dom.calibStatusGuide.textContent = 'Mode: Default Gate (P4-P3). Click 4 corners (TL, TR, BR, BL). Drag points to adjust.';
      drawCanvas();
    });

    dom.optCardCustom.addEventListener('click', () => {
      state.gateOption = 'custom';
      dom.radioCustomGate.checked = true;
      dom.optCardCustom.classList.add('selected');
      dom.optCardDefault.classList.remove('selected');
      dom.calibStatusGuide.textContent = 'Mode: Custom Gate. Click 2 points across the lane to set trigger line (G1 to G2). Drag points to adjust.';
      drawCanvas();
    });

    dom.btnResetAllCalib.addEventListener('click', () => {
      if (state.gateOption === 'custom' && state.customGateLine.length > 0) {
        state.customGateLine = [];
      } else {
        state.calibPoints = [];
      }
      drawCanvas();
    });

    dom.btnSaveCalibration.addEventListener('click', async () => {
      if (state.calibPoints.length !== 4) {
        return alert('Please place all 4 road corners first (P1, P2, P3, P4).');
      }

      let gate = null;
      if (state.gateOption === 'custom' && state.customGateLine.length === 2) {
        gate = [
          [state.customGateLine[0].x, state.customGateLine[0].y],
          [state.customGateLine[1].x, state.customGateLine[1].y]
        ];
      } else {
        // Option 1: Default between P4 and P3
        gate = [
          [state.calibPoints[3].x, state.calibPoints[3].y],
          [state.calibPoints[2].x, state.calibPoints[2].y]
        ];
      }

      const payload = {
        pixel_points: state.calibPoints.map(p => [p.x, p.y]),
        custom_gate_line: gate,
        road_width_m: parseFloat(dom.calibRoadWidth.value) || 3.5,
        road_length_m: parseFloat(dom.calibRoadLength.value) || 20.0
      };

      try {
        const res = await fetch('/api/calibration/save', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });
        if (res.ok) {
          alert('Radar Calibration & Enforcement Gate successfully updated!');
          calibrationModal.style.display = 'none';
        } else {
          alert('Failed to save calibration.');
        }
      } catch (err) {
        console.error(err);
      }
    });

    // Mouse Interactions (Click + Drag & Drop)
    calibrationCanvas.addEventListener('mousedown', (e) => {
      const pos = getAccurateMousePos(e);
      const nearby = findNearbyPoint(pos);

      if (nearby) {
        state.dragTarget = nearby;
        calibrationCanvas.style.cursor = 'grabbing';
        return;
      }

      // If clicking on empty space, place new points
      if (state.gateOption === 'default' || state.calibPoints.length < 4) {
        if (state.calibPoints.length < 4) {
          state.calibPoints.push(pos);
          drawCanvas();
        }
      } else if (state.gateOption === 'custom') {
        if (state.customGateLine.length < 2) {
          state.customGateLine.push(pos);
        } else {
          state.customGateLine = [pos];
        }
        drawCanvas();
      }
    });

    window.addEventListener('mousemove', (e) => {
      if (state.dragTarget) {
        const pos = getAccurateMousePos(e);
        if (state.dragTarget.type === 'road') {
          state.calibPoints[state.dragTarget.index] = pos;
        } else if (state.dragTarget.type === 'gate') {
          state.customGateLine[state.dragTarget.index] = pos;
        }
        drawCanvas();
      } else if (calibrationModal.style.display === 'flex') {
        const pos = getAccurateMousePos(e);
        calibrationCanvas.style.cursor = findNearbyPoint(pos) ? 'grab' : 'crosshair';
      }
    });

    window.addEventListener('mouseup', () => {
      if (state.dragTarget) {
        state.dragTarget = null;
        calibrationCanvas.style.cursor = 'crosshair';
      }
    });
  }

  function loadCalibrationReference() {
    state.calibImg.src = `/api/calibration/reference-frame?t=${Date.now()}`;
    state.calibImg.onload = () => {
      const canvas = dom.calibrationCanvas;
      canvas.width = state.calibImg.naturalWidth;
      canvas.height = state.calibImg.naturalHeight;
      drawCanvas();
    };
  }

  async function loadCurrentCalibrationData() {
    try {
      const res = await fetch('/api/calibration/current');
      if (!res.ok) return;
      const data = await res.json();

      if (data.calibration_points && data.calibration_points.length === 4) {
        state.calibPoints = data.calibration_points.map(pt => ({ x: pt[0], y: pt[1] }));
      }
      if (data.custom_gate_line && data.custom_gate_line.length === 2) {
        state.customGateLine = data.custom_gate_line.map(pt => ({ x: pt[0], y: pt[1] }));
        state.gateOption = 'custom';
        dom.radioCustomGate.checked = true;
        dom.optCardCustom.classList.add('selected');
        dom.optCardDefault.classList.remove('selected');
      }
      drawCanvas();
    } catch (_) {}
  }

  function drawCanvas() {
    const canvas = dom.calibrationCanvas;
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (state.calibImg.complete) {
      ctx.drawImage(state.calibImg, 0, 0, canvas.width, canvas.height);
    }

    const refDim = Math.max(canvas.width, canvas.height);
    const nodeR = Math.max(10, Math.round(refDim * 0.012));
    const lineW = Math.max(3, Math.round(refDim * 0.0035));
    const fontSz = Math.max(14, Math.round(refDim * 0.016));

    // 1. Draw 4-point Road Box
    const pts = state.calibPoints;
    if (pts.length >= 2) {
      ctx.beginPath();
      ctx.moveTo(pts[0].x, pts[0].y);
      for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i].x, pts[i].y);
      if (pts.length === 4) {
        ctx.closePath();
        ctx.fillStyle = 'rgba(88, 166, 255, 0.2)';
        ctx.fill();
      }
      ctx.strokeStyle = '#58a6ff';
      ctx.lineWidth = lineW;
      ctx.stroke();
    }

    pts.forEach((p, i) => {
      ctx.beginPath();
      ctx.arc(p.x, p.y, nodeR, 0, 2 * Math.PI);
      ctx.fillStyle = '#58a6ff';
      ctx.fill();
      ctx.lineWidth = 2;
      ctx.strokeStyle = '#ffffff';
      ctx.stroke();

      ctx.fillStyle = '#ffffff';
      ctx.font = `bold ${fontSz}px -apple-system, sans-serif`;
      ctx.fillText(`P${i + 1}`, p.x + nodeR + 4, p.y - nodeR);
    });

    // 2. Draw Trigger Gate (Default P4-P3 or Custom G1-G2)
    let gate = null;
    let label = 'TRIGGER GATE (P4-P3)';

    if (state.gateOption === 'custom' && state.customGateLine.length === 2) {
      gate = state.customGateLine;
      label = 'CUSTOM TRIGGER GATE';
    } else if (pts.length === 4) {
      gate = [pts[3], pts[2]];
    }

    if (gate && gate.length === 2) {
      ctx.beginPath();
      ctx.moveTo(gate[0].x, gate[0].y);
      ctx.lineTo(gate[1].x, gate[1].y);
      ctx.strokeStyle = '#ff007f';
      ctx.lineWidth = lineW * 1.6;
      ctx.stroke();

      if (state.gateOption === 'custom') {
        gate.forEach((gp, idx) => {
          ctx.beginPath();
          ctx.arc(gp.x, gp.y, nodeR * 1.1, 0, 2 * Math.PI);
          ctx.fillStyle = '#ff007f';
          ctx.fill();
          ctx.lineWidth = 2;
          ctx.strokeStyle = '#ffffff';
          ctx.stroke();

          ctx.fillStyle = '#ff007f';
          ctx.font = `bold ${fontSz}px -apple-system, sans-serif`;
          ctx.fillText(`G${idx + 1}`, gp.x + nodeR + 4, gp.y + nodeR + 10);
        });
      }

      const midX = (gate[0].x + gate[1].x) / 2;
      const midY = (gate[0].y + gate[1].y) / 2;
      ctx.fillStyle = '#ff007f';
      ctx.font = `bold ${fontSz}px -apple-system, sans-serif`;
      ctx.fillText(label, midX - 60, midY - 10);
    }
  }

})();