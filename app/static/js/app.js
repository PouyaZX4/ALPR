/* ==========================================================================
   Persian Speed ALPR & Radar Control - Robust Client Script
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
    calibImg: new Image(),
    calibPoints: [],
    activeDragIdx: -1,
    scaleX: 1.0,
    scaleY: 1.0
  };

  const PERSIAN_DIGITS = ['۰', '۱', '۲', '۳', '۴', '۵', '۶', '۷', '۸', '۹'];
  const toPersianDigits = (str) => String(str).replace(/[0-9]/g, (d) => PERSIAN_DIGITS[parseInt(d, 10)]);

  const dom = {
    wsStatus: document.getElementById('wsStatus'),
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
    btnOpenCalibration: document.getElementById('btnOpenCalibration'),
    calibrationModal: document.getElementById('calibrationModal'),
    btnCloseCalibration: document.getElementById('btnCloseCalibration'),
    calibrationCanvas: document.getElementById('calibrationCanvas'),
    calibRoadWidth: document.getElementById('calibRoadWidth'),
    calibRoadLength: document.getElementById('calibRoadLength'),
    btnResetCalibPoints: document.getElementById('btnResetCalibPoints'),
    btnSaveCalibration: document.getElementById('btnSaveCalibration')
  };

  document.addEventListener('DOMContentLoaded', () => {
    initWebSocket();
    initUploadListeners();
    initRotationControls();
    initStreamControls();
    initCalibrationCanvas();
    fetchStats();
    fetchViolations();
    fetchConfig();
  });

  function initWebSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws/live`;

    if (state.ws) {
      try { state.ws.close(); } catch (_) {}
    }

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
      if (!state.wsReconnectTimer) {
        state.wsReconnectTimer = setInterval(initWebSocket, 4000);
      }
    };
  }

  function initUploadListeners() {
    const { dropZone, videoFileInput } = dom;
    if (!dropZone || !videoFileInput) return;

    ['dragenter', 'dragover'].forEach(name => {
      dropZone.addEventListener(name, (e) => {
        e.preventDefault();
        dropZone.style.borderColor = 'var(--accent)';
      });
    });

    ['dragleave', 'drop'].forEach(name => {
      dropZone.addEventListener(name, (e) => {
        e.preventDefault();
        dropZone.style.borderColor = 'var(--border)';
      });
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
      if (e.lengthComputable) {
        dom.uploadProgressBar.style.width = `${Math.round((e.loaded / e.total) * 100)}%`;
      }
    };

    xhr.onload = () => {
      dom.uploadProgressContainer.style.display = 'none';
      if (xhr.status >= 200 && xhr.status < 300) {
        const resp = JSON.parse(xhr.responseText);
        state.videoLoaded = true;
        state.currentVideoFile = resp.filename;
        dom.videoMetaText.textContent = `${resp.filename} | ${resp.fps} FPS`;
        dom.btnStartStream.disabled = false;
        
        // Show initial frame
        dom.streamOverlayPlaceholder.style.display = 'none';
        dom.liveStreamFeed.src = `/api/calibration/reference-frame?t=${Date.now()}`;
        loadCalibrationReference();
      } else {
        alert('Upload failed: ' + xhr.status);
      }
    };

    xhr.send(fd);
  }

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
            // Re-point live stream to reload immediately with new rotation
            dom.liveStreamFeed.src = `/api/stream/video?t=${bust}`;
          } else if (state.videoLoaded) {
            // If idle, refresh reference frame preview
            dom.liveStreamFeed.src = `/api/calibration/reference-frame?t=${bust}`;
          }
          loadCalibrationReference();
        } catch (err) {
          console.error(err);
        }
      });
    });
  }

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
      try {
        await fetch('/api/stream/stop', { method: 'POST' });
      } catch (_) {}
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

  function parsePlate(raw) {
    if (!raw || raw === 'UNKNOWN') return { p1: '---', l: '?', p2: '---', pr: '--' };
    const m = raw.trim().match(/^(\d{2})\s*([^\d\s]+)\s*(\d{3})\s*[-_]?\s*(\d{2})$/);
    if (m) return { p1: toPersianDigits(m[1]), l: m[2], p2: toPersianDigits(m[3]), pr: toPersianDigits(m[4]) };
    return { p1: toPersianDigits(raw), l: '', p2: '', pr: 'ایران' };
  }

  function handleIncomingViolation(v) {
    if (dom.emptyViolationsState) dom.emptyViolationsState.style.display = 'none';

    const p = parsePlate(v.plate_text);
    const card = document.createElement('div');
    card.className = 'violation-card-item';
    card.innerHTML = `
      <div class="violation-card-top">
        <div class="v-veh-crop-wrap">
          <img src="${v.vehicle_image_url || ''}" class="v-crop-veh" onerror="this.style.display='none';">
        </div>
        <div class="v-plate-info">
          <div class="plate-box-iranian" dir="rtl">
            <div class="plate-blue-strip"><span>IRAN</span></div>
            <div class="plate-main-text">
              <span>${p.p1}</span>
              <span class="plate-letter">${p.l}</span>
              <span>${p.p2}</span>
            </div>
            <div class="plate-province-zone"><span>${p.pr}</span></div>
          </div>
          <div class="v-crop-plate-wrap">
            <img src="${v.plate_image_url || ''}" class="v-crop-plate" onerror="this.style.display='none';">
          </div>
        </div>
      </div>
      <div class="violation-card-meta">
        <div class="meta-item"><span class="meta-lbl">SPEED</span><span class="meta-val text-red">${parseFloat(v.speed_kmh).toFixed(1)} km/h</span></div>
        <div class="meta-item"><span class="meta-lbl">LIMIT</span><span class="meta-val">${parseFloat(v.speed_limit || 60).toFixed(0)} km/h</span></div>
        <div class="meta-item"><span class="meta-lbl">TIME</span><span class="meta-val">${v.video_time || '--:--'}</span></div>
        <div class="meta-item"><span class="meta-lbl">ACC</span><span class="meta-val">${Math.round((v.ocr_confidence || 0) * 100)}%</span></div>
      </div>
    `;
    dom.violationsList.insertBefore(card, dom.violationsList.firstChild);
  }

  async function fetchViolations() {
    try {
      const res = await fetch('/api/violations?limit=50');
      const data = await res.json();
      const list = data.violations || [];
      dom.violationsList.innerHTML = '';
      if (list.length === 0) {
        if (dom.emptyViolationsState) {
          dom.violationsList.appendChild(dom.emptyViolationsState);
          dom.emptyViolationsState.style.display = 'block';
        }
        return;
      }
      if (dom.violationCountBadge) dom.violationCountBadge.textContent = list.length;
      list.slice().reverse().forEach(handleIncomingViolation);
    } catch (_) {}
  }

  async function fetchStats() {
    try {
      const res = await fetch('/api/stats');
      const stats = await res.json();
      dom.statTotalViolations.textContent = stats.total_violations || 0;
      dom.statMaxSpeed.innerHTML = `${(stats.max_speed || 0).toFixed(1)} <small>km/h</small>`;
      dom.statAvgSpeed.innerHTML = `${(stats.avg_speed || 0).toFixed(1)} <small>km/h</small>`;
    } catch (_) {}
  }

  async function fetchConfig() {
    try {
      const res = await fetch('/api/config');
      const conf = await res.json();
      if (conf.thresholds?.speed_kmh) dom.inputSpeedLimit.value = conf.thresholds.speed_kmh;
    } catch (_) {}
  }

  function initCalibrationCanvas() {
    const { btnOpenCalibration, btnCloseCalibration, calibrationModal, calibrationCanvas } = dom;

    btnOpenCalibration.addEventListener('click', () => {
      calibrationModal.style.display = 'flex';
      loadCalibrationReference();
    });

    btnCloseCalibration.addEventListener('click', () => {
      calibrationModal.style.display = 'none';
    });

    dom.btnResetCalibPoints.addEventListener('click', () => {
      state.calibPoints = [];
      drawCanvas();
    });

    dom.btnSaveCalibration.addEventListener('click', async () => {
      if (state.calibPoints.length !== 4) return alert('Select 4 points');
      const pts = state.calibPoints.map(p => [Math.round(p.x / state.scaleX), Math.round(p.y / state.scaleY)]);
      await fetch('/api/calibration/save', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          pixel_points: pts,
          custom_gate_line: [pts[3], pts[2]],
          road_width_m: parseFloat(dom.calibRoadWidth.value) || 3.5,
          road_length_m: parseFloat(dom.calibRoadLength.value) || 20.0
        })
      });
      calibrationModal.style.display = 'none';
    });

    calibrationCanvas.addEventListener('mousedown', (e) => {
      const rect = calibrationCanvas.getBoundingClientRect();
      const pos = { x: e.clientX - rect.left, y: e.clientY - rect.top };
      if (state.calibPoints.length < 4) {
        state.calibPoints.push(pos);
        drawCanvas();
      }
    });
  }

  function loadCalibrationReference() {
    state.calibImg.src = `/api/calibration/reference-frame?t=${Date.now()}`;
    state.calibImg.onload = () => {
      const canvas = dom.calibrationCanvas;
      const wrap = canvas.parentElement;
      canvas.width = wrap.clientWidth || 600;
      canvas.height = (canvas.width * state.calibImg.height) / state.calibImg.width;
      state.scaleX = canvas.width / state.calibImg.width;
      state.scaleY = canvas.height / state.calibImg.height;
      drawCanvas();
    };
  }

  function drawCanvas() {
    const ctx = dom.calibrationCanvas.getContext('2d');
    ctx.clearRect(0, 0, dom.calibrationCanvas.width, dom.calibrationCanvas.height);
    if (state.calibImg.complete) ctx.drawImage(state.calibImg, 0, 0, dom.calibrationCanvas.width, dom.calibrationCanvas.height);

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
      ctx.lineWidth = 2;
      ctx.stroke();
    }

    pts.forEach((p, i) => {
      ctx.beginPath();
      ctx.arc(p.x, p.y, 6, 0, 2 * Math.PI);
      ctx.fillStyle = i >= 2 ? '#f85149' : '#58a6ff';
      ctx.fill();
    });
  }

})();