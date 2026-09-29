document.addEventListener("DOMContentLoaded", () => {
    const API_BASE = "";
    const WS_URL = `${window.location.protocol === "https:" ? "wss:" : "ws:"}//${window.location.host}/ws/live`;

    // DOM Navigation
    const navItems = document.querySelectorAll(".nav-item");
    const tabPanes = document.querySelectorAll(".tab-pane");

    // Metrics
    const valTotalViolations = document.getElementById("valTotalViolations");
    const valMaxSpeed = document.getElementById("valMaxSpeed");
    const valAvgSpeed = document.getElementById("valAvgSpeed");
    const valSpeedLimit = document.getElementById("valSpeedLimit");

    // Stream Controls
    const videoFileInput = document.getElementById("videoFileInput");
    const uploadBtnText = document.getElementById("uploadBtnText");
    const uploadLabel = document.getElementById("uploadLabel");
    const btnStartStream = document.getElementById("btnStartStream");
    const btnStopStream = document.getElementById("btnStopStream");
    const liveStreamImg = document.getElementById("liveStreamImg");
    const streamPlaceholder = document.getElementById("streamPlaceholder");
    const streamBadge = document.getElementById("streamBadge");

    const thresholdRange = document.getElementById("thresholdRange");
    const thresholdVal = document.getElementById("thresholdVal");

    // Live Feed & History Grids
    const liveViolationsGrid = document.getElementById("liveViolationsGrid");
    const noViolationsMsg = document.getElementById("noViolationsMsg");
    const historyViolationsGrid = document.getElementById("historyViolationsGrid");
    const violationsTableBody = document.getElementById("violationsTableBody");
    const tableFilter = document.getElementById("tableFilter");

    // Status
    const wsStatusDot = document.getElementById("wsStatusDot");
    const wsStatusText = document.getElementById("wsStatusText");

    // Calibration
    const calibrationCanvas = document.getElementById("calibrationCanvas");
    const ctx = calibrationCanvas.getContext("2d");
    const canvasHint = document.getElementById("canvasHint");
    const btnResetPoints = document.getElementById("btnResetPoints");
    const btnSaveCalibration = document.getElementById("btnSaveCalibration");
    const roadWidthInput = document.getElementById("roadWidthInput");
    const roadLengthInput = document.getElementById("roadLengthInput");

    const ptBadges = [
        document.getElementById("ptBadge1"),
        document.getElementById("ptBadge2"),
        document.getElementById("ptBadge3"),
        document.getElementById("ptBadge4")
    ];

    // Modal
    const imageModal = document.getElementById("imageModal");
    const modalClose = document.getElementById("modalClose");
    const modalVehImage = document.getElementById("modalVehImage");
    const modalPlateImage = document.getElementById("modalPlateImage");
    const modalInfo = document.getElementById("modalInfo");

    let calibrationPoints = [];
    let refImage = new Image();
    let isRefImageLoaded = false;
    let ws = null;
    let allViolationsCache = [];

    // ---------------- TAB NAVIGATION ----------------
    function switchTab(tabId) {
        navItems.forEach(n => n.classList.remove("active"));
        tabPanes.forEach(p => p.classList.remove("active"));

        const targetBtn = Array.from(navItems).find(n => n.getAttribute("data-tab") === tabId);
        const targetPane = document.getElementById(tabId);

        if (targetBtn) targetBtn.classList.add("active");
        if (targetPane) targetPane.classList.add("active");

        if (tabId === "calibration-tab") {
            loadReferenceFrame();
        } else if (tabId === "history-tab") {
            fetchViolations();
        }
    }

    navItems.forEach(item => {
        item.addEventListener("click", () => {
            switchTab(item.getAttribute("data-tab"));
        });
    });

    // ---------------- WEBSOCKET ----------------
    function setupWebSocket() {
        try {
            ws = new WebSocket(WS_URL);

            ws.onopen = () => {
                if (wsStatusDot) wsStatusDot.className = "status-indicator online";
                if (wsStatusText) wsStatusText.textContent = "Live Connected";
            };

            ws.onmessage = (event) => {
                try {
                    const message = JSON.parse(event.data);
                    if (message.type === "NEW_VIOLATION") {
                        handleNewViolation(message.data);
                    }
                } catch (err) {
                    console.error("WS parse error:", err);
                }
            };

            ws.onclose = () => {
                if (wsStatusDot) wsStatusDot.className = "status-indicator offline";
                if (wsStatusText) wsStatusText.textContent = "Disconnected (Retrying)";
                setTimeout(setupWebSocket, 3000);
            };

            ws.onerror = () => {
                if (ws) ws.close();
            };
        } catch (e) {
            console.error("WebSocket setup error:", e);
        }
    }

    // ---------------- DIRECT FILE UPLOAD HANDLER ----------------
    if (videoFileInput) {
        videoFileInput.addEventListener("change", function () {
            if (!this.files || this.files.length === 0) return;

            const file = this.files[0];
            const fileSizeMB = (file.size / (1024 * 1024)).toFixed(1);

            uploadLabel.style.pointerEvents = "none";
            if (uploadBtnText) uploadBtnText.textContent = `Uploading: 0% (${fileSizeMB} MB)...`;

            const formData = new FormData();
            formData.append("file", file);

            const xhr = new XMLHttpRequest();
            xhr.open("POST", `${API_BASE}/api/upload`, true);
            xhr.timeout = 900000; // 15 mins for large 300MB-1GB files

            xhr.upload.onprogress = (event) => {
                if (event.lengthComputable) {
                    const percent = Math.round((event.loaded / event.total) * 100);
                    const loadedMB = (event.loaded / (1024 * 1024)).toFixed(1);
                    if (uploadBtnText) uploadBtnText.textContent = `⏳ ${percent}% (${loadedMB}/${fileSizeMB} MB)`;
                }
            };

            xhr.onload = () => {
                uploadLabel.style.pointerEvents = "auto";
                if (uploadBtnText) uploadBtnText.textContent = "Upload Video";

                if (xhr.status === 200) {
                    try {
                        const data = JSON.parse(xhr.responseText);
                        alert(`✅ Video Uploaded Successfully!\n\nFile: ${data.filename}\nFPS: ${data.fps} | Frames: ${data.total_frames}\n\nYou can now set road dots in Road Calibration or click 'Start Live Stream'.`);
                        btnStartStream.disabled = false;
                        btnStartStream.classList.add("btn-primary");
                    } catch (err) {
                        alert("Uploaded, but server returned invalid response format.");
                    }
                } else {
                    alert(`❌ Upload failed with status ${xhr.status}: ${xhr.statusText}`);
                }
            };

            xhr.onerror = () => {
                uploadLabel.style.pointerEvents = "auto";
                if (uploadBtnText) uploadBtnText.textContent = "Upload Video";
                alert("❌ Upload failed due to a network connection error.");
            };

            xhr.ontimeout = () => {
                uploadLabel.style.pointerEvents = "auto";
                if (uploadBtnText) uploadBtnText.textContent = "Upload Video";
                alert("❌ Upload timed out.");
            };

            xhr.send(formData);
        });
    }

    // ---------------- STREAM CONTROLS (ANTI-CACHE FIX) ----------------
    btnStartStream.addEventListener("click", () => {
        streamPlaceholder.classList.add("hidden");
        liveStreamImg.classList.remove("hidden");

        // Cache-busting query parameter forces fresh MJPEG socket
        const streamUrl = `${API_BASE}/api/stream/video?t=${Date.now()}`;
        liveStreamImg.src = streamUrl;

        streamBadge.textContent = "Streaming Live";
        streamBadge.className = "badge badge-alert";

        btnStartStream.disabled = true;
        btnStopStream.disabled = false;
    });

    btnStopStream.addEventListener("click", async () => {
        try {
            await fetch(`${API_BASE}/api/stream/stop`, { method: "POST" });
        } catch (e) {}

        // Drop the image stream source to terminate the client connection cleanly
        liveStreamImg.src = "";
        liveStreamImg.classList.add("hidden");
        streamPlaceholder.classList.remove("hidden");
        streamBadge.textContent = "Stream Stopped";
        streamBadge.className = "badge";

        btnStartStream.disabled = false;
        btnStopStream.disabled = true;

        fetchStats();
        fetchViolations();
    });

    // ---------------- ROAD CALIBRATION ENGINE ----------------
    function loadReferenceFrame() {
        refImage = new Image();
        refImage.src = `${API_BASE}/api/calibration/reference-frame?t=${Date.now()}`;
        refImage.onload = () => {
            isRefImageLoaded = true;
            if (canvasHint) canvasHint.classList.add("hidden");
            renderCalibrationCanvas();
        };
        refImage.onerror = () => {
            if (canvasHint) {
                canvasHint.classList.remove("hidden");
                canvasHint.textContent = "Upload a video in the Live Operations tab to load the reference road frame.";
            }
        };
    }

    function renderCalibrationCanvas() {
        if (!isRefImageLoaded) return;

        calibrationCanvas.width = refImage.naturalWidth || 1280;
        calibrationCanvas.height = refImage.naturalHeight || 720;

        ctx.clearRect(0, 0, calibrationCanvas.width, calibrationCanvas.height);
        ctx.drawImage(refImage, 0, 0, calibrationCanvas.width, calibrationCanvas.height);

        // Draw connecting polygon
        if (calibrationPoints.length > 1) {
            ctx.beginPath();
            ctx.moveTo(calibrationPoints[0].x, calibrationPoints[0].y);
            for (let i = 1; i < calibrationPoints.length; i++) {
                ctx.lineTo(calibrationPoints[i].x, calibrationPoints[i].y);
            }
            if (calibrationPoints.length === 4) {
                ctx.closePath();
                ctx.fillStyle = "rgba(0, 229, 255, 0.25)";
                ctx.fill();
            }
            ctx.strokeStyle = "#00e5ff";
            ctx.lineWidth = 3;
            ctx.stroke();
        }

        // Draw marker dots
        const pointLabels = ["P1 (Top-L)", "P2 (Top-R)", "P3 (Bot-R)", "P4 (Bot-L)"];
        calibrationPoints.forEach((pt, idx) => {
            ctx.beginPath();
            ctx.arc(pt.x, pt.y, 8, 0, 2 * Math.PI);
            ctx.fillStyle = "#ff1744";
            ctx.fill();
            ctx.strokeStyle = "#ffffff";
            ctx.lineWidth = 2.5;
            ctx.stroke();

            ctx.fillStyle = "#00e5ff";
            ctx.font = "bold 15px 'JetBrains Mono', monospace";
            ctx.fillText(pointLabels[idx], pt.x + 12, pt.y - 8);
        });

        updatePointBadges();
    }

    calibrationCanvas.addEventListener("click", (e) => {
        if (!isRefImageLoaded) return;

        if (calibrationPoints.length >= 4) {
            alert("All 4 road points are placed! Click 'Reset Spots' to re-plot.");
            return;
        }

        const rect = calibrationCanvas.getBoundingClientRect();
        const scaleX = calibrationCanvas.width / rect.width;
        const scaleY = calibrationCanvas.height / rect.height;

        const x = Math.round((e.clientX - rect.left) * scaleX);
        const y = Math.round((e.clientY - rect.top) * scaleY);

        calibrationPoints.push({ x, y });
        renderCalibrationCanvas();
    });

    btnResetPoints.addEventListener("click", () => {
        calibrationPoints = [];
        renderCalibrationCanvas();
    });

    function updatePointBadges() {
        const defaultNames = ["P1: Top-Left", "P2: Top-Right", "P3: Bottom-Right", "P4: Bottom-Left"];
        ptBadges.forEach((badge, idx) => {
            if (idx < calibrationPoints.length) {
                const pt = calibrationPoints[idx];
                badge.className = "point-badge set";
                badge.textContent = `${defaultNames[idx]} (${pt.x}, ${pt.y})`;
            } else {
                badge.className = "point-badge";
                badge.textContent = `${defaultNames[idx]} (Unset)`;
            }
        });
    }

    btnSaveCalibration.addEventListener("click", async () => {
        if (calibrationPoints.length !== 4) {
            alert("⚠️ Please click 4 spots on the road:\n1. Top-Left\n2. Top-Right\n3. Bottom-Right\n4. Bottom-Left");
            return;
        }

        const roadWidth = parseFloat(roadWidthInput.value);
        const roadLength = parseFloat(roadLengthInput.value);

        const payload = {
            pixel_points: calibrationPoints.map(p => [p.x, p.y]),
            road_width_m: roadWidth,
            road_length_m: roadLength
        };

        try {
            const res = await fetch(`${API_BASE}/api/calibration/save`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload)
            });
            const data = await res.json();
            if (res.ok) {
                alert("🎉 Road Calibration Saved! Return to 'Live Operations' and start stream.");
                switchTab("live-tab");
            } else {
                alert("Calibration error: " + data.detail);
            }
        } catch (err) {
            alert("Failed to save calibration: " + err.message);
        }
    });

    // ---------------- REUSABLE VIOLATION CARD TEMPLATE ----------------
    function createViolationCardHTML(v) {
        const delta = (v.speed_kmh - (v.speed_limit || 60)).toFixed(1);
        return `
            <div class="violation-card">
                <div class="vcard-header">
                    <span class="vcard-tag">TRACK ID #${v.track_id}</span>
                    <span class="vcard-speed">${v.speed_kmh.toFixed(1)} <small>km/h (+${delta})</small></span>
                </div>
                <div class="vcard-crops">
                    <div class="vcard-crop-box">
                        <label>Vehicle Snapshot</label>
                        <img src="${v.vehicle_image_url || '/' + v.vehicle_image_path}" alt="Vehicle" onerror="this.src='/static/placeholder.png'">
                    </div>
                    <div class="vcard-crop-box">
                        <label>License Plate</label>
                        <img src="${v.plate_image_url || '/' + v.plate_image_path}" alt="Plate" onerror="this.src='/static/placeholder.png'">
                    </div>
                </div>
                <div class="vcard-body">
                    <div class="vcard-plate-box">
                        <span class="plate-badge-styled">${v.plate_text || 'UNKNOWN'}</span>
                        <span class="ocr-conf-tag">OCR: ${((v.ocr_confidence || 0) * 100).toFixed(0)}%</span>
                    </div>
                    <div class="vcard-footer">
                        <span>Time: <strong class="vcard-time">${v.video_time || '00:00.00'}</strong></span>
                        <button class="btn btn-secondary btn-sm" onclick="openInspectionModal('${v.vehicle_image_url || '/' + v.vehicle_image_path}', '${v.plate_image_url || '/' + v.plate_image_path}', '${v.plate_text}', ${v.speed_kmh}, '${v.video_time}', '${v.timestamp}')">
                            Inspect
                        </button>
                    </div>
                </div>
            </div>
        `;
    }

    function handleNewViolation(v) {
        if (noViolationsMsg) noViolationsMsg.style.display = "none";

        // Prepend to Live Feed Grid on Tab 1
        const tempDiv = document.createElement("div");
        tempDiv.innerHTML = createViolationCardHTML(v);
        liveViolationsGrid.prepend(tempDiv.firstElementChild);

        fetchStats();
    }

    // ---------------- STATS & HISTORY ----------------
    async function fetchStats() {
        try {
            const res = await fetch(`${API_BASE}/api/stats`);
            const data = await res.json();
            valTotalViolations.textContent = data.total_violations || 0;
            valMaxSpeed.innerHTML = `${data.max_speed || 0} <small>km/h</small>`;
            valAvgSpeed.innerHTML = `${data.avg_speed || 0} <small>km/h</small>`;
        } catch (err) {}
    }

    async function fetchConfig() {
        try {
            const res = await fetch(`${API_BASE}/api/config`);
            const data = await res.json();
            const spd = data.thresholds?.speed_kmh || 60;
            thresholdRange.value = spd;
            thresholdVal.textContent = spd;
            valSpeedLimit.textContent = spd;
        } catch (err) {}
    }

    thresholdRange.addEventListener("input", (e) => {
        thresholdVal.textContent = e.target.value;
        valSpeedLimit.textContent = e.target.value;
    });

    thresholdRange.addEventListener("change", async (e) => {
        await fetch(`${API_BASE}/api/config/threshold`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ speed_kmh: parseFloat(e.target.value) })
        });
    });

    async function fetchViolations() {
        try {
            const res = await fetch(`${API_BASE}/api/violations?limit=100`);
            const data = await res.json();
            allViolationsCache = data.violations || [];
            renderViolationsHistory(allViolationsCache);
        } catch (err) {
            console.error("Violations fetch error:", err);
        }
    }

    function renderViolationsHistory(violations) {
        // 1. Render History Cards
        if (historyViolationsGrid) {
            if (violations.length === 0) {
                historyViolationsGrid.innerHTML = `<div class="no-violations-msg"><p>No violations recorded in database yet.</p></div>`;
            } else {
                historyViolationsGrid.innerHTML = violations.map(v => createViolationCardHTML(v)).join('');
            }
        }

        // 2. Render History Table View
        if (violationsTableBody) {
            if (violations.length === 0) {
                violationsTableBody.innerHTML = `<tr><td colspan="9" class="empty-row">No violations recorded yet.</td></tr>`;
            } else {
                violationsTableBody.innerHTML = violations.map(v => `
                    <tr>
                        <td>#${v.id}</td>
                        <td><strong>Track ${v.track_id}</strong></td>
                        <td><img src="/${v.vehicle_image_path}" class="tbl-crop-img" alt="Veh"></td>
                        <td><img src="/${v.plate_image_path}" class="tbl-crop-img" alt="Plate"></td>
                        <td><span class="plate-badge-styled">${v.plate_text || 'UNKNOWN'}</span></td>
                        <td><span style="color: var(--danger); font-weight: 700;">${v.speed_kmh.toFixed(1)} km/h</span></td>
                        <td><span style="color: var(--primary); font-weight: 600;">${v.video_time || '00:00.00'}</span></td>
                        <td>${((v.ocr_confidence || 0) * 100).toFixed(0)}%</td>
                        <td>${v.timestamp}</td>
                    </tr>
                `).join('');
            }
        }
    }

    if (tableFilter) {
        tableFilter.addEventListener("input", (e) => {
            const q = e.target.value.toLowerCase();
            const filtered = allViolationsCache.filter(v =>
                (v.plate_text && v.plate_text.toLowerCase().includes(q)) ||
                String(v.track_id).includes(q)
            );
            renderViolationsHistory(filtered);
        });
    }

    window.openInspectionModal = function (vehUrl, plateUrl, plateText, speed, videoTime, timestamp) {
        modalVehImage.src = vehUrl.replace(/\\/g, "/");
        modalPlateImage.src = plateUrl.replace(/\\/g, "/");
        modalInfo.innerHTML = `
            <div><strong>Plate Text:</strong> <span class="plate-badge-styled">${plateText || 'UNKNOWN'}</span></div>
            <div><strong>Recorded Speed:</strong> <span style="color: var(--danger); font-weight: 700;">${speed.toFixed(1)} km/h</span></div>
            <div><strong>Video Timestamp:</strong> <span style="color: var(--primary);">${videoTime}</span></div>
            <div><strong>System Date:</strong> ${timestamp}</div>
        `;
        imageModal.classList.remove("hidden");
    };

    modalClose.addEventListener("click", () => imageModal.classList.add("hidden"));
    imageModal.addEventListener("click", (e) => {
        if (e.target === imageModal) imageModal.classList.add("hidden");
    });

    document.getElementById("btnRefreshStats").addEventListener("click", () => {
        fetchStats();
        fetchViolations();
    });

    // Start services
    setupWebSocket();
    fetchStats();
    fetchConfig();
    fetchViolations();
});