document.addEventListener("DOMContentLoaded", () => {
    // API Endpoints
    const API_BASE = "";

    // DOM Elements
    const dropzone = document.getElementById("dropzone");
    const videoFileInput = document.getElementById("videoFileInput");
    const btnSelectFile = document.getElementById("btnSelectFile");
    const fileInfo = document.getElementById("fileInfo");
    const fileNameSpan = document.getElementById("fileName");
    const btnUploadProcess = document.getElementById("btnUploadProcess");
    const btnRefresh = document.getElementById("btnRefresh");

    const valTotalViolations = document.getElementById("valTotalViolations");
    const valMaxSpeed = document.getElementById("valMaxSpeed");
    const valAvgSpeed = document.getElementById("valAvgSpeed");

    const thresholdRange = document.getElementById("thresholdRange");
    const thresholdVal = document.getElementById("thresholdVal");
    const badgeThresholdVal = document.getElementById("badgeThresholdVal");

    const violationsTableBody = document.getElementById("violationsTableBody");
    const tableFilter = document.getElementById("tableFilter");

    const imageModal = document.getElementById("imageModal");
    const modalClose = document.getElementById("modalClose");
    const modalImage = document.getElementById("modalImage");
    const modalInfo = document.getElementById("modalInfo");

    let selectedFile = null;
    let pollInterval = null;

    // Load initial stats & config
    fetchStats();
    fetchViolations();
    fetchConfig();

    // ----------------------------------------------------
    // EVENT LISTENERS
    // ----------------------------------------------------
    btnSelectFile.addEventListener("click", () => videoFileInput.click());

    videoFileInput.addEventListener("change", (e) => {
        if (e.target.files.length > 0) {
            handleFileSelect(e.target.files[0]);
        }
    });

    dropzone.addEventListener("dragover", (e) => {
        e.preventDefault();
        dropzone.classList.add("dragover");
    });

    dropzone.addEventListener("dragleave", () => {
        dropzone.classList.remove("dragover");
    });

    dropzone.addEventListener("drop", (e) => {
        e.preventDefault();
        dropzone.classList.remove("dragover");
        if (e.dataTransfer.files.length > 0) {
            handleFileSelect(e.dataTransfer.files[0]);
        }
    });

    btnUploadProcess.addEventListener("click", uploadAndProcess);
    btnRefresh.addEventListener("click", () => {
        fetchStats();
        fetchViolations();
    });

    thresholdRange.addEventListener("input", (e) => {
        const val = e.target.value;
        thresholdVal.textContent = val;
        badgeThresholdVal.textContent = val;
    });

    thresholdRange.addEventListener("change", (e) => {
        updateThreshold(e.target.value);
    });

    tableFilter.addEventListener("input", (e) => {
        filterTable(e.target.value);
    });

    modalClose.addEventListener("click", () => {
        imageModal.classList.add("hidden");
    });

    // ----------------------------------------------------
    // FILE HANDLING & UPLOAD
    // ----------------------------------------------------
    function handleFileSelect(file) {
        selectedFile = file;
        fileNameSpan.textContent = `${file.name} (${(file.size / (1024 * 1024)).toFixed(1)} MB)`;
        fileInfo.classList.remove("hidden");
    }

    async function uploadAndProcess() {
        if (!selectedFile) return;

        btnUploadProcess.disabled = True;
        btnUploadProcess.textContent = "Uploading...";

        const formData = new FormData();
        formData.append("file", selectedFile);

        try {
            const res = await fetch(`${API_BASE}/api/upload`, {
                method: "POST",
                body: formData
            });

            const data = await res.json();
            if (res.ok) {
                alert("Video uploaded! ALPR Background Processing started.");
                btnUploadProcess.textContent = "Processing...";
                startStatusPolling();
            } else {
                alert("Upload failed: " + (data.detail || "Server error"));
                btnUploadProcess.disabled = false;
                btnUploadProcess.textContent = "Upload & Process";
            }
        } catch (err) {
            alert("Failed to fetch: Make sure server is running at http://127.0.0.1:8000");
            btnUploadProcess.disabled = false;
            btnUploadProcess.textContent = "Upload & Process";
        }
    }

    // ----------------------------------------------------
    // LIVE STATUS POLLING
    // ----------------------------------------------------
    function startStatusPolling() {
        if (pollInterval) clearInterval(pollInterval);

        pollInterval = setInterval(async () => {
            try {
                const res = await fetch(`${API_BASE}/api/status`);
                const status = await res.json();

                if (status.status === "processing") {
                    btnUploadProcess.textContent = `Processing: ${status.progress}% (${status.current_frame}/${status.total_frames} frames)`;
                    fetchStats();
                    fetchViolations();
                } else if (status.status === "completed") {
                    clearInterval(pollInterval);
                    btnUploadProcess.textContent = "Upload & Process";
                    btnUploadProcess.disabled = false;
                    alert("Processing Completed! All violations logged.");
                    fetchStats();
                    fetchViolations();
                } else if (status.status === "error") {
                    clearInterval(pollInterval);
                    btnUploadProcess.textContent = "Upload & Process";
                    btnUploadProcess.disabled = false;
                    alert("Processing Error: " + status.error);
                }
            } catch (err) {
                console.error("Status polling error:", err);
            }
        }, 2000);
    }

    // ----------------------------------------------------
    // API CALLS
    // ----------------------------------------------------
    async function fetchStats() {
        try {
            const res = await fetch(`${API_BASE}/api/stats`);
            const data = await res.json();
            valTotalViolations.textContent = data.total_violations || 0;
            valMaxSpeed.innerHTML = `${data.max_speed || 0} <small>km/h</small>`;
            valAvgSpeed.innerHTML = `${data.avg_speed || 0} <small>km/h</small>`;
        } catch (err) {
            console.error("Fetch stats error:", err);
        }
    }

    async function fetchViolations() {
        try {
            const res = await fetch(`${API_BASE}/api/violations?limit=50`);
            const data = await res.json();
            renderTable(data.violations || []);
        } catch (err) {
            console.error("Fetch violations error:", err);
        }
    }

    async function fetchConfig() {
        try {
            const res = await fetch(`${API_BASE}/api/config`);
            const data = await res.json();
            const spd = data.thresholds.speed_kmh || 60;
            thresholdRange.value = spd;
            thresholdVal.textContent = spd;
            badgeThresholdVal.textContent = spd;
        } catch (err) {
            console.error("Fetch config error:", err);
        }
    }

    async function updateThreshold(newSpeed) {
        try {
            await fetch(`${API_BASE}/api/config/threshold`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ speed_kmh: parseFloat(newSpeed) })
            });
            fetchConfig();
        } catch (err) {
            console.error("Update threshold error:", err);
        }
    }

    // ----------------------------------------------------
    // TABLE RENDERING & MODAL
    // ----------------------------------------------------
    function renderTable(violations) {
        if (!violations || violations.length === 0) {
            violationsTableBody.innerHTML = `<tr><td colspan="7" class="empty-row">No violations recorded yet. Upload a video to start analysis.</td></tr>`;
            return;
        }

        violationsTableBody.innerHTML = violations.map(v => `
            <tr>
                <td>#${v.id}</td>
                <td><span class="tag tag-id">Track ${v.track_id}</span></td>
                <td><strong class="plate-display">${v.plate_text || 'UNKNOWN'}</strong></td>
                <td><span class="speed-tag">${v.speed_kmh.toFixed(1)} km/h</span></td>
                <td>${(v.ocr_confidence * 100).toFixed(0)}%</td>
                <td>${v.timestamp}</td>
                <td>
                    <button class="btn btn-sm btn-accent" onclick="openModal('${v.plate_image_path || v.vehicle_image_path}', '${v.plate_text}', ${v.speed_kmh}, '${v.timestamp}')">
                        View Image
                    </button>
                </td>
            </tr>
        `).join('');
    }

    function filterTable(query) {
        const rows = violationsTableBody.querySelectorAll("tr");
        const term = query.toLowerCase();
        rows.forEach(row => {
            const text = row.textContent.toLowerCase();
            row.style.display = text.includes(term) ? "" : "none";
        });
    }

    window.openModal = function(imgSrc, plateText, speed, timestamp) {
        if (!imgSrc) {
            alert("No image crop available for this record.");
            return;
        }
        // Normalize path for server URL
        const cleanPath = imgSrc.replace(/\\/g, "/");
        modalImage.src = `/${cleanPath}`;
        modalInfo.innerHTML = `
            <p><strong>Plate Text:</strong> ${plateText}</p>
            <p><strong>Recorded Speed:</strong> ${speed.toFixed(1)} km/h</p>
            <p><strong>Timestamp:</strong> ${timestamp}</p>
        `;
        imageModal.classList.remove("hidden");
    };
});