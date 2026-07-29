document.addEventListener("DOMContentLoaded", () => {
    // Elements
    const dropzone = document.getElementById("dropzone");
    const videoFileInput = document.getElementById("videoFileInput");
    const btnSelectFile = document.getElementById("btnSelectFile");
    const fileInfo = document.getElementById("fileInfo");
    const fileName = document.getElementById("fileName");
    const btnUploadProcess = document.getElementById("btnUploadProcess");
    const btnRefresh = document.getElementById("btnRefresh");
    
    const valTotalViolations = document.getElementById("valTotalViolations");
    const valMaxSpeed = document.getElementById("valMaxSpeed");
    const valAvgSpeed = document.getElementById("valAvgSpeed");
    const violationsTableBody = document.getElementById("violationsTableBody");
    const tableFilter = document.getElementById("tableFilter");
    
    const imageModal = document.getElementById("imageModal");
    const modalClose = document.getElementById("modalClose");
    const modalImage = document.getElementById("modalImage");
    const modalInfo = document.getElementById("modalInfo");

    let allViolations = [];

    // File Upload Handlers
    btnSelectFile.addEventListener("click", (e) => {
        e.stopPropagation();
        videoFileInput.click();
    });

    dropzone.addEventListener("click", () => videoFileInput.click());

    dropzone.addEventListener("dragover", (e) => {
        e.preventDefault();
        dropzone.style.borderColor = "var(--primary)";
    });

    dropzone.addEventListener("dragleave", () => {
        dropzone.style.borderColor = "rgba(255, 255, 255, 0.15)";
    });

    dropzone.addEventListener("drop", (e) => {
        e.preventDefault();
        dropzone.style.borderColor = "rgba(255, 255, 255, 0.15)";
        if (e.dataTransfer.files.length) {
            handleSelectedFile(e.dataTransfer.files[0]);
        }
    });

    videoFileInput.addEventListener("change", (e) => {
        if (e.target.files.length) {
            handleSelectedFile(e.target.files[0]);
        }
    });

    function handleSelectedFile(file) {
        fileName.textContent = file.name;
        fileInfo.classList.remove("hidden");
    }

    btnUploadProcess.addEventListener("click", async (e) => {
        e.stopPropagation();
        const file = videoFileInput.files[0];
        if (!file) return;

        const formData = new FormData();
        formData.append("file", file);

        btnUploadProcess.disabled = true;
        btnUploadProcess.textContent = "Uploading...";

        try {
            const response = await fetch("/api/upload", {
                method: "POST",
                body: formData
            });
            const data = await response.json();
            alert("Video uploaded successfully: " + data.filename);
            fetchStats();
            fetchViolations();
        } catch (err) {
            alert("Upload failed: " + err.message);
        } finally {
            btnUploadProcess.disabled = false;
            btnUploadProcess.textContent = "Upload & Process";
        }
    });

    // Fetch Stats
    async function fetchStats() {
        try {
            const res = await fetch("/api/stats");
            const data = await res.json();
            valTotalViolations.textContent = data.total_violations || 0;
            valMaxSpeed.innerHTML = `${data.max_speed || 0} <small>km/h</small>`;
            valAvgSpeed.innerHTML = `${data.avg_speed || 0} <small>km/h</small>`;
        } catch (err) {
            console.error("Failed to fetch stats:", err);
        }
    }

    // Fetch Violations Table
    async function fetchViolations() {
        try {
            const res = await fetch("/api/violations");
            const data = await res.json();
            allViolations = data.violations || [];
            renderTable(allViolations);
        } catch (err) {
            console.error("Failed to fetch violations:", err);
        }
    }

    function renderTable(violations) {
        if (!violations.length) {
            violationsTableBody.innerHTML = `<tr><td colspan="7" class="empty-row">No violations recorded yet. Run video processing to populate logs.</td></tr>`;
            return;
        }

        violationsTableBody.innerHTML = violations.map(v => `
            <tr>
                <td>#${v.id}</td>
                <td>Track-${v.track_id}</td>
                <td><span class="plate-badge">${v.plate_text || "UNKNOWN"}</span></td>
                <td><span class="speed-tag">${v.speed_kmh} km/h</span></td>
                <td>${v.ocr_confidence ? (v.ocr_confidence * 100).toFixed(1) + "%" : "N/A"}</td>
                <td>${v.timestamp || "N/A"}</td>
                <td>
                    <button class="btn btn-secondary btn-sm" onclick="viewPlateCrop('${v.plate_image_path}', '${v.plate_text}')">View Plate</button>
                </td>
            </tr>
        `).join("");
    }

    window.viewPlateCrop = function(path, text) {
        if (!path) {
            alert("No image path for plate crop.");
            return;
        }
        modalImage.src = "/" + path;
        modalInfo.innerHTML = `<strong>Plate:</strong> ${text || "N/A"}`;
        imageModal.classList.remove("hidden");
    };

    modalClose.addEventListener("click", () => imageModal.classList.add("hidden"));

    btnRefresh.addEventListener("click", () => {
        fetchStats();
        fetchViolations();
    });

    tableFilter.addEventListener("input", (e) => {
        const query = e.target.value.toLowerCase();
        const filtered = allViolations.filter(v => 
            (v.plate_text && v.plate_text.toLowerCase().includes(query)) ||
            (v.track_id && String(v.track_id).includes(query))
        );
        renderTable(filtered);
    });

    // Initial load
    fetchStats();
    fetchViolations();
});
