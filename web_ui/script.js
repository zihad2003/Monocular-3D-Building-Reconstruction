document.addEventListener('DOMContentLoaded', () => {
    const dropzone = document.getElementById('dropzone');
    const fileInput = document.getElementById('file-upload');
    const generateBtn = document.getElementById('generate-btn');
    const logConsole = document.getElementById('log-console');
    const viewerContainer = document.getElementById('viewer-container');
    const viewerActions = document.getElementById('viewer-actions');
    const viewerTabs = document.getElementById('viewer-tabs');
    const emptyState = document.getElementById('empty-state');
    const threeCanvas = document.getElementById('three-canvas');
    const previewImg2D = document.getElementById('2d-preview');
    const viewerHint = document.getElementById('viewer-hint');
    const modelStats = document.getElementById('model-stats');
    const downloadObjBtn = document.getElementById('download-obj-btn');
    const exportHeightBtn = document.getElementById('export-height-btn');

    const gsdInput = document.getElementById('gsd-input');
    const gsdDisplay = document.getElementById('gsd-display');
    if (gsdInput && gsdDisplay) {
        gsdInput.addEventListener('input', () => {
            gsdDisplay.innerText = `${parseFloat(gsdInput.value || 0.3).toFixed(2)} m/px`;
        });
    }

    // Determine API root: if opened locally via file://, point to http://127.0.0.1:8000
    const API_BASE = (window.location.protocol === 'file:' || !window.location.host)
        ? 'http://127.0.0.1:8000'
        : '';

    const tab3D = document.getElementById('tab-3d');
    const tabMask = document.getElementById('tab-mask');
    const tabHeight = document.getElementById('tab-height');

    let currentFile = null;
    let lastResult = null;

    // Three.js State
    let scene = null;
    let camera = null;
    let renderer = null;
    let controls = null;
    let currentMesh = null;
    let animationFrameId = null;

    // --- Drag and Drop File Upload ---
    if (dropzone) {
        dropzone.addEventListener('click', () => fileInput.click());

        dropzone.addEventListener('dragover', (e) => {
            e.preventDefault();
            dropzone.style.borderColor = '#3b82f6';
        });

        dropzone.addEventListener('dragleave', () => {
            dropzone.style.borderColor = 'rgba(255,255,255,0.2)';
        });

        dropzone.addEventListener('drop', (e) => {
            e.preventDefault();
            dropzone.style.borderColor = 'rgba(255,255,255,0.2)';
            if (e.dataTransfer.files.length) {
                handleFile(e.dataTransfer.files[0]);
            }
        });

        fileInput.addEventListener('change', function () {
            if (this.files.length) {
                handleFile(this.files[0]);
            }
        });
    }

    function handleFile(file) {
        currentFile = file;
        const reader = new FileReader();
        reader.onload = (e) => {
            dropzone.innerHTML = `
                <img src="${e.target.result}" style="max-height: 120px; max-width: 100%; border-radius: 8px; margin-bottom: 0.5rem; object-fit: contain;">
                <p><strong>${file.name}</strong> ready (${(file.size / 1024).toFixed(1)} KB)</p>
            `;
        };
        reader.readAsDataURL(file);
        logMsg(`Loaded input image: ${file.name}`);
    }

    // --- Quick Sample Buttons ---
    document.querySelectorAll('.sample-btn').forEach(btn => {
        btn.addEventListener('click', async () => {
            const sampleName = btn.dataset.sample;
            logMsg(`Loading sample: ${sampleName}...`);
            try {
                const res = await fetch(`${API_BASE}/test_samples/${sampleName}`);
                if (!res.ok) throw new Error("Could not fetch sample");
                const blob = await res.blob();
                const file = new File([blob], sampleName, { type: "image/png" });
                handleFile(file);
                // Auto trigger generation
                setTimeout(() => {
                    if (generateBtn) generateBtn.click();
                }, 300);
            } catch (err) {
                console.error(err);
                if (window.location.protocol === 'file:') {
                    logMsg(`Error loading sample: Failed to connect to server at ${API_BASE}. Make sure the server is started with: python src/demo/server.py`);
                } else {
                    logMsg(`Error loading sample: ${err.message}`);
                }
            }
        });
    });

    // --- Generate 3D Model API Call ---
    if (generateBtn) {
        generateBtn.addEventListener('click', async () => {
            if (!currentFile) {
                alert("Please select or drop a satellite image first.");
                return;
            }

            generateBtn.disabled = true;
            generateBtn.innerText = "Reconstructing 3D...";
            logMsg("Sending image to neural network inference server...");

            const gsdVal = gsdInput ? (parseFloat(gsdInput.value) || 0.3) : 0.3;
            const formData = new FormData();
            formData.append("file", currentFile);
            formData.append("ground_sample_distance", gsdVal);

            let attempt = 0;
            const maxAttempts = 6;

            async function runInference() {
                attempt++;
                try {
                    const response = await fetch(`${API_BASE}/api/generate`, {
                        method: 'POST',
                        body: formData
                    });

                    if (!response.ok) {
                        if ((response.status === 502 || response.status === 503 || response.status === 504) && attempt < maxAttempts) {
                            logMsg(`server waking up (free tier)... (attempt ${attempt}/${maxAttempts}, retrying in 5s)`);
                            await new Promise(r => setTimeout(r, 5000));
                            return await runInference();
                        }
                        if (response.status === 422) {
                            const errJson = await response.json().catch(() => null);
                            const errMsg = (errJson && errJson.error) ? errJson.error : "No buildings detected";
                            throw new Error(errMsg);
                        }
                        const errText = await response.text().catch(() => '');
                        throw new Error(`Server returned HTTP ${response.status} ${errText ? ': ' + errText : ''}`);
                    }

                    const data = await response.json();
                    if (!data.success && data.error) {
                        throw new Error(data.error);
                    }
                    lastResult = data;

                    // Log steps returned by server
                    if (data.logs) {
                        data.logs.forEach((msg, idx) => {
                            setTimeout(() => logMsg(msg), idx * 250);
                        });
                    }

                    // Update UI Stats
                    if (data.stats) {
                        modelStats.innerText = `Buildings: ${data.stats.buildings_detected} | Max H: ${data.stats.max_height_m}m | Mean H: ${data.stats.mean_height_m}m | GSD: ${data.stats.ground_sample_distance}m/px`;
                    }

                    // Setup Download Links
                    downloadObjBtn.href = `${API_BASE}${data.obj_url}?t=${Date.now()}`;
                    exportHeightBtn.href = `${API_BASE}${data.height_url}?t=${Date.now()}`;

                    // Reveal Viewer
                    emptyState.style.display = 'none';
                    viewerTabs.style.display = 'flex';
                    viewerActions.style.display = 'flex';

                    // Render Three.js 3D Model
                    show3DView();
                    loadObjModel(`${API_BASE}${data.obj_url}?t=${Date.now()}`);

                } catch (err) {
                    console.error(err);
                    const isNetworkErr = err.message.includes("Failed to fetch") || err.message.includes("NetworkError") || (err.name === "TypeError");
                    if (isNetworkErr && attempt < maxAttempts) {
                        logMsg(`server waking up (free tier)... (attempt ${attempt}/${maxAttempts}, retrying in 5s)`);
                        await new Promise(r => setTimeout(r, 5000));
                        return await runInference();
                    }

                    logMsg(`Inference notice: ${err.message}`);
                    if (err.message.includes("No buildings detected")) {
                        alert("No buildings detected in this image. Please upload a satellite image containing building footprints.");
                    } else if (attempt >= maxAttempts) {
                        logMsg(`Server did not respond. If running locally, please ensure 'python src/demo/server.py' is running at http://127.0.0.1:8000.`);
                    }
                } finally {
                    if (attempt >= maxAttempts || (lastResult && lastResult.success)) {
                        generateBtn.disabled = false;
                        generateBtn.innerText = "Generate 3D Model";
                    }
                }
            }

            await runInference();
        });
    }

    // --- Tab Switching ---
    tab3D.addEventListener('click', () => {
        setTabActive(tab3D);
        show3DView();
    });

    tabMask.addEventListener('click', () => {
        if (!lastResult) return;
        setTabActive(tabMask);
        show2DView(lastResult.mask_url);
    });

    tabHeight.addEventListener('click', () => {
        if (!lastResult) return;
        setTabActive(tabHeight);
        show2DView(lastResult.height_url);
    });

    function setTabActive(activeBtn) {
        [tab3D, tabMask, tabHeight].forEach(btn => {
            btn.style.background = 'rgba(255,255,255,0.1)';
            btn.style.border = '1px solid rgba(255,255,255,0.2)';
        });
        activeBtn.style.background = '#3b82f6';
        activeBtn.style.border = 'none';
    }

    function show3DView() {
        threeCanvas.style.display = 'block';
        previewImg2D.style.display = 'none';
        viewerHint.style.display = 'block';
        if (renderer && camera) {
            onWindowResize();
        }
    }

    function show2DView(imgUrl) {
        threeCanvas.style.display = 'none';
        previewImg2D.style.display = 'block';
        viewerHint.style.display = 'none';
        const fullUrl = imgUrl.startsWith('http') ? imgUrl : `${API_BASE}${imgUrl}`;
        previewImg2D.src = fullUrl + `?t=${Date.now()}`;
    }

    // --- Three.js Scene Setup & Model Loading ---
    function initThree() {
        if (renderer) return;

        const width = viewerContainer.clientWidth;
        const height = viewerContainer.clientHeight;

        scene = new THREE.Scene();
        scene.background = new THREE.Color(0x070b14);

        camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 2000);
        camera.position.set(60, 80, 100);

        renderer = new THREE.WebGLRenderer({ canvas: threeCanvas, antialias: true, alpha: true });
        renderer.setSize(width, height);
        renderer.setPixelRatio(window.devicePixelRatio);
        renderer.shadowMap.enabled = true;

        controls = new THREE.OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;
        controls.dampingFactor = 0.05;
        controls.maxPolarAngle = Math.PI / 2.05; // Don't go below ground

        // Lighting
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.7);
        scene.add(ambientLight);

        const dirLight1 = new THREE.DirectionalLight(0xfff3e0, 0.9);
        dirLight1.position.set(50, 100, 50);
        scene.add(dirLight1);

        const dirLight2 = new THREE.DirectionalLight(0x80d8ff, 0.4);
        dirLight2.position.set(-50, 50, -50);
        scene.add(dirLight2);

        // Ground Grid
        const grid = new THREE.GridHelper(200, 40, 0x3b82f6, 0x1e293b);
        grid.position.y = -0.1;
        scene.add(grid);

        window.addEventListener('resize', onWindowResize);

        function animate() {
            animationFrameId = requestAnimationFrame(animate);
            controls.update();
            renderer.render(scene, camera);
        }
        animate();
    }

    function onWindowResize() {
        if (!renderer || !camera) return;
        const width = viewerContainer.clientWidth;
        const height = viewerContainer.clientHeight;
        camera.aspect = width / height;
        camera.updateProjectionMatrix();
        renderer.setSize(width, height);
    }

    function loadObjModel(url) {
        initThree();

        if (currentMesh) {
            scene.remove(currentMesh);
            currentMesh = null;
        }

        const loader = new THREE.OBJLoader();
        loader.load(
            url,
            (obj) => {
                // Apply realistic architectural styling to each building mesh
                obj.traverse((child) => {
                    if (child.isMesh) {
                        child.castShadow = true;
                        child.receiveShadow = true;

                        // Modern clean architectural building material
                        child.material = new THREE.MeshStandardMaterial({
                            color: 0xf1f5f9,      // Crisp off-white architectural concrete
                            roughness: 0.45,
                            metalness: 0.1,
                            side: THREE.DoubleSide
                        });

                        // Crisp architectural edge lines (like Google Earth 3D / SketchUp models)
                        try {
                            const edges = new THREE.EdgesGeometry(child.geometry, 28);
                            const edgeLine = new THREE.LineSegments(
                                edges,
                                new THREE.LineBasicMaterial({ color: 0x1e293b, linewidth: 1.5 })
                            );
                            child.add(edgeLine);
                        } catch (e) {
                            // Skip edges if geometry is non-standard
                        }
                    }
                });

                // Compute bounding box and center object cleanly
                const box = new THREE.Box3().setFromObject(obj);
                const center = box.getCenter(new THREE.Vector3());
                const size = box.getSize(new THREE.Vector3());

                obj.position.x -= center.x;
                obj.position.y -= box.min.y; // Sit flat on ground
                obj.position.z -= center.z;

                scene.add(obj);
                currentMesh = obj;

                // Position camera with beautiful 45-degree bird's-eye architectural perspective
                const maxDim = Math.max(size.x, size.z, 20);
                camera.position.set(maxDim * 1.1, maxDim * 1.3, maxDim * 1.4);
                controls.target.set(0, size.y * 0.4, 0);
                controls.update();

                logMsg("3D architectural mesh loaded successfully into viewer.");
            },
            (xhr) => {
                // Progress
            },
            (err) => {
                console.error("OBJ Load Error:", err);
                logMsg("Warning: Could not render OBJ in Three.js.");
            }
        );
    }

    function logMsg(msg) {
        if (!logConsole) return;
        const p = document.createElement('p');
        p.innerText = `> ${msg}`;
        logConsole.appendChild(p);
        logConsole.scrollTop = logConsole.scrollHeight;
    }
});
