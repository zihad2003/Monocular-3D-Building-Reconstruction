document.addEventListener('DOMContentLoaded', () => {
    // Dashboard Logic
    const dropzone = document.getElementById('dropzone');
    const fileInput = document.getElementById('file-upload');
    const generateBtn = document.getElementById('generate-btn');
    const logConsole = document.getElementById('log-console');
    const viewerContainer = document.getElementById('viewer-container');
    const viewerActions = document.getElementById('viewer-actions');

    if(dropzone) {
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
            if(e.dataTransfer.files.length) {
                handleFile(e.dataTransfer.files[0]);
            }
        });

        fileInput.addEventListener('change', function() {
            if(this.files.length) {
                handleFile(this.files[0]);
            }
        });
    }

    function handleFile(file) {
        dropzone.innerHTML = `<span class="upload-icon">✅</span><p>${file.name} ready for processing.</p>`;
        logMsg(`File loaded: ${file.name}`);
    }

    if(generateBtn) {
        generateBtn.addEventListener('click', () => {
            logMsg("Initializing neural network backend...");
            generateBtn.disabled = true;
            generateBtn.innerText = "Processing...";
            
            setTimeout(() => logMsg("Running footprint segmentation (BONAI pre-trained)..."), 1000);
            setTimeout(() => logMsg("Estimating height maps via DSM regression..."), 2500);
            setTimeout(() => logMsg("Extruding 2D polygons to 3D meshes..."), 4000);
            
            setTimeout(() => {
                logMsg("Generation complete! Rendering in viewer.");
                generateBtn.disabled = false;
                generateBtn.innerText = "Generate 3D Model";
                
                // Simulate loading a 3D model
                viewerContainer.innerHTML = `<div style="text-align:center; color:#3b82f6;">
                    <p style="font-size:3rem; margin-bottom:1rem; animation: pulse 2s infinite;">🧊</p>
                    <p>Interactive 3D Render Here</p>
                    <p style="font-size:0.8rem; color:#94a3b8;">(Requires Three.js Integration)</p>
                </div>`;
                viewerActions.style.display = 'flex';
                viewerContainer.style.background = 'rgba(59, 130, 246, 0.1)';
            }, 5500);
        });
    }

    function logMsg(msg) {
        if(!logConsole) return;
        const p = document.createElement('p');
        p.innerText = `> ${msg}`;
        logConsole.appendChild(p);
        logConsole.scrollTop = logConsole.scrollHeight;
    }
});
