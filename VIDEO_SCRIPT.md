# 🎬 3–4 Minute Video Presentation Script & Recording Guide
**Project:** Geo3D — Monocular 3D Building Reconstruction from Optical Satellite Imagery  
**Target Duration:** 3 minutes 30 seconds – 4 minutes  
**Target Audience:** Course Evaluators, Faculty, ML Engineers & Industry Reviewers  
**Tone:** Confident, clear, engaging, and professional (Simple conversational words, zero confusing jargon)

---

## 📌 Instructions for the Presenter / Video Creator (ভিডিও যিনি রেকর্ড করবেন তার জন্য নির্দেশিকা)

> [!TIP]
> **Recording Setup Checklist:**
> 1. **Screen Resolution:** Record your desktop at **1920x1080 (1080p, 60fps)** using OBS Studio, Loom, or Windows Game Bar (`Win + Alt + R`).
> 2. **Browser Setup:** Open `http://127.0.0.1:8000/index.html` (Landing Page) and `http://127.0.0.1:8000/dashboard.html` in Chrome or Edge. Press `F11` (or maximize window for a clean view without messy bookmarks).
> 3. **Audio Quality:** Use a clean microphone or headset. Speak at a relaxed, steady pace (~130 words per minute).
> 4. **Mouse Movement:** Move the cursor smoothly without shaking or sudden erratic jerks.
> 5. **Camera/Webcam (Optional):** If showing facecam, keep a small circular frame in the bottom-right corner during Intro & Outro; minimize during the 3D viewer demo.

---

## ⏱️ Video Timeline Breakdown

| Section | Timestamp | Focus Area | Visual on Screen |
| :--- | :--- | :--- | :--- |
| **Scene 1: Hook & Problem** | `0:00 – 0:45` | Real-world problem & limitation of LiDAR | Landing Page Hero + Satellite overview |
| **Scene 2: Core Architecture** | `0:45 – 1:30` | How our AI model works | Pipeline architecture / Workflow graphic |
| **Scene 3: Live Interactive Demo** | `1:30 – 2:50` | Uploading image & 3D generation | Web Dashboard, Inference, 3D Orbit & Zoom |
| **Scene 4: Training & Results** | `2:50 – 3:30` | 40-epoch results (89.4% IoU, 2.98m MAE) | Metrics Section & Training curve plot |
| **Scene 5: Real-World Use & Outro** | `3:30 – 4:00` | Future impact & wrap up | Dashboard with 3D model spinning slowly |

---

## 🎙️ Detailed Scene-by-Scene Script & Cues

### Scene 1: The Problem & The Big Idea (0:00 – 0:45)
**Screen Action:**
- Show the Landing Page: `http://127.0.0.1:8000/index.html`.
- Cursor hovers gently on the Hero title: *"Single Optical Satellite to Precision 3D Cities"*.
- Smoothly scroll down slightly to show the side-by-side comparison (Stage 01 2D tile vs Stage 02 3D mesh).

**Presenter Voiceover (বাংলা):**
> "Creating a 3D digital model of an entire city used to require expensive drone flights, LiDAR sensors, or multiple stereo satellite passes. But what if we could reconstruct an entire 3D city from just **a single standard 2D satellite image**?
> 
> Hello everyone! Welcome to **Geo3D**. In this project, we built a deep-learning-driven pipeline that takes an ordinary optical satellite picture and reconstructs millimeter-accurate 3D building models with heights in just seconds."

*(Optional English alternative for international presentation):*
> *"Traditionally, creating 3D city models requires costly LiDAR or multi-view stereo flights. Today, we present Geo3D: a deep learning system that reconstructs full 3D building geometries and metric heights from a single monocular 2D satellite image."*

---

### Scene 2: System Architecture & How It Works (0:45 – 1:30)
**Screen Action:**
- Scroll down to the **"Reconstruction Pipeline"** section on the Landing Page.
- Point to the 3 pipeline steps:
  1. *Footprint Segmentation*
  2. *DSM Height Estimation*
  3. *Polygonal 3D Extrusion*

**Presenter Voiceover (বাংলা):**
> "Let’s look under the hood at how Geo3D works. 
> 
> Under the hood, we use a custom **Multi-Task Neural Network** built with a shared ResNet encoder and **two parallel U-Net decoders**:
> - **Decoder 1** learns building footprints — detecting the exact rooftop borders using a combined BCE and Dice loss.
> - **Decoder 2** regresses the metric height of each pixel using a masked Smooth-L1 loss, predicting the real elevation in meters.
> 
> After the neural network makes its prediction, our geometric extrusion engine isolates individual building contours, simplifies their polygons, and extrudes them vertically into standard textured 3D `.OBJ` meshes."

---

### Scene 3: Live Interactive Demo (1:30 – 2:50)  ⭐ *Most Important Section*
**Screen Action:**
- Click on **"Launch Dashboard"** button (or navigate to `http://127.0.0.1:8000/dashboard.html`).
- Show the clean dark-mode interface: upload dropzone on the left, interactive 3D viewport on the right.
- In the left sidebar under "Quick Test Samples", click **"Skyscrapers"** (or drag & drop `01_tall_skyscraper_towers.png`).
- Watch the progress bar advance and log feed show:
  - *Footprint segmentation complete*
  - *DSM height regression complete*
  - *Extruding 3D polygons...*
- When the 3D model appears in the viewer:
  - **Left-click and drag** to smoothly rotate the camera around the 3D buildings.
  - **Right-click and drag** to pan across the streets.
  - **Scroll mouse wheel** to zoom in close to individual building walls and rooftops.
  - Point out the bottom statistics badges: Buildings Detected, Mean Height, Max Height.
- *(Optional bonus):* Click on **"Commercial Downtown"** or **"Residential Houses"** to show how it handles smaller houses with lower elevations!

**Presenter Voiceover (বাংলা):**
> "Now, let’s see the real-time system in action.
> 
> Here is our interactive web dashboard. On the left, we can upload any satellite image or pick one of our pre-calibrated test scenes. Let’s click on **'Skyscrapers'**.
> 
> Instantly, the backend FastAPI server processes the tile through PyTorch. In less than a second, the footprint mask and heightmap are generated. And here is the result: a full 3D interactive mesh rendered right inside Three.js!
> 
> Notice how tall commercial skyscrapers have sharp, vertical elevations reaching over 45 meters, while surrounding structures match realistic lower heights. We can freely rotate around the block, inspect angles, zoom in, and even export the model directly for GIS, urban planning, or game engines."

---

### Scene 4: Training & Experimental Performance (2:50 – 3:30)
**Screen Action:**
- Switch back to the Landing Page or bring up the **Training Curves** graphic (`training_curves_40epochs.png`).
- Point to the metrics banner:
  - **89.4% Validation IoU**
  - **2.98 Meters Height MAE**
  - **6,999+ Training Samples**
  - **< 850 ms Inference Latency**

**Presenter Voiceover (বাংলা):**
> "Let’s discuss the model’s training performance.
> 
> We trained our model for **40 epochs** on high-resolution satellite imagery paired with normalized digital surface models (nDSM). 
> 
> The results speak for themselves:
> - Our building footprint segmentation achieved a **Validation IoU of 89.4%**, which accurately separates dense, touching buildings without merging them.
> - For height estimation, our model reached a **Mean Absolute Error of just 2.98 meters** — that’s less than the height of a single building floor!
> - The entire pipeline runs with an end-to-end latency of under **850 milliseconds** on CPU inference."

---

### Scene 5: Applications, Future Scope & Outro (3:30 – 4:00)
**Screen Action:**
- Return to the 3D Dashboard with the 3D model rotating in auto-orbit mode or gentle manual rotation.
- Show the clean footer: *"Geo3D — Deep Learning for Monocular Geospatial Intelligence"*.
- Presenter smile / sign-off.

**Presenter Voiceover (বাংলা):**
> "Geo3D demonstrates that high-fidelity 3D geospatial reconstruction is now possible using affordable, widely available single satellite imagery. This technology opens massive opportunities for **disaster damage assessment**, **rapid urban planning**, **solar energy potential calculation**, and **smart city twins**.
> 
> Everything is open-source and ready for production deployment. Thank you so much for watching!"

---

## 💡 Quick Tips for the Recording Person
1. **Pacing:** Don't rush through the 3D model interaction. Spend at least 40 seconds smoothly spinning and zooming the 3D buildings. This is the visual "wow factor" that evaluators remember.
2. **Cursor focus:** While mentioning "89.4% IoU" or "2.98m MAE", circle or hover the mouse over those stats so the viewer’s eye follows naturally.
3. **Audio recording:** If you make a mistake while reading, pause for 2 seconds, take a breath, and re-read the sentence from the start. You can easily cut out the pause in video editing (CapCut, Premiere, or DaVinci Resolve).
