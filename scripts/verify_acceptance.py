"""Validation script ensuring all acceptance criteria are met before completion."""
import io
import sys
from pathlib import Path
from PIL import Image
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.demo.server import app, get_model

def main():
    print("==> Initializing model and TestClient...")
    model = get_model()
    client = TestClient(app)

    # 1. Check /healthz
    print("\n[Check 1] Verifying /healthz ...")
    res = client.get("/healthz")
    assert res.status_code == 200, f"Healthz failed: {res.text}"
    health_data = res.json()
    assert health_data["model_loaded"] is True
    assert "torch_version" in health_data
    assert "smp_version" in health_data
    print(f"  OK: status={health_data['status']}, model_loaded={health_data['model_loaded']}, smp={health_data['smp_version']}")

    # 2. Check /api/generate with 01_tall_skyscraper_towers.png
    print("\n[Check 2] Testing /api/generate with test_samples/01_tall_skyscraper_towers.png ...")
    sample_p = ROOT / "test_samples" / "01_tall_skyscraper_towers.png"
    assert sample_p.exists()
    with open(sample_p, "rb") as f:
        res = client.post(
            "/api/generate",
            files={"file": ("sample.png", f, "image/png")},
            data={"ground_sample_distance": 0.3},
        )
    assert res.status_code == 200, f"Generate failed: {res.text}"
    gen_data = res.json()
    assert gen_data["success"] is True
    assert gen_data["stats"]["buildings_detected"] > 0
    obj_url = gen_data["obj_url"]
    obj_disk_p = ROOT / obj_url.lstrip("/")
    assert obj_disk_p.exists(), f"OBJ file not found on disk at: {obj_disk_p}"
    obj_text = obj_disk_p.read_text(encoding="utf-8")
    assert "v " in obj_text and "f " in obj_text, "OBJ file missing vertices or faces"
    v_count = obj_text.count("\nv ")
    f_count = obj_text.count("\nf ")
    print(f"  OK: Success={gen_data['success']}, Buildings={gen_data['stats']['buildings_detected']}, OBJ vertices={v_count}, faces={f_count}")

    # 3. Check 1024x1024 upload
    print("\n[Check 3] Testing /api/generate with 1024x1024 image ...")
    img_1024 = Image.open(sample_p).resize((1024, 1024))
    buf = io.BytesIO()
    img_1024.save(buf, format="PNG")
    buf.seek(0)
    res = client.post(
        "/api/generate",
        files={"file": ("sample_1024.png", buf, "image/png")},
        data={"ground_sample_distance": 0.3},
    )
    assert res.status_code == 200, f"1024x1024 generation failed: {res.text}"
    data_1024 = res.json()
    assert data_1024["stats"]["buildings_detected"] > 0
    assert data_1024["stats"]["max_height_m"] > 0
    print(f"  OK: 1024x1024 reconstructed {data_1024['stats']['buildings_detected']} buildings, Max H={data_1024['stats']['max_height_m']}m")

    # 4. Check empty-field returns HTTP 422 {"error": "No buildings detected"}
    print("\n[Check 4] Testing /api/generate with empty field image (expect 422) ...")
    green_field = Image.new("RGB", (256, 256), (50, 100, 50))
    buf_empty = io.BytesIO()
    green_field.save(buf_empty, format="PNG")
    buf_empty.seek(0)
    res = client.post(
        "/api/generate",
        files={"file": ("empty_field.png", buf_empty, "image/png")},
    )
    assert res.status_code == 422, f"Expected 422, got {res.status_code}: {res.text}"
    assert res.json() == {"error": "No buildings detected"}
    print(f"  OK: HTTP 422 returned with {res.json()}")

    print("\n" + "=" * 55)
    print("ALL ACCEPTANCE CRITERIA VERIFIED SUCCESSFULLY!")
    print("=" * 55)

if __name__ == "__main__":
    main()
