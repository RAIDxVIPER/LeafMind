"""
scripts/reproduce_500.py
Reproduce 500 serialization error with real JPEG bytes through TestClient.
"""
import io
from pathlib import Path
import sys
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image, ImageFilter
from fastapi.testclient import TestClient
from src.api.app import app

client = TestClient(app, raise_server_exceptions=True)

def test_repro():
    squash_img_path = Path("Dataset_2/Train/Squash___Powdery_mildew/PV__0056878d-a244-4653-ad56-7e97080c7fd2___UMD_Powd.M 9962.JPG")
    with open(squash_img_path, "rb") as f:
        squash_bytes = f.read()

    apple_img_path = Path("Dataset_2/Test_OOD/Apple___Apple_scab/PD__Apple Scab Leaf (1).jpg")
    with open(apple_img_path, "rb") as f:
        apple_bytes = f.read()

    print(f"Loaded real Squash JPEG ({len(squash_bytes)} bytes, starts with {squash_bytes[:4].hex()})")
    print(f"Loaded real Apple JPEG ({len(apple_bytes)} bytes, starts with {apple_bytes[:4].hex()})")

    # (a1) Escalated case: Squash with crop Squash
    print("\n--- Test (a1): Squash with crop Squash ---")
    try:
        resp_a1 = client.post(
            "/analyze",
            data={"crop": "Squash"},
            files={"image": ("squash.jpg", squash_bytes, "image/jpeg")},
        )
        print(f"Status: {resp_a1.status_code}")
    except Exception as exc:
        print("EXCEPTION RAISED in (a1):")
        traceback.print_exc()

    # (a2) Escalated case: Apple scab OOD with crop Apple
    print("\n--- Test (a2): Apple scab OOD with crop Apple ---")
    try:
        resp_a2 = client.post(
            "/analyze",
            data={"crop": "Apple"},
            files={"image": ("apple.jpg", apple_bytes, "image/jpeg")},
        )
        print(f"Status: {resp_a2.status_code}")
    except Exception as exc:
        print("EXCEPTION RAISED in (a2):")
        traceback.print_exc()

    # (b) Follow-up with previous_case_id
    print("\n--- Test (b): Follow-up case ---")
    tomato_img_path = Path("Dataset_2/Test_ID/Tomato___Early_blight/PV__0034a551-9512-44e5-ba6c-827f85ecc688___RS_Erly.B 9432.JPG")
    with open(tomato_img_path, "rb") as f:
        tomato_bytes = f.read()

    base_client = TestClient(app, raise_server_exceptions=False)
    resp_base = base_client.post(
        "/analyze",
        data={"crop": "Tomato"},
        files={"image": ("tomato.jpg", tomato_bytes, "image/jpeg")},
    )
    print(f"Base case status: {resp_base.status_code}")
    base_case_id = None
    if resp_base.status_code == 200:
        base_case_id = resp_base.json().get("case_id")
        print(f"Base case_id: {base_case_id}")
    else:
        print(f"Base case failed: {resp_base.text[:300]}")

    if base_case_id:
        try:
            resp_b = client.post(
                "/analyze",
                data={"crop": "Tomato", "previous_case_id": base_case_id},
                files={"image": ("tomato2.jpg", tomato_bytes, "image/jpeg")},
            )
            print(f"Follow-up status: {resp_b.status_code}")
        except Exception as exc:
            print("EXCEPTION RAISED in (b):")
            traceback.print_exc()

    # (c) Blurry image
    print("\n--- Test (c): Blurry image ---")
    pil_img = Image.open(io.BytesIO(tomato_bytes))
    blurry_pil = pil_img.filter(ImageFilter.GaussianBlur(radius=15))
    buf = io.BytesIO()
    blurry_pil.save(buf, format="JPEG")
    blurry_bytes = buf.getvalue()

    try:
        resp_c = client.post(
            "/analyze",
            data={"crop": "Tomato"},
            files={"image": ("blurry.jpg", blurry_bytes, "image/jpeg")},
        )
        print(f"Blurry status: {resp_c.status_code}")
    except Exception as exc:
        print("EXCEPTION RAISED in (c):")
        traceback.print_exc()

if __name__ == "__main__":
    test_repro()
