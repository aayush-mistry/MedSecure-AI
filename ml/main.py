import os
import re
import json
import time
import threading
import sqlite3
import difflib
import ssl
import numpy as np
import cv2
from PIL import Image, ImageEnhance, ImageFilter
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import easyocr

# Disable SSL certificate verification for model downloads (macOS fix)
ssl._create_default_https_context = ssl._create_unverified_context

# Barcode detection - optional, works if ZBar native library is installed
try:
    from pyzbar.pyzbar import decode as zbar_decode
    from pyzbar.pyzbar import ZBarSymbol
    HAS_ZBAR = True
except Exception:
    HAS_ZBAR = False
    def zbar_decode(*args, **kwargs):
        return []
    print("ZBar not available. Barcode detection disabled.")

app = FastAPI(title="MedSecure ML Inference Service v4")

print("Loading EasyOCR model (CPU)...")
reader = easyocr.Reader(['en'], gpu=False, verbose=False)
print("EasyOCR ready.")

DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "db", "medsecure.db"))

# In-memory progress store for async scan processing
scan_progress_store = {}
scan_progress_lock = threading.Lock()

STAGES = [
    "preprocessing",
    "ocr_extraction",
    "barcode_decoding",
    "visual_analysis",
    "medicine_lookup",
    "batch_lookup",
    "field_comparison",
    "scoring"
]

class ScanRequest(BaseModel):
    scan_id: str
    file_path: str

class ProgressResponse(BaseModel):
    scan_id: str
    stage: str
    stage_index: int
    total_stages: int
    progress: float
    status: str

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def set_progress(scan_id, stage, stage_index, progress, status="processing"):
    with scan_progress_lock:
        scan_progress_store[scan_id] = {
            "scan_id": scan_id,
            "stage": stage,
            "stage_index": stage_index,
            "total_stages": len(STAGES),
            "progress": progress,
            "status": status
        }

# ─── Image Preprocessing ──────────────────────────────────────────────────────

def preprocess_image(file_path):
    """Enhance image for better OCR and CV analysis: deskew, denoise, contrast."""
    img = cv2.imread(file_path)
    if img is None:
        return None, ["Could not read image file"]

    anomalies = []
    h, w = img.shape[:2]

    # 1. Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2. Denoise
    denoised = cv2.fastNlMeansDenoising(gray, h=10, templateWindowSize=7, searchWindowSize=21)

    # 3. Adaptive contrast enhancement (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)

    # 4. Deskew - detect rotation angle and correct
    coords = np.column_stack(np.where(enhanced > 0))
    if len(coords) > 0:
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = 90 + angle
        if abs(angle) > 0.5:
            center = (w // 2, h // 2)
            matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            enhanced = cv2.warpAffine(enhanced, matrix, (w, h),
                                      flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            anomalies.append(f"Image deskewed by {angle:.1f} degrees")

    # 5. Sharpen
    sharpen_kernel = np.array([[-1, -1, -1],
                               [-1,  9, -1],
                               [-1, -1, -1]])
    sharpened = cv2.filter2D(enhanced, -1, sharpen_kernel)

    # 6. Convert back to BGR for EasyOCR compatibility
    final = cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)

    return final, anomalies

# ─── Barcode Extraction ───────────────────────────────────────────────────────

def extract_barcodes(file_path):
    """Detect and decode barcodes from image using pyzbar."""
    img = cv2.imread(file_path)
    if img is None:
        return [], []

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    barcodes = zbar_decode(gray)

    results = []
    for barcode in barcodes:
        data = barcode.data.decode("utf-8")
        barcode_type = barcode.type
        rect = barcode.rect
        results.append({
            "data": data,
            "type": barcode_type,
            "x": rect.left,
            "y": rect.top,
            "w": rect.width,
            "h": rect.height
        })

    return results

# ─── Date Normalization ───────────────────────────────────────────────────────

def normalize_date(date_str):
    """Normalize date to MM/YYYY format for comparison.
    Accepts: 08/2027, 08-2027, 08/27, 08-27, 2027/08, etc.
    Returns: MM/YYYY string or original string if unparseable.
    """
    if not date_str:
        return ""
    date_str = date_str.strip()

    # Try MM/YYYY or MM-YYYY
    m = re.match(r'^(\d{2})[/-](\d{4})$', date_str)
    if m:
        return f"{m.group(1)}/{m.group(2)}"

    # Try MM/YY or MM-YY  (expand to MM/20YY)
    m = re.match(r'^(\d{2})[/-](\d{2})$', date_str)
    if m:
        return f"{m.group(1)}/20{m.group(2)}"

    # Try YYYY/MM or YYYY-MM
    m = re.match(r'^(\d{4})[/-](\d{2})$', date_str)
    if m:
        return f"{m.group(2)}/{m.group(1)}"

    return date_str

# ─── OCR Field Extraction ─────────────────────────────────────────────────────

def extract_fields(ocr_results):
    """Extract structured medicine fields from EasyOCR results."""
    lines = [r[1].strip() for r in ocr_results if len(r[1].strip()) >= 2]
    full = " ".join(lines)

    fields = {
        "name": "",
        "manufacturer": "",
        "batch_number": "",
        "expiry_date": "",
        "mfg_date": "",
        "mrp": "",
        "license_number": "",
        "ocr_boxes": [],
        "barcodes": []
    }

    # Batch number
    batch_m = re.search(
        r'(?:batch\s*(?:no|number|n\.?o?\.?)|b\.?\s*n\.?\s*o?\.?)\s*[:\-\s]*([A-Z0-9][A-Z0-9\-/]{2,})',
        full, re.IGNORECASE)
    if batch_m:
        fields["batch_number"] = batch_m.group(1).strip()
    else:
        standalone = re.search(r'\b([A-Z]{2}\d{4,6})\b', full)
        if standalone:
            fields["batch_number"] = standalone.group(1)

    # Expiry date
    exp_m = re.search(
        r'(?:exp\.?\s*(?:date|dt)?|expiry)\s*[:\-\s]*((?:\d{2})[/\-](?:\d{2,4}))',
        full, re.IGNORECASE)
    if exp_m:
        fields["expiry_date"] = normalize_date(exp_m.group(1))

    # Manufacturing date
    mfg_m = re.search(
        r'(?:mfg\.?\s*(?:date|dt)?|mfd\.?)\s*[:\-\s]*((?:\d{2})[/\-](?:\d{2,4}))',
        full, re.IGNORECASE)
    if mfg_m:
        fields["mfg_date"] = normalize_date(mfg_m.group(1))

    # MRP
    mrp_m = re.search(
        r'(?:mrp|m\.?r\.?p\.?|price)\s*[:\-\s]*(?:rs\.?\s*)?(\d+\.?\d*)',
        full, re.IGNORECASE)
    if mrp_m:
        fields["mrp"] = f"₹{mrp_m.group(1)}"

    # Manufacturer
    mfr_m = re.search(
        r'(?:mfg\.?\s*by|manufactured\s*by|mfr\.?\s*by)\s*[:\-\s]*(.+?)(?:\r|\n|$)',
        full, re.IGNORECASE)
    if mfr_m:
        fields["manufacturer"] = mfr_m.group(1).strip()[:80]

    # Manufacturing License Number
    lic_m = re.search(
        r'(?:mfg\.?\s*lic\.?(?:\s*no\.?)?|manufacturing\s*licen[sc]e(?:\s*no\.?)?|lic\.?\s*no\.?)\s*[:\-\s]*([A-Z0-9/\-\.]{6,})',
        full, re.IGNORECASE)
    if lic_m:
        fields["license_number"] = lic_m.group(1).strip()[:60]

    return fields, lines

# ─── Visual Quality Analysis ──────────────────────────────────────────────────

def analyze_visual_quality(img, expected_colors_json):
    """Full visual analysis: blur, color deviation, edge density, saturation, JPEG artifacts."""
    if img is None:
        return 50.0, ["Image could not be loaded for visual analysis"]

    anomalies = []
    score = 100.0
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = img.shape[:2]

    # 1. Blur detection (Laplacian variance)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    if laplacian_var < 30:
        score -= 45
        anomalies.append(f"Severe image blur detected (sharpness: {laplacian_var:.0f}). Indicates low-quality photocopied reproduction.")
    elif laplacian_var < 60:
        score -= 30
        anomalies.append(f"High print blur (sharpness: {laplacian_var:.0f}). Possible scanned/reprinted packaging.")
    elif laplacian_var < 120:
        score -= 15
        anomalies.append(f"Moderate blur in print (sharpness: {laplacian_var:.0f}). Possible quality issue.")

    # 2. Color profile deviation
    try:
        expected = json.loads(expected_colors_json) if isinstance(expected_colors_json, str) else expected_colors_json
        primary_hex = expected.get("primary", "#ffffff").lstrip('#')
        expected_rgb = np.array([int(primary_hex[i:i+2], 16) for i in (0, 2, 4)], dtype=np.float64)
        expected_bgr = expected_rgb[::-1]

        top_strip = img[0:int(h*0.12), :].reshape(-1, 3)
        bottom_strip = img[int(h*0.88):, :].reshape(-1, 3)

        non_white_mask = ~(np.all(top_strip > 200, axis=1))
        top_colors = top_strip[non_white_mask] if np.any(non_white_mask) else top_strip

        non_white_mask = ~(np.all(bottom_strip > 200, axis=1))
        bottom_colors = bottom_strip[non_white_mask] if np.any(non_white_mask) else bottom_strip

        if len(top_colors) > 0 and len(bottom_colors) > 0:
            mean_color = (np.mean(top_colors, axis=0) + np.mean(bottom_colors, axis=0)) / 2
            dist = np.linalg.norm(expected_bgr - mean_color)
            if dist > 120:
                score -= 35
                anomalies.append(f"Major packaging color mismatch (delta: {dist:.0f}). Expected primary hue #{primary_hex}, detected significantly different palette.")
            elif dist > 70:
                score -= 15
                anomalies.append(f"Color variance detected (delta: {dist:.0f}). Possible printing batch color drift.")
    except Exception:
        pass

    # 3. Edge density and text presence
    edges = cv2.Canny(gray, 50, 150)
    edge_ratio = np.sum(edges > 0) / edges.size
    if edge_ratio < 0.01:
        score -= 20
        anomalies.append("Extremely low edge density. Packaging appears blank or severely degraded.")
    elif edge_ratio < 0.02:
        score -= 10
        anomalies.append("Low text/edge density. Packaging may have missing printed content.")

    # 4. Unnatural color saturation
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mean_sat = np.mean(hsv[:, :, 1])
    if mean_sat > 180:
        score -= 10
        anomalies.append("Atypically high color saturation. Common in digitally reprinted packaging.")

    # 5. JPEG compression artifacts
    block_variances = []
    for i in range(0, h - 8, 8):
        for j in range(0, w - 8, 8):
            block_variances.append(np.var(gray[i:i+8, j:j+8]))
    mean_block_var = np.mean(block_variances) if block_variances else 0
    if mean_block_var < 20:
        score -= 5
        anomalies.append("Heavy JPEG compression artifacts detected. Suggests digital re-encoding.")

    return max(0.0, score), anomalies

# ─── Medicine Lookup ──────────────────────────────────────────────────────────

def lookup_medicine(extracted_name, lines, full_text, conn):
    """Find the best matching medicine in the DB using fuzzy text matching."""
    cur = conn.cursor()
    cur.execute("""
        SELECT id, name, generic_name, manufacturer_name, approved_batch_format,
               composition, expected_colors, barcode_required
        FROM medicines
    """)
    medicines = [dict(r) for r in cur.fetchall()]

    best_med = None
    best_score = 0.0

    # First pass: exact substring match in full_text
    for med in medicines:
        if med["name"].lower() in full_text.lower():
            ratio = 0.95
            if ratio > best_score:
                best_score = ratio
                best_med = med

    # Second pass: fuzzy line-by-line
    if best_score < 0.7:
        for line in lines:
            if len(line) < 3:
                continue
            for med in medicines:
                ratio = difflib.SequenceMatcher(None, line.lower(), med["name"].lower()).ratio()
                if ratio > best_score:
                    best_score = ratio
                    best_med = med

    return best_med, best_score

# ─── Batch Lookup ─────────────────────────────────────────────────────────────

def lookup_batch(medicine_id, batch_number, conn):
    """Look up a genuine batch record in medicine_batches."""
    if not medicine_id or not batch_number:
        return None
    cur = conn.cursor()
    row = cur.execute(
        """SELECT * FROM medicine_batches
           WHERE medicine_id = ? AND batch_number = ?""",
        (medicine_id, batch_number)
    ).fetchone()
    return dict(row) if row else None

# ─── Field Comparison ─────────────────────────────────────────────────────────

def compare_fields(extracted, batch_row, medicine):
    """Compare every extracted OCR field against the stored batch / medicine record.
    Returns a dict of field comparison results.
    """
    results = {}

    def make_result(extracted_val, stored_val, match, note=None):
        r = {"extracted": extracted_val, "stored": stored_val, "match": match}
        if note:
            r["note"] = note
        return r

    # ── Batch Number ──
    ext_batch = (extracted.get("batch_number") or "").strip()
    stored_batch = (batch_row.get("batch_number") or "") if batch_row else None
    if batch_row is None:
        results["batch_number"] = make_result(ext_batch, None, False, "Batch not found in genuine batch database")
    else:
        results["batch_number"] = make_result(ext_batch, stored_batch, ext_batch.upper() == (stored_batch or "").upper())

    # ── Manufacturer ──
    ext_mfr = (extracted.get("manufacturer") or "").strip()
    stored_mfr = (batch_row.get("manufacturer") or "") if batch_row else (medicine.get("manufacturer_name") or "")
    if ext_mfr:
        # Fuzzy match for manufacturer names (allow partial/abbreviation differences)
        ratio = difflib.SequenceMatcher(None, ext_mfr.lower(), stored_mfr.lower()).ratio()
        match = ratio >= 0.75
        results["manufacturer"] = make_result(ext_mfr, stored_mfr, match,
            None if match else f"Manufacturer name similarity: {ratio:.0%}")
    else:
        results["manufacturer"] = make_result("", stored_mfr, False, "Manufacturer not detected in OCR")

    # ── Manufacturing Date ──
    ext_mfg = normalize_date(extracted.get("mfg_date") or "")
    stored_mfg = normalize_date((batch_row.get("manufacturing_date") or "") if batch_row else "")
    if not ext_mfg:
        results["manufacturing_date"] = make_result("", stored_mfg, False, "Manufacturing date not detected")
    elif not stored_mfg:
        results["manufacturing_date"] = make_result(ext_mfg, None, False, "No stored date to compare")
    else:
        results["manufacturing_date"] = make_result(ext_mfg, stored_mfg, ext_mfg == stored_mfg)

    # ── Expiry Date ──
    ext_exp = normalize_date(extracted.get("expiry_date") or "")
    stored_exp = normalize_date((batch_row.get("expiry_date") or "") if batch_row else "")
    if not ext_exp:
        results["expiry_date"] = make_result("", stored_exp, False, "Expiry date not detected")
    elif not stored_exp:
        results["expiry_date"] = make_result(ext_exp, None, False, "No stored date to compare")
    else:
        results["expiry_date"] = make_result(ext_exp, stored_exp, ext_exp == stored_exp)

    # ── MRP ──
    ext_mrp = (extracted.get("mrp") or "").strip()
    stored_mrp = (batch_row.get("mrp") or "") if batch_row else ""
    if not ext_mrp:
        results["mrp"] = make_result("", stored_mrp, False, "MRP not detected")
    elif not stored_mrp:
        results["mrp"] = make_result(ext_mrp, None, False, "No stored MRP to compare")
    else:
        # Normalize: strip ₹ and whitespace, compare numeric value
        def parse_mrp(s):
            m = re.search(r'(\d+\.?\d*)', s.replace('₹', '').replace('Rs', ''))
            return float(m.group(1)) if m else None
        ev = parse_mrp(ext_mrp)
        sv = parse_mrp(stored_mrp)
        if ev is not None and sv is not None:
            # Allow ±1 rupee tolerance
            match = abs(ev - sv) <= 1.0
            results["mrp"] = make_result(ext_mrp, stored_mrp, match,
                None if match else f"MRP mismatch: extracted ₹{ev}, stored ₹{sv}")
        else:
            results["mrp"] = make_result(ext_mrp, stored_mrp, ext_mrp.strip() == stored_mrp.strip())

    # ── Manufacturing License ──
    ext_lic = (extracted.get("license_number") or "").strip()
    stored_lic = (batch_row.get("manufacturing_license") or "") if batch_row else ""
    if not ext_lic:
        results["license_number"] = make_result("", stored_lic, False, "License number not detected in OCR")
    elif not stored_lic:
        results["license_number"] = make_result(ext_lic, None, False, "No stored license to compare")
    else:
        match = ext_lic.upper().replace(" ", "") == stored_lic.upper().replace(" ", "")
        results["license_number"] = make_result(ext_lic, stored_lic, match)

    return results

# ─── Barcode Verification ─────────────────────────────────────────────────────

def verify_barcode(barcodes_detected, batch_row, medicine):
    """Verify barcode against database record.
    - If barcode_required is False: skip, return neutral status.
    - If barcode_required is True: decode and compare.
    """
    barcode_required = False
    if batch_row:
        barcode_required = bool(batch_row.get("barcode_required", 0))
    elif medicine:
        barcode_required = bool(medicine.get("barcode_required", 0))

    if not barcode_required:
        return {
            "required": False,
            "found": len(barcodes_detected) > 0,
            "match": None,
            "note": "Barcode not required for this pack type"
        }, None  # score contribution: None (excluded from calculation)

    # Barcode IS required
    if not barcodes_detected:
        return {
            "required": True,
            "found": False,
            "match": False,
            "note": "Barcode required but not detected on packaging"
        }, 0.0

    # Try to match any detected barcode against stored value
    stored_value = (batch_row.get("barcode_value") or "") if batch_row else ""
    for bc in barcodes_detected:
        decoded = bc.get("data", "")
        if stored_value and decoded.strip() == stored_value.strip():
            return {
                "required": True,
                "found": True,
                "match": True,
                "decoded_value": decoded,
                "stored_value": stored_value,
                "note": "Barcode verified successfully"
            }, 100.0

    # Barcodes found but none match
    decoded_values = [bc.get("data", "") for bc in barcodes_detected]
    return {
        "required": True,
        "found": True,
        "match": False,
        "decoded_value": decoded_values[0] if decoded_values else "",
        "stored_value": stored_value,
        "note": "Barcode found but does not match registered value"
    }, 0.0

# ─── Scoring Engine ───────────────────────────────────────────────────────────

def calculate_score(field_comparisons, image_score, barcode_score, medicine_name_match):
    """Weighted scoring with optional barcode contribution.

    Weights (barcode required):
      batch_number     35%
      mfg_date         15%
      expiry_date      15%
      manufacturer     10%
      medicine_name    10%
      image_analysis   10%
      barcode           5%

    If barcode is not required (barcode_score is None), redistribute its 5%
    proportionally across batch_number, mfg_date, expiry_date, manufacturer, medicine_name.
    """
    BASE_WEIGHTS = {
        "batch_number":       0.35,
        "manufacturing_date": 0.15,
        "expiry_date":        0.15,
        "manufacturer":       0.10,
        "medicine_name":      0.10,
        "image_analysis":     0.10,
        "barcode":            0.05,
    }

    # Convert field comparison results to scores (100 if match, 0 if not)
    def field_score(key):
        if key not in field_comparisons:
            return 0.0
        fc = field_comparisons[key]
        if fc.get("extracted") == "" and fc.get("stored") in (None, ""):
            return 50.0  # neutral — field simply not present
        return 100.0 if fc.get("match") else 0.0

    scores = {
        "batch_number":       field_score("batch_number"),
        "manufacturing_date": field_score("manufacturing_date"),
        "expiry_date":        field_score("expiry_date"),
        "manufacturer":       field_score("manufacturer"),
        "medicine_name":      100.0 if medicine_name_match else 0.0,
        "image_analysis":     image_score,
        "barcode":            barcode_score,  # None or float
    }

    weights = dict(BASE_WEIGHTS)

    if barcode_score is None:
        # Redistribute barcode weight proportionally among non-image, non-barcode fields
        redistribute = weights.pop("barcode")
        redistributable_keys = ["batch_number", "manufacturing_date", "expiry_date", "manufacturer", "medicine_name"]
        total_w = sum(weights[k] for k in redistributable_keys)
        for k in redistributable_keys:
            weights[k] += redistribute * (weights[k] / total_w)
        scores.pop("barcode")

    composite = sum(scores[k] * weights[k] for k in scores)
    composite = max(0.0, min(100.0, round(composite, 1)))

    # Build signal_breakdown dict for frontend display
    breakdown = {k: round(v, 1) if v is not None else None for k, v in scores.items()}
    return composite, breakdown

# ─── Main Pipeline ────────────────────────────────────────────────────────────

def run_full_pipeline(scan_id, file_path):
    """Execute the full database-driven scan pipeline with progress tracking."""
    try:
        # ── Stage 0: Preprocessing ──────────────────────────────────────────
        set_progress(scan_id, "preprocessing", 0, 0.0)
        time.sleep(0.1)

        if not os.path.exists(file_path):
            with scan_progress_lock:
                scan_progress_store[scan_id] = {
                    "scan_id": scan_id, "stage": "error", "stage_index": -1,
                    "total_stages": len(STAGES), "progress": 0.0, "status": "error",
                    "error": "Image file not found"
                }
            return

        img_cv = cv2.imread(file_path)
        if img_cv is not None:
            height, width = img_cv.shape[:2]
        else:
            width, height = 800, 600

        processed_img, preproc_anomalies = preprocess_image(file_path)
        set_progress(scan_id, "ocr_extraction", 1, 0.14)
        time.sleep(0.1)

        # ── Stage 1: OCR Extraction ─────────────────────────────────────────
        if processed_img is not None:
            temp_path = file_path + "_enhanced.jpg"
            cv2.imwrite(temp_path, processed_img)
            ocr_results = reader.readtext(temp_path)
            try:
                os.remove(temp_path)
            except Exception:
                pass
        else:
            ocr_results = reader.readtext(file_path)

        set_progress(scan_id, "barcode_decoding", 2, 0.28)
        time.sleep(0.1)

        # ── Stage 2: Barcode Decoding ───────────────────────────────────────
        barcodes_detected = extract_barcodes(file_path)

        # Build OCR boxes for frontend overlay
        ocr_boxes = []
        for bbox, text, conf in ocr_results:
            try:
                xs = [pt[0] for pt in bbox]
                ys = [pt[1] for pt in bbox]
                x_min = min(xs) / width * 100
                y_min = min(ys) / height * 100
                x_max = max(xs) / width * 100
                y_max = max(ys) / height * 100
                ocr_boxes.append({
                    "text": text,
                    "confidence": float(conf),
                    "x": round(x_min, 1),
                    "y": round(y_min, 1),
                    "w": round(x_max - x_min, 1),
                    "h": round(y_max - y_min, 1)
                })
            except Exception:
                pass

        fields, lines = extract_fields(ocr_results)
        fields["ocr_boxes"] = ocr_boxes
        fields["barcodes"] = barcodes_detected
        full_text = " ".join(lines)

        set_progress(scan_id, "visual_analysis", 3, 0.42)
        time.sleep(0.1)

        # ── Stage 3: Visual Analysis ────────────────────────────────────────
        # Will be completed after medicine lookup (needs expected_colors)
        # placeholder — actual analysis happens after medicine lookup

        set_progress(scan_id, "medicine_lookup", 4, 0.56)
        time.sleep(0.1)

        # ── Stage 4: Medicine Lookup ────────────────────────────────────────
        conn = get_db()

        matched_medicine, match_ratio = lookup_medicine(
            fields.get("name", ""), lines, full_text, conn
        )

        anomalies = list(preproc_anomalies)
        medicine_id = None
        medicine_name_match = False

        if matched_medicine and match_ratio >= 0.5:
            medicine_id = matched_medicine["id"]
            fields["name"] = matched_medicine["name"]
            if not fields["manufacturer"]:
                fields["manufacturer"] = matched_medicine["manufacturer_name"]
            medicine_name_match = True
        else:
            fields["name"] = "Unidentified Medicine"
            fields["manufacturer"] = fields.get("manufacturer") or "Unknown Manufacturer"
            anomalies.append("No matching CDSCO-registered medicine brand identified from packaging text.")

        # Run visual analysis now that we have expected_colors
        if matched_medicine:
            visual_score, vis_anomalies = analyze_visual_quality(
                img_cv if img_cv is not None else cv2.imread(file_path),
                matched_medicine.get("expected_colors", "{}")
            )
        else:
            visual_score = 35.0
            vis_anomalies = ["Cannot assess visual quality without identified medicine reference."]

        for a in vis_anomalies:
            if "deskewed" not in a.lower():
                anomalies.append(a)

        set_progress(scan_id, "batch_lookup", 5, 0.70)
        time.sleep(0.1)

        # ── Stage 5: Batch Lookup ───────────────────────────────────────────
        batch_row = None
        batch_id = None

        if medicine_id and fields.get("batch_number"):
            batch_row = lookup_batch(medicine_id, fields["batch_number"], conn)
            if batch_row:
                batch_id = batch_row["id"]
                if batch_row.get("status") == "recalled":
                    anomalies.append(
                        f"⚠️ BATCH RECALLED: Batch {fields['batch_number']} has been officially recalled. "
                        "Do not use this product."
                    )
            else:
                anomalies.append(
                    f"Batch '{fields['batch_number']}' not found in genuine batch database for {fields['name']}."
                )
        elif medicine_id and not fields.get("batch_number"):
            anomalies.append("Batch number not detected on packaging. Field may be obscured or absent.")

        set_progress(scan_id, "field_comparison", 6, 0.82)
        time.sleep(0.1)

        # ── Stage 6: Field Comparison ───────────────────────────────────────
        if matched_medicine:
            field_comparisons = compare_fields(fields, batch_row, matched_medicine)
        else:
            # All fields fail if medicine not identified
            field_comparisons = {
                k: {"extracted": fields.get(k, ""), "stored": None, "match": False}
                for k in ["batch_number", "manufacturer", "manufacturing_date", "expiry_date", "mrp", "license_number"]
            }

        # Barcode verification
        barcode_status, barcode_score = verify_barcode(barcodes_detected, batch_row, matched_medicine)

        # Community alert check
        community_note = None
        if medicine_id and fields.get("batch_number"):
            cur = conn.cursor()
            alert = cur.execute(
                "SELECT report_count FROM alerts WHERE medicine_id=? AND batch_number=?",
                (medicine_id, fields["batch_number"])
            ).fetchone()
            if alert:
                rc = alert["report_count"]
                if rc >= 3:
                    anomalies.append(
                        f"🚨 Active community alert: {rc} independent reports confirm this batch as counterfeit."
                    )
                    # Apply community penalty to visual score (existing behaviour)
                    visual_score = min(visual_score, 20.0)
                elif rc >= 1:
                    anomalies.append(
                        f"⚠️ Community caution: {rc} suspect report(s) filed for batch {fields['batch_number']}."
                    )

        conn.close()

        # ── Stage 7: Scoring ────────────────────────────────────────────────
        set_progress(scan_id, "scoring", 7, 0.94)
        time.sleep(0.1)

        composite, signal_breakdown = calculate_score(
            field_comparisons, visual_score, barcode_score, medicine_name_match
        )

        image_analysis_result = {
            "score": round(visual_score, 1),
            "anomalies": [a for a in anomalies if any(kw in a.lower() for kw in [
                "blur", "color", "edge", "saturation", "jpeg", "deskew"
            ])]
        }

        result = {
            "medicine_id": medicine_id,
            "batch_id": batch_id,
            "authenticity_score": composite,
            "ocr_extracted": fields,
            "db_match_results": field_comparisons,
            "image_analysis": image_analysis_result,
            "barcode_status": barcode_status,
            "anomalies": anomalies,
            "signal_breakdown": signal_breakdown
        }

        with scan_progress_lock:
            scan_progress_store[scan_id] = {
                "scan_id": scan_id, "stage": "complete", "stage_index": 8,
                "total_stages": len(STAGES), "progress": 1.0, "status": "complete",
                "result": result
            }

    except Exception as e:
        print(f"Pipeline error for {scan_id}: {e}")
        import traceback
        traceback.print_exc()
        with scan_progress_lock:
            scan_progress_store[scan_id] = {
                "scan_id": scan_id, "stage": "error", "stage_index": -1,
                "total_stages": len(STAGES), "progress": 0.0, "status": "error",
                "error": str(e)
            }

# ─── API Endpoints ────────────────────────────────────────────────────────────

@app.post("/process_scan")
def process_scan(req: ScanRequest):
    """Start async scan processing and return immediately."""
    if not os.path.exists(req.file_path):
        raise HTTPException(status_code=404, detail="Image file not found")

    thread = threading.Thread(target=run_full_pipeline, args=(req.scan_id, req.file_path))
    thread.daemon = True
    thread.start()

    return {
        "scan_id": req.scan_id,
        "status": "started",
        "message": "Scan pipeline initiated. Poll /scan_progress/{scan_id} for updates."
    }

@app.get("/scan_progress/{scan_id}")
def get_scan_progress(scan_id: str):
    """Get current progress of an async scan."""
    with scan_progress_lock:
        progress = scan_progress_store.get(scan_id)
        if progress is None:
            return {"scan_id": scan_id, "status": "not_found", "progress": 0.0}
        return progress

@app.get("/health")
def health():
    return {"status": "healthy", "service": "ml", "version": "4.0.0"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
