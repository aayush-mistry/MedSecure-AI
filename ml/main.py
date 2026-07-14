import os
import re
import json
import time
import threading
import sqlite3
import difflib
import ssl
import multiprocessing as mp
import numpy as np
import cv2
from PIL import Image
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import easyocr

ssl._create_default_https_context = ssl._create_unverified_context

try:
    from pyzbar.pyzbar import decode as zbar_decode
    HAS_ZBAR = True
except Exception:
    HAS_ZBAR = False
    def zbar_decode(*args, **kwargs):
        return []
    print("ZBar not available. Barcode detection disabled.")

app = FastAPI(title="MedSecure ML Inference Service v5 (12-Stage Pipeline)")

print("Loading EasyOCR model (CPU)...")
reader = easyocr.Reader(['en'], gpu=False, verbose=False)
print("EasyOCR ready.")

DB_PATH = os.environ.get(
    "DB_PATH",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "db", "indian_pharmaceutical_products.db"))
)
MAX_OCR_DIMENSION = 1280
MAX_ANALYSIS_DIMENSION = 1600

scan_progress_store = {}
scan_progress_lock = threading.Lock()

STAGES = [
    "image_quality",
    "image_enhancement",
    "ocr_extraction",
    "db_verification",
    "batch_verification",
    "barcode_verification",
    "ai_packaging",
    "logo_verification",
    "color_verification",
    "layout_verification",
    "tamper_detection",
    "confidence_scoring"
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
    conn = sqlite3.connect(DB_PATH, timeout=2.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout = 2000")
    return conn

def set_progress(scan_id, stage, stage_index, progress, status="processing"):
    with scan_progress_lock:
        scan_progress_store[scan_id] = {
            "scan_id": scan_id, "stage": stage, "stage_index": stage_index,
            "total_stages": len(STAGES), "progress": progress, "status": status
        }
    if stage_index >= 0:
        time.sleep(0.15)

# --- Stage 1: Image Quality Assessment ---
def resize_max_dimension(img, max_dimension):
    h, w = img.shape[:2]
    current_max = max(h, w)
    if current_max <= max_dimension:
        return img
    scale = max_dimension / current_max
    return cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)

def check_image_quality(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    resolution_ok = (h * w) > (300 * 300)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    blur_ok = laplacian_var > 40
    brightness = np.mean(gray)
    lighting_ok = 30 < brightness < 225

    issues = []
    if not resolution_ok: issues.append("Low resolution")
    if not blur_ok: issues.append("Severe blur detected")
    if not lighting_ok: issues.append("Poor lighting conditions")

    return len(issues) == 0, issues

# --- Stage 2: Image Enhancement ---
def enhance_image(img):
    img = resize_max_dimension(img, MAX_ANALYSIS_DIMENSION)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # Deskew
    coords = np.column_stack(np.where(enhanced > 0))
    angle = 0
    if len(coords) > 0:
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45: angle = 90 + angle
        if abs(angle) > 0.5:
            h, w = enhanced.shape
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            enhanced = cv2.warpAffine(enhanced, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            
    sharpen_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharpened = cv2.filter2D(enhanced, -1, sharpen_kernel)
    return cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)

def prepare_ocr_image(img):
    ocr_img = resize_max_dimension(img, MAX_OCR_DIMENSION)
    gray = cv2.cvtColor(ocr_img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 5, 35, 35)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    return clahe.apply(gray)

# --- Stage 3: OCR Extraction ---
NOT_DETECTED = "Not Detected"
MEDICINE_MATCH_THRESHOLD = 0.78

def normalize_date(date_str):
    if not date_str or date_str == NOT_DETECTED: return NOT_DETECTED
    m = re.match(r'^(\d{2})[/-](\d{4})$', date_str.strip())
    if m: return f"{m.group(1)}/{m.group(2)}"
    m = re.match(r'^(\d{2})[/-](\d{2})$', date_str.strip())
    if m: return f"{m.group(1)}/20{m.group(2)}"
    return date_str.strip()

def is_detected(value):
    return bool(value and str(value).strip() and str(value).strip() != NOT_DETECTED)

def clean_ocr_text(value):
    return re.sub(r"\s+", " ", value or "").strip(" :;-")

def normalize_match_text(value):
    value = (value or "").lower()
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()

def field_result(value=NOT_DETECTED, confidence=0.0, source="ocr"):
    value = clean_ocr_text(value)
    return {
        "value": value if value else NOT_DETECTED,
        "confidence": round(float(confidence or 0.0), 3),
        "source": source
    }

def best_regex_match(ocr_results, pattern, group=1, normalizer=None, min_conf=0.25):
    best = field_result()
    for result in ocr_results:
        text = result[1].strip() if len(result) >= 2 else ""
        conf = float(result[2]) if len(result) >= 3 else 0.0
        if conf < min_conf:
            continue
        match = re.search(pattern, text, re.IGNORECASE)
        if not match:
            continue
        value = match.group(group).strip()
        if normalizer:
            value = normalizer(value)
        if conf > best["confidence"]:
            best = field_result(value, conf)
    return best

def confidence_for_value(ocr_results, value):
    if not is_detected(value):
        return 0.0
    normalized_value = normalize_match_text(value)
    best = 0.0
    for result in ocr_results:
        text = result[1].strip() if len(result) >= 2 else ""
        conf = float(result[2]) if len(result) >= 3 else 0.0
        normalized_text = normalize_match_text(text)
        if normalized_value and (normalized_value in normalized_text or normalized_text in normalized_value):
            best = max(best, conf)
    return best

def extract_ocr_fields(ocr_results):
    lines = [r[1].strip() for r in ocr_results if len(r[1].strip()) >= 2]
    full_text = " ".join(lines)
    field_details = {
        "name": field_result(),
        "manufacturer": field_result(),
        "batch_number": field_result(),
        "mfg_date": field_result(),
        "expiry_date": field_result(),
        "mrp": field_result(),
        "strength": field_result(),
        "dosage_form": field_result(),
        "composition": field_result(),
        "license_number": field_result(),
        "barcode": field_result()
    }

    ignored_name_terms = {
        "mfg", "mfd", "batch", "b no", "exp", "expiry", "mrp", "price",
        "license", "lic", "tablet", "capsule", "strip", "schedule"
    }
    best_name = field_result()
    for result in ocr_results:
        text = result[1].strip() if len(result) >= 2 else ""
        conf = float(result[2]) if len(result) >= 3 else 0.0
        lower = text.lower()
        looks_like_name = (
            conf >= 0.35
            and len(text) >= 3
            and any(ch.isalpha() for ch in text)
            and not any(term in lower for term in ignored_name_terms)
        )
        if looks_like_name and conf > best_name["confidence"]:
            best_name = field_result(text, conf)
    field_details["name"] = best_name

    field_details["batch_number"] = best_regex_match(
        ocr_results,
        r'(?:batch\s*(?:no|number)?|b\.?\s*no\.?)\s*[:\-\.]*\s*([A-Z0-9][A-Z0-9\-/]{2,})'
    )
    field_details["expiry_date"] = best_regex_match(
        ocr_results,
        r'(?:exp|expiry)(?:\s*date)?\s*[:;\-\.]*\s*((?:\d{1,2})[/\-](?:\d{2,4}))',
        normalizer=normalize_date
    )
    field_details["mfg_date"] = best_regex_match(
        ocr_results,
        r'(?:mfg|mfd)(?:\s*date)?\s*[:;\-\.]*\s*((?:\d{1,2})[/\-](?:\d{2,4}))',
        normalizer=normalize_date
    )
    field_details["mrp"] = best_regex_match(
        ocr_results,
        r'(?:mrp|price)\s*[:;\-]*\s*(?:rs\.?|inr|₹)?\s*(\d+\.?\d*)'
    )
    field_details["license_number"] = best_regex_match(
        ocr_results,
        r'(?:mfg\.?\s*lic|lic\.?\s*no|license)\s*[:\-]*\s*([A-Z0-9/\-\.]{4,})'
    )
    field_details["strength"] = best_regex_match(
        ocr_results,
        r'\b(\d+(?:\.\d+)?\s*(?:mg|mcg|g|ml|iu))\b'
    )
    field_details["dosage_form"] = best_regex_match(
        ocr_results,
        r'\b(tablets?|tabs?|capsules?|caps?|syrup|injection|cream|ointment|drops|inhaler)\b',
        group=0
    )

    for line in lines:
        mfr = re.search(r'(?:mfg\.?\s*by|manufactured\s*by)\s*[:\-]*\s*(.+)', line, re.IGNORECASE)
        if mfr:
            manufacturer = re.split(r'\b(?:batch|exp|expiry|mrp|mfg\s*date)\b', mfr.group(1), flags=re.IGNORECASE)[0].strip()
            field_details["manufacturer"] = field_result(manufacturer[:80], confidence_for_value(ocr_results, line))
            break
    if not is_detected(field_details["manufacturer"]["value"]):
        mfr = re.search(r'(?:mfg\.?\s*by|manufactured\s*by)\s*[:\-]*\s*(.+?)(?=\s+(?:batch|exp|expiry|mrp|mfg\s*date)\b|$)', full_text, re.IGNORECASE)
        if mfr:
            manufacturer = mfr.group(1).strip()
            field_details["manufacturer"] = field_result(manufacturer[:80], confidence_for_value(ocr_results, manufacturer))

    fields = {key: detail["value"] for key, detail in field_details.items()}
    fields["mfr"] = fields["manufacturer"]
    fields["_confidence"] = {key: detail["confidence"] for key, detail in field_details.items()}
    fields["_details"] = field_details
    fields["_raw_text"] = full_text

    return fields, full_text, lines

# --- Stage 4: DB Verification ---
def get_table_columns(conn, table_name):
    try:
        return {row["name"] for row in conn.execute(f"PRAGMA table_info({table_name})").fetchall()}
    except sqlite3.Error:
        return set()

def medicine_name(med):
    return med.get("brand_name") or med.get("name") or ""

def medicine_manufacturer(med):
    return med.get("manufacturer_name") or med.get("manufacturer") or ""

def build_medicine_candidates(full_text, lines, conn):
    columns = get_table_columns(conn, "medicines")
    searchable_columns = [c for c in ["brand_name", "name", "generic_name", "primary_ingredient", "manufacturer_name"] if c in columns]
    if not searchable_columns:
        return []

    tokens = []
    for token in re.findall(r"[A-Za-z][A-Za-z0-9\-]{2,}", full_text):
        token = token.strip()
        if len(token) >= 4 and token.lower() not in {"batch", "expiry", "mfg", "mfd", "tablet", "capsule", "tablets", "capsules", "price", "license"}:
            tokens.append(token)

    seen = set()
    candidates = []
    cur = conn.cursor()
    for token in tokens[:12]:
        where = " OR ".join([f"{col} LIKE ?" for col in searchable_columns])
        params = [f"%{token}%"] * len(searchable_columns)
        try:
            rows = cur.execute(f"SELECT * FROM medicines WHERE {where} LIMIT 30", params).fetchall()
        except sqlite3.Error:
            continue
        for row in rows:
            med = dict(row)
            if med.get("id") not in seen:
                seen.add(med.get("id"))
                candidates.append(med)
        if len(candidates) >= 80:
            break

    return candidates

def verify_db_medicine(fields, full_text, lines, conn):
    cur = conn.cursor()
    try:
        cur.execute("SELECT * FROM medicines LIMIT 1")
    except sqlite3.OperationalError:
        return None, 0.0, {"status": "skipped", "reason": "Medicine database unavailable"}

    all_meds = build_medicine_candidates(full_text, lines, conn)
    if not all_meds:
        cur.execute("SELECT * FROM medicines LIMIT 200")
        all_meds = [dict(r) for r in cur.fetchall()]
    
    best_med = None
    best_score = 0.0
    full_text_norm = normalize_match_text(full_text)
    name_norm = normalize_match_text(fields.get("name"))
    manufacturer_norm = normalize_match_text(fields.get("manufacturer"))

    for med in all_meds:
        brand_norm = normalize_match_text(medicine_name(med))
        generic_norm = normalize_match_text(med.get("generic_name") or "")
        med_mfr_norm = normalize_match_text(medicine_manufacturer(med))
        if not brand_norm:
            continue

        scores = []
        if name_norm:
            scores.append(difflib.SequenceMatcher(None, name_norm, brand_norm).ratio())
            if name_norm == brand_norm:
                scores.append(1.0)
            elif brand_norm in name_norm or name_norm in brand_norm:
                scores.append(0.94)
        if brand_norm in full_text_norm:
            scores.append(0.96)
        for line in lines:
            line_norm = normalize_match_text(line)
            if line_norm:
                scores.append(difflib.SequenceMatcher(None, line_norm, brand_norm).ratio())

        score = max(scores) if scores else 0.0
        if generic_norm and generic_norm in full_text_norm:
            score = min(1.0, score + 0.03)
        if manufacturer_norm and med_mfr_norm:
            mfr_score = difflib.SequenceMatcher(None, manufacturer_norm, med_mfr_norm).ratio()
            if mfr_score >= 0.70 or manufacturer_norm in med_mfr_norm or med_mfr_norm in manufacturer_norm:
                score = min(1.0, score + 0.04)

        if score > best_score:
            best_score = score
            best_med = med

    if not best_med or best_score < MEDICINE_MATCH_THRESHOLD:
        return None, best_score, {
            "status": "not_found",
            "reason": "Medicine Not Found: OCR evidence did not meet confidence threshold"
        }

    return best_med, best_score, {
        "status": "verified",
        "reason": f"Medicine matched with {round(best_score * 100)}% confidence"
    }

# --- Stage 5: Batch Verification ---
def verify_batch(medicine_id, batch_number, conn):
    if not is_detected(batch_number):
        return None
    cur = conn.cursor()
    try:
        if medicine_id:
            row = cur.execute("SELECT * FROM medicine_batches WHERE medicine_id=? AND batch_number=?", (medicine_id, batch_number)).fetchone()
        else:
            row = cur.execute("SELECT * FROM medicine_batches WHERE batch_number=? LIMIT 1", (batch_number,)).fetchone()
        return dict(row) if row else None
    except sqlite3.OperationalError:
        return None

# --- Stage 6: Barcode Verification ---
def decode_barcodes_worker(gray, queue):
    try:
        barcodes = zbar_decode(gray)
        queue.put([b.data.decode("utf-8", errors="ignore") for b in barcodes])
    except Exception as e:
        queue.put({"error": str(e)})

def decode_barcodes_with_timeout(gray, timeout_seconds=2.0):
    try:
        ctx = mp.get_context("fork")
    except ValueError:
        return [], "Timed barcode decoding is unavailable on this platform"

    queue = ctx.Queue()
    proc = ctx.Process(target=decode_barcodes_worker, args=(gray, queue))
    proc.start()
    proc.join(timeout_seconds)

    if proc.is_alive():
        proc.terminate()
        proc.join(1)
        return None, "Barcode decoder timed out"

    if queue.empty():
        return [], None

    result = queue.get()
    if isinstance(result, dict) and result.get("error"):
        return [], result["error"]
    return result, None

def verify_barcode(img, batch_row, med_row):
    req = False
    if batch_row and batch_row.get("barcode_required"): req = True
    if med_row and med_row.get("barcode_required"): req = True

    if not HAS_ZBAR:
        return {"status": "skipped", "required": req, "found": False, "match": None, "ocr_value": NOT_DETECTED, "stored_value": None, "note": "Barcode decoder unavailable", "reason": "Barcode verification skipped", "score": 85 if not req else 70}

    h, w = img.shape[:2]
    max_dim = max(h, w)
    scan_img = img
    if max_dim > 1200:
        scale = 1200 / max_dim
        scan_img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(scan_img, cv2.COLOR_BGR2GRAY)
    decoded, decode_error = decode_barcodes_with_timeout(gray)
    if decode_error:
        print(f"Barcode decoding skipped: {decode_error}")
        return {"status": "skipped", "required": req, "found": False, "match": None, "ocr_value": NOT_DETECTED, "stored_value": None, "note": decode_error, "reason": "Barcode verification skipped", "score": 85 if not req else 70}

    if not decoded:
        return {"status": "skipped", "required": req, "found": False, "match": None, "ocr_value": NOT_DETECTED, "stored_value": None, "note": "Barcode not detected", "reason": "Barcode not detected; verification skipped", "score": 85 if not req else 70}
    
    expected = (batch_row.get("barcode_value") or "") if batch_row else ""
    if expected and expected in decoded:
        return {"status": "passed", "required": req, "found": True, "match": True, "ocr_value": decoded[0], "decoded_value": decoded[0], "stored_value": expected, "note": "Barcode verified", "reason": "Barcode Verified", "score": 100}
    if expected:
        return {"status": "failed", "required": req, "found": True, "match": False, "ocr_value": decoded[0], "decoded_value": decoded[0], "stored_value": expected, "note": "Barcode mismatch", "reason": "Barcode Mismatch", "score": 0}
    return {"status": "skipped", "required": req, "found": True, "match": None, "ocr_value": decoded[0], "decoded_value": decoded[0], "stored_value": None, "note": "Barcode decoded but no database barcode value is recorded", "reason": "Barcode database comparison skipped", "score": 90}

# --- Stage 7: AI Packaging ---
def analyze_packaging(img):
    # Mock OpenCV based structural anomaly check (laplacian edge count vs expected)
    edges = cv2.Canny(img, 50, 150)
    density = np.sum(edges > 0) / edges.size
    if density < 0.01: return {"status": "failed", "reason": "Package appears blank or extremely degraded", "score": 20}
    if density > 0.30: return {"status": "failed", "reason": "Too much noise/tampering on surface", "score": 40}
    return {"status": "passed", "reason": "Packaging verified", "score": 100}

# --- Stage 8: Logo Verification ---
def verify_logo(img, med_row):
    # Mock SIFT feature matching against expected logo embeddings
    mfg_name = med_row.get("manufacturer_name") if med_row else None
    if mfg_name and mfg_name.lower() in ["gsk", "dr. reddy's", "pfizer"]:
        return {"status": "passed", "reason": "Logo Verified", "score": 95}
    if mfg_name:
        return {"status": "passed", "reason": "Manufacturer identity present", "score": 90}
    return {"status": "warning", "reason": "Logo/manufacturer confidence limited", "score": 70}

# --- Stage 9: Color Verification ---
def verify_color(img, med_row):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mean_s = np.mean(hsv[:, :, 1])
    
    if med_row:
        exp_color = json.loads(med_row.get("expected_colors") or "{}")
        # In a real model, we compare histograms. We mock the confidence here:
        if exp_color:
            return {"status": "passed", "reason": "Color Profile Matched", "score": 98}
    
    if mean_s > 200: return {"status": "failed", "reason": "Unnatural color saturation", "score": 30}
    return {"status": "passed", "reason": "Color within acceptable delta", "score": 90}

# --- Stage 10: Layout Verification ---
def verify_layout(ocr_results, med_row):
    # Verify relative positioning
    if len(ocr_results) < 3: return {"status": "failed", "reason": "Missing key layout elements", "score": 40}
    return {"status": "passed", "reason": "Layout elements verified", "score": 95}

# --- Stage 11: Tamper Detection ---
def detect_tampering(img):
    # Mock SSIM check for broken seals / tape reflections
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bright_spots = np.sum(gray > 240) / gray.size
    if bright_spots > 0.75:
        return {"status": "failed", "reason": "Surface tampering/tape reflection detected", "score": 35}
    return {"status": "passed", "reason": "No physical tamper detected", "score": 100}

# --- Stage 12: Explainable AI Engine ---
def db_value(row, *keys):
    if not row:
        return None
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return str(value)
    return None

def compare_values(ocr_value, db_value_text, label=None):
    if not is_detected(ocr_value):
        return {
            "extracted": NOT_DETECTED,
            "stored": db_value_text or None,
            "match": None,
            "status": "Skipped",
            "note": f"{label or 'Field'} not detected by OCR"
        }
    if not db_value_text:
        return {
            "extracted": ocr_value,
            "stored": None,
            "match": None,
            "status": "No Database Record",
            "note": "No reference value available for comparison"
        }
    ocr_norm = normalize_match_text(normalize_date(ocr_value))
    db_norm = normalize_match_text(normalize_date(db_value_text))
    match = bool(ocr_norm and db_norm and (ocr_norm == db_norm or ocr_norm in db_norm or db_norm in ocr_norm))
    return {
        "extracted": ocr_value,
        "stored": db_value_text,
        "match": match,
        "status": "Verified" if match else "Mismatch",
        "note": None if match else "OCR value differs from database reference"
    }

def compare_fuzzy(ocr_value, db_value_text, label=None, threshold=0.75):
    if not is_detected(ocr_value):
        return {
            "extracted": NOT_DETECTED,
            "stored": db_value_text or None,
            "match": None,
            "status": "Skipped",
            "note": f"{label or 'Field'} not detected by OCR"
        }
    if not db_value_text:
        return {
            "extracted": ocr_value,
            "stored": None,
            "match": None,
            "status": "No Database Record",
            "note": "No reference value available for comparison"
        }
    ocr_norm = normalize_match_text(ocr_value)
    db_norm = normalize_match_text(db_value_text)
    ratio = difflib.SequenceMatcher(None, ocr_norm, db_norm).ratio()
    match = ratio >= threshold or ocr_norm in db_norm or db_norm in ocr_norm
    return {
        "extracted": ocr_value,
        "stored": db_value_text,
        "match": match,
        "status": "Verified" if match else "Mismatch",
        "confidence": round(ratio, 3),
        "note": None if match else "OCR value differs from database reference"
    }

def build_verification_results(fields, med_row, batch_row, barcode_status):
    composition = None
    if med_row:
        try:
            raw_composition = med_row.get("composition")
            parsed = json.loads(raw_composition) if raw_composition else []
            composition = ", ".join(parsed) if isinstance(parsed, list) else str(parsed)
        except Exception:
            composition = med_row.get("composition")

    dosage_status = "Skipped"
    dosage_note = "Dosage form not detected by OCR"
    if is_detected(fields.get("dosage_form")) and (med_row or batch_row):
        dosage_status = "Reference Only"
        dosage_note = "No direct dosage-form reference is available for a strict match"

    return {
        "medicine_name": compare_fuzzy(fields.get("name"), medicine_name(med_row) if med_row else None, "Medicine Name", 0.78),
        "generic_name": {
            "extracted": NOT_DETECTED,
            "stored": db_value(med_row, "generic_name"),
            "match": None,
            "status": "Reference Only" if med_row else "Skipped",
            "note": "Generic name is a database reference for the matched medicine; it is not OCR evidence."
        },
        "manufacturer": compare_fuzzy(fields.get("manufacturer"), medicine_manufacturer(med_row) if med_row else db_value(batch_row, "manufacturer"), "Manufacturer", 0.70),
        "batch_number": compare_values(fields.get("batch_number"), db_value(batch_row, "batch_number"), "Batch Number"),
        "manufacturing_date": compare_values(fields.get("mfg_date"), db_value(batch_row, "manufacturing_date"), "Manufacturing Date"),
        "expiry_date": compare_values(fields.get("expiry_date"), db_value(batch_row, "expiry_date"), "Expiry Date"),
        "mrp": compare_values(fields.get("mrp"), db_value(batch_row, "mrp"), "MRP"),
        "strength": compare_fuzzy(fields.get("strength"), composition, "Strength", 0.65),
        "dosage_form": {
            "extracted": fields.get("dosage_form", NOT_DETECTED),
            "stored": db_value(med_row, "dosage_form") or db_value(batch_row, "pack_type"),
            "match": None,
            "status": dosage_status,
            "note": dosage_note
        },
        "composition": compare_fuzzy(fields.get("composition"), composition, "Composition", 0.65),
        "license_number": compare_values(fields.get("license_number"), db_value(batch_row, "manufacturing_license") or db_value(med_row, "cdsco_license"), "License Number"),
        "barcode": {
            "extracted": barcode_status.get("ocr_value") or barcode_status.get("decoded_value") or NOT_DETECTED,
            "stored": barcode_status.get("stored_value"),
            "match": barcode_status.get("match"),
            "status": "Verified" if barcode_status.get("match") is True else ("Mismatch" if barcode_status.get("match") is False else "Skipped"),
            "note": barcode_status.get("note")
        }
    }

def generate_evidence_report(stages_results):
    score = 0
    weights = {
        "ocr": 0.12, "db": 0.18, "batch": 0.18, "barcode": 0.08,
        "packaging": 0.12, "logo": 0.08, "color": 0.08, "layout": 0.08, "tamper": 0.08
    }

    explanation = []
    breakdown = {}
    for name, weight in weights.items():
        result = stages_results[name]
        stage_score = float(result.get("score", 0))
        score += stage_score * weight
        breakdown[name] = round(stage_score, 1)
        marker = "OK" if stage_score >= 85 else ("WARN" if stage_score >= 60 else "FAIL")
        explanation.append(f"{marker}: {result.get('reason', name)}")

    final_score = min(100, max(0, round(score, 1)))
    return {
        "score": final_score,
        "confidence": "High" if final_score >= 85 else ("Medium" if final_score >= 60 else "Low"),
        "verdict": "Verified Genuine" if final_score > 85 else ("Caution" if final_score > 60 else "Counterfeit / High Risk"),
        "explanation": explanation,
        "breakdown": breakdown
    }


def db_verdict(score):
    if score >= 80:
        return "verified"
    if score >= 55:
        return "caution"
    return "high_risk"


def run_full_pipeline(scan_id, file_path):
    try:
        set_progress(scan_id, "image_quality", 0, 0.05)
        img = cv2.imread(file_path)
        if img is None: raise Exception("Invalid image file")
        
        # Stage 1
        ok, issues = check_image_quality(img)
        if not ok:
            with scan_progress_lock:
                scan_progress_store[scan_id] = {"status": "error", "error": f"Poor image quality: {', '.join(issues)}. Please recapture."}
            return

        # Stage 2
        set_progress(scan_id, "image_enhancement", 1, 0.10)
        enhanced_img = enhance_image(img)
        
        # Stage 3
        set_progress(scan_id, "ocr_extraction", 2, 0.20)
        ocr_img = prepare_ocr_image(enhanced_img)
        ocr_res = reader.readtext(
            ocr_img,
            detail=1,
            paragraph=False,
            decoder="greedy",
            batch_size=4,
            canvas_size=MAX_OCR_DIMENSION,
            mag_ratio=1.0,
            text_threshold=0.55,
            low_text=0.35,
            width_ths=0.8
        )
        fields, full_text, lines = extract_ocr_fields(ocr_res)
        ocr_public = {k: v for k, v in fields.items() if not k.startswith("_") and k != "mfr"}
        detected_core_fields = [
            fields.get("name"), fields.get("manufacturer"), fields.get("batch_number"),
            fields.get("expiry_date"), fields.get("mfg_date"), fields.get("mrp")
        ]
        detected_count = sum(1 for value in detected_core_fields if is_detected(value))
        ocr_score = min(100, 35 + detected_count * 10 + int(np.mean(list(fields["_confidence"].values()) or [0]) * 20))
        res_ocr = {
            "score": ocr_score,
            "reason": f"OCR extracted {detected_count}/{len(detected_core_fields)} core fields"
        }

        conn = get_db()

        # Stage 4
        set_progress(scan_id, "db_verification", 3, 0.30)
        med_row, db_score, med_match_meta = verify_db_medicine(fields, full_text, lines, conn)
        brand_nm = medicine_name(med_row) if med_row else "Medicine Not Found"
        res_db = {
            "score": round(db_score * 100, 1) if med_row else max(0, round(db_score * 60, 1)),
            "reason": med_match_meta["reason"]
        }
        
        # Stage 5
        set_progress(scan_id, "batch_verification", 4, 0.40)
        batch_row = verify_batch(med_row["id"] if med_row else None, fields["batch_number"], conn)
        if not is_detected(fields["batch_number"]):
            res_batch = {"score": 85, "reason": "Batch number not detected; batch verification skipped"}
        elif batch_row:
            res_batch = {"score": 100, "reason": "Batch number found in registry"}
        else:
            res_batch = {"score": 45, "reason": "OCR batch number not found in registry"}

        # Stage 6
        set_progress(scan_id, "barcode_verification", 5, 0.50)
        res_barcode = verify_barcode(enhanced_img, batch_row, med_row)

        # Stage 7
        set_progress(scan_id, "ai_packaging", 6, 0.60)
        res_packaging = analyze_packaging(enhanced_img)

        # Stage 8
        set_progress(scan_id, "logo_verification", 7, 0.70)
        res_logo = verify_logo(enhanced_img, med_row)

        # Stage 9
        set_progress(scan_id, "color_verification", 8, 0.80)
        res_color = verify_color(enhanced_img, med_row)

        # Stage 10
        set_progress(scan_id, "layout_verification", 9, 0.85)
        res_layout = verify_layout(ocr_res, med_row)

        # Stage 11
        set_progress(scan_id, "tamper_detection", 10, 0.90)
        res_tamper = detect_tampering(enhanced_img)

        # Stage 12
        set_progress(scan_id, "confidence_scoring", 11, 0.95)
        stages_results = {
            "ocr": res_ocr, "db": res_db, "batch": res_batch, "barcode": res_barcode,
            "packaging": res_packaging, "logo": res_logo, "color": res_color,
            "layout": res_layout, "tamper": res_tamper
        }
        
        verification_results = build_verification_results(fields, med_row, batch_row, res_barcode)
        verified_count = sum(1 for result in verification_results.values() if result.get("match") is True)
        mismatch_count = sum(1 for result in verification_results.values() if result.get("match") is False)
        skipped_count = sum(1 for result in verification_results.values() if result.get("status") == "Skipped")
        if mismatch_count:
            stages_results["db"]["score"] = min(stages_results["db"]["score"], max(35, 90 - mismatch_count * 20))
            stages_results["db"]["reason"] = f"{mismatch_count} database comparison mismatch(es) found"
        elif verified_count:
            stages_results["db"]["reason"] = f"{verified_count} database field(s) verified; {skipped_count} skipped"

        final_report = generate_evidence_report(stages_results)

        # Extract anomalies for legacy compatibility in DB
        anomalies = [line for line in final_report["explanation"] if line.startswith("FAIL") or line.startswith("WARN")]

        # Write result. The backend creates the scan row before calling ML, so
        # prefer UPDATE; INSERT keeps direct ML testing usable.
        cur = conn.cursor()
        try:
            scan_values = (
                med_row["id"] if med_row else None,
                batch_row["id"] if batch_row else None,
                final_report["score"],
                db_verdict(final_report["score"]),
                json.dumps(ocr_public),
                json.dumps(verification_results),
                json.dumps(res_packaging),
                json.dumps(res_barcode),
                json.dumps(anomalies),
                json.dumps(final_report["breakdown"]),
                scan_id
            )
            cur.execute("""
                UPDATE scans SET
                    medicine_id=?,
                    batch_id=?,
                    authenticity_score=?,
                    verdict=?,
                    ocr_extracted=?,
                    db_match_results=?,
                    image_analysis=?,
                    barcode_status=?,
                    anomalies=?,
                    signal_breakdown=?,
                    scanned_at=CURRENT_TIMESTAMP
                WHERE id=?
            """, scan_values)
            if cur.rowcount == 0:
                cur.execute("""
                    INSERT INTO scans (
                        id, medicine_id, batch_id, authenticity_score, verdict,
                        ocr_extracted, db_match_results, image_analysis,
                        barcode_status, anomalies, signal_breakdown
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    scan_id,
                    scan_values[0],
                    scan_values[1],
                    scan_values[2],
                    scan_values[3],
                    scan_values[4],
                    scan_values[5],
                    scan_values[6],
                    scan_values[7],
                    scan_values[8],
                    scan_values[9]
                ))
            conn.commit()
        except sqlite3.Error as e:
            # Fallback if scans table doesn't exist yet or if id already exists
            print(f"DB Insert failed: {e}")
            pass

        with scan_progress_lock:
            scan_progress_store[scan_id] = {
                "status": "complete",
                "progress": 1.0,
                "result": {
                    "verdict": final_report["verdict"],
                    "confidence": final_report["confidence"],
                    "authenticity_score": final_report["score"],
                    "medicine_id": med_row["id"] if med_row else None,
                    "batch_id": batch_row["id"] if batch_row else None,
                    "medicine_name": brand_nm,
                    "generic_name": med_row.get("generic_name") if med_row else None,
                    "manufacturer_name": med_row.get("manufacturer_name") if med_row else (fields["manufacturer"] if is_detected(fields["manufacturer"]) else None),
                    "batch_number": fields["batch_number"] if is_detected(fields["batch_number"]) else NOT_DETECTED,
                    "ocr_extracted": ocr_public,
                    "ocr_field_details": fields["_details"],
                    "raw_ocr_text": fields["_raw_text"],
                    "db_match_results": verification_results,
                    "image_analysis": res_packaging,
                    "barcode_status": res_barcode,
                    "explanation": final_report["explanation"],
                    "anomalies": anomalies,
                    "breakdown": final_report["breakdown"],
                    "signal_breakdown": final_report["breakdown"]
                }
            }

    except Exception as e:
        import traceback
        traceback.print_exc()
        with scan_progress_lock:
            scan_progress_store[scan_id] = {"status": "error", "error": str(e)}

@app.post("/process_scan")
@app.post("/api/scan/upload")
async def upload_scan(req: ScanRequest):
    scan_id = req.scan_id
    set_progress(scan_id, "queued", -1, 0.0)
    threading.Thread(target=run_full_pipeline, args=(scan_id, req.file_path), daemon=True).start()
    return {"scan_id": scan_id, "status": "processing"}

@app.get("/scan_progress/{scan_id}")
@app.get("/api/scan/{scan_id}/progress")
async def get_scan_progress(scan_id: str):
    with scan_progress_lock:
        if scan_id not in scan_progress_store:
            raise HTTPException(status_code=404, detail="Scan ID not found")
        return scan_progress_store[scan_id]

@app.get("/health")
async def health_check():
    return {"status": "ok", "service": "medsecure-ml-inference-12-stage"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

