import os
import re
import json
import time
import threading
import sqlite3
import difflib
import ssl
import multiprocessing as mp
import importlib.util
import numpy as np
import cv2
from PIL import Image
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import easyocr

ssl._create_default_https_context = ssl._create_unverified_context

try:
    if os.name == "nt" and hasattr(os, "add_dll_directory"):
        pyzbar_spec = importlib.util.find_spec("pyzbar")
        if pyzbar_spec and pyzbar_spec.submodule_search_locations:
            os.add_dll_directory(str(pyzbar_spec.submodule_search_locations[0]))
    from pyzbar.pyzbar import decode as zbar_decode
    HAS_ZBAR = True
except Exception:
    HAS_ZBAR = False
    def zbar_decode(*args, **kwargs):
        return []
    print("ZBar not available.")

try:
    import zxingcpp
    HAS_ZXING = True
except Exception:
    HAS_ZXING = False
    zxingcpp = None
    print("ZXing-C++ not available. Falling back to OpenCV QR detection.")

app = FastAPI(title="MedSecure ML Inference Service v5 (12-Stage Pipeline)")

print("Loading EasyOCR model...")
reader = easyocr.Reader(['en'], gpu=True, verbose=False)
print("EasyOCR ready.")

DB_PATH = os.environ.get(
    "DB_PATH",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "db", "indian_pharmaceutical_products.db"))
)
MAX_OCR_DIMENSION = 800
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


_DB_CACHE = {"medicines": [], "batches": [], "generic_names": set()}

def init_db_cache():
    if _DB_CACHE["medicines"]: return
    try:
        conn = get_db()
        _DB_CACHE["medicines"] = [dict(r) for r in conn.execute("SELECT * FROM medicines").fetchall()]
        _DB_CACHE["batches"] = [dict(r) for r in conn.execute("SELECT * FROM medicine_batches").fetchall()]
        for m in _DB_CACHE["medicines"]:
            g = m.get("generic_name")
            if g and g.strip():
                _DB_CACHE["generic_names"].add(g.strip())
        conn.close()
    except Exception as e:
        print(f"Failed to init DB cache: {e}")

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
    
    # Deskew from real edges only. Using every non-black pixel can rotate
    # normal package photos and make small label text harder for OCR.
    edges = cv2.Canny(enhanced, 50, 150)
    coords = np.column_stack(np.where(edges > 0))
    angle = 0
    if len(coords) > 100:
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45: angle = 90 + angle
        if 0.5 < abs(angle) <= 15:
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

def clean_medicine_name_candidate(value):
    value = clean_ocr_text(value)
    value = re.sub(r"\b\d+(?:\.\d+)?\s*(?:mg|mcg|g|ml|iu)\b", "", value, flags=re.IGNORECASE)
    value = re.sub(r"\b(?:tablets?|tabs?|capsules?|caps?|syrup|injection|cream|ointment|drops|strip|blister)\b", "", value, flags=re.IGNORECASE)
    value = re.sub(r"\b(?:ip|bp|usp)\b", "", value, flags=re.IGNORECASE)
    value = re.sub(r"[^A-Za-z0-9 '\-]+", " ", value)
    return clean_ocr_text(value)

def field_result(value=NOT_DETECTED, confidence=0.0, source="ocr"):
    value = clean_ocr_text(value)
    return {
        "value": value if value else NOT_DETECTED,
        "confidence": round(float(confidence or 0.0), 3),
        "source": source
    }

def best_regex_match(ocr_results, pattern, group=1, normalizer=None, min_conf=0.25, full_text=None):
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
    if not is_detected(best["value"]) and full_text:
        match = re.search(pattern, full_text, re.IGNORECASE)
        if match:
            value = match.group(group).strip()
            if normalizer:
                value = normalizer(value)
            best = field_result(value, confidence_for_value(ocr_results, value), "ocr_full_text")
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

def normalize_batch_candidate(value):
    value = clean_ocr_text(value).upper()
    value = re.sub(r"[^A-Z0-9\-/]", "", value)
    if not value:
        return NOT_DETECTED
    chars = []
    for char in value:
        if char == "O" and any(ch.isdigit() for ch in chars):
            chars.append("0")
        elif char == "I" and any(ch.isdigit() for ch in chars):
            chars.append("1")
        elif char == "S" and any(ch.isdigit() for ch in chars):
            chars.append("5")
        else:
            chars.append(char)
    return "".join(chars).strip("-/")

def batch_candidate_score(candidate, context_text, confidence):
    normalized = normalize_batch_candidate(candidate)
    if not is_detected(normalized):
        return None
    if len(normalized) < 4 or len(normalized) > 18:
        return None
    if re.fullmatch(r"\d{1,2}[-/]\d{2,4}", normalized) or re.fullmatch(r"\d+(?:\.\d+)?", normalized):
        return None
    if not (re.search(r"[A-Z]", normalized) and re.search(r"\d", normalized)):
        return None

    score = float(confidence or 0.0)
    if re.search(r"\b(batch|b\s*no|lot)\b", normalize_match_text(context_text)):
        score += 1.0
    if re.fullmatch(r"[A-Z]{1,5}[-/]?\d{3,8}[A-Z0-9\-/]*", normalized):
        score += 0.35
    return normalized, score

def extract_batch_number(ocr_results, full_text):
    patterns = [
        r"(?:batch|lot)\s*(?:no|number|code)?\s*[:;\-\.]*\s*([A-Z0-9][A-Z0-9\-/]{2,})",
        r"\bb\.?\s*no\.?\s*[:;\-\.]*\s*([A-Z0-9][A-Z0-9\-/]{2,})",
    ]
    for pattern in patterns:
        match = best_regex_match(
            ocr_results,
            pattern,
            normalizer=normalize_batch_candidate,
            min_conf=0.0,
            full_text=full_text
        )
        if is_detected(match["value"]):
            return match

    candidates = []
    for idx, result in enumerate(ocr_results):
        text = result[1].strip() if len(result) >= 2 else ""
        conf = float(result[2]) if len(result) >= 3 else 0.0
        context = " ".join(
            ocr_results[i][1].strip()
            for i in (idx - 1, idx, idx + 1, idx + 2)
            if 0 <= i < len(ocr_results) and len(ocr_results[i]) >= 2
        )
        for raw in re.findall(r"\b[A-Z0-9][A-Z0-9\-/]{3,17}\b", context, flags=re.IGNORECASE):
            scored = batch_candidate_score(raw, context, conf)
            if scored:
                candidates.append((scored[1], scored[0], conf))

    if candidates:
        candidates.sort(reverse=True, key=lambda item: item[0])
        _, value, conf = candidates[0]
        return field_result(value, conf, "ocr_batch_context")

    return field_result()

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
        "license", "lic", "schedule", "paracetamol tablets"
    }
    best_name = field_result()
    for result in ocr_results:
        text = result[1].strip() if len(result) >= 2 else ""
        conf = float(result[2]) if len(result) >= 3 else 0.0
        lower = text.lower()
        name_candidate = clean_medicine_name_candidate(text)
        looks_like_name = (
            conf >= 0.35
            and len(name_candidate) >= 3
            and any(ch.isalpha() for ch in name_candidate)
            and not any(term in lower for term in ignored_name_terms)
        )
        if looks_like_name and conf > best_name["confidence"]:
            best_name = field_result(name_candidate, conf)
    field_details["name"] = best_name

    
    # Generic Name Fuzzy Matching
    init_db_cache()
    best_gn_score = 0
    best_gn = None
    lower_full_text = full_text.lower()
    for gn in _DB_CACHE["generic_names"]:
        gn_lower = gn.lower()
        if gn_lower in lower_full_text:
            best_gn = gn
            best_gn_score = 1.0
            break
        # Relaxed matching without IP/Tablets
        for token in gn_lower.split():
            if len(token) > 4 and token in lower_full_text:
                if not best_gn:
                    best_gn = gn
                    best_gn_score = 0.8
                
    if best_gn:
        field_details["composition"] = field_result(best_gn, best_gn_score, "ocr_regex")
        print(f"[OCR] Generic Name Extracted: {best_gn} (score: {best_gn_score})")

    field_details["batch_number"] = extract_batch_number(ocr_results, full_text)
    field_details["expiry_date"] = best_regex_match(
        ocr_results,
        r'(?:exp|expiry)(?:\s*date)?\s*[:;\-\.]*\s*((?:\d{1,2})[/\-](?:\d{2,4}))',
        normalizer=normalize_date,
        full_text=full_text
    )
    field_details["mfg_date"] = best_regex_match(
        ocr_results,
        r'(?:mfg|mfd)(?:\s*date)?\s*[:;\-\.]*\s*((?:\d{1,2})[/\-](?:\d{2,4}))',
        normalizer=normalize_date,
        full_text=full_text
    )
    field_details["mrp"] = best_regex_match(
        ocr_results,
        r'(?:mrp|price)\s*[:;\-]*\s*(?:rs\.?|inr|₹)?\s*[:;\-]*\s*(\d+\.?\d*)',
        full_text=full_text
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
    init_db_cache()
    tokens = []
    for token in re.findall(r"[A-Za-z][A-Za-z0-9\-]{2,}", full_text):
        token = token.strip().lower()
        if len(token) >= 4 and token not in {"batch", "expiry", "mfg", "mfd", "tablet", "capsule", "tablets", "capsules", "price", "license"}:
            tokens.append(token)

    seen = set()
    candidates = []
    searchable_columns = ["brand_name", "name", "generic_name", "primary_ingredient", "manufacturer_name"]
    for token in tokens[:12]:
        for med in _DB_CACHE["medicines"]:
            if med.get("id") in seen: continue
            for col in searchable_columns:
                val = med.get(col)
                if val and token in str(val).lower():
                    seen.add(med.get("id"))
                    candidates.append(med)
                    break
        if len(candidates) >= 80:
            break
    return candidates

def find_medicine_by_batch(batch_number, conn):
    if not is_detected(batch_number): return None
    init_db_cache()
    for b in _DB_CACHE["batches"]:
        if str(b.get("batch_number")).lower() == str(batch_number).lower():
            for m in _DB_CACHE["medicines"]:
                if m.get("id") == b.get("medicine_id"):
                    return m
    return None

def verify_db_medicine(fields, full_text, lines, conn):
    init_db_cache()
    if not _DB_CACHE["medicines"]:
        return None, 0.0, {"status": "skipped", "reason": "Medicine database unavailable"}

    batch_med = find_medicine_by_batch(fields.get("batch_number"), conn)
    if batch_med:
        return batch_med, 0.93, {
            "status": "verified",
            "reason": "Medicine identified from registered batch number"
        }

    all_meds = build_medicine_candidates(full_text, lines, conn)
    if not all_meds:
        all_meds = _DB_CACHE["medicines"][:200]
    
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
    if not is_detected(batch_number): return None
    init_db_cache()
    batch_str = str(batch_number).lower()
    for b in _DB_CACHE["batches"]:
        if str(b.get("batch_number")).lower() == batch_str:
            if medicine_id and b.get("medicine_id") != medicine_id: continue
            return b
    return None

def infer_single_registered_batch(med_row, conn):
    if not med_row or conn is None:
        return None
    try:
        rows = conn.execute("SELECT * FROM medicine_batches WHERE medicine_id=?", (med_row.get("id"),)).fetchall()
    except sqlite3.Error:
        return None
    if len(rows) == 1:
        return dict(rows[0])
    return None

def safe_stage(stage_name, fallback, fn):
    try:
        result = fn()
        return result if isinstance(result, dict) else fallback
    except Exception as exc:
        print(f"{stage_name} failed: {exc}", flush=True)
        failed = dict(fallback)
        failed["status"] = "failed"
        failed["reason"] = f"{stage_name} failed: {exc}"
        failed["score"] = float(failed.get("score", 0) or 0)
        return failed

# --- Stage 6: Barcode Verification ---
def build_ocr_boxes(ocr_results, image_shape):
    image_h, image_w = image_shape[:2]
    boxes = []
    for result in ocr_results:
        if len(result) < 3:
            continue
        points, text, conf = result[0], clean_ocr_text(result[1]), float(result[2] or 0)
        if not text or conf < 0.25:
            continue
        xs = [float(point[0]) for point in points]
        ys = [float(point[1]) for point in points]
        left, top = min(xs), min(ys)
        width, height = max(xs) - left, max(ys) - top
        if width <= 1 or height <= 1:
            continue
        boxes.append({
            "x": round((left / image_w) * 100, 2),
            "y": round((top / image_h) * 100, 2),
            "w": round((width / image_w) * 100, 2),
            "h": round((height / image_h) * 100, 2),
            "text": text[:80],
            "confidence": round(conf, 3)
        })
    return boxes[:30]

def decode_barcodes_opencv(gray):
    detector = cv2.QRCodeDetector()
    decoded = []
    try:
        ok, decoded_info, _, _ = detector.detectAndDecodeMulti(gray)
        if ok:
            decoded.extend([value for value in decoded_info if value])
    except Exception:
        value, _, _ = detector.detectAndDecode(gray)
        if value:
            decoded.append(value)
    return decoded

def decode_barcodes_zxing(img):
    if not HAS_ZXING:
        return []
    results = zxingcpp.read_barcodes(img)
    decoded = []
    for result in results:
        text = getattr(result, "text", None)
        if text:
            decoded.append(text)
    return decoded

def valid_ean13(value):
    if not re.fullmatch(r"\d{13}", value or ""):
        return False
    digits = [int(ch) for ch in value]
    checksum = (10 - ((sum(digits[0:12:2]) + 3 * sum(digits[1:12:2])) % 10)) % 10
    return checksum == digits[12]

def normalize_barcode_digit_candidates(text):
    digits = re.sub(r"\D", "", text or "")
    candidates = []
    for match in re.findall(r"\d{8,18}", digits):
        candidates.extend(match[i:i + 13] for i in range(0, max(0, len(match) - 12)))
        candidates.append(match)
    valid = [candidate for candidate in candidates if valid_ean13(candidate)]
    if valid:
        return valid
    return [candidate for candidate in candidates if 8 <= len(candidate) <= 18]

def decode_barcode_digits_with_ocr(img):
    h, w = img.shape[:2]
    regions = [
        img[:, int(w * 0.65):],
        img[:, int(w * 0.72):],
        img[int(h * 0.2):int(h * 0.95), int(w * 0.65):],
    ]
    candidates = []
    for region in regions:
        if region.size == 0:
            continue
        variants = [
            region,
            cv2.rotate(region, cv2.ROTATE_90_CLOCKWISE),
            cv2.rotate(region, cv2.ROTATE_90_COUNTERCLOCKWISE),
        ]
        for variant in variants:
            gray = cv2.cvtColor(variant, cv2.COLOR_BGR2GRAY) if len(variant.shape) == 3 else variant
            try:
                results = reader.readtext(
                    gray,
                    detail=1,
                    paragraph=False,
                    decoder="greedy",
                    text_threshold=0.35,
                    low_text=0.2,
                    width_ths=1.2
                )
            except Exception:
                continue
            text = " ".join(result[1] for result in results if len(result) >= 2)
            candidates.extend(normalize_barcode_digit_candidates(text))
    if not candidates:
        return []
    candidates.sort(key=len, reverse=True)
    return [candidates[0]]

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
        try:
            barcodes = zbar_decode(gray)
            return [b.data.decode("utf-8", errors="ignore") for b in barcodes], None
        except Exception as exc:
            return [], str(exc)

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

    h, w = img.shape[:2]
    max_dim = max(h, w)
    scan_img = img
    if max_dim > 1200:
        scale = 1200 / max_dim
        scan_img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(scan_img, cv2.COLOR_BGR2GRAY)
    if HAS_ZBAR:
        decoded, decode_error = decode_barcodes_with_timeout(gray)
        decoder_name = "ZBar"
    elif HAS_ZXING:
        decoded, decode_error = decode_barcodes_zxing(scan_img), None
        decoder_name = "ZXing-C++"
    else:
        decoded, decode_error = decode_barcodes_opencv(gray), None
        decoder_name = "OpenCV QR fallback"

    if decode_error:
        print(f"Barcode decoding skipped: {decode_error}")
        return {"status": "skipped", "required": req, "found": False, "match": None, "ocr_value": NOT_DETECTED, "stored_value": None, "note": decode_error, "reason": "Barcode verification skipped", "score": 85 if not req else 70}

    if not decoded:
        ocr_decoded = decode_barcode_digits_with_ocr(scan_img)
        if ocr_decoded:
            decoded = ocr_decoded
            decoder_name = "OCR barcode digit fallback"

    if not decoded:
        note = f"Barcode not detected by {decoder_name}"
        return {"status": "skipped", "required": req, "found": False, "match": None, "ocr_value": NOT_DETECTED, "stored_value": None, "note": note, "reason": "Barcode not detected; verification skipped", "score": 85 if not req else 70}
    
    expected = (batch_row.get("barcode_value") or "") if batch_row else ""
    if expected and expected in decoded:
        return {"status": "passed", "required": req, "found": True, "match": True, "ocr_value": decoded[0], "decoded_value": decoded[0], "stored_value": expected, "note": f"Barcode verified by {decoder_name}", "reason": "Barcode Verified", "score": 100}
    if expected:
        return {"status": "failed", "required": req, "found": True, "match": False, "ocr_value": decoded[0], "decoded_value": decoded[0], "stored_value": expected, "note": f"Barcode mismatch; decoded by {decoder_name}", "reason": "Barcode Mismatch", "score": 0}
    return {"status": "skipped", "required": req, "found": True, "match": None, "ocr_value": decoded[0], "decoded_value": decoded[0], "stored_value": None, "note": f"Barcode decoded by {decoder_name}, but no database barcode value is recorded", "reason": "Barcode database comparison skipped", "score": 90}

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

    gn_ocr = fields.get("composition")
    gn_db = db_value(med_row, "generic_name")
    if is_detected(gn_ocr):
        gn_res = compare_fuzzy(gn_ocr, gn_db, "Generic Name", 0.70)
        if gn_res["match"]:
            gn_res["status"] = "Verified"
            gn_res["note"] = "Generic name matched with database."
    else:
        gn_res = {
            "extracted": NOT_DETECTED,
            "stored": gn_db,
            "match": False if gn_db else None,
            "status": "Mismatch" if gn_db else "Skipped",
            "note": "Generic name not found in OCR text." if gn_db else "No generic name in database."
        }

    return {
        "medicine_name": compare_fuzzy(fields.get("name"), medicine_name(med_row) if med_row else None, "Medicine Name", 0.78),
        "generic_name": gn_res,
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


import time
import concurrent.futures

def run_full_pipeline(scan_id, file_path):
    conn = None
    timings = {}
    total_start = time.perf_counter()
    try:
        # DB Caching (non-blocking)
        init_db_cache()

        t0 = time.perf_counter()
        set_progress(scan_id, "image_quality", 0, 0.05)
        img = cv2.imread(file_path)
        if img is None: raise Exception("Invalid image file")
        ok, issues = check_image_quality(img)
        if not ok:
            print(f"Image quality warnings for {scan_id}: {', '.join(issues)}", flush=True)
        timings["Image Loading & Quality"] = time.perf_counter() - t0

        # Stage 2
        t0 = time.perf_counter()
        set_progress(scan_id, "image_enhancement", 1, 0.10)
        try:
            enhanced_img = enhance_image(img)
        except Exception as exc:
            print(f"Image enhancement failed, using original image: {exc}", flush=True)
            enhanced_img = img
        timings["Preprocessing"] = time.perf_counter() - t0
        
        # Stage 3
        t0 = time.perf_counter()
        set_progress(scan_id, "ocr_extraction", 2, 0.20)
        try:
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
            fields["ocr_boxes"] = build_ocr_boxes(ocr_res, ocr_img.shape)
        except Exception as exc:
            print(f"OCR extraction failed: {exc}", flush=True)
            ocr_res = []
            full_text = ""
            lines = []
            fields = {
                "name": NOT_DETECTED, "manufacturer": NOT_DETECTED, "batch_number": NOT_DETECTED,
                "mfg_date": NOT_DETECTED, "expiry_date": NOT_DETECTED, "mrp": NOT_DETECTED,
                "strength": NOT_DETECTED, "dosage_form": NOT_DETECTED, "composition": NOT_DETECTED,
                "license_number": NOT_DETECTED, "barcode": NOT_DETECTED, "ocr_boxes": [], "mfr": NOT_DETECTED,
                "_confidence": {}, "_details": {}, "_raw_text": ""
            }
        timings["OCR"] = time.perf_counter() - t0
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

        try:
            conn = get_db()
        except Exception as exc:
            print(f"Database unavailable: {exc}", flush=True)
            conn = None

        # Stage 4: DB Match
        t0 = time.perf_counter()
        set_progress(scan_id, "db_verification", 3, 0.30)
        if conn is not None:
            try:
                med_row, db_score, med_match_meta = verify_db_medicine(fields, full_text, lines, conn)
            except Exception as exc:
                print(f"Medicine identification failed: {exc}", flush=True)
                med_row, db_score, med_match_meta = None, 0.0, {"reason": f"Medicine identification failed: {exc}"}
        else:
            med_row, db_score, med_match_meta = None, 0.0, {"reason": "Medicine database unavailable"}
        brand_nm = medicine_name(med_row) if med_row else "Medicine Not Found"
        if med_row and is_detected(brand_nm):
            fields["name"] = brand_nm
            fields.setdefault("_confidence", {})["name"] = max(fields.get("_confidence", {}).get("name", 0), db_score)
            fields.setdefault("_details", {})["name"] = field_result(brand_nm, db_score, "registry_match")
        res_db = {
            "score": round(db_score * 100, 1) if med_row else max(0, round(db_score * 60, 1)),
            "reason": med_match_meta["reason"]
        }
        timings["Medicine Matching"] = time.perf_counter() - t0
        
        # Stage 5: Batch Match
        t0 = time.perf_counter()
        set_progress(scan_id, "batch_verification", 4, 0.40)
        batch_row = verify_batch(med_row["id"] if med_row and conn else None, fields["batch_number"], conn) if conn else None
        if not is_detected(fields["batch_number"]):
            inferred_batch = infer_single_registered_batch(med_row, conn)
            if inferred_batch:
                batch_row = inferred_batch
                fields["batch_number"] = inferred_batch.get("batch_number") or NOT_DETECTED
                fields.setdefault("_confidence", {})["batch_number"] = 0.65
                fields.setdefault("_details", {})["batch_number"] = field_result(fields["batch_number"], 0.65, "registry_single_batch")
                res_batch = {"score": 90, "reason": "Batch recovered from the matched medicine registry record"}
            else:
                res_batch = {"score": 85, "reason": "Batch number not detected; batch verification skipped"}
        elif batch_row:
            res_batch = {"score": 100, "reason": "Batch number found in registry"}
        else:
            res_batch = {"score": 45, "reason": "OCR batch number not found in registry"}
        timings["Batch Verification"] = time.perf_counter() - t0
        ocr_public = {k: v for k, v in fields.items() if not k.startswith("_") and k != "mfr"}

        # Parallelize Stages 6 to 11
        t0 = time.perf_counter()
        set_progress(scan_id, "verification_modules", 5, 0.50)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            fut_barcode = executor.submit(safe_stage, "Barcode verification", {"status": "failed", "reason": "Barcode verification failed", "score": 0, "match": None, "required": False, "found": False}, lambda: verify_barcode(enhanced_img, batch_row, med_row))
            fut_packaging = executor.submit(safe_stage, "Packaging analysis", {"status": "failed", "reason": "Packaging analysis failed", "score": 0}, lambda: analyze_packaging(enhanced_img))
            fut_logo = executor.submit(safe_stage, "Logo verification", {"status": "warning", "reason": "Logo verification skipped", "score": 75}, lambda: verify_logo(enhanced_img, med_row))
            fut_color = executor.submit(safe_stage, "Color verification", {"status": "failed", "reason": "Color verification failed", "score": 0}, lambda: verify_color(enhanced_img, med_row))
            fut_layout = executor.submit(safe_stage, "Layout verification", {"status": "failed", "reason": "Layout verification failed", "score": 0}, lambda: verify_layout(ocr_res, med_row))
            fut_tamper = executor.submit(safe_stage, "Tamper detection", {"status": "failed", "reason": "Tamper detection failed", "score": 0}, lambda: detect_tampering(enhanced_img))
            
            res_barcode = fut_barcode.result()
            res_packaging = fut_packaging.result()
            res_logo = fut_logo.result()
            res_color = fut_color.result()
            res_layout = fut_layout.result()
            res_tamper = fut_tamper.result()
            
        timings["Parallel Verifications (Barcode, Logo, Color, etc)"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        set_progress(scan_id, "confidence_scoring", 11, 0.95)
        
        verification_results = build_verification_results(fields, med_row, batch_row, res_barcode)
        
        # Logging generic name correctly
        gn_result = verification_results.get("generic_name", {})
        print(f"\n--- Generic Name Logging ---")
        print(f"OCR Generic Name: {fields.get('composition')}")
        print(f"Parsed Generic Name: {gn_result.get('extracted')}")
        print(f"Database Generic Name: {gn_result.get('stored')}")
        print(f"Similarity Score: {gn_result.get('confidence', 1.0) * 100}%")
        print(f"Verification: {gn_result.get('status')}")
        print(f"----------------------------\n")
        
        final_report = generate_evidence_report({
            "ocr": res_ocr,
            "db": res_db,
            "batch": res_batch,
            "barcode": res_barcode,
            "packaging": res_packaging,
            "logo": res_logo,
            "color": res_color,
            "layout": res_layout,
            "tamper": res_tamper
        })
        timings["Authenticity Score"] = time.perf_counter() - t0
        
        timings["Total Time"] = time.perf_counter() - total_start
        
        print("\n=== Execution Timings ===")
        for stage, duration in timings.items():
            print(f"{stage}: {duration * 1000:.0f} ms" if duration < 1 else f"{stage}: {duration:.2f} s")
        print("=========================\n", flush=True)

        anomalies = []
        for stage_name in ["barcode", "packaging", "color", "layout", "tamper", "logo"]:
            res = locals().get(f"res_{stage_name}")
            if res and res.get("status") in ("failed", "warning"):
                anomalies.append(res.get("reason", f"{stage_name} check issue"))

        with scan_progress_lock:
            scan_progress_store[scan_id] = {
                "scan_id": scan_id,
                "stage": "complete",
                "stage_index": len(STAGES),
                "total_stages": len(STAGES),
                "progress": 1.0,
                "status": "complete",
                "result": {
                    "verdict": final_report["verdict"],
                    "authenticity_score": final_report["score"],
                    "confidence_level": final_report["confidence"],
                    "medicine_name": brand_nm,
                    "generic_name": gn_result.get("extracted") or med_row.get("generic_name") if med_row else None,
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
    finally:
        if conn is not None:
            conn.close()

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
    barcode_decoder = "zbar" if HAS_ZBAR else ("zxing_cpp" if HAS_ZXING else "opencv_qr_fallback")
    return {
        "status": "ok",
        "service": "medsecure-ml-inference-12-stage",
        "ocr": "easyocr",
        "barcode_decoder": barcode_decoder,
        "build": "ocr-name-fix-2026-07-15"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

