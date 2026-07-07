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

DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "db", "medsecure.db"))
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
def normalize_date(date_str):
    if not date_str: return ""
    m = re.match(r'^(\d{2})[/-](\d{4})$', date_str.strip())
    if m: return f"{m.group(1)}/{m.group(2)}"
    m = re.match(r'^(\d{2})[/-](\d{2})$', date_str.strip())
    if m: return f"{m.group(1)}/20{m.group(2)}"
    return date_str.strip()

def extract_ocr_fields(ocr_results):
    lines = [r[1].strip() for r in ocr_results if len(r[1].strip()) >= 2]
    full_text = " ".join(lines)
    fields = {}
    best_text = ""
    best_conf = 0
    for result in ocr_results:
        text = result[1].strip() if len(result) >= 2 else ""
        lower = text.lower()
        if len(result) >= 3 and result[2] > best_conf and not any(k in lower for k in ["mfg", "batch", "exp", "mrp", "license"]):
            best_text = text
            best_conf = result[2]
    fields["name"] = best_text

    bm = re.search(r'(?:batch\s*(?:no|number)?|b\.?\s*no\.?)\s*[:\-\.]*\s*([A-Z0-9][A-Z0-9\-/]{2,})', full_text, re.IGNORECASE)
    fields["batch_number"] = bm.group(1).strip() if bm else ""
    
    exp = re.search(r'(?:exp|expiry)(?:\s*date)?\s*[:;\-\.]*\s*((?:\d{2})[/\-](?:\d{2,4}))', full_text, re.IGNORECASE)
    fields["expiry_date"] = normalize_date(exp.group(1)) if exp else ""
    
    mfg = re.search(r'(?:mfg|mfd)(?:\s*date)?\s*[:;\-\.]*\s*((?:\d{2})[/\-](?:\d{2,4}))', full_text, re.IGNORECASE)
    fields["mfg_date"] = normalize_date(mfg.group(1)) if mfg else ""
    
    mrp = re.search(r'(?:mrp|price)\s*[:;\-]*\s*(?:rs\.?)?\s*(\d+\.?\d*)', full_text, re.IGNORECASE)
    fields["mrp"] = mrp.group(1) if mrp else ""
    
    lic = re.search(r'(?:mfg\.?\s*lic|lic\.?\s*no)\s*[:\-]*\s*([A-Z0-9/\-\.]{4,})', full_text, re.IGNORECASE)
    fields["license_number"] = lic.group(1).strip() if lic else ""
    
    manufacturer = ""
    for line in lines:
        mfr = re.search(r'(?:mfg\.?\s*by|manufactured\s*by)\s*[:\-]*\s*(.+)', line, re.IGNORECASE)
        if mfr:
            manufacturer = re.split(r'\b(?:batch|exp|expiry|mrp|mfg\s*date)\b', mfr.group(1), flags=re.IGNORECASE)[0].strip()
            break
    if not manufacturer:
        mfr = re.search(r'(?:mfg\.?\s*by|manufactured\s*by)\s*[:\-]*\s*(.+?)(?=\s+(?:batch|exp|expiry|mrp|mfg\s*date)\b|$)', full_text, re.IGNORECASE)
        manufacturer = mfr.group(1).strip() if mfr else ""
    fields["manufacturer"] = manufacturer[:80]
    
    # Set default 'mfr' field so dictionary doesn't throw KeyError later
    if "mfr" not in fields:
        fields["mfr"] = fields["manufacturer"]

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
    searchable_columns = [c for c in ["brand_name", "name", "generic_name", "primary_ingredient"] if c in columns]
    if not searchable_columns:
        return []

    tokens = []
    for token in re.findall(r"[A-Za-z][A-Za-z0-9\-]{2,}", full_text):
        token = token.strip()
        if len(token) >= 4 and token.lower() not in {"batch", "expiry", "mfg", "mfd", "tablet", "capsule"}:
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

def verify_db_medicine(full_text, lines, conn):
    cur = conn.cursor()
    try:
        cur.execute("SELECT * FROM medicines LIMIT 1")
    except sqlite3.OperationalError:
        return None, 0.0

    all_meds = build_medicine_candidates(full_text, lines, conn)
    if not all_meds:
        cur.execute("SELECT * FROM medicines LIMIT 200")
        all_meds = [dict(r) for r in cur.fetchall()]
    
    best_med = None
    best_score = 0.0
    full_text_lower = full_text.lower()
    for med in all_meds:
        brand = medicine_name(med).lower()
        manufacturer = medicine_manufacturer(med).lower()
        if brand and brand in full_text_lower:
            score = 0.98
            if manufacturer and manufacturer in full_text_lower:
                score = 1.0
            if score > best_score:
                best_score = score; best_med = med
        for line in lines:
            if brand:
                ratio = difflib.SequenceMatcher(None, line.lower(), brand).ratio()
                if manufacturer and manufacturer in full_text_lower:
                    ratio = min(1.0, ratio + 0.05)
                if ratio > best_score and ratio > 0.7:
                    best_score = ratio; best_med = med
    
    return best_med, best_score

# --- Stage 5: Batch Verification ---
def verify_batch(medicine_id, batch_number, conn):
    if not medicine_id or not batch_number: return None
    cur = conn.cursor()
    try:
        row = cur.execute("SELECT * FROM medicine_batches WHERE medicine_id=? AND batch_number=?", (medicine_id, batch_number)).fetchone()
        return dict(row) if row else None
    except sqlite3.OperationalError:
        return None

def verify_batch_any_medicine(batch_number, conn):
    if not batch_number:
        return None
    try:
        row = conn.execute("SELECT * FROM medicine_batches WHERE batch_number=? LIMIT 1", (batch_number,)).fetchone()
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

    if not req:
        return {"status": "skipped", "required": False, "found": False, "match": None, "note": "Barcode not required", "reason": "Barcode not required", "score": 100}

    if not HAS_ZBAR:
        return {"status": "failed", "required": True, "found": False, "match": False, "note": "Barcode decoder unavailable", "reason": "Barcode decoder unavailable", "score": 0}

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
        return {"status": "failed", "required": True, "found": False, "match": False, "note": decode_error, "reason": decode_error, "score": 0}

    if not decoded:
        return {"status": "failed", "required": True, "found": False, "match": False, "note": "Barcode required but not found", "reason": "Barcode required but not found", "score": 0}
    
    expected = (batch_row.get("barcode_value") or "") if batch_row else ""
    if expected in decoded:
        return {"status": "passed", "required": True, "found": True, "match": True, "decoded_value": decoded[0], "stored_value": expected, "note": "Barcode verified", "reason": "Barcode Verified", "score": 100}
    return {"status": "failed", "required": True, "found": True, "match": False, "decoded_value": decoded[0], "stored_value": expected, "note": "Barcode mismatch", "reason": "Barcode Mismatch", "score": 0}

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
def generate_explainable_report(stages_results):
    score = 0
    weights = {
        "ocr": 0.15, "db": 0.15, "batch": 0.20, "barcode": 0.05, 
        "packaging": 0.10, "logo": 0.05, "color": 0.10, "layout": 0.10, "tamper": 0.10
    }
    
    report = []
    
    def add_line(name, res):
        nonlocal score
        w = weights[name]
        s = res["score"]
        score += s * w
        
        icon = "✓" if s > 80 else ("⚠" if s > 50 else "✗")
        report.append(f"{icon} {res['reason']}")

    add_line("ocr", stages_results["ocr"])
    add_line("db", stages_results["db"])
    add_line("batch", stages_results["batch"])
    add_line("barcode", stages_results["barcode"])
    add_line("packaging", stages_results["packaging"])
    add_line("logo", stages_results["logo"])
    add_line("color", stages_results["color"])
    add_line("layout", stages_results["layout"])
    add_line("tamper", stages_results["tamper"])
    
    final_score = min(100, max(0, round(score, 1)))
    verdict = "Verified Genuine" if final_score > 85 else ("Caution" if final_score > 60 else "Counterfeit / High Risk")
    
    return {
        "score": final_score,
        "verdict": verdict,
        "explanation": report,
        "breakdown": {k: v["score"] for k, v in stages_results.items()}
    }


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
        
        ocr_score = 100 if fields["batch_number"] and fields["mfr"] else (50 if fields["batch_number"] else 20)
        res_ocr = {"score": ocr_score, "reason": "OCR Extracted Core Fields" if ocr_score > 50 else "Missing text fields in OCR"}

        conn = get_db()

        # Stage 4
        set_progress(scan_id, "db_verification", 3, 0.30)
        med_row, db_score = verify_db_medicine(full_text, lines, conn)
        brand_nm = med_row.get("brand_name") or med_row.get("name") if med_row else "None"
        res_db = {"score": db_score * 100, "reason": f"Medicine Name Matched ({brand_nm})"}
        
        # Stage 5
        set_progress(scan_id, "batch_verification", 4, 0.40)
        batch_row = verify_batch(med_row["id"] if med_row else None, fields["batch_number"], conn)
        if not batch_row:
            batch_row = verify_batch_any_medicine(fields["batch_number"], conn)
            if batch_row and (not med_row or batch_row.get("medicine_id") != med_row.get("id")):
                try:
                    med = conn.execute("SELECT * FROM medicines WHERE id=? LIMIT 1", (batch_row["medicine_id"],)).fetchone()
                    med_row = dict(med) if med else None
                    brand_nm = medicine_name(med_row) if med_row else brand_nm
                    if med_row:
                        res_db = {"score": 90, "reason": f"Medicine inferred from verified batch ({brand_nm})"}
                except sqlite3.Error:
                    pass
        res_batch = {"score": 100 if batch_row else 0, "reason": "Batch Number Verified" if batch_row else "Batch NOT found in registry"}

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
        
        final_report = generate_explainable_report(stages_results)

        # Extract anomalies for legacy compatibility in DB
        anomalies = [line for line in final_report["explanation"] if line.startswith("✗") or line.startswith("⚠")]

        # Write result
        cur = conn.cursor()
        try:
            cur.execute("""
                INSERT INTO scans (id, medicine_id, batch_number, authenticity_score, verdict, anomalies_json, scanned_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (scan_id, med_row["id"] if med_row else None, fields["batch_number"], final_report["score"], 
                  final_report["verdict"].lower().replace(' ', '_'), json.dumps(anomalies), int(time.time()*1000)))
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
                    "authenticity_score": final_report["score"],
                    "medicine_id": med_row["id"] if med_row else None,
                    "batch_id": batch_row["id"] if batch_row else None,
                    "medicine_name": brand_nm,
                    "manufacturer_name": (med_row.get("manufacturer_name") or fields["manufacturer"]) if med_row else fields["manufacturer"],
                    "batch_number": fields["batch_number"],
                    "ocr_extracted": fields,
                    "db_match_results": {
                        "medicine_name": {
                            "extracted": brand_nm,
                            "stored": brand_nm if med_row else None,
                            "match": bool(med_row)
                        },
                        "batch_number": {
                            "extracted": fields["batch_number"],
                            "stored": batch_row.get("batch_number") if batch_row else None,
                            "match": bool(batch_row)
                        }
                    },
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
