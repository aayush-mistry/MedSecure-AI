import sys
import cv2
import easyocr
import asyncio
from main import extract_ocr_fields, build_medicine_candidates, verify_db_medicine, init_db_cache

print("Loading EasyOCR...")
reader = easyocr.Reader(['en'], gpu=False)
img = cv2.imread('perfect_mock_medicine.jpg')
if img is None:
    print("Could not load image")
    sys.exit(1)

print("Running OCR...")
ocr_res = reader.readtext(img, detail=1, paragraph=False, decoder="greedy")

print("Extracting fields...")
fields, full_text, lines = extract_ocr_fields(ocr_res)

print("DB Verification...")
import sqlite3
conn = sqlite3.connect('../db/indian_pharmaceutical_products.db')
conn.row_factory = sqlite3.Row
init_db_cache(conn)

med, score, reason = verify_db_medicine(fields, full_text, lines, conn)
print("FINAL MED:", med["name"] if med else None, "SCORE:", score)
