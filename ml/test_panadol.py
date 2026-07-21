import sys
import sqlite3
from main import extract_ocr_fields, build_medicine_candidates, verify_db_medicine, init_db_cache

# Construct ocr_results from debug_ocr2.py
ocr_results = [
    ([[10, 10], [50, 10], [50, 59], [10, 59]], 'Pain', 1.0),
    ([[10, 10], [50, 10], [50, 55], [10, 55]], 'MFD Date:', 0.626),
    ([[10, 10], [50, 10], [50, 46], [10, 46]], 'EXP', 1.0),
    ([[10, 10], [50, 10], [50, 55], [10, 55]], '12/2030', 0.698),
    ([[10, 10], [50, 10], [50, 42], [10, 42]], 'MRP:', 0.999),
    ([[10, 10], [50, 10], [50, 42], [10, 42]], 'Rs.', 0.745),
    ([[10, 10], [50, 10], [50, 42], [10, 42]], 'gsk', 0.558),
    ([[10, 10], [50, 10], [50, 167], [10, 167]], 'Panadol', 0.997),
    ([[10, 10], [50, 10], [50, 97], [10, 97]], 'Paracetamol', 0.851),
    ([[10, 10], [50, 10], [50, 97], [10, 97]], '50Omg', 0.507)
]

fields, full_text, lines = extract_ocr_fields(ocr_results)
print("BEST NAME:", fields.get("name"))
