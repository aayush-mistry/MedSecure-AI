import sys
import sqlite3
from main import extract_ocr_fields, build_medicine_candidates, verify_db_medicine, init_db_cache

# Construct ocr_results from the db row
ocr_results = [
    ([[5.25, 12.5], [84.13, 12.5], [84.13, 29.67], [5.25, 29.67]], "Calpol 50Omg Tablet", 0.563),
    ([[6.12, 26.5], [65, 26.5], [65, 36.83], [6.12, 36.83]], "Paracetamol Tablets IP 500mg", 0.976),
    ([[6.88, 40.17], [18.38, 40.17], [18.38, 46.5], [6.88, 46.5]], "Mifg By", 0.63),
    ([[6.88, 45.67], [62.13, 45.67], [62.13, 52], [6.88, 52]], "Glaxo SmithKline Pharmaceuticals Ltd", 0.769),
    ([[7.25, 55.33], [22.25, 55.33], [22.25, 60.66], [7.25, 60.66]], "Batch No", 0.992),
    ([[25.87, 54.67], [41.12, 54.67], [41.12, 60.67], [25.87, 60.67]], "GP43210", 0.999),
    ([[7.38, 61.83], [39.88, 61.83], [39.88, 68.16], [7.38, 68.16]], "MFG Date: 08/2024", 0.763),
    ([[65, 66], [71.25, 66], [71.25, 71], [65, 71]], "Ssk", 0.524),
    ([[7.5, 69.33], [22.75, 69.33], [22.75, 74.66], [7.5, 74.66]], "EXP Date", 0.954),
    ([[25.87, 68.83], [39.87, 68.83], [39.87, 74.83], [25.87, 74.83]], "07/2027", 1),
    ([[7.5, 76], [20.5, 76], [20.5, 81.33], [7.5, 81.33]], "MRP Rs.", 0.794),
    ([[26.5, 76], [35.75, 76], [35.75, 81.33], [26.5, 81.33]], "16.65", 0.998),
    ([[75, 29.38], [87.87, 29.38], [87.87, 42.78], [75, 42.78]], "gsk", 1)
]

fields, full_text, lines = extract_ocr_fields(ocr_results)

conn = sqlite3.connect('../db/indian_pharmaceutical_products.db')
conn.row_factory = sqlite3.Row
init_db_cache()

med, score, reason = verify_db_medicine(fields, full_text, lines, conn)
print("MFR:", fields["manufacturer"]); print("FINAL MED:", med["name"] if med else None, "SCORE:", score, "REASON:", reason)
