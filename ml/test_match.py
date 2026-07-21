import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__)))

from main import verify_db_medicine, normalize_match_text, get_db

full_text = "panadol paracetamol 500mg for effective pain relief film-coated tablets gsk glaxosmithkline each tablet contains paracetamol bp 500mg for headache pain fever 810001228854385 mfd date 12/2025 exp date 12/2030 mrp rs 150.50"
lines = full_text.split()
fields = {
    "name": "panadol",
    "manufacturer": "gsk",
    "batch_number": None
}

conn = get_db()
best_med, score, meta = verify_db_medicine(fields, full_text, lines, conn)
print("BEST MED:", best_med.get("name") if best_med else None)
print("SCORE:", score)
print("META:", meta)
