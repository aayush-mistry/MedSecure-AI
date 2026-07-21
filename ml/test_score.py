import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__)))

from main import generate_evidence_report

stages_results = {
    "ocr": {"score": 95},
    "db": {"score": 100},
    "batch": {"score": 85},
    "barcode": {"score": 70},
    "packaging": {"score": 100},
    "logo": {"score": 100},
    "color": {"score": 100},
    "layout": {"score": 100},
    "tamper": {"score": 100}
}

verification_results = {
    "medicine_name": {"status": "Verified"},
    "batch_number": {"status": "Skipped"},
    "manufacturing_date": {"status": "Skipped"},
    "expiry_date": {"status": "Skipped"},
    "mrp": {"status": "Skipped"},
    "barcode": {"status": "Skipped"}
}

report = generate_evidence_report(stages_results, verification_results)
print("FINAL SCORE:", report["score"])
print("EXPLANATION:", report["explanation"])
