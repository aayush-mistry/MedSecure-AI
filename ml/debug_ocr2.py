import sys
import cv2
import easyocr

print("Loading EasyOCR...")
reader = easyocr.Reader(['en'], gpu=False)
img = cv2.imread(r'C:\Users\aayus\.gemini\antigravity\brain\f59557ee-8440-4efc-a77d-83046fa174ae\mock_medicine_box_1784607741814.png')

print("Running OCR...")
ocr_res = reader.readtext(img, detail=1, paragraph=False, decoder="greedy")

for r in ocr_res:
    box = r[0]
    height = box[2][1] - box[0][1]
    print(f"'{r[1]}' conf {r[2]:.3f} height {height}")
