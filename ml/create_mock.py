from PIL import Image, ImageDraw, ImageFont

img = Image.new('RGB', (800, 600), color = (255, 255, 255))
d = ImageDraw.Draw(img)

try:
    font_large = ImageFont.truetype("arial.ttf", 60)
    font_medium = ImageFont.truetype("arial.ttf", 36)
    font_small = ImageFont.truetype("arial.ttf", 24)
except IOError:
    font_large = ImageFont.load_default()
    font_medium = ImageFont.load_default()
    font_small = ImageFont.load_default()

d.text((50, 50), "Panadol", fill=(0,0,0), font=font_large)
d.text((50, 120), "Paracetamol 500mg", fill=(100,100,100), font=font_medium)
d.text((50, 180), "Manufactured by: GlaxoSmithKline", fill=(0,0,0), font=font_medium)

d.text((50, 260), "Batch No: XYZ9876", fill=(0,0,0), font=font_medium)
d.text((50, 310), "MFD DATE: 12/2025", fill=(0,0,0), font=font_medium)
d.text((50, 360), "EXP DATE: 12/2030", fill=(0,0,0), font=font_medium)
d.text((50, 410), "MRP Rs. 150.50", fill=(0,0,0), font=font_medium)
d.text((50, 460), "Lic No: 12345/ABC", fill=(0,0,0), font=font_medium)
d.text((50, 510), "Barcode: 1234567890123", fill=(0,0,0), font=font_medium)

img.save('perfect_mock_medicine.jpg')
print("Image created.")
