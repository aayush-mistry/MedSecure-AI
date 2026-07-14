import os
from PIL import Image, ImageDraw, ImageFont, ImageFilter

def generate_samples():
    # Define paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    public_dir = os.path.abspath(os.path.join(base_dir, "..", "frontend", "public", "samples"))
    os.makedirs(public_dir, exist_ok=True)
    
    # 1. Genuine Calpol 500mg Tablet (Indian CSV product med-33928)
    print("Generating sample 1: Genuine Calpol 500mg Tablet...")
    img1 = Image.new('RGB', (800, 500), color='#ffffff')
    draw1 = ImageDraw.Draw(img1)
    
    # Draw Green stripes
    draw1.rectangle([0, 0, 800, 60], fill='#10b981')
    draw1.rectangle([0, 440, 800, 500], fill='#10b981')
    
    # Draw text
    # We use default font since we are running headless, which works perfectly
    draw1.text((50, 100), "Calpol 500mg Tablet", fill='#111827', font_size=40)
    draw1.text((50, 160), "Paracetamol Tablets IP 500mg", fill='#4b5563', font_size=20)
    draw1.text((50, 220), "Mfg By: Glaxo SmithKline Pharmaceuticals Ltd", fill='#1f2937', font_size=22)
    draw1.text((50, 270), "Batch No: GP43210", fill='#1f2937', font_size=22)
    draw1.text((50, 310), "EXP Date: 08-2027", fill='#1f2937', font_size=22)
    draw1.text((50, 350), "MRP Rs. 16.65 (Incl. of all taxes)", fill='#1f2937', font_size=22)
    
    img1.save(os.path.join(public_dir, "calpol_genuine.jpg"), "JPEG")
    
    # 2. Counterfeit Crocin Advance Tablet (Indian CSV product med-33924)
    print("Generating sample 2: Counterfeit Crocin Advance Tablet...")
    img2 = Image.new('RGB', (800, 500), color='#ffffff')
    draw2 = ImageDraw.Draw(img2)
    
    # Draw Red stripes
    draw2.rectangle([0, 0, 800, 60], fill='#de2c2c')
    draw2.rectangle([0, 440, 800, 500], fill='#de2c2c')
    
    # Draw text
    draw2.text((50, 100), "Crocin Advance Tablet", fill='#111827', font_size=40)
    draw2.text((50, 160), "Paracetamol Analgesic 500mg", fill='#4b5563', font_size=20)
    draw2.text((50, 220), "Mfg By: GlaxoSmithKline Consumer Healthcare", fill='#1f2937', font_size=22)
    draw2.text((50, 270), "Batch No: INVALID-999-BATCH", fill='#1f2937', font_size=22) # Mismatch
    draw2.text((50, 310), "EXP Date: 12-2028", fill='#1f2937', font_size=22)
    draw2.text((50, 350), "MRP Rs. 22.62", fill='#1f2937', font_size=22)
    
    img2.save(os.path.join(public_dir, "crocin_counterfeit.jpg"), "JPEG")

    # 3. Counterfeit Bromez 20mg Capsule (Indian CSV product med-28898)
    print("Generating sample 3: Counterfeit Bromez 20mg Capsule...")
    img3 = Image.new('RGB', (800, 500), color='#ffffff')
    draw3 = ImageDraw.Draw(img3)
    
    # Mismatched background color: Blue instead of Red/Rose
    draw3.rectangle([0, 0, 800, 60], fill='#3b82f6')
    draw3.rectangle([0, 440, 800, 500], fill='#3b82f6')
    
    # Draw text
    draw3.text((50, 100), "Bromez 20mg Capsule", fill='#111827', font_size=40)
    draw3.text((50, 160), "Omeprazole Gastro-resistant Capsules IP", fill='#4b5563', font_size=20)
    draw3.text((50, 220), "Mfg By: New Medicon Pharma Labs Pvt Ltd", fill='#1f2937', font_size=22)
    draw3.text((50, 270), "Batch No: MC8872", fill='#1f2937', font_size=22)
    draw3.text((50, 310), "EXP Date: 12-2028", fill='#1f2937', font_size=22)
    draw3.text((50, 350), "MRP Rs. 55.00", fill='#1f2937', font_size=22)
    
    # Apply Gaussian Blur to simulate cheap scanning printing quality defect
    img3_blurred = img3.filter(ImageFilter.GaussianBlur(radius=5))
    img3_blurred.save(os.path.join(public_dir, "omez_counterfeit.jpg"), "JPEG")
    
    print("All sample images generated successfully.")

if __name__ == "__main__":
    generate_samples()
