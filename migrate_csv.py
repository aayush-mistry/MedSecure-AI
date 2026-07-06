import sqlite3
import csv
import ast
import json
import time

DB_PATH = 'db/medsecure.db'
CSV_PATH = 'indian_pharmaceutical_products_clean.csv'

def migrate_db():
    print("Connecting to database...")
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    print("Dropping existing medicines table if needed...")
    cur.execute("DROP TABLE IF EXISTS medicines")

    print("Creating new optimized medicines table...")
    cur.execute("""
    CREATE TABLE medicines (
        id TEXT PRIMARY KEY,
        brand_name TEXT,
        name TEXT,                  -- backward compatibility alias
        generic_name TEXT,          -- backward compatibility alias
        manufacturer_name TEXT,     -- backward compatibility alias
        manufacturer TEXT,
        dosage_form TEXT,
        pack_size REAL,
        pack_unit TEXT,
        primary_ingredient TEXT,
        primary_strength TEXT,
        active_ingredients TEXT,
        composition TEXT,
        therapeutic_class TEXT,
        packaging_raw TEXT,
        price_inr REAL,
        is_discontinued INTEGER DEFAULT 0,
        cdsco_license TEXT,
        approved_batch_format TEXT DEFAULT '^[A-Z0-9]{2,}\\d{4,6}$',
        expected_colors TEXT DEFAULT '{"primary":"#2563eb","secondary":"#ffffff"}',
        reference_image_url TEXT DEFAULT '/reference/default.jpg',
        logo_embedding TEXT DEFAULT '[]',
        barcode_required INTEGER DEFAULT 0
    )
    """)

    print("Creating indexes...")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_medicines_brand_name ON medicines(brand_name)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_medicines_manufacturer ON medicines(manufacturer)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_medicines_primary_ingredient ON medicines(primary_ingredient)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_medicines_therapeutic_class ON medicines(therapeutic_class)")

    print("Reading CSV and parsing records...")
    records = []
    seen = set()
    
    start_time = time.time()
    
    with open(CSV_PATH, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            brand = row['brand_name'].strip()
            mfg = row['manufacturer'].strip()
            
            # Deduplicate by brand and manufacturer
            key = (brand.lower(), mfg.lower())
            if key in seen:
                continue
            seen.add(key)
            
            med_id = f"med-{row['product_id']}"
            
            # Parse active ingredients securely
            raw_ai = row.get('active_ingredients', '[]')
            parsed_ai = []
            composition_list = []
            if raw_ai and raw_ai.strip() not in ('', '[]', 'None'):
                try:
                    # CSV contains stringified Python list of dicts: "[{'name': 'Amoxycillin', ...}]"
                    parsed_ai = ast.literal_eval(raw_ai)
                    if isinstance(parsed_ai, list):
                        for item in parsed_ai:
                            # Extract composition strings for backward compatibility
                            name_str = item.get('name', '')
                            strength_str = item.get('strength') or ''
                            if name_str:
                                comp_str = f"{name_str} {strength_str}".strip()
                                composition_list.append(comp_str)
                except Exception:
                    pass

            # Default barcode_required (assume 1 for Box, 0 for Strip/others generally for demo purposes, 
            # but we can default to 0 and rely on batch level barcode checks)
            barcode_required = 1 if 'box' in row.get('pack_unit', '').lower() else 0

            # Default colors
            default_colors = json.dumps({"primary": "#2563eb", "secondary": "#ffffff"})

            record = (
                med_id,
                brand,
                brand,  # name
                row.get('primary_ingredient', ''),  # generic_name
                mfg,  # manufacturer_name
                mfg,
                row.get('dosage_form', ''),
                float(row['pack_size']) if row.get('pack_size') else 0.0,
                row.get('pack_unit', ''),
                row.get('primary_ingredient', ''),
                row.get('primary_strength', ''),
                json.dumps(parsed_ai),
                json.dumps(composition_list),
                row.get('therapeutic_class', ''),
                row.get('packaging_raw', ''),
                float(row['price_inr']) if row.get('price_inr') else 0.0,
                1 if row.get('is_discontinued') == 'True' else 0,
                '',  # cdsco_license
                r'^[A-Z0-9]{2,}\d{4,6}$',  # approved_batch_format
                default_colors,  # expected_colors
                '/reference/default.jpg',  # reference_image_url
                '[]',  # logo_embedding
                barcode_required
            )
            records.append(record)
            
            if i % 50000 == 0 and i > 0:
                print(f"  Parsed {i} rows...")

    print(f"Finished parsing. Unique records to insert: {len(records)}")
    
    print("Inserting into SQLite (this might take a few moments)...")
    cur.executemany("""
    INSERT INTO medicines (
        id, brand_name, name, generic_name, manufacturer_name, manufacturer,
        dosage_form, pack_size, pack_unit, primary_ingredient, primary_strength,
        active_ingredients, composition, therapeutic_class, packaging_raw,
        price_inr, is_discontinued, cdsco_license, approved_batch_format,
        expected_colors, reference_image_url, logo_embedding, barcode_required
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, records)

    print("Updating existing demo batches to point to new CSV records...")
    # Mapping old demo med IDs to new real CSV product IDs
    # Crocin -> med-33924
    # Calpol -> med-33928
    # Pantocid -> med-173591
    # Omez -> med-161640
    # Dolo 650 -> med-58235
    
    mapping = {
        'med-33924': ['med-0', 'med-1', 'med-2'],
        'med-33928': ['med-3', 'med-4'],
        'med-173591': ['med-8', 'med-9'],
        'med-161640': ['med-10', 'med-11'],
        'med-58235': ['med-33']
    }
    
    for new_id, old_ids in mapping.items():
        placeholders = ','.join(['?'] * len(old_ids))
        query = f"UPDATE medicine_batches SET medicine_id = ? WHERE medicine_id IN ({placeholders})"
        cur.execute(query, [new_id] + old_ids)
        
        # Update scans table as well just in case
        query_scans = f"UPDATE scans SET medicine_id = ? WHERE medicine_id IN ({placeholders})"
        cur.execute(query_scans, [new_id] + old_ids)
        
        # Update alerts table
        query_alerts = f"UPDATE alerts SET medicine_id = ? WHERE medicine_id IN ({placeholders})"
        cur.execute(query_alerts, [new_id] + old_ids)

    conn.commit()
    conn.close()
    
    elapsed = time.time() - start_time
    print(f"Migration completed successfully in {elapsed:.2f} seconds!")

if __name__ == "__main__":
    migrate_db()
