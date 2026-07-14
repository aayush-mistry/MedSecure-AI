import ast
import csv
import json
import os
import sqlite3
import time

DB_PATH = os.path.join("db", "indian_pharmaceutical_products.db")
CSV_PATH = "indian_pharmaceutical_products_clean.csv"


def parse_float(value, default=0.0):
    try:
        return float(value) if value not in (None, "") else default
    except ValueError:
        return default


def parse_bool(value):
    return 1 if str(value).strip().lower() == "true" else 0


def parse_ingredients(raw_value):
    parsed = []
    composition = []
    if raw_value and raw_value.strip() not in ("", "[]", "None"):
        try:
            parsed = ast.literal_eval(raw_value)
            if isinstance(parsed, list):
                for item in parsed:
                    name = item.get("name", "")
                    strength = item.get("strength") or ""
                    if name:
                        composition.append(f"{name} {strength}".strip())
            else:
                parsed = []
        except (SyntaxError, ValueError):
            parsed = []
    return parsed, composition


def create_app_schema(cur):
    cur.executescript(
        """
        DROP TABLE IF EXISTS reports;
        DROP TABLE IF EXISTS alerts;
        DROP TABLE IF EXISTS scans;
        DROP TABLE IF EXISTS medicine_batches;
        DROP TABLE IF EXISTS medicines;

        CREATE TABLE medicines (
            id TEXT PRIMARY KEY,
            brand_name TEXT,
            name TEXT,
            generic_name TEXT,
            manufacturer_name TEXT,
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
        );

        CREATE TABLE medicine_batches (
            id TEXT PRIMARY KEY,
            medicine_id TEXT NOT NULL,
            batch_number TEXT NOT NULL,
            manufacturer TEXT,
            manufacturing_date TEXT,
            expiry_date TEXT,
            mrp TEXT,
            manufacturing_license TEXT,
            barcode_value TEXT,
            barcode_required INTEGER DEFAULT 0,
            pack_type TEXT,
            pack_size TEXT,
            country_of_origin TEXT DEFAULT 'India',
            status TEXT DEFAULT 'active' CHECK(status IN ('active', 'recalled', 'expired')),
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(medicine_id) REFERENCES medicines(id)
        );

        CREATE TABLE users (
            id TEXT PRIMARY KEY,
            email TEXT UNIQUE,
            password_hash TEXT,
            role TEXT CHECK(role IN ('consumer', 'pharmacist', 'healthcare_worker', 'inspector')),
            verified INTEGER DEFAULT 0,
            license_number TEXT,
            pin_code TEXT,
            language TEXT DEFAULT 'en',
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE scans (
            id TEXT PRIMARY KEY,
            user_id TEXT,
            medicine_id TEXT,
            batch_id TEXT,
            image_url TEXT,
            authenticity_score REAL,
            verdict TEXT CHECK(verdict IN ('verified', 'caution', 'high_risk')),
            ocr_extracted TEXT,
            db_match_results TEXT,
            image_analysis TEXT,
            barcode_status TEXT,
            anomalies TEXT,
            signal_breakdown TEXT,
            lat REAL,
            lng REAL,
            scanned_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(user_id) REFERENCES users(id),
            FOREIGN KEY(medicine_id) REFERENCES medicines(id),
            FOREIGN KEY(batch_id) REFERENCES medicine_batches(id)
        );

        CREATE TABLE alerts (
            id TEXT PRIMARY KEY,
            medicine_id TEXT,
            batch_number TEXT,
            report_count INTEGER DEFAULT 1,
            lat REAL,
            lng REAL,
            severity TEXT CHECK(severity IN ('caution', 'high')),
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            last_updated TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(medicine_id) REFERENCES medicines(id)
        );

        CREATE TABLE reports (
            id TEXT PRIMARY KEY,
            scan_id TEXT,
            user_id TEXT,
            medicine_id TEXT,
            batch_number TEXT,
            notes TEXT,
            lat REAL,
            lng REAL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(scan_id) REFERENCES scans(id),
            FOREIGN KEY(user_id) REFERENCES users(id),
            FOREIGN KEY(medicine_id) REFERENCES medicines(id)
        );

        CREATE INDEX idx_medicines_brand_name ON medicines(brand_name);
        CREATE INDEX idx_medicines_name ON medicines(name);
        CREATE INDEX idx_medicines_manufacturer ON medicines(manufacturer);
        CREATE INDEX idx_medicines_primary_ingredient ON medicines(primary_ingredient);
        CREATE INDEX idx_medicines_therapeutic_class ON medicines(therapeutic_class);
        CREATE INDEX idx_batches_medicine_batch ON medicine_batches(medicine_id, batch_number);
        CREATE INDEX idx_batches_batch_number ON medicine_batches(batch_number);
        """
    )


def medicine_record(row):
    parsed_ai, composition = parse_ingredients(row.get("active_ingredients", "[]"))
    pack_unit = row.get("pack_unit", "")
    packaging = row.get("packaging_raw", "")
    barcode_required = 1 if "box" in f"{pack_unit} {packaging}".lower() else 0
    return (
        f"med-{row['product_id']}",
        row.get("brand_name", "").strip(),
        row.get("brand_name", "").strip(),
        row.get("primary_ingredient", "").strip(),
        row.get("manufacturer", "").strip(),
        row.get("manufacturer", "").strip(),
        row.get("dosage_form", "").strip(),
        parse_float(row.get("pack_size")),
        pack_unit.strip(),
        row.get("primary_ingredient", "").strip(),
        row.get("primary_strength", "").strip(),
        json.dumps(parsed_ai),
        json.dumps(composition),
        row.get("therapeutic_class", "").strip(),
        packaging.strip(),
        parse_float(row.get("price_inr")),
        parse_bool(row.get("is_discontinued")),
        "",
        r"^[A-Z0-9]{2,}\d{4,6}$",
        json.dumps({"primary": "#2563eb", "secondary": "#ffffff"}),
        "/reference/default.jpg",
        "[]",
        barcode_required,
    )


def seed_batches(cur):
    fixtures = [
        ("batch-calpol-500-001", "med-33928", "GP43210", "08/2024", "07/2027", "CALPOL500-GP43210-IN"),
        ("batch-calpol-650-001", "med-33933", "GP51009", "02/2025", "01/2028", "CALPOL650-GP51009-IN"),
        ("batch-crocin-001", "med-33924", "BT4521", "03/2025", "02/2028", "CROCIN-BT4521-IN"),
        ("batch-dolo-650-001", "med-58235", "GP99210", "05/2024", "04/2027", "DOLO650-GP99210-ML-IN"),
        ("batch-bromez-20-001", "med-28898", "OMZ441", "12/2024", "11/2027", "BROMEZ20-OMZ441-IN"),
    ]
    for batch_id, medicine_id, batch_number, mfg_date, exp_date, barcode in fixtures:
        med = cur.execute(
            "SELECT manufacturer_name, packaging_raw, price_inr, pack_size, pack_unit FROM medicines WHERE id=?",
            (medicine_id,),
        ).fetchone()
        if not med:
            continue
        pack_size = f"{med[3]:g} {med[4]}".strip() if med[3] else med[4]
        cur.execute(
            """
            INSERT INTO medicine_batches (
                id, medicine_id, batch_number, manufacturer, manufacturing_date,
                expiry_date, mrp, manufacturing_license, barcode_value, barcode_required,
                pack_type, pack_size, country_of_origin, status
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'India', 'active')
            """,
            (
                batch_id,
                medicine_id,
                batch_number,
                med[0],
                mfg_date,
                exp_date,
                f"Rs. {float(med[2] or 0):.2f}",
                "CDSCO/IND/REF",
                barcode,
                0,
                med[4] or "pack",
                pack_size,
            ),
        )


def seed_alerts(cur):
    alerts = [
        ("alt-1", "med-33924", "INVALID-999-BATCH", 14, 23.0225, 72.5714, "high"),
        ("alt-2", "med-28898", "MC8872", 4, 21.1702, 72.8311, "caution"),
        ("alt-3", "med-33928", "GP43210", 8, 22.3072, 73.1812, "high"),
    ]
    cur.executemany(
        """
        INSERT INTO alerts (id, medicine_id, batch_number, report_count, lat, lng, severity)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        alerts,
    )


def migrate_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    start_time = time.time()
    print(f"Building canonical Indian pharmaceutical DB: {DB_PATH}")
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    create_app_schema(cur)

    records = []
    seen = set()
    with open(CSV_PATH, mode="r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader, start=1):
            brand = row.get("brand_name", "").strip()
            mfg = row.get("manufacturer", "").strip()
            if not brand or not mfg:
                continue
            key = (brand.lower(), mfg.lower())
            if key in seen:
                continue
            seen.add(key)
            records.append(medicine_record(row))
            if i % 50000 == 0:
                print(f"  Parsed {i} CSV rows...")

    print(f"Inserting {len(records)} unique Indian product records...")
    cur.executemany(
        """
        INSERT INTO medicines (
            id, brand_name, name, generic_name, manufacturer_name, manufacturer,
            dosage_form, pack_size, pack_unit, primary_ingredient, primary_strength,
            active_ingredients, composition, therapeutic_class, packaging_raw,
            price_inr, is_discontinued, cdsco_license, approved_batch_format,
            expected_colors, reference_image_url, logo_embedding, barcode_required
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        records,
    )

    seed_batches(cur)
    seed_alerts(cur)
    conn.commit()
    conn.close()
    print(f"Done in {time.time() - start_time:.2f}s.")


if __name__ == "__main__":
    migrate_db()
