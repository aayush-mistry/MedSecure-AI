import sqlite3
import os

db_path = os.path.join("db", "indian_pharmaceutical_products.db")
conn = sqlite3.connect(db_path)
try:
    conn.execute("INSERT INTO medicines (id, name, brand_name, generic_name, manufacturer_name, price_inr) VALUES ('med_panadol', 'Panadol', 'Panadol', 'Paracetamol', 'GlaxoSmithKline', 150.50)")
    conn.commit()
    print("Inserted Panadol")
except sqlite3.IntegrityError:
    print("Already inserted")
conn.close()
