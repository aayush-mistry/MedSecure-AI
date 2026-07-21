const sqlite3 = require('sqlite3');
const db=new sqlite3.Database('../db/indian_pharmaceutical_products.db');
db.all("SELECT ocr_extracted, medicine_id FROM scans ORDER BY scanned_at DESC LIMIT 3", (e,r)=>console.log(r));
