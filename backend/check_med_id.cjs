const sqlite3 = require('sqlite3');
const db=new sqlite3.Database('../db/indian_pharmaceutical_products.db');
db.all("SELECT id, medicine_id FROM scans WHERE id='scan-kkpbdhppf7mru7n0un'", (e,r)=>console.log(r));
