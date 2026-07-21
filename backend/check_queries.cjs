const sqlite3 = require('sqlite3').verbose();
const db = new sqlite3.Database('../db/indian_pharmaceutical_products.db');

db.all(`SELECT a.*, m.name as medicine_name, m.generic_name, m.manufacturer_name
     FROM alerts a JOIN medicines m ON a.medicine_id=m.id ORDER BY a.last_updated DESC LIMIT 30`, (err, rows) => {
  if (err) console.error("Error 1:", err);
  else console.log("Query 1 OK, rows:", rows.length);
});

db.all(`SELECT s.id, s.image_url, s.authenticity_score, s.verdict, s.scanned_at,
     m.name as medicine_name, m.generic_name, m.manufacturer_name
     FROM scans s LEFT JOIN medicines m ON s.medicine_id=m.id
     ORDER BY s.scanned_at DESC LIMIT 50`, (err, rows) => {
  if (err) console.error("Error 2:", err);
  else console.log("Query 2 OK, rows:", rows.length);
});
