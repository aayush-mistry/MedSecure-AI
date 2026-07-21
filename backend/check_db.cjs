const sqlite3 = require('sqlite3').verbose();
const db = new sqlite3.Database('../db/indian_pharmaceutical_products.db');

db.serialize(() => {
  db.get("SELECT * FROM scans LIMIT 1", (err, row) => {
    if (err) console.error("Scans error:", err);
    else console.log("Scans OK");
  });
  db.get("SELECT * FROM alerts LIMIT 1", (err, row) => {
    if (err) console.error("Alerts error:", err);
    else console.log("Alerts OK");
  });
});
db.close();
