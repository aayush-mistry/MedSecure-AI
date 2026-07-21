const sqlite3 = require('sqlite3').verbose();
const db = new sqlite3.Database('db/indian_pharmaceutical_products.db');

db.serialize(() => {
  db.get("SELECT sql FROM sqlite_master WHERE name='medicines'", (err, row) => {
    if (err) console.error(err);
    else console.log(row.sql);
  });
});
db.close();
