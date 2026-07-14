const BASE_URL = "http://localhost:3001/api/v1";

async function testMlEndpoint() {
  try {
    const res = await fetch("http://localhost:8000/process_scan", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ scan_id: "test-scan-123", file_path: "d:/MedSecure Ai/ml/sample.jpg" })
    });
    console.log("ML status:", res.status);
    const text = await res.text();
    console.log("ML response:", text);
  } catch (err) {
    console.error("ML Error:", err.message);
  }
}

testMlEndpoint();
