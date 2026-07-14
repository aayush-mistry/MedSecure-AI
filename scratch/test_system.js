const BASE_URL = "http://localhost:3001/api/v1";

async function testApi() {
  console.log("Testing MedSecure AI API...");
  
  let token = null;
  
  // 1. Test Auth Login
  try {
    const res = await fetch(`${BASE_URL}/auth/login`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ email: "inspector@medsecure.gov.in", password: "secure-inspector-password" })
    });
    
    if (res.ok) {
      console.log("Auth: Login successful");
      const data = await res.json();
      token = data.token;
    } else {
      console.log(`Auth: Login failed ${res.status}`);
      const text = await res.text();
      console.log(text);
    }
  } catch (e) {
    console.log(`Auth: Exception ${e.message}`);
  }

  const headers = token ? { "Authorization": `Bearer ${token}` } : {};

  // 2. Test GET Dashboard
  try {
    const res = await fetch(`${BASE_URL}/dashboard/pharmacist`, { headers });
    if (res.ok) {
      console.log("Dashboard: Success");
    } else {
      console.log(`Dashboard: Failed ${res.status}`);
      const text = await res.text();
      console.log(text);
    }
  } catch (e) {
    console.log(`Dashboard: Exception ${e.message}`);
  }
}

testApi();
