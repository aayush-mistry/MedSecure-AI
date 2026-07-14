import requests
import json
import time

BASE_URL = "http://localhost:3001/api/v1"

def test_api():
    print("Testing MedSecure AI API...")
    
    # 1. Test Auth Login
    try:
        login_res = requests.post(f"{BASE_URL}/auth/login", json={"email": "pharmacist1@medsecure.ai", "password": "password"})
        if login_res.status_code == 200:
            print("Auth: Login successful")
            token = login_res.json().get("token")
        else:
            print(f"Auth: Login failed {login_res.status_code}")
            token = None
    except Exception as e:
        print(f"Auth: Exception {e}")
        token = None

    headers = {"Authorization": f"Bearer {token}"} if token else {}

    # Test GET Dashboard
    try:
        dash_res = requests.get(f"{BASE_URL}/dashboard/pharmacist", headers=headers)
        if dash_res.status_code == 200:
            print("Dashboard: Success")
        else:
            print(f"Dashboard: Failed {dash_res.status_code}")
    except Exception as e:
        print(f"Dashboard: Exception {e}")

if __name__ == "__main__":
    test_api()
