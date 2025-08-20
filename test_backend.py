#!/usr/bin/env python3
"""
Simple test script for the AI Video Editor backend API
"""

import requests
import time
import os

BASE_URL = "http://localhost:8000"

def test_api_health():
    """Test if the API is running"""
    try:
        response = requests.get(f"{BASE_URL}/docs")
        if response.status_code == 200:
            print("✅ API is running and accessible")
            return True
        else:
            print(f"❌ API returned status {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to API. Make sure it's running on http://localhost:8000")
        return False

def test_upload_endpoint():
    """Test the upload endpoint with a dummy file"""
    print("\n🧪 Testing upload endpoint...")
    
    # Create a dummy video file (just a text file for testing)
    test_file_path = "test_video.txt"
    with open(test_file_path, "w") as f:
        f.write("This is a test video file")
    
    try:
        with open(test_file_path, "rb") as f:
            files = {"file": ("test_video.mp4", f, "video/mp4")}
            response = requests.post(f"{BASE_URL}/upload", files=files)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Upload successful! Upload ID: {result['upload_id']}")
            return result['upload_id']
        else:
            print(f"❌ Upload failed with status {response.status_code}: {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ Upload test failed: {e}")
        return None
    finally:
        # Clean up test file
        if os.path.exists(test_file_path):
            os.remove(test_file_path)

def test_processing_endpoints(upload_id):
    """Test the processing endpoints"""
    if not upload_id:
        print("❌ Cannot test processing without upload ID")
        return
    
    print(f"\n🧪 Testing processing endpoints with upload ID: {upload_id}")
    
    endpoints = [
        "background-removal",
        "subtitles", 
        "scene-detection",
        "voice-translate",
        "style",
        "object-remove"
    ]
    
    for endpoint in endpoints:
        print(f"  Testing {endpoint}...")
        try:
            response = requests.post(
                f"{BASE_URL}/process/{endpoint}",
                params={"upload_id": upload_id}
            )
            
            if response.status_code == 200:
                print(f"    ✅ {endpoint} endpoint working")
            else:
                print(f"    ❌ {endpoint} failed: {response.status_code}")
                
        except Exception as e:
            print(f"    ❌ {endpoint} error: {e}")

def test_status_endpoint(upload_id):
    """Test the status endpoint"""
    if not upload_id:
        return
    
    print(f"\n🧪 Testing status endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/status/{upload_id}")
        if response.status_code == 200:
            status = response.json()
            print(f"✅ Status endpoint working: {status['status']}")
        else:
            print(f"❌ Status endpoint failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Status test failed: {e}")

def main():
    """Run all tests"""
    print("🚀 AI Video Editor Backend API Test")
    print("=" * 40)
    
    # Test 1: API Health
    if not test_api_health():
        print("\n❌ API is not accessible. Please start the backend server first:")
        print("   cd backend")
        print("   python main.py")
        return
    
    # Test 2: Upload
    upload_id = test_upload_endpoint()
    
    # Test 3: Processing endpoints
    test_processing_endpoints(upload_id)
    
    # Test 4: Status endpoint
    test_status_endpoint(upload_id)
    
    print("\n🎉 Backend API tests completed!")
    print("\nTo start the full application:")
    print("1. Backend: cd backend && python main.py")
    print("2. Frontend: cd frontend && npm install && npm run dev")
    print("3. Or use: start.bat (Windows) or start.ps1 (PowerShell)")

if __name__ == "__main__":
    main()
