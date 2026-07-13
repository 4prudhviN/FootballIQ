"""
Quick integration test for the FootballIQ FastAPI backend.
Run this AFTER starting the server on port 8000.

Usage:
    python test_backend.py
"""
import requests

# The URL where your FastAPI bridge is running
url = "http://localhost:8000/api/upload-video"

# Path to a sample short football video clip on your machine
video_path = "input_movement.mp4"

try:
    print("🚀 Sending test video to FootballIQ local backend...")
    with open(video_path, "rb") as video_file:
        files = {"file": (video_path, video_file, "video/mp4")}
        response = requests.post(url, files=files)

    if response.status_code == 200:
        data = response.json()
        print("\n✅ Test Success! Server responded perfectly:")
        print(f"🔗 Analyzed Video Stream URL: {data.get('video_url')}")
        print(f"🚨 Triggered Warnings: {data.get('warnings')}")
    else:
        print(f"❌ Server error (Status Code: {response.status_code}): {response.text}")

except FileNotFoundError:
    print(f"📁 Error: Could not find '{video_path}'. Place a short sample video in this folder to test.")
except requests.exceptions.ConnectionError:
    print("🔌 Error: Could not connect to the backend. Make sure your FastAPI server is running via Uvicorn!")
except Exception as e:
    print(f"💥 Unexpected error: {e}")