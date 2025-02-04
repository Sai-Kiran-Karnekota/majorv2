# backend/tests/test_ml_service.py
import pytest
from fastapi.testclient import TestClient
from app.main import app
import numpy as np
import cv2

client = TestClient(app)

def test_process_frame():
    # Create dummy frame
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    _, frame_bytes = cv2.imencode('.jpg', frame)
    
    response = client.post(
        "/process_frame",
        json={
            "frame": frame_bytes.tobytes(),
            "frame_id": 1
        }
    )
    
    assert response.status_code == 200
    assert "detections" in response.json()

def test_object_history():
    response = client.get("/object/person_1/history")
    assert response.status_code == 200
    assert "history" in response.json()