# ml/app/main.py
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
import torch
import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Dict
import asyncio
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class MLService:
    def __init__(self):
        self.model = YOLO('yolov8n.pt')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Initialize object tracking
        self.tracker = None
        self.object_history: Dict[str, List] = {}
        
    async def process_frame(self, frame_data: np.ndarray, frame_id: int) -> Dict:
        try:
            # Convert frame data to format expected by YOLO
            results = self.model(frame_data)[0]
            
            detections = []
            for det in results.boxes.data.tolist():
                x1, y1, x2, y2, conf, cls = det
                detection = {
                    'bbox': [float(x1), float(y1), float(x2-x1), float(y2-y1)],
                    'confidence': float(conf),
                    'class': int(cls),
                    'class_name': results.names[int(cls)],
                    'frame_id': frame_id
                }
                detections.append(detection)
            
            # Update tracking
            if self.tracker is None:
                self.tracker = cv2.TrackerCSRT_create()
            
            # Update object history
            for det in detections:
                obj_id = f"{det['class_name']}_{frame_id}"
                if obj_id not in self.object_history:
                    self.object_history[obj_id] = []
                self.object_history[obj_id].append({
                    'frame_id': frame_id,
                    'bbox': det['bbox'],
                    'confidence': det['confidence']
                })
            
            return {
                'frame_id': frame_id,
                'detections': detections,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error processing frame: {e}")
            raise

    async def get_object_history(self, object_id: str) -> List:
        return self.object_history.get(object_id, [])

# Initialize ML service
ml_service = MLService()

@app.websocket("/ws/process")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            # Receive frame data
            data = await websocket.receive_json()
            frame_data = np.frombuffer(data['frame'], dtype=np.uint8)
            frame_data = cv2.imdecode(frame_data, cv2.IMREAD_COLOR)
            
            # Process frame
            results = await ml_service.process_frame(frame_data, data['frame_id'])
            
            # Send results back
            await websocket.send_json(results)
            
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await websocket.close()

@app.get("/object/{object_id}/history")
async def get_object_history(object_id: str):
    history = await ml_service.get_object_history(object_id)
    return {"object_id": object_id, "history": history}