import cv2
import numpy as np
from ultralytics import YOLO

class DroneDetector:
    def __init__(self, model_path: str = "yolo11n.pt", conf_threshold: float = 0.4):
        self.conf_threshold = conf_threshold
        self.model = YOLO(model_path)
        print(f"[Detector] Model {model_path} hazır. Eşik: {self.conf_threshold}")

    def set_threshold(self, new_threshold: float):
        self.conf_threshold = new_threshold

    def detect(self, frame: np.ndarray) -> list:
        detections = [] 
        results = self.model(frame, conf=self.conf_threshold, imgsz=1088, verbose=False)
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy.tolist()[0]
                conf = float(box.conf[0])

                detections.append([int(x1), int(y1), int(x2), int(y2), conf])
                
        return detections
