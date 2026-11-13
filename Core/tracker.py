import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

class Track:
    """Her bir dronun kendi Kalman Filtresini ve durumunu yönettiği sınıf."""
    def __init__(self, bbox, track_id):
        self.track_id = track_id
        self.bbox = bbox  # [x1, y1, x2, y2]
        self.state = "TRACKING"
        self.lost_counter = 0
        
        #Kalman filtresi
        self.kf = cv2.KalmanFilter(4, 2, 0)
        self.kf.transitionMatrix = np.array(
            [
                [1, 0, 1, 0],
                [0, 1, 0, 1],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ],
            dtype=np.float32,
        )
        
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0],[0, 1, 0, 0]], dtype=np.float32)
        
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.1
        
        cx, cy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
        self.kf.statePost = np.array([[cx], [cy], [0], [0]], dtype=np.float32)

    def predict(self):
        prediction = self.kf.predict()
        cx, cy = prediction[0][0], prediction[1][0]
        
        w = self.bbox[2] - self.bbox[0]
        h = self.bbox[3] - self.bbox[1]
        self.bbox = [int(cx - w/2), int(cy - h/2), int(cx + w/2), int(cy + h/2)]
        return self.bbox

    def update(self, bbox):
        cx, cy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
        measurement = np.array([[cx], [cy]], dtype=np.float32)
        self.kf.correct(measurement)
        self.bbox = bbox


class DroneTracker:
    def __init__(self, max_lost_frames: int = 30):
        self.tracks = []
        self.next_id = 1
        self.max_lost_frames = max_lost_frames

    def _calculate_iou(self, box1, box2):
        x_left = max(box1[0], box2[0])
        y_top = max(box1[1], box2[1])
        x_right = min(box1[2], box2[2])
        y_bottom = min(box1[3], box2[3])

        if x_right < x_left or y_bottom < y_top:
            return 0.0

        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union_area = box1_area + box2_area - intersection_area
        if union_area <= 0:
            return 0.0
        return intersection_area / float(union_area)

    def _get_center(self, box):
        return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)

    def update(self, detections: list) -> list:
        for track in self.tracks:
            track.predict()

        if len(self.tracks) == 0:
            for det in detections:
                self.tracks.append(Track(det[:4], self.next_id))
                self.next_id += 1
        else:
            cost_matrix = np.zeros((len(self.tracks), len(detections)), dtype=np.float32)
            
            for t_idx, track in enumerate(self.tracks):
                t_cx, t_cy = self._get_center(track.bbox)
                for d_idx, det in enumerate(detections):
                    d_cx, d_cy = self._get_center(det[:4])
                    
                    iou = self._calculate_iou(track.bbox, det[:4])
                    distance = np.sqrt((t_cx - d_cx)**2 + (t_cy - d_cy)**2)
                    
                    cost_matrix[t_idx, d_idx] = (1.0 - iou) + (distance * 0.001)

            #Macar algoritması
            track_indices, det_indices = linear_sum_assignment(cost_matrix)

            matched_tracks = set()
            matched_dets = set()

            for t, d in zip(track_indices, det_indices):
                if cost_matrix[t, d] < 1.2:  
                    if self.tracks[t].state in ["LOST", "UNCERTAIN"]:
                        self.tracks[t].state = "REACQUIRED"
                    else:
                        self.tracks[t].state = "TRACKING"
                        
                    self.tracks[t].update(detections[d][:4])
                    self.tracks[t].lost_counter = 0
                    matched_tracks.add(t)
                    matched_dets.add(d)

            for t_idx, track in enumerate(self.tracks):
                if t_idx not in matched_tracks:
                    track.lost_counter += 1
                    
                    if track.lost_counter <= 3:
                        track.state = "UNCERTAIN"
                    elif track.lost_counter <= self.max_lost_frames:
                        track.state = "LOST"
                    else:
                        track.state = "DEAD"

            for d_idx, det in enumerate(detections):
                if d_idx not in matched_dets:
                    self.tracks.append(Track(det[:4], self.next_id))
                    self.next_id += 1

            self.tracks = [t for t in self.tracks if t.state != "DEAD"]

        return [[t.bbox, t.track_id, t.state] for t in self.tracks]
