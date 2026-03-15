import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
import random
import os
import urllib.request

# Download gesture recognizer model if missing
def ensure_model_exists():
    model_path = 'gesture_recognizer.task'
    if not os.path.exists(model_path):
        print(f"Downloading gesture recognizer model...")
        url = 'https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task'
        try:
            urllib.request.urlretrieve(url, model_path)
            print(f"Model downloaded successfully: {model_path}")
        except Exception as e:
            print(f"Error downloading model: {e}")
            raise RuntimeError(f"Failed to download gesture_recognizer.task: {e}")
    return os.path.abspath(model_path)

class ProArtCanvas:
    def __init__(self):
        # Initialize Mediapipe Tasks
        model_path = ensure_model_exists()
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.GestureRecognizerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.8
        )
        self.recognizer = vision.GestureRecognizer.create_from_options(options)
        
        # Camera & Hardware
        self.cam_list = [0, 1, 2]
        self.cam_idx = 0
        self.cap = cv2.VideoCapture(self.cam_list[self.cam_idx])
        
        # Canvas & Brush States
        self.canvas = None
        self.colors = [(255, 60, 0), (60, 255, 0), (0, 60, 255), (255, 255, 255)]
        self.color_idx = 0
        self.brush_modes = ["INK", "SPRAY", "FLAT"]
        self.mode_idx = 0
        self.xp, self.yp = 0, 0
        self.last_timestamp = 0

    def get_thickness(self, x1, y1, x2, y2):
        """Dynamic thickness based on movement speed."""
        dist = np.hypot(x2 - x1, y2 - y1)
        # Faster movement = thinner line
        return max(2, min(18, int(20 - dist/5)))

    def draw_ink(self, img, x1, y1, x2, y2, color):
        thickness = self.get_thickness(x1, y1, x2, y2)
        cv2.line(img, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)

    def draw_spray(self, img, x, y, color):
        for _ in range(8):
            offset_x = int(random.gauss(0, 12))
            offset_y = int(random.gauss(0, 12))
            cv2.circle(img, (x + offset_x, y + offset_y), random.randint(1, 3), color, -1)

    def draw_flat(self, img, x1, y1, x2, y2, color):
        """Calligraphy flat nib."""
        pts = np.linspace((x1, y1), (x2, y2), 8)
        for p in pts:
            cv2.ellipse(img, (int(p[0]), int(p[1])), (18, 4), 30, 0, 360, color, -1)

    def run(self):
        while self.cap.isOpened():
            success, frame = self.cap.read()
            if not success: continue
            
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            if self.canvas is None or self.canvas.shape != frame.shape:
                self.canvas = np.zeros_like(frame)

                # Mediapipe Processing
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            timestamp = int(time.time() * 1000)
            if timestamp > self.last_timestamp:
                results = self.recognizer.recognize_for_video(mp_image, timestamp)
                self.last_timestamp = timestamp
            
            # --- MENU UI ---
            cv2.rectangle(frame, (0, 0), (w, 80), (20, 20, 20), -1)
            for i, c in enumerate(self.colors):
                cv2.circle(frame, (50 + i*80, 40), 25, c, -1)
                if i == self.color_idx: cv2.circle(frame, (50 + i*80, 40), 28, (255, 255, 255), 3)

            # Process Landmarks
            if results.hand_landmarks:
                for i, hand_lms in enumerate(results.hand_landmarks):
                    handedness = results.handedness[i][0].category_name
                    
                    # Coordinate Mapping
                    tip = hand_lms[8]
                    cx, cy = int(tip.x * w), int(tip.y * h)
                    
                    # Logic: INDEX UP and MIDDLE UP
                    idx_up = tip.y < hand_lms[6].y
                    mid_up = hand_lms[12].y < hand_lms[10].y

                    # ERASER (LEFT HAND)
                    if handedness == "Left": # Actual Right Hand due to flip
                        x_pts = [int(lm.x * w) for lm in hand_lms]
                        y_pts = [int(lm.y * h) for lm in hand_lms]
                        cv2.rectangle(self.canvas, (min(x_pts)-20, min(y_pts)-20), (max(x_pts)+20, max(y_pts)+20), (0,0,0), -1)
                        cv2.rectangle(frame, (min(x_pts)-20, min(y_pts)-20), (max(x_pts)+20, max(y_pts)+20), (200,200,200), 2)

                    # PAINTER (RIGHT HAND)
                    else:
                        if idx_up and mid_up:
                            self.xp, self.yp = 0, 0
                            if cy < 80: self.color_idx = (cx - 20) // 80 if cx < 320 else self.color_idx
                            cv2.circle(frame, (cx, cy), 15, self.colors[self.color_idx], 2)
                        elif idx_up:
                            if self.xp == 0: self.xp, self.yp = cx, cy
                            
                            color = self.colors[self.color_idx]
                            if self.mode_idx == 0: self.draw_ink(self.canvas, self.xp, self.yp, cx, cy, color)
                            elif self.mode_idx == 1: self.draw_spray(self.canvas, cx, cy, color)
                            elif self.mode_idx == 2: self.draw_flat(self.canvas, self.xp, self.yp, cx, cy, color)
                            
                            self.xp, self.yp = cx, cy
                        else: self.xp, self.yp = 0, 0

            # Combined Display
            gray = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2GRAY)
            _, inv = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY_INV)
            inv = cv2.cvtColor(inv, cv2.COLOR_GRAY2BGR)
            frame = cv2.bitwise_and(frame, inv)
            frame = cv2.bitwise_or(frame, self.canvas)

            cv2.imshow("Mediapipe Tasks Art Studio", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): break
            if key == ord('n'): # Next Cam
                self.cam_idx = (self.cam_idx + 1) % len(self.cam_list)
                self.cap.release()
                self.cap = cv2.VideoCapture(self.cam_list[self.cam_idx])
            if key == ord('m'): self.mode_idx = (self.mode_idx + 1) % 3 # Change Mode

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    ProArtCanvas().run()