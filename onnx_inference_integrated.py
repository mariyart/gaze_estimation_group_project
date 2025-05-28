import cv2
import uniface
import argparse
import numpy as np
import onnxruntime as ort
import logging
import time
import mediapipe as mp

from typing import Tuple
from utils.helpers import draw_bbox_gaze

# =================== Kalman Filter ===================
class KalmanFilter1D:
    def __init__(self, process_variance=1e-3, measurement_variance=1e-1):
        self.x, self.P = 0.0, 1.0
        self.Q, self.R = process_variance, measurement_variance

    def update(self, measurement):
        self.P += self.Q
        K = self.P / (self.P + self.R)
        self.x += K * (measurement - self.x)
        self.P *= (1 - K)
        return self.x

# =================== Gaze Zones ===================
gaze_zones = {
    "left_mirror": {"pitch": (-47.98, -27.46), "yaw": (-6.92, 4.71)},
    "left": {"pitch": (-53.11, -27.78), "yaw": (-1.09, 5.71)},
    "front": {"pitch": (-47.14, 7.78), "yaw": (-1.41, 9.06)},
    "center_mirror": {"pitch": (4.83, 26.34), "yaw": (-3.00, 10.06)},
    "front_right": {"pitch": (23.91, 31.33), "yaw": (-2.23, 9.28)},
    "right_mirror": {"pitch": (29.47, 46.29), "yaw": (-5.95, -0.39)},
    "right": {"pitch": (42.01, 54.04), "yaw": (-4.68, 3.40)},
    "infotainment": {"pitch": (22.50, 47.32), "yaw": (-22.60, -1.04)},
    "steering_wheel": {"pitch": (-11.63, 25.46), "yaw": (-40.24, 10.39)},
    "not_valid": {"pitch": (-14.78, -5.54), "yaw": (-7.58, 9.03)}
}

def identify_gaze_zone(pitch_deg, yaw_deg):
    for zone, limits in gaze_zones.items():
        if limits["pitch"][0] <= pitch_deg <= limits["pitch"][1] and limits["yaw"][0] <= yaw_deg <= limits["yaw"][1]:
            return zone
    return "not_valid"

# =================== ONNX Estimation ===================
class GazeEstimationONNX:
    def __init__(self, model_path: str):
        self.session = ort.InferenceSession(
            model_path,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        self._bins = 90
        self._binwidth = 4
        self._angle_offset = 180
        self.idx_tensor = np.arange(self._bins, dtype=np.float32)
        self.input_mean = [0.485, 0.456, 0.406]
        self.input_std = [0.229, 0.224, 0.225]
        self.input_size = tuple(self.session.get_inputs()[0].shape[2:][::-1])
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, self.input_size).astype(np.float32) / 255.0
        image = (image - self.input_mean) / self.input_std
        image = np.transpose(image, (2, 0, 1))
        return np.expand_dims(image, axis=0).astype(np.float32)

    def softmax(self, x: np.ndarray) -> np.ndarray:
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e_x / e_x.sum(axis=1, keepdims=True)

    def decode(self, pitch_logits: np.ndarray, yaw_logits: np.ndarray) -> Tuple[float, float, float, float]:
        pitch_probs = self.softmax(pitch_logits)
        yaw_probs = self.softmax(yaw_logits)
        pitch = np.sum(pitch_probs * self.idx_tensor, axis=1) * self._binwidth - self._angle_offset
        yaw = np.sum(yaw_probs * self.idx_tensor, axis=1) * self._binwidth - self._angle_offset
        return np.radians(pitch[0]), np.radians(yaw[0]), pitch[0], yaw[0]

    def estimate(self, face_image: np.ndarray) -> Tuple[float, float, float, float]:
        input_tensor = self.preprocess(face_image)
        outputs = self.session.run(self.output_names, {self.input_name: input_tensor})
        return self.decode(outputs[0], outputs[1])

# =================== Main ===================
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", default=None)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    cap = cv2.VideoCapture(int(args.source) if args.source.isdigit() else args.source)
    if not cap.isOpened():
        raise RuntimeError("Video source not available.")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30

    writer = None
    if args.output:
        writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
    CAMERA_MATRIX = np.array([[578.7566, 0.0, 349.099], [0.0, 575.6480, 239.78], [0.0, 0.0, 1.0]], dtype="double")
    IRIS_DIAMETER_MM = 11.7
    alpha = 0.9
    smoothed_depth = None

    engine = GazeEstimationONNX(args.model)
    detector = uniface.RetinaFace()

    pitch_kf, yaw_kf = KalmanFilter1D(), KalmanFilter1D()

    while cap.isOpened():
        start = time.time()
        ret, frame = cap.read()
        if not ret:
            break

        # Depth Estimation + Iris Position
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                lm = face_landmarks.landmark
                for iris_ids in ([474, 475, 476, 477], [469, 470, 471, 472]):
                    iris = np.array([(lm[i].x * width, lm[i].y * height) for i in iris_ids])
                    iris_center = np.mean(iris, axis=0).astype(int)
                    cv2.circle(frame, tuple(iris_center), 3, (0, 0, 255), -1)
                iris_diameter_pixels = np.linalg.norm(iris[0] - iris[2])
                if iris_diameter_pixels > 0:
                    focal_px = (CAMERA_MATRIX[0, 0] + CAMERA_MATRIX[1, 1]) / 2
                    depth_now = (focal_px * IRIS_DIAMETER_MM) / (iris_diameter_pixels * 1000)
                    smoothed_depth = depth_now if smoothed_depth is None else alpha * smoothed_depth + (1 - alpha) * depth_now
                    cv2.putText(frame, f"Depth: {smoothed_depth:.2f} m", (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        bboxes, _ = detector.detect(frame)
        for bbox in bboxes:
            x_min, y_min, x_max, y_max = map(int, bbox[:4])
            face = frame[y_min:y_max, x_min:x_max]
            if face.size == 0:
                continue

            try:
                pitch_rad, yaw_rad, pitch_deg, yaw_deg = engine.estimate(face)
            except:
                continue

            pitch_sm = pitch_kf.update(pitch_deg)
            yaw_sm = yaw_kf.update(yaw_deg)
            zone = identify_gaze_zone(pitch_sm, yaw_sm)
            draw_bbox_gaze(frame, bbox, np.radians(pitch_sm), np.radians(yaw_sm))
            cv2.putText(frame, zone, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, f"Pitch: {pitch_sm:.1f} Yaw: {yaw_sm:.1f}", (x_min, y_max + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        fps_text = f"FPS: {1.0 / (time.time() - start):.2f}"
        cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        if writer:
            writer.write(frame)

        cv2.imshow("Gaze Estimation", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
