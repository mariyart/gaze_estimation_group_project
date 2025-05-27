import cv2
import uniface
import argparse
import numpy as np
import onnxruntime as ort
import logging
import time

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
            providers=["CPUExecutionProvider", "CUDAExecutionProvider"]
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

# =================== Main Loop ===================
def parse_args():
    parser = argparse.ArgumentParser(description="Gaze Estimation ONNX with Zones, Depth, Kalman, FPS")
    parser.add_argument("--source", type=str, required=True, help="Video path or camera index (e.g., 0)")
    parser.add_argument("--model", type=str, required=True, help="Path to ONNX model")
    parser.add_argument("--output", type=str, default=None, help="Path to save output video")
    return parser.parse_args()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    args = parse_args()

    try:
        source = int(args.source)
    except ValueError:
        source = args.source

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise IOError(f"Failed to open video source: {args.source}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    writer = None
    if args.output:
        writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    engine = GazeEstimationONNX(model_path=args.model)
    detector = uniface.RetinaFace()
    CAMERA_MATRIX = np.array([[578.7566, 0.0, 349.099], [0.0, 575.6480, 239.78], [0.0, 0.0, 1.0]], dtype="double")
    IRIS_DIAMETER_MM = 11.7
    alpha = 0.9
    smoothed_depth = None

    pitch_kf, yaw_kf = KalmanFilter1D(), KalmanFilter1D()

    while cap.isOpened():
        start_time = time.time()
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]

        bboxes, keypoints = detector.detect(frame)
        for bbox, kps in zip(bboxes, keypoints):
            x_min, y_min, x_max, y_max = map(int, bbox[:4])
            face_crop = frame[y_min:y_max, x_min:x_max]

            if face_crop is None or face_crop.size == 0:
                logging.warning("Empty image crop, skipping...")
                continue

            try:
                pitch_rad, yaw_rad, pitch_deg, yaw_deg = engine.estimate(face_crop)
            except Exception as e:
                logging.warning(f"Estimation failed: {e}")
                continue

            pitch_deg_sm = pitch_kf.update(pitch_deg)
            yaw_deg_sm = yaw_kf.update(yaw_deg)
            zone = identify_gaze_zone(pitch_deg_sm, yaw_deg_sm)
            draw_bbox_gaze(frame, bbox, np.radians(pitch_deg_sm), np.radians(yaw_deg_sm))
            cv2.putText(frame, f"{zone}", (x_min, y_min - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Heuristic iris depth estimation using keypoints (eye corners)
            left_eye = np.array(kps[0])
            right_eye = np.array(kps[1])
            iris_diameter_pixels = np.linalg.norm(left_eye - right_eye) / 2  # Approximate radius
            if iris_diameter_pixels > 0:
                focal_length_px = (CAMERA_MATRIX[0, 0] + CAMERA_MATRIX[1, 1]) / 2
                depth_current = (focal_length_px * IRIS_DIAMETER_MM) / (iris_diameter_pixels * 1000)
                smoothed_depth = depth_current if smoothed_depth is None else alpha * smoothed_depth + (1 - alpha) * depth_current
                cv2.putText(frame, f"Depth: {smoothed_depth:.2f} m", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        fps_calc = 1.0 / (time.time() - start_time)
        cv2.putText(frame, f"FPS: {fps_calc:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        if writer:
            writer.write(frame)

        cv2.imshow("Gaze Estimation", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
