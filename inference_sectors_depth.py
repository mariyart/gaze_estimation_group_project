import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
import argparse
import logging
import time
import warnings
import json

from config import data_config
from utils.helpers import get_model, draw_bbox_gaze
import uniface

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format='%(message)s')

# Gaze zones
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

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="resnet34")
    parser.add_argument("--weight", type=str, default="resnet34.pt")
    parser.add_argument("--dataset", type=str, default="gaze360")
    parser.add_argument("--view", action="store_true", default=True)
    parser.add_argument("--source", type=str, default="0")
    parser.add_argument("--output", type=str, default=None)
    return parser.parse_args()

def pre_process(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    transform = transforms.Compose([
        transforms.ToPILImage(), transforms.Resize(448), transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

def draw_grid(frame):
    h, w, _ = frame.shape
    for i in range(1, 6):
        cv2.line(frame, (int(i * w / 6), 0), (int(i * w / 6), h), (255, 255, 255), 1)
        cv2.line(frame, (0, int(i * h / 6)), (w, int(i * h / 6)), (255, 255, 255), 1)

def draw_gaze_arrow(frame, start, end, color=(255, 0, 255)):
    start = tuple(np.int32(start))
    end = tuple(np.int32(end))
    cv2.arrowedLine(frame, start, end, color, 2, tipLength=0.2)

def identify_gaze_zone(pitch, yaw):
    for zone, limits in gaze_zones.items():
        if limits["pitch"][0] <= pitch <= limits["pitch"][1] and limits["yaw"][0] <= yaw <= limits["yaw"][1]:
            return zone
    return "not_valid"

def main(params):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load gaze model
    model = get_model(params.model, data_config[params.dataset]["bins"], inference_mode=True)
    model.load_state_dict(torch.load(params.weight, map_location=device))
    model.to(device).eval()
    idx_tensor = torch.arange(data_config[params.dataset]["bins"], dtype=torch.float32, device=device)

    pitch_kf, yaw_kf = KalmanFilter1D(), KalmanFilter1D()
    face_detector = uniface.RetinaFace()

    # MediaPipe FaceMesh
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

    # Camera and iris constants
    CAMERA_MATRIX = np.array([[578.7566, 0.0, 349.099],
                              [0.0, 575.6480, 239.78],
                              [0.0, 0.0, 1.0]], dtype="double")
    IRIS_DIAMETER_MM = 11.7
    smoothed_depth = None
    alpha = 0.9

    cap = cv2.VideoCapture(int(params.source)) if params.source.isdigit() else cv2.VideoCapture(params.source)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_delay = 1 / fps if fps > 0 else 0.033
    out = None

    if params.output:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(params.output, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                lm = face_landmarks.landmark

                for eye_start, iris_ids, center_id in [(473, [474, 475, 476, 477], 473), (468, [469, 470, 471, 472], 468)]:
                    iris = np.array([(lm[i].x * w, lm[i].y * h) for i in iris_ids])
                    iris_center = np.mean(iris, axis=0)
                    eye_center = np.array((lm[center_id].x * w, lm[center_id].y * h))
                    draw_gaze_arrow(frame, eye_center, eye_center + 2 * (iris_center - eye_center))

                iris_diameter_pixels = np.linalg.norm(iris[0] - iris[2])
                if iris_diameter_pixels > 0:
                    focal_length_px = (CAMERA_MATRIX[0, 0] + CAMERA_MATRIX[1, 1]) / 2
                    depth_current = (focal_length_px * IRIS_DIAMETER_MM) / (iris_diameter_pixels * 1000)
                    smoothed_depth = depth_current if smoothed_depth is None else alpha * smoothed_depth + (1 - alpha) * depth_current
                    cv2.putText(frame, f"Depth: {smoothed_depth:.2f} m", (10, 160),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        bboxes, keypoints = face_detector.detect(frame)
        for bbox in bboxes:
            x_min, y_min, x_max, y_max = map(int, bbox[:4])
            image = frame[y_min:y_max, x_min:x_max]
            image = pre_process(image).to(device)

            pitch, yaw = model(image)
            pitch_deg = torch.sum(F.softmax(pitch, dim=1) * idx_tensor) * data_config[params.dataset]["binwidth"] - data_config[params.dataset]["angle"]
            yaw_deg = torch.sum(F.softmax(yaw, dim=1) * idx_tensor) * data_config[params.dataset]["binwidth"] - data_config[params.dataset]["angle"]
            pitch_sm = pitch_kf.update(pitch_deg.item())
            yaw_sm = yaw_kf.update(yaw_deg.item())
            zone = identify_gaze_zone(pitch_sm, yaw_sm)

            draw_bbox_gaze(frame, bbox, np.radians(pitch_sm), np.radians(yaw_sm))
            cv2.putText(frame, f"Pitch: {pitch_sm:.1f}° Yaw: {yaw_sm:.1f}° Zone: {zone}",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        draw_grid(frame)

        if params.view:
            cv2.imshow("Integrated Gaze & Depth Estimation", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

        if out:
            out.write(frame)

        time.sleep(frame_delay)

    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    args = parse_args()
    if args.dataset in data_config:
        cfg = data_config[args.dataset]
        args.bins = cfg["bins"]
        args.binwidth = cfg["binwidth"]
        args.angle = cfg["angle"]
    else:
        raise ValueError("Unknown dataset")
    main(args)
