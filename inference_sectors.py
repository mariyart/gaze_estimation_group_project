import cv2
import logging
import argparse
import warnings
import numpy as np
import json

import torch
import torch.nn.functional as F
from torchvision import transforms

from config import data_config
from utils.helpers import get_model, draw_bbox_gaze

import uniface

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format='%(message)s')

# Define the gaze zones with their respective pitch and yaw ranges
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
        self.x = 0.0  # initial state
        self.P = 1.0  # initial covariance
        self.Q = process_variance  # process noise
        self.R = measurement_variance  # measurement noise

    def update(self, measurement):
        self.P += self.Q
        K = self.P / (self.P + self.R)
        self.x += K * (measurement - self.x)
        self.P *= (1 - K)
        return self.x

def parse_args():
    parser = argparse.ArgumentParser(description="Gaze estimation inference")
    parser.add_argument("--model", type=str, default="resnet34", help="Model name, default `resnet18`")
    parser.add_argument(
        "--weight",
        type=str,
        default="resnet34.pt",
        help="Path to gaze estimation model weights"
    )
    parser.add_argument("--view", action="store_true", default=True, help="Display the inference results")
    parser.add_argument("--source", type=str, default="0",  # Default to webcam
                        help="Path to source video file or camera index")
    parser.add_argument("--output", type=str, default="output.mp4", help="Path to save output file")
    parser.add_argument("--dataset", type=str, default="gaze360", help="Dataset name to get dataset related configs")
    args = parser.parse_args()

    if args.dataset in data_config:
        dataset_config = data_config[args.dataset]
        args.bins = dataset_config["bins"]
        args.binwidth = dataset_config["binwidth"]
        args.angle = dataset_config["angle"]
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}. Available options: {list(data_config.keys())}")

    return args

def pre_process(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    transform = transforms.Compose([ 
        transforms.ToPILImage(), 
        transforms.Resize(448),
        transforms.ToTensor(), 
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
    ]) 
    image = transform(image) 
    image_batch = image.unsqueeze(0)
    return image_batch

def draw_grid(frame):
    # Drawing a simple grid overlay
    height, width, _ = frame.shape
    grid_size = 6  # Number of rows and columns

    # Draw vertical lines dynamically based on frame width
    for i in range(1, grid_size):
        x = int(i * width / grid_size)
        cv2.line(frame, (x, 0), (x, height), (255, 255, 255), 1)

    # Draw horizontal lines dynamically based on frame height
    for i in range(1, grid_size):
        y = int(i * height / grid_size)
        cv2.line(frame, (0, y), (width, y), (255, 255, 255), 1)

def identify_gaze_zone(pitch_deg, yaw_deg):
    for zone, ranges in gaze_zones.items():
        pitch_range = ranges["pitch"]
        yaw_range = ranges["yaw"]

        if pitch_range[0] <= pitch_deg <= pitch_range[1] and yaw_range[0] <= yaw_deg <= yaw_range[1]:
            return zone
    return "not_valid"  # Default to "not_valid" if no match is found

def main(params):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(torch.cuda.is_available())
    print(device)

    idx_tensor = torch.arange(params.bins, device=device, dtype=torch.float32)

    face_detector = uniface.RetinaFace()

    try:
        gaze_detector = get_model(params.model, params.bins, inference_mode=True)
        state_dict = torch.load(params.weight, map_location=device)
        gaze_detector.load_state_dict(state_dict)
        logging.info("Gaze Estimation model weights loaded.")
    except Exception as e:
        logging.info(f"Exception occurred while loading pre-trained weights of gaze estimation model. Exception: {e}")

    gaze_detector.to(device)
    gaze_detector.eval()

    pitch_kalman = KalmanFilter1D()
    yaw_kalman = KalmanFilter1D()

    video_source = params.source
    if video_source.isdigit() or video_source == '0':
        cap = cv2.VideoCapture(int(video_source))  # Open webcam if source is '0'
    else:
        cap = cv2.VideoCapture(video_source)

    if params.output:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(params.output, fourcc, cap.get(cv2.CAP_PROP_FPS), (width, height))

    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    gaze_data = []  # List to store pitch, yaw, and frame data

    with torch.no_grad():
        while True:
            success, frame = cap.read()
            frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))  # Get current frame number

            if not success:
                logging.info("Failed to obtain frame or EOF")
                break

            bboxes, keypoints = face_detector.detect(frame)
            for bbox, keypoint in zip(bboxes, keypoints):
                x_min, y_min, x_max, y_max = map(int, bbox[:4])

                image = frame[y_min:y_max, x_min:x_max]
                image = pre_process(image).to(device)

                pitch, yaw = gaze_detector(image)

                pitch_soft = F.softmax(pitch, dim=1)
                yaw_soft = F.softmax(yaw, dim=1)

                pitch_deg = torch.sum(pitch_soft * idx_tensor, dim=1).item() * params.binwidth - params.angle
                yaw_deg = torch.sum(yaw_soft * idx_tensor, dim=1).item() * params.binwidth - params.angle

                pitch_deg_smoothed = pitch_kalman.update(pitch_deg)
                yaw_deg_smoothed = yaw_kalman.update(yaw_deg)

                pitch_predicted = np.radians(pitch_deg_smoothed)
                yaw_predicted = np.radians(yaw_deg_smoothed)

                # Identify the gaze zone
                gaze_zone = identify_gaze_zone(pitch_deg_smoothed, yaw_deg_smoothed)

                draw_bbox_gaze(frame, bbox, pitch_predicted, yaw_predicted)

                # Add grid to the frame
                draw_grid(frame)

                # Print action, gaze direction, and zone for current frame
                gaze_text = f"Pitch: {pitch_deg_smoothed:.2f}° | Yaw: {yaw_deg_smoothed:.2f}° | Zone: {gaze_zone}"
                logging.info(f"Frame {frame_number}: {gaze_text}")
                
                # Reduce text size by changing font scale to 0.7 (smaller text)
                cv2.putText(frame, gaze_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            if params.output:
                out.write(frame)

            if params.view:
                cv2.imshow('Demo', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    cap.release()
    if params.output:
        out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    args = parse_args()

    if not args.view and not args.output:
        raise Exception("At least one of --view or --output must be provided.")

    main(args)
