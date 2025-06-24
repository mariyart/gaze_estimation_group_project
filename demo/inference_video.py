import tqdm
import torch
import torch.backends.cudnn as cudnn
import os
import argparse
import cv2
import numpy as np
import csv

from utils import config as cfg, update_config, get_logger, Timer, draw_results
from models import FaceDetectorIF as FaceDetector
from models import GazePredictorHandler as GazePredictor

cudnn.benchmark = cfg.CUDNN.BENCHMARK
torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
cudnn.enabled = cfg.CUDNN.ENABLED
os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'


# ---------------------- GAZE ZONE MAPPING ----------------------
GAZE_ZONES = {
    "driver_side_window": {
        "pitch": (-23.84, -23.2),
        "yaw": (7.11, 10.31)
    },
    "phone_wheel": {
        "pitch": (-22.7, -22.02),
        "yaw": (7.7, 9.41)
    },
    "infotainment": {
        "pitch": (-22.9, -22.4),
        "yaw": (6.42, 7.0)
    },
    "passenger_face": {
        "pitch": (-22.68, -22.54),
        "yaw": (5.85, 7.53)
    },
    "passenger_side_window": {
        "pitch": (-22.41, -22.3),
        "yaw": (7.72, 8.13)
    },
    "rear_mirror": {
        "pitch": (-18, -14),
        "yaw": (6, 18)
    },
    "rear_passenger": {
        "pitch": (-23.52, -23.41),
        "yaw": (11.02, 15.83)
    },
    "passenger_footwell": {
        "pitch": (-18.92, -12.26),
        "yaw": (21.57, 40.52)
    }
}

def get_zone_from_pitch_yaw(pitch, yaw):
    for zone, ranges in GAZE_ZONES.items():
        pmin, pmax = ranges["pitch"]
        ymin, ymax = ranges["yaw"]
        if pmin <= pitch <= pmax and ymin <= yaw <= ymax:
            return zone
    return "normal_driving"


# ---------------------- KALMAN FILTER ----------------------
class KalmanFilter1D:
    def __init__(self, process_variance=1e-6, measurement_variance=5e-1):
        self.x, self.P = 0.0, 1.0
        self.Q = process_variance
        self.R = measurement_variance

    def update(self, measurement):
        self.P += self.Q
        K = self.P / (self.P + self.R)
        self.x += K * (measurement - self.x)
        self.P *= (1 - K)
        return self.x

class KalmanFilter3D:
    def __init__(self, process_variance=1e-8, measurement_variance=1.0):
        self.filters = [KalmanFilter1D(process_variance, measurement_variance) for _ in range(3)]
        self.last_valid_vector = np.array([0.0, 0.0, 1.0])

    def update(self, vector):
        norm = np.linalg.norm(vector)
        if not 0.7 < norm < 1.3 or np.any(np.isnan(vector)):
            return self.last_valid_vector

        smoothed = np.array([f.update(v) for f, v in zip(self.filters, vector)])
        norm_s = np.linalg.norm(smoothed)
        if norm_s == 0 or np.isnan(norm_s):
            return self.last_valid_vector

        smoothed /= norm_s
        self.last_valid_vector = smoothed
        return smoothed


# ---------------------- INFERENCE HELPERS ----------------------
def average_output(out_dict, prev_dict):
    out_dict['gaze_out'] += prev_dict['gaze_out']
    out_dict['gaze_out'] /= np.linalg.norm(out_dict['gaze_out'])
    return out_dict

def get_pitch_yaw_from_vector(vector):
    x, y, z = vector
    yaw = np.degrees(np.arctan2(x, z))
    pitch = -np.degrees(np.arcsin(y))
    return pitch, yaw

@Timer(name='Forward', fps=True, pprint=False)
def infer_once(img, detector, predictor, draw, prev_dict=None):
    out_img = None
    out_dict = None

    bboxes, lms5, _ = detector.run(img)

    if isinstance(bboxes, (list, np.ndarray)) and len(bboxes) > 0:
        bboxes = np.array(bboxes)
        lms5 = np.array(lms5)

        if bboxes.ndim == 1 and bboxes.shape[0] == 4:
            bboxes = bboxes.reshape(1, 4)
            lms5 = lms5.reshape(1, 5, 2)

        if bboxes.ndim == 2 and bboxes.shape[1] >= 4:
            idxs_sorted = sorted(range(len(bboxes)), key=lambda k: bboxes[k][3] - bboxes[k][1])
            best_idx = idxs_sorted[-1]
            best_lms5 = lms5[best_idx]

            out_dict = predictor(img, best_lms5, undo_roll=True)

            if prev_dict is not None:
                out_dict = average_output(out_dict, prev_dict)

            if draw and out_dict is not None:
                out_img = draw_results(img, best_lms5, out_dict)

        else:
            print("Skipping frame: bboxes shape invalid", bboxes.shape)
    else:
        print("No bboxes returned")

    return out_img, out_dict


# ---------------------- VIDEO INFERENCE ----------------------
def inference_video(cfg, video_path, draw=True, smooth=True):
    detector = FaceDetector(cfg.DETECTOR.THRESHOLD, cfg.DETECTOR.IMAGE_SIZE)
    predictor = GazePredictor(cfg.PREDICTOR, device=cfg.DEVICE)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {video_path}")

    is_file = isinstance(video_path, str) and os.path.isfile(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if is_file else 0
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # Output paths only for file input
    if is_file:
        base_name = os.path.splitext(os.path.basename(video_path))[0]
    else:
        base_name = "webcam"

    base_dir = os.path.join("log", cfg.EXP_NAME)
    run_name = f"{base_name}_out_{cfg.PREDICTOR.BACKBONE_TYPE}"
    save_dir = os.path.join(base_dir, run_name)

    counter = 1
    while os.path.exists(save_dir):
        save_dir = os.path.join(base_dir, f"{run_name}_{counter}")
        counter += 1

    os.makedirs(save_dir, exist_ok=True)
    save_video_path = os.path.join(save_dir, 'output.mp4')
    csv_path = os.path.join(save_dir, 'gaze_angles.csv')

    writer = cv2.VideoWriter(save_video_path, fourcc, fps, (width, height))

    if is_file:
        csv_file = open(csv_path, mode='w', newline='')
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['frame_id', 'pitch', 'yaw', 'zone'])
    else:
        csv_file = None
        csv_writer = None

    frame_id = 0
    prev_dict = None
    gaze_filter = KalmanFilter3D()

    frame_iter = tqdm.trange(total_frames, desc="Processing Video") if is_file else iter(int, 1)

    for _ in frame_iter:
        ret, frame = cap.read()
        if not ret:
            break

        out_img, out_dict = infer_once(frame, detector, predictor, draw, prev_dict)

        if out_dict and 'gaze_out' in out_dict:
            gaze_vector = out_dict['gaze_out']
            if smooth:
                gaze_vector = gaze_filter.update(gaze_vector)
            pitch, yaw = get_pitch_yaw_from_vector(gaze_vector)
            zone = get_zone_from_pitch_yaw(pitch, yaw)
            prev_dict = out_dict.copy() if smooth else None
        else:
            pitch, yaw = 0.0, 0.0
            zone = "normal_driving"
            prev_dict = None

        if csv_writer:
            csv_writer.writerow([frame_id, pitch, yaw, zone])

        result_frame = out_img if draw and out_img is not None else frame
        if draw:
            text = f"Pitch: {pitch:.2f}°, Yaw: {yaw:.2f}°, Zone: {zone}"
            cv2.putText(result_frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 0), 2)

        writer.write(result_frame)

        if draw:
            cv2.imshow("3DGazeNet Output", result_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        frame_id += 1

    cap.release()
    writer.release()
    if csv_file:
        csv_file.close()
    if draw:
        cv2.destroyAllWindows()

    if is_file:
        print(f"Video saved to: {save_video_path}")
        print(f"CSV saved to:   {csv_path}")

# ---------------------- ARGUMENT PARSING ----------------------
def parse_args():
    parser = argparse.ArgumentParser(description='3DGazeNet: Inference on video or webcam')
    parser.add_argument('--cfg', help='experiment config file', required=True, type=str)
    parser.add_argument('--video_path', help='Path to input video file. Leave empty for webcam.', type=str, default=None)
    parser.add_argument('--gpu_id', help='GPU id to use', default=0, type=int)
    parser.add_argument('--no_draw', help='Disable drawing the results', action='store_true')
    parser.add_argument('--smooth_predictions', help='Apply Kalman filter smoothing to 3D gaze vector', action='store_true')
    return parser.parse_args()


# ---------------------- ENTRY POINT ----------------------
if __name__ == '__main__':
    args = parse_args()
    update_config(args.cfg)
    exp_save_path = f'log/{cfg.EXP_NAME}'
    logger = get_logger(exp_save_path, save=True, use_tqdm=False)
    Timer.save_path = exp_save_path

    with torch.no_grad():
        video_input = args.video_path if args.video_path else 0
        inference_video(cfg=cfg, video_path=video_input, draw=not args.no_draw, smooth=args.smooth_predictions)
