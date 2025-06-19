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


def inference_video(cfg, video_path, draw=True, smooth=True):
    detector = FaceDetector(cfg.DETECTOR.THRESHOLD, cfg.DETECTOR.IMAGE_SIZE)
    predictor = GazePredictor(cfg.PREDICTOR, device=cfg.DEVICE)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {video_path}")

    is_file = isinstance(video_path, str)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if is_file else 0
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    base_name = os.path.splitext(os.path.basename(video_path))[0] if is_file else "webcam"
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

    with open(csv_path, mode='w', newline='') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(['frame_id', 'pitch', 'yaw'])

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
                prev_dict = out_dict.copy() if smooth else None
                csv_writer.writerow([frame_id, pitch, yaw])
            else:
                pitch, yaw = 0.0, 0.0
                prev_dict = None
                csv_writer.writerow([frame_id, pitch, yaw])

            result_frame = out_img if draw and out_img is not None else frame
            if draw:
                text = f"Pitch: {pitch:.2f}°, Yaw: {yaw:.2f}°"
                cv2.putText(result_frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            writer.write(result_frame)

            if not is_file:
                cv2.imshow("3DGazeNet Real-Time", result_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            frame_id += 1

    cap.release()
    writer.release()
    if not is_file:
        cv2.destroyAllWindows()

    print(f"Video saved to: {save_video_path}")
    print(f"CSV saved to:   {csv_path}")


def parse_args():
    parser = argparse.ArgumentParser(description='3DGazeNet: Inference on video or webcam')
    parser.add_argument('--cfg', help='experiment config file', required=True, type=str)
    parser.add_argument('--video_path', help='Path to input video file. Leave empty for webcam.', type=str, default=None)
    parser.add_argument('--gpu_id', help='GPU id to use', default=0, type=int)
    parser.add_argument('--no_draw', help='Disable drawing the results', action='store_true')
    parser.add_argument('--smooth_predictions', help='Apply Kalman filter smoothing to 3D gaze vector', action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    update_config(args.cfg)
    exp_save_path = f'log/{cfg.EXP_NAME}'
    logger = get_logger(exp_save_path, save=True, use_tqdm=False)
    Timer.save_path = exp_save_path

    with torch.no_grad():
        video_input = args.video_path if args.video_path else 0
        inference_video(cfg=cfg, video_path=video_input, draw=not args.no_draw, smooth=args.smooth_predictions)
