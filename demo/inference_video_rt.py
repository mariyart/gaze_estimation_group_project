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
torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED
os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'


def average_output(out_dict, prev_dict):
    out_dict['gaze_out'] += prev_dict['gaze_out']
    out_dict['gaze_out'] /= np.linalg.norm(out_dict['gaze_out'])
    if out_dict['verts_eyes'] is not None:
        scale_l = np.linalg.norm(out_dict['verts_eyes']['left']) / np.linalg.norm(prev_dict['verts_eyes']['left'])
        scale_r = np.linalg.norm(out_dict['verts_eyes']['right']) / np.linalg.norm(prev_dict['verts_eyes']['right'])
        out_dict['verts_eyes']['left'] *= (1 + (scale_l - 1) / 2) / scale_l
        out_dict['verts_eyes']['right'] *= (1 + (scale_r - 1) / 2) / scale_r
        out_dict['verts_eyes']['left'][:, :2] += - out_dict['verts_eyes']['left'][out_dict['iris_idxs']][:, :2].mean(axis=0) + out_dict['centers_iris']['left']
        out_dict['verts_eyes']['right'][:, :2] += - out_dict['verts_eyes']['right'][out_dict['iris_idxs']][:, :2].mean(axis=0) + out_dict['centers_iris']['right']
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
            best_bbox = bboxes[best_idx]
            best_lms5 = lms5[best_idx]

            out_dict = predictor(img, best_lms5, undo_roll=True)

            if prev_dict is not None:
                out_dict = average_output(out_dict, prev_dict)

            if out_dict is not None and 'gaze_out' in out_dict:
                pitch, yaw = get_pitch_yaw_from_vector(out_dict['gaze_out'])
                out_dict['pitch'] = pitch
                out_dict['yaw'] = yaw

            if draw and out_dict is not None:
                out_img = draw_results(img, best_lms5, out_dict)
                text = f"Pitch: {out_dict['pitch']:.2f}°, Yaw: {out_dict['yaw']:.2f}°"
                cv2.putText(out_img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        else:
            print("⚠️ Skipping frame: bboxes shape invalid:", bboxes.shape)
    else:
        print("⚠️ No bboxes returned")

    return out_img, out_dict


def realtime_inference(cfg, draw=True, smooth=True):
    detector = FaceDetector(cfg.DETECTOR.THRESHOLD, cfg.DETECTOR.IMAGE_SIZE)
    predictor = GazePredictor(cfg.PREDICTOR, device=cfg.DEVICE)

    cap = cv2.VideoCapture(0)
    prev_dict = None

    frame_id = 0
    save_path = os.path.join("log", cfg.EXP_NAME, "gaze_angles.csv")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['frame_id', 'pitch', 'yaw'])

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            out_img, out_dict = infer_once(frame, detector, predictor, draw, prev_dict)

            if out_dict is not None and 'pitch' in out_dict and 'yaw' in out_dict:
                prev_dict = out_dict.copy() if smooth else None
                writer.writerow([frame_id, out_dict['pitch'], out_dict['yaw']])
            else:
                prev_dict = None

            if draw and out_img is not None:
                cv2.imshow("3DGazeNet Real-Time", out_img)
            else:
                cv2.imshow("3DGazeNet Real-Time", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            frame_id += 1

    cap.release()
    cv2.destroyAllWindows()


def parse_args():
    parser = argparse.ArgumentParser(description='Real-time Gaze Inference')
    parser.add_argument('--cfg', help='experiment config file', required=True, type=str)
    parser.add_argument('--gpu_id', help='id of the gpu to utilize', default=0, type=int)
    parser.add_argument('--no_draw', help='Disable drawing the results', action='store_true')
    parser.add_argument('--smooth_predictions', help='Average predictions between consecutive frames', action='store_true')
    args = parser.parse_args()
    update_config(args.cfg)
    return args


if __name__ == '__main__':
    args = parse_args()
    exp_save_path = f'log/{cfg.EXP_NAME}'
    logger = get_logger(exp_save_path, save=True, use_tqdm=False)
    Timer.save_path = exp_save_path

    with torch.no_grad():
        realtime_inference(cfg=cfg, draw=not args.no_draw, smooth=args.smooth_predictions)
