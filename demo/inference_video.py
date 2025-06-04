import tqdm
import torch
import torch.backends.cudnn as cudnn
import os
import argparse
import csv
import numpy as np

from utils import config as cfg, update_config, get_logger, Timer, VideoLoader, VideoSaver, show_result, draw_results
from models import FaceDetectorIF as FaceDetector
from models import GazePredictorHandler as GazePredictor

IMG_FORMATS = 'bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp'
VID_FORMATS = 'asf', 'avi', 'gif', 'm4v', 'mkv', 'mov', 'mp4', 'mpeg', 'mpg', 'ts', 'wmv'

cudnn.benchmark = cfg.CUDNN.BENCHMARK
torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED
os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'


def get_pitch_yaw_from_vector(vector):
    x, y, z = vector
    yaw = np.degrees(np.arctan2(x, z))
    pitch = -np.degrees(np.arcsin(y))
    return pitch, yaw


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
                import cv2
                text = f"Pitch: {pitch:.2f}°, Yaw: {yaw:.2f}°"
                cv2.putText(out_img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            print("⚠️ Skipping frame: bboxes shape invalid:", bboxes.shape)
    else:
        print("⚠️ No bboxes returned")

    return out_img, out_dict


def inference(cfg, video_path, draw, smooth):
    detector = FaceDetector(cfg.DETECTOR.THRESHOLD, cfg.DETECTOR.IMAGE_SIZE)
    predictor = GazePredictor(cfg.PREDICTOR, device=cfg.DEVICE)

    loader = VideoLoader(video_path, cfg.DETECTOR.IMAGE_SIZE, use_letterbox=False)
    save_dir = video_path[:video_path.rfind('.')] + f'_out_{cfg.PREDICTOR.BACKBONE_TYPE}_x{cfg.PREDICTOR.IMAGE_SIZE[0]}_{cfg.PREDICTOR.MODE}'
    saver = VideoSaver(output_dir=save_dir, fps=loader.fps, img_size=loader.vid_size, vid_size=640, save_images=False)
    tq = tqdm.tqdm(loader, file=logger)

    prev_dict = None
    pred_gaze_all = []

    csv_path = os.path.join(save_dir, 'gaze_angles.csv')
    os.makedirs(save_dir, exist_ok=True)
    csv_file = open(csv_path, mode='w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['frame_id', 'pitch', 'yaw'])

    for frame_idx, input in tq:
        if input is None:
            break

        out_img, out_dict = infer_once(input, detector, predictor, draw, prev_dict)

        if out_img is not None and out_dict is not None and 'gaze_out' in out_dict:
            prev_dict = out_dict.copy() if smooth else None
            pred_gaze_all += [out_dict['gaze_out']]
            csv_writer.writerow([frame_idx, out_dict['pitch'], out_dict['yaw']])
        else:
            prev_dict = None
            pred_gaze_all += [(0., 0., 0.)]
            csv_writer.writerow([frame_idx, 0.0, 0.0])

        description = '{fwd} {ft:.2f} | {det} {det_res:.2f} | {ep} {pred:.2f}'.format(
            fwd='Inference avg fps:',
            det='Detector avg fps:',
            ep='Eye predictor avg fps:',
            ft=Timer.metrics.avg('ForwardGazePredictor'),
            det_res=Timer.metrics.avg('Detector'),
            pred=Timer.metrics.avg('GazePredictor'))
        tq.set_description_str(description)

        if draw and out_img is not None:
            saver(out_img, frame_idx)

    csv_file.close()

    with open(f'{save_dir}/predicted_gaze_vectors.txt', 'w') as f:
        f.writelines([f"{', '.join(str(v) for v in p)}\n" for p in pred_gaze_all])


def parse_args():
    parser = argparse.ArgumentParser(description='Inference Gaze')
    parser.add_argument('--cfg', help='experiment configure file name', required=True, type=str)
    known_args, rest = parser.parse_known_args()
    update_config(known_args.cfg)
    parser.add_argument('--video_path', help='Video file to run', default="data/test_videos/ms_30s.mp4", type=str)
    parser.add_argument('--gpu_id', help='id of the gpu to utilize', default=0, type=int)
    parser.add_argument('--no_draw', help='Draw and save the results', action='store_true')
    parser.add_argument('--smooth_predictions', help='Average predictions between consecutive frames', action='store_true')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    exp_save_path = f'log/{cfg.EXP_NAME}'
    logger = get_logger(exp_save_path, save=True, use_tqdm=True)
    Timer.save_path = exp_save_path

    with torch.no_grad():
        inference(cfg=cfg, video_path=args.video_path, draw=not args.no_draw, smooth=args.smooth_predictions)
