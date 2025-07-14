# driver_monitor_with_3dgazenet.py
import cv2
import mediapipe as mp
import numpy as np
import time
import os
import torch
import torch.backends.cudnn as cudnn
import csv


# 3DGazeNet imports
from utils import config as cfg, update_config, draw_results
from models import FaceDetectorIF as FaceDetector
from models import GazePredictorHandler as GazePredictor
update_config("demo/configs/infer_res18_x128_all_vfhq_vert.yaml")

cudnn.benchmark = cfg.CUDNN.BENCHMARK
torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
cudnn.enabled = cfg.CUDNN.ENABLED
os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'

# 3DGazeNet init
detector = FaceDetector(cfg.DETECTOR.THRESHOLD, cfg.DETECTOR.IMAGE_SIZE)
predictor = GazePredictor(cfg.PREDICTOR, device=cfg.DEVICE)

gaze_data = []  # Stores [frame_idx, pitch, yaw]
frame_idx = 0

def add_gaze_graphics(frame):
    bboxes, lms5, _ = detector.run(frame)
    if not bboxes or not lms5:
        return frame
    best = np.argmax([b[3] - b[1] for b in bboxes])
    out = predictor(frame, lms5[best], undo_roll=True)
    if out is not None:
        overlay = frame.copy()
        draw_results(overlay, lms5[best], out)
        mask = cv2.cvtColor(overlay, cv2.COLOR_BGR2GRAY) != cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame[mask] = overlay[mask]
    return frame

# ---------------- MediaPipe & Kalman ----------------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

class Kalman1D:
    def __init__(self, q=1e-3, r=1e-2): self.x, self.P, self.Q, self.R = 0, 1, q, r
    def update(self, z):
        self.P += self.Q; K = self.P / (self.P + self.R)
        self.x += K * (z - self.x); self.P *= (1 - K); return self.x

kf_yaw = Kalman1D()
kf_depth = Kalman1D()

GAZE_ZONES = {
    "driver_side_window": {"pitch": (21.94, 25.94), "yaw": (45, 59)},
    "infotainment": {"pitch": (44, 48), "yaw": (-45, -40.27)},
    "normal_driving": {"pitch": (48.57, 51.57), "yaw": (0.51, 3.51)},
    "passenger_face": {"pitch": (-11, 34), "yaw": (-72, -43.19)},
    "passenger_footwell": {"pitch": (34.09, 38.09), "yaw": (-63.52, -60.52)},
    "passenger_side_window": {"pitch": (32.99, 45.99), "yaw": (-60.01, -52.01)},
    "phone_wheel": {"pitch": (32.26, 38.26), "yaw": (33.56, 40.56)},
    "rear_mirror": {"pitch": (53.57, 57.57), "yaw": (-45.81, -41.81)},
    "rear_passenger": {"pitch": (36.22, 40.22), "yaw": (-48.36, -44.36)}
}

MODEL_POINTS = np.array([
    (0.0, 0.0, 0.0),
    (-30.0, -30.0, -30.0),
    (30.0, -30.0, -30.0),
    (-24.0, 32.0, -30.0),
    (24.0, 32.0, -30.0)
], dtype=np.float32)
DIST_COEFFS = np.zeros((4, 1), dtype=np.float32)

def get_camera_matrix(w, h, f=None):
    f = 1.2 * max(w, h) if f is None else f
    return np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], dtype=np.float32)

def solve_head_pose(lms5, cam_mtx):
    img_points = np.array([lms5[2], lms5[0], lms5[1], lms5[3], lms5[4]], dtype=np.float32)
    ok, rvec, tvec = cv2.solvePnP(MODEL_POINTS, img_points, cam_mtx, DIST_COEFFS, flags=cv2.SOLVEPNP_EPNP)
    return (np.linalg.norm(tvec), rvec, tvec) if ok else (None, None, None)

def pitch_yaw_from_vec(vec):
    yaw = np.degrees(np.arctan2(vec[0], vec[2]))
    pitch = -np.degrees(np.arcsin(vec[1]))
    return pitch, yaw

def zone_from_pitch_yaw(p, y):
    for z, r in GAZE_ZONES.items():
        pmin, pmax = r["pitch"]; ymin, ymax = r["yaw"]
        if pmin <= p <= pmax and ymin <= y <= ymax:
            return z
    return "normal_driving"

# ---------------- Driver Monitoring Logic ----------------
EAR_THRESH = 0.12
PARTIAL_EAR_THRESH = 0.15
BLINK_QUICK_SPAN = 1.5
BLINK_QUICK_THRESHOLD = 3
YAWN_THRESH = 0.2
YAW_DISTRACT_DEG = 10
DISTRACT_DUR = 2.0

CALIB_DURATION = 1.0
KNOWN_DISTANCE_M = 0.55

def eye_aspect_ratio(pts):
    A = np.linalg.norm(pts[1] - pts[5])
    B = np.linalg.norm(pts[2] - pts[4])
    C = np.linalg.norm(pts[0] - pts[3])
    return (A + B) / (2.0 * C)

def mouth_aspect_ratio(pts):
    return np.linalg.norm(pts[0] - pts[1]) / np.linalg.norm(pts[2] - pts[3])

# ---------------- Video Setup ----------------
#cap = cv2.VideoCapture(r"P:\gaze-estimation-main\data\test_videos\in car\18_52_47MJPG-0009.avi")
cap = cv2.VideoCapture(0)
fps = cap.get(cv2.CAP_PROP_FPS) or 25
w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_delay = int(1000 / fps)
out = cv2.VideoWriter("output_with_annotations.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (720, 480))

calibrated, calib_start, ipd_list, focal_length = False, None, [], None
eye_closed, eye_closed_start, blink_times = False, None, []
yawn_start, yawn_times, distract_start = None, [], None
flash_toggle, last_flash_time = False, 0

# ---------------- Main Loop ----------------
while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)
    t = time.time()

    distraction_msg, alerts = "", []
    depth_display, show_iris = True, True

    if results.multi_face_landmarks:
        lm = results.multi_face_landmarks[0].landmark

        # HEAD POSE
        lms5 = np.array([
            [lm[33].x * w, lm[33].y * h],
            [lm[263].x * w, lm[263].y * h],
            [lm[1].x * w, lm[1].y * h],
            [lm[61].x * w, lm[61].y * h],
            [lm[291].x * w, lm[291].y * h]
        ], dtype=np.float32)

        cam_mtx = get_camera_matrix(w, h)
        dist_mm, rvec, tvec = solve_head_pose(lms5, cam_mtx)
        if rvec is not None:
            R, _ = cv2.Rodrigues(rvec)
            vec = R[:, 2]
            yaw_sm = kf_yaw.update(vec[0])
            pitch, yaw = pitch_yaw_from_vec(vec)
            if rvec is not None:
                R, _ = cv2.Rodrigues(rvec)
                vec = R[:, 2]
                yaw_sm = kf_yaw.update(vec[0])
                pitch, yaw = pitch_yaw_from_vec(vec)
                zone = zone_from_pitch_yaw(pitch, yaw)
            else:
                pitch = yaw = 0.0
                zone = "normal_driving"
            gaze_data.append([frame_idx, pitch, yaw, zone])
            frame_idx += 1
            zone = zone_from_pitch_yaw(pitch, yaw)
        else:
            pitch = yaw = 0.0
            zone = "normal_driving"

        yaw_deg = yaw
        if abs(yaw_deg) > YAW_DISTRACT_DEG:
            if distract_start is None: distract_start = t
            elif t - distract_start >= DISTRACT_DUR:
                distraction_msg = "DISTRACTED"
                depth_display = show_iris = False
        else:
            distract_start = None

        # Eyes & blink
        re = np.array([(lm[i].x*w, lm[i].y*h) for i in [33,160,158,133,153,144]])
        le = np.array([(lm[i].x*w, lm[i].y*h) for i in [362,385,387,263,373,380]])
        ear = (eye_aspect_ratio(re) + eye_aspect_ratio(le)) / 2.0
        if ear < PARTIAL_EAR_THRESH and not eye_closed:
            eye_closed, eye_closed_start, blink_start = True, t, t
        elif ear >= EAR_THRESH and eye_closed:
            blink_dur = t - blink_start
            if blink_dur < 0.5: blink_times.append(t)
            eye_closed = False

        blink_times = [bt for bt in blink_times if t - bt <= BLINK_QUICK_SPAN]
        if len(blink_times) >= BLINK_QUICK_THRESHOLD:
            alerts.append("FREQUENT BLINKING - DROWSINESS ALERT")
        elif eye_closed and eye_closed_start:
            cd = t - eye_closed_start
            if 2 <= cd < 3: alerts.append("MICROSLEEP DETECTED"); depth_display=False
            elif 3 <= cd < 6: alerts.append("SLEEP DETECTED"); depth_display=False
            elif cd >= 6:
                alerts.append("UNRESPONSIVE DRIVER"); depth_display=False
                cv2.putText(frame, "WARNING! DRIVER UNRESPONSIVE!", (50, h//2),
                            cv2.FONT_HERSHEY_DUPLEX, 1.0, (255,255,255), 3)

        # Yawn
        mp_pts = np.array([(lm[i].x*w, lm[i].y*h) for i in [13,14,78,308]])
        mar = mouth_aspect_ratio(mp_pts)
        if mar > YAWN_THRESH:
            if yawn_start is None: yawn_start = t
            elif t - yawn_start >= 2.0:
                if not yawn_times or t - yawn_times[-1] > 2.0:
                    yawn_times.append(t)
                alerts.append("YAWNING DETECTED")
        else:
            yawn_start = None
        yawn_times = [yt for yt in yawn_times if t - yt <= 60]
        if len(yawn_times) >= 2 and (yawn_times[-1] - yawn_times[-2]) < 30 and mar > YAWN_THRESH:
            alerts.append("FREQUENT YAWNING - DROWSINESS")

        # Depth via IPD
        if depth_display:
            il = np.mean([(lm[i].x*w, lm[i].y*h) for i in [474,475,476,477]], axis=0)
            ir = np.mean([(lm[i].x*w, lm[i].y*h) for i in [469,470,471,472]], axis=0)
            ipd = abs(il[0] - ir[0])
            if not calibrated:
                if calib_start is None: calib_start = t
                ipd_list.append(ipd)
                elapsed = t - calib_start
                cv2.putText(frame, f"Calibrating... {elapsed:.1f}s", (10,60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255),2)
                if elapsed >= CALIB_DURATION:
                    focal_length = np.mean(ipd_list) * KNOWN_DISTANCE_M
                    calibrated = True
            else:
                corr = np.cos(yaw_sm)
                ipd_corr = ipd / corr if abs(yaw_sm) < np.radians(30) else ipd
                if focal_length and ipd_corr > 0:
                    depth_m = focal_length / ipd_corr
                    d = kf_depth.update(depth_m)
                    cv2.putText(frame, f"Depth: {d:.2f} m", (10,90),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0),2)
            if show_iris:
                cv2.circle(frame, tuple(il.astype(int)),2,(0,255,0),-1)
                cv2.circle(frame, tuple(ir.astype(int)),2,(0,255,0),-1)

        # Overlay info
        critical_states = {"SLEEP DETECTED", "UNRESPONSIVE DRIVER", "MICROSLEEP DETECTED"}

        # Pitch-yaw read-out
        cv2.putText(frame, f"Pitch:{pitch:.1f}°  Yaw:{yaw:.1f}°",
                    (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 0), 2)

        # Zone text only if NOT in a critical state
        if not any(msg in critical_states for msg in alerts):
            cv2.putText(frame, f"Zone: {zone}", (10, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        # Distraction status
        if distraction_msg:
            cv2.putText(frame, distraction_msg, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # Show the first alert message (if any)
        for msg in alerts:
            cv2.putText(frame, msg, (10, 180),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            break

    # Draw arrow + eye from 3DGazeNet after all logic
    frame = add_gaze_graphics(frame)

    r = cv2.resize(frame, (720,480))
    out.write(r)
    cv2.imshow("Driver Monitoring", r)
    if cv2.waitKey(frame_delay) & 0xFF == 27: break

with open("gaze_mediapipe_output.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["frame", "pitch", "yaw", "zone"])
    writer.writerows(gaze_data)

cap.release()
out.release()
cv2.destroyAllWindows()
