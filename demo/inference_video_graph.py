import sys
import numpy as np
import cv2
import torch
import mediapipe as mp
from PyQt6.QtWidgets import QApplication, QWidget, QHBoxLayout, QLabel, QVBoxLayout
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QImage, QPixmap, QVector3D
import pyqtgraph.opengl as gl
import pyqtgraph as pg
import argparse

from utils import config as cfg, update_config, draw_results
from models import FaceDetectorIF as FaceDetector
from models import GazePredictorHandler as GazePredictor

update_config("demo/configs/infer_res18_x128_all_vfhq_vert.yaml")

cudnn = torch.backends.cudnn
cudnn.benchmark = cfg.CUDNN.BENCHMARK
cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
cudnn.enabled = cfg.CUDNN.ENABLED

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, refine_landmarks=True)

def get_pitch_yaw_from_vector(vector):
    x, y, z = vector
    yaw = np.degrees(np.arctan2(x, z))
    pitch = -np.degrees(np.arcsin(y))
    return pitch, yaw

def get_zone_color(pitch, yaw):
    zones = {
        'left': ((-60, 60), (10, 180), (1, 0, 0, 1)),
        'right': ((-60, 60), (-180, -10), (0, 1, 0, 1)),
        'up': ((-90, -10), (-60, 60), (0, 0, 1, 1)),
        'down': ((10, 90), (-60, 60), (1, 1, 0, 1)),
        'center': ((-10, 10), (-10, 10), (1, 0, 1, 1))
    }
    for _, (pr, yr, color) in zones.items():
        if pr[0] <= pitch <= pr[1] and yr[0] <= yaw <= yr[1]:
            return color
    return (0.5, 0.5, 0.5, 1)

def create_plane_item(width=1.0, height=1.0, color=(0.5, 0.5, 0.5, 0.2)):
    vertices = np.array([
        [0, 0, 0],
        [width, 0, 0],
        [width, height, 0],
        [0, height, 0],
    ])
    faces = np.array([
        [0, 1, 2],
        [0, 2, 3],
    ])
    md = gl.MeshData(vertexes=vertices, faces=faces)
    plane = gl.GLMeshItem(meshdata=md, color=color, smooth=False, shader='shaded', drawFaces=True)
    plane.setGLOptions('translucent')
    return plane, [vertices[face] for face in faces]

def ray_intersects_triangle(ray_origin, ray_vector, triangle):
    epsilon = 1e-6
    v0, v1, v2 = triangle
    edge1 = v1 - v0
    edge2 = v2 - v0
    h = np.cross(ray_vector, edge2)
    a = np.dot(edge1, h)
    if -epsilon < a < epsilon:
        return False, None
    f = 1.0 / a
    s = ray_origin - v0
    u = f * np.dot(s, h)
    if u < 0.0 or u > 1.0:
        return False, None
    q = np.cross(s, edge1)
    v = f * np.dot(ray_vector, q)
    if v < 0.0 or u + v > 1.0:
        return False, None
    t = f * np.dot(edge2, q)
    if t > epsilon:
        return True, t
    return False, None

def infer_once(img, detector, predictor, draw=True, prev_dict=None):
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
                out_dict['gaze_out'] += prev_dict['gaze_out']
                out_dict['gaze_out'] /= np.linalg.norm(out_dict['gaze_out'])

            if 'gaze_out' in out_dict:
                pitch, yaw = get_pitch_yaw_from_vector(out_dict['gaze_out'])
                out_dict['pitch'] = pitch
                out_dict['yaw'] = yaw

            if draw:
                img = draw_results(img, best_lms5, out_dict)

    return out_dict, img

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=str, default="0", help="Path to video file")
    args = parser.parse_args()

    # Allow numeric 0 for webcam
    video_source = int(args.video) if args.video.isdigit() else args.video
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"Could not open video: {args.video}")
        sys.exit(1)

    app = QApplication(sys.argv)
    window = QWidget()
    layout = QHBoxLayout()
    window.setLayout(layout)

    view = gl.GLViewWidget()
    view.setCameraPosition(pos=QVector3D(0, 0, 0), distance=2)
    axis = gl.GLAxisItem()
    axis.setSize(1, 1, 1)
    view.addItem(axis)
    layout.addWidget(view)

    vbox = QVBoxLayout()
    cam_label = QLabel()
    plane_label = QLabel("Plane hit: None")
    vbox.addWidget(cam_label)
    vbox.addWidget(plane_label)
    layout.addLayout(vbox)

    window.setGeometry(100, 100, 1280, 720)
    window.show()

    detector = FaceDetector(cfg.DETECTOR.THRESHOLD, cfg.DETECTOR.IMAGE_SIZE)
    predictor = GazePredictor(cfg.PREDICTOR, device=cfg.DEVICE)

    arrow_ref = [None]
    prev_dict = None

    planes = {}
    mesh_tris = {}
    plane_specs = {
        'front': ((0.5, 0.5), (1, 0, 0, 0.4)),
        'right': ((0.5, 0.5), (0, 1, 0, 0.4)),
        'left':  ((0.5, 0.5), (0.6, 0, 0.6, 0.4)),
        'top':   ((1.0, 1.0), (0, 0, 1, 0.4))
    }

    for name, ((w, h), color) in plane_specs.items():
        item, tris = create_plane_item(w, h, color)
        planes[name] = item
        mesh_tris[name] = tris
        view.addItem(item)

    def update():
        nonlocal prev_dict
        ret, frame = cap.read()
        if not ret:
            return

        out, frame_drawn = infer_once(frame, detector, predictor, draw=True, prev_dict=prev_dict)
        frame_rgb = cv2.cvtColor(frame_drawn, cv2.COLOR_BGR2RGB)
        h, w, ch = frame_rgb.shape
        qimg = QImage(frame_rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        cam_label.setPixmap(QPixmap.fromImage(qimg).scaled(640, 480, Qt.AspectRatioMode.KeepAspectRatio))

        results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if results.multi_face_landmarks:
            for face in results.multi_face_landmarks:
                lm = face.landmark
                if len(lm) >= 468:
                    left_eye = np.array([lm[133].x - 0.5, -(lm[133].y - 0.5), -lm[133].z])
                    right_eye = np.array([lm[362].x - 0.5, -(lm[362].y - 0.5), -lm[362].z])
                    origin = (left_eye + right_eye) / 2 * 2

                    planes['front'].resetTransform()
                    planes['front'].translate(origin[0] - 0.25, origin[1] - 0.25, origin[2] + 0.5)

                    planes['right'].resetTransform()
                    planes['right'].rotate(90, 0, 1, 0)
                    planes['right'].translate(origin[0] + 0.25, origin[1] - 0.25, origin[2] + 0.25)

                    planes['left'].resetTransform()
                    planes['left'].rotate(90, 0, 1, 0)
                    planes['left'].translate(origin[0] - 0.75, origin[1] - 0.25, origin[2] + 0.25)

                    planes['top'].resetTransform()
                    planes['top'].rotate(90, 1, 0, 0)
                    planes['top'].translate(origin[0] - 0.5, origin[1] + 0.5, origin[2] - 0.5)

                    if out and 'gaze_out' in out:
                        prev_dict = out.copy()
                        D = out['gaze_out'] / np.linalg.norm(out['gaze_out'])
                        pitch, yaw = out['pitch'], out['yaw']
                        color = get_zone_color(pitch, yaw)

                        closest_t, closest_name = float('inf'), None
                        for name, tris in mesh_tris.items():
                            for tri in tris:
                                tri = [planes[name].transform().map(QVector3D(*p)) for p in tri]
                                tri = [np.array([p.x(), p.y(), p.z()]) for p in tri]
                                hit, t = ray_intersects_triangle(origin, D, tri)
                                if hit and t < closest_t:
                                    closest_t = t
                                    closest_name = name

                        end_point = origin + D * 0.5
                        plane_label.setText(f"Plane hit: {closest_name if closest_name else 'None'}")

                        arrow_pos = np.array([origin, end_point])
                        if arrow_ref[0] is not None:
                            view.removeItem(arrow_ref[0])
                        arrow = gl.GLLinePlotItem(pos=arrow_pos, color=color, width=2, antialias=True)
                        view.addItem(arrow)
                        arrow_ref[0] = arrow

    timer = QTimer()
    timer.timeout.connect(update)
    timer.start(33)
    sys.exit(app.exec())

if __name__ == '__main__':
    with torch.no_grad():
        main()
