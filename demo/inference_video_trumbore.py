import sys
import numpy as np
import cv2
import torch
import mediapipe as mp
from PyQt6.QtWidgets import QApplication, QWidget, QHBoxLayout, QLabel, QVBoxLayout
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QImage, QPixmap, QVector3D, QMatrix4x4
import pyqtgraph.opengl as gl
import pyqtgraph as pg
import argparse
from PIL import Image

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

def create_plane(width=2.0, height=2.0):
    vertices = np.array([
        [-width/2, -height/2, 0],
        [ width/2, -height/2, 0],
        [ width/2,  height/2, 0],
        [-width/2,  height/2, 0],
    ])
    faces = np.array([
        [0, 1, 2],
        [0, 2, 3],
    ])
    return vertices, faces

def apply_transform(vertices, matrix):
    return np.array([[vec.x(), vec.y(), vec.z()] for vec in [matrix.map(QVector3D(*v)) for v in vertices]])

def ray_intersects_triangle(ray_origin, ray_vector, triangle):
    epsilon = 1e-6
    v0, v1, v2 = triangle
    edge1 = v1 - v0
    edge2 = v2 - v0
    h = np.cross(ray_vector, edge2)
    a = np.dot(edge1, h)
    if abs(a) < epsilon:
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
    return (t > epsilon), t

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
    parser.add_argument('--video', type=str, default=0)
    args = parser.parse_args()

    app = QApplication(sys.argv)
    window = QWidget()
    layout = QHBoxLayout(window)
    view = gl.GLViewWidget()
    view.setCameraPosition(pos=np.array([0, 0, 1]), distance=2)
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

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Could not open video: {args.video}")
        sys.exit(1)

    detector = FaceDetector(cfg.DETECTOR.THRESHOLD, cfg.DETECTOR.IMAGE_SIZE)
    predictor = GazePredictor(cfg.PREDICTOR, device=cfg.DEVICE)

    arrow_ref = [None]
    arrow_head_ref = [None]
    prev_dict = None

    specs = {
        'front':  (0,   QVector3D(0, 0, 1), QVector3D(0, 0, 1.5), (1, 0, 0, 0.4)),
        'right':  (90,  QVector3D(0, 1, 0), QVector3D(1.5, 0, 0),  (0, 1, 0, 0.4)),
        'left':  (-90,  QVector3D(0, 1, 0), QVector3D(-1.5, 0, 0), (1, 0, 1, 0.4)),
        'top':    (90,  QVector3D(1, 0, 0), QVector3D(0, 1.5, 0),  (0, 0, 1, 0.4)),
        'bottom': (-90, QVector3D(1, 0, 0), QVector3D(0, -1.5, 0), (1, 1, 0, 0.4))
    }

    image_files = {
        'front': r"C:\Users\maria\Desktop\front.jpg",
    }

    base_vertices, base_faces = create_plane()
    plane_tris = {}

    loaded_images = {
        name: (np.array(Image.open(path).convert("RGBA")), Image.open(path).size)
        for name, path in image_files.items()
    }

    for name, (angle, axis, translation, color) in specs.items():
        matrix = QMatrix4x4()
        matrix.translate(translation)
        matrix.rotate(angle, axis)

        transformed_vertices = apply_transform(base_vertices, matrix)
        plane_tris[name] = [
            [transformed_vertices[idx] for idx in face] for face in base_faces
        ]

        # Proper image alignment
        if name in image_files:
            # Load and convert image
            img = Image.open(image_files[name]).convert("RGBA")
            img_np = np.array(img)
            img_item = gl.GLImageItem(img_np)

            # Scale image to exactly 2x2 world units (independent of resolution)
            h, w = img_np.shape[:2]
            image_transform = QMatrix4x4()
            image_transform.translate(-1.0, -1.0, 0)  # Center image at (0,0)
            image_transform.scale(2.0 / w, 2.0 / h, 1.0)  # Normalize pixels to 2x2

            # Orientation and position
            placement = QMatrix4x4()

            if name == 'front':
                placement.translate(0, 0, 1.5)  # move to front
                placement.rotate(180, QVector3D(0, 1, 0))  # flip horizontally (Y-axis)
                placement.rotate(180, QVector3D(0, 0, 1))  # flip vertically (Z-axis)
                placement.rotate(-90, QVector3D(0, 0, 1))  # rotate 90 degrees clockwise around Z-axis
            elif name == 'right':  # goes on LEFT wall (x = -1.5)
                placement.rotate(90, QVector3D(0, 1, 0))
                placement.translate(-1.5, 0, 0)
            elif name == 'left':  # goes on RIGHT wall (x = +1.5)
                placement.rotate(-90, QVector3D(0, 1, 0))
                placement.translate(1.5, 0, 0)

            final_transform = placement * image_transform
            img_item.setTransform(final_transform)
            view.addItem(img_item)

        else:
            md = gl.MeshData(vertexes=transformed_vertices, faces=base_faces)
            plane_item = gl.GLMeshItem(meshdata=md, color=color, smooth=False, shader='shaded', drawFaces=True)
            plane_item.setGLOptions('translucent')
            view.addItem(plane_item)

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
                    view.opts['center'] = pg.Vector(origin[0], origin[1], origin[2])

                    if out and 'gaze_out' in out:
                        prev_dict = out.copy()
                        D = out['gaze_out'] / np.linalg.norm(out['gaze_out'])
                        pitch, yaw = out['pitch'], out['yaw']
                        color = get_zone_color(pitch, yaw)

                        min_t, hit_name = float('inf'), None
                        for name, tris in plane_tris.items():
                            for tri in tris:
                                hit, t = ray_intersects_triangle(origin, D, tri)
                                if hit and t < min_t:
                                    min_t = t
                                    hit_name = name

                        plane_label.setText(f"Plane hit: {hit_name if hit_name else 'None'}")
                        if min_t != float('inf'):
                            arrow_length = max(min_t - 0.1, 0.1)  # keep a small offset from the plane
                        else:
                            arrow_length = 2.0  # shorter fallback length

                        end_point = origin + D * arrow_length
                        arrow_pos = np.array([origin, end_point])
                        if arrow_ref[0] is not None:
                            view.removeItem(arrow_ref[0])
                        arrow = gl.GLLinePlotItem(pos=arrow_pos, color=color, width=2, antialias=True)
                        view.addItem(arrow)
                        arrow_ref[0] = arrow

                        if arrow_head_ref[0] is not None:
                            view.removeItem(arrow_head_ref[0])

                        cone_height = 0.2
                        cone_radius = 0.05
                        meshdata = gl.MeshData.cylinder(rows=10, cols=20, radius=[cone_radius, 0], length=cone_height)

                        up = np.array([0, 0, 1])
                        axis = np.cross(up, D)
                        angle = np.degrees(np.arccos(np.clip(np.dot(up, D), -1.0, 1.0)))
                        transform = QMatrix4x4()
                        transform.translate(QVector3D(*end_point))
                        if np.linalg.norm(axis) > 1e-6:
                            transform.rotate(angle, QVector3D(*axis))

                        vertices = meshdata.vertexes()
                        vertices = apply_transform(vertices, transform)
                        cone_mesh = gl.MeshData(vertexes=vertices, faces=meshdata.faces())
                        arrow_head = gl.GLMeshItem(meshdata=cone_mesh, color=color, smooth=True, shader='shaded', drawFaces=True)
                        arrow_head.setGLOptions('translucent')
                        view.addItem(arrow_head)
                        arrow_head_ref[0] = arrow_head

    timer = QTimer()
    timer.timeout.connect(update)
    timer.start(33)
    sys.exit(app.exec())

if __name__ == '__main__':
    with torch.no_grad():
        main()
