import cv2
import mediapipe as mp
import numpy as np
import time

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

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

CAMERA_MATRIX = np.array([[578.7566, 0.0, 349.099],
                          [0.0, 575.6480, 239.78],
                          [0.0, 0.0, 1.0]], dtype="double")
IRIS_DIAMETER_MM = 11.7
alpha = 0.9
smoothed_depth = None

# Kalman filters for gaze smoothing
gaze_kf_x_left = KalmanFilter1D()
gaze_kf_y_left = KalmanFilter1D()
gaze_kf_x_right = KalmanFilter1D()
gaze_kf_y_right = KalmanFilter1D()

cap = cv2.VideoCapture(0)
fps = cap.get(cv2.CAP_PROP_FPS)
frame_delay = 1 / fps if fps > 0 else 0.033

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

            # Geometric head pose estimation
            nose_tip = np.array([lm[1].x * w, lm[1].y * h])
            left_eye = np.array([lm[33].x * w, lm[33].y * h])
            right_eye = np.array([lm[263].x * w, lm[263].y * h])
            eye_distance = np.linalg.norm(left_eye - right_eye)
            imbalance = nose_tip[0] - (left_eye[0] + right_eye[0]) / 2
            yaw_geom = np.arctan(imbalance / eye_distance)
            pitch_geom = np.arctan((nose_tip[1] - left_eye[1]) / eye_distance)

            cv2.putText(frame, f"Pitch: {np.degrees(pitch_geom):.1f}°", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Yaw: {np.degrees(yaw_geom):.1f}°", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Iris depth and smoothed gaze estimation
            for iris_ids, center_id, kf_x, kf_y in [([474, 475, 476, 477], 473, gaze_kf_x_right, gaze_kf_y_right),
                                                    ([469, 470, 471, 472], 468, gaze_kf_x_left, gaze_kf_y_left)]:
                iris = np.array([(lm[i].x * w, lm[i].y * h) for i in iris_ids])
                eye_center = np.array((lm[center_id].x * w, lm[center_id].y * h))
                iris_center = np.mean(iris, axis=0)

                # Kalman filter smoothing
                gaze_x = kf_x.update(iris_center[0])
                gaze_y = kf_y.update(iris_center[1])
                smoothed_gaze = np.array([gaze_x, gaze_y])

                # Draw arrow
                end_point = eye_center + 2 * (smoothed_gaze - eye_center)
                cv2.arrowedLine(frame, tuple(np.int32(eye_center)), tuple(np.int32(end_point)), (255, 0, 255), 2, tipLength=0.2)

                # Depth estimation
                iris_diameter_pixels = np.linalg.norm(iris[0] - iris[2])
                if iris_diameter_pixels > 0:
                    focal_length_px = (CAMERA_MATRIX[0, 0] + CAMERA_MATRIX[1, 1]) / 2
                    depth_current = (focal_length_px * IRIS_DIAMETER_MM) / (iris_diameter_pixels * 1000)
                    smoothed_depth = depth_current if smoothed_depth is None else alpha * smoothed_depth + (1 - alpha) * depth_current
                    cv2.putText(frame, f"Depth: {smoothed_depth:.2f} m", (10, 90),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    cv2.imshow("Jetson Gaze and Pose Estimation", frame)
    if cv2.waitKey(int(frame_delay * 1000)) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
