import sys
import numpy as np
import math
from PyQt6.QtWidgets import QApplication
import pyqtgraph.opengl as gl

# === Load gaze vectors ===
def load_gaze_vectors(filepath):
    return np.loadtxt(filepath, delimiter=",")

# === Convert to pitch/yaw (degrees) ===
def vector_to_pitch_yaw(vec):
    x, y, z = vec
    norm = math.sqrt(x*x + y*y + z*z)
    x /= norm
    y /= norm
    z /= norm
    pitch = math.degrees(math.atan2(-y, math.sqrt(x**2 + z**2)))
    yaw = math.degrees(math.atan2(-x, -z))
    return pitch, yaw

# === Assign color based on zone ===
def get_zone_color(pitch, yaw):
    # Loosened thresholds to force matches
    zones = {
        'left':     {'range': ((-60, 60), (10, 180)),   'color': (1, 0, 0, 1)},     # red
        'right':    {'range': ((-60, 60), (-180, -10)), 'color': (0, 1, 0, 1)},     # green
        'up':       {'range': ((-90, -10), (-60, 60)),  'color': (0, 0, 1, 1)},     # blue
        'down':     {'range': ((10, 90), (-60, 60)),    'color': (1, 1, 0, 1)},     # yellow
        'center':   {'range': ((-10, 10), (-10, 10)),   'color': (1, 0, 1, 1)},     # magenta
    }

    for name, z in zones.items():
        pr, yr = z['range']
        if pr[0] <= pitch <= pr[1] and yr[0] <= yaw <= yr[1]:
            return z['color']
    return (0.5, 0.5, 0.5, 1)  # gray fallback

# === Main 3D visualization ===
def visualize_gaze_directions(gaze_vectors):
    app = QApplication(sys.argv)
    w = gl.GLViewWidget()
    w.setWindowTitle('3D Gaze with Zones (DEBUG)')
    w.setGeometry(0, 0, 1000, 800)
    w.setCameraPosition(distance=2)
    w.show()

    axes = gl.GLAxisItem()
    axes.setSize(x=1, y=1, z=1)
    w.addItem(axes)

    origin = np.array([0, 0, 0])

    # Show debug for first N vectors
    print("=== Pitch/Yaw Debug ===")
    for i, vec in enumerate(gaze_vectors):
        pitch, yaw = vector_to_pitch_yaw(vec)
        color = get_zone_color(pitch, yaw)

        if i < 10:
            print(f"[{i}] Pitch: {pitch:.2f}°, Yaw: {yaw:.2f}°, Color: {color}")

        endpoint = origin + vec / np.linalg.norm(vec) * 0.1  # Normalize
        arrow = gl.GLLinePlotItem(pos=np.array([origin, endpoint]), color=color, width=2.0, antialias=True)
        w.addItem(arrow)

    sys.exit(app.exec())

# === Run ===
if __name__ == "__main__":
    filepath = r"P:\3DGazeNet\data\test_videos\a-05282025154703-0000_out_resnet_x128_vertex\predicted_gaze_vectors.txt"
    gaze_vectors = load_gaze_vectors(filepath)
    visualize_gaze_directions(gaze_vectors)
