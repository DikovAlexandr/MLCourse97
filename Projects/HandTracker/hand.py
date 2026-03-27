"""
Hand Tracking → Shadow Hand (dex-retargeting + SAPIEN)
=======================================================
Требования:
  conda install sapien -c sapien
  pip install dex-retargeting mediapipe opencv-python
  git clone https://github.com/dexsuite/dex-urdf
  wget hand_landmarker.task

Управление viewer:
  ЛКМ      — вращение камеры
  ПКМ      — перемещение
  Колёсико — зум
  Q        — выход
"""

import cv2
import numpy as np
import time
import sapien
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from dex_retargeting.retargeting_config import RetargetingConfig

# ══════════════════════════════════════════════════════════════════════
# 1. DEX-RETARGETING
# ══════════════════════════════════════════════════════════════════════

CONFIG_PATH = "C:/Users/pawly/anaconda3/envs/dex_env/Lib/site-packages/dex_retargeting/configs/offline/shadow_hand_right.yml"
URDF_DIR    = "dex-urdf/robots/hands"
URDF_PATH   = "dex-urdf/robots/hands/shadow_hand/shadow_hand_right.urdf"

RetargetingConfig.set_default_urdf_dir(URDF_DIR)
config      = RetargetingConfig.load_from_file(CONFIG_PATH)
retargeting = config.build()

print("    dex-retargeting загружен из конфига")
print(f"   Тип оптимизатора: {type(retargeting.optimizer).__name__}")
print(f"   Суставов: {len(retargeting.joint_names)}")

# Индексы точек MediaPipe из конфига
human_indices = retargeting.optimizer.target_link_human_indices
print(f"   Индексы точек: {human_indices}")



class AngleSmoother:
    def __init__(self, alpha=0.3):
        self.alpha = alpha
        self.state = None

    def update(self, angles):
        if self.state is None:
            self.state = angles.copy()
        else:
            self.state = self.alpha * angles + (1 - self.alpha) * self.state
        return self.state

smoother = AngleSmoother(alpha=0.3)

# ══════════════════════════════════════════════════════════════════════
# 2. SAPIEN
# ══════════════════════════════════════════════════════════════════════

scene = sapien.Scene()
scene.set_timestep(1 / 240)
scene.add_ground(altitude=0)
scene.set_ambient_light([0.5, 0.5, 0.5])
scene.add_directional_light([0, 1, -1], [0.5, 0.5, 0.5])

viewer = scene.create_viewer()
viewer.set_camera_xyz(0.5, 0, 0.4)
viewer.set_camera_rpy(0, 0, 3.14)

# Загружаем Shadow Hand
loader = scene.create_urdf_loader()
loader.fix_root_link = True
sapien_robot = loader.load(URDF_PATH)
sapien_robot.set_root_pose(sapien.Pose([0, 0, 0.2]))

# Маппинг суставов retargeting → SAPIEN (исключаем dummy_* суставы)
sapien_joint_names      = [j.get_name() for j in sapien_robot.get_active_joints()]
retargeting_joint_names = retargeting.joint_names
real_names              = [n for n in retargeting_joint_names
                            if not n.startswith('dummy')]

retargeting_to_sapien = np.array([
    retargeting_joint_names.index(name)
    for name in sapien_joint_names
    if name in real_names
]).astype(int)

sapien_from_retargeting = np.array([
    i for i, name in enumerate(sapien_joint_names)
    if name in real_names
]).astype(int)

print(f"\n SAPIEN загружен")
print(f"   Суставов SAPIEN: {len(sapien_joint_names)}")
print(f"   Маппинг: {len(retargeting_to_sapien)} суставов совпадают")



# 3. MEDIAPIPE


base_opts = mp_python.BaseOptions(model_asset_path="hand_landmarker.task")
options   = vision.HandLandmarkerOptions(base_options=base_opts, num_hands=1)
detector  = vision.HandLandmarker.create_from_options(options)

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
]

def draw_skeleton(frame, landmarks):
    h, w = frame.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (0, 255, 0), 2)
    for pt in pts:
        cv2.circle(frame, pt, 4, (0, 0, 255), -1)


# 4. ОСНОВНОЙ ЦИКЛ


video_path = "human_hand_video.mp4"
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise IOError(f"Не удалось открыть: {video_path}")

fps        = cap.get(cv2.CAP_PROP_FPS) or 30.0
width      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_time = 1.0 / fps
print(f"\nВидео: {width}x{height} @ {fps:.1f} fps")

out_video = cv2.VideoWriter(
    "output_hand_robot.mp4",
    cv2.VideoWriter_fourcc(*'mp4v'),
    fps, (width, height),
)

prev_time = time.time()

while cap.isOpened() and not viewer.closed:

    # Синхронизация с видео
    elapsed = time.time() - prev_time
    if elapsed < frame_time:
        time.sleep(frame_time - elapsed)
    prev_time = time.time()

    ret, frame = cap.read()
    if not ret:
        break

    # MediaPipe детекция
    rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    res    = detector.detect(mp_img)

    # Скелет на кадре
    if res.hand_landmarks:
        draw_skeleton(frame, res.hand_landmarks[0])
        cv2.putText(frame, "Hand detected", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "No hand", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.imshow("Hand Tracking", frame)
    out_video.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Ретаргетинг и управление роботом
    if res.hand_world_landmarks:
        pts = np.array([[lm.x, lm.y, lm.z]
                        for lm in res.hand_world_landmarks[0]])

        # PositionOptimizer ожидает координаты нужных точек
        ref_value     = pts[human_indices]  # (10, 3)
        retarget_qpos = retargeting.retarget(ref_value)
        retarget_qpos = smoother.update(retarget_qpos)
        # Применяем к SAPIEN с правильным маппингом
        qpos = sapien_robot.get_qpos()
        qpos[sapien_from_retargeting] = retarget_qpos[retargeting_to_sapien]
        sapien_robot.set_qpos(qpos)

    scene.step()
    scene.update_render()
    viewer.render()


# 5. ЗАВЕРШЕНИЕ


cap.release()
out_video.release()
cv2.destroyAllWindows()
detector.close()
print("\n Видео: output_hand_robot.mp4")