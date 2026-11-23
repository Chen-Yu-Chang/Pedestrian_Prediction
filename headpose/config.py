"""Paths and constants shared by the head-pose scripts.

The JAAD paths mirror run_pipeline.py / run_pipeline_fixed.py; they are repeated here so the
head-pose scripts do not need ultralytics or seaborn just to import a path.
"""
import os

import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, 'data')

# ---- JAAD (same layout as run_pipeline.py) ----
BASE_PATH = os.path.join(DATA, 'vehic-ped-intuition')          # images/{train,val,test}/video_XXXX_fYYYY.jpg
XML_PATH = os.path.join(DATA, 'attributes-label', 'annotations_attributes')
JAAD_ANN = os.path.join(DATA, 'JAAD', 'annotations')            # video_XXXX.xml (boxes + 'look' attribute)
POSE_DIR = os.path.join(ROOT, 'outputs_pose')                   # pose_{split}.pkl from pose_extract.py

# ---- head-pose data / outputs ----
HPE_DATA = os.path.join(DATA, 'HPE')                            # 300W_LP/ and AFLW2000/ (SixDRepNet layout)
TEACHER_WEIGHTS = os.path.join(DATA, 'sixdrepnet', 'best_a2.pt')  # SixDRepNet release v1, RepVGG-A2
HEAD_DIR = os.path.join(ROOT, 'outputs_head')
os.makedirs(HEAD_DIR, exist_ok=True)

# ---- geometry of the stored JAAD frames ----
DET_SIZE = 640                 # boxes / keypoints in pose_{split}.pkl are in 640x640 detector coords
ORIG_W, ORIG_H = 1920, 1080    # JAAD original resolution
FRAME_STRIDE = 5

# ---- head crops ----
STUDENT_SIZE = 64              # student input (pixels, square)
TEACHER_SIZE = 224             # SixDRepNet input
TEACHER_MIN_PX = 32            # only trust the teacher on JAAD heads at least this many *stored* pixels tall
KP_CONF = 0.3                  # same keypoint threshold as exp_classifiers.py


def enhance_image(img):
    """CLAHE on the L channel, identical to run_pipeline.enhance_image."""
    if img is None:
        return None
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    cl = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)
