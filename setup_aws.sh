#!/usr/bin/env bash
# One-time setup on an AWS GPU instance (e.g. g5.xlarge or g4dn.xlarge, "Deep Learning AMI GPU PyTorch").
#
# Run from the root of the pedestrian-intention repo, after unzipping the add-on there:
#     bash setup_aws.sh
#
# Optional: copy data from S3 at the same time (paths are examples, use your own bucket):
#     S3_DATA=s3://my-bucket/pedestrian/data  S3_POSE=s3://my-bucket/pedestrian/outputs_pose  bash setup_aws.sh
#   S3_DATA  -> copied into data/          (JAAD frames + annotations, HPE faces, detector weights ...)
#   S3_POSE  -> copied into outputs_pose/  (pose_train.pkl, pose_val.pkl, pose_test.pkl)
#
# Optional: GET_JAAD=1 bash setup_aws.sh
#   clones the JAAD repo (annotations), downloads the JAAD clips (~3 GB) and, if the pose_*.pkl files are
#   already in outputs_pose/, cuts the needed frames at full resolution into data/jaad_fullres/.
#
# Optional: GET_FACES=1 bash setup_aws.sh
#   downloads the raw 300W-LP (2.8 GB, Google Drive) and AFLW2000-3D (87 MB) zips into data/HPE/ and checks
#   them against the SHA-256 checksums published by TensorFlow Datasets (same files TFDS downloads).
set -euo pipefail
cd "$(dirname "$0")"

echo "== 1. Python and GPU"
PY=${PY:-python3}
$PY --version
if command -v nvidia-smi >/dev/null; then
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
  echo "WARNING: no NVIDIA GPU found. Everything still runs, but steps 1-3 will be very slow on CPU."
fi

echo "== 2. Python packages"
$PY -m pip install --upgrade pip >/dev/null
$PY -m pip install -r requirements.txt
$PY - <<'EOF'
import torch
v = tuple(int(x) for x in torch.__version__.split('+')[0].split('.')[:2])
print(f'torch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')
if v < (2, 5):
    raise SystemExit('ERROR: torch >= 2.5 is needed (safe loading of the teacher checkpoint). '
                     'Run: pip install --upgrade torch torchvision')
EOF

echo "== 3. exp_classifiers.py from the team repo (the only repo file the pipeline imports)"
if [ ! -f exp_classifiers.py ]; then
  curl -L --fail -o exp_classifiers.py \
    https://raw.githubusercontent.com/use08174/pedestrian-intention/main/exp_classifiers.py
fi
ls -l exp_classifiers.py

echo "== 3b. Folders"
mkdir -p data/sixdrepnet data/HPE data/JAAD/annotations outputs_pose outputs_head logs

echo "== 4. Teacher weights (SixDRepNet RepVGG-A2)"
if [ ! -s data/sixdrepnet/best_a2.pt ]; then
  curl -L --fail -o data/sixdrepnet/best_a2.pt \
    https://github.com/Shohruh72/SixDRepNet/releases/download/v1/best_a2.pt
fi
ls -lh data/sixdrepnet/best_a2.pt

echo "== 5. Data from S3 (optional)"
if [ -n "${S3_DATA:-}" ]; then aws s3 sync "$S3_DATA" data/; else echo "S3_DATA not set, skipping"; fi
if [ -n "${S3_POSE:-}" ]; then aws s3 sync "$S3_POSE" outputs_pose/; else echo "S3_POSE not set, skipping"; fi

echo "== 6. JAAD annotations, clips and full-resolution frames (optional)"
if [ "${GET_JAAD:-0}" = "1" ]; then
  [ -d data/JAAD_repo ] || git clone --depth 1 https://github.com/ykotseruba/JAAD.git data/JAAD_repo
  mkdir -p data/JAAD/annotations && cp -u data/JAAD_repo/annotations/*.xml data/JAAD/annotations/
  if [ ! -d data/JAAD_repo/JAAD_clips ]; then (cd data/JAAD_repo && bash download_clips.sh); fi
  if ls outputs_pose/pose_*.pkl >/dev/null 2>&1; then
    $PY cut_jaad_frames.py --clips data/JAAD_repo/JAAD_clips --out data/jaad_fullres
    echo "Full-resolution frames ready in data/jaad_fullres (run_all.sh uses them automatically)"
  else
    echo "pose_*.pkl not found yet: put them in outputs_pose/, then run"
    echo "  python cut_jaad_frames.py --clips data/JAAD_repo/JAAD_clips"
  fi
else
  echo "GET_JAAD not set, skipping"
fi

echo "== 7. Face datasets for the head-pose student (optional)"
# Each dataset: skip if already unpacked; use a zip you uploaded yourself; otherwise try to download.
# A failed download prints what to do by hand and setup carries on (it never stops the rest).
unpack_zip() {   # unpack_zip <zip> <sha256 or ""> <target dir name>
  if [ -n "$2" ] && [ "$(sha256sum "$1" | cut -d' ' -f1)" != "$2" ]; then
    echo "   WARNING: $1 does not match the official checksum (could be a re-packed mirror); unpacking anyway"
  fi
  unzip -q -o "$1" -d data/HPE/_unpack && rm -f "$1"
  # find the folder that holds the jpg + mat pairs and move it into place
  src=$($PY - "$3" <<'PYEOF'
import os, sys
best, n = None, 0
for d, _, fs in os.walk('data/HPE/_unpack'):
    k = sum(f.endswith('.jpg') and f[:-4] + '.mat' in fs for f in fs)
    if k > n:
        best, n = d, k
if sys.argv[1] == '300W_LP' and best:            # 300W-LP has many sub-folders: take their common parent
    best = os.path.dirname(best)
print(best or '')
PYEOF
)
  if [ -z "$src" ]; then echo "   ERROR: no jpg + mat pairs found in $1"; return 1; fi
  rm -rf "data/HPE/$3" && mv "$src" "data/HPE/$3" && rm -rf data/HPE/_unpack
  echo "   ready: data/HPE/$3 ($(find "data/HPE/$3" -name '*.mat' | wc -l) .mat files)"
}
if [ "${GET_FACES:-0}" = "1" ]; then
  mkdir -p data/HPE
  # ---- AFLW2000-3D (test faces) ----
  if [ -d data/HPE/AFLW2000 ]; then
    echo "   AFLW2000 already there"
  elif ls data/HPE/[Aa][Ff][Ll][Ww]2000*.zip >/dev/null 2>&1; then
    unpack_zip "$(ls data/HPE/[Aa][Ff][Ll][Ww]2000*.zip | head -1)" 252bc35274d65ff27b6e573aa96c2f4c116ad88452cc984fb882258c0ed6e2d8 AFLW2000 || true
  elif curl -L --fail --connect-timeout 30 -o data/HPE/AFLW2000-3D.zip \
         http://www.cbsr.ia.ac.cn/users/xiangyuzhu/projects/3DDFA/Database/AFLW2000-3D.zip; then
    unpack_zip data/HPE/AFLW2000-3D.zip 252bc35274d65ff27b6e573aa96c2f4c116ad88452cc984fb882258c0ed6e2d8 AFLW2000 || true
  elif [ -f ~/.kaggle/kaggle.json ]; then
    rm -f data/HPE/AFLW2000-3D.zip
    echo "   official server unreachable, trying the Kaggle mirror mohamedadlyi/aflw2000-3d"
    $PY -m pip install -q kaggle
    $PY -m kaggle datasets download -d mohamedadlyi/aflw2000-3d -p data/HPE \
      && unpack_zip "$(ls data/HPE/aflw2000*.zip | head -1)" "" AFLW2000 || true
  else
    rm -f data/HPE/AFLW2000-3D.zip
    echo "   COULD NOT DOWNLOAD AFLW2000-3D. Do one of these, then run this script again:"
    echo "     a) download http://www.cbsr.ia.ac.cn/users/xiangyuzhu/projects/3DDFA/Database/AFLW2000-3D.zip"
    echo "        or the Kaggle mirror https://www.kaggle.com/datasets/mohamedadlyi/aflw2000-3d in your browser,"
    echo "        and upload the zip to data/HPE/  (the name must start with AFLW2000 or aflw2000)"
    echo "     b) put your Kaggle API token in ~/.kaggle/kaggle.json (kaggle.com > Settings > API)"
  fi
  # ---- 300W-LP (training faces) ----
  if [ -d data/HPE/300W_LP ]; then
    echo "   300W_LP already there"
  elif ls data/HPE/300W*LP*.zip >/dev/null 2>&1; then
    unpack_zip "$(ls data/HPE/300W*LP*.zip | head -1)" 250c366c417ad48f72522b629ff83dbb2b2a57945803ad4a530a696bb6b13ee7 300W_LP || true
  else
    $PY -m pip install -q gdown
    if $PY -m gdown 0B7OEHD3T4eCkVGs0TkhUWFN6N1k -O data/HPE/300W-LP.zip && [ -s data/HPE/300W-LP.zip ]; then
      unpack_zip data/HPE/300W-LP.zip 250c366c417ad48f72522b629ff83dbb2b2a57945803ad4a530a696bb6b13ee7 300W_LP || true
    else
      rm -f data/HPE/300W-LP.zip
      echo "   COULD NOT DOWNLOAD 300W-LP. Download it in your browser:"
      echo "     https://drive.google.com/uc?export=download&id=0B7OEHD3T4eCkVGs0TkhUWFN6N1k"
      echo "   upload it to data/HPE/300W-LP.zip, then run this script again"
    fi
  fi
  ls data/HPE
else
  echo "GET_FACES not set, skipping"
fi

echo "== 8. What is still missing"
bash run_all.sh --check || true
