#!/usr/bin/env bash
# Run the whole head-pose + Transformer pipeline in order, from the root of the pedestrian-intention repo.
#
#   bash run_all.sh            # run every step; steps whose output already exists are skipped
#   bash run_all.sh --check    # only check that the inputs are in place, run nothing
#   bash run_all.sh --from 3   # start at step 3 (earlier outputs must exist)
#   FORCE=1 bash run_all.sh    # re-run steps even if their output exists
#   QUICK=1 bash run_all.sh    # tiny run (few epochs, 2 folds, 1 seed) to test the setup end to end
#   bash run_all.sh --faces-only   # part A only: teacher labels + student stage 1 + face evaluation.
#                                  # Needs only the face datasets and teacher weights (no pose_*.pkl yet)
#
# Long run: start it so it survives a dropped SSH connection, then watch the log:
#   nohup bash run_all.sh > logs/run_all.log 2>&1 &
#   tail -f logs/run_all.log
#
# Settings (environment variables):
#   WORKERS     data-loading workers for training      (default: number of CPUs, max 8)
#   LIMIT       use only N 300W-LP faces in step 1     (default: all; e.g. 60000 if RAM is small)
#   FRAMES_DIR  folder with images/{split}/... at full resolution, for sharper head crops (default: stored frames)
#   MOD_DROP    modality dropout in the Transformer   (default: 0.15)
#   HEAD_FROM   keypoints (default) = place the head crop with the nose/eye/ear positions in pose_*.pkl
#               box = place it at the top of the pedestrian box only (no keypoints used anywhere)
#   BODY_TEST   1 = also run the separate box + body-pose (17 keypoints) test as step 8 (default: off)
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs outputs_head outputs_pose

PY=${PY:-python3}
CHECK=0; FROM=0; FACES_ONLY=0
while [ $# -gt 0 ]; do
  case "$1" in
    --check) CHECK=1 ;;
    --faces-only) FACES_ONLY=1 ;;
    --from) FROM="$2"; shift ;;
    *) echo "unknown option $1"; exit 1 ;;
  esac
  shift
done

# use full-resolution frames automatically once cut_jaad_frames.py has made them
if [ -z "${FRAMES_DIR:-}" ] && [ -d data/jaad_fullres/images ]; then FRAMES_DIR=data/jaad_fullres; fi
NCPU=$(nproc 2>/dev/null || echo 4)
WORKERS=${WORKERS:-$(( NCPU < 8 ? NCPU : 8 ))}
MOD_DROP=${MOD_DROP:-0.15}
if [ "${QUICK:-0}" = "1" ]; then
  E1="--epochs 1 --batch 32"; E2="--epochs 1 --batch 32"; TF="--folds 2 --seeds 0 --epochs 2"
  LIMIT=${LIMIT:-500}; echo ">>> QUICK mode: tiny run to test the setup, results are meaningless"
else
  E1=""; E2=""; TF=""
fi

# ---------- input checks ----------
missing=0
need() {   # need <path> <what it is / where it comes from>
  if [ ! -e "$1" ] || { [ -d "$1" ] && [ -z "$(ls -A "$1" 2>/dev/null)" ]; }; then
    echo "  MISSING  $1"; echo "           -> $2"; missing=1
  else
    echo "  ok       $1"
  fi
}
echo "== Checking inputs"
need data/sixdrepnet/best_a2.pt  "teacher weights: run setup_aws.sh (downloads it automatically)"
need data/HPE/300W_LP            "300W-LP faces: GET_FACES=1 bash setup_aws.sh"
need data/HPE/AFLW2000           "AFLW2000 faces: GET_FACES=1 bash setup_aws.sh"
if [ "$FACES_ONLY" = "0" ]; then
need exp_classifiers.py          "team repo file: setup_aws.sh downloads it"
need data/JAAD/annotations       "JAAD annotations: GET_JAAD=1 bash setup_aws.sh"
for s in train val test; do
  need outputs_pose/pose_$s.pkl  "pedestrian boxes + keypoints made by pose_extract.py - ask your teammate for this file (it is not on GitHub)"
  need "${FRAMES_DIR:-data/vehic-ped-intuition}/images/$s" "JAAD frames for the $s split - GET_JAAD=1 bash setup_aws.sh (cuts them from the JAAD clips once the pkl files are in place)"
done
fi
if [ "$missing" = "1" ]; then
  echo "Some inputs are missing (see above)."
  [ "$CHECK" = "1" ] && exit 1
  echo "Fix them, then run again."; exit 1
fi
echo "All inputs found."
[ "$CHECK" = "1" ] && exit 0

# a QUICK test run leaves tiny-model outputs behind; the next real run must redo every step
if [ "${QUICK:-0}" = "1" ]; then
  FORCE=1; touch outputs_head/.quick_run
elif [ -e outputs_head/.quick_run ]; then
  echo ">>> Last run was a QUICK test: redoing all steps for real"; FORCE=1; rm -f outputs_head/.quick_run
fi

# ---------- steps ----------
step() {   # step <number> <title> <output file that marks it as done> <command...>
  local n="$1" title="$2" done_file="$3"; shift 3
  if [ "$n" -lt "$FROM" ]; then echo "== Step $n skipped (--from $FROM)"; return; fi
  if [ -e "$done_file" ] && [ "${FORCE:-0}" != "1" ]; then
    echo "== Step $n ($title): output $done_file exists, skipping (FORCE=1 to redo)"; return
  fi
  echo "== Step $n: $title"
  echo "   $*"
  local t0=$SECONDS
  "$@" 2>&1 | tee "logs/step$n.log"
  echo "   done in $(( (SECONDS - t0) / 60 )) min (log: logs/step$n.log)"
}

FR=""; [ -n "${FRAMES_DIR:-}" ] && FR="--frames_dir $FRAMES_DIR"
LIM=""; [ -n "${LIMIT:-}" ] && LIM="--limit $LIMIT"

# Part A - head-pose student on faces (no JAAD data needed)
step 1 "teacher labels on 300W-LP / AFLW2000"   outputs_head/faces_AFLW2000.npz    $PY hp_01_label_faces.py $LIM
step 2 "student stage 1: distillation"          outputs_head/student_stage1.pt     $PY hp_03_train_student.py --stage 1 --workers "$WORKERS" $E1
if [ "$FACES_ONLY" = "1" ]; then
  step 5 "evaluate teacher vs student (faces)"   outputs_head/eval_aflw_by_size.csv $PY hp_04_eval_heads.py
  echo; echo "== Part A finished. Result: outputs_head/eval_aflw_by_size.csv (pose error vs head size)"
  echo "   Next, once pose_*.pkl and JAAD frames are in place:  bash run_all.sh"
  exit 0
fi
# Part B - JAAD heads (needs pose_*.pkl and frames)
step 3 "crop JAAD pedestrian heads"             outputs_head/heads_test.npz        $PY hp_02_extract_jaad_heads.py $FR --head_from "${HEAD_FROM:-keypoints}"
step 4 "student stage 2: looking fine-tune"     outputs_head/student_stage2.pt     $PY hp_03_train_student.py --stage 2 --workers "$WORKERS" $E2
step 5 "evaluate teacher vs student"            outputs_head/eval_jaad_look.csv    $PY hp_04_eval_heads.py
step 6 "per-frame head features"                outputs_head/head_test.pkl         $PY hp_05_head_features.py
# Part C - crossing prediction
step 7 "Transformer: box vs box + head pose"     outputs_pose/results_transformer_cv_bbox_bbox+head.csv \
       $PY exp_transformer.py --sets bbox bbox+head --mod_drop "$MOD_DROP" $TF
if [ "${BODY_TEST:-0}" = "1" ]; then
  step 8 "separate test: box + body pose"        outputs_pose/results_transformer_cv_bbox+pose.csv \
         $PY exp_transformer.py --sets bbox+pose $TF
fi

echo
echo "== Finished. Main results:"
echo "   outputs_head/eval_aflw_by_size.csv         head-pose error vs head size (teacher vs student)"
echo "   outputs_head/eval_jaad_look.csv            'looking' AUC vs head size"
echo "   outputs_pose/results_transformer_cv_bbox_bbox+head.csv   main test: box vs box + head pose"
if [ "${BODY_TEST:-0}" = "1" ]; then
  echo "   outputs_pose/results_transformer_cv_bbox+pose.csv        separate test: box + body pose"
fi
