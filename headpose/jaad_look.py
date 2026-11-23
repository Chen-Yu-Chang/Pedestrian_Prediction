"""Per-frame 'look' labels from the JAAD annotation files (data/JAAD/annotations/video_XXXX.xml).

Behaviour-annotated pedestrians carry a per-box attribute look = 'looking' / 'not-looking'.
Returns {(ped_id, frame): 1 or 0}. Frame numbers are original video frames, the same numbering as the
file names parsed in pose_extract.py.
"""
import os
import xml.etree.ElementTree as ET

from .config import JAAD_ANN


def load_look_labels(v_id, ann_dir=JAAD_ANN):
    path = os.path.join(ann_dir, f'video_{v_id}.xml')
    if not os.path.exists(path):
        return {}
    labels = {}
    for t in ET.parse(path).getroot().findall('track'):
        if t.get('label') not in ('pedestrian', 'ped'):
            continue
        for b in t.findall('box'):
            attrs = {a.get('name'): (a.text or '').strip() for a in b.findall('attribute')}
            look = attrs.get('look')
            if attrs.get('id') and look in ('looking', 'not-looking'):
                labels[(attrs['id'], int(b.get('frame')))] = int(look == 'looking')
    return labels
