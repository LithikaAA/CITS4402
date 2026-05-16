"""
CITS4402 Computer Vision Project - Face Detection and Matching
Group Members:
  - Jason Rarey - 22681384
  - Lithika Senthil Kumar - 23812347

Description:
    A GUI application that performs face detection, facial landmark detection,
    face alignment, and identity clustering on images.

Dependencies:
    pip install opencv-python mediapipe scikit-learn Pillow numpy scipy
"""
import os
import time
import threading
import tkinter as tk
from tkinter import filedialog, messagebox
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_distances
 
import cv2
import numpy as np
from PIL import Image, ImageTk
import requests
 
# ── Download MediaPipe models on first run ──────────────────────────────────
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
os.makedirs(MODELS_DIR, exist_ok=True)
 
DETECTOR_MODEL_PATH   = os.path.join(MODELS_DIR, 'blaze_face_short_range.tflite')
DETECTOR_FULL_MODEL_PATH = os.path.join(MODELS_DIR, 'blaze_face_full_range.tflite')
DETECTOR_FULL_URL = ('https://storage.googleapis.com/mediapipe-models/'
                     'face_detector/blaze_face_full_range/float16/1/'
                     'blaze_face_full_range.tflite')
LANDMARKER_MODEL_PATH = os.path.join(MODELS_DIR, 'face_landmarker.task')
SFACE_MODEL_PATH = os.path.join(MODELS_DIR, 'face_recognition_sface_2021dec.onnx')
 
DETECTOR_URL   = ('https://storage.googleapis.com/mediapipe-models/'
                  'face_detector/blaze_face_short_range/float16/1/'
                  'blaze_face_short_range.tflite')
LANDMARKER_URL = ('https://storage.googleapis.com/mediapipe-models/'
                  'face_landmarker/face_landmarker/float16/1/'
                  'face_landmarker.task')
SFACE_URL = ('https://github.com/opencv/opencv_zoo/raw/main/models/'
             'face_recognition_sface/face_recognition_sface_2021dec.onnx')
 
 
def _download_if_missing(path, url):
    if not os.path.exists(path):
        print(f'Downloading {os.path.basename(path)} ...')
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        with open(path, 'wb') as f:
            f.write(r.content)
        print('  Done.')
 
 
_download_if_missing(DETECTOR_MODEL_PATH,   DETECTOR_URL)
_download_if_missing(DETECTOR_FULL_MODEL_PATH, DETECTOR_FULL_URL)
_download_if_missing(LANDMARKER_MODEL_PATH, LANDMARKER_URL)
_download_if_missing(SFACE_MODEL_PATH, SFACE_URL)
 
# ── MediaPipe Tasks imports ─────────────────────────────────────────────────
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import mediapipe as mp
 
# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
TARGET_LANDMARKS = np.float32([
    [40, 40],   # right eye
    [85, 40],   # left eye
    [63, 70],   # nose tip
])
OUTPUT_SIZE = 125
 
# ─────────────────────────────────────────────
# SKIN COLOUR SEGMENTATION
# ─────────────────────────────────────────────
 
def skin_mask(bgr_img):
    hsv   = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
    ycrcb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2YCrCb)
    mask_hsv   = cv2.inRange(hsv,   np.array([0,  20,  70]),  np.array([25, 255, 255]))
    mask_ycrcb = cv2.inRange(ycrcb, np.array([0, 133,  77]),  np.array([255, 173, 127]))
    mask   = cv2.bitwise_and(mask_hsv, mask_ycrcb)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,   kernel, iterations=1)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel, iterations=2)
    return mask
 
 
def skin_ratio_in_box(mask, x, y, w, h):
    # Only check top 60% of box - avoids clothing at bottom affecting score
    face_h = int(h * 0.6)
    roi = mask[y:y+face_h, x:x+w]
    if roi.size == 0:
        return 0.0
    return float(np.count_nonzero(roi)) / roi.size
 
 
# ─────────────────────────────────────────────
# FACE DETECTION
# ─────────────────────────────────────────────
 
def _build_detector_short():
    opts = mp_vision.FaceDetectorOptions(
        base_options=mp_python.BaseOptions(model_asset_path=DETECTOR_MODEL_PATH),
        min_detection_confidence=0.25,
    )
    return mp_vision.FaceDetector.create_from_options(opts)

def _build_detector_full():
    opts = mp_vision.FaceDetectorOptions(
        base_options=mp_python.BaseOptions(model_asset_path=DETECTOR_FULL_MODEL_PATH),
        min_detection_confidence=0.25,
    )
    return mp_vision.FaceDetector.create_from_options(opts)

def _detect_opencv_haar(bgr_img):
    gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = cascade.detectMultiScale(
        gray,
        scaleFactor=1.05,
        minNeighbors=5,
        minSize=(60, 60)
    )
    raw = []
    if len(faces) > 0:
        for (x, y, w, h) in faces:
            h_img = bgr_img.shape[0]
            box_centre_y = y + h / 2
            if box_centre_y > h_img * 0.75:
                continue
            aspect = w / max(h, 1)
            if aspect < 0.7 or aspect > 1.4:
                continue
            raw.append((x, y, w, h, 1.0))  # high score = trusted
    return raw
 
def detect_faces(bgr_img, skin_threshold=0.05):
    h_img, w_img = bgr_img.shape[:2]
    s_mask = skin_mask(bgr_img)
    rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    raw = []

    # MediaPipe short range + full range detectors
    for detector in [_build_detector_short(), _build_detector_full()]:
        result = detector.detect(mp_img)
        if result.detections:
            for det in result.detections:
                bb = det.bounding_box
                x, y = max(0, bb.origin_x), max(0, bb.origin_y)
                w = min(bb.width,  w_img - x)
                h = min(bb.height, h_img - y)
                score = det.categories[0].score if det.categories else 0.5
                raw.append((x, y, w, h, score))

    # Haar cascade as third detector
    raw += _detect_opencv_haar(bgr_img)

    print("ALL RAW BOXES:")
    for (x, y, w, h, s) in raw:
        skin = skin_ratio_in_box(s_mask, x, y, w, h)
        print(f"  x={x} y={y} w={w} h={h} score={s:.2f} skin={skin:.3f}")

    # Use skin colour to filter false positives
    filtered = [(x, y, w, h, s) for (x, y, w, h, s) in raw
                if s >= 1.0 or skin_ratio_in_box(s_mask, x, y, w, h) >= skin_threshold]
    if not filtered:
        filtered = raw
    if not filtered:
        return []

    print("AFTER SKIN FILTER:")
    for (x, y, w, h, s) in filtered:
        print(f"  x={x} y={y} w={w} h={h} score={s:.2f}")

    boxes_np  = np.array([[x, y, x+w, y+h] for (x, y, w, h, _) in filtered], dtype=np.float32)
    scores_np = np.array([s for (_, _, _, _, s) in filtered], dtype=np.float32)
    indices   = _nms(boxes_np, scores_np)

    print("AFTER NMS:")
    for i in indices:
        print(f"  x={int(boxes_np[i][0])} y={int(boxes_np[i][1])} w={int(boxes_np[i][2]-boxes_np[i][0])} h={int(boxes_np[i][3]-boxes_np[i][1])}")

    return [(int(boxes_np[i][0]), int(boxes_np[i][1]),
             int(boxes_np[i][2] - boxes_np[i][0]),
             int(boxes_np[i][3] - boxes_np[i][1])) for i in indices]

def remove_duplicate_face_boxes(boxes, centre_threshold=45):
    unique = []

    for box in boxes:
        x, y, w, h = box
        cx = x + w / 2
        cy = y + h / 2

        is_duplicate = False

        for ux, uy, uw, uh in unique:
            ucx = ux + uw / 2
            ucy = uy + uh / 2

            centre_dist = np.hypot(cx - ucx, cy - ucy)

            if centre_dist < centre_threshold:
                is_duplicate = True
                break

        if not is_duplicate:
            unique.append(box)

    return unique
 
 
def _nms(boxes, scores, iou_threshold=0.3):
    if len(boxes) == 0:
        return []
    order = np.argsort(scores)[::-1]
    keep  = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        ious = _iou(boxes[i], boxes[rest])
        order = rest[ious < iou_threshold]
    return keep
 
 
def _iou(box, others):
    x1 = np.maximum(box[0], others[:, 0])
    y1 = np.maximum(box[1], others[:, 1])
    x2 = np.minimum(box[2], others[:, 2])
    y2 = np.minimum(box[3], others[:, 3])
    inter  = np.maximum(0, x2-x1) * np.maximum(0, y2-y1)
    area_b = (box[2]-box[0]) * (box[3]-box[1])
    area_o = (others[:,2]-others[:,0]) * (others[:,3]-others[:,1])
    return inter / (area_b + area_o - inter + 1e-6)
 
 
# ─────────────────────────────────────────────
# FACIAL LANDMARK DETECTION
# ─────────────────────────────────────────────
 
_RIGHT_EYE_IDX = [33, 133, 160, 159, 158, 144, 145, 153]
_LEFT_EYE_IDX  = [362, 263, 387, 386, 385, 373, 374, 380]
_NOSE_TIP_IDX  = 4
 
 
def _build_landmarker():
    opts = mp_vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=LANDMARKER_MODEL_PATH),
        num_faces=4,
        min_face_detection_confidence=0.2,  # lower from 0.4
        min_face_presence_confidence=0.2,   # lower from 0.4
        min_tracking_confidence=0.2,        # lower from 0.4
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )

    return mp_vision.FaceLandmarker.create_from_options(opts)
 
 
def detect_landmarks(bgr_img, face_boxes):
    h_img, w_img = bgr_img.shape[:2]
    rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = _build_landmarker().detect(mp_img)
    print(f"FaceMesh found {len(result.face_landmarks) if result.face_landmarks else 0} landmark sets")

    if not result.face_landmarks:
        return [None] * len(face_boxes)

    mesh_pts = [
        np.array([[lm.x * w_img, lm.y * h_img] for lm in fl], dtype=np.float32)
        for fl in result.face_landmarks
    ]

    centres = [pts.mean(axis=0) for pts in mesh_pts]

    out = []
    used_landmark_indices = set()

    for (bx, by, bw, bh) in face_boxes:
        box_cx = bx + bw / 2
        box_cy = by + bh / 2

        best_idx = None
        best_dist = float("inf")

        for i, centre in enumerate(centres):
            if i in used_landmark_indices:
                continue

            dist = np.hypot(centre[0] - box_cx, centre[1] - box_cy)

            if dist < best_dist:
                best_dist = dist
                best_idx = i

        if best_idx is None:
            out.append(None)
            continue

        used_landmark_indices.add(best_idx)

        pts = mesh_pts[best_idx]

        re = pts[_RIGHT_EYE_IDX].mean(axis=0)
        le = pts[_LEFT_EYE_IDX].mean(axis=0)
        nt = pts[_NOSE_TIP_IDX]

        out.append((
            tuple(re.astype(int)),
            tuple(le.astype(int)),
            tuple(nt.astype(int))
        ))

    while len(out) < len(face_boxes):
        out.append(None)

    return out
 
 
# ─────────────────────────────────────────────
# FACIAL ALIGNMENT
# ─────────────────────────────────────────────
 
def align_face(bgr_img, right_eye, left_eye, nose_tip):
    src  = np.float32([right_eye, left_eye, nose_tip])
    M, _ = cv2.estimateAffinePartial2D(src, TARGET_LANDMARKS, method=cv2.LMEDS)
    if M is None:
        M = cv2.getAffineTransform(src, TARGET_LANDMARKS)
    warped = cv2.warpAffine(bgr_img, M,
                            (bgr_img.shape[1], bgr_img.shape[0]),
                            flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_REFLECT_101)
    crop = warped[0:OUTPUT_SIZE, 0:OUTPUT_SIZE]
    if crop.shape[:2] != (OUTPUT_SIZE, OUTPUT_SIZE):
        crop = cv2.resize(crop, (OUTPUT_SIZE, OUTPUT_SIZE))
    return crop, M
 
 
def transform_landmarks(M, points):
    pts = np.float32([[p] for p in points])
    return [tuple(p[0].astype(int)) for p in cv2.transform(pts, M)]
 
 
# ─────────────────────────────────────────────
# DRAW UTILITIES
# ─────────────────────────────────────────────
 
def draw_box(img, x, y, w, h):
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
 
 
def draw_landmarks(img, re, le, nt, r=5):
    cv2.circle(img, re, r, (0,   0, 255), -1)   # red   – right eye
    cv2.circle(img, le, r, (0, 255,   0), -1)   # green – left eye
    cv2.circle(img, nt, r, (255, 0,   0), -1)   # blue  – nose tip
 
 
def overlay_faces(out_img, faces, lmks_list):
    h, w = out_img.shape[:2]
    corners = [(0, 0), (w-OUTPUT_SIZE, 0),
               (0, h-OUTPUT_SIZE), (w-OUTPUT_SIZE, h-OUTPUT_SIZE)]
    for i, (face, lmks) in enumerate(zip(faces, lmks_list)):
        if i >= 4:
            break
        cx, cy = corners[i]
        patch  = face.copy()
        if lmks:
            draw_landmarks(patch, *lmks, r=4)
        cy2 = min(cy + OUTPUT_SIZE, h)
        cx2 = min(cx + OUTPUT_SIZE, w)
        out_img[cy:cy2, cx:cx2] = patch[:cy2-cy, :cx2-cx]
 
 
# ─────────────────────────────────────────────
# FULL PIPELINE
# ─────────────────────────────────────────────
 
def process_image(bgr_img, draw=True):
    out = bgr_img.copy()

    # First remove obvious duplicate face boxes before doing landmarks/alignment.
    boxes = remove_duplicate_face_boxes(detect_faces(bgr_img), centre_threshold=40)[:4]

    if len(boxes) == 0:
        return out, [], [], 0

    all_lmks = detect_landmarks(bgr_img, boxes)

    faces = []
    embedding_faces = []
    aligned_lmks = []

    kept_box_centres = []
    kept_landmark_triplets = []
    kept_face_hashes = []

    for (x, y, w, h), lmks in zip(boxes, all_lmks):
        if lmks is None:
            lmks = (
                (x + w // 4, y + h // 3),
                (x + 3 * w // 4, y + h // 3),
                (x + w // 2, y + h // 2)
            )

        re, le, nt = lmks

        current_centre = np.array([x + w / 2, y + h / 2], dtype=np.float32)
        current_triplet = np.array([re, le, nt], dtype=np.float32)

        is_duplicate = False

        # Duplicate check 1: same source-image face location.
        for old_centre in kept_box_centres:
            centre_dist = np.linalg.norm(current_centre - old_centre)
            if centre_dist < 5:
                is_duplicate = True
                break

        if is_duplicate:
            continue

        # Duplicate check 2: only run if landmarks are real (not estimated)
        if lmks is not None:
            for old_triplet in kept_landmark_triplets:
                landmark_dist = np.mean(
                    np.linalg.norm(current_triplet - old_triplet, axis=1)
                )
                if landmark_dist < 5:
                    is_duplicate = True
                    break

        if is_duplicate:
            print(f"DROPPED as duplicate: box centre={current_centre}")
            continue

        face, M = align_face(bgr_img, re, le, nt)

        # Duplicate check 3: very similar aligned crop.
        face_gray = cv2.resize(
            cv2.cvtColor(face, cv2.COLOR_BGR2GRAY),
            (32, 32)
        )
        face_hash = face_gray.astype(np.float32).flatten()
        face_hash = face_hash / (np.linalg.norm(face_hash) + 1e-6)

        for old_hash in kept_face_hashes:
            similarity = float(np.dot(face_hash, old_hash))

            if similarity > 0.995:
                is_duplicate = True
                break

        if is_duplicate:
            continue

        kept_box_centres.append(current_centre)
        kept_landmark_triplets.append(current_triplet)
        kept_face_hashes.append(face_hash)

        faces.append(face)

        # Important: use the same aligned crop for identity embedding.
        # This keeps the saved crop and clustering input consistent.
        embedding_faces.append(face)

        if draw:
            draw_box(out, x, y, w, h)
            draw_landmarks(out, re, le, nt)

        def clamp(p):
            return (
                int(np.clip(p[0], 0, OUTPUT_SIZE - 1)),
                int(np.clip(p[1], 0, OUTPUT_SIZE - 1))
            )

        re_a, le_a, nt_a = transform_landmarks(M, [re, le, nt])
        aligned_lmks.append((clamp(re_a), clamp(le_a), clamp(nt_a)))

    overlay_faces(out, faces, aligned_lmks)

    return out, faces, embedding_faces, len(faces)
 
 
# ─────────────────────────────────────────────
# FEATURE EXTRACTION USING DEEPFACE AND CROP HELPER
# ─────────────────────────────────────────────
def crop_face_for_embedding(bgr_img, box, margin=0.35):
    x, y, w, h = box
    h_img, w_img = bgr_img.shape[:2]

    cx = x + w / 2
    cy = y + h / 2

    size = int(max(w, h) * (1.0 + margin))

    x1 = int(max(0, cx - size / 2))
    y1 = int(max(0, cy - size / 2))
    x2 = int(min(w_img, cx + size / 2))
    y2 = int(min(h_img, cy + size / 2))

    crop = bgr_img[y1:y2, x1:x2]

    if crop.size == 0:
        return None

    return crop



from deepface import DeepFace

DEEPFACE_MODEL_NAME = "Facenet512"


def extract_embedding(face_bgr):
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)

    reps = DeepFace.represent(
        img_path=face_rgb,
        model_name=DEEPFACE_MODEL_NAME,
        detector_backend="skip",
        enforce_detection=False,
        align=False
    )

    if isinstance(reps, list):
        embedding = reps[0]["embedding"]
    else:
        embedding = reps["embedding"]

    embedding = np.asarray(embedding, dtype=np.float32)
    embedding = embedding / (np.linalg.norm(embedding) + 1e-6)

    return embedding
 
 
# ─────────────────────────────────────────────
# CLUSTERING
# ─────────────────────────────────────────────
 
def cluster_identities(embeddings, distance_threshold=0.65):
    if not embeddings:
        return np.array([], dtype=int)

    

    X = np.vstack(embeddings)
    D = cosine_distances(X)

    print("Cosine distance matrix:")
    print(np.round(D, 3))

    try:
        model = AgglomerativeClustering(
            n_clusters=None,
            metric="precomputed",
            linkage="complete",
            distance_threshold=distance_threshold
        )
    except TypeError:
        model = AgglomerativeClustering(
            n_clusters=None,
            affinity="precomputed",
            linkage="complete",
            distance_threshold=distance_threshold
        )

    labels = model.fit_predict(D)

    unique_labels = sorted(set(labels))
    remap = {old: new for new, old in enumerate(unique_labels)}
    labels = np.array([remap[x] for x in labels], dtype=int)

    return labels
 
 
# ─────────────────────────────────────────────
# BULK PROCESSING
# ─────────────────────────────────────────────
 
SUPPORTED_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
 
 
def bulk_process(folder_path, progress_callback=None):
    t0    = time.time()
    files = sorted([f for f in os.listdir(folder_path)
                    if os.path.splitext(f)[1].lower() in SUPPORTED_EXTS])
 
    if not files:
        return dict(n_images=0, n_faces=0, n_identities=0,
                    elapsed=0, output_folder=None)
 
    out_folder = os.path.join(folder_path, 'Processed_Images')
    if os.path.exists(out_folder):
        for f in os.listdir(out_folder):
            os.remove(os.path.join(out_folder, f))
    else:
        os.makedirs(out_folder)
 
    embeddings, records = [], []
    total_faces, n_done = 0, 0
 
    for fname in files:
        bgr = cv2.imread(os.path.join(folder_path, fname))
        if bgr is None:
            continue
        _, aligned, embedding_faces, n = process_image(bgr, draw=False)
        total_faces += n
        n_done      += 1
        for display_face, embed_face in zip(aligned, embedding_faces):
            display_face_rs = cv2.resize(display_face, (OUTPUT_SIZE, OUTPUT_SIZE))

            embeddings.append(extract_embedding(embed_face))

            global_face_id = len(records)
            records.append((global_face_id, display_face_rs, fname))
        if progress_callback:
            progress_callback(n_done, len(files))
 
    n_identities = 0

    if embeddings:
        labels = cluster_identities(embeddings)
        n_identities = int(labels.max()) + 1

        filtered_records = []
        filtered_labels = []
        seen_source_identity = set()

        for record, lbl in zip(records, labels):
            face_id, img, source_fname = record
            source_base = os.path.splitext(source_fname)[0]
            key = (source_base, int(lbl))

            if key in seen_source_identity:
                print(f"Skipping duplicate from source image {source_base}: face {face_id} in Identity_{lbl}")
                continue

            seen_source_identity.add(key)
            filtered_records.append(record)
            filtered_labels.append(lbl)

        records = filtered_records
        labels = np.array(filtered_labels, dtype=int)

        n_identities = len(set(labels))
        total_faces = len(records)

        for (m, img, source_fname), lbl in zip(records, labels):
            base_name = os.path.splitext(source_fname)[0]
            cv2.imwrite(
                os.path.join(out_folder, f'Identity_{lbl}_{base_name}_face_{m}.jpg'),
                img
            )

        print("Cluster labels:", labels)

        groups = {}
        for (face_id, _, source_fname), lbl in zip(records, labels):
            groups.setdefault(lbl, []).append(f"{source_fname}: face {face_id}")

        print("Identity groups:")
        for lbl, face_ids in groups.items():
            print(f"Identity {lbl}: faces {face_ids}")

        print("Number of identities:", n_identities)
 
    return dict(n_images=n_done, n_faces=total_faces, n_identities=n_identities,
                elapsed=time.time()-t0, output_folder=out_folder)
 
 
# ─────────────────────────────────────────────
# GUI  (styled to match CITS4402 lab code)
# ─────────────────────────────────────────────
 
DISPLAY_W = 380
DISPLAY_H = 285
BG        = "#6600f4"   # soft lilac
BTN_CLR   = '#f57bbc'
BTN_ACT   = '#f8a8d2'
 
# Keep PhotoImage references alive so tkinter doesn't garbage-collect them
_photo_left  = None
_photo_right = None
 
 
def show_image(bgr, label, side):
    global _photo_left, _photo_right
    rgb     = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w    = rgb.shape[:2]
    scale   = min(DISPLAY_W / w, DISPLAY_H / h)
    resized = cv2.resize(rgb, (int(w * scale), int(h * scale)))
    tk_img  = ImageTk.PhotoImage(Image.fromarray(resized))
    label.config(image=tk_img)
    if side == 'left':
        _photo_left  = tk_img
    else:
        _photo_right = tk_img
 
 
def draw_progress(fraction):
    prog_canvas.delete('all')
    prog_canvas.create_rectangle(0, 0, int(860 * fraction), 10,
                                 fill='#4caf50', outline='')
 
 
def single_image():
    path = filedialog.askopenfilename(
        title='Select an image',
        filetypes=[('Images', '*.jpg *.jpeg *.png *.bmp *.tiff *.webp'),
                   ('All files', '*.*')])
    if not path:
        return
    bgr = cv2.imread(path)
    if bgr is None:
        messagebox.showerror('Error', f'Cannot read:\n{path}')
        return
 
    show_image(bgr, lbl_input, 'left')
    var_msg1.set('Processing...')
    var_msg2.set('')
    root.update()
 
    t0 = time.time()
    out, _, _, n_faces = process_image(bgr)
    elapsed = time.time() - t0
 
    show_image(out, lbl_output, 'right')
    var_msg1.set(f'Single image processed in {elapsed:.2f} seconds')
    var_msg2.set(f'Single image found {n_faces} face{"s" if n_faces != 1 else ""}')
    draw_progress(0)
 
 
def bulk_processing():
    folder = filedialog.askdirectory(title='Select folder of images')
    if not folder:
        return
    var_msg1.set('Bulk processing... please wait.')
    var_msg2.set('')
    draw_progress(0)
    root.update()
 
    def _run():
        def _prog(done, total):
            root.after(0, draw_progress, done / max(total, 1))
            root.after(0, var_msg1.set, f'Processing image {done}/{total}...')
        stats = bulk_process(folder, progress_callback=_prog)
        root.after(0, _bulk_done, stats)
 
    threading.Thread(target=_run, daemon=True).start()
 
 
def _bulk_done(stats):
    draw_progress(1.0)
    n  = stats['n_images']
    f  = stats['n_faces']
    z  = stats['n_identities']
    s  = stats['elapsed']
    of = stats['output_folder']
    var_msg1.set(f'Total {n} images processed in {s:.2f} seconds.')
    var_msg2.set(f'{f} faces detected corresponding to '
                 f'{z} unique identit{"ies" if z != 1 else "y"}.')
    if of:
        messagebox.showinfo('Bulk Processing Complete',
                            f'Results saved to:\n{of}\n\n'
                            f'{n} images | {f} faces | {z} identities')
 
 
# ── Build the window (place-based layout, same style as lab code) ─────────────
root = tk.Tk()
root.title('Image GUI')
root.geometry('920x530')
root.resizable(False, False)
root.configure(bg=BG)
 
# Panel labels
tk.Label(root, text='Input Image',     bg=BG, font=('Arial', 11)).place(x=30,  y=5)
tk.Label(root, text='Processed Image', bg=BG, font=('Arial', 11)).place(x=510, y=5)
 
# Image display boxes
lbl_input  = tk.Label(root, bg='black', relief='solid', bd=1)
lbl_input.place(x=30, y=28, width=DISPLAY_W, height=DISPLAY_H)
 
lbl_output = tk.Label(root, bg='black', relief='solid', bd=1)
lbl_output.place(x=510, y=28, width=DISPLAY_W, height=DISPLAY_H)
 
# Status messages
var_msg1 = tk.StringVar(value='')
var_msg2 = tk.StringVar(value='')
tk.Label(root, textvariable=var_msg1, bg=BG, anchor='w',
         font=('Arial', 10)).place(x=30, y=325)
tk.Label(root, textvariable=var_msg2, bg=BG, anchor='w',
         font=('Arial', 10)).place(x=30, y=348)
 
# Progress bar
prog_canvas = tk.Canvas(root, width=860, height=10, bg=BG, highlightthickness=0)
prog_canvas.place(x=30, y=375)
 
# Hint text
tk.Label(root,
         text='Press "Single Image" to load and process one image,  '
              'or "Bulk Processing" to process a whole folder.',
         bg=BG, font=('Arial', 10)).place(x=30, y=392)
 
# Buttons
tk.Button(root, text='Single Image',    font=('Arial', 12),
          bg=BTN_CLR, activebackground=BTN_ACT,
          command=single_image).place(x=30, y=430)
 
tk.Button(root, text='Bulk Processing', font=('Arial', 12),
          bg=BTN_CLR, activebackground=BTN_ACT,
          command=bulk_processing).place(x=220, y=430)
 
# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == '__main__':
    root.mainloop()