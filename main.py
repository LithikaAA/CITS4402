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
 
import cv2
import numpy as np
from PIL import Image, ImageTk
import requests
 
# ── Download MediaPipe models on first run ──────────────────────────────────
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
os.makedirs(MODELS_DIR, exist_ok=True)
 
DETECTOR_MODEL_PATH   = os.path.join(MODELS_DIR, 'blaze_face_short_range.tflite')
LANDMARKER_MODEL_PATH = os.path.join(MODELS_DIR, 'face_landmarker.task')
 
DETECTOR_URL   = ('https://storage.googleapis.com/mediapipe-models/'
                  'face_detector/blaze_face_short_range/float16/1/'
                  'blaze_face_short_range.tflite')
LANDMARKER_URL = ('https://storage.googleapis.com/mediapipe-models/'
                  'face_landmarker/face_landmarker/float16/1/'
                  'face_landmarker.task')
 
 
def _download_if_missing(path, url):
    if not os.path.exists(path):
        print(f'Downloading {os.path.basename(path)} ...')
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        with open(path, 'wb') as f:
            f.write(r.content)
        print('  Done.')
 
 
_download_if_missing(DETECTOR_MODEL_PATH,   DETECTOR_URL)
_download_if_missing(LANDMARKER_MODEL_PATH, LANDMARKER_URL)
 
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
    roi = mask[y:y+h, x:x+w]
    if roi.size == 0:
        return 0.0
    return float(np.count_nonzero(roi)) / roi.size
 
 
# ─────────────────────────────────────────────
# FACE DETECTION
# ─────────────────────────────────────────────
 
def _build_detector():
    opts = mp_vision.FaceDetectorOptions(
        base_options=mp_python.BaseOptions(model_asset_path=DETECTOR_MODEL_PATH),
        min_detection_confidence=0.4,
    )
    return mp_vision.FaceDetector.create_from_options(opts)
 
 
def detect_faces(bgr_img, skin_threshold=0.10):
    h_img, w_img = bgr_img.shape[:2]
    s_mask  = skin_mask(bgr_img)
    rgb     = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    mp_img  = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result  = _build_detector().detect(mp_img)
 
    raw = []
    if result.detections:
        for det in result.detections:
            bb   = det.bounding_box
            x, y = max(0, bb.origin_x), max(0, bb.origin_y)
            w    = min(bb.width,  w_img - x)
            h    = min(bb.height, h_img - y)
            score = det.categories[0].score if det.categories else 0.5
            raw.append((x, y, w, h, score))
 
    # Use skin colour to filter false positives
    filtered = [(x, y, w, h, s) for (x, y, w, h, s) in raw
                if skin_ratio_in_box(s_mask, x, y, w, h) >= skin_threshold]
    if not filtered:
        filtered = raw  # fallback: keep all if skin filter removes everything
    if not filtered:
        return []
 
    boxes_np  = np.array([[x, y, x+w, y+h] for (x, y, w, h, _) in filtered], dtype=np.float32)
    scores_np = np.array([s for (_, _, _, _, s) in filtered], dtype=np.float32)
    indices   = _nms(boxes_np, scores_np)
 
    return [(int(boxes_np[i][0]), int(boxes_np[i][1]),
             int(boxes_np[i][2] - boxes_np[i][0]),
             int(boxes_np[i][3] - boxes_np[i][1])) for i in indices]
 
 
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
        min_face_detection_confidence=0.4,
        min_face_presence_confidence=0.4,
        min_tracking_confidence=0.4,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )
    return mp_vision.FaceLandmarker.create_from_options(opts)
 
 
def detect_landmarks(bgr_img, face_boxes):
    h_img, w_img = bgr_img.shape[:2]
    rgb    = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = _build_landmarker().detect(mp_img)
 
    if not result.face_landmarks:
        return [None] * len(face_boxes)
 
    mesh_pts = [
        np.array([[lm.x * w_img, lm.y * h_img] for lm in fl], dtype=np.float32)
        for fl in result.face_landmarks
    ]
    centres = [pts.mean(axis=0) for pts in mesh_pts]
 
    out = []
    for (bx, by, bw, bh) in face_boxes:
        cx, cy = bx + bw / 2, by + bh / 2
        best   = int(np.argmin([np.hypot(c[0]-cx, c[1]-cy) for c in centres]))
        pts    = mesh_pts[best]
        re = pts[_RIGHT_EYE_IDX].mean(axis=0)
        le = pts[_LEFT_EYE_IDX ].mean(axis=0)
        nt = pts[_NOSE_TIP_IDX]
        out.append((tuple(re.astype(int)), tuple(le.astype(int)), tuple(nt.astype(int))))
 
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
    out     = bgr_img.copy()
    boxes   = detect_faces(bgr_img)[:4]
    n_faces = len(boxes)
 
    for (x, y, w, h) in boxes:
        draw_box(out, x, y, w, h)
 
    if n_faces == 0:
        return out, [], 0
 
    all_lmks      = detect_landmarks(bgr_img, boxes)
    faces         = []
    aligned_lmks  = []
 
    for (x, y, w, h), lmks in zip(boxes, all_lmks):
        if lmks is None:
            lmks = ((x+w//4, y+h//3), (x+3*w//4, y+h//3), (x+w//2, y+h//2))
        re, le, nt = lmks
 
        if draw:
            draw_landmarks(out, re, le, nt)
 
        face, M = align_face(bgr_img, re, le, nt)
        faces.append(face)
 
        def clamp(p):
            return (int(np.clip(p[0], 0, OUTPUT_SIZE-1)),
                    int(np.clip(p[1], 0, OUTPUT_SIZE-1)))
 
        re_a, le_a, nt_a = transform_landmarks(M, [re, le, nt])
        aligned_lmks.append((clamp(re_a), clamp(le_a), clamp(nt_a)))
 
    overlay_faces(out, faces, aligned_lmks)
    return out, faces, n_faces
 
 
# ─────────────────────────────────────────────
# FEATURE EXTRACTION
# ─────────────────────────────────────────────
 
def extract_embedding(face_bgr):
    gray = cv2.resize(cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY), (64, 64))
 
    lbp = _lbp_hist(gray)
 
    hog_desc = cv2.HOGDescriptor(
        _winSize=(64,64), _blockSize=(16,16), _blockStride=(8,8),
        _cellSize=(8,8),  _nbins=9)
    hog = hog_desc.compute(gray).flatten()
 
    hsv = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2HSV)
    hh  = cv2.calcHist([hsv], [0], None, [16], [0, 180]).flatten()
    sh  = cv2.calcHist([hsv], [1], None, [16], [0, 256]).flatten()
    hh /= (hh.sum() + 1e-6)
    sh /= (sh.sum() + 1e-6)
 
    feat = np.concatenate([lbp, hog, hh, sh])
    return feat / (np.linalg.norm(feat) + 1e-6)
 
 
def _lbp_hist(gray, P=8, R=1, bins=256):
    h, w   = gray.shape
    lbp    = np.zeros_like(gray, dtype=np.uint8)
    angles = [2 * np.pi * p / P for p in range(P)]
    offs   = [(R * np.cos(a), -R * np.sin(a)) for a in angles]
    for y in range(1, h-1):
        for x in range(1, w-1):
            c    = int(gray[y, x])
            code = 0
            for bit, (dy, dx) in enumerate(offs):
                ny = int(np.clip(round(y+dy), 0, h-1))
                nx = int(np.clip(round(x+dx), 0, w-1))
                code |= (int(gray[ny, nx]) >= c) << bit
            lbp[y, x] = code
    hist, _ = np.histogram(lbp.flatten(), bins=bins, range=(0, 256))
    hist = hist.astype(np.float32)
    return hist / (hist.sum() + 1e-6)
 
 
# ─────────────────────────────────────────────
# CLUSTERING
# ─────────────────────────────────────────────
 
def cluster_identities(embeddings, eps=0.35, min_samples=1):
    if not embeddings:
        return np.array([], dtype=int)
    from sklearn.cluster import DBSCAN
    labels  = DBSCAN(eps=eps, min_samples=min_samples,
                     metric='cosine').fit_predict(np.vstack(embeddings))
    max_lbl = max(labels.max(), -1)
    new_lbl = labels.copy()
    for i, l in enumerate(labels):
        if l == -1:
            max_lbl += 1
            new_lbl[i] = max_lbl
    return new_lbl
 
 
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
        _, aligned, n = process_image(bgr, draw=False)
        total_faces += n
        n_done      += 1
        for m, face in enumerate(aligned):
            face_rs = cv2.resize(face, (OUTPUT_SIZE, OUTPUT_SIZE))
            embeddings.append(extract_embedding(face_rs))
            records.append((m, face_rs))
        if progress_callback:
            progress_callback(n_done, len(files))
 
    n_identities = 0
    if embeddings:
        labels       = cluster_identities(embeddings)
        n_identities = int(labels.max()) + 1
        for (m, img), lbl in zip(records, labels):
            cv2.imwrite(os.path.join(out_folder, f'Identity_{lbl}_face_{m}.jpg'), img)
 
    return dict(n_images=n_done, n_faces=total_faces, n_identities=n_identities,
                elapsed=time.time()-t0, output_folder=out_folder)
 
 
# ─────────────────────────────────────────────
# GUI  (styled to match CITS4402 lab code)
# ─────────────────────────────────────────────
 
DISPLAY_W = 380
DISPLAY_H = 285
BG        = '#e8e1f2'   # soft lilac
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
    out, _, n_faces = process_image(bgr)
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