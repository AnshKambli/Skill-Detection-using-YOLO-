#!/usr/bin/env python3
"""
DERMA-SCAN: real-time skin analysis HUD

Webcam -> MediaPipe face landmarks -> face zones -> your YOLOv8 model
(Acnes / Dark Circle / Wrinkle) + live colour metrics (shine, redness) -> sci-fi overlay.

Cosmetic tracking only. This is not a medical device and the score is an experimental
composite. Everything runs locally; frames are never saved unless you press S.

Keys:  B set baseline | S save snapshot + log row | M toggle mesh dots | D debug view
       [ and ] lower / raise detection confidence | Q or ESC quit
Test on a still photo instead of the webcam:  python skin_scanner.py --image photo.jpg
"""
import argparse
import csv
import os
import sys
import threading
import time
import urllib.request
from collections import deque
from datetime import datetime

import cv2
import numpy as np

# ----------------------------------------------------------------------------------
# Settings you may want to tune
# ----------------------------------------------------------------------------------
NAMES = ["Acnes", "Dark Circle", "Wrinkle"]      # must match the training order

# Skin index = 100 - penalties (weights are arbitrary, adjust to taste)
ACNE_PTS, ACNE_CAP = 3.0, 40.0                   # points per blemish, max penalty
EYE_PTS = 8.0                                    # per eye with a dark circle
WRINKLE_PTS, WRINKLE_CAP = 3.0, 15.0
SHINE_FREE, SHINE_PTS, SHINE_CAP = 1.0, 2.0, 10.0   # % of T-zone that is shiny

SHINE_V, SHINE_S = 225, 70       # a pixel counts as shine if V >= 225 and S <= 70
SMALL_W = 320                    # width the face crop is shrunk to for cheap metrics
PANEL_W = 340                    # side panel width at 720p

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
             "face_landmarker/float16/latest/face_landmarker.task")

# ----------------------------------------------------------------------------------
# Face landmark groups (MediaPipe 478-point topology; "left" = left side of the image)
# ----------------------------------------------------------------------------------
OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400,
        377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
FOREHEAD = [21, 54, 103, 67, 109, 10, 338, 297, 332, 284, 251, 300, 293, 334, 296, 336, 9,
            107, 66, 105, 63, 70]
HULLS = {
    "nose":        [168, 129, 358, 98, 327, 2],
    "chin":        [176, 148, 152, 377, 400, 18],
}
LID_L = [33, 7, 163, 144, 145, 153, 154, 155, 133]
LID_R = [263, 249, 390, 373, 374, 380, 381, 382, 362]
ZONE_ORDER = ["under_eye_left", "under_eye_right", "forehead", "nose",
              "cheek_left", "cheek_right", "chin", "other_face"]
LABEL = {"under_eye_left": "L EYE", "under_eye_right": "R EYE", "forehead": "FOREHEAD",
         "nose": "NOSE", "cheek_left": "L CHEEK", "cheek_right": "R CHEEK",
         "chin": "CHIN", "other_face": "OTHER"}

# ----------------------------------------------------------------------------------
# Look and feel (BGR)
# ----------------------------------------------------------------------------------
FONT = cv2.FONT_HERSHEY_SIMPLEX
CYAN, MAGENTA, AMBER = (255, 220, 40), (255, 60, 200), (0, 190, 255)
GREEN, RED, DIM, WHITE = (120, 255, 120), (70, 70, 255), (140, 140, 140), (235, 235, 235)
CLS_COL = {"Acnes": MAGENTA, "Dark Circle": AMBER, "Wrinkle": GREEN}


# ----------------------------------------------------------------------------------
# Geometry and zones
# ----------------------------------------------------------------------------------
def face_roi(pts, shape, margin=0.18):
    h, w = shape[:2]
    o = pts[OVAL]
    x0, y0 = o.min(axis=0)
    x1, y1 = o.max(axis=0)
    mx, my = (x1 - x0) * margin, (y1 - y0) * margin
    return (max(0, int(x0 - mx)), max(0, int(y0 - my)),
            min(w, int(x1 + mx)), min(h, int(y1 + my)))


def poly_mask(shape, pts):
    m = np.zeros(shape[:2], np.uint8)
    cv2.fillPoly(m, [np.asarray(pts).astype(np.int32)], 255)
    return m > 0


def under_eye_poly(pts, ids):
    lid = pts[ids]
    eye_w = np.linalg.norm(pts[ids[0]] - pts[ids[-1]])
    return np.vstack([lid, (lid + [0, 0.30 * eye_w])[::-1]])


def build_zones(shape, pts):
    """Exclusive (non-overlapping) boolean zone masks inside the face oval."""
    face = poly_mask(shape, pts[OVAL])
    raw = {"under_eye_left": poly_mask(shape, under_eye_poly(pts, LID_L)),
           "under_eye_right": poly_mask(shape, under_eye_poly(pts, LID_R)),
           "forehead": poly_mask(shape, pts[FOREHEAD])}
    for k, ids in HULLS.items():
        hull = cv2.convexHull(pts[ids].astype(np.int32)).reshape(-1, 2)
        raw[k] = poly_mask(shape, hull)
    raw["cheek_left"] = poly_mask(shape, cheek_poly(pts, LID_L, 234, 129, 61, [58, 132, 93]))
    raw["cheek_right"] = poly_mask(shape, cheek_poly(pts, LID_R, 454, 358, 291, [288, 361, 323]))
    taken = np.zeros(shape[:2], bool)
    zones = {}
    for k in ZONE_ORDER[:-1]:
        zones[k] = raw[k] & face & ~taken
        taken |= zones[k]
    zones["other_face"] = face & ~taken
    return zones


def cheek_poly(pts, lid, edge, wing, mouth, jaw):
    """Region under the eye, beside the nose, down to the mouth corner and jaw line."""
    eye_w = np.linalg.norm(pts[lid[0]] - pts[lid[-1]])
    top = pts[lid] + np.array([0, 0.25 * eye_w])          # just inside the under-eye band
    return np.vstack([pts[[edge]], top, pts[[wing, mouth]], pts[jaw]])


def zone_contours(zones, scale, x0, y0):
    out = {}
    for z, m in zones.items():
        cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        out[z] = [(c.astype(np.float32) / scale + [x0, y0]).astype(np.int32)
                  for c in cs if len(c) >= 3]
    return out


# ----------------------------------------------------------------------------------
# Colour metrics and signal quality
# ----------------------------------------------------------------------------------
def zone_metrics(roi_s, zones):
    """Per-zone redness (a* minus face mean) and shine (% of pixels that are glare)."""
    lab = cv2.cvtColor(roi_s, cv2.COLOR_BGR2LAB).astype(np.float32)
    a = lab[..., 1] - 128
    hsv = cv2.cvtColor(roi_s, cv2.COLOR_BGR2HSV)
    spec = (hsv[..., 2] >= SHINE_V) & (hsv[..., 1] <= SHINE_S)
    face = np.logical_or.reduce(list(zones.values()))
    if not face.any():
        return {z: dict(red=0.0, shine=0.0) for z in zones}, 0.0
    a_face = float(a[face].mean())
    out = {}
    for z, m in zones.items():
        if m.sum() < 30:
            out[z] = dict(red=0.0, shine=0.0)
        else:
            out[z] = dict(red=float(a[m].mean() - a_face), shine=100.0 * float(spec[m].mean()))
    return out, float(lab[..., 0][face].mean() * 100 / 255)


def assess_quality(L_face, pts, W):
    score, hints = 100, []
    if L_face < 30:
        score -= 40; hints.append("MORE LIGHT NEEDED")
    elif L_face > 82:
        score -= 40; hints.append("TOO BRIGHT, REDUCE GLARE")
    ratio = (pts[454][0] - pts[234][0]) / W
    if ratio < 0.20:
        score -= 30; hints.append("MOVE CLOSER")
    elif ratio > 0.70:
        score -= 20; hints.append("MOVE BACK")
    dl = np.linalg.norm(pts[1] - pts[234])
    dr = np.linalg.norm(pts[1] - pts[454])
    if abs(dl - dr) / max(dl + dr, 1e-6) > 0.15:
        score -= 30; hints.append("FACE THE CAMERA")
    return max(score, 0), (hints or ["SIGNAL GOOD"])


def compute_index(acne, dc_eyes, wrinkles, shine):
    pen = (min(ACNE_CAP, ACNE_PTS * acne) + EYE_PTS * dc_eyes
           + min(WRINKLE_CAP, WRINKLE_PTS * wrinkles)
           + min(SHINE_CAP, max(0.0, shine - SHINE_FREE) * SHINE_PTS))
    return max(0.0, 100.0 - pen)


def make_verdict(S):
    if S["quality"] < 50:
        return ["SIGNAL TOO WEAK", S["hints"][0]]
    out = []
    if S["acne"] == 0:
        out.append("NO BLEMISHES DETECTED")
    else:
        z = max(S["acne_by_zone"], key=S["acne_by_zone"].get)
        out.append(f"{S['acne']} BLEMISHES, MOST ON {LABEL[z]}")
    dc = S["dc_l"] + S["dc_r"]
    if dc == 2:
        out.append("DARK CIRCLES: BOTH EYES")
    elif dc == 1:
        out.append("DARK CIRCLE: " + ("L" if S["dc_l"] else "R") + " EYE")
    else:
        out.append("UNDER-EYE AREA CLEAR")
    if S["wrinkles"]:
        out.append(f"{S['wrinkles']} WRINKLE LINES")
    if S["shine"] > SHINE_FREE + 3:
        out.append("T-ZONE SHINE ELEVATED")
    return out[:4]


# ----------------------------------------------------------------------------------
# Face landmarks (MediaPipe Tasks API, works with old and new mediapipe versions)
# ----------------------------------------------------------------------------------
class FaceTracker:
    def __init__(self, model_path="face_landmarker.task"):
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision
        self.mp = mp
        if not os.path.exists(model_path):
            print("Downloading face landmark model (one time, about 4 MB)...")
            try:
                urllib.request.urlretrieve(MODEL_URL, model_path)
            except Exception as e:
                sys.exit(f"Could not download {MODEL_URL}\n({e})\n"
                         f"Download it manually and save it as '{model_path}' next to this script.")
        opts = vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.VIDEO, num_faces=1,
            min_face_detection_confidence=0.5, min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5)
        self.lm = vision.FaceLandmarker.create_from_options(opts)
        self.t0, self.last = time.perf_counter(), -1

    def process(self, bgr):
        hh, ww = bgr.shape[:2]
        src = bgr if ww <= 640 else cv2.resize(bgr, (640, int(hh * 640 / ww)), interpolation=cv2.INTER_AREA)
        rgb = np.ascontiguousarray(cv2.cvtColor(src, cv2.COLOR_BGR2RGB))
        img = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb)
        ts = max(int((time.perf_counter() - self.t0) * 1000), self.last + 1)
        self.last = ts
        res = self.lm.detect_for_video(img, ts)
        if not res.face_landmarks:
            return None
        h, w = bgr.shape[:2]
        return np.array([(p.x * w, p.y * h) for p in res.face_landmarks[0]], np.float32)


# ----------------------------------------------------------------------------------
# Background detector so the video stays smooth while YOLO runs
# ----------------------------------------------------------------------------------
class Detector(threading.Thread):
    def __init__(self, model, imgsz=640, conf=0.40, device=None):
        super().__init__(daemon=True)
        self.model, self.imgsz, self.conf, self.device = model, imgsz, conf, device
        self.lock, self.ev = threading.Lock(), threading.Event()
        self.job, self.busy, self.alive = None, False, True
        self.result, self.error = None, None

    def submit(self, roi, zones_s, scale):
        with self.lock:
            if self.busy or self.job is not None:
                return False
            self.job = (roi.copy(), zones_s, scale)
        self.ev.set()
        return True

    def run(self):
        while self.alive:
            self.ev.wait(0.5)
            self.ev.clear()
            with self.lock:
                job, self.job = self.job, None
                if job is None:
                    continue
                self.busy = True
            try:
                self.result = self.infer(*job)
            except Exception as e:           # keep the video running, show the error later
                self.error = repr(e)
            finally:
                self.busy = False

    def infer(self, roi, zones_s, scale):
        h, w = roi.shape[:2]
        hs, ws = next(iter(zones_s.values())).shape
        kw = dict(imgsz=self.imgsz, conf=self.conf, iou=0.5, verbose=False)
        if self.device is not None:
            kw["device"] = self.device
        t0 = time.time()
        r = self.model.predict(roi, **kw)[0]
        counts = {z: {c: 0 for c in NAMES} for z in ZONE_ORDER}
        boxes, dropped = [], []
        for b in r.boxes:
            cname = NAMES[int(b.cls)]
            x1, y1, x2, y2 = [float(v) for v in b.xyxy[0].cpu().numpy()]
            cx = int(np.clip((x1 + x2) / 2 * scale, 0, ws - 1))
            cy = int(np.clip((y1 + y2) / 2 * scale, 0, hs - 1))
            zone = next((z for z in ZONE_ORDER if zones_s[z][cy, cx]), None)
            # drop detections outside the face, and dark circles anywhere except under the eyes
            if zone is None or (cname == "Dark Circle" and not zone.startswith("under_eye")):
                dropped.append(dict(cls=cname, conf=float(b.conf), box=(x1 / w, y1 / h, x2 / w, y2 / h)))
                continue
            counts[zone][cname] += 1
            boxes.append(dict(cls=cname, conf=float(b.conf), zone=zone,
                              box=(x1 / w, y1 / h, x2 / w, y2 / h)))
        return dict(boxes=boxes, dropped=dropped, counts=counts, t=time.time(),
                    ms=int((time.time() - t0) * 1000))


# ----------------------------------------------------------------------------------
# Drawing
# ----------------------------------------------------------------------------------
def put(img, text, org, scale=0.5, color=CYAN, thick=1):
    cv2.putText(img, text, (int(org[0]), int(org[1])), FONT, scale, color, thick, cv2.LINE_AA)


def tag(img, text, org, color=CYAN, scale=0.45):
    (tw, th), _ = cv2.getTextSize(text, FONT, scale, 1)
    x, y = int(org[0]), int(org[1])
    cv2.rectangle(img, (x - 3, y - th - 4), (x + tw + 3, y + 3), (12, 10, 10), -1)
    put(img, text, (x, y), scale, color)


def brackets(img, x0, y0, x1, y1, color, L=26, t=2):
    for x, y, dx, dy in [(x0, y0, 1, 1), (x1, y0, -1, 1), (x0, y1, 1, -1), (x1, y1, -1, -1)]:
        cv2.line(img, (x, y), (x + dx * L, y), color, t, cv2.LINE_AA)
        cv2.line(img, (x, y), (x, y + dy * L), color, t, cv2.LINE_AA)


def scan_sweep(frame, box, oval, t):
    x0, y0, x1, y1 = box
    sub = frame[y0:y1, x0:x1]
    h, w = sub.shape[:2]
    if h < 20 or w < 20:
        return
    mask = np.zeros((h, w), np.uint8)
    cv2.fillPoly(mask, [(oval - [x0, y0]).astype(np.int32)], 255)
    y = int(((t * 0.55) % 1.0) * h)
    r0, r1 = max(0, y - 45), min(h, y + 1)
    if r1 > r0:
        rows = np.arange(r0, r1, dtype=np.float32)
        ramp = np.clip(1.0 - (y - rows) / 45.0, 0, 1)
        alpha = (ramp[:, None] * (mask[r0:r1].astype(np.float32) / 255.0) * 0.40)[..., None]
        seg = sub[r0:r1]
        seg[:] = (seg.astype(np.float32) * (1 - alpha) + np.array(CYAN, np.float32) * alpha).astype(np.uint8)
    line = np.zeros((h, w), np.uint8)
    cv2.line(line, (0, y), (w - 1, y), 255, 1)
    sub[cv2.bitwise_and(line, mask) > 0] = WHITE


def bar(p, x, y, w, h, frac, color):
    cv2.rectangle(p, (x, y), (x + w, y + h), (60, 55, 50), 1)
    fw = int((w - 2) * float(np.clip(frac, 0, 1)))
    if fw > 0:
        cv2.rectangle(p, (x + 1, y + 1), (x + 1 + fw, y + h - 1), color, -1)


def make_panel(h, S):
    """Side panel, drawn at a fixed 720 px height and resized to fit the camera frame."""
    H = 720
    p = np.full((H, PANEL_W, 3), (18, 14, 12), np.uint8)
    cv2.line(p, (0, 0), (0, H), (90, 80, 30), 1)
    put(p, "DERMA-SCAN", (16, 36), 0.85, CYAN, 2)
    put(p, "REAL-TIME SKIN ANALYSIS", (16, 58), 0.4, DIM)

    if not S["face"]:
        put(p, "AWAITING SUBJECT", (16, 120), 0.6, AMBER, 1)
        put(p, "Look at the camera", (16, 150), 0.45, DIM)
    else:
        idx = S["index"]
        col = GREEN if idx >= 75 else AMBER if idx >= 50 else RED
        if S["quality"] < 50:
            col = DIM
        put(p, "SKIN INDEX", (16, 100), 0.45, DIM)
        num = f"{idx:.0f}"
        put(p, num, (16, 165), 2.3, col, 3)
        put(p, "/100", (16 + cv2.getTextSize(num, FONT, 2.3, 3)[0][0] + 8, 165), 0.6, DIM)
        bar(p, 16, 180, 308, 9, idx / 100, col)
        if S["quality"] < 50:
            note = "LOW CONFIDENCE"
        elif S["baseline"] is not None:
            note = f"{idx - S['baseline']:+.0f} VS BASELINE"
        else:
            note = "EXPERIMENTAL SCORE (B = SET BASELINE)"
        put(p, note, (16, 208), 0.4, DIM)

        put(p, "FINDINGS", (16, 252), 0.5, CYAN, 1)
        rows = [("ACNE", str(S["acne"]), S["acne"] / 15, MAGENTA),
                ("DARK CIRCLES", ("L" if S["dc_l"] else "-") + " / " + ("R" if S["dc_r"] else "-"),
                 (S["dc_l"] + S["dc_r"]) / 2, AMBER),
                ("WRINKLES", str(S["wrinkles"]), S["wrinkles"] / 10, GREEN)]
        y = 282
        for name, val, frac, c in rows:
            put(p, name, (16, y), 0.5, WHITE)
            put(p, val, (230, y), 0.5, c, 1)
            bar(p, 16, y + 8, 308, 6, frac, c)
            y += 36

        put(p, "SURFACE", (16, 400), 0.5, CYAN, 1)
        put(p, "T-ZONE SHINE", (16, 430), 0.5, WHITE)
        put(p, f"{S['shine']:.1f}%", (230, 430), 0.5, AMBER)
        bar(p, 16, 438, 308, 6, S["shine"] / 10, AMBER)
        put(p, "CHEEK REDNESS", (16, 466), 0.5, WHITE)
        put(p, f"{S['red']:+.1f} a*", (230, 466), 0.5, RED)
        bar(p, 16, 474, 308, 6, (S["red"] + 5) / 10, RED)

        qc = GREEN if S["quality"] >= 75 else AMBER if S["quality"] >= 50 else RED
        put(p, "SIGNAL", (16, 520), 0.5, CYAN, 1)
        put(p, f"{S['quality']:.0f}%", (230, 520), 0.5, qc)
        bar(p, 16, 528, 308, 6, S["quality"] / 100, qc)
        put(p, S["hints"][0], (16, 552), 0.42, qc)

        put(p, "VERDICT", (16, 592), 0.5, CYAN, 1)
        y = 616
        for line in S["verdict"][:3]:
            put(p, "> " + line, (16, y), 0.42, WHITE)
            y += 22
    put(p, f"{S['fps']:.0f} FPS", (262, 36), 0.4, DIM)
    put(p, S.get("diag", ""), (16, 684), 0.38, AMBER if S.get("debug") else DIM)
    put(p, "B base S save M mesh D debug [ ] conf Q quit", (16, 704), 0.36, DIM)
    return cv2.resize(p, (max(120, int(PANEL_W * h / H)), h), interpolation=cv2.INTER_AREA)


def render(frame, S):
    H, W = frame.shape[:2]
    if S["face"]:
        x0, y0, x1, y1 = S["box"]
        zt = S["zone_total"]
        ov = frame.copy()
        for z, cs in S["contours"].items():
            if z != "other_face" and cs:
                cv2.fillPoly(ov, cs, AMBER if zt.get(z, 0) else CYAN)
        cv2.addWeighted(ov, 0.16, frame, 0.84, 0, frame)
        scan_sweep(frame, S["box"], S["oval"], S["t"])
        for z, cs in S["contours"].items():
            if z != "other_face" and cs:
                cv2.polylines(frame, cs, True, AMBER if zt.get(z, 0) else CYAN, 1, cv2.LINE_AA)
        if S["mesh"]:
            for q in S["pts"][::4]:
                cv2.circle(frame, (int(q[0]), int(q[1])), 1, CYAN, -1)
        rw, rh = x1 - x0, y1 - y0
        for d in S["dets"]:
            u1, v1, u2, v2 = d["box"]
            cv2.rectangle(frame, (int(x0 + u1 * rw), int(y0 + v1 * rh)),
                          (int(x0 + u2 * rw), int(y0 + v2 * rh)), CLS_COL[d["cls"]], 1, cv2.LINE_AA)
        if S.get("debug"):
            for d in S.get("dropped", []):
                u1, v1, u2, v2 = d["box"]
                cv2.rectangle(frame, (int(x0 + u1 * rw), int(y0 + v1 * rh)),
                              (int(x0 + u2 * rw), int(y0 + v2 * rh)), DIM, 1, cv2.LINE_AA)
        for z, cs in S["contours"].items():
            if z == "other_face" or not cs or not zt.get(z, 0):
                continue
            m = cv2.moments(max(cs, key=cv2.contourArea))
            if m["m00"] > 0:
                tag(frame, f"{LABEL[z]} {zt[z]}", (m["m10"] / m["m00"] - 30, m["m01"] / m["m00"]), AMBER, 0.4)
        o = S["oval"]
        ox0, oy0 = o.min(axis=0)
        ox1, oy1 = o.max(axis=0)
        brackets(frame, int(ox0) - 12, int(oy0) - 12, int(ox1) + 12, int(oy1) + 12, CYAN)
        tag(frame, "TARGET LOCKED", (ox0 - 12, oy0 - 20), CYAN)
    else:
        pulse = 0.5 + 0.5 * np.sin(S["t"] * 4)
        c = tuple(int(v * (0.4 + 0.6 * pulse)) for v in AMBER)
        put(frame, "NO FACE DETECTED", (W // 2 - 150, H // 2), 0.9, c, 2)
        put(frame, "ALIGN FACE WITH CAMERA", (W // 2 - 130, H // 2 + 32), 0.55, DIM)
    put(frame, "DERMA-SCAN // LIVE", (14, 26), 0.6, CYAN, 2)
    cv2.circle(frame, (W - 20, 20), 5, RED if int(S["t"] * 2) % 2 == 0 else (40, 40, 90), -1)
    return np.hstack([frame, make_panel(H, S)])


# ----------------------------------------------------------------------------------
# Main loop
# ----------------------------------------------------------------------------------
def open_camera(src, w, h):
    """src is a camera number (0, 1, ...) or a stream URL from a phone IP-camera app."""
    src = str(src)
    if not src.isdigit():
        cap = cv2.VideoCapture(src)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)       # keep the stream live, not delayed
        return cap
    idx = int(src)
    cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW) if sys.platform.startswith("win") else cv2.VideoCapture(idx)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    return cap


def main():
    ap = argparse.ArgumentParser(description="DERMA-SCAN real-time skin scanner")
    ap.add_argument("--model", default="best.pt", help="your trained YOLOv8 weights")
    ap.add_argument("--cam", default="0", help="camera number, or a phone stream URL such as http://192.168.1.5:8080/video")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--imgsz", type=int, default=800)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--device", default=None, help="cpu, or 0 for the first GPU")
    ap.add_argument("--image", default=None, help="analyse a still photo instead of the webcam")
    ap.add_argument("--no-mirror", action="store_true", help="do not flip the camera image")
    args = ap.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        sys.exit("ultralytics is not installed: pip install -r requirements.txt")
    try:
        import torch
        torch.set_num_threads(max(2, (os.cpu_count() or 4) // 2))
    except Exception:
        pass
    model = YOLO(args.model)
    mnames = [model.names[i] for i in sorted(model.names)]
    if mnames != NAMES:
        print(f"WARNING: model classes {mnames} differ from expected {NAMES}")

    tracker = FaceTracker()
    detector = Detector(model, args.imgsz, args.conf, args.device)
    detector.start()
    static, cap = None, None
    if args.image:
        static = cv2.imread(args.image)
        if static is None:
            sys.exit(f"Could not read the image: {args.image}")
        k0 = min(1.0, 1280 / max(static.shape[:2]))
        if k0 < 1:
            static = cv2.resize(static, None, fx=k0, fy=k0, interpolation=cv2.INTER_AREA)
    else:
        cap = open_camera(args.cam, args.width, args.height)
        if not cap.isOpened():
            sys.exit("Could not open the camera. Try --cam 1 or 2, close other apps using it, "
                     "or check the stream URL.")

    os.makedirs("skin_scans", exist_ok=True)
    cv2.namedWindow("DERMA-SCAN", cv2.WINDOW_NORMAL)
    hist = deque(maxlen=5)
    last_ts, idx_ema, baseline, fps = None, None, None, 15.0
    shine_ema, red_ema, mesh, lost_since, debug = 0.0, 0.0, True, None, False
    t_prev, snap = time.time(), None

    try:
        while True:
            if static is not None:
                ok, frame = True, static.copy()
                time.sleep(0.03)
            else:
                ok, frame = cap.read()
                if not ok:
                    break
                if not args.no_mirror:
                    frame = cv2.flip(frame, 1)    # mirror view
            H, W = frame.shape[:2]
            now = time.time()
            fps = 0.9 * fps + 0.1 / max(now - t_prev, 1e-3)
            t_prev = now
            dres = detector.result
            diag = ("DETECTOR WARMING UP" if dres is None else
                    f"RAW {len(dres['boxes']) + len(dres['dropped'])}  KEPT {len(dres['boxes'])}  "
                    f"{dres['ms']} MS  CONF {detector.conf:.2f}")
            S = dict(face=False, t=now, mesh=mesh, fps=fps, baseline=baseline,
                     debug=debug, dropped=[], diag=diag)

            pts = tracker.process(frame)
            if pts is not None:
                lost_since = None
                box = face_roi(pts, frame.shape)
                x0, y0, x1, y1 = box
                rw, rh = x1 - x0, y1 - y0
                if rw > 40 and rh > 40:
                    scale = SMALL_W / rw
                    roi = frame[y0:y1, x0:x1]
                    roi_s = cv2.resize(roi, (SMALL_W, max(1, int(rh * scale))), interpolation=cv2.INTER_AREA)
                    zones_s = build_zones(roi_s.shape, (pts - [x0, y0]) * scale)
                    met, L_face = zone_metrics(roi_s, zones_s)
                    quality, hints = assess_quality(L_face, pts, W)
                    detector.submit(roi, zones_s, scale)

                    res = detector.result
                    if res is not None and res["t"] != last_ts:
                        last_ts = res["t"]
                        c = res["counts"]
                        hist.append((sum(c[z]["Acnes"] for z in ZONE_ORDER),
                                     sum(c[z]["Wrinkle"] for z in ZONE_ORDER),
                                     int(c["under_eye_left"]["Dark Circle"] > 0),
                                     int(c["under_eye_right"]["Dark Circle"] > 0)))
                    fresh = res is not None and now - res["t"] < 3.0
                    if hist:
                        med = np.median(np.array(hist), axis=0)
                        acne, wrinkles, dc_l, dc_r = int(med[0]), int(med[1]), int(med[2] > 0.5), int(med[3] > 0.5)
                    else:
                        acne = wrinkles = dc_l = dc_r = 0
                    counts = res["counts"] if fresh else {z: {c_: 0 for c_ in NAMES} for z in ZONE_ORDER}
                    shine = np.mean([met["forehead"]["shine"], met["nose"]["shine"]])
                    red = np.mean([met["cheek_left"]["red"], met["cheek_right"]["red"]])
                    shine_ema += 0.2 * (shine - shine_ema)
                    red_ema += 0.2 * (red - red_ema)
                    index = compute_index(acne, dc_l + dc_r, wrinkles, shine_ema)
                    idx_ema = index if idx_ema is None else idx_ema + 0.15 * (index - idx_ema)
                    S.update(face=True, pts=pts, box=box, oval=pts[OVAL],
                             contours=zone_contours(zones_s, scale, x0, y0),
                             dets=res["boxes"] if fresh else [],
                             dropped=res["dropped"] if fresh else [],
                             zone_total={z: sum(counts[z].values()) for z in ZONE_ORDER},
                             acne=acne, wrinkles=wrinkles, dc_l=dc_l, dc_r=dc_r,
                             acne_by_zone={z: counts[z]["Acnes"] for z in ZONE_ORDER},
                             shine=shine_ema, red=red_ema, index=idx_ema,
                             quality=quality, hints=hints)
                    S["verdict"] = make_verdict(S)
            if not S["face"]:
                lost_since = lost_since or now
                if now - lost_since > 1.0:
                    hist.clear(); idx_ema = None

            canvas = render(frame, S)
            if detector.error:
                put(canvas, "DETECTOR ERROR: " + detector.error[:70], (14, canvas.shape[0] - 14), 0.45, RED)
            cv2.imshow("DERMA-SCAN", canvas)
            snap = (canvas, S)

            k = cv2.waitKey(1) & 0xFF
            if k in (ord("q"), 27):
                break
            if k == ord("m"):
                mesh = not mesh
            if k == ord("d"):
                debug = not debug
            if k == ord("]"):
                detector.conf = min(0.90, round(detector.conf + 0.05, 2))
            if k == ord("["):
                detector.conf = max(0.05, round(detector.conf - 0.05, 2))
            if k == ord("b") and S["face"]:
                baseline = S["index"]
            if k == ord("s") and S["face"]:
                stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                cv2.imwrite(f"skin_scans/scan_{stamp}.jpg", canvas)
                new = not os.path.exists("skin_scans/log.csv")
                with open("skin_scans/log.csv", "a", newline="") as f:
                    w = csv.writer(f)
                    if new:
                        w.writerow(["time", "index", "acne", "dark_circle_L", "dark_circle_R",
                                    "wrinkles", "tzone_shine_pct", "cheek_redness_a", "signal"])
                    w.writerow([stamp, round(S["index"], 1), S["acne"], S["dc_l"], S["dc_r"],
                                S["wrinkles"], round(S["shine"], 2), round(S["red"], 2), S["quality"]])
                print("Saved scan", stamp)
    finally:
        detector.alive = False
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()