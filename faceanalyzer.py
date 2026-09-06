
import argparse
import threading
import time
from collections import deque
 
import cv2
import numpy as np
from deepface import DeepFace
 
 
def parse_args():
    parser = argparse.ArgumentParser(description="Face Analyzer")
    parser.add_argument(
        "--emotion",
        type=str,
        default="happy",
        choices=["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"],
        help="Emotion to detect",
    )
    parser.add_argument("--image", required=True, help="Path to the input image")
    parser.add_argument("--threshold", type=float, default=50.0)
    parser.add_argument("--hold-frames", type=int, default=3)
    parser.add_argument(
        "--detector",
        type=str,
        default="mtcnn",
        choices=["opencv", "mtcnn", "mediapipe", "ssd", "yunet"],
        help="Face detector backend. mediapipe/opencv/yunet are fast; mtcnn is slow.",
    )
    parser.add_argument(
        "--analysis-width",
        type=int,
        default=320,
        help="Frame is downscaled to this width before analysis (speeds up detection a lot).",
    )
    parser.add_argument("--camera-index", type=int, default=None)
    parser.add_argument("--capture-width", type=int, default=640)
    parser.add_argument("--capture-height", type=int, default=480)
    return parser.parse_args()
 
 
def load_overlay_image(path, target_height):
    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Image not found at {path}")
    h, w = image.shape[:2]
    scale = target_height / h
    new_w, new_h = int(w * scale), target_height
    return cv2.resize(image, (new_w, new_h))
 
 
def overlay_transparent(background, overlay, x, y):
    bg_h, bg_w = background.shape[:2]
    ov_h, ov_w = overlay.shape[:2]
    if x >= bg_w or y >= bg_h:
        return background
 
    ov_w = min(ov_w, bg_w - x)
    ov_h = min(ov_h, bg_h - y)
    overlay = overlay[:ov_h, :ov_w]
 
    if overlay.shape[2] == 4:
        alpha = overlay[:, :, 3] / 255.0
        for c in range(3):
            background[y:y + ov_h, x:x + ov_w, c] = (
                alpha * overlay[:, :, c]
                + (1 - alpha) * background[y:y + ov_h, x:x + ov_w, c]
            )
    else:
        background[y:y + ov_h, x:x + ov_w] = overlay
    return background
 
 
def open_real_camera(forced_index=None, max_index=4, warmup_frames=15, min_mean=3.0,
                      capture_width=640, capture_height=480):
    candidates = [forced_index] if forced_index is not None else range(max_index)
 
    for idx in candidates:
        cap = cv2.VideoCapture(idx, cv2.CAP_AVFOUNDATION)
        if not cap.isOpened():
            cap.release()
            continue
 
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, capture_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, capture_height)
 
        print(f"Testing camera index {idx}...")
        best_mean = 0.0
        got_real_frame = False
 
        for _ in range(warmup_frames):
            ret, frame = cap.read()
            if ret and frame is not None:
                mean_val = float(np.mean(frame))
                best_mean = max(best_mean, mean_val)
                if mean_val >= min_mean:
                    got_real_frame = True
                    break
            time.sleep(0.1)
 
        if got_real_frame:
            print(f"Using camera index {idx} (mean pixel value {best_mean:.2f}).")
            return cap
 
        print(f"Camera index {idx} returned only black/empty frames -- skipping.")
        cap.release()
 
    return None
 
 
class EmotionAnalyzer:
    """
    Runs DeepFace.analyze on a background thread so the main video loop
    never blocks waiting for it. The main loop just reads whatever the
    latest result is
    """
 
    def __init__(self, detector_backend, analysis_width):
        self.detector_backend = detector_backend
        self.analysis_width = analysis_width
        self.lock = threading.Lock()
        self.latest_frame = None
        self.last_emotion_label = "unknown"
        self.last_confidence = 0.0
        self._latest_emotions = {}
        self.running = True
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()
 
    def submit_frame(self, frame):
        with self.lock:
            self.latest_frame = frame
 
    def _worker(self):
        while self.running:
            with self.lock:
                frame = self.latest_frame
                self.latest_frame = None  
 
            if frame is None:
                time.sleep(0.01)
                continue
 
            # Downscale for faster detection/analysis.
            h, w = frame.shape[:2]
            scale = self.analysis_width / w
            small = cv2.resize(frame, (self.analysis_width, int(h * scale)))
 
            try:
                result = DeepFace.analyze(
                    small, actions=["emotion"], enforce_detection=False,
                    silent=True, detector_backend=self.detector_backend,
                )
                if isinstance(result, list):
                    if not result:
                        raise ValueError("No face detected")
                    result = result[0]
                emotions = result.get("emotion", {})
                if not emotions:
                    raise ValueError("No emotion data")
 
                label = max(emotions, key=emotions.get)
                confidence = float(emotions[label])
                with self.lock:
                    self.last_emotion_label = label
                    self.last_confidence = confidence
                    self._latest_emotions = emotions
            except Exception as e:
                print("Analyzer error:", e)
                with self.lock:
                    self.last_emotion_label = "unknown"
                    self.last_confidence = 0.0
                    self._latest_emotions = {}
 
    def get_target_confidence(self, target_emotion):
        with self.lock:
            return (
                float(self._latest_emotions.get(target_emotion, 0.0)),
                self.last_emotion_label,
                self.last_confidence,
            )
 
    def stop(self):
        self.running = False
 
 
def main():
    args = parse_args()
 
    cap = open_real_camera(
        forced_index=args.camera_index,
        capture_width=args.capture_width,
        capture_height=args.capture_height,
    )
    if cap is None:
        print("Error: No camera produced a real (non-black) frame.")
        print("Try turning off Continuity Camera on a nearby iPhone and rerun.")
        return
 
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or args.capture_width
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or args.capture_height
    print(f"Frame size: {frame_width} x {frame_height}")
    print(f"Detector backend: {args.detector} (analysis width {args.analysis_width}px)")
 
    overlay_img = load_overlay_image(args.image, target_height=frame_height)
    canvas_w = frame_width + overlay_img.shape[1]
 
    analyzer = EmotionAnalyzer(args.detector, args.analysis_width)
    hits = deque(maxlen=args.hold_frames)
    frame_count = 0
    submit_every = 3  # hand off a frame for analysis every N displayed frames
 
    print(f"Watching for '{args.emotion}' (threshold {args.threshold}%). Press 'q' to quit.")
 
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Lost camera feed.")
                break
 
            frame_count += 1
            if frame_count % submit_every == 0:
                analyzer.submit_frame(frame.copy())
 
            target_confidence, label, confidence = analyzer.get_target_confidence(args.emotion)
            hits.append(target_confidence >= args.threshold)
            show_overlay = len(hits) == args.hold_frames and all(hits)
 
            canvas = np.zeros((frame_height, canvas_w, 3), dtype=np.uint8)
            canvas[:, :frame_width] = frame
 
            cv2.putText(canvas, f"{label} ({confidence:.0f}%)", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
 
            if show_overlay:
                canvas = overlay_transparent(canvas, overlay_img, frame_width, 0)
 
            cv2.imshow("Emotion Overlay", canvas)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        analyzer.stop()
        cap.release()
        cv2.destroyAllWindows()
 
 
if __name__ == "__main__":
    main()
 
