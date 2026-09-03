import argparse
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
    parser.add_argument(
        "--image",
        required=True,
        help="Path to the input image",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=50.0
        ,
        help="Threshold for emotion detection",
    )
    parser.add_argument(
        "--hold-frames",
        type=int,
        default=3,
        help="Number of frames to hold the detected emotion",
    )
    parser.add_argument(
        "--analyze-every",
        type=int,
        default=5,
        help="Run the emotion analysis every N frames",
    )

    return parser.parse_args()

def load_image(path, target_height):
    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Image not found at {path}")
    h, w = image.shape[:2]
    scale = target_height / h
    new_w, new_h = int(w * scale), target_height
    return cv2.resize(image, (new_w, new_h))


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
        for c in range(0, 3):
            background[y:y + ov_h, x:x + ov_w, c] = (
                alpha * overlay[:, :, c]
                + (1 - alpha) * background[y:y + ov_h, x:x + ov_w, c]
            )
    else:
        background[y:y + ov_h, x:x + ov_w] = overlay
    return background
def main():
    args = parse_args()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    for _ in range(10):
        ret, _ = cap.read()
        if ret:
            break
        time.sleep(0.1)
    else:
        print("Error: Camera opened but never returned a frame.")
        cap.release()
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480

    print(f"Frame size: {frame_width} x {frame_height}")

    overlay_img = load_overlay_image(args.image, target_height=frame_height)
    canvas_w = frame_width + overlay_img.shape[1]
    hits = deque(maxlen=args.hold_frames)
    frame_count = 0
    last_emotion = "..."
    last_confidence = 0.0
    last_emotion_label = "unknown"
    print(f"Watching for '{args.emotion}' (threshold {args.threshold}%). Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
 
        frame_count += 1
        triggered_this_frame = False
 
        if frame_count % args.analyze_every == 0:
            try:
                result = DeepFace.analyze(
                    frame, actions=["emotion"], enforce_detection=False, silent=True,
                    detector_backend="mtcnn"
                )
                if isinstance(result, list):
                    if not result:
                        raise ValueError("No face detected")
                    result = result[0]
                emotions = result.get("emotion", {})
                if not emotions:
                    raise ValueError("No emotion data")
                last_emotion_label = max(emotions, key=emotions.get)
                last_confidence = float(emotions.get(last_emotion_label, 0.0))

                target_confidence = float(emotions.get(args.emotion, 0.0))
                triggered_this_frame = target_confidence >= args.threshold
            except Exception as e:
                print("DeepFace error:", e)
                triggered_this_frame = False
                last_emotion_label = "unknown"
                last_confidence = 0.0

            hits.append(triggered_this_frame)

        show_overlay = len(hits) == args.hold_frames and all(hits)
 
        # Build canvas: webcam frame + blank space (or overlay) to the side
        canvas = np.zeros((frame_height, canvas_w, 3), dtype=np.uint8)
        canvas[:, :frame_width] = frame
 
        label = f"{last_emotion_label} ({last_confidence:.0f}%)"
        cv2.putText(canvas, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
 
        if show_overlay:
            canvas = overlay_transparent(canvas, overlay_img, frame_width                                   , 0)
 
        cv2.imshow("Emotion Overlay", canvas)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
 
    cap.release()
    cv2.destroyAllWindows()
 
 
if __name__ == "__main__":
    main()
    

    
