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
            background[y:y+ov_h, x:x+ov_w, c] = (alpha * overlay[:, :, c] +
                                                 (1 - alpha) * background[y:y+ov_h, x:x+ov_w, c])
        else:
            background[y:y+ov_h, x:x+ov_w] = overlay
        return background
    
