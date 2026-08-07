import os
import cv2 
import urllib.request
import mediapipe as mp 
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision 
import numpy as np 

 
# --- Landmark index groups for different facial regions ---
# These indices are fixed by MediaPipe's 468-point face mesh topology.
REGIONS = {
    "left_eyebrow": [70, 63, 105, 66, 107],
    "right_eyebrow": [336, 296, 334, 293, 300],
    "left_eye": [33, 160, 158, 133, 153, 144],
    "right_eye": [362, 385, 387, 263, 373, 380],
    "mouth": [61, 291, 13, 14, 78, 308],
    "jaw": [152, 148, 176, 149, 150, 377],
}
 
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "face_landmarker.task")
 
 
def EnsureModelDownloaded():
    """
    Downloads the FaceLandmarker model file if it isn't already present
    next to this script. Only happens once.
    """
    if not os.path.exists(MODEL_PATH):
        print(f"Downloading face landmarker model to {MODEL_PATH} ...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Model download complete.")
    return MODEL_PATH


  
def LoadFacialMesh(max_num_faces=1, min_detection_confidence=0.5, min_tracking_confidence=0.5):
    """
    Initializes MediaPipe's FaceLandmarker.
 
    Returns:
        A FaceLandmarker object. Call .close() on it when done.
    """
    model_path = EnsureModelDownloaded()
 
    base_options = mp_python.BaseOptions(model_asset_path=model_path)
    options = mp_vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_faces=max_num_faces,
        min_face_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )
    face_landmarker = mp_vision.FaceLandmarker.create_from_options(options)
    return face_landmarker
 
 
def GetLandmarkPositions(landmarks, frame_width, frame_height):
    """
    Converts FaceLandmarker's normalized landmarks into a NumPy array
    of pixel coordinates, shape (478, 2).
    """
    points = np.array(
        [[lm.x * frame_width, lm.y * frame_height] for lm in landmarks]
    )
    return points
 
 
def TrackFacialMovement(frame, face_landmarker, timestamp_ms, prev_points=None,
                         movement_threshold=2.0, draw=True):
    """
    Detects face landmarks in a frame and measures movement per region
    by comparing against the previous frame's landmark positions.
 
    Args:
        frame: The current BGR frame.
        face_landmarker: A FaceLandmarker object from LoadFacialMesh().
        timestamp_ms: Monotonically increasing timestamp in milliseconds.
            Required by the Tasks API in VIDEO mode.
        prev_points: The (478, 2) landmark array from the previous frame,
            or None on the first frame / if no face was previously found.
        movement_threshold: Average pixel displacement above which a
            region is considered "moving."
        draw: If True, draws the mesh points and per-region movement labels.
 
    Returns:
        current_points: (478, 2) array of this frame's landmark positions,
            or None if no face was detected.
        movement: dict mapping region name -> average pixel displacement.
        frame: The (optionally annotated) frame.
    """
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
 
    result = face_landmarker.detect_for_video(mp_image, timestamp_ms)
 
    movement = {}
 
    if not result.face_landmarks:
        return None, movement, frame
 
    frame_height, frame_width = frame.shape[:2]
    landmarks = result.face_landmarks[0]  # first detected face
    current_points = GetLandmarkPositions(landmarks, frame_width, frame_height)
 
    if draw:
        for (x, y) in current_points:
            cv2.circle(frame, (int(x), int(y)), 1, (100, 100, 100), -1)
 
    if prev_points is not None:
        y_offset = 30
        for region_name, indices in REGIONS.items():
            current_region = current_points[indices]
            prev_region = prev_points[indices]
 
            displacements = np.linalg.norm(current_region - prev_region, axis=1)
            avg_displacement = float(np.mean(displacements))
            movement[region_name] = avg_displacement
 
            is_moving = avg_displacement > movement_threshold
 
            if draw:
                color = (0, 165, 255) if is_moving else (150, 150, 150)
                for (x, y) in current_region:
                    cv2.circle(frame, (int(x), int(y)), 2, color, -1)
 
                label = f"{region_name}: {avg_displacement:.1f}" + (" (moving)" if is_moving else "")
                cv2.putText(
                    frame, label, (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1
                )
                y_offset += 20
 
    return current_points, movement, frame