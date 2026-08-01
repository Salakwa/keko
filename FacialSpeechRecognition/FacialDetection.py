import sys 
import os
import cv2 


def LoadFacialDetection(): 
    """
    Loads OpenCV's pretrained Haar Cascade face detector.
    This XML file ships with opencv-python, so no extra download is needed.
    """
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)
 
    if face_cascade.empty():
        raise IOError(f"Failed to load Haar Cascade from {cascade_path}")
 
    return face_cascade



def DetectFaces(frame, face_cascade, draw=True): 
    """
    Detects faces in a single BGR frame.
 
    Args:
        frame: The image/frame (as returned by camera.read()).
        face_cascade: A loaded cv2.CascadeClassifier.
        draw: If True, draws rectangles around detected faces on the frame.
 
    Returns:
        faces: List of (x, y, w, h) bounding boxes for each detected face.
        frame: The (optionally annotated) frame.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
 
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,      # how much the image size is reduced at each scale
        minNeighbors=5,       # higher = fewer false positives
        minSize=(30, 30)      # ignore detections smaller than this
    )
 
    if draw:
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(
                frame,
                "Face",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )
 
    return faces, frame