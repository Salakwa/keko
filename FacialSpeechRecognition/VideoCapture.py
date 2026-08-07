import sys
import cv2   
import time

#File Imports 
import FacialDetection 
import FacialMesh


# Initialize Default Camera 
camera = cv2.VideoCapture(0)   
if camera is None or not camera.isOpened():
    raise ValueError("Device has No Camera Detected.") 
print("Camera successfully detected and connected.")
   

# Get Camera Specs (Default frame width and height)
frame_width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT)) 

# Load in the Facial Detection 
face_cascade = FacialDetection.LoadFacialDetection() 

# Load in & Declare Vars for Facial Mesh 
face_landmarker = FacialMesh.LoadFacialMesh()
prev_points = None
start_time = time.time()

try:
    while True:
        ret, frame = camera.read() 
        if not ret:  
            raise RuntimeError("Camera Disconnected Unexpectedly.") 

        # Run Frame through Facial Detection 
        faces, frame = FacialDetection.DetectFaces(frame, face_cascade, draw=True) 

        # Run Frame through Mesh Logic 
        timestamp_ms = int((time.time() - start_time) * 1000)

        prev_points, movement, frame = FacialMesh.TrackFacialMovement(
            frame, face_landmarker, timestamp_ms,
            prev_points=prev_points, movement_threshold=2.0, draw=True
        )

        # Display the Captured Frames Live
        cv2.imshow('Live Camera Feed', frame)

        # Press 'q' to exit the loop
        if cv2.waitKey(1) == ord('q'): 
            print("Live Camera Feed Ended by User")
            break
except Exception as e:
    print(f"UNEXPECTED CRASH: {e}")
finally:
    # Release the capture and writer objects
    camera.release()
    cv2.destroyAllWindows() 
    face_landmarker.close()