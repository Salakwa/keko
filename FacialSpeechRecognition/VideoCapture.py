import sys
import cv2 


# Initialize Default Camera 
camera = cv2.VideoCapture(0)   
if camera is None or not camera.isOpened():
    raise ValueError("Device has No Camera Detected.") 
print("Camera successfully detected and connected.")
   

# Get Camera Specs (Default frame width and height)
frame_width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))


try:
    while True:
        ret, frame = camera.read() 
        if not ret:  
            raise RuntimeError("Camera Disconnected Unexpectedly.")

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