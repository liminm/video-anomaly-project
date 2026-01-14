import cv2
import glob
import os
import numpy as np

# Path to your data
VIDEO_DIR = "data/UCSD_Anomaly_Dataset.v1p2/UCSDped2/Test/Test001"

def view_motion_mask():
    # MOG2 is a standard background subtraction algorithm
    # history=500: Learns the background over 500 frames
    # varThreshold=16: Sensitivity (Lower = detects more motion)
    fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=False)
    
    frame_paths = sorted(glob.glob(os.path.join(VIDEO_DIR, "*.tif")))
    
    for path in frame_paths:
        frame = cv2.imread(path)
        
        # 1. Extract Motion Mask
        # This turns static concrete BLACK (0) and moving people WHITE (255)
        fgmask = fgbg.apply(frame)
        
        # 2. Clean up noise (Optional "Opening" operation)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
        
        # 3. Apply Mask to Frame
        # The result is the person in color, floating in a black void
        masked_frame = cv2.bitwise_and(frame, frame, mask=fgmask)
        
        # Show side-by-side
        combined = np.hstack((frame, masked_frame))
        cv2.imshow('Left: Raw | Right: Motion Only', combined)
        
        if cv2.waitKey(30) & 0xFF == 27: # Press ESC to quit
            break
            
    cv2.destroyAllWindows()

if __name__ == "__main__":
    view_motion_mask()