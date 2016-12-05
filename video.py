import cv2
from scipy import cluster
import numpy as np;

cap = cv2.VideoCapture(0)
#gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)



# Setup SimpleBlobDetector parameters.
params = cv2.SimpleBlobDetector_Params()

# Change thresholds
params.minThreshold = 10;
params.maxThreshold = 200;
 
# Filter by Circularity
params.filterByCircularity = True
params.minCircularity = 0.85
 
# Create a detector with the parameters
ver = (cv2.__version__).split('.')
if int(ver[0]) < 3 :
	detector = cv2.SimpleBlobDetector(params)
else : 
	detector = cv2.SimpleBlobDetector_create(params)


	
while(1):
    ret, frame = cap.read()

    fgmask = frame;

    # Detect blobs.
    keypoints = detector.detect(fgmask)
    # Draw detected blobs as red circles.
    # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
    im_with_keypoints = cv2.drawKeypoints(frame, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    #print keypoints

    cv2.imshow('frame',im_with_keypoints)

    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()	
	
