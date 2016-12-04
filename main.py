# Standard imports
import cv2
from scipy import cluster
import numpy as np;
 
""""Read image"""
im = cv2.imread("3.jpg", cv2.IMREAD_GRAYSCALE)

def blob_detect(): 
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

	# Detect blobs.
	keypoints = detector.detect(im)
	return keypoints


# Get X,Y cordinates in Numpy Array
def cordinates(keypoints=blob_detect()):
	n = len(keypoints)
	cordinates = np.zeros(shape=(n,2))
	for points in range(len(keypoints)):
		cordinates[points] = keypoints[points].pt
	return cordinates

def dice_ids(data=cordinates()):
	centroids,_ = cluster.vq.kmeans(obs=data, k_or_guess = 2, iter = 2, thresh = 1e-05, check_finite = True)
	#assign sample to cluster
	ids,_ = cluster.vq.vq(data, centroids)
	return ids

def plt_dice(keypoints=blob_detect()):
	# Draw detected blobs as red circles.
	# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
	im_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
	cv2.imshow("Keypoints", im_with_keypoints)
	cv2.waitKey(0)

def dice_value(values = dice_ids()):
	dice1, dice2 = 0, 0
	for index in range(len(values)):
		if values[index] == 0:
			dice1 += 1
		elif values[index] == 1:
			dice2 += 1
		elif values[index] != 0 or values[index] != 1:
			print "unknown values detected"
	return dice1, dice2

	
a, b = dice_value()
print 'value dice 1 =', a
print 'value dice 2 =', b

plt_dice()
