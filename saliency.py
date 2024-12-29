# import the necessary packages
from imutils.video import VideoStream
import imutils
import time
import cv2
import numpy as np
from framesPerSecond import FramesPerSecond
# initialize the motion saliency object and start the video stream
saliency = None
vs = VideoStream(src=0).start()
time.sleep(2.0)
fps = FramesPerSecond()

def null(x):
     pass

cv2.namedWindow('settings')
cv2.createTrackbar("d", "settings", 9, 25, null)
cv2.createTrackbar("sigmaColor", "settings", 75, 255, null)
cv2.createTrackbar("sigmaSpace", "settings", 75, 255, null)
cv2.createTrackbar("threshold", "settings", 5, 255, null)
cv2.setTrackbarMin("threshold", "settings", 1)
cv2.setTrackbarMin("d", "settings", 1)

cv2.namedWindow('capture areas')
cv2.createTrackbar("y1", "capture areas", 80, 640, null)
cv2.createTrackbar("y2", "capture areas", 280, 640, null)
cv2.createTrackbar("x1", "capture areas", 150, 480, null)
cv2.createTrackbar("x2", "capture areas", 330, 480, null)

count = 0
total = 0
# loop over frames from the video file stream
while True:
	# grab the frame from the threaded video stream and resize it
	# to 500px (to speedup processing)
	frame = vs.read()
	frame = imutils.resize(frame, width=450)
	renderframe = frame.copy()


	frame1y1 = cv2.getTrackbarPos('y1','capture areas')
	cv2.setTrackbarMin("y2", "capture areas", frame1y1)
	frame1y2 = cv2.getTrackbarPos('y2','capture areas')
	cv2.setTrackbarMax("y1", "capture areas", frame1y2 - 1)

	frame1x1 = cv2.getTrackbarPos('x1','capture areas')
	cv2.setTrackbarMin("x2", "capture areas", frame1x1)
	frame1x2 = cv2.getTrackbarPos('x2','capture areas')
	cv2.setTrackbarMax("x1", "capture areas", frame1x2 -1)

	frame1 = frame[frame1x1:frame1x2, frame1y1:frame1y2]

	# frame2 = frame[80:280, 330:510]

	# if our saliency object is None, we need to instantiate it
	# if saliency is None:
	# 	saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
	# 	# saliency = cv2.saliency.MotionSaliencyBinWangApr2014_create()
	# 	# saliency.setImagesize(frame.shape[1], frame.shape[0])
	# 	# saliency.init()
    # convert the input frame to grayscale and compute the saliency
	# map based on the motion model
	bialteralD = cv2.getTrackbarPos('d','settings')
	bilateralSigmaColor = cv2.getTrackbarPos('sigmaColor','settings')
	bilateralSigmaSpace = cv2.getTrackbarPos('sigmaSpace','settings')
	treshholdManual = cv2.getTrackbarPos('threshold','settings')

	blur = cv2.bilateralFilter(frame1,bialteralD,bilateralSigmaColor,bilateralSigmaSpace)
	gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
	# (success, saliencyMap) = saliency.computeSaliency(gray)

	# ret, threshMap = cv2.threshold((saliencyMap * 255).astype('uint8'), treshholdManual, 255, cv2.THRESH_BINARY)
	# ret, threshMap = cv2.threshold((saliencyMap * 255).astype('uint8'), treshholdManual, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
	# saliencyMap = (saliencyMap * 255).astype("uint8")


	ret2, threshMap2 = cv2.threshold(gray, treshholdManual, 255, 0)
	inverted = cv2.bitwise_not(threshMap2)
	contours, hierarchy = cv2.findContours(inverted, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	# mean = np.mean(saliencyMap, axis=(0,1))
	# std = np.std(saliencyMap, axis=(0,1))
	# mask = (np.abs(saliencyMap - mean) / std >= 4.5).any(axis=1)
	# mask_u8 = mask.astype(np.uint8) * 255
	

	# proportion = np.mean(saliencyMap)
	proportion = (cv2.countNonZero(inverted) / threshMap2.size) * 100 #np.mean(saliencyMap)
	count += 1
	total += proportion
	average = total / count
	proportion = "Proportion: "+str(np.round(average, 2))+"%"

	cv2.drawContours(renderframe, contours, -1, (0,255,0), 3, None, None, None, (frame1y1, frame1x1))
	cv2.putText(renderframe, proportion, (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
	cv2.rectangle(renderframe, (frame1y1, frame1x1), (frame1y2, frame1x2), (255, 0, 0), 1)
	fpsFrame = fps.applyFrameRate(renderframe)
	# display the image to our screen
	cv2.imshow("Frame", renderframe)
	# cv2.imshow("Map", saliencyMap)
	# cv2.imshow("Treshmap", threshMap)

	# cv2.imshow("Treshmap2", threshMap2)

	# cv2.imshow("blur", blur)
	key = cv2.waitKey(1) & 0xFF
 
	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

	if key == ord("r"):
		count = 0
		total = 0

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()