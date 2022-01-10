from skimage.metrics import structural_similarity
import numpy as np
import cv2 as cv
import imutils
import time

	##captures the webcam (webcam 0 is the built-in one, 1 is the usb)
#cap = cv.VideoCapture('added bugs with light.mp4')
#cap = cv.VideoCapture('newCamTest2.avi')
cap = cv.VideoCapture(0)
if not cap.isOpened():
	print('can not open')
	exit()
firstFrame = None
lastDelta = None
##contourFrame = None
##startTime = round(time.time() * 1000)

	##continuously reads frames from the cam
while True:
	ret, frame = cap.read()
	if not ret:
		print('can not read frame')
		break
			##makes frame grayscale
	gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
	#cv.imshow('frame', gray)

			##exits program if q is pressed
	if cv.waitKey(1) == ord('q'):
		break
	
	if firstFrame is None:
		firstFrame = gray
		continue
			##gets frame difference between current frame and previous frame
	frameDelta = cv.absdiff(firstFrame, gray)
	if lastDelta is None:
		lastDelta = frameDelta
		continue
	
			##just testing what filters do
	##kernel = np.ones((5,5), np.float32)/25
	##dst = cv.filter2D(frameDelta, -1, kernel)
	##blur = cv.blur(frameDelta, (5, 5))
	##medianbl = cv.medianBlur(frameDelta, 5)
	##gaussbl = cv.GaussianBlur(frameDelta, (5, 5), 0)
	bilatFilter = cv.bilateralFilter(frameDelta, 9, 75, 75)

			##blob detection using sift
	sift = cv.SIFT_create()
	kp = sift.detect(bilatFilter, None)
	##kp = sift.detect(frameDelta, None)
	blobs = bilatFilter
	##blobs = frameDelta
	blobs = cv.drawKeypoints(bilatFilter, kp, blobs)
	##blobs = cv.drawKeypoints(frameDelta, kp, blobs)

			##blob detection using surf [NOT WORKING] surf algorithm patented
	##surf = cv.xfeatures2d.SURF_create(50000)
	##surf.setHessianThreshold(50000)
	##kp, des = surf.detectAndCompute(bilatFilter, None)
	##blobs = cv.drawKeypoints(bilatFilter, kp, None, (255, 0, 0), 4)


			##playing around with difference detection
	##(score, diff) = structural_similarity(lastDelta, frameDelta, full=True)
	##print("Image similarity", score)
	##diff = (diff * 255).astype("uint8")

	##thresh = cv.threshold(diff, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)[1]
	##contours = cv.findContours(thresh.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
	##contours = contours[0] if len(contours) == 2 else contours[1]
	##mask = np.zeros(lastDelta.shape, dtype='uint8')
	##filled_after = frameDelta.copy()
	##for c in contours:
		##area = cv.contourArea(c)
		##if area > 10:
			##x,y,w,h = cv.boundingRect(c)
			##cv.rectangle(frameDelta, (x, y), (x + w, y + h), (36,255,12), 2)
			##cv.rectangle(lastDelta, (x, y), (x + w, y + h), (36,255,12), 2)
			##cv.drawContours(mask, [c], 0, (0,255,0), -1)
			##cv.drawContours(filled_after, [c], 0, (0,255,0), -1)

	##bilatFilter = cv.resize(bilatFilter, blobs.shape)
	grayBlobs = cv.cvtColor(blobs, cv.COLOR_BGR2GRAY)
	(score, diff) = structural_similarity(grayBlobs, bilatFilter, full=True)
	print("Image similarity", score)
	##finalDif = cv.absdiff(lastDelta, frameDelta)
	##finalFrame = frameDelta

			##makes rectangle contours
	diff = (diff * 255).astype("uint8")
	thresh = cv.threshold(diff, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)[1]
	contours = cv.findContours(thresh.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
	contours = contours[0] if len(contours) == 2 else contours[1]
	contourFrama = bilatFilter
	for c in contours:
		area = cv.contourArea(c)
		if area > 40:
			x,y,w,h = cv.boundingRect(c)
			cv.rectangle(bilatFilter, (x, y), (x + w, y + h), (36,255,12), 2)
	
			##shows mesage if motion detected [NOT WORKING] not sure how to check if keypoints are present
	finalFrame = blobs
	##finalFrame = bilatFilter
	##finalFrame = frameDelta
	if score == 1.0 : finalFrame = cv.putText(blobs, 'no motion', (0, 60), cv.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 5, cv.LINE_AA)
	else: finalFrame = cv.putText(blobs, 'MOTION', (0, 60), cv.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 5, cv.LINE_AA)
	
			##video output
	##cv.imshow('bilateral filtering', bilatFilter)
	##cv.imshow('median blur', medianbl)
	cv.imshow('point detection', finalFrame)
	cv.imshow('point detection1', bilatFilter)
	firstFrame = gray
	lastDelta = frameDelta


cap.release()
cv.destroyAllWindows()
