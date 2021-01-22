# python contours.py --video beach.mp4 --out . --height 480


# Unresolved Bugs

import os
import numpy as np 
import cv2
import argparse
import time
import math
import matplotlib.pyplot as plt
from PIL import Image


def main(video, outpath, height):

	print("reading ", video)

	filename = os.path.splitext(os.path.basename(video))[0]
	# if not os.path.exists(outpath + "/" + filename):
	# 	os.makedirs(outpath + "/" + filename)

	# init video capture with video
	cap = cv2.VideoCapture(video)

	# get default video FPS
	fps = cap.get(cv2.CAP_PROP_FPS)

	# get total number of video frames
	num_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

	print(num_frames)

	# read the first frame
	ret, previous_frame = cap.read()

	# proceed if frame reading was successful
	if not ret: return

	# width after resize
	width = math.floor(previous_frame.shape[1] * 
			height / (previous_frame.shape[0]))

	# resize frame
	resized_frame = cv2.resize(previous_frame, (width, height))

	timex_float = None
	timex_uint = None

	(rAvg, gAvg, bAvg) = (None, None, None)

	frame_count = 0
	while True:

		# capture frame-by-frame
		ret, frame = cap.read()

		# if frame reading was not successful, break
		if not ret:
			break

		resized_frame = cv2.resize(frame, (width, height))

		'''
		create aggregated image
		'''
		(B, G, R) = cv2.split(resized_frame.astype("float"))

		# if the frame averages are None, initialize them
		if rAvg is None:
			rAvg = R
			bAvg = B
			gAvg = G
		# otherwise, compute the weighted average between the history of
		# frames and the current frames
		else:
			rAvg = ((frame_count * rAvg) + (1 * R)) / (frame_count + 1.0)
			gAvg = ((frame_count * gAvg) + (1 * G)) / (frame_count + 1.0)
			bAvg = ((frame_count * bAvg) + (1 * B)) / (frame_count + 1.0)
		
		avg = cv2.merge([bAvg, gAvg, rAvg]).astype("uint8")


		# visualization
		cv2.imshow("timex", avg)
		
		k = cv2.waitKey(1)
		if k == 27:
			break

		if frame_count == 1200:
			break

		if k == 115:
			#cv2.imwrite(outpath + "/contours_root.jpg", flow_contours_root)
			# cv2.imwrite(outpath + "/flow_sub.jpg", cpu_flow_mean_sub_masked_bgr)
			cv2.imwrite(outpath + "/" + filename + "/timex.jpg", avg)

		frame_count += 1

	#print(outpath + "/" + filename + "/timex.jpg")
	cv2.imwrite(outpath + "/" + filename + "_timex.jpg", avg)

	# release the capture
	cap.release()

	# destroy all windows
	cv2.destroyAllWindows()


if __name__ == "__main__":

	# init argument parser
	parser = argparse.ArgumentParser(description="Rip Currents Detection with CUDA enabled")

	parser.add_argument(
		"--video", help="path to .mp4 video file", required=True, type=str,
	)

	parser.add_argument(
		"--out", help="path and file name of the output file without .mp4", required=True, type=str,
	)

	parser.add_argument(
		"--height", help="resized height of the output", required=False, type=int, default=480,
	)

	# parsing script arguments
	args = parser.parse_args()
	video = args.video
	outpath = args.out
	height = args.height

	# run pipeline
	main(video, outpath, height)