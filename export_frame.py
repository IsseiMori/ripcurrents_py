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


def main(video, outpath, height, extract_frame):

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

	frame_count = 0
	while True:

		# capture frame-by-frame
		ret, frame = cap.read()

		# if frame reading was not successful, break
		if not ret:
			break


		previous_frame = frame;

		frame_count += 1

		if frame_count == extract_frame: break

	#print(outpath + "/" + filename + "/timex.jpg")
	cv2.imwrite(outpath + "/" + filename + "_" + str(frame_count) +".jpg", previous_frame)

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

	parser.add_argument(
		"--frame", help="resized height of the output", required=False, type=int, default=10000,
	)

	# parsing script arguments
	args = parser.parse_args()
	video = args.video
	outpath = args.out
	height = args.height
	extract_frame = args.frame

	# run pipeline
	main(video, outpath, height, extract_frame)