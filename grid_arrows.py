# python contours.py --video beach.mp4 --out . --height 480 --window 900


# Unresolved Bugs

import os
import numpy as np 
import cv2
import argparse
import time
import math
import matplotlib.pyplot as plt
from PIL import Image
import copy

def unitize_xy(x, y):
	theta = math.atan2(y,x)
	return (math.cos(theta), math.sin(theta))

def draw_arrows(img, flow, vertices_root_pos_2d, dt):

	img_ret = img.copy()

	max_x = np.amax(flow[:][:][0])
	max_y = np.amax(flow[:][:][1])

	max_len = math.sqrt(max_x * max_x + max_y * max_y)

	for row in range(len(flow)):
		for col in range(len(flow[0])):
			dx = flow[row][col][0]
			dy = flow[row][col][1]
			dx, dy = unitize_xy(dx, dy)
			x0 = vertices_root_pos_2d[row][col][0]
			y0 = vertices_root_pos_2d[row][col][1]

			cv2.arrowedLine(img_ret, (int(x0), int(y0)), (int(x0 + dx * dt), int(y0 + dy * dt)), (255, 0, 0), 3, tipLength = 0.5)

	return img_ret


def main(video, outpath, height, window_size, grid_size, bin_size):

	# init dict to track time for every stage at each iteration
	timers = {
		"full pipeline": [],
		"reading": [],
		"pre-process": [],
		"optical flow": [],
		"post-process": [],
	}
	
	print("reading ", video)

	filename = os.path.splitext(os.path.basename(video))[0]
	# if not os.path.exists(outpath + "/" + filename):
	# 	os.makedirs(outpath + "/" + filename)

	# init video capture with video
	cap = cv2.VideoCapture(video)

	# get default video FPS
	fps = cap.get(cv2.CAP_PROP_FPS)
	print(fps, " fps")

	# get total number of video frames
	num_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

	# read the first frame
	ret, frame = cap.read()

	# proceed if frame reading was successful
	if not ret: return

	# width after resize
	width = math.floor(frame.shape[1] * 
			height / (frame.shape[0]))

	video_out = cv2.VideoWriter(outpath + "/" + filename + "_timelines.avi", cv2.VideoWriter_fourcc(*'MJPG'), fps, (width, height)) 
	video_out2 = cv2.VideoWriter(outpath + "/" + filename + "_timelines_norm.avi", cv2.VideoWriter_fourcc(*'MJPG'), fps, (width, height)) 

	# resize frame
	resized_frame = cv2.resize(frame, (width, height))


	# Calculate number of arrows
	arrow_count_w = math.floor(width / grid_size)
	arrow_count_h = math.floor(height / grid_size)

	print(arrow_count_h, arrow_count_w)

	# 1D array of vertices position
	vertices_root =  np.array([], dtype=np.float32)
	vertices_root_pos_2d = np.zeros((arrow_count_h, arrow_count_w, 2), dtype=np.float32)
	for row in range (0, arrow_count_h):
		for col in range (0, arrow_count_w):
			x0 = col*grid_size + grid_size/2
			y0 = row*grid_size + grid_size/2
			if row == 0 and col == 0: 
				vertices_root =  np.array([x0, y0], dtype=np.float32)
			else:
				vertices_root = np.vstack((vertices_root, np.array([x0, y0], dtype=np.float32)))
			vertices_root_pos_2d[row][col][0] = x0
			vertices_root_pos_2d[row][col][1] = y0

	buffer_flow = np.zeros((window_size, arrow_count_h, arrow_count_w, 2), dtype=np.float32)
	buffer_unit_flow = np.zeros((window_size, arrow_count_h, arrow_count_w, 2), dtype=np.float32)
	buffer_bin = np.zeros((window_size, arrow_count_h, arrow_count_w), dtype=np.uint8)
	buffer_speed = np.zeros((window_size, arrow_count_h, arrow_count_w), dtype=np.float32)


	lk_params = dict(winSize = (15, 15),
					maxLevel = 5,
					criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

	# convert to gray
	old_gray = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)

	frame_count = 0
	while True:
		# start full pipeline timer
		start_full_time = time.time()

		ret, frame = cap.read()

		# if frame reading was not successful, break
		if not ret:
			break

		resized_frame = cv2.resize(frame, (width, height))

		gray_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)


		# Optical Flow LK
		new_points, status, error = cv2.calcOpticalFlowPyrLK(old_gray, gray_frame, vertices_root, None, **lk_params)

		# Calculate flow at each point
		flow = np.zeros((arrow_count_h, arrow_count_w, 2), dtype=np.float32)
		for i in range(len(new_points)):
			row = math.floor(i / arrow_count_w)
			col = math.floor(i % arrow_count_w)
			# print(row, col)
			flow[row][col][0] = new_points[i][0] - vertices_root[i][0]
			flow[row][col][1] = new_points[i][1] - vertices_root[i][1]

		vis_flow = draw_arrows(resized_frame, flow, vertices_root_pos_2d, grid_size / 2)


		old_gray = gray_frame.copy()


		# end full pipeline timer, and record
		end_full_time = time.time()
		timers["full pipeline"].append(end_full_time - start_full_time)

		# visualization
		cv2.imshow("timelines", vis_flow)
		# cv2.imshow("timelines norm", frame_timelines_norm)

		video_out.write(vis_flow)
		# video_out2.write(frame_timelines_norm)

		k = cv2.waitKey(1)
		if k == 27:
			break


		if k == 115:
			cv2.imwrite(outpath + "/" + filename + "/flow_average.jpg", cpu_flow_average_bgr)
			cv2.imwrite(outpath + "/" + filename + "/flow_average_overlay.jpg", cpu_flow_overlay)
			

		frame_count += 1

	video_out.release()

	# cv2.imwrite(outpath + "/" + filename + "_timelines.jpg", frame_timelines)
	# cv2.imwrite(outpath + "/" + filename + "_timelines_norm.jpg", frame_timelines_norm)

	# release the capture
	cap.release()

	# destroy all windows
	cv2.destroyAllWindows()

	# print results
	print("Number of frames : ", frame_count)

	# calculate frames per second
	print("Default video FPS : {:0.3f}".format(fps))

	full_fps = (frame_count - 1) / sum(timers["full pipeline"])
	print("Full pipeline FPS : {:0.3f}".format(full_fps))


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
		"--window", help="resized height of the output", required=False, type=int, default=900,
	)

	parser.add_argument(
		"--grid", help="grid per pixel", required=False, type=int, default=20,
	)

	parser.add_argument(
		"--bins", help="number of bins", required=False, type=int, default=6,
	)

	# parsing script arguments
	args = parser.parse_args()
	video = args.video
	outpath = args.out
	height = args.height
	window_size = args.window
	grid_size = args.grid
	bin_num = args.bins


	# run pipeline
	main(video, outpath, height, window_size, grid_size, bin_num)