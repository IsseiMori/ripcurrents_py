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

import my_flow

def main(video, outpath, height, window_size, grid_size, bin_size, wave_dir):

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

	video_out1 = cv2.VideoWriter(outpath + "/" + filename + "_" + str(bin_size) + "_vis_flow.avi", cv2.VideoWriter_fourcc(*'MJPG'), fps, (width, height)) 

	# resize frame
	resized_frame = cv2.resize(frame, (width, height))


	# Calculate number of arrows
	arrow_count_w = math.floor(width / grid_size)
	arrow_count_h = math.floor(height / grid_size)

	print(arrow_count_h, arrow_count_w)

	mask_img = cv2.imread("E:/ripcurrents/flow_paper/figures/rip1/masks.png", 0)
	#mask_img = cv2.resize(mask_img, (width, height))

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

	sum_flow = np.zeros((arrow_count_h, arrow_count_w, 2), dtype=np.float32)
	sum_unit_flow = np.zeros((arrow_count_h, arrow_count_w, 2), dtype=np.float32)

	# stores the bin direction for each frame for windowsize frames
	buffer_bin = np.zeros((window_size, arrow_count_h, arrow_count_w), dtype=np.uint16)

	# stores the number of frames in each bin for the previous windowsize frames
	bin_hist = np.zeros((arrow_count_h, arrow_count_w, bin_num), dtype=np.uint16)

	# bin counts are weighted based on their magnitude
	buffer_bin_weighted_hist = np.zeros((window_size, arrow_count_h, arrow_count_w, bin_num), dtype=np.uint16)

	sum_bin_weighted_hist = np.zeros((arrow_count_h, arrow_count_w, bin_num), dtype=np.float32)




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

		# start optical flow timer
		start_of = time.time()

		# Optical Flow LK
		new_points, status, error = cv2.calcOpticalFlowPyrLK(old_gray, gray_frame, vertices_root, None, **lk_params)

		# end of timer, and record
		end_of = time.time()
		timers["optical flow"].append(end_of - start_of)

		# Calculate flow at each point
		flow = np.zeros((arrow_count_h, arrow_count_w, 2), dtype=np.float32)
		for i in range(len(new_points)):
			row = math.floor(i / arrow_count_w)
			col = math.floor(i % arrow_count_w)
			# print(row, col)
			flow[row][col][0] = new_points[i][0] - vertices_root[i][0]
			flow[row][col][1] = new_points[i][1] - vertices_root[i][1]

		# Calculate unit flow
		flow_unit = my_flow.flow_to_unit(flow)

		flow_bins = my_flow.flow_to_bins(flow, bin_num)

		current_buffer_i = frame_count % window_size



		# update total flow
		sum_flow -= buffer_flow[current_buffer_i]
		buffer_flow[current_buffer_i] = flow
		sum_flow += flow

		# update total unit flow
		sum_unit_flow -= buffer_unit_flow[current_buffer_i]
		buffer_unit_flow[current_buffer_i] = flow
		sum_unit_flow += flow_unit


		# update bin hist
		if frame_count >= window_size:
			bin_hist = my_flow.remove_from_bin_hist(bin_hist, buffer_bin[current_buffer_i])
		buffer_bin[current_buffer_i] = flow_bins
		bin_hist = my_flow.append_to_bin_hist(bin_hist, flow_bins)
		max_bins = my_flow.hist_to_max_bin(bin_hist)

		# update bin weighted hist
		bin_weighted = my_flow.flow_to_bin_weighted(flow, bin_num)
		sum_bin_weighted_hist -= buffer_bin_weighted_hist[current_buffer_i]
		buffer_bin_weighted_hist[current_buffer_i] = bin_weighted
		sum_bin_weighted_hist += bin_weighted
		max_bins_weighted = my_flow.hist_to_max_bin(sum_bin_weighted_hist)

		# start optical flow timer
		start_of = time.time()

		threshold_min_mag = min(window_size, frame_count+1) * 0
		vis_flow, _ = my_flow.draw_arrows_flow_mask(resized_frame, sum_flow, bin_num, vertices_root_pos_2d, grid_size * 0.8, wave_dir, mask_img, grid_size, threshold_min_mag)

		# end of timer, and record
		end_of = time.time()
		timers["post-process"].append(end_of - start_of)

		vis_flow = cv2.putText(vis_flow, str(frame_count), (30,30), cv2.FONT_HERSHEY_SIMPLEX,  
                   0.8, (255,255,255), 1, cv2.LINE_AA) 

		old_gray = gray_frame.copy()


		# end full pipeline timer, and record
		end_full_time = time.time()
		timers["full pipeline"].append(end_full_time - start_full_time)

		# visualization
		cv2.imshow("vis_flow", vis_flow)

		video_out1.write(vis_flow)

		k = cv2.waitKey(1)
		if k == 27:
			break

		frame_count += 1

	video_out1.release()

	# cv2.imwrite(outpath + "/" + filename + "_timelines.jpg", frame_timelines)
	# cv2.imwrite(outpath + "/" + filename + "_timelines_norm.jpg", frame_timelines_norm)

	# release the capture
	cap.release()

	# destroy all windows
	cv2.destroyAllWindows()

	# print results
	print("Number of frames : ", frame_count)

	print("Elapsed time")
	for stage, seconds in timers.items():
		print("-", stage, ": {:0.3f} seconds".format(sum(seconds)))

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

	parser.add_argument(
		"--wave_dir", help="incoming dir", required=False, type=int, default=-1,
	)

	# parsing script arguments
	args = parser.parse_args()
	video = args.video
	outpath = args.out
	height = args.height
	window_size = args.window
	grid_size = args.grid
	bin_num = args.bins
	wave_dir = args.wave_dir


	# run pipeline
	main(video, outpath, height, window_size, grid_size, bin_num, wave_dir)