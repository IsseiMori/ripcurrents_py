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

def flow_to_unit(flow):
	flow_unit = flow.copy()

	for row in range(len(flow_unit)):
		for col in range(len(flow_unit[0])):
			dx = flow[row][col][0]
			dy = flow[row][col][1]
			dx, dy = unitize_xy(dx, dy)
			flow_unit[row][col][0] = dx
			flow_unit[row][col][1] = dy

	return flow_unit

def xy_to_bin(x, y, bin_num):
	theta = math.atan2(y,x)
	bin_dir = (int)((theta + math.pi) / (2 * math.pi) * bin_num)
	return bin_dir % bin_num

# calculate bin for each arrow
def flow_to_bins(flow, bin_num):
	flow_bins = np.zeros((flow.shape[0], flow.shape[1]), dtype=np.uint16)

	for row in range(len(flow)):
		for col in range(len(flow[0])):
			dx = flow[row][col][0]
			dy = flow[row][col][1]
			bin_dir = xy_to_bin(dx, dy, bin_num)
			flow_bins[row][col] = bin_dir

	return flow_bins

def flow_to_bin_weighted(flow, bin_num):
	bin_weighted = np.zeros((flow.shape[0], flow.shape[1], bin_num), dtype=np.float32)

	for row in range(len(flow)):
		for col in range(len(flow[0])):
			dx = flow[row][col][0]
			dy = flow[row][col][1]
			magnitude = math.sqrt(dx * dx + dy * dy)
			bin_dir = xy_to_bin(dx, dy, bin_num)
			bin_weighted[row][col][bin_dir] += magnitude

	return bin_weighted


# increment the bin count
def append_to_bin_hist(bin_hist, flow_bins):
	
	for row in range(len(flow_bins)):
		for col in range(len(flow_bins[0])):
			bin_dir = flow_bins[row][col]
			bin_hist[row][col][bin_dir] += 1

	return bin_hist

# increment the bin count
def remove_from_bin_hist(bin_hist, flow_bins):
	
	for row in range(len(flow_bins)):
		for col in range(len(flow_bins[0])):
			bin_dir = flow_bins[row][col]
			bin_hist[row][col][bin_dir] -= 1

	return bin_hist

def hist_to_max_bin(bin_hist):

	max_bins = np.zeros((bin_hist.shape[0], bin_hist.shape[1]), dtype=np.uint16)

	for row in range(len(bin_hist)):
		for col in range(len(bin_hist[0])):
			bin_dir = np.argmax(bin_hist[row][col])
			max_bins[row][col] = bin_dir

	return max_bins


def bin_to_flow(bin_dir, bin_num):
	angle = bin_dir / float(bin_num) * math.pi * 2 - math.pi + 0.5 / float(bin_num) * math.pi * 2
	dx = math.cos(angle)
	dy = math.sin(angle)
	return (dx, dy)

def max_bin_to_rip(bin_dir, bin_num):
	opposit = bin_dir - bin_num / 2 if bin_dir + bin_num / 2 > bin_num - 1 else bin_dir + bin_num / 2
	opp_near1 = bin_dir - bin_num / 2 + 1 if bin_dir + bin_num / 2 + 1 > bin_num - 1 else bin_dir + bin_num / 2 + 1
	opp_near2 = bin_dir - bin_num / 2 - 1 if bin_dir + bin_num / 2 - 1 > bin_num - 1 else bin_dir + bin_num / 2 - 1

	return int(opposit), int(opp_near1), int(opp_near2)

def mat_mode_bin(flow_bins, bin_num):

	count = np.zeros(bin_num, dtype=np.uint16)

	for row in range(len(flow_bins)):
		for col in range(len(flow_bins[0])):
			bin_dir = flow_bins[row][col]
			count[bin_dir] += 1

	return np.argmax(count)

def draw_arrows_flow(img, flow, bin_num, vertices_root_pos_2d, dt, wave_dir):

	img_ret = img.copy()
	img_ret_seg = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
	img_ret_seg.fill(0)

	max_x = np.amax(flow[:][:][0])
	max_y = np.amax(flow[:][:][1])

	max_len = math.sqrt(max_x * max_x + max_y * max_y)

	flow_bins = flow_to_bins(flow, bin_num)
	max_bin = mat_mode_bin(flow_bins, bin_num)

	if (wave_dir != -1): max_bin = wave_dir

	opposit, opp_near1, opp_near2 = max_bin_to_rip(max_bin, bin_num)

	count_max = 0
	count_near1 = 0
	count_near2 = 0
	count_opposite = 0

	for row in range(len(flow)):
		for col in range(len(flow[0])):
			dx = flow[row][col][0]
			dy = flow[row][col][1]
			dx, dy = unitize_xy(dx, dy)
			# dx = dx / math.sqrt(4)
			# dy = dy / math.sqrt(4)
			x0 = vertices_root_pos_2d[row][col][0]
			y0 = vertices_root_pos_2d[row][col][1]

			bin_dir = xy_to_bin(dx, dy, bin_num)
			if (bin_dir == max_bin) : 
				color = (0, 0, 0)
				count_max += 1
			elif (bin_dir == opposit) : 
				color = (0, 0, 255)
				count_opposite += 1
			elif (bin_dir == opp_near1) : 
				color = (0, 200, 255)
				count_near1 += 1
			elif (bin_dir == opp_near2) : 
				color = (0, 255, 200)
				count_near2 += 1
			else : color = (128, 128, 128)


			cv2.arrowedLine(img_ret, (int(x0), int(y0)), (int(x0 + dx * dt), int(y0 + dy * dt)), color, 3, tipLength = 0.5)

			if (bin_dir == opposit or bin_dir == opp_near1 or bin_dir == opp_near2): 
				cv2.rectangle(img_ret_seg, (int(x0), int(y0)), (int(x0 + 20), int(y0 + 20)), (255, 255, 255), -1)

	img_ret = cv2.putText(img_ret, str(count_max), (30,50), cv2.FONT_HERSHEY_SIMPLEX,  
                   0.6, (0, 0, 0), 2, cv2.LINE_AA) 
	img_ret = cv2.putText(img_ret, str(count_near1), (30,70), cv2.FONT_HERSHEY_SIMPLEX,  
                   0.6, (0, 0, 255), 2, cv2.LINE_AA) 
	img_ret = cv2.putText(img_ret, str(count_near2), (30,90), cv2.FONT_HERSHEY_SIMPLEX,  
                   0.6, (0, 200, 255), 2, cv2.LINE_AA) 
	img_ret = cv2.putText(img_ret, str(count_opposite), (30,110), cv2.FONT_HERSHEY_SIMPLEX,  
                   0.6, (0, 255, 200), 2, cv2.LINE_AA) 
	
	return img_ret, img_ret_seg

def draw_arrows_unit_flow(img, flow, bin_num, vertices_root_pos_2d, dt):

	img_ret = img.copy()

	for row in range(len(flow)):
		for col in range(len(flow[0])):
			dx = flow[row][col][0]
			dy = flow[row][col][1]
			dx, dy = unitize_xy(dx, dy)
			x0 = vertices_root_pos_2d[row][col][0]
			y0 = vertices_root_pos_2d[row][col][1]

			cv2.arrowedLine(img_ret, (int(x0), int(y0)), (int(x0 + dx * dt), int(y0 + dy * dt)), (255, 0, 0), 3, tipLength = 0.5)

	return img_ret

#   1 
# 0    2 
# 5    3
#   4
def draw_arrows_bins(img, flow_bins, bin_num, vertices_root_pos_2d, dt, wave_dir):

	img_ret = img.copy()
	img_ret_seg = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
	img_ret_seg.fill(0)

	max_bin = mat_mode_bin(flow_bins, bin_num)

	if (wave_dir != -1): max_bin = wave_dir

	opposit, opp_near1, opp_near2 = max_bin_to_rip(max_bin, bin_num)

	count_max = 0
	count_near1 = 0
	count_near2 = 0
	count_opposite = 0

	for row in range(len(flow_bins)):
		for col in range(len(flow_bins[0])):
			bin_dir = flow_bins[row][col]
			dx, dy = bin_to_flow(bin_dir, bin_num)
			x0 = vertices_root_pos_2d[row][col][0]
			y0 = vertices_root_pos_2d[row][col][1]

			if (bin_dir == max_bin) : 
				color = (0, 0, 0)
				count_max += 1
			elif (bin_dir == opposit) : 
				color = (0, 0, 255)
				count_opposite += 1
			elif (bin_dir == opp_near1) : 
				color = (0, 200, 255)
				count_near1 += 1
			elif (bin_dir == opp_near2) : 
				color = (0, 255, 200)
				count_near2 += 1
			else : color = (128, 128, 128)

			cv2.arrowedLine(img_ret, (int(x0), int(y0)), (int(x0 + dx * dt), int(y0 + dy * dt)), color, 3, tipLength = 0.5)

			if (bin_dir == opposit or bin_dir == opp_near1 or bin_dir == opp_near2): 
				cv2.rectangle(img_ret_seg, (int(x0), int(y0)), (int(x0 + 20), int(y0 + 20)), (255, 255, 255), -1)


	img_ret = cv2.putText(img_ret, str(count_max), (30,50), cv2.FONT_HERSHEY_SIMPLEX,  
                   0.6, (0, 0, 0), 2, cv2.LINE_AA) 
	img_ret = cv2.putText(img_ret, str(count_near1), (30,70), cv2.FONT_HERSHEY_SIMPLEX,  
                   0.6, (0, 0, 255), 2, cv2.LINE_AA) 
	img_ret = cv2.putText(img_ret, str(count_near2), (30,90), cv2.FONT_HERSHEY_SIMPLEX,  
                   0.6, (0, 200, 255), 2, cv2.LINE_AA) 
	img_ret = cv2.putText(img_ret, str(count_opposite), (30,110), cv2.FONT_HERSHEY_SIMPLEX,  
                   0.6, (0, 255, 200), 2, cv2.LINE_AA) 

	return img_ret, img_ret_seg


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
	video_out2 = cv2.VideoWriter(outpath + "/" + filename + "_" + str(bin_size) + "_vis_unit_flow.avi", cv2.VideoWriter_fourcc(*'MJPG'), fps, (width, height)) 
	video_out3 = cv2.VideoWriter(outpath + "/" + filename + "_" + str(bin_size) + "_vis_bin_hist.avi", cv2.VideoWriter_fourcc(*'MJPG'), fps, (width, height)) 
	video_out4 = cv2.VideoWriter(outpath + "/" + filename + "_" + str(bin_size) + "_vis_bin_weighted_hist.avi", cv2.VideoWriter_fourcc(*'MJPG'), fps, (width, height)) 

	video_out1_seg = cv2.VideoWriter(outpath + "/" + filename + "_" + str(bin_size) + "_vis_flow_seg.avi", cv2.VideoWriter_fourcc(*'MJPG'), fps, (width, height)) 
	video_out2_seg = cv2.VideoWriter(outpath + "/" + filename + "_" + str(bin_size) + "_vis_unit_flow_seg.avi", cv2.VideoWriter_fourcc(*'MJPG'), fps, (width, height)) 
	video_out3_seg = cv2.VideoWriter(outpath + "/" + filename + "_" + str(bin_size) + "_vis_bin_hist_seg.avi", cv2.VideoWriter_fourcc(*'MJPG'), fps, (width, height)) 
	video_out4_seg = cv2.VideoWriter(outpath + "/" + filename + "_" + str(bin_size) + "_vis_bin_weighted_hist_seg.avi", cv2.VideoWriter_fourcc(*'MJPG'), fps, (width, height)) 


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
		flow_unit = flow_to_unit(flow)

		flow_bins = flow_to_bins(flow, bin_num)

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
			bin_hist = remove_from_bin_hist(bin_hist, buffer_bin[current_buffer_i])
		buffer_bin[current_buffer_i] = flow_bins
		bin_hist = append_to_bin_hist(bin_hist, flow_bins)
		max_bins = hist_to_max_bin(bin_hist)

		# update bin weighted hist
		bin_weighted = flow_to_bin_weighted(flow, bin_num)
		sum_bin_weighted_hist -= buffer_bin_weighted_hist[current_buffer_i]
		buffer_bin_weighted_hist[current_buffer_i] = bin_weighted
		sum_bin_weighted_hist += bin_weighted
		max_bins_weighted = hist_to_max_bin(sum_bin_weighted_hist)

		# start optical flow timer
		start_of = time.time()

		vis_flow, vis_flow_seg = draw_arrows_flow(resized_frame, sum_flow, bin_num, vertices_root_pos_2d, grid_size * 0.8, wave_dir)
		vis_unit_flow, vis_unit_flow_seg = draw_arrows_flow(resized_frame, sum_unit_flow, bin_num, vertices_root_pos_2d, grid_size * 0.8, wave_dir)
		vis_bin_hist, vis_bin_hist_seg = draw_arrows_bins(resized_frame, max_bins, bin_num, vertices_root_pos_2d, grid_size * 0.8, wave_dir)
		vis_bin_weighted_hist, vis_bin_weighted_hist_seg = draw_arrows_bins(resized_frame, max_bins_weighted, bin_num, vertices_root_pos_2d, grid_size * 0.8, wave_dir)

		# end of timer, and record
		end_of = time.time()
		timers["post-process"].append(end_of - start_of)

		vis_flow = cv2.putText(vis_flow, str(frame_count), (30,30), cv2.FONT_HERSHEY_SIMPLEX,  
                   0.8, (255,255,255), 1, cv2.LINE_AA) 
		vis_unit_flow = cv2.putText(vis_unit_flow, str(frame_count), (30,30), cv2.FONT_HERSHEY_SIMPLEX,  
                   0.8, (255,255,255), 1, cv2.LINE_AA) 
		vis_bin_hist = cv2.putText(vis_bin_hist, str(frame_count), (30,30), cv2.FONT_HERSHEY_SIMPLEX,  
                   0.8, (255,255,255), 1, cv2.LINE_AA) 
		vis_bin_weighted_hist = cv2.putText(vis_bin_weighted_hist, str(frame_count), (30,30), cv2.FONT_HERSHEY_SIMPLEX,  
                   0.8, (255,255,255), 1, cv2.LINE_AA) 


		old_gray = gray_frame.copy()


		# end full pipeline timer, and record
		end_full_time = time.time()
		timers["full pipeline"].append(end_full_time - start_full_time)

		# visualization
		cv2.imshow("vis_flow", vis_flow)
		cv2.imshow("vis_unit_flow", vis_unit_flow)
		cv2.imshow("vis_bin_hist", vis_bin_hist)
		cv2.imshow("vis_bin_weighted_hist", vis_bin_weighted_hist)

		video_out1.write(vis_flow)
		video_out2.write(vis_unit_flow)
		video_out3.write(vis_bin_hist)
		video_out4.write(vis_bin_weighted_hist)

		video_out1_seg.write(vis_flow_seg)
		video_out2_seg.write(vis_unit_flow_seg)
		video_out3_seg.write(vis_bin_hist_seg)
		video_out4_seg.write(vis_bin_weighted_hist_seg)

		k = cv2.waitKey(1)
		if k == 27:
			break


		if k == 115:
			cv2.imwrite(outpath + "/" + filename + "/flow_average.jpg", cpu_flow_average_bgr)
			cv2.imwrite(outpath + "/" + filename + "/flow_average_overlay.jpg", cpu_flow_overlay)
			

		frame_count += 1

	video_out1.release()
	video_out2.release()
	video_out3.release()
	video_out4.release()

	video_out1_seg.release()
	video_out2_seg.release()
	video_out3_seg.release()
	video_out4_seg.release()

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