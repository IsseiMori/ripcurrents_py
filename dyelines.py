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

# return degree angle and normalized magnitude
def calc_angle_from_flow_cpu(cpu_flow):
	cpu_flow_x = cpu_flow[:,:,0]
	cpu_flow_y = cpu_flow[:,:,1]

	cpu_flow_magnitude, cpu_flow_angle = cv2.cartToPolar(
		cpu_flow_x, cpu_flow_y, angleInDegrees=True,
	)

	cv2.normalize(cpu_flow_magnitude, cpu_flow_magnitude, 0.0, 1.0, cv2.NORM_MINMAX)

	return cpu_flow_angle, cpu_flow_magnitude

def calc_bgr_from_angle_magnitude(cpu_flow_angle, cpu_flow_magnitude):
	cpu_flow_hsv = cv2.merge((
		cpu_flow_angle, 
		np.ones_like(cpu_flow_angle, np.float32),
		cpu_flow_magnitude
	))

	cpu_flow_bgr = cv2.cvtColor(cpu_flow_hsv, cv2.COLOR_HSV2BGR) * 255
	cpu_flow_bgr = cpu_flow_bgr.astype(np.uint8)

	return cpu_flow_bgr

def calc_bgr_from_angle_magnitude_root(cpu_flow_angle, cpu_flow_magnitude):
	cpu_flow_hsv = cv2.merge((
		cpu_flow_angle, 
		np.ones_like(cpu_flow_angle, np.float32),
		cpu_flow_magnitude
	))

	cpu_flow_bgr = np.sqrt(cv2.cvtColor(cpu_flow_hsv, cv2.COLOR_HSV2BGR)) * 255
	cpu_flow_bgr = cpu_flow_bgr.astype(np.uint8)

	return cpu_flow_bgr

# return degree angle and normalized magnitude
def calc_unit_flow_cpu(cpu_flow):
	cpu_flow_x = cpu_flow[:,:,0]
	cpu_flow_y = cpu_flow[:,:,1]

	cpu_flow_magnitude, cpu_flow_angle = cv2.cartToPolar(
		cpu_flow_x, cpu_flow_y, angleInDegrees=True,
	)

	x = np.cos(cpu_flow_angle * math.pi / 180)
	y = np.sin(cpu_flow_angle * math.pi / 180)

	cpu_unit_flow = cv2.merge([x,y])

	return cpu_unit_flow

def zero_edge_flow(cpu_flow):

	height, width, _ = cpu_flow.shape
	offset = 50

	cpu_flow[0:offset,:,0] = 0
	cpu_flow[0:offset,:,1] = 0
	cpu_flow[height-offset:height,:,0] = 0
	cpu_flow[height-offset:height,:,1] = 0
	cpu_flow[:,0:offset,0] = 0
	cpu_flow[:,0:offset,1] = 0
	cpu_flow[:,width-offset:width,0] = 0
	cpu_flow[:,width-offset:width,1] = 0

	return cpu_flow

def remove_outlier(cpu_flow):
	
	q3_x = np.quantile(cpu_flow[:,:,0], (0.75))
	q3_y = np.quantile(cpu_flow[:,:,1], (0.75))
	q1_x = np.quantile(cpu_flow[:,:,0], (0.25))
	q1_y = np.quantile(cpu_flow[:,:,1], (0.25))

	ior_x = q3_x - q1_x
	ior_y = q3_y - q1_y

	cpu_flow[:,:,0] = np.where(cpu_flow[:,:,0] > (q3_x + 1.5 * ior_x), 0, cpu_flow[:,:,0])
	cpu_flow[:,:,1] = np.where(cpu_flow[:,:,1] > (q3_y + 1.5 * ior_y), 0, cpu_flow[:,:,1])
	cpu_flow[:,:,0] = np.where(cpu_flow[:,:,0] > (q3_y + 1.5 * ior_y), 0, cpu_flow[:,:,0])
	cpu_flow[:,:,1] = np.where(cpu_flow[:,:,1] > (q3_x + 1.5 * ior_x), 0, cpu_flow[:,:,1])

	return cpu_flow

# calculate bin for each arrow
def flow_to_bins(flow, bin_num):
	flow_bins = np.zeros((flow.shape[0], flow.shape[1]), dtype=np.uint8)

	for row in range(len(flow)):
		for col in range(len(flow[0])):
			dx = flow[row][col][0]
			dy = flow[row][col][1]
			bin_dir = xy_to_bin(dx, dy, bin_num)
			flow_bins[row][col] = bin_dir

	return flow_bins

def xy_to_bin(x, y, bin_num):
	theta = math.atan2(y,x)
	bin_dir = (int)((theta + math.pi) / (2 * math.pi) * bin_num)
	return bin_dir % bin_num

def mat_mode_bin(flow_bins, bin_num):

	count = np.zeros(bin_num, dtype=np.uint16)

	for row in range(len(flow_bins)):
		for col in range(len(flow_bins[0])):
			bin_dir = flow_bins[row][col]
			count[bin_dir] += 1

	return np.argmax(count)

def max_bin_to_rip(bin_dir, bin_num):
	opposit = bin_dir - bin_num / 2 if bin_dir + bin_num / 2 > bin_num - 1 else bin_dir + bin_num / 2
	opp_near1 = bin_dir - bin_num / 2 + 1 if bin_dir + bin_num / 2 + 1 > bin_num - 1 else bin_dir + bin_num / 2 + 1
	opp_near2 = bin_dir - bin_num / 2 - 1 if bin_dir + bin_num / 2 - 1 > bin_num - 1 else bin_dir + bin_num / 2 - 1

	return int(opposit), int(opp_near1), int(opp_near2)

box_pos = []
def draw_box(event, x, y, flags, param):
	if event == cv2.EVENT_LBUTTONDOWN:
		print(x, y)
		pos = np.array([x, y])
		box_pos.append(pos)

def order_points(pts):
	# initialzie a list of coordinates that will be ordered
	# such that the first entry in the list is the top-left,
	# the second entry is the top-right, the third is the
	# bottom-right, and the fourth is the bottom-left
	rect = np.zeros((4, 2), dtype = "float32")
	# the top-left point will have the smallest sum, whereas
	# the bottom-right point will have the largest sum
	s = pts.sum(axis = 1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]
	# now, compute the difference between the points, the
	# top-right point will have the smallest difference,
	# whereas the bottom-left will have the largest difference
	diff = np.diff(pts, axis = 1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]
	# return the ordered coordinates
	rect[0] = pts[0]
	rect[1] = pts[1]
	rect[2] = pts[2]
	rect[3] = pts[3]
	return rect

def four_point_transform(image, width, height, pts):
	# obtain a consistent order of the points and unpack them
	# individually
	rect = order_points(pts)
	(tl, tr, br, bl) = rect

	dst = np.array([
		[0, 0],
		[width - 1, 0],
		[width - 1, height - 1],
		[0, height - 1]], dtype = "float32")
	# compute the perspective transform matrix and then apply it
	M = cv2.getPerspectiveTransform(rect, dst)
	warped = cv2.warpPerspective(image, M, (width, height))
	# return the warped image
	return warped

line_pos = []
def draw_lines(event, x, y, flags, param):
	if event == cv2.EVENT_LBUTTONDOWN:
		print(x, y)
		pos = np.array([x, y])
		line_pos.append(pos)

class Timeline:
	def __init__(self, start, end, vnum):
		self.vertices_origin = []
		self.vertices = []
		self.vnum = vnum
		
		spacing = (end - start) / (vnum - 1)
		for i_vertex in range(0, vnum):
			vertex = start + spacing * i_vertex
			self.vertices_origin.append(vertex)
			vertices_list = []
			self.vertices.append(vertices_list)

	def birth_line(self):
		new_vertices = copy.deepcopy(self.vertices_origin)
		for i_vnum in range(0, self.vnum):
			self.vertices[i_vnum].append(new_vertices[i_vnum])

	def move_vertices(self, flow, step_size):

		# move each vertex based on the flow
		for i_vnum in range(0, self.vnum):
			for i_vertex in range(len(self.vertices[i_vnum])):
				x = self.vertices[i_vnum][i_vertex][0]
				y = self.vertices[i_vnum][i_vertex][1]
				flow_vec = flow[math.floor(y)][math.floor(x)]
				x += flow_vec[0] * step_size
				y += flow_vec[1] * step_size
				self.vertices[i_vnum][i_vertex][0] = x
				self.vertices[i_vnum][i_vertex][1] = y

	# draw timelines and return the image
	def draw_lines(self, img):
		ret_img = img.copy()

		# draw initial line
		for i_vertex in range(len(self.vertices_origin) - 1):
			x1 = math.floor(self.vertices_origin[i_vertex][0])
			y1 = math.floor(self.vertices_origin[i_vertex][1])
			x2 = math.floor(self.vertices_origin[i_vertex + 1][0])
			y2 = math.floor(self.vertices_origin[i_vertex + 1][1])
			ret_img = cv2.circle(ret_img, (x1, y1), 4, (70, 70, 70), -1)
			ret_img = cv2.line(ret_img, (x1, y1), (x2, y2), (70, 70, 70), 4) 
			
			# draw the last point
			if i_vertex == len(self.vertices_origin) - 1:
				ret_img = cv2.circle(ret_img, (x2, y2), 4, (70, 70, 70), -1)

		img_dye = np.zeros(ret_img.shape, dtype=np.uint8)
		opacity = 5.0 / len(self.vertices)

		# draw moving vertices
		for i_vnum in range(0, self.vnum):
			for i_vertex in range(len(self.vertices[i_vnum])):
				x1 = math.floor(self.vertices[i_vnum][i_vertex][0])
				y1 = math.floor(self.vertices[i_vnum][i_vertex][1])

				img_circle = np.zeros(ret_img.shape, dtype=np.uint8)
				img_circle = cv2.circle(img_circle, (x1, y1), 40, (0, 255, 0), -1)
				img_dye = cv2.addWeighted(img_dye, 1, img_circle, opacity, 0)

		ret_img = cv2.addWeighted(ret_img, 1, img_dye, 1, 0)
		
		return ret_img

	def delete_dye(self, flow):
		bin_num = 6 # only works for 6
		bin_mat = flow_to_bins(flow, bin_num)
		max_dir = mat_mode_bin(bin_mat, bin_num)
		rip_dir1, rip_dir2, rip_dir3 = max_bin_to_rip(max_dir, bin_num)

		for i_vnum in range(0, self.vnum):
			new_vertices = []
			for i_vertex in range(len(self.vertices[i_vnum])):
				x = math.floor(self.vertices[i_vnum][i_vertex][0])
				y = math.floor(self.vertices[i_vnum][i_vertex][1])
				bin_dir = xy_to_bin(x, y, bin_num)
				if bin_dir == rip_dir1 or bin_dir == rip_dir2 or bin_dir == rip_dir3:
					new_vertices.append(self.vertices[i_vnum][i_vertex])
			self.vertices[i_vnum] = new_vertices



def add_color_wheel(img, wheel):
	wheel_resized = cv2.resize(wheel, (50, 50))
	img[10:10+wheel_resized.shape[0], 10:10+wheel_resized.shape[1]] = wheel_resized

def main(video, outpath, height, window_size, correct_perspective, birthrate):

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

	# get total number of video frames
	num_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

	# read the first frame
	ret, frame = cap.read()

	# proceed if frame reading was successful
	if not ret: return

	# width after resize
	width = math.floor(frame.shape[1] * 
			height / (frame.shape[0]))

	video_out = cv2.VideoWriter(outpath + "/" + filename + "_flow_overlay.avi", cv2.VideoWriter_fourcc(*'MJPG'), fps, (width, height)) 

	# load mask img
	mask_img = cv2.imread("mask3.png", 0)
	# make sure the size is correct. This should not be necessary
	mask_img = cv2.resize(mask_img, (width, height))
	mask_img_bgr = cv2.cvtColor(mask_img, cv2.COLOR_GRAY2BGR)

	# resize frame
	resized_frame = cv2.resize(frame, (width, height))

	perspective_corrected = False
	if correct_perspective == 1:
		'''
		perspective correction init
		'''
		cv2.namedWindow('click to draw a box')
		cv2.setMouseCallback('click to draw a box', draw_box)

		cv2.imshow("click to draw a box", frame)

		# wait for clicks until enter is hit
		while(1):
			k = cv2.waitKey()
			if k == 13:
				break

		if len(box_pos) > 0: 
			perspective_corrected = True
			pts = np.array([(box_pos[0][0], box_pos[0][1]), (box_pos[1][0], box_pos[1][1]), (box_pos[2][0], box_pos[2][1]), (box_pos[3][0], box_pos[3][1])])
			# apply the four point tranform to obtain a "birds eye view" of
			# the image


			'''
			export img with perspective correction box
			'''
			frame_with_box = frame.copy()
			cv2.line(frame_with_box, (box_pos[0][0], box_pos[0][1]), (box_pos[1][0], box_pos[1][1]), (255,255,255), 3)
			cv2.line(frame_with_box, (box_pos[1][0], box_pos[1][1]), (box_pos[2][0], box_pos[2][1]), (255,255,255), 3)
			cv2.line(frame_with_box, (box_pos[2][0], box_pos[2][1]), (box_pos[3][0], box_pos[3][1]), (255,255,255), 3)
			cv2.line(frame_with_box, (box_pos[3][0], box_pos[3][1]), (box_pos[0][0], box_pos[0][1]), (255,255,255), 3)
			cv2.imwrite(outpath + "/" + filename + "_box_position.jpg", frame_with_box)

			resized_frame = four_point_transform(frame, width, height, pts)



	# upload resized frame to GPU
	gpu_frame = cv2.cuda_GpuMat()
	gpu_frame.upload(resized_frame)

	# convert to gray
	previous_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)

	# upload pre-processed frame to GPU
	gpu_previous = cv2.cuda_GpuMat()
	gpu_previous.upload(previous_frame)

	# create gpu_hsv output for optical flow
	gpu_hsv = cv2.cuda_GpuMat(gpu_frame.size(), cv2.CV_32FC3)
	gpu_hsv_8u = cv2.cuda_GpuMat(gpu_frame.size(), cv2.CV_8UC3)

	gpu_h = cv2.cuda_GpuMat(gpu_frame.size(), cv2.CV_32FC1)
	gpu_s = cv2.cuda_GpuMat(gpu_frame.size(), cv2.CV_32FC1)
	gpu_v = cv2.cuda_GpuMat(gpu_frame.size(), cv2.CV_32FC1)

	# set saturation to 1
	gpu_s.upload(np.ones_like(previous_frame, np.float32))

	cpu_flow_array = []

	color_wheel = cv2.imread("colorWheel.jpg")






	# Init timeline
	cv2.namedWindow('click to draw timelines')
	cv2.setMouseCallback('click to draw timelines', draw_lines)

	cv2.imshow("click to draw timelines", resized_frame)

	# wait for clicks until enter is hit
	while(1):
		k = cv2.waitKey()
		if k == 13:
			break


	# initialize timelines
	timelines = []
	for i_vertex in range(0, len(line_pos) - 1, 2):
		timeline = Timeline(line_pos[i_vertex], line_pos[i_vertex + 1], 20)
		timelines.append(timeline)

	frame_count = 0
	while True:
		# start full pipeline timer
		start_full_time = time.time()

		# start reading timer
		start_read_time = time.time()

		ret, frame = cap.read()

		# if frame reading was not successful, break
		if not ret:
			break

		resized_frame = cv2.resize(frame, (width, height))
		if perspective_corrected:
			resized_frame = four_point_transform(frame, width, height, pts)

		# upload frame to GPU
		gpu_frame.upload(resized_frame)

		# end reading timer, and record
		end_read_time = time.time()
		timers["reading"].append(end_read_time - start_read_time)


		# start pre-process timer
		start_pre_time = time.time()

		# resize frame
		gpu_frame = cv2.cuda.resize(gpu_frame, (width, height))

		# convert to gray
		gpu_current = cv2.cuda.cvtColor(gpu_frame, cv2.COLOR_BGR2GRAY)

		# end pre-process timer, and record
		end_pre_time = time.time()
		timers["pre-process"].append(end_pre_time - start_pre_time)

		# start optical flow timer
		start_of = time.time()


		# create optical flow instance
		# create (int numLevels=5, double pyrScale=0.5, bool fastPyramids=false, int winSize=13, int numIters=10, int polyN=5, double polySigma=1.1, int flags=0)
		gpu_flow = cv2.cuda_FarnebackOpticalFlow.create(
			4, 0.5, False, 7, 10, 7, 1.5, 0,
		)
		# calculate optical flow
		gpu_flow = cv2.cuda_FarnebackOpticalFlow.calc(
			gpu_flow, gpu_previous, gpu_current, None,
		)

		# end of timer, and record
		end_of = time.time()
		timers["optical flow"].append(end_of - start_of)

		# start post-process timer
		start_post_time = time.time()


		
		cpu_flow = gpu_flow.download()
		cpu_flow = calc_unit_flow_cpu(cpu_flow)

		# prevent bug on edge
		#cpu_flow = zero_edge_flow(cpu_flow)
		cpu_flow = remove_outlier(cpu_flow)



		'''
		create aggregated flow 
		'''
		cpu_flow_divided = cpu_flow / window_size
		cpu_flow_array.append(cpu_flow_divided)

		if frame_count == 0:
			cpu_flow_average = cpu_flow_divided.copy()
		elif  frame_count < window_size:
			cpu_flow_average += cpu_flow_divided
		else:
			cpu_flow_average += cpu_flow_divided
			cpu_flow_average -= cpu_flow_array[0]
			cpu_flow_array.pop(0)


		cpu_flow_average_angle, cpu_flow_average_magnitude = calc_angle_from_flow_cpu(cpu_flow_average)


		if frame_count == 0:
			cpu_flow_angle, cpu_flow_magnitude = calc_angle_from_flow_cpu(cpu_flow_average)
			cpu_flow_mag_max = cpu_flow_magnitude.copy()
			cpu_flow_max = cpu_flow_average.copy()
		else:
			cpu_flow_angle, cpu_flow_magnitude = calc_angle_from_flow_cpu(cpu_flow_average)
			cpu_flow_max_angle, cpu_flow_max_magnitude = calc_angle_from_flow_cpu(cpu_flow_max)
			cpu_flow_max[:,:,0] = (cpu_flow_magnitude > cpu_flow_max_magnitude) * cpu_flow_average[:,:,0]\
							+ (cpu_flow_magnitude < cpu_flow_max_magnitude) * cpu_flow_max[:,:,0]
			cpu_flow_max[:,:,1] = (cpu_flow_magnitude > cpu_flow_max_magnitude) * cpu_flow_average[:,:,1]\
							+ (cpu_flow_magnitude < cpu_flow_max_magnitude) * cpu_flow_max[:,:,1]



		# create new timeline
		if frame_count == 0 or frame_count % birthrate == 0:
			for timeline in timelines:
				timeline.delete_dye(cpu_flow_average)
				timeline.birth_line()

	
		# move timelines
		for timeline in timelines:
			timeline.move_vertices(cpu_flow, 1.0)


		cpu_flow_average_bgr = calc_bgr_from_angle_magnitude(cpu_flow_average_angle, cpu_flow_average_magnitude)

		cpu_flow_overlay = cpu_flow_average_bgr.copy()
		cv2.addWeighted(cpu_flow_average_bgr, 1, resized_frame, 1, 0, cpu_flow_overlay)

		add_color_wheel(cpu_flow_average_bgr, color_wheel)
		add_color_wheel(cpu_flow_overlay, color_wheel)

		# draw timelines
		frame_timelines = resized_frame.copy()
		for timeline in timelines:
			frame_timelines = timeline.draw_lines(frame_timelines)

		frame_timelines = cv2.putText(frame_timelines, str(frame_count), (30,30), cv2.FONT_HERSHEY_SIMPLEX,  
                   0.8, (255,255,255), 1, cv2.LINE_AA) 

		# update previous_frame value
		gpu_previous = gpu_current
	

		# end post-process timer, and record
		end_post_time = time.time()
		timers["post-process"].append(end_post_time - start_post_time)

		# end full pipeline timer, and record
		end_full_time = time.time()
		timers["full pipeline"].append(end_full_time - start_full_time)

		# visualization
		#cv2.imshow("original", frame)
		cv2.imshow("timelines", frame_timelines)

		video_out.write(frame_timelines)

		k = cv2.waitKey(1)
		if k == 27:
			break

		if frame_count == 1500:
			break


		if k == 115:
			cv2.imwrite(outpath + "/" + filename + "/flow_average.jpg", cpu_flow_average_bgr)
			cv2.imwrite(outpath + "/" + filename + "/flow_average_overlay.jpg", cpu_flow_overlay)
			

		frame_count += 1

	video_out.release()

	# release the capture
	cap.release()

	# destroy all windows
	cv2.destroyAllWindows()

	# print results
	print("Number of frames : ", frame_count)

	# elapsed time at each stage
	print("Elapsed time")
	for stage, seconds in timers.items():
		print("-", stage, ": {:0.3f} seconds".format(sum(seconds)))

	# calculate frames per second
	print("Default video FPS : {:0.3f}".format(fps))

	of_fps = (frame_count - 1) / sum(timers["optical flow"])
	print("Optical flow FPS : {:0.3f}".format(of_fps))

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
		"--correct_perspective", help="correct perspective? 1 or 0", required=False, type=int, default=0,
	)

	parser.add_argument(
		"--birthrate", help="birth rate", required=False, type=int, default=30,
	)

	# parsing script arguments
	args = parser.parse_args()
	video = args.video
	outpath = args.out
	height = args.height
	window_size = args.window
	correct_perspective = args.correct_perspective
	birthrate = args.birthrate


	# run pipeline
	main(video, outpath, height, window_size, correct_perspective, birthrate)