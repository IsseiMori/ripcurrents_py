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
		self.vertices_origin = np.array([], dtype=np.float32)
		self.vertices = np.array([], dtype=np.float32)
		
		spacing = (end - start) / (vnum - 1)
		for i_vertex in range(0, vnum):
			for j_vertex in range(0, vnum):
				vertex = start + np.array([spacing[0], 0]) * i_vertex + np.array([0, spacing[1]]) * j_vertex
				print(i_vertex, j_vertex)
				print(vertex)
				if i_vertex == 0 and j_vertex == 0:
					self.vertices_origin = np.array([vertex[0], vertex[1]], dtype=np.float32)
				else:
					self.vertices_origin = np.vstack((self.vertices_origin, np.array([vertex[0], vertex[1]], dtype=np.float32)))
		print(len(self.vertices_origin))
		
	def birth_line(self):
		if len(self.vertices) == 0:
			self.vertices = copy.deepcopy(self.vertices_origin)
		else:
			self.vertices = np.vstack((self.vertices, self.vertices_origin))

	def move_vertices(self, flow, step_size):

		# move each vertex based on the flow
		for i_vertex in range(len(self.vertices)):
			x = self.vertices[i_vertex][0]
			y = self.vertices[i_vertex][1]
			flow_vec = flow[math.floor(y)][math.floor(x)]
			x += flow_vec[0] * step_size
			y += flow_vec[1] * step_size
			self.vertices[i_vertex][0] = x
			self.vertices[i_vertex][1] = y


	# draw timelines and return the image
	def draw_lines(self, img, color):
		ret_img = img.copy()

		# # draw initial line
		# for i_vertex in range(len(self.vertices_origin) - 1):
		# 	x1 = math.floor(self.vertices_origin[i_vertex][0])
		# 	y1 = math.floor(self.vertices_origin[i_vertex][1])
		# 	x2 = math.floor(self.vertices_origin[i_vertex + 1][0])
		# 	y2 = math.floor(self.vertices_origin[i_vertex + 1][1])
		# 	ret_img = cv2.circle(ret_img, (x1, y1), 4, (70, 70, 70), -1)
		# 	ret_img = cv2.line(ret_img, (x1, y1), (x2, y2), (70, 70, 70), 4) 
			
		# 	# draw the last point
		# 	if i_vertex == len(self.vertices_origin) - 1:
		# 		ret_img = cv2.circle(ret_img, (x2, y2), 4, (70, 70, 70), -1)

		# draw moving vertices
		for i_vertex in range(len(self.vertices)):
			x1 = math.floor(self.vertices[i_vertex][0])
			y1 = math.floor(self.vertices[i_vertex][1])
			ret_img = cv2.circle(ret_img, (x1, y1), 4, color, -1)
		
		return ret_img

def add_color_wheel(img, wheel):
	wheel_resized = cv2.resize(wheel, (50, 50))
	img[10:10+wheel_resized.shape[0], 10:10+wheel_resized.shape[1]] = wheel_resized

def change_vertices_step(new_points, old_points, dt, max_dist, is_norm):

	for i_point in range(len(new_points)):
		old_x, old_y = old_points[i_point]
		new_x, new_y = new_points[i_point]

		x, y = new_x - old_x, new_y - old_y
		
		theta = math.atan2(y,x)

		if is_norm:
			length = 1
		else:
			length = math.sqrt(x*x + y*y)
		

		if length > max_dist:
			new_points[i_point][0] = old_x
			new_points[i_point][1] = old_y
			continue

		new_points[i_point][0] = old_x + math.cos(theta) * length * dt
		new_points[i_point][1] = old_y + math.sin(theta) * length * dt


	return new_points

def main(video, outpath, height, window_size, correct_perspective):

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


	# Init timeline
	cv2.namedWindow('click to draw timelines')
	cv2.setMouseCallback('click to draw timelines', draw_lines)

	cv2.imshow("click to draw timelines", resized_frame)

	# wait for clicks until enter is hit
	while(1):
		k = cv2.waitKey()
		if k == 13:
			break


	# line_pos.append(np.array([251, 214]))
	# line_pos.append(np.array([439, 172]))

	# initialize timelines
	timelines = []
	for i_vertex in range(0, len(line_pos) - 1, 2):
		timeline = Timeline(line_pos[i_vertex], line_pos[i_vertex + 1], 10)
		timelines.append(timeline)


	lk_params = dict(winSize = (15, 15),
					maxLevel = 5,
					criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

	# convert to gray
	old_gray = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)

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


		# end reading timer, and record
		end_read_time = time.time()
		timers["reading"].append(end_read_time - start_read_time)


		# start pre-process timer
		start_pre_time = time.time()


		# end pre-process timer, and record
		end_pre_time = time.time()
		timers["pre-process"].append(end_pre_time - start_pre_time)

		# start optical flow timer
		start_of = time.time()



		gray_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)

		# create new timeline
		if frame_count == 0:
			for timeline in timelines:
				timeline.birth_line()


		for timeline in timelines:
			old_points = timeline.vertices
			new_points, status, error = cv2.calcOpticalFlowPyrLK(old_gray, gray_frame, old_points, None, **lk_params)
			timeline.vertices = change_vertices_step(new_points, old_points, 0.5, 30, False)


		# end of timer, and record
		end_of = time.time()
		timers["optical flow"].append(end_of - start_of)

		# start post-process timer
		start_post_time = time.time()

		color_timelines = [(255,0,0), (0,255,0), (0,0,255), (255,255,0), (0,255,255), (255,0,255)]

		# draw timelines
		frame_timelines = resized_frame.copy()

		for i_timeline in range(len(timelines)):
			frame_timelines = timelines[i_timeline].draw_lines(frame_timelines, color_timelines[i_timeline])

		frame_timelines = cv2.putText(frame_timelines, str(frame_count), (30,30), cv2.FONT_HERSHEY_SIMPLEX,  
                   0.8, (255,255,255), 1, cv2.LINE_AA) 


		old_gray = gray_frame.copy()



		# end post-process timer, and record
		end_post_time = time.time()
		timers["post-process"].append(end_post_time - start_post_time)

		# end full pipeline timer, and record
		end_full_time = time.time()
		timers["full pipeline"].append(end_full_time - start_full_time)

		# visualization
		cv2.imshow("timelines", frame_timelines)

		video_out.write(frame_timelines)

		k = cv2.waitKey(1)
		if k == 27:
			break


		if k == 115:
			cv2.imwrite(outpath + "/" + filename + "/flow_average.jpg", cpu_flow_average_bgr)
			cv2.imwrite(outpath + "/" + filename + "/flow_average_overlay.jpg", cpu_flow_overlay)
			

		frame_count += 1

	print(outpath + "/" + filename + "_timelines.jpg")

	video_out.release()
	cv2.imwrite(outpath + "/" + filename + "_timelines.jpg", frame_timelines)

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

	# parsing script arguments
	args = parser.parse_args()
	video = args.video
	outpath = args.out
	height = args.height
	window_size = args.window
	correct_perspective = args.correct_perspective


	# run pipeline
	main(video, outpath, height, window_size, correct_perspective)