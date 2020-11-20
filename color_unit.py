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


def main(video, outpath, height, window_size):

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
	if not os.path.exists(outpath + "/" + filename):
		os.makedirs(outpath + "/" + filename)

	# init video capture with video
	cap = cv2.VideoCapture(video)

	# get default video FPS
	fps = cap.get(cv2.CAP_PROP_FPS)

	# get total number of video frames
	num_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

	# read the first frame
	ret, previous_frame = cap.read()

	# proceed if frame reading was successful
	if not ret: return

	# width after resize
	width = math.floor(previous_frame.shape[1] * 
			height / (previous_frame.shape[0]))

	# load mask img
	mask_img = cv2.imread("mask3.png", 0)
	# make sure the size is correct. This should not be necessary
	mask_img = cv2.resize(mask_img, (width, height))
	mask_img_bgr = cv2.cvtColor(mask_img, cv2.COLOR_GRAY2BGR)

	# resize frame
	frame = cv2.resize(previous_frame, (width, height))
	# frame[:,:,0] = mask_img * frame[:,:,0]
	# frame[:,:,1] = mask_img * frame[:,:,1]
	# frame[:,:,2] = mask_img * frame[:,:,2] 

	# upload resized frame to GPU
	gpu_frame = cv2.cuda_GpuMat()
	gpu_frame.upload(frame)

	# convert to gray
	previous_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

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

	cpu_flow_max = None
	cpu_flow_mag_max = None



	frame_count = 0
	while True:
		# start full pipeline timer
		start_full_time = time.time()

		# start reading timer
		start_read_time = time.time()

		# capture frame-by-frame
		ret, frame = cap.read()

		# if frame reading was not successful, break
		if not ret:
			break

		resized_frame = cv2.resize(frame, (width, height))
		# resized_frame[:,:,0] = mask_img * resized_frame[:,:,0]
		# resized_frame[:,:,1] = mask_img * resized_frame[:,:,1]
		# resized_frame[:,:,2] = mask_img * resized_frame[:,:,2] 

		# upload frame to GPU
		gpu_frame.upload(frame)

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
		cpu_flow = zero_edge_flow(cpu_flow)



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


		cpu_flow_average_bgr = calc_bgr_from_angle_magnitude(cpu_flow_average_angle, cpu_flow_average_magnitude)
		cpu_flow_average_bgr_strong = calc_bgr_from_angle_magnitude(cpu_flow_average_angle, np.ones_like(cpu_flow_average_angle, np.float32))


		cpu_flow_overlay = cpu_flow_average_bgr.copy()
		cv2.addWeighted(cpu_flow_average_bgr, 1, resized_frame, 1, 0, cpu_flow_overlay)

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
		cv2.imshow("flow", cpu_flow_average_bgr)
		cv2.imshow("flow sub", cpu_flow_average_bgr_strong)
		cv2.imshow("flow overlay", cpu_flow_overlay)

		k = cv2.waitKey(1)
		if k == 27:
			break

		if frame_count == 1500:
			break


		if k == 115:
			cv2.imwrite(outpath + "/" + filename + "/flow_average_unit.jpg", cpu_flow_average_bgr)
			cv2.imwrite(outpath + "/" + filename + "/flow_average_unit_strong.jpg", cpu_flow_average_bgr_strong)
			cv2.imwrite(outpath + "/" + filename + "/flow_average_unit_overlay.jpg", cpu_flow_overlay)

		frame_count += 1

	cv2.imwrite(outpath + "/" + filename + "/flow_average_unit.jpg", cpu_flow_average_bgr)
	cv2.imwrite(outpath + "/" + filename + "/flow_average_unit_strong.jpg", cpu_flow_average_bgr_strong)
	cv2.imwrite(outpath + "/" + filename + "/flow_average_unit_overlay.jpg", cpu_flow_overlay)


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

	# parsing script arguments
	args = parser.parse_args()
	video = args.video
	outpath = args.out
	height = args.height
	window_size = args.window


	# run pipeline
	main(video, outpath, height, window_size)