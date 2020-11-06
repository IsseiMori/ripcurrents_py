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
		cpu_flow_average_magnitude = np.sqrt(cpu_flow_average_magnitude)


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


		'''
		find dominant direction and opposite direction
		'''
		angle_masked = (mask_img > 0) * cpu_flow_average_angle
		hist,bins = np.histogram(angle_masked,bins = [0,45,90,130,180,225,270,315,360]) 
		bin_low = bins[np.argmax(hist)]
		bin_high = bins[np.argmax(hist)+1]
		max_hist_count = hist[np.argmax(hist)]

		opp_bin_low =  bins[np.argmax(hist) + 4 - 8 if np.argmax(hist) > 8 else np.argmax(hist) + 4]
		opp_bin_high = bins[np.argmax(hist) + 1 + 4 - 8 if np.argmax(hist) + 1 > 8 else np.argmax(hist) + 4 + 1]
		
		# force opp_bin_low to be lower
		if opp_bin_low > opp_bin_high:
			tmp = opp_bin_low
			opp_bin_low = opp_bin_high
			opp_bin_highf = tmp



		'''
		find mean velocity in max bin and subtruct from flow
		'''
		max_hist_flow_x = ((angle_masked > bin_low) * (angle_masked < bin_high) * cpu_flow_average[:,:,0]).sum() / float(max_hist_count)
		max_hist_flow_y = ((angle_masked > bin_low) * (angle_masked < bin_high) * cpu_flow_average[:,:,1]).sum() / float(max_hist_count)
		
		'''
		find max velocity in max bin and subtruct from flow
		'''
		# max_hist_flow_x_max = ((angle > bin_low) * (angle < bin_high) * cpu_flow_average[:,:,0]).max()
		# max_hist_flow_x_min = ((angle > bin_low) * (angle < bin_high) * cpu_flow_average[:,:,0]).min()
		# max_hist_flow_x =  max_hist_flow_x_max if abs(max_hist_flow_x_max) > abs(max_hist_flow_x_min) else max_hist_flow_x_min
		# max_hist_flow_y_max = ((angle > bin_low) * (angle < bin_high) * cpu_flow_average[:,:,1]).max()
		# max_hist_flow_y_min = ((angle > bin_low) * (angle < bin_high) * cpu_flow_average[:,:,1]).min()
		# max_hist_flow_y = max_hist_flow_y_max if abs(max_hist_flow_y_max) > abs(max_hist_flow_y_min) else max_hist_flow_y_min


		'''
		subtract above flow
		'''
		cpu_flow_mean_sub = cpu_flow_average.copy()
		cpu_flow_mean_sub[:,:,0] -= max_hist_flow_x
		cpu_flow_mean_sub[:,:,1] -= max_hist_flow_y

		cpu_flow_mean_sub[:,:,0] = (mask_img > 0) * cpu_flow_mean_sub[:,:,0]
		cpu_flow_mean_sub[:,:,1] = (mask_img > 0) * cpu_flow_mean_sub[:,:,1]


		cpu_flow_mean_sub_angle, cpu_flow_mean_sub_magnitude = calc_angle_from_flow_cpu(cpu_flow_mean_sub)



		'''
		Mask out everything that is not in opposite direction
		'''
		cpu_flow_mean_sub_masked = cpu_flow_mean_sub.copy()
		cpu_flow_mean_sub_masked[:,:,0] = ((cpu_flow_mean_sub_angle > opp_bin_low) * (cpu_flow_mean_sub_angle < opp_bin_high)) * cpu_flow_mean_sub[:,:,0]
		cpu_flow_mean_sub_masked[:,:,1] = ((cpu_flow_mean_sub_angle > opp_bin_low) * (cpu_flow_mean_sub_angle < opp_bin_high)) * cpu_flow_mean_sub[:,:,1]


		cpu_flow_mean_sub_masked_angle, cpu_flow_mean_sub_masked_magnitude = calc_angle_from_flow_cpu(cpu_flow_mean_sub_masked)


		cpu_flow_max_angle, cpu_flow_max_magnitude = calc_angle_from_flow_cpu(cpu_flow_max)
		cpu_flow_max_bgr = calc_bgr_from_angle_magnitude(cpu_flow_max_angle, cpu_flow_max_magnitude)

		cpu_flow_average_bgr = calc_bgr_from_angle_magnitude(cpu_flow_average_angle, cpu_flow_average_magnitude)
		cpu_flow_average_sub_bgr = calc_bgr_from_angle_magnitude(cpu_flow_mean_sub_angle, cpu_flow_mean_sub_magnitude)
		cpu_flow_average_sub_bgr_strong = calc_bgr_from_angle_magnitude(cpu_flow_mean_sub_angle, np.ones_like(cpu_flow_mean_sub_masked_angle, np.float32))
		cpu_flow_average_root_bgr = calc_bgr_from_angle_magnitude_root(cpu_flow_average_angle, cpu_flow_average_magnitude)
		cpu_flow_mean_sub_masked_root_bgr = calc_bgr_from_angle_magnitude_root(cpu_flow_mean_sub_masked_angle, cpu_flow_mean_sub_masked_magnitude)
		cpu_flow_mean_sub_masked_bgr = calc_bgr_from_angle_magnitude(cpu_flow_mean_sub_masked_angle, cpu_flow_mean_sub_masked_magnitude)

		cpu_flow_mean_sub_masked_bgr_strong = calc_bgr_from_angle_magnitude(cpu_flow_mean_sub_masked_angle, np.ones_like(cpu_flow_mean_sub_masked_angle, np.float32))



		# update previous_frame value
		gpu_previous = gpu_current



		'''
		find contours on sub flow
		'''
		flow_rip_gray = cv2.cvtColor(cpu_flow_mean_sub_masked_bgr,cv2.COLOR_BGR2GRAY)
		ret,thresh = cv2.threshold(flow_rip_gray,0,255,0)
		contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
		contours_selected = []
		contours_touched = []
		contours_tiny = []
		for cnt in contours:
			# Boundary Check
			boundary_touched = False
			for cnt_p in cnt:
				cnt_p_mask_color = mask_img[cnt_p[0][1]][cnt_p[0][0]];
				if cnt_p_mask_color < 255:
					boundary_touched = True
					break

			# Minimum area requirement check
			if not boundary_touched:
				area = cv2.contourArea(cnt)
				if area > 50:
					contours_selected.append(cnt)
				else: 
					contours_tiny.append(cnt)
			else:
				contours_touched.append(cnt)
				
		flow_contours = cv2.drawContours(resized_frame, contours_selected, -1, (0,255,0), 2)
		flow_contours = cv2.drawContours(flow_contours, contours_tiny, -1, (125,0,0), 2)
		flow_contours = cv2.drawContours(flow_contours, contours_touched, -1, (125,0,0), 2)

		flow_contours = cv2.putText(flow_contours, str(frame_count), (50, 50) , cv2.FONT_HERSHEY_SIMPLEX ,  
					   1, (255,255,255), 2, cv2.LINE_AA) 


		'''
		find contours on sub root flow
		'''
		# flow_rip_gray = cv2.cvtColor(cpu_flow_average_root_bgr,cv2.COLOR_BGR2GRAY)
		# ret,thresh = cv2.threshold(flow_rip_gray,0,255,0)
		# contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
		# contours_selected = []
		# contours_touched = []
		# contours_tiny = []
		# for cnt in contours:
		# 	# Boundary Check
		# 	boundary_touched = False
		# 	for cnt_p in cnt:
		# 		cnt_p_mask_color = mask_img[cnt_p[0][1]][cnt_p[0][0]];
		# 		if cnt_p_mask_color < 255:
		# 			boundary_touched = True
		# 			break

		# 	# Minimum area requirement check
		# 	if not boundary_touched:
		# 		area = cv2.contourArea(cnt)
		# 		if area > 50:
		# 			contours_selected.append(cnt)
		# 		else: 
		# 			contours_tiny.append(cnt)
		# 	else:
		# 		contours_touched.append(cnt)
				
		# flow_contours_root = cv2.drawContours(resized_frame, contours_selected, -1, (0,255,0), 2)
		# flow_contours_root = cv2.drawContours(flow_contours_root, contours_tiny, -1, (125,0,0), 2)
		# flow_contours_root = cv2.drawContours(flow_contours_root, contours_touched, -1, (125,0,0), 2)

		# flow_contours_root = cv2.putText(flow_contours_root, str(frame_count), (50, 50) , cv2.FONT_HERSHEY_SIMPLEX ,  
		# 			   1, (255,255,255), 2, cv2.LINE_AA) 





		# end post-process timer, and record
		end_post_time = time.time()
		timers["post-process"].append(end_post_time - start_post_time)

		# end full pipeline timer, and record
		end_full_time = time.time()
		timers["full pipeline"].append(end_full_time - start_full_time)

		# visualization
		#cv2.imshow("original", frame)
		cv2.imshow("flow", cpu_flow_average_bgr)
		cv2.imshow("flow sub", cpu_flow_average_sub_bgr)
		cv2.imshow("flow sub root strong", cpu_flow_average_sub_bgr_strong)
		cv2.imshow("flow sub root mask", cpu_flow_mean_sub_masked_root_bgr)
		# cv2.imshow("flow sub", cpu_flow_mean_sub_masked_bgr)
		cv2.imshow("contours", flow_contours)
		#cv2.imshow("contours root", flow_contours_root)
		#cv2.imshow("max", cpu_flow_max_bgr)
		k = cv2.waitKey(1)
		if k == 27:
			break

		if frame_count == 1500:
			break


		if k == 115:
			#cv2.imwrite(outpath + "/contours_root.jpg", flow_contours_root)
			# cv2.imwrite(outpath + "/flow_sub.jpg", cpu_flow_mean_sub_masked_bgr)
			cv2.imwrite(outpath + "/" + filename + "/flow_average_unit.jpg", cpu_flow_average_bgr)
			cv2.imwrite(outpath + "/" + filename + "/flow_sub_unit.jpg", cpu_flow_average_sub_bgr)
			cv2.imwrite(outpath + "/" + filename + "/flow_sub_root_strong_unit.jpg", cpu_flow_average_sub_bgr_strong)
			cv2.imwrite(outpath + "/" + filename + "/flow_sub_root_mask_unit.jpg", cpu_flow_mean_sub_masked_root_bgr)
			cv2.imwrite(outpath + "/" + filename + "/contours_unit.jpg", flow_contours)

		frame_count += 1

	cv2.imwrite(outpath + "/" + filename + "/flow_average_unit.jpg", cpu_flow_average_bgr)
	cv2.imwrite(outpath + "/" + filename + "/flow_sub_unit.jpg", cpu_flow_average_sub_bgr)
	cv2.imwrite(outpath + "/" + filename + "/flow_sub_root_strong_unit.jpg", cpu_flow_average_sub_bgr_strong)
	cv2.imwrite(outpath + "/" + filename + "/flow_sub_root_mask_unit.jpg", cpu_flow_mean_sub_masked_root_bgr)
	cv2.imwrite(outpath + "/" + filename + "/contours_unit.jpg", flow_contours)

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