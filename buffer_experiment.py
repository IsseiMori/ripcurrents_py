# python contours.py --video beach.mp4 --out . --height 480 --window 900


# Unresolved Bugs

import os
import numpy as np 
import cv2
import argparse
import time
import math
import matplotlib.pyplot as plt
import copy

ITERATION = 1000
BUFFER_SIZE = 10
MAT_SIZE = 50

generated_mat = np.zeros((ITERATION, MAT_SIZE, MAT_SIZE))
mat_buffer = np.zeros((BUFFER_SIZE, MAT_SIZE, MAT_SIZE))
buffer_sum = np.zeros((MAT_SIZE, MAT_SIZE))

print(ITERATION, BUFFER_SIZE, MAT_SIZE)

for i in range(ITERATION):
	generated_mat[i] = np.random.randint(5, size=(MAT_SIZE,MAT_SIZE))

# Use buffer to actively add and subtruct from the sum
start_timer = time.time()

for i in range(ITERATION):
	new_mat = generated_mat[i]

	# Buffer position to update in this iteration
	current_buffer_i = i % BUFFER_SIZE

	# Remove the oldest data in the buffer from the sum
	buffer_sum -= mat_buffer[current_buffer_i] / BUFFER_SIZE

	# Update the buffer
	mat_buffer[current_buffer_i] = new_mat

	# Add new data to the sum
	buffer_sum += new_mat / BUFFER_SIZE

end_timer = time.time()
print(end_timer - start_timer)

print(buffer_sum)

# reset buffer
buffer_sum = np.zeros((MAT_SIZE, MAT_SIZE))

# Simply calculate the sum everytime using buffer

start_timer = time.time()

for i in range(ITERATION):
	new_mat = generated_mat[i]

	# Buffer position to update in this iteration
	current_buffer_i = i % BUFFER_SIZE

	# Update the buffer
	mat_buffer[current_buffer_i] = new_mat

	buffer_sum = mat_buffer.sum(axis=0) / BUFFER_SIZE

end_timer = time.time()
print(end_timer - start_timer)

print(buffer_sum)

