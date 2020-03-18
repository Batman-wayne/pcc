# Point Cloud Computing -- ECE 408
# Author: Harris Mohamed 

import sys
import os
import numpy as np
import heapq

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 408 CODE BEGINS HERE. Above lines are for grabbing the AWS data, which won't work for you.
# TODO: Loop through coordinate.txt and add it to actual_coordinates. The code below is 
# expecting a list of lists, like [[x0, y0, z0], [x1, y1, z2], ....]
actual_coordinates = []

# Helper function to find distance between points 
def distPoints(p0, p1):
    return (np.sqrt((p0[0] - p1[0])**2 + (p0[1] - p1[1])**2 + (p0[2] - p1[2])**2))

POINTS_TO_EXPLORE = 10  # Number of points to explore. An LSRP is calculated, and then the normal vector is extracted from there.
points_dict = {}
nearest_neighbors = {}

count = 0
# Loop through all points, compute normal vector for each cluster of points 
for coordinate in actual_coordinates:
    # Explore the nearest n neighbors. Doing this without any optimizations as of now.
    # if count == 10:   # There are 24,330 points in the file. Takes my computer a little over a minute to compute for all of them. Use this to test on smaller number of points
    #     break
    count = count + 1
    actual_points = []
    actual_x = []
    actual_y = []
    actual_z = []
    n_points = []
    heapq.heapify(n_points)
    mod_list = actual_coordinates.copy()
    mod_list.remove(coordinate)
    for point in mod_list:
        heapq.heappush(n_points, (distPoints(coordinate, point), point))
    
    # Find the nearest 10 points. Not optimized at all.
    for i in range(POINTS_TO_EXPLORE):
        curr_point = heapq.heappop(n_points)[1]
        actual_points.append(curr_point)
        actual_x.append(curr_point[0])
        actual_y.append(curr_point[1])
        actual_z.append(curr_point[2])

        actual_coordinates.remove(curr_point)
    
    # Find the plane classifying the 10 points 
    tmp_A = []
    tmp_B = []
    for i in range(POINTS_TO_EXPLORE):
        tmp_A.append([actual_x[i], actual_y[i], 1])
        tmp_B.append(actual_z[i])
    b = np.matrix(tmp_B).T 
    A = np.matrix(tmp_A)
    fit = (A.T * A).I * A.T * b

    # fit is the numpy matrix holding the plane parameters. fit[0,0] is A, fit[1,0] is B, fit[2,0] is C in the equation Ax + By + Cz = D. 
    mag = np.sqrt(fit[0,0]**2 + fit[1,0]**2 + fit[2,0]**2)
    nearest_neighbors[str(coordinate)] = actual_points

    # Here we are normalizing the normal vector of the plane
    normalized = []
    normalized.append((fit[0,0] / mag))
    normalized.append((fit[1,0] / mag))
    normalized.append((fit[2,0] / mag))
    points_dict[str(coordinate)] = normalized

# Might need to tweak this, this is what determines if two planes are part of the same feature or not.
# TODO: Finish the feature identification.
ANGLE_LOW_THRESHOLD = 80
num_features = 0
features = {}
# Now loop through the dictionary and average normal vectors
for key, value in points_dict.items():
    empty = []
    if len(features) == 0:
        empty.append(0)
        empty.append(value)
        features[num_features] = empty 
    else:
        for k, v in features.items():
            angle_compare = v[0]
            angle = np.degrees(np.arccos(angle_compare[0]*v[0] + angle_compare[1]*v[1] + angle_compare[2]*v[2]))
            if (angle < 15):


        curr = features[0]
        # print(curr)
        # print("Current plane: ", value)
        angle = np.degrees(np.arccos(curr[0]*value[0] + curr[1]*value[1] + curr[2]*value[2]))
        print(angle)
    
    
# Left these comments in, they can be used to visualize the outputs. 
# print(type(fit))
# print(fit)
# errors = b - A * fit
# residual = np.linalg.norm(errors)

# plt.figure()
# ax = plt.subplot(111, projection='3d')
# ax.scatter(actual_x, actual_y, actual_z, color='b') 

# xlim = ax.get_xlim()
# ylim = ax.get_ylim()
# X,Y = np.meshgrid(np.arange(xlim[0], xlim[1]),
#                   np.arange(ylim[0], ylim[1]))
# Z = np.zeros(X.shape)
# for r in range(X.shape[0]):
#     for c in range(X.shape[1]):
#         Z[r,c] = fit[0] * X[r,c] + fit[1] * Y[r,c] + fit[2]
# ax.plot_wireframe(X,Y,Z, color='k')

# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.set_zlabel('z')
# plt.show()

# break
# # plt.figure()
# # ax = plt.subplot(111, projection='3d')
# # ax.scatter(actual_x, actual_y, actual_z, color='b') 

# xlim = ax.get_xlim()
# ylim = ax.get_ylim()
# X,Y = np.meshgrid(np.arange(xlim[0], xlim[1]),
#                   np.arange(ylim[0], ylim[1]))
# Z = np.zeros(X.shape)
# for r in range(X.shape[0]):
#     for c in range(X.shape[1]):
#         Z[r,c] = fit[0] * X[r,c] + fit[1] * Y[r,c] + fit[2]
# ax.plot_wireframe(X,Y,Z, color='k')

# # ax.set_xlabel('x')
# # ax.set_ylabel('y')
# # ax.set_zlabel('z')
# # plt.show()
    
