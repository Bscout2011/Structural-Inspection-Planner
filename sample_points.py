#!/usr/bin/env python3

import rospy
import numpy as np
import open3d as o3d
import copy
from os.path import exists

DEG2RAD = np.pi/180
RAD2DIG = 1/DEG2RAD

MIN_PROJECTION_DIST = 0.3 # m
MAX_PROJECTION_DIST = 1 # m
MAX_FEASIBLE_HEIGHT = 2
VOXEL_DOWNSAMPLE_SIZE = 1 # m
MAX_VISION_RANGE = 10 # m
MAX_VISION_ANGLE = 60 # deg

# Get the center of gravity of a given triangle
def getTriangleCOG(triangle):
    triangle_np = np.asarray(triangle)
    v1 = np.asarray(mesh.vertices[triangle_np[0]])
    v2 = np.asarray(mesh.vertices[triangle_np[1]])
    v3 = np.asarray(mesh.vertices[triangle_np[2]])
    cog = (v1 + v2 + v3)/3.0
    return cog

# Project the point outward along the normal at a random distance between MIN_PROJECTION_DIST and MAX_PROJECTION_DIST
def projectPointNormallyOutward(point, normal):
    return point + np.asarray(normal) * np.random.uniform(MIN_PROJECTION_DIST, MAX_PROJECTION_DIST)

# Return true if the normals are antiparallel up to MAX_VISION_ANGLE and the cog is within MAX_VISION_RANGE of the point
def isTriangleVisibleFromPoint(point, orientation, triangle_normal, cog):
    return np.dot(orientation, triangle_normal) < -np.cos(MAX_VISION_ANGLE * DEG2RAD) and np.linalg.norm(point - cog) < MAX_VISION_RANGE

def getRandomOrientation():
    orientation = np.random.uniform(0, 1, size=(1, 3))
    return orientation / np.linalg.norm(orientation)

# Read mesh file
model = "Mesh/Elevated_Tank.STL"  # choose the mesh model
if not exists(model):
    raise FileNotFoundError(f"{model} not found in current file folder.")

mesh = o3d.io.read_triangle_mesh(model)
# mesh.scale(0.0005, center = np.array([0,0,0]))

# Preprocessing
mesh.compute_vertex_normals()

# Visualize mesh
# o3d.visualization.draw_geometries([mesh])

sampled_points = o3d.geometry.PointCloud()
points = []
cogs = []
for i in range(len(mesh.triangles)):
    # Project the COG's of the triangles normally outward
    cog = getTriangleCOG(mesh.triangles[i])
    cogs.append(cog)
    projected_point = projectPointNormallyOutward(cog, mesh.triangle_normals[i])
    points.append(projected_point)

# Create pointcloud
points_np = np.asarray(points)
sampled_points.points = o3d.utility.Vector3dVector(points_np)

o3d.visualization.draw_geometries([sampled_points, mesh])

# Sparsify pointcloud
sampled_points = sampled_points.voxel_down_sample(voxel_size = VOXEL_DOWNSAMPLE_SIZE)

# Visualize mesh and pointcloud
o3d.visualization.draw_geometries([sampled_points, mesh])

feasible_points = []
infeasible_points = []

for i in range(len(sampled_points.points)):
    if sampled_points.points[i][2] < MAX_FEASIBLE_HEIGHT:
        feasible_points.append(np.asarray(sampled_points.points[i]))
    else:
        infeasible_points.append(sampled_points.points[i])

feasible_points_np = np.asarray(feasible_points)
feasible_pointcloud = o3d.geometry.PointCloud()
feasible_pointcloud.points = o3d.utility.Vector3dVector(feasible_points_np)
feasible_pointcloud.paint_uniform_color([1, 0.706, 0])

infeasible_points_np = np.asarray(infeasible_points)
infeasible_pointcloud = o3d.geometry.PointCloud()
infeasible_pointcloud.points = o3d.utility.Vector3dVector(infeasible_points_np)
infeasible_pointcloud.paint_uniform_color([0, 0.1, 1])

# Visualize mesh and pointcloud
o3d.visualization.draw_geometries([feasible_pointcloud, infeasible_pointcloud, mesh])

# point = np.asarray(sampled_points.points[0])
# orientation = np.asarray([1, 0, 0])
# for i in range(len(mesh.triangles)):
#     if isTriangleVisibleFromPoint(point, orientation, np.asarray(mesh.triangle_normals[i]), cogs[i]):
#         print("yes")
#         triangle = np.asarray(mesh.triangles[i])
#         for j in range(3):
#             vertex_index = np.asarray(mesh.vertices[triangle[j]])
#             mesh.vertex_colors[vertex_index] = np.array([1, 0.706, 0])
#     else:
#         print("no")

# print(getRandomOrientation())

# o3d.visualization.draw_geometries([sampled_points, mesh])

# Filter sampled points in collision with the mesh