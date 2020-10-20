"""
Visual computational geometry and viewpoint path constrained optimization.
Andrew Washburn, Prateek Arora, Nikhil Khedekar
University of Nevada Reno
CS791 Special Topics (Robotics)
Instructor: Christos Papachristos
Fall 2020
"""

# Imports
from os.path import exists
import numpy as np
from scipy.spatial.transform.rotation import Rotation as R
from scipy.cluster import hierarchy as shc
from scipy.spatial.distance import pdist, squareform
from numpy.linalg import norm
import math
from stl import mesh as stl_mesh
from mpl_toolkits import mplot3d
from matplotlib import pyplot, cm
from InvokeLKH import writeTSPLIBfile_FE, run_LKHsolver_cmd, rm_solution_file_cmd, copy_toTSPLIBdir_cmd


# Initialize constants
camera_position = np.array([30, 20, 30])
BIG_BEN = "Mesh/BigBen.stl"
TANK = "Mesh/Elevated_Tank.STL"
INCIDENCE_ANGLE = np.pi / 6  # facet field of view
FOV = np.pi / 3  # camera field of view
DMIN = 5
DMAX = 10

model = TANK  # choose the mesh model
if not exists(model):
    raise FileNotFoundError(f"{model} not found in current file folder.")


# Helpers
def normalize(vector):
    return vector / np.linalg.norm(vector, axis=1, keepdims=True)


def dot_v(directions1, directions2):
    return np.sum(directions1 * directions2, axis=1)


def incidence_plane(facet, normal):
    """
    facets is a (3 x 3) array of triangle points in 3D space
    normals is (3, ) array corresponds to each triangle's outward normals
    """
    plane_normals = np.zeros((3, 3))
    for p0, p1 in ((0, 1), (1, 2), (2, 0)):
        q = facet[p1] - facet[p0]
        q = q / norm(q)
        m = R.from_rotvec(INCIDENCE_ANGLE * q)  # euler vector
        n = m.apply(normal)
        plane_normals[p0] = n
    return plane_normals


def visible_facets(viewpoint, mesh):
    plane_origin = np.array([0, 0, viewpoint[2]])
    # compute heading towards the xy origin
    camera_xy_direction = (plane_origin - viewpoint) / norm(plane_origin - viewpoint)

    # Calculate direction from the Camera to each facet's center
    unit_vectors = normalize(mesh.v0 - viewpoint)
    camera_angles = np.arccos(dot_v(camera_xy_direction, unit_vectors))
    visible_facets_idx = np.argwhere(camera_angles <= FOV)

    normals = dot_v(your_mesh.normals, unit_vectors) # values < 0 are pointing towards camera
    theta = np.arccos(normals)
    feasible_facets_idx = np.argwhere(theta[visible_facets_idx] >= (np.pi - INCIDENCE_ANGLE))
    return feasible_facets_idx[:,0]

def plot_tsp_path(axes, viewpoints):
    path = []
    with open("TSPLIB/CoveragePlanner.txt") as fh:
        on_tour = False
        for line in fh:
            line = line.strip()
            if on_tour:
                point = int(line)
                if point == -1:
                    on_tour = False
                else:
                    path.append(point - 1)

            elif not on_tour and line == "TOUR_SECTION":
                on_tour = True
            
            # print(line, "\t| ", on_tour)
    # plot TSP path lines
    for i in range(n-1):
        p = path[i]
        pn = path[i+1]
        axes.plot(
            xs=[viewpoints[p, 0], viewpoints[pn, 0]],
            ys=[viewpoints[p, 1], viewpoints[pn, 1]],
            zs=[viewpoints[p, 2], viewpoints[pn, 2]],
            color='green'
        )

# Load the STL files and add the vectors to the plot
your_mesh = stl_mesh.Mesh.from_file(model)
n = len(your_mesh.points)
# if model == TANK:
#     # rotate the mesh 90deg about z and 90 deg about y
#     your_mesh.rotate([0, 0, 1], math.radians(90))
#     your_mesh.rotate([0, 1, 0], math.radians(90))


# Compute triangle mesh center
mesh_centers = np.stack([np.average(your_mesh.x, axis=1), 
                        np.average(your_mesh.y, axis=1), 
                        np.average(your_mesh.z, axis=1)], 
                        axis=-1)

# Compute incidence planes
facets = your_mesh.points.reshape(-1, 3, 3)
incidence_normals = np.zeros(facets.shape)  # shape (n, 3, 3)
for i in range(facets.shape[0]):
    incidence_normals[i] = incidence_plane(facets[i], your_mesh.normals[i])

# Initialize viewpoints along facet normal
viewpoints = your_mesh.v0 + your_mesh.normals

# Compute Cost Matrix for distance between each point
distMatrix = pdist(viewpoints)
Z = shc.average(distMatrix)
cluster_groups = shc.fcluster(Z, 4, criterion='distance')
d1 = shc.dendrogram(Z)

# Run Traveling Salesman solver on initialized viewpoints
fname_tsp = "CoveragePlanner"
user_comment = "Compute a path between feasible viewpoints for a triangular mesh object"
# [fileID1,fileID2] = writeTSPLIBfile_FE(fname_tsp,CostMatrix,user_comment)
# run_LKHsolver_cmd(fname_tsp)
# copy_toTSPLIBdir_cmd(fname_tsp)
# rm_solution_file_cmd(fname_tsp)


rand_idx = np.random.randint(viewpoints.shape[0], size=10)
camera_positions = viewpoints[rand_idx]
visible = [visible_facets(pos, your_mesh) for pos in camera_positions]
visible = np.unique(np.concatenate(visible))

# Rasterization algorithm to determine if facet is not obscured by another object
# TODO: How to determine whether a facet is in front of another facet given a camera point

print("{} facets out of {} are visible to the {} viewpoints".format(visible.shape[0], n, camera_positions.shape[0]))



# Create a new plot
figure = pyplot.figure()
axes = mplot3d.Axes3D(figure)

# Set colors.
cmap = cm.get_cmap('Blues')
# colors = [cmap(-1 * normal) for normal in normals]
colors = np.ones(n)
colors[visible] = 0
colors = cmap(colors)
# Add polygon with view color to matplotlib figure
polygon = mplot3d.art3d.Poly3DCollection(your_mesh.vectors, facecolors=colors)

axes.add_collection3d(polygon)

# Add Camera to plot
# axes.scatter(xs=camera_positions[0], ys=camera_positions[1], zs=camera_positions[2], marker='o', color='red')

# Add line from camera to feasibile mesh center
# for idx in feasible_facets_idx:
#     axes.plot(
#         xs=[camera_position[0], mesh_centers[idx][0,0]],
#         ys=[camera_position[1], mesh_centers[idx][0,1]],
#         zs=[camera_position[2], mesh_centers[idx][0,2]],
#         color='green'
#     )
# Add marker to visible facets
# axes.scatter(xs=viewpoints[feasible_facets_idx, 0], 
#             ys=viewpoints[feasible_facets_idx, 1], 
#             zs=viewpoints[feasible_facets_idx, 2], 
#             marker='o', color='green')

# Show all viewpoints and Optimal path between them
colors = cm.get_cmap()
axes.scatter(xs=viewpoints[:, 0], ys=viewpoints[:, 1], zs=viewpoints[:, 2], marker='o', c=cluster_groups)
# axes.scatter(xs=your_mesh.v0[:, 0], ys=your_mesh.v0[:, 1], zs=your_mesh.v0[:, 2], marker='o', color='green')

# plot_tsp_path(axes, viewpoints)


# Auto scale to the mesh size
scale = your_mesh.points.flatten()
axes.auto_scale_xyz(scale, scale, scale)
axes.set_xlabel('X')
axes.set_ylabel('Y')
axes.set_zlabel('Z')

# Show the plot to the screen
pyplot.show()
