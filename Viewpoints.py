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
import os
import numpy as np
from scipy.spatial.transform.rotation import Rotation as R
from scipy.cluster import hierarchy as shc
from scipy.spatial.distance import pdist, squareform
from numpy.linalg import norm
import math
from stl import mesh as stl_mesh
from mpl_toolkits import mplot3d
from matplotlib import pyplot, cm
from matplotlib import pyplot as plt
from InvokeLKH import writeTSPLIBfile_FE, run_LKHsolver_cmd, rm_solution_file_cmd, copy_toTSPLIBdir_cmd


# Initialize constants
camera_position = np.array([30, 20, 30])
BIG_BEN = "Mesh/BigBen.stl"
TANK = "Mesh/Elevated_Tank.STL"
INCIDENCE_ANGLE = np.pi / 6  # facet field of view
FOV = np.pi / 3  # camera field of view
DMIN = 5
DMAX = 10
D_CLUSTER = 0.8  # [m]
CWD = os.path.dirname(os.path.abspath(__file__))
model = TANK  # choose the mesh model


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

    normals = dot_v(mesh.normals, unit_vectors) # values < 0 are pointing towards camera
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


def load_mesh(model):
    """
    Load the STL files and add the vectors to the plot
    \nReturns: 
    mesh: Mesh object from numpy-stl   
    facets: (n, 3, 3) coordinate array of triangular facets
    incidence_normals: incidence planes created by facet boundary 
    mesh_centers: numpy array of facet center coordinates
    n: number of facets
    """
    if not exists(os.path.join(CWD, model)):
        raise ImportError("{} not found in current file folder.".format(model))

    your_mesh = stl_mesh.Mesh.from_file(os.path.join(CWD, model))
    n = len(your_mesh.points)

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

    return your_mesh, facets, incidence_normals, mesh_centers, n


def create_obstacles(facets):
    """
    Create Obstaces in 2D
    Returns 2D array of XY obstacle points
    """
    obstacles = []
    for f in facets:
        for p0, p1 in ((0, 1), (1, 2), (2, 0)):
            dx = f[p1, 0] - f[p0, 0]
            dy = f[p1, 1] - f[p0, 1]
            d = np.hypot(dx, dy)
            theta = np.arctan2(dy, dx)
            x = f[p0, 0:2]
            
            D = 0.25
            nstep = int(d / D)
            obs = np.ones((nstep, 2)) * x
            d_vec = np.vstack([np.arange(nstep), np.arange(nstep)]).T * np.array([D*np.cos(theta), D*np.sin(theta)])
            obs = obs + d_vec
            obstacles.append(obs)

    obstacles = np.concatenate(obstacles)
    return obstacles


def obstacle_perimeter(obstacles):
    """
    Create obstacle perimiter 
    """
    perim = 5
    D = 0.25
    xmin = np.min(obstacles[:,0]) - perim  # [m]
    xmax = np.max(obstacles[:,0]) + perim  # [m]
    ymin = np.min(obstacles[:,1]) - perim  # [m]
    ymax = np.max(obstacles[:,1]) + perim  # [m]

    top = np.arange(xmin, xmax, step=D).reshape(-1,1)
    top = np.concatenate([top, np.ones(top.shape) * ymax], axis=1)
    right = np.arange(ymax, ymin, step=-D).reshape(-1,1)
    right = np.concatenate([np.ones(right.shape) * xmax, right], axis=1)
    bottom = np.arange(xmax, xmin, step=-D).reshape(-1,1)
    bottom = np.concatenate([bottom, np.ones(bottom.shape) * ymin], axis=1)
    left = np.arange(ymin, ymax, step=D).reshape(-1,1)
    left = np.concatenate([np.ones(left.shape) * xmin, left], axis=1)

    obstacles = np.concatenate([obstacles, np.concatenate([top, right, bottom, left])])
    return obstacles


def viewpoint_clusters(viewpoints, d_cluster=D_CLUSTER):
    """
    Compute Cost Matrix for distance between each point
    \nReturns:
    cluster_groups: a list of group numbers for each viewpoint
    cluster_centers: a 2D XY array of cluster centers
    """
    distMatrix = pdist(viewpoints)
    Z = shc.average(distMatrix)
    cluster_groups = shc.fcluster(Z, D_CLUSTER, criterion='distance')

    n_clusters = max(cluster_groups)
    cluster_centers = np.zeros((n_clusters, 2))
    for c in range(n_clusters):
        group = cluster_groups == c+1
        view_group = viewpoints[group, 0:2]
        cluster_centers[c] = np.median(view_group, axis=0)

    return cluster_groups, cluster_centers


def main():

    # Load the model
    mesh_model, facets, incidence_normals, mesh_centers, n = load_mesh(model)

    # Initialize viewpoints along facet normal
    viewpoints = mesh_model.v0 + mesh_model.normals

    n_lengths = np.linalg.norm(mesh_model.normals, axis=-1)
    print "Mesh Normals: ", n_lengths

    rand_idx = np.random.randint(viewpoints.shape[0], size=10)
    camera_positions = viewpoints[rand_idx]
    visible = [visible_facets(pos, mesh_model) for pos in camera_positions]
    visible = np.unique(np.concatenate(visible))

    # Rasterization algorithm to determine if facet is not obscured by another object
    # TODO: How to determine whether a facet is in front of another facet given a camera point

    print("{} facets out of {} are visible to the {} viewpoints".format(visible.shape[0], mesh_model.v0.shape[0], camera_positions.shape[0]))

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
    polygon = mplot3d.art3d.Poly3DCollection(mesh_model.vectors, facecolors=colors)

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
    cluster_groups = viewpoint_clusters(viewpoints)
    colors = cm.get_cmap()
    axes.scatter(xs=viewpoints[:, 0], ys=viewpoints[:, 1], zs=viewpoints[:, 2], marker='o', c=cluster_groups, cmap='RdYlBu')

    # plot_tsp_path(axes, viewpoints)


    # Auto scale to the mesh size
    scale = mesh_model.points.flatten()
    axes.auto_scale_xyz(scale, scale, scale)
    axes.set_xlabel('X')
    axes.set_ylabel('Y')
    axes.set_zlabel('Z')

    # Show the plot to the screen
    pyplot.show()

    obstacles = create_obstacles(facets)

    # Plot 2D obstacles
    fig, ax = plt.subplots()
    ax.plot(obstacles[:,0], obstacles[:,1], ".k")
    ax.plot(viewpoints[:,0], viewpoints[:,1], '+g')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    pyplot.show()


if __name__ == "__main__":
    main()