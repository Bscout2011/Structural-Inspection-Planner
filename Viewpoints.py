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
TANK = "Mesh/ElevatedTank.stl"
INCIDENCE_ANGLE = np.pi / 6  # facet field of view
FOV = np.pi / 3  # camera field of view
DMIN = 5
DMAX = 10
D_CLUSTER = 0.8  # [m]
CWD = os.path.dirname(os.path.abspath(__file__))
model = TANK  # choose the mesh model



def rand_cmap(nlabels, type='bright', first_color_black=True, last_color_black=False, verbose=True):
    """
    Creates a random colormap to be used together with matplotlib. Useful for segmentation tasks
    :param nlabels: Number of labels (size of colormap)
    :param type: 'bright' for strong colors, 'soft' for pastel colors
    :param first_color_black: Option to use first color as black, True or False
    :param last_color_black: Option to use last color as black, True or False
    :param verbose: Prints the number of labels and shows the colormap. True or False
    :return: colormap for matplotlib
    """
    from matplotlib.colors import LinearSegmentedColormap
    import colorsys
    import numpy as np


    if type not in ('bright', 'soft'):
        print ('Please choose "bright" or "soft" for type')
        return

    if verbose:
        print('Number of labels: ' + str(nlabels))

    # Generate color map for bright colors, based on hsv
    if type == 'bright':
        randHSVcolors = [(np.random.uniform(low=0.0, high=1),
                          np.random.uniform(low=0.2, high=1),
                          np.random.uniform(low=0.9, high=1)) for i in range(nlabels)]

        # Convert HSV list to RGB
        randRGBcolors = []
        for HSVcolor in randHSVcolors:
            randRGBcolors.append(colorsys.hsv_to_rgb(HSVcolor[0], HSVcolor[1], HSVcolor[2]))

        if first_color_black:
            randRGBcolors[0] = [0, 0, 0]

        if last_color_black:
            randRGBcolors[-1] = [0, 0, 0]

        random_colormap = LinearSegmentedColormap.from_list('new_map', randRGBcolors, N=nlabels)

    # Generate soft pastel colors, by limiting the RGB spectrum
    if type == 'soft':
        low = 0.6
        high = 0.95
        randRGBcolors = [(np.random.uniform(low=low, high=high),
                          np.random.uniform(low=low, high=high),
                          np.random.uniform(low=low, high=high)) for i in range(nlabels)]

        if first_color_black:
            randRGBcolors[0] = [0, 0, 0]

        if last_color_black:
            randRGBcolors[-1] = [0, 0, 0]
        random_colormap = LinearSegmentedColormap.from_list('new_map', randRGBcolors, N=nlabels)

    # Display colorbar
    if verbose:
        from matplotlib import colors, colorbar
        from matplotlib import pyplot as plt
        fig, ax = plt.subplots(1, 1, figsize=(15, 0.5))

        bounds = np.linspace(0, nlabels, nlabels + 1)
        norm = colors.BoundaryNorm(bounds, nlabels)

        cb = colorbar.ColorbarBase(ax, cmap=random_colormap, norm=norm, spacing='proportional', ticks=None,
                                   boundaries=bounds, format='%1i', orientation=u'horizontal')

    return random_colormap


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
    
    Returns: 

    mesh: Mesh object from numpy-stl   
    facets: (n, 3, 3) coordinate array of triangular facets
    unit_normals: unit normals for each facet
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
    unit_normals = your_mesh.normals / np.linalg.norm(your_mesh.normals, axis=1)[:, None]

    return your_mesh, facets, unit_normals, mesh_centers, n


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
    Takes points and uses complete hierarchial clustering to return cluster centers in 2D
    \nReturns:
    cluster_groups: a list of group numbers for each viewpoint
    cluster_centers: a 2D XY array of cluster centers
    """
    # TODO: use PRM to compute path distances.
    distMatrix = pdist(viewpoints)
    Z = shc.complete(distMatrix)
    cluster_groups = shc.fcluster(Z, D_CLUSTER, criterion='distance')

    n_clusters = max(cluster_groups)
    cluster_centers = np.zeros((n_clusters, 2))
    for c in range(n_clusters):
        group = cluster_groups == c+1
        view_group = viewpoints[group, 0:2]
        cluster_centers[c] = np.median(view_group, axis=0)

    return cluster_groups, cluster_centers


def create_viewpoints(mesh_model, incidence_angle=INCIDENCE_ANGLE, dmin=.1, dmax=2):
    """
    Given a mesh model, create a viewpoint 1m away from the first vertex.
    Constraints:
    Below height threshold.
    Above the ground.
    Not inside an object.
    Within incidence angle of a facet
    """
    unit_norm = mesh_model.normals / np.linalg.norm(mesh_model.normals, axis=1)[:, None]

    # For all points in the mesh calculate a rectangular region to sample points from
    viewpoints = mesh_model.v0 + unit_norm
    normal = viewpoints - mesh_model.v0
    return viewpoints, normal


def plot_3d_object_viewpoints(mesh_model, viewpoints):
    # Create a new plot
    figure = pyplot.figure()
    axes = mplot3d.Axes3D(figure)

    # Add polygon with view color to matplotlib figure
    polygon = mplot3d.art3d.Poly3DCollection(mesh_model.vectors)
    axes.add_collection3d(polygon)
    axes.scatter(xs=viewpoints[:, 0], ys=viewpoints[:, 1], zs=viewpoints[:, 2], marker='o', c='r')

    # Auto scale to the mesh size
    scale = mesh_model.points.flatten()
    axes.auto_scale_xyz(scale, scale, scale)
    axes.set_xlabel('X')
    axes.set_ylabel('Y')
    axes.set_zlabel('Z')

    # Show the plot to the screen
    pyplot.show()


def main():

    # Load the model
    mesh_model, facets, incidence_normals, mesh_centers, n = load_mesh(model)
    plot_3d_object_viewpoints(mesh_model, np.zeros((1, 3)))
    # Initialize viewpoints along facet normal
    viewpoints = mesh_model.v0 + mesh_model.normals

    # Rasterization algorithm to determine if facet is not obscured by another object
    # TODO: How to determine whether a facet is in front of another facet given a camera point


    
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
    # figure = pyplot.figure()
    # axes = mplot3d.Axes3D(figure)
    cluster_groups, cluster_centers = viewpoint_clusters(viewpoints)
    colors = cm.get_cmap()
    new_cmap = rand_cmap(np.max(cluster_groups[0]), type='bright', first_color_black=True, last_color_black=False, verbose=False)
    # axes.scatter(xs=viewpoints[:, 0], ys=viewpoints[:, 1], zs=viewpoints[:, 2], marker='o', c=cluster_groups, cmap='RdYlBu')
    # axes.scatter(xs=viewpoints[:, 0], ys=viewpoints[:, 1], zs=viewpoints[:, 2], marker='o', c=cluster_groups[0], cmap=new_cmap)
    # plot_tsp_path(axes, viewpoints)


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