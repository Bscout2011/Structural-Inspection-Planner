"""
Author: Andrew Washburn
UNR CS791 Robotics Fall 2020
Instructor: Christos Papachristos

This program imports a STL polyhedron mesh file and computes an optimal set of viewpoints that observe the entire object.
Idea credited to "Planning Robot Motions for Range-Image Acquisition and Automatic 3D Model Construction" by Banos and Latombe 1998
"""

import numpy as np
from stl import mesh as stl_mesh
from numpy.linalg import norm
from scipy.spatial.transform import Rotation as R
from scipy.spatial import cKDTree

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from mpl_toolkits import mplot3d

from os.path import exists, join
import os

from probabilistic_road_map import sample_points, generate_road_map


# Set input viewpoint constraints
INCIDENCE_ANGLE = np.pi / 6  # 30deg
DMIN = 0.1  # [m]
DMAX = 2  # [m]
CEILING = 2  # [m]
FLOOR = 0.1  # [m]
ROBOT_RADIUS = 0.5  # [m]
ARM_LENGTH = 0.7  # [m]
TANK = "Mesh/ElevatedTank.stl"  # object to inspect
CWD = os.path.dirname(os.path.abspath(__file__))


def load_mesh(model):
    """
    Load the STL files and add the vectors to the plot
    Arguments:
        model: path name of stl file
    Returns: 
        mesh: Mesh object from numpy-stl   
        facets: (n, 3, 3) coordinate array of triangular facets
        unit_normals: unit normals for each facet
        mesh_centers: numpy array of facet center coordinates
        n: number of facets
    """
    if not exists(join(CWD, model)):
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


def sample_cone_region(mu=500, incidence_angle=INCIDENCE_ANGLE, dmin=DMIN, dmax=DMAX):
    """
    Generate mu points within a cone.

    Returns:
        cone_points: ndarray [mu, 3]
    """
    cone_points = np.zeros((mu, 3))
    rng = np.random.default_rng()
    width = np.tan(incidence_angle) * dmax
    center = np.array((0, 0, 1))
    i = 0

    while i < mu:
        z = rng.uniform(dmin, dmax)
        x, y = rng.uniform(-width, width, 2)
        point = np.array((x, y, z))
        # Check if point is within incidence cone
        length = norm(point)
        theta = np.arccos(np.dot(center, point) / norm(point))
        if (theta <= incidence_angle) & (length >= dmin) & (length <= dmax):
            # add to cone_points
            cone_points[i] = point
            i = i + 1
    return cone_points


def transform_cone(cone_points, point, normal):
    """
    Cross product of z = [0,0,1] with unit normal for this point will give axis of rotation
    """
    z = np.array((0, 0, 1))
    direction = np.dot(z, normal)
    theta = np.arccos(direction)
    rot_vec = np.cross(z, normal)

    if norm(rot_vec) == 0:  # case for 0 cross product, set rotation axis as x-axis
        rot_vec = np.array((1, 0, 0))
    rot_vec = rot_vec / norm(rot_vec) * theta
    rotation = R.from_rotvec(rot_vec)
    rotated_cone_points = rotation.apply(cone_points)  # rotate cone
    rotated_cone_points = rotated_cone_points + point  # translate

    return rotated_cone_points


def compute_visible_points(region, point, points, normals, viewed_points, arm_length, free_space_kdtree,
                        fov_angle=np.pi/4, incidence_angle=INCIDENCE_ANGLE, dmin=DMIN, dmax=DMAX, floor=FLOOR, ceiling=CEILING):
    """
    From each viewpoint, calculate the number of points this viewpoint can see. Return viewpoint that sees the most unique coverage.

    Arguments:
        region: viewpoints in the observable space
        point: the point where sampled a viewable region
        points: all the points in the object
        normals: corresponding normals to each point
        viewed_points: list of viewed points (0 corresponds to unseen)
        arm_length: max distance from point in free space to viewpoint
        free_space_kdtree: free configuration space of mobile robot
        fov_angle: [radians] camera field of view

    Output:
        best_view: a viewpoint position and orientation
        viewed_points: an updated list of seen and unseen points
        viewpoint_visible_point_indices: a list of indices of viewed mesh points from the best_view
    """

    viewpoint_visible_point_indices = []  # indicies of viewed points
    viewpoint_visible_point_count = 0  # max number of visible points
    best_view = np.zeros((1, 6))
    fov_cos_angle = np.cos(fov_angle)
    incidence_cos_angle = np.cos(incidence_angle)
    
    for viewpoint in region:
        if (viewpoint[2] < floor) or (viewpoint[2] > ceiling):
            continue  # viewpoint is out of reach
        viewpoint_dir = point - viewpoint
        viewpoint_dir = viewpoint_dir / norm(viewpoint_dir)

        view_vectors = points - viewpoint
        view_vectors_dir = view_vectors / norm(view_vectors, axis=1)[:, np.newaxis]
        # Filter points within min and max distance
        view_vector_distance = norm(view_vectors, axis=1)
        view_vector_distance = (view_vector_distance >= dmin) & (view_vector_distance <= dmax)
        # Filter points within viewpoint field of View
        fov_cos_theta = np.dot(view_vectors_dir, viewpoint_dir)
        fov_visible = np.arccos(fov_cos_theta) <= fov_angle

        # Filter points pointed towards viewpoint
        incidence_cos_theta = np.dot(normals, viewpoint_dir)
        incidence_visible = np.arccos(incidence_cos_theta) <= incidence_angle
        # Filter points that haven't been seen yet
        unseen_boundary = viewed_points == 0
        
        # TODO: ray-tracing to determine if there's a facet in front of this line of sight

        visible_points = np.all(np.stack((view_vector_distance, fov_visible, incidence_visible, unseen_boundary), axis=1), axis=1)
        visible_point_indices = np.argwhere(visible_points).squeeze()

        if visible_point_indices.size > viewpoint_visible_point_count:
            viewpoint_visible_point_indices = visible_point_indices
            viewpoint_visible_point_count = visible_point_indices.shape[0]
            best_view = np.concatenate((viewpoint, viewpoint_dir))
    
    if viewpoint_visible_point_count == 0:
        raise Exception("Nothing new can be seen from this point's observable space")
    
    # Update unseen points
    for idx in viewpoint_visible_point_indices:
        viewed_points[idx] = 1

    return best_view, viewed_points, viewpoint_visible_point_indices


def create_obstacles(facets):
    """
    Input: triangular facets from a STL mesh
    Returns KD tree of XY obstacle points
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
    obstacles = cKDTree(obstacles)
    return obstacles


def generate_prm(polygon, robot_radius, n_sample=500):
    """
    Given a stl mesh, generate a road map around the object representing the mobile robot's free space. 
    Arguments:
        polygon: a stl mesh object
        robot_radius: radius for collision checking free space
        n_sample: number of road map vertices
    Output: 
        sample_kdtree: a cKDTree of vertices representing the mobile robot's free space 
        road_map: an adjacency list graph representing the free configuration space edges
    """
    facets = polygon.points.reshape(-1, 3, 3)
    obstacle_kd_tree = create_obstacles(facets)
    sample_x, sample_y = sample_points(robot_radius, obstacle_kd_tree)
    road_map, sample_kdtree = generate_road_map(sample_x, sample_y, robot_radius, obstacle_kd_tree)

    return sample_kdtree, road_map


def dual_viewpoint_sampling(polygon, robot_radius, arm_length, mu=500, constraints=None, plot=False):
    """
    Arguments:
        polygon: path filename to a STL object
        robot_radius: robot size for obstacle collision checking
        mu: number of samples per viewpoint iteration
        constraints: (optional) geometric viewing restrictions 
        plot: show the model and computed viewpoints

    Output: 
        viewpoint_set: a near optimal set S of viewpoints covering the object's boundary

    1. Select a point p from the unseen portion of the object and compute the region O(p) from which such a point is visible.
    2. Sample O(p) mu times, and select the position with the highest coverage. Add this position to the set S.
    3. Update the data structure representing the unseen portion of the object.
    4. Repeat the algorithm until the unssen boundary is neglectable.
    """

    # load the polygon object
    rng = np.random.default_rng()
    mesh_model, facets, unit_norms, mesh_centers, n = load_mesh(polygon)
    unit_norms = np.array([val for val in unit_norms for _ in range(3)])
    points = mesh_model.points.reshape(-1, 3)
    num_points = points.shape[0]
    # Initialize set list of viewpoints
    viewpoint_set = []
    viewed_points_set = []
    # Initialize data structure of unseen points
    unseen = np.zeros(num_points, dtype=int)
    # Initialize samples for observable region
    cone_points = sample_cone_region(mu)

    # Generate a Probabilistic Road Map of the free space around the object
    free_space_kdtree, road_map = generate_prm(mesh_model, robot_radius)

    # Loop until all points are seen, or could not be seen
    while np.any(unseen == 0):
        # 1. 
        # Choose a point from the unseen portion of the object.
        unseen_point_idices = np.argwhere(unseen == 0)
        point_idx = rng.choice(unseen_point_idices)
        point = points[point_idx].squeeze()
        normal = unit_norms[point_idx].squeeze()
        # Copy cone points and transform for this point
        region = transform_cone(cone_points, point, normal)
        # 2.
        # Select viewpoint with highest coverage and return viewpoint position and unseen points
        try:
            viewpoint, unseen, viewed_points = compute_visible_points(region, point, points, unit_norms, unseen, arm_length, free_space_kdtree)
            viewpoint_set.append(viewpoint)
            viewed_points_set.append(viewed_points)

        except Exception:
            unseen[point_idx] = 2
        # 3. 
        # Unseen points list is updated. Set of viewpoints is added
    
    viewpoint_set = np.array(viewpoint_set)
    
    if plot:
        plot_3d_object_viewpoints(mesh_model, viewpoint_set, viewed_points_set)
    return viewpoint_set


def plot_3d_object_viewpoints(mesh_model, viewpoints, viewed_points_set):
    # Create a new plot
    figure = plt.figure()
    axes = mplot3d.Axes3D(figure)
    points = mesh_model.points.reshape(-1, 3)
    color=iter(plt.cm.rainbow(np.linspace(0,1,viewpoints.shape[0])))

    # Add polygon with view color to matplotlib figure
    polygon = mplot3d.art3d.Poly3DCollection(mesh_model.vectors, linewidth=.1, edgecolor=(0, 0, 0), facecolor=(0, 0, 1, .2))
    axes.add_collection3d(polygon)

    for i in range(viewpoints.shape[0]):
        vp = viewpoints[i][0:3]
        viewed_points = [points[v] for v in viewed_points_set[i]]
        viewed_points = np.array(viewed_points)
        c = next(color)
        axes.scatter(vp[0], vp[1], vp[2], c=c)
        axes.scatter(xs=viewed_points[:, 0], ys=viewed_points[:, 1], zs=viewed_points[:, 2], c=c)
        # plot lines from viewpoint to points
        for j in range(viewed_points.shape[0]):
            axes.plot(
                [vp[0], viewed_points[j, 0]],
                [vp[1], viewed_points[j, 1]],
                [vp[2], viewed_points[j, 2]],
                c=c
            )


    # Auto scale to the mesh size
    scale = mesh_model.points.flatten()
    axes.auto_scale_xyz(scale, scale, scale)
    axes.set_xlabel('X')
    axes.set_ylabel('Y')
    axes.set_zlabel('Z')

    # Show the plot to the screen
    plt.show()


def plot_sideview(mesh_model):
    cone_points = sample_cone_region()
    # get points on y-z max plane
    mesh_points = mesh_model.points.reshape(-1, 3, 3)
    rng = np.random.default_rng()
    rand_idx = rng.integers(mesh_points.shape[0])
    facet = mesh_points[rand_idx]
    normal = mesh_model.normals[rand_idx]
    unit_normal = normal / norm(normal)
    region = transform_cone(cone_points, facet[0], normal)

    diag = np.sqrt(2)/2
    points = [
        [-.5, 1],
        [.5, 1],
        [.5+diag, 1+diag],
        [.5+diag, 2+diag],
        [.5, 2+2*diag],
        [-.5, 2+2*diag],
        [-.5-diag, 2+diag],
        [-.5-diag, 1+diag]
    ]
    points = np.array(points)
    
    fig, ax = plt.subplots()
    ax.plot(points[:, 0], points[:, 1], 'bo')
    plt.show()



def main():
    viewpoints = dual_viewpoint_sampling(TANK, ROBOT_RADIUS, ARM_LENGTH, plot=True)
    # stuff = load_mesh(TANK)
    # mesh_model = stuff[0]
    # plot_sideview(mesh_model)


if __name__ == "__main__":
    main()