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
HEIGHT = .5  # robot arm base height
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


def sample_cone_region(mu=500, incidence_angle=INCIDENCE_ANGLE, dmin=DMIN, dmax=DMAX, plot=False):
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

    if plot:
        # Create plot window with robot config sample space
        fig, ax = plt.subplots()
        ax.plot(cone_points[:,0], cone_points[:,2], 'bo')
        ax.axis('equal')
        ax.set_ylim([-1, dmax*1.05])
        ax.axis('off')
        
        ax.plot(  # Plot boundary Points
            [-1, 0, 1], 
            [-.5, 0, -.5], 
            'ko-')
        ax.plot(  # p_i-1 normal
            [-1, -1.25],
            [-.5, 0],
            'k--'
        )
        ax.plot(  # p_i+1 normal
            [1, 1.25],
            [-.5, 0],
            'k--'
        )
        ax.annotate(r"$p_i$", [-.1, -.3], size=20)
        ax.annotate(r"$p_{i+1}$", [1, -.75], size=20)
        ax.annotate(r"$\hat u_{i+1}$", [1.25, 0.1], size=20)
        ax.annotate(r"$p_{i-1}$", [-1, -.75], size=20)
        ax.annotate(r"$\hat u_{i-1}$", [-1.35, 0.1], size=20)

        # Plot centerline
        ax.plot([0, 0], [0, .6], 'k--')
        ax.plot(
            [0, .55 * np.sin(incidence_angle)],
            [0, .55 * np.cos(incidence_angle)],
            'k-'
        )
        # Plot the angle constraint edges
        corners = np.array(
            [[dmin * np.sin(incidence_angle), dmin*np.cos(incidence_angle)],
            [dmax * np.sin(incidence_angle), dmax*np.cos(incidence_angle)]]
        )
        ax.plot(corners[:,0], corners[:,1], 'k--')
        ax.plot(-corners[:,0], corners[:,1], 'k--')
        # Plot the distance min and max edges
        angles = np.linspace(-incidence_angle, incidence_angle, 50)
        arc_min_x = dmin * np.sin(angles)
        arc_min_y = dmin * np.cos(angles)
        arc_max_x = dmax * np.sin(angles)
        arc_max_y = dmax * np.cos(angles)
        ax.plot(arc_min_x, arc_min_y, 'k--')
        ax.plot(arc_max_x, arc_max_y, 'k--')
        # Annotate plot
        ax.annotate(r"$\tau$", [.3, .3], size=25)
        draw_angle(incidence_angle, np.pi / 2 - incidence_angle)
        ax.annotate("dmin", [-1*corners[0,0]-.8, corners[0, 1]], size=25)
        ax.annotate("dmax", [-1*corners[1,0]-.4, corners[1, 1]+.2], size=25)
        ax.annotate(r"$S_i$", [1, 1], size=25)
        ax.arrow(1, 1, -.75, .1, width=.01, joinstyle='round', zorder=3)

        plt.show()

    return cone_points


def transform_points(points, theta, origin):
    T = np.array([[np.cos(theta), -np.sin(theta), origin[0]],
                  [np.sin(theta), np.cos(theta), origin[1]],
                  [0, 0, 1]])
    return np.matmul(T, np.array(points))


def draw_angle(angle, offset=0, origin=[0, 0], r=0.5, n_points=100):
        """
        Effect:
            draws on pyplot an arc
        """
        x_start = r*np.cos(angle)
        x_end = r
        dx = (x_end - x_start)/(n_points-1)
        coords = [[0 for _ in range(n_points)] for _ in range(3)]
        x = x_start
        for i in range(n_points-1):
            y = np.sqrt(r**2 - x**2)
            coords[0][i] = x
            coords[1][i] = y
            coords[2][i] = 1
            x += dx
        coords[0][-1] = r
        coords[2][-1] = 1
        coords = transform_points(coords, offset, origin)
        plt.plot(coords[0], coords[1], 'k--')


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


def ray_tracing(goal_idx, p_0, d, facets, normals):
    """
    Returns true if the goal facet as seen from the viewpoint is collision free.

    Arguments:
        goal_idx: index of facet to view
        p_0: position to view from
        d: viewpoint unit direction
        facets: (n, 3, 3) ndarray of triangular facets
        normals: (n, 3) unit normals corresponding to each facet

    Returns:
        True if goal facet can be seen. 
        False if another facet is in the way.
    """
    n = facets.shape[0]     # number of facets
    p = np.zeros((n, 3))    # intersection points
    t = np.zeros(n)         # intersection distance

    # Compute distance to intersect each facet plane
    for i in range(n):
        n_d = np.dot(facets[i, 0] - p_0, normals[i])
        t[i] = n_d / np.dot(d, normals[i])
        if t[i] > 0:  # only compute intersection if plane is in front of viewpoint
            p[i] = p_0 + t[i] * d
        # else:
            # print("Plane is not visible from this direction")

    # Given the intersection distance, is the intersected point within the triangluar facet
    t_goal = t[goal_idx]
    for i in range(n):
        if (t[i] <= 0) or (t[i] > t_goal):
            continue  # skip points behind viewpoint or further than goal
        
        edges = np.array([facets[i,v1] - facets[i,v0] for v0, v1 in ((0,1), (1, 2), (2, 0))])  
        C = p[i] - facets[i]
        result = np.array([np.dot(normals[i], np.cross(edges[j], C[j])) for j in range(3)])
        res = np.all(result >= 0)
        if res and t[i] < t_goal:
            # print(f"Goal plane is blocked by plane {i}")
            return False
    return True


def compute_visible_points(region, point, points, normals, viewed_points, arm_length, free_space_kdtree,
                        fov_angle=np.pi/4, incidence_angle=INCIDENCE_ANGLE, dmin=DMIN, dmax=DMAX, floor=FLOOR, ceiling=CEILING):
    """
    From each viewpoint, calculate the number of points this viewpoint can see. Return viewpoint that sees the most unique coverage.

    Arguments:
        region: viewpoints in the observable space
        point: the point where sampled a viewable region
        points: all the points in the object
        normals: corresponding unit normals to each point
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
    facets = points.reshape(-1, 3, 3)
    facet_normals = normals.reshape(-1, 3, 3)[:,0,:]
    fov_cos_angle = np.cos(fov_angle)
    incidence_cos_angle = np.cos(incidence_angle)
    
    for viewpoint in region:
        if (viewpoint[2] < floor) or (viewpoint[2] > ceiling):
            continue  # viewpoint is out of reach
        
        # Check if within radius of point in the free workspace.
        nearest_neighbors = free_space_kdtree.query_ball_point(viewpoint, arm_length)
        if len(nearest_neighbors) == 0:
            continue  # viewpoint is out of reach
        else:
            # viewpoint within arm length. Check IK solution is feasible.
            # Get IK solution for closest free
            pass

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
        
        visible_points = np.all(np.stack((view_vector_distance, fov_visible, incidence_visible, unseen_boundary), axis=1), axis=1)
        visible_point_indices = np.argwhere(visible_points).squeeze()

        # Ray-tracing to determine if there's a facet in front of this line of sight
        ray_trace_visible = []
        for point_idx in visible_point_indices:
            facet_idx = point_idx // 3  # divide by three to get facet indexes
            if ray_tracing(facet_idx, viewpoint, viewpoint_dir, facets, facet_normals):
                ray_trace_visible.append(point_idx)
        visible_point_indices = np.array(ray_trace_visible)

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


def generate_prm(polygon, robot_radius, height, n_sample=500):
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
    road_map, sample_kdtree = generate_road_map(sample_x, sample_y, robot_radius, obstacle_kd_tree, height)

    return sample_kdtree, road_map


def dual_viewpoint_sampling(polygon, robot_radius, arm_length, height, mu=500, constraints=None, plot=False):
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
    free_space_kdtree, road_map = generate_prm(mesh_model, robot_radius, height)

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
            print(f"{np.sum(unseen == 0)} / {unseen.shape[0]} mesh points unseen")

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
    axes.axis('off')

    # Show the plot to the screen
    plt.show()


def plot_model_normals(model):
    mesh_model, facets, unit_norms, mesh_centers, n = load_mesh(model)
    # Create a new plot
    figure = plt.figure()
    axes = mplot3d.Axes3D(figure)
    points = mesh_model.points.reshape(-1, 3)
    # unit_norms = np.array([val for val in unit_norms for _ in range(3)])
    
    polygon = mplot3d.art3d.Poly3DCollection(mesh_model.vectors, linewidth=.1, edgecolor=(0, 0, 0), facecolor=(0, 0, 1, .2))
    axes.add_collection3d(polygon)

    for i in range(facets.shape[0]):
        axes.plot(
            [mesh_centers[i,0], mesh_centers[i,0] + unit_norms[i,0]],
            [mesh_centers[i,1], mesh_centers[i,1] + unit_norms[i,1]],
            [mesh_centers[i,2], mesh_centers[i,2] + unit_norms[i,2]],
        )
    plt.show()


def main():
    viewpoints = dual_viewpoint_sampling(TANK, ROBOT_RADIUS, ARM_LENGTH, HEIGHT, plot=True)
    np.save("viewpoints", viewpoints)
    # stuff = load_mesh(TANK)
    # mesh_model = stuff[0]
    # plot_sideview(mesh_model)
    # plot_model_normals(TANK)
    # sample_cone_region(dmin=.7, plot=True)


if __name__ == "__main__":
    main()