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
from scipy.spatial import distance
import scipy.cluster.hierarchy as hcluster
from scipy.cluster.hierarchy import linkage, fcluster

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from mpl_toolkits import mplot3d

from os.path import exists, join
import os
import sys
from geometry_msgs.msg import Pose, PoseArray
from probabilistic_road_map import sample_points, generate_road_map, dijkstra_planning

from scipy.spatial.transform import Rotation as R
import rospkg
r = rospkg.RosPack()
path  = r.get_path('robowork_planning')
path += "/scripts/"
print(path)
sys.path.append(path)
from get_ik import GetIK
import rospy

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


def generate_prm(polygon, robot_radius, height=0, n_sample=500):
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


def cluster_base_position(ell, viewpoints, sample_kd_tree, road_map):
    """
    Find a minimal set of base positions for the viewpoint set.
    Find pairwise distances between all viewpoints. If a viewpoint pair shares
    a nearest neighbor within ell distance in the sample space, 
    then distance between these viewpoint is the distance between that nearest neighbor and
    the viewpoint. Else viewpoints don't share a nearest neighbor, distance between the two 
    is length to closest point in sample space and the dijkstra path length between the free space points.
    Pairwise distances feed into a heirarchial clustering algorithm using a complete linkage.
    The dendrogram is pruned at 2*ell. For each viewpoint cluster, find a common nearest neighbor
    that is optimal. We choose the neighbor that minimizes the distance to all viewpoints.
    Store the viewpoints in a list structure `G` that corresponds to each base position. 

    Params:
        ell: [m] robot arm length
        viewpoints: (n_vp, 6) viewpoint poses
        sample_kd_tree: cKDTree of points representing free space
        road_map: adjacency list of sample space

    Returns:
        B: (n_b, 6) an array of base poses
        G: list of lists of viewpoint poses. First index corresponds to base pose index. Second contains corresponding viewpoints
    """
    # Find pair wise distances between each viewpoint
    n_vp = viewpoints.shape[0]
    p_dist = np.zeros((n_vp, n_vp))
    for i in range(n_vp):
        for j in range(i+1, n_vp):
            # First: find common nearest neighbors
            vp0 = viewpoints[i, 0:3]
            vp1 = viewpoints[j, 0:3]
            nn0 = sample_kd_tree.query_ball_point(vp0, ell)
            nn1 = sample_kd_tree.query_ball_point(vp1, ell)
            common_nn = set(nn0).intersection(set(nn1))
            if len(common_nn) > 0:  # viewpoint pair has a common base position
                commom_nn_pos = np.array([sample_kd_tree.data[nn] for nn in common_nn])
                distances = distance.cdist((vp0, vp1), commom_nn_pos)
                p_dist[i, j] = distances.sum(axis=0).min()  # store minimum common distance
            else:  # no common base position. Compute path length between closest base positions
                b0 = nn0[0]
                b1 = nn1[0]
                path_list, path_found = dijkstra_planning(b0, b1, road_map, sample_kd_tree.data[:,0], sample_kd_tree.data[:,1])
                if not path_found:
                    raise Exception("No path found between viewpoints")
                path_length = distance.euclidean(vp0, sample_kd_tree.data[b0])
                for s in range(len(path_list) - 1):
                    path_length = path_length + distance.euclidean(
                        sample_kd_tree.data[path_list[s]], sample_kd_tree.data[path_list[s+1]])
                path_length = path_length + distance.euclidean(vp1, sample_kd_tree.data[b1])
                p_dist[i, j] = path_length

    # Hierarchial clustering
    v_dist = distance.squareform(p_dist + p_dist.T)
    Z = linkage(v_dist, 'complete')
    cluster_assignment = fcluster(Z, 2*ell, 'distance')
    # Find a common base position for each cluster group
    n_c = max(cluster_assignment)
    B = np.zeros((n_c, 6))
    G = []
    for i in range(n_c):
        vpos = np.array([viewpoints[j, 0:3] for j, c in enumerate(cluster_assignment) if c == i+1])
        G.append(vpos)
        bp = [set(sample_kd_tree.query_ball_point(p, ell)) for p in vpos]
        common_bp = set.intersection(*bp)
        base_pos_candidates  = np.array([sample_kd_tree.data[p] for p in common_bp])
        # vpos_bp_dist: each row corresponds to a viewpoint and column to a base position; 
        # entry is euclidean distance between the two
        vpos_bp_dist = distance.cdist(vpos, base_pos_candidates)
        # Choose base position with min total distance. What would be other metrics?
        opt_bp_idx = vpos_bp_dist.sum(axis=0).argmin()
        opt_bp_pos = base_pos_candidates[opt_bp_idx]
        opt_bp_pos[2] = 0  # set position height to ground level
        # TODO: Determine base orientation
        B[i, 0:2] = opt_bp_pos[0:2]
    
    return B, G


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
        free_space_kdtree: cKDTree of samples representing the free space
        road_map: adjacency list of the free space

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
            print("{np.sum(unseen == 0)} / {unseen.shape[0]} mesh points unseen")

        except Exception:
            unseen[point_idx] = 2
        # 3. 
        # Unseen points list is updated. Set of viewpoints is added
    
    viewpoint_set = np.array(viewpoint_set)
    
    if plot:
        plot_3d_object_viewpoints(mesh_model, viewpoint_set, viewed_points_set)
    return viewpoint_set, free_space_kdtree, road_map


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

def getEuler(v):
    # x = np.array([1,0,0], dtype=np.float64)
    # y = np.array([0,1,0], dtype=np.float64)
    # z = np.array([0,0,1], dtype=np.float64)
    roll =  np.arccos(v[0]/np.linalg.norm(v))
    pitch =  np.arccos(v[1]/np.linalg.norm(v))
    yaw =  np.arccos(v[2]/np.linalg.norm(v))
    return roll, pitch, yaw

def convertToPose(viewpoint):
    g = viewpoint[:3]
    v = viewpoint[3:]

    r, p, y = getEuler(v)
    rr = R.from_euler('xyz',[r,p,y], degrees=False)
    q = rr.as_quat()
    # q = tfs.quaternion_from_euler(r,p,y)
    
    p = Pose()
    p.position.x = g[0]
    p.position.y = g[1]
    p.position.z = g[2]

    p.orientation.x = q[0]
    p.orientation.y = q[1]
    p.orientation.z = q[2]
    p.orientation.w = q[3]
    
    return p


import open3d as o3d
# import numpy as np

def collision_check(m_input, location, normal):
    # Input mesh
    # m_input = o3d.io.read_triangle_mesh(mesh)
    # m_input = o3d.geometry.TriangleMesh.create_cone(0.1, 0.3, 20,1)
    pose = pose_from_vector3D(np.hstack((location,normal)))
    rot = o3d.geometry.TriangleMesh.get_rotation_matrix_from_quaternion(np.array([pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]))
    tx = np.zeros((4,4))
    tx[:3,:3] = rot
    tx[3,3] = 1
    test_size = 0.10
    m_test = o3d.geometry.TriangleMesh.create_box(test_size, test_size, test_size)
    # Location to be tested
    test_location = location
    # Translate the test mesh to the test location
    m_test = m_test.translate(test_location)
    m_test_arrow=  o3d.geometry.TriangleMesh.create_arrow(0.05,0.07,0.1,0.05,20,4,1)
    m_test_arrow = m_test_arrow.transform(tx)
    m_test_arrow = m_test_arrow.translate(test_location)
    # Unimportant stuff
        # m_test_arrow.paint_uniform_color([1, 0.706, 0])
    # m_test.paint_uniform_color([0, 0.706, 0])
    # Actual collision check
    if (m_test.is_intersecting(m_input)):
        # print("Intersecting")
        return True, m_test_arrow
    else:
        # m_test_arrow.paint_uniform_color([0, 0.706, 1])
        # print("Not Intersecting")
        return False, m_test_arrow
        # Visualize to confirm
    # o3d.visualization.draw_geometries([m_input, m_test])


# Returns true if in collision
def collision_check_with_robot_model(m_input, location, orientation):
    # SET HERE THE SIZE OF THE ROBOT. The m_test is a box representing the robot.
    m_test = o3d.geometry.TriangleMesh.create_box(0.8, 0.8, 0.5)
    rot = o3d.geometry.TriangleMesh.get_rotation_matrix_from_quaternion(np.array([orientation[1], orientation[2], orientation[3], orientation[0]]))
    tx = np.zeros((4,4))
    tx[:3,:3] = rot
    tx[3,3] = 1
    # Rotate the mesh
    m_test = m_test.transform(tx)
    # Translate the test mesh to the test location
    m_test = m_test.translate(location)
    # Actual collision check
    if (m_test.is_intersecting(m_input)):
        # print("Intersecting")
        return True
    else:
        return False

def  pose_from_vector3D(waypoint):
    #http://lolengine.net/blog/2013/09/18/beautiful-maths-quaternion-from-vectors
    offset  = np.array([1.5, 2.5, 0], dtype=np.float64)

    pose= Pose()
    pose.position.x = waypoint[0] + offset[0]
    pose.position.y = waypoint[1] + offset[1]
    pose.position.z = waypoint[2] + offset[2]
    #calculating the half-way vector.
    u = [1,0,0]
    norm = np.linalg.norm(waypoint[3:])
    v = np.asarray(waypoint[3:])/norm 
    if (np.array_equal(u, v)):
        pose.orientation.w = 1
        pose.orientation.x = 0
        pose.orientation.y = 0
        pose.orientation.z = 0
    elif (np.array_equal(u, np.negative(v))):
        pose.orientation.w = 0
        pose.orientation.x = 0
        pose.orientation.y = 0
        pose.orientation.z = 1
    else:
        half = [u[0]+v[0], u[1]+v[1], u[2]+v[2]]
        pose.orientation.w = np.dot(u, half)
        temp = np.cross(u, half)
        pose.orientation.x = temp[0]
        pose.orientation.y = temp[1]
        pose.orientation.z = temp[2]
    norm = np.sqrt(pose.orientation.x*pose.orientation.x + pose.orientation.y*pose.orientation.y + 
        pose.orientation.z*pose.orientation.z + pose.orientation.w*pose.orientation.w)
    if norm == 0:
        norm = 1
    pose.orientation.x /= norm
    pose.orientation.y /= norm
    pose.orientation.z /= norm
    pose.orientation.w /= norm
    return pose

def getPoseArray(viewpoints):

    pa = PoseArray()
    print("viewpoints len = ", len(viewpoints))
    for i in range(len(viewpoints)):
        p = pose_from_vector3D(viewpoints[i])
        pa.poses.append(p)

    pa.header.frame_id = 'map'
    pa.header.stamp = rospy.Time.now()
    return pa


def height_filter(g):

    max_arm_reach = 1.7
    min_arm_reach = 0.25
    if g[2]> max_arm_reach or g[2]<min_arm_reach:
        return False
    else:
        return True


def getValidVP(viewpoints):

    filter_vp = []
    mesh = o3d.io.read_triangle_mesh(TANK)
    arrow_arr = []
    for i in range(len(viewpoints)):
        g = viewpoints[i][:3]
        v = viewpoints[i][3:]
        # print("vp = ",viewpoints[i])
        # print("g = ",g)
        # print("v = ",v)
        reachable = height_filter(g)
        collision, arrow = collision_check(mesh,g,v)
        # arrow.paint_uniform_color([1, 0.706, 0])
        if(collision):
            arrow.paint_uniform_color([1, 0.706, 0])
        if(not reachable):
            arrow.paint_uniform_color([0.706, 1, 0])
        if(not collision and reachable):
            arrow.paint_uniform_color([0, 0.706, 1])
            filter_vp.append(viewpoints[i])
        arrow_arr.append(arrow)

    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=np.array([0., 0., 0.]))

    arrow_arr.append(mesh)
    arrow_arr.append(axis)
    # o3d.visualization.draw_geometries(arrow_arr)
    
    return np.array(filter_vp)


def getPlanePoints(viewpoints, plane, sign):
    cluster = []
    not_clusters = []
    for i in range(len(viewpoints)):
        g = viewpoints[i][:3]
        print("pt test = ", g[0]*plane[0] + g[1]*plane[1] + g[2]*plane[2] + plane[3])
        if(sign*(g[0]*plane[0] + g[1]*plane[1] + g[2]*plane[2] + plane[3]) > 0.0):
            cluster.append(viewpoints[i])
        else:
            not_clusters.append(viewpoints[i])
    return np.array(cluster), np.array(not_clusters)

def getClusters(viewpoints):
    p1 = np.array([0,1,0, -0.3],dtype=np.float64) #sign = +1
    p2 = np.array([0,1,0,-5.0+0.3],dtype=np.float64) #sign = -1
    p3 = np.array([1,0,0,-1.0-0.2],dtype=np.float64) #sign = +1
    p4 = np.array([1,0,0,-1.0-1.8+0.2],dtype=np.float64) #sign = -1

    mesh = o3d.io.read_triangle_mesh(TANK)
    mesh.translate(np.array([1.0,2.5,0.0],dtype=np.float64))
    all_clusters = []
    print("c1")
    cluster_1, rest = getPlanePoints(viewpoints, p1, -1.0) 
    print("c2")
    cluster_2, rest = getPlanePoints(rest, p2, 1.0)    
    print("c3")
    cluster_3, rest = getPlanePoints(rest, p3, -1.0)    
    print("c4")
    cluster_4, rest = getPlanePoints(rest, p4, 1.0)

    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=np.array([0., 0., 0.]))

    all_clusters.append(cluster_1)
    all_clusters.append(cluster_2)
    all_clusters.append(cluster_3)
    all_clusters.append(cluster_4)

    test_size = 0.1
    plot_pt = []
    for i, cluster in enumerate(all_clusters):
        print("cluster {} and size of cluster = {}".format(i,cluster.shape))
        for j in range(len(cluster)):
            g = cluster[j][:3]
            m_test = o3d.geometry.TriangleMesh.create_box(test_size, test_size, test_size)
            m_test = m_test.translate(g)
            if(i == 0):
                m_test.paint_uniform_color([1, 0.706, 0])
                plot_pt.append(m_test)
            if(i == 1):
                m_test.paint_uniform_color([1, 0.706, 1])
                plot_pt.append(m_test)
            if(i == 2):
                m_test.paint_uniform_color([0, 0.706, 1])
                plot_pt.append(m_test)
            if(i == 3):
                m_test.paint_uniform_color([0, 0.706, 0])
                plot_pt.append(m_test)

    
    plot_pt.append(mesh)
    plot_pt.append(axis)
    # o3d.visualization.draw_geometries(plot_pt)

    return all_clusters

# def getOffsetVP(viewpoints):
#     offset  = np.array([1.0, 2.5, 0.0], dtype=np.float64)
    
#     offsetVP = []
#     for i in range(len(viewpoints)):
#         vp = viewpoints[i]
#         vp[0] += offset[0]
#         vp[1] += offset[1]
#         vp[2] += offset[2]
#         offsetVP.append(vp)

#     return np.array(offsetVP)

def posearray_to_nparrays(posearr):
  positions = []
  orientations = []
  for pose in posearr.poses:
    positions.append([pose.position.x, pose.position.y, pose.position.z])
    orientations.append([pose.orientation.w, pose.orientation.x, pose.orientation.y, pose.orientation.z])
  return np.array(positions), np.array(orientations)


def seperate_points_to_clusters(points, clusters, orientations):
  #This array contains arrays of points that are in a cluster. 
  # clustered_points = [cluster1, cluster2, ..]
  # cluster1 = [point1, point2, ...]
  clustered_points = []
  clustered_orientations = []
  # The clustered orientations is simply so that we can maintain the orientiations in a similar data structure as the positions. 
  num_clusters = np.amax(clusters)
  for i in range(num_clusters):
    clustered_points.append([])
    clustered_orientations.append([])
  
  for i in range(len(points)):
    clustered_points[clusters[i]-1].append(points[i])
    clustered_orientations[clusters[i]-1].append(orientations[i])
    # The -1 is required as the clusters numbering starts from 1
  
  return clustered_points, clustered_orientations


#Note: Clusterization does not consider the viewpoint angles
def clusterize_viewpoints(posearr):
  locations, orientations = posearray_to_nparrays(posearr)
  distance_threshold = 1.5
  clusters = hcluster.fclusterdata(locations, distance_threshold, criterion="distance")
  return seperate_points_to_clusters(locations, clusters, orientations)


def check_ik(base_location, base_orientation, viewpoint_location, viewpoint_orientation):
  # print("base_orientation type = {}, shape = {}", type(base_orientation),base_orientation.shape)
  PLANNING_GROUP_ = "main_arm_SIM"
  ik_ob = GetIK(group=PLANNING_GROUP_)

  arm_pose = PoseStamped()
  arm_pose.header.frame_id = 'map'
  arm_pose.pose.position.x = viewpoint_location[0]
  arm_pose.pose.position.y = viewpoint_location[1]
  arm_pose.pose.position.z = viewpoint_location[2]
  arm_pose.pose.orientation.x = viewpoint_orientation[0]
  arm_pose.pose.orientation.y = viewpoint_orientation[1]
  arm_pose.pose.orientation.z = viewpoint_orientation[2]
  arm_pose.pose.orientation.w = viewpoint_orientation[3]
  
  base_pose = PoseStamped()
  base_pose.header.frame_id = 'map'
  base_pose.pose.position.x = base_location[0]
  base_pose.pose.position.y = base_location[1]
  base_pose.pose.position.z = base_location[2]
  base_pose.pose.orientation.x = base_orientation[0]
  base_pose.pose.orientation.y = base_orientation[1]
  base_pose.pose.orientation.z = base_orientation[2]
  base_pose.pose.orientation.w = base_orientation[3]
  response = ik_ob.get_ik(arm_pose, base_pose)
  print("ik response code", response.error_code)
  if(response.error_code.val == MoveItErrorCodes.SUCCESS):
      # print("ik found\n")
    return True
  else:
    # print("No solution found ")
    return False


# cluster is a np array of points
def get_base_location_for_cluster(position_cluster, orientation_cluster):
  cluster_center = np.mean(position_cluster, axis=0)
  # project center to ground
  cluster_center[2] = 0
  projected_circle_radius = 1
  base_offset_from_ground = 0.2
  mesh = o3d.io.read_triangle_mesh(TANK)

  for i in range(100):
    # Sample point for base location in a circle around the projected cluster_center
    length = np.sqrt(np.random.uniform(0, projected_circle_radius))
    angle = np.pi * np.random.uniform(0, 2)
    x = cluster_center[0] + length * np.cos(angle)
    y = cluster_center[1] + length * np.sin(angle)
    base_location = np.array([x, y, base_offset_from_ground])
    base_orientation = np.mean(np.array(orientation_cluster), axis=0)
    # wxyz

    #Check if base location is possible - i.e. check if it is collision free and it is ik feasible for all points in the position_cluster
    if not collision_check_with_robot_model(mesh, base_location, base_orientation):
      # The location is not in collision with the object mesh and now the ik should be checked for all points in the position_cluster
      all_ik_feasible = False
      for j in range(len(position_cluster)):
        if not check_ik(base_location, base_orientation, position_cluster[j], orientation_cluster[j]):
          break
        if j == (len(position_cluster) - 1):
          all_ik_feasible = True
      if all_ik_feasible:
        return base_location, base_orientation
    else: 
      print("in collision")
  print('Fatal: Unable to find a possible base location after 100 tries. Maybe try increasing the projected_circle_radius? Voxblox could also be reporting the collision with the ground. This can be fixed by increasing the base_offset')
  

def np_to_pose(position, orientation):
  p = Pose()
  print(position)
  p.position.x = position[0]
  p.position.y = position[1]
  p.position.z = position[2]
  p.orientation.w = orientation[0]
  p.orientation.x = orientation[1]
  p.orientation.y = orientation[2]
  p.orientation.z = orientation[3]
  return p


def main():
    rospy.init_node('viewpointg_gen', anonymous=True)
    viewpoint_pub = rospy.Publisher('viewpoints',PoseArray,queue_size=5, latch=True)
    base_poses_pub = rospy.Publisher('base_poses',PoseArray,queue_size=5, latch=True)

    viewpoints, sample_kdtree, road_map = dual_viewpoint_sampling(TANK, ROBOT_RADIUS, ARM_LENGTH, HEIGHT, plot=False)
    filtered_vp = getValidVP(viewpoints)
    # print("viewpoint shape = ", filtered_vp.shape)

    rate = rospy.Rate(5)
    posearr = getPoseArray(filtered_vp)
    print("old vp = {} filteredvp = {}".format(viewpoints.shape,filtered_vp.shape))

    # Now we have the final viewpoints for the camera. They need to be clustered, the base location needs to be found for each cluster and then two pose arrays need to be published in a new message
    clustered_positions, clustered_orientations = clusterize_viewpoints(posearr)
    print("sizes: ")
    print(len(clustered_positions))
    print(len(clustered_orientations))
    clustered_base_positions = []
    clustered_base_orientations = []
    for i in range(len(clustered_positions)):
      clustered_base_position, clustered_base_orientation = get_base_location_for_cluster(clustered_positions[i], clustered_orientations[i])
      clustered_base_positions.append(clustered_base_position)
      clustered_base_orientations.append(clustered_base_orientation)


    print("Clustered positions: ")
    print(clustered_positions)

    print("Clustered base positions: ")
    print(clustered_base_positions)

    # convert all clusters back to two pose arrays
    ee_posearr = PoseArray()
    b_posearr = PoseArray()
    for i in range(len(clustered_positions)):
      base_pose = np_to_pose(clustered_base_positions[i], clustered_base_orientations[i])
      for j in range(len(clustered_positions[i])):
        print (i, j)
        viewpoint_pose = np_to_pose(clustered_positions[i][j], clustered_orientations[i][j])
        ee_posearr.poses.append(viewpoint_pose)
        b_posearr.poses.append(base_pose)

    # offset_filtered_vp = getOffsetVP(filtered_vp)
    # ac = getClusters(offset_filtered_vp)
    while(not rospy.is_shutdown()):
        viewpoint_pub.publish(ee_posearr)
        base_poses_pub.publish(b_posearr)
        rate.sleep()
   
    # stuff = load_mesh(TANK)
    # mesh_model = stuff[0]
    # plot_sideview(mesh_model)
    # plot_model_normals(TANK)
    # sample_cone_region(dmin=.7, plot=True)


def viewpoint_base_generate():
    viewpoints, sample_kdtree, road_map = dual_viewpoint_sampling(TANK, ROBOT_RADIUS, ARM_LENGTH, HEIGHT, plot=False)
    B, G = cluster_base_position(ARM_LENGTH, viewpoints, sample_kdtree, road_map)


if __name__ == "__main__":
    viewpoint_base_generate()