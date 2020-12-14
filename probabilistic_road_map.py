"""

Probabilistic Road Map (PRM) Planner

author: Atsushi Sakai (@Atsushi_twi)
modified by: Andrew Washburn
github: https://github.com/AtsushiSakai/PythonRobotics/blob/master/PathPlanning/ProbabilisticRoadMap/probabilistic_road_map.py
Accessed: 20 Oct 2020

"""

import random
import pickle
import math
import os
import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
import matplotlib
from scipy.spatial import cKDTree, distance
from scipy.cluster.hierarchy import linkage, fcluster
from InvokeLKH import writeTSPLIBfile_FE, run_LKHsolver_cmd, copy_toTSPLIBdir_cmd, return_tsp_path_list, rm_solution_file_cmd, run_tsp
from Viewpoints import TANK, create_obstacles, load_mesh, create_viewpoints, viewpoint_clusters


# parameter
N_SAMPLE = 500  # number of sample_points
N_KNN = 10  # number of edge from one sampled point
MAX_EDGE_LEN = 5.0  # [m] Maximum edge length
CWD = os.path.dirname(os.path.abspath(__file__))

show_animation = True


class Node:
    """
    Node class for dijkstra search
    """

    def __init__(self, x, y, cost, parent_index):
        self.x = x
        self.y = y
        self.cost = cost
        self.parent_index = parent_index

    def __str__(self):
        return str(self.x) + "," + str(self.y) + "," +\
               str(self.cost) + "," + str(self.parent_index)


def prm_planning(goals, obstacles, rr):
    """
    sx, sy: start xy position
    goals: ndarray(n_g, 2) goal positions
    obstacles: ndarray(n, 2) obstacle points
    rr: robot radius [m]

    Returns:
    goal_points: feasible goal xy coordinates
    feasible_distance: distance from requested goal to feasible goal
    tsp_path: list of indexes in goal_points for shortest path
    """
    n_g = goals.shape[0]
    # put obstacles in a kdtree
    obstacle_kd_tree = cKDTree(obstacles)
    # Sample points contains N_SAMPLE + n_g points
    print("PRM sample points")
    sample_x, sample_y = sample_points(rr, obstacle_kd_tree)
    
    # Add goals as an array. 
    print("Building road map")
    road_map, sample_kd_tree = generate_road_map(sample_x, sample_y, rr, obstacle_kd_tree)

    plt.scatter(sample_kd_tree.data[:,0], sample_kd_tree.data[:,1], c='b')
    plt.scatter(obstacle_kd_tree.data[:,0], obstacle_kd_tree.data[:,1], c='k')

    # Ensure all goals can be found from a single starting point
    bottom_left = sample_kd_tree.mins
    _, start_idx = sample_kd_tree.query(bottom_left)
    # If goal is infeasible, build path to closest sampled point in road map
    goal_edges = connect_goals(start_idx, goals, road_map, sample_kd_tree)

    # For all the feasible points, get pairwise distances
    print("Computing pair wise distances.")
    feasible_goal_idx = [e[1] for e in goal_edges]
    cost_matrix = pair_wise_distances(feasible_goal_idx, road_map, sample_x, sample_y)

    # Get TSP shortest path. tsp_path is a list whose elements correspond to feasible_goal_idx
    tsp_path = run_tsp(cost_matrix, "ClusterPaths", "TSP shortest complete paths for all feasible clustered viewpoints.")
    
    feasible_dist = [e[1] for e in goal_edges]
    goal_points = [[sample_x[i], sample_y[i]] for i in feasible_goal_idx]
    goal_points = np.array(goal_points)

    # Return goal coordinates, and tsp_path
    return goal_points, feasible_dist, tsp_path, np.array((sample_x, sample_y)).T


def num_infeasible_goals(feasible_dist):
    # Show off total distance and number of infeasible goals
    t_dist = 0
    num_infeasible_goals = 0
    for d in feasible_dist:
        t_dist += d
        if d > 0:
            num_infeasible_goals += 1

    print("Total infeasible goals {} / {}. Total distance shortened {:.1f} [m]".format(num_infeasible_goals, len(feasible_dist), t_dist))


def plot_tsp(goal_points, feasible_dist, tsp_path, obstacles, samples):

    # Plot 2D obstacles
    fig, ax = plt.subplots()

    # ax.plot(sample_x, sample_y, ".b")
    ax.plot(obstacles[:,0], obstacles[:,1], ".k")
    # ax.plot(goals[:,0], goals[:,1], 'xc')
    ax.grid(True)
    ax.axis("equal")
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    # For each goal, label goal point and feasible point with an edge
    for g, goal in enumerate(goal_points):
        
        # plt.gcf().canvas.mpl_connect(
        #     'key_release_event',
        #     lambda event: [exit(0) if event.key == 'escape' else None])
        if feasible_dist[g] == 0:
            # print(f"Goal {g} is feasible.")
            ax.plot(goal[0], goal[1], "+b")
        else:
            # print(f"Goal {g} is infeasible. Closest point is {goal[0]:.1f} [m] away. ", end="")
            g_idx = g + N_SAMPLE
            neighbor_p = (samples[g_idx, 0], samples[g_idx, 1])
            # print(f"Goal ({goal_p[0]:.1f}, {goal_p[1]:.1f}), Nearest Neighbor ({neighbor_p[0]:.1f}, {neighbor_p[1]:.1f}).")
            # ax.plot(
            #     [neighb or_p[0], goal_p[0]],
            #     [neighbor_p[1], goal_p[1]],
            #     "-r"
            # )
            ax.plot(goal[0], goal[1], "+b")
            ax.plot(neighbor_p[0], neighbor_p[1], ".b")
            
        # plt.pause(0.01)

    # Plot the TSP path
    n_p = len(tsp_path)
    for i in range(n_p - 1):
        p0 = goal_points[tsp_path[i]]
        p1 = goal_points[tsp_path[i + 1]]
        ax.plot(
            [p0[0], p1[0]],
            [p0[1], p1[1]],
            "-g"
        )

    plt.show()

    # Show edges to goal points 
    # plot_goal_road_map(road_map, goals, sample_kd_tree)
    

    # # Compute pairwise paths and distances between all goals
    # num_paths = n_g * (n_g - 1) / 2
    # print(f"There are {n_g} goals and {num_paths} pairwise paths.")
    # paths = []
    # for g1 in range(N_SAMPLE, N_SAMPLE + n_g-1):
    #     for g2 in range(g1, N_SAMPLE + n_g):
    #         rx, ry, path_found = dijkstra_planning(g1, g2, road_map, sample_x, sample_y, debug=False)
    #         if not path_found and False:
    #             fig, ax = plt.subplots()
    #             ax.plot(obstacles[:,0], obstacles[:,1], ".k")
    #             ax.plot(sample_x, sample_y, ".b")
    #             ax.plot(sample_x[g1], sample_y[g1], "^g")
    #             ax.plot(sample_x[g2], sample_y[g2], "^r")
    #             g1_edges = road_map[g1]
    #             g2_edges = road_map[g2]
    #             for e1 in g1_edges:
    #                 ax.plot([sample_x[g1], sample_x[e1]], 
    #                         [sample_y[g1], sample_y[e1]], "-k")
    #             for e2 in g2_edges:
    #                 ax.plot([sample_x[g2], sample_x[e2]], 
    #                         [sample_y[g2], sample_y[e2]], "-k")

    #             ax.plot(rx, ry, "xg")
    #             plt.show()

    #         paths.append((rx, ry))
    return None


def is_collision(sx, sy, gx, gy, rr, obstacle_kd_tree):
    x = sx
    y = sy
    dx = gx - sx
    dy = gy - sy
    yaw = math.atan2(dy, dx)
    d = math.hypot(dx, dy)

    if d >= MAX_EDGE_LEN:
        return True

    D = rr
    n_step = int(d / D)

    for i in range(n_step):
        dist, _ = obstacle_kd_tree.query([x, y])
        if dist <= rr:
            return True  # collision
        x += D * math.cos(yaw)
        y += D * math.sin(yaw)

    # goal point check
    dist, _ = obstacle_kd_tree.query([gx, gy])
    if dist <= rr:
        return True  # collision

    return False  # OK


def generate_road_map(sample_x, sample_y, rr, obstacle_kd_tree, height=0):
    """
    Road map generation of the free workspace. Returned graph is wholly connected.
    Inputs:
        sample_x: [m] x positions of sampled points
        sample_y: [m] y positions of sampled points
        rr: Robot Radius[m]
        obstacle_kd_tree: KDTree object of obstacles

    Returns:
        road_map: adjacency list of graph edges
        sample_kd_tree: cKDTree of nodes in the free space
    """

    road_map = []
    n_sample = len(sample_x)
    sample_kd_tree = cKDTree(np.vstack((sample_x, sample_y)).T)

    for (ix, iy) in zip(sample_x, sample_y):

        _, indexes = sample_kd_tree.query([ix, iy], k=n_sample)
        edge_id = []

        for ii in range(1, len(indexes)):
            nx = sample_x[indexes[ii]]
            ny = sample_y[indexes[ii]]

            if not is_collision(ix, iy, nx, ny, rr, obstacle_kd_tree):
                edge_id.append(indexes[ii])

            if len(edge_id) >= N_KNN:
                break

        road_map.append(edge_id)

    # Now prune the road map to include only the free workspace accessible from a long way away.
    bottom_left = sample_kd_tree.mins
    _, start_idx = sample_kd_tree.query(bottom_left)
    # Breadth First Search through graph, getting all connected nodes.
    colors = np.zeros(n_sample)  # 0 - white; 1 - gray; 2 - black
    distance = np.zeros(n_sample) + np.Inf
    parents = np.zeros(n_sample) + np.NAN
    
    colors[start_idx] = 1
    distance[start_idx] = 0
    parents[start_idx] = np.NAN

    queue = [start_idx]
    while len(queue) > 0:
        u = queue.pop(0)
        for v in road_map[u]:
            if colors[v] == 0:
                colors[v] = 1
                distance[v] = distance[u] + 1
                parents[v] = u
                queue.append(v)
        colors[u] = 2
    # Extract point indicies in free workspace
    free_points = np.argwhere(colors).squeeze()
    free_graph = [road_map[p] for p in free_points]
    x_free = [sample_x[i] for i in free_points]
    y_free = [sample_y[i] for i in free_points]
    free_kdtree = cKDTree(np.vstack((x_free, y_free, np.ones(len(x_free)) * height)).T)
    
    return free_graph, free_kdtree


def connect_goals(start_idx, goals, road_map, sample_kd_tree):
    """
    The start_idx is a point in the free configuration space.
    The roap map contains a collision free graph of the configuration space.
    Iterate through all the goals to make sure they're in the free configuration space.

    returns a list each index corresponds to a goal, elements are (dist, closest_point_index).

    start_idx: data index from sample_kd_tree 
    goals: ndarray(n_g, 2) goals to connect to the road map.
    road_map: a list of lists where each index corresponds to the point in sample_kd_tree.data. Goals' edges are in last n_g indicies
    sample_kd_tree: kdtree of all the sampled points and goals in the configuration space
    """
    n_g = goals.shape[0]
    n_sample = sample_kd_tree.n - n_g  # subtract the number of goal nodes from the sampled points
    goal_edge_list = [None] * n_g
    sample_x = sample_kd_tree.data[:,0]
    sample_y = sample_kd_tree.data[:,1]

    for g in range(n_g):
        # Run Planning algorithm
        # If no path found. Return the closed set and use this to create an edge to the unfeasible goal.
        # IDEA: Instead of creating an edge, set the closest closed set point as goal?
        closed_set, path_found = dijkstra_planning(start_idx, n_sample + g, road_map, sample_x, sample_y)

        # Find the closest point in the closed set.
        dists, indexes = sample_kd_tree.query(goals[g], k=N_SAMPLE+n_g)

        i = 0  # first kdtree query result is the goal itself
        end = len(indexes)

        # If no path found, iterate through all nearest neighbors until found a point in the closed set
        if not path_found:
            while i < end and indexes[i] not in closed_set:
                i += 1
            if i == end:
                raise IndexError("Could not find a path between the start and goal.")
        
        # Store distance and node id in goal_edge_list
        # IDEA: store the distance in a separate list as a measurement for experimenting with different algorithms.
        goal_edge_list[g] = (dists[i], indexes[i])

    return goal_edge_list


def pair_wise_distances(points, road_map, sample_x, sample_y):
    """
    points: a list of indexes correspoinding to samples
    road_map: adjacency list of all connected samples
    sample_xy: xy point coordinates
    """
    n_p = len(points)
    path_lengths = np.zeros((n_p, n_p))  # [start, end] 

    for i in range(n_p):
        for j in range(i+1, n_p):
            path, path_found = dijkstra_planning(points[i], points[j], road_map, sample_x, sample_y)
            if not path_found:
                raise IndexError("Could not find a path between feasible points.")
            # path is a 2D list of point_indexes. Get the distance for each n-1 edges and sum
            for p in range(len(path) - 1):
                p0_idx = path[p]
                p0 = np.array(sample_x[p0_idx], sample_y[p0_idx])
                p1_idx = path[p+1]
                p1 = np.array(sample_x[p1_idx], sample_y[p1_idx])
                length = distance.euclidean(p0, p1)
                path_lengths[i, j] += length
            path_lengths[j, i] = path_lengths[i, j]  # make full matrix
    
    return path_lengths


def dijkstra_planning(start_idx, goal_idx, road_map, sample_x, sample_y):
    """
    s_x: start x position [m]
    s_y: start y position [m]
    gx: goal x position [m]
    gy: goal y position [m]
    ox: x position list of Obstacles [m]
    oy: y position list of Obstacles [m]
    rr: robot radius [m]
    road_map: ??? [m]
    sample_x: ??? [m]
    sample_y: ??? [m]

    return: Two lists of path coordinates ([x1, x2, ...], [y1, y2, ...]), empty list when no path was found
    If no path is found, the closed set of visited node indexes is returned as rx and ry is an empty list
    """

    start_node = Node(sample_x[start_idx], sample_y[start_idx], 0.0, -1)
    goal_node = Node(sample_x[goal_idx], sample_y[goal_idx], 0.0, -1)

    open_set, closed_set = dict(), dict()
    open_set[start_idx] = start_node

    path_found = True

    while True:
        if not open_set:
            # No more nodes to explore
            path_found = False
            break

        c_id = min(open_set, key=lambda o: open_set[o].cost)
        current = open_set[c_id]

        # show graph
        if False and len(closed_set.keys()) % 2 == 0:
            # for stopping simulation with the esc key.
            plt.gcf().canvas.mpl_connect(
                'key_release_event',
                lambda event: [exit(0) if event.key == 'escape' else None])
            plt.plot(current.x, current.y, "xg")
            plt.pause(0.001)

        if c_id == (goal_idx):
            # print("goal is found!")
            goal_node.parent_index = current.parent_index
            goal_node.cost = current.cost
            break

        # Remove the item from the open set
        del open_set[c_id]
        # Add it to the closed set
        closed_set[c_id] = current

        # expand search grid based on motion model
        for i in range(len(road_map[c_id])):
            n_id = road_map[c_id][i]
            dx = sample_x[n_id] - current.x
            dy = sample_y[n_id] - current.y
            d = math.hypot(dx, dy)
            node = Node(sample_x[n_id], sample_y[n_id],
                        current.cost + d, c_id)

            if n_id in closed_set:
                continue
            # Otherwise if it is already in the open set
            if n_id in open_set:
                if open_set[n_id].cost > node.cost:
                    open_set[n_id].cost = node.cost
                    open_set[n_id].parent_index = c_id
            else:
                open_set[n_id] = node

    if path_found is False:
        # return indicies inside the closed set. 
        return list(closed_set), path_found

    # generate final course
    path_ids = [c_id]
    c_id = goal_node.parent_index
    while c_id != -1:
        path_ids.insert(0, c_id)
        n = closed_set[c_id]
        c_id = n.parent_index

    return path_ids, path_found


def plot_goal_road_map(road_map, goals, sample_kd_tree):
    n_g = goals.shape[0]
    n_sample = sample_kd_tree.n - n_g
    goal_edges = road_map[n_sample: n_sample + n_g]

    for g, edges in enumerate(goal_edges):
        goal = goals[g]
        for e in edges:
            point = sample_kd_tree.data[e]

            plt.plot(
                [goal[0], point[0]],
                [goal[1], point[1]], 
                "-g")


def sample_points(rr, obstacle_kd_tree, boundary=3, n_sample=N_SAMPLE):
    """
    Randomly sample points in the free configuration space.
    Returns sample_x, sample_y
    """
    max_x, max_y = obstacle_kd_tree.maxes
    min_x, min_y = obstacle_kd_tree.mins
    max_x += boundary
    min_x -= boundary
    max_y += boundary
    min_y -= boundary

    sample_x, sample_y = [], []

    while len(sample_x) < n_sample:
        tx = (random.random() * (max_x - min_x)) + min_x
        ty = (random.random() * (max_y - min_y)) + min_y

        dist, index = obstacle_kd_tree.query([tx, ty])

        if dist >= rr:
            sample_x.append(tx)
            sample_y.append(ty)

    # Removing goals from initial PRM generation. -ALW 23Nov20
    # for g in range(goals.shape[0]):
    #     sample_x.append(goals[g,0])
    #     sample_y.append(goals[g,1])

    return sample_x, sample_y


def check_inline(p, s, u, incidence_angle, dmin=0, dmax=None):
    """
    Checks if a point s lies within a cone incidence constraint.
    I think this works in any dimension so long as p, s, and u are the same dimension.

    Params:
        p: point defining cone base
        s: point to check if within incidence cone
        u: unit direction of cone from p
        incidence_angle: width of cone 
    """ 
    vec = s - p
    length = norm(vec)
    unit_vec = vec / length
    theta = np.arccos(np.dot(vec, u))
    to_ret = length > dmin
    if dmax is not None:
        to_ret = to_ret and (length < dmax)
    to_ret = to_ret and (theta <= incidence_angle)
    return to_ret


def cluster_base_position(ell, viewpoints, sample_kd_tree, road_map, arm_angle=np.pi/3):
    """
    Find a minimal set of base positions for the viewpoint set.
    
    Params:
        ell: [m] robot arm length
        arm_angle: [rad] base position angle to viewpoint direction constraint. Default 60deg.
        viewpoints: (n_vp, 6) viewpoint poses
        sample_kd_tree: cKDTree of points representing free space
        road_map: adjacency list of sample space

    Returns:
        B: (n_b, 6) an array of base poses
        G: list of lists of viewpoint poses. First index corresponds to base pose index. Second contains corresponding viewpoints

    First, calculate pairwise distances between all viewpoints. If a viewpoint pair shares
    a nearest neighbor within ell distance in the sample space, 
    then distance between these viewpoint is the distance between that nearest neighbor and
    the viewpoint. Else viewpoints don't share a nearest neighbor, distance between the two 
    is length to closest point in sample space and the dijkstra path length between the free space points.
    Pairwise distances feed into a heirarchial clustering algorithm using a complete linkage.
    The dendrogram is pruned at 2*ell. For each viewpoint cluster, find a common nearest neighbor
    that is optimal. We choose the neighbor that minimizes the distance to all viewpoints.
    Store the viewpoints in a list structure `G` that corresponds to each base position. 
    """
    # Find pair wise distances between each viewpoint
    n_vp = viewpoints.shape[0]
    p_dist = np.zeros((n_vp, n_vp))
    for i in range(n_vp):
        for j in range(i+1, n_vp):
            # First: find common nearest neighbors
            vp0 = viewpoints[i, 0:3]
            vu0 = viewpoints[i, 4:6]
            vp1 = viewpoints[j, 0:3]
            vu1 = viewpoints[i, 4:6]
            nn0 = sample_kd_tree.query_ball_point(vp0, ell)
            nn1 = sample_kd_tree.query_ball_point(vp1, ell)
            common_nn = set(nn0).intersection(set(nn1))  # contains indices of sample_kd_tree data
            common_nn = [x for x in common_nn if \
                    (check_inline(vp0[:2], sample_kd_tree.data[x][:2], -vu0, arm_angle, 0.1)) and \
                    (check_inline(vp1[:2], sample_kd_tree.data[x][:2], -vu1, arm_angle, 0.1))
                ]
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
        vpose = np.array([viewpoints[j] for j, c in enumerate(cluster_assignment) if c == i+1])
        vpos = vpose[:, 0:3]
        vorientation = vpose[:, 3:6]
        G.append(vpose)
        bp = [set(sample_kd_tree.query_ball_point(p, ell)) for p in vpos]  # query basepositions within an arm length
        common_bp = set.intersection(*bp)
        # Filter base positions behind the viewpoints
        base_pos_candidates = []
        for idx in common_bp:
            base_position = sample_kd_tree.data[idx]
            feasible = True
            # TODO: Uncomment
            # for vp in vpose:
            #     feasible = feasible and check_inline(vp[0:2], base_position[:2], -vp[4:6], arm_angle, 0.1)
            if feasible:
                base_pos_candidates.append(base_position)
        if len(base_pos_candidates) == 0:
            raise Exception("No base positions found for a cluster.")
        base_pos_candidates  = np.array(base_pos_candidates)
        # vpos_bp_dist: each row corresponds to a viewpoint and column to a base position; 
        # entry is euclidean distance between the two
        vpos_bp_dist = distance.cdist(vpose[:, 0:3], base_pos_candidates)
        # Choose base position with min total distance. 
        # TODO: What would be other metrics?
        opt_bp_idx = vpos_bp_dist.sum(axis=0).argmin()
        opt_bp_pos = base_pos_candidates[opt_bp_idx]
        opt_bp_pos[2] = 0  # set position height to ground level
        # TODO: Determine base orientation
        B[i, 0:2] = opt_bp_pos[0:2]
    
    return B, G


def plot_freespace(rr, ell, boundary, n_samples):
    """
    param
        rr: robot radius for calculating obstacle free space
        ell: arm length reachable viewpoints
        boundary: perimeter of free space around obstacle
        n_samples: number of points in PRM
    """
    viewpoints = np.load(CWD + "/viewpoints.npy")
    mesh_model, facets, incidence_normals, mesh_centers, n = load_mesh(TANK)
    obstacles = create_obstacles(facets)
    obstacle_kd_tree = cKDTree(obstacles)
    print("PRM sample points")
    sample_x, sample_y = sample_points(rr, obstacle_kd_tree, boundary, n_samples)
    
    print("Building road map")
    road_map, sample_kd_tree = generate_road_map(sample_x, sample_y, rr, obstacle_kd_tree, height=.5)

    B, G = cluster_base_position(ell, viewpoints, sample_kd_tree, road_map)


    # Plot PRM, obstacle, and viewpoints
    plt.scatter(sample_kd_tree.data[:,0], sample_kd_tree.data[:,1], c='b', alpha=.2)
    plt.scatter(obstacle_kd_tree.data[:,0], obstacle_kd_tree.data[:,1], c='k', alpha=.2)
    # Draw a Circle with arm_length radius around each viewpoint
    n_c = B.shape[0]
    cmap = plt.cm.rainbow
    norm = matplotlib.colors.Normalize(vmin=0, vmax=n_c)
    # Plot base position and corresponding viewpoint
    for i in range(n_c):
        b = B[i]
        vp = G[i]
        plt.scatter(b[0], b[1], color=cmap(norm(i)), marker="v")
        for v in vp:
            plt.plot(
                [b[0], v[0]],
                [b[1], v[1]],
                color=cmap(norm(i)),
                linestyle="dashdot"
            )
            plt.scatter(v[0], v[1], color=cmap(norm(i)))
            circle = plt.Circle((v[0], v[1]), radius=ell, color=cmap(norm(i)), fill=False, linestyle="dashed")
            plt.gcf().gca().add_artist(circle)

    plt.axis('equal')
    plt.show()


def main():
    print(__file__ + " start!!")
    
    print("Loading Model")
    mesh_model, facets, incidence_normals, mesh_centers, n = load_mesh(TANK)
    obstacles = create_obstacles(facets)
    viewpoints, normals = create_viewpoints(mesh_model)
    cluster_groups, cluster_centers = viewpoint_clusters(viewpoints)

    n_clusters = cluster_centers.shape[0]

    # Probabilistic Road Map algorithm
    print("Running Probabilistic Road Map algorithm on {} clusters.".format(n_clusters))
    robot_size = .5  # [m]
    goal_points, feasible_dist, tsp_path, samples = prm_planning(cluster_centers, obstacles, robot_size)
    plot_tsp(goal_points, feasible_dist, tsp_path, obstacles, samples)


if __name__ == '__main__':
    plot_freespace(.5, 1, 2, 1000)