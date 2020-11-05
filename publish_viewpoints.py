#!/usr/bin/env python2


import roslib; roslib.load_manifest('visualization_marker_tutorials')
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
import rospy
import math
import numpy as np
import rospy
from Viewpoints import TANK, load_mesh, viewpoint_clusters
from probabilistic_road_map import prm_planning
from matplotlib import cm
from Viewpoints import TANK, create_obstacles, load_mesh, create_viewpoints, viewpoint_clusters
from geometry_msgs.msg import PoseArray, Pose
MARKER_TOPIC = 'visualization_traj'
BASE_POSE_TOPIC = 'base_goal_poses'
ARM_POSE_TOPIC = 'arm_goal_poses'


def readPoints():
    # Load the model
    print("Loading Model")
    mesh_model, facets, incidence_normals, mesh_centers, n = load_mesh(TANK)
    obstacles = create_obstacles(facets)
    viewpoints, normals = create_viewpoints(mesh_model)
    cluster_groups, cluster_centers = viewpoint_clusters(viewpoints)

    n_clusters = cluster_centers.shape[0]

    # Probabilistic Road Map algorithm
    print("Running Probabilistic Road Map algorithm on {} clusters.".format(n_clusters))
    robot_size = .6  # [m]
    goal_points, feasible_dist, tsp_path, samples = prm_planning(cluster_centers, obstacles, robot_size)

    # For each goal point, get heading to first viewpoint in that group
    goal_heading_xy = np.zeros((n_clusters, 2))
    for g in range(n_clusters):
        goal = goal_points[g]
        view_group_idx = cluster_groups == (g+1)
        view_group = viewpoints[view_group_idx]
        goal_heading_xy[g] = view_group[0][:2] - goal

    return viewpoints, normals, cluster_groups, goal_points, goal_heading_xy


def  pose_from_vector3D(position, dir_vec):
    #http://lolengine.net/blog/2013/09/18/beautiful-maths-quaternion-from-vectors
    # print("pos shape = {}, dir shape = {}".format(np.shape(position), np.shape(dir_vec)))
    pose= Pose()
    pose.position.x = position[0]
    pose.position.y = position[1]
    pose.position.z = position[2] 
    #calculating the half-way vector.
    u = np.array([1,0,0])
    norm = np.linalg.norm(dir_vec)
    v = np.asarray(dir_vec)/norm 
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
    norm = math.sqrt(pose.orientation.x*pose.orientation.x + pose.orientation.y*pose.orientation.y + 
        pose.orientation.z*pose.orientation.z + pose.orientation.w*pose.orientation.w)
    if norm == 0:
        norm = 1
    pose.orientation.x /= norm
    pose.orientation.y /= norm
    pose.orientation.z /= norm
    pose.orientation.w /= norm
    return pose

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
                          np.random.uniform(low=0.9, high=1)) for i in xrange(nlabels)]

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
                          np.random.uniform(low=low, high=high)) for i in xrange(nlabels)]

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
        # plt.show()

    return random_colormap

def get3DDistance(a, b):
    return np.linalg.norm(a-b)

def main():
    viewpoints, normals, cluster_groups, goal_points, goal_heading_xy = readPoints()
    # list_, viewpoint_normals, groups, centers, goal_heading_xy = readPoints()
    offset = np.array([1, 2.5, 0])

    viewpoints += offset  # offset the model from the origin in Gazebo
    goal_points += offset[:2]
    # cmap = cm.get_cmap('RdBu')
    print("cluster max",np.max(cluster_groups))
    cmap = rand_cmap(np.max(cluster_groups)+2, type='bright', first_color_black=True, last_color_black=False, verbose=True)
    

    # print("Publishing topic {}".format(topic))
    publisher = rospy.Publisher(MARKER_TOPIC, MarkerArray, queue_size=5)
    base_pub = rospy.Publisher(BASE_POSE_TOPIC, PoseArray, queue_size=5)
    arm_pub = rospy.Publisher(ARM_POSE_TOPIC, PoseArray, queue_size=5)
    

    rospy.init_node('register')

    markerArray = MarkerArray()
    count = 0
    # MARKERS_MAX = len(list_)
    r = rospy.Rate(0.5)

    base_poses = PoseArray()
    arm_poses = PoseArray()
    print("len of viewpoints = {}, normal = {}, cluster_groups = {}, goal_pts = {}, goal_dir = {}".format(np.shape(viewpoints), np.shape(normals), np.shape(cluster_groups),np.shape(goal_points), np.shape(goal_heading_xy)))
    
    # Add cluster centers on the ground
    i = 0
    valid_idx = []
    unique_goal_points = []
    for base_pos, base_dir in zip(goal_points, goal_heading_xy):
    # for i, c in enumerate(goal_points):
        marker = Marker()
        color = cmap(i)
        base_pose = Pose()
        marker.header.frame_id = "map"
        marker.type = marker.ARROW
        marker.action = marker.ADD
        marker.scale.x = 0.5
        marker.scale.y = 0.5
        marker.scale.z = 0
        marker.color.a = color[3]
        marker.color.r = color[0]
        marker.color.g = color[1]
        marker.color.b = color[2]
        if( (base_pos[1]<-0.15) or (base_pos[0]<0.8 and base_pos[1]<3.0 )):
            valid_idx.append(i)
            if len(unique_goal_points) == 0:
                base_pose = pose_from_vector3D(np.array([base_pos[0], base_pos[1] ,0.0]), np.array([base_dir[0], base_dir[1], 0.0]))
                unique_goal_points.append(base_pos)
                base_poses.poses.append(base_pose)
                marker.pose = base_pose
                markerArray.markers.append(marker)
            else:
                exist_flag = False
                for goal in unique_goal_points:
                    if get3DDistance( np.array([base_pos[0], base_pos[1], 0.0]), np.array([goal[0], goal[1], 0.0]) ) < 0.2:
                        exist_flag = True
                        break
                if exist_flag == False:
                    base_pose = pose_from_vector3D(np.array([base_pos[0], base_pos[1] ,0.0]), np.array([base_dir[0], base_dir[1], 0.0]))
                    unique_goal_points.append(base_pos)
                    base_poses.poses.append(base_pose)
                    marker.pose = base_pose
                    markerArray.markers.append(marker)

        i+=1
    
    # Publish the Viewpoint Markers
    for view_pos, dir_vec, group in zip(viewpoints, normals, cluster_groups):
        marker = Marker()
        if(view_pos[2]>1.3 or view_pos[2]<0.0):
            color = cmap(0)
            continue
        else:
            color = cmap(group)
        # print("color = ", color)
        arm_pose = Pose()

        if(group in valid_idx and (view_pos[1]<-0.05 or view_pos[0]<0.9) ):
        # if(view_pos[1]<-0.15 or view_pos[0]<0.8):
            arm_pose = pose_from_vector3D(np.array([view_pos[0], view_pos[1] ,view_pos[2]]), np.array([-dir_vec[0], -dir_vec[1], -dir_vec[2]]))

            # marker.header.frame_id = "/bvr_SIM/bvr_base_inertia"
            # bvr_SIM/bvr_base_link
            marker.header.frame_id = "map"
            marker.type = marker.ARROW
            marker.action = marker.ADD
            marker.scale.x = 0.1
            marker.scale.y = 0.1
            marker.scale.z = 0.1
            marker.color.a = 1.0
            marker.color.r = color[0]
            marker.color.g = color[1]
            marker.color.b = color[2]
            # marker.pose.orientation.x = view_pos[5]
            # marker.pose.orientation.y = view_pos[4]
            # marker.pose.orientation.z = view_pos[3]
            # marker.pose.orientation.w = view_pos[6]
            marker.pose = arm_pose
            markerArray.markers.append(marker)
            arm_poses.poses.append(arm_pose)


    # Renumber the marker IDs
    id = 0
    for m in markerArray.markers:
        m.id = id
        id += 1



        # Publish the MarkerArray
    while not rospy.is_shutdown():
        publisher.publish(markerArray)
        base_pub.publish(base_poses)
        arm_pub.publish(arm_poses)

        count += 1

        r.sleep()


if __name__ == "__main__":
    main()
