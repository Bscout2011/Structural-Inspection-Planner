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

MARKER_TOPIC = 'visualization_traj'
BASE_POSE_TOPIC = 'base_goal_poses'


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
    robot_size = .5  # [m]
    goal_points, feasible_dist, tsp_path = prm_planning(cluster_centers, obstacles, robot_size)

    # For each goal point, get heading to first viewpoint in that group
    goal_heading_xy = np.zeros((n_clusters, 2))
    for g in range(n_clusters):
        goal = goal_points[g]
        view_group_idx = cluster_groups == (g+1)
        view_group = viewpoints[view_group_idx]
        goal_heading_xy[g] = view_group[0][:2] - goal

    return viewpoints, normals, cluster_groups, goal_points, goal_heading_xy


def main():
    list_, viewpoint_normals, groups, centers, goal_heading_xy = readPoints()
    offset = np.array([2, 5, 0])
    list_ += offset  # offset the model from the origin in Gazebo
    centers += offset[:2]
    cmap = cm.get_cmap('RdBu')

    print("Publishing topic {}".format(topic))
    publisher = rospy.Publisher(topic, MarkerArray, queue_size=5)

    rospy.init_node('register')

    markerArray = MarkerArray()
    count = 0
    MARKERS_MAX = len(list_)
    r = rospy.Rate(1)

    # Publish the Viewpoint Markers
    for elem, group in zip(list_, groups):
        marker = Marker()
        color = cmap(group)
        # marker.header.frame_id = "/bvr_SIM/bvr_base_inertia"
        # bvr_SIM/bvr_base_link
        marker.header.frame_id = "map"
        marker.type = marker.CUBE
        marker.action = marker.ADD
        marker.scale.x = 0.1
        marker.scale.y = 0.1
        marker.scale.z = 0.1
        marker.color.a = color[3]
        marker.color.r = color[0]
        marker.color.g = color[1]
        marker.color.b = color[2]
        # marker.pose.orientation.x = elem[5]
        # marker.pose.orientation.y = elem[4]
        # marker.pose.orientation.z = elem[3]
        # marker.pose.orientation.w = elem[6]
        marker.pose.position.x = elem[0]
        marker.pose.position.y = elem[1]
        marker.pose.position.z = elem[2] 

        markerArray.markers.append(marker)

    # Add cluster centers on the ground
    for i, c in enumerate(centers):
        marker = Marker()
        color = cmap(i)

        marker.header.frame_id = "map"
        marker.type = marker.SPHERE
        marker.action = marker.ADD
        marker.scale.x = 0.5
        marker.scale.y = 0.5
        marker.scale.z = 0
        marker.color.a = color[3]
        marker.color.r = color[0]
        marker.color.g = color[1]
        marker.color.b = color[2]
        # marker.pose.orientation.x = elem[5]
        # marker.pose.orientation.y = elem[4]
        # marker.pose.orientation.z = elem[3]
        # marker.pose.orientation.w = elem[6]
        marker.pose.position.x = c[0]
        marker.pose.position.y = c[1]
        marker.pose.position.z = 0 

        markerArray.markers.append(marker)

    # Renumber the marker IDs
    id = 0
    for m in markerArray.markers:
        m.id = id
        id += 1



        # Publish the MarkerArray
    while not rospy.is_shutdown():
        publisher.publish(markerArray)

        count += 1

        r.sleep()


if __name__ == "__main__":
    main()
