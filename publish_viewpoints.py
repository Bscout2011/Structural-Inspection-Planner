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

    return viewpoints, cluster_groups, goal_points, goal_heading_xy


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

def main():

    list_, groups, centers, goal_heading_xy = readPoints()
    offset = np.array([2, 5, 0])

    list_ += offset  # offset the model from the origin in Gazebo
    centers += offset[:2]
    # cmap = cm.get_cmap('RdBu')
    cmap = rand_cmap(np.max(groups)+2, type='bright', first_color_black=True, last_color_black=False, verbose=True)
    

    topic = 'visualization_traj'
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
        if(elem[2]>1.5 or elem[2]<0.0):
            color = cmap(0)
        else:
            color = cmap(group)
        print("color = ", color)
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
