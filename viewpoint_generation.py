# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import numpy as np
from numpy.linalg import norm
from scipy.spatial.transform import Rotation as R

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from Viewpoints import load_mesh, TANK, plot_3d_object_viewpoints


# %%
# Set input viewpoint constraints
INCIDENCE_ANGLE = np.pi / 6  # 30deg
DMIN = 0.1  # [m]
DMAX = 2  # [m]
CEILING = 2
FLOOR = 0.1


# %%
def sample_viewable_region(mu=500, incidence_angle=INCIDENCE_ANGLE, dmin=DMIN, dmax=DMAX):
    # Generate mu points within a cone 
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
        theta = np.arccos(np.dot(center, point) / norm(point))
        if theta < incidence_angle:
            # add to cone_points
            cone_points[i] = point
            i = i + 1
    return cone_points


def transform_cone(cone_points, point, normal):
    # Cross product of z = [0,0,1] with unit normal for this point will give axis of rotation
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
    # TODO: add other constraints on the viewpoint location like height

    return rotated_cone_points


def compute_visible_points(region, point, points, normals, viewed_points, 
                        fov_angle=np.pi/4, incidence_angle=INCIDENCE_ANGLE, dmin=DMIN, dmax=DMAX, floor=FLOOR, ceiling=CEILING):
    """
    Now we have a set of viewpoints that are all in the visible space for this facet point.
    From each viewpoint, calculate the number of points this viewpoint can see
    
    region: viewpoints in the observable space
    point: the point where sampled a viewable region
    points: all the points in the object
    normals: corresponding normals to each point
    fov_angle: [radians] camera field of view

    Output:
    best_view: a viewpoint position and orientation
    viewed_points: an updated list of seen and unseen points
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

    return best_view, viewed_points


# %%
def dual_viewpoint_sampling(polygon, mu=500, constraints=None, plot=False):
    """
    Inputs:
    polygon: path filename to a STL object
    mu: (default: 500)number of samples per viewpoint iteration
    constraints: (optional) geometric viewing restrictions 

    Output: a near optimal set S of viewpoints covering the object's boundary

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
    # Initialize data structure of unseen points
    unseen = np.zeros(num_points, dtype=int)
    # Initialize samples for observable region
    cone_points = sample_viewable_region(mu)
    # Loop until done, or timeout. TODO: what are timeout conditions
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
            viewpoint, unseen = compute_visible_points(region, point, points, unit_norms, unseen)
            viewpoint_set.append(viewpoint)
        except Exception:
            unseen[point_idx] = 2
        # 3. 
        # Unseen points list is updated. Set of viewpoints is added
    
    viewpoint_set = np.array(viewpoint_set)
    
    if plot:
        plot_3d_object_viewpoints(mesh_model, viewpoint_set)
    return viewpoint_set


def main():
    viewpoints = dual_viewpoint_sampling(TANK, plot=True)


if __name__ == "__main__":
    main()