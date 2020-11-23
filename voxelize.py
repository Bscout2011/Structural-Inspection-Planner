import open3d as o3d
from open3d.visualization import draw_geometries
import numpy as np


print('input')
model = "Mesh/Elevated_Tank.STL"  # choose the mesh model
if not exists(model):
    raise FileNotFoundError(f"{model} not found in current file folder.")

memesh = o3dtut.get_bunny_mesh()

draw_geometries([mesh])

print('voxelization')
voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh(mesh,
                                                              voxel_size=0.05)
o3d.visualization.draw_geometries([voxel_grid])