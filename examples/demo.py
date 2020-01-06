import copy
import numpy as np
import open3d as o3
import sdrsac

# load source and target point cloud
source = o3.io.read_point_cloud('bunny.pcd')
target = copy.deepcopy(source)
th = np.deg2rad(90.0)
target.transform(np.array([[np.cos(th), -np.sin(th), 0.0, 0.0],
                           [np.sin(th), np.cos(th), 0.0, 0.0],
                           [0.0, 0.0, 1.0, 0.5],
                           [0.0, 0.0, 0.0, 1.0]]))
source = source.voxel_down_sample(voxel_size=0.005)
target = target.voxel_down_sample(voxel_size=0.005)

cbs = [sdrsac.Open3dVisualizerCallback(source, target), lambda rot, t: print(rot, t)]
rot, t = sdrsac.sdrsac(np.array(source.points).T, np.array(target.points).T, callbacks=cbs)
print(rot, t)