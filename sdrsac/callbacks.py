import copy
import numpy as np
import open3d as o3


class Open3dVisualizerCallback(object):
    """Display the 3D registration result of each iteration.

    Args:
        source (numpy.ndarray): Source point cloud data.
        target (numpy.ndarray): Target point cloud data.
        save (bool, optional): If this flag is True,
            each iteration image is saved in a sequential number.
        keep_window (bool, optional): If this flag is True,
            the drawing window blocks after registration is finished.
    """
    def __init__(self, source, target, save=False,
                 keep_window=True):
        self._vis = o3.visualization.Visualizer()
        self._vis.create_window()
        self._source = source
        self._target = target
        self._result = copy.deepcopy(self._source)
        self._save = save
        self._keep_window = keep_window
        self._source.paint_uniform_color([1, 0, 0])
        self._target.paint_uniform_color([0, 1, 0])
        self._result.paint_uniform_color([0, 0, 1])
        self._vis.add_geometry(self._source)
        self._vis.add_geometry(self._target)
        self._vis.add_geometry(self._result)
        self._cnt = 0

    def __del__(self):
        if self._keep_window:
            self._vis.run()
        self._vis.destroy_window()

    def __call__(self, rot, t):
        self._result.points = copy.deepcopy(self._source.points)
        trans = np.zeros((4, 4))
        trans[:3, :3] = rot
        trans[:3, 3] = t
        self._result.transform(trans)
        self._vis.update_geometry(self._source)
        self._vis.update_geometry(self._target)
        self._vis.update_geometry(self._result)
        self._vis.poll_events()
        self._vis.update_renderer()
        if self._save:
            self._vis.capture_screen_image("image_%04d.jpg" % self._cnt)
        self._cnt += 1
