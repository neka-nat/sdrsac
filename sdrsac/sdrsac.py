import numpy as np
import open3d as o3
import scipy.spatial as ss
from . import sdp


def to_pointcloud(arr):
    pc = o3.PointCloud()
    pc.points = o3.Vector3dVector(arr.T)
    return pc


def sdrsac(m, b, max_itr=10000,
           n_sample=9,
           k=4,
           d_diff_thresh=1.0e-4,
           pair_dist_thresh=1.0e-2,
           n_pair_thres=5,
           corr_count_eps=0.015,
           icp_thres=0.02,
           callbacks=[]):
    btree = ss.KDTree(b.T)
    max_inls = 0
    ps = 0.99
    best_rot = np.identity(3)
    best_t = np.zeros(3)
    iter = 0
    t_max = 1e10
    stop = False
    while iter < max_itr and not stop:
        idx_m = np.random.choice(m.shape[1], n_sample, False)
        smp_m = m[:, idx_m]
        b_to_sample = b
        scount = 0
        while b_to_sample.shape[1] > n_sample * 2.0 and scount < 2:
            scount += 1
            idx_b = np.random.choice(b_to_sample.shape[1], n_sample, False)
            smp_b = b_to_sample[:, idx_b]
            b_to_sample = np.delete(b_to_sample, idx_b, axis=1)

            rot, t, _, corr_b = sdp.sdp_reg(smp_m, smp_b,
                                            k, d_diff_thresh,
                                            pair_dist_thresh,
                                            n_pair_thres)

            if not corr_b is None and len(corr_b) > 0:
                tm = (np.dot(rot, m).transpose() + t).transpose()
                reg_p2p = o3.registration.registration_icp(to_pointcloud(tm), to_pointcloud(smp_b),
                                                           icp_thres, np.identity(4),
                                                           o3.registration.TransformationEstimationPointToPoint())
                rot_icp = reg_p2p.transformation[:3, :3]
                t_icp = reg_p2p.transformation[:3, 3]
                tm_icp = (np.dot(rot_icp, tm).transpose() + t_icp).transpose()
                inls_icp = sdp.count_correspondence(tm_icp, btree, corr_count_eps)
                if inls_icp > max_inls:
                    max_inls = inls_icp
                    best_rot = np.dot(rot_icp, rot)
                    best_t = np.dot(rot_icp, t) + t_icp
                    p_i = max_inls / m.shape[1]
                    t_max = np.log(1.0 - ps) / max(np.log(1.0 - p_i**k), 1.0e-8)
                    for c in callbacks:
                        c(best_rot, best_t)
            iter += iter
            if iter >= t_max and iter >= 5:
                stop = True
    return best_rot, best_t