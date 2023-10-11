import os
import pickle
import numpy as np

# from data3d.datasets.openlane import OpenLaneDataset
# from data3d.datasets.once_3dlanes import Once3DLanesDataset
# from data3d.datasets.private_lane import PrivateLaneDataset
# from data3d.visualization.lane2d_visualizer import Lane2DVisualizer


class Lane2DDemoManager:
    # _DATASETS = {
    #     "OpenLane": OpenLaneDataset,
    #     "Once3DLanes": Once3DLanesDataset,
    #     "PrivateLane": PrivateLaneDataset,
    # }
    # _IMG_KEYS = {
    #     "OpenLane": "FRONT",
    #     "Once3DLanes": "cam01",
    #     "PrivateLane": "camera_15",
    # }

    def _get_pts(self, lanes, img_metas):
        _2d_pts, _3d_pts = [], []
        if lanes is not None:
            for lane in lanes:
                pts = np.array(lane)
                d_pts=pts.copy()
                d_pts[:,0]=d_pts[:,0]*(-1)
                ds_pts=np.array([d_pts[:,1],d_pts[:,0],d_pts[:,2]]).T
                _3d_pts.append(ds_pts)
                pts = np.hstack([pts, np.ones_like(pts[:, 2:3])])
                pts=pts@img_metas[0]['ego2img'].T
                pts = pts[abs(pts[:, 2]) > 1e-6]
                pts /= pts[:, 2:3]
                _2d_pts.append(pts[:, :2])
        return _2d_pts, _3d_pts


    def _get_gt_pts(self, img_metas):
        _2d_pts, _3d_pts = [], []
        if True:
            for lane_n in range(len(img_metas[0]['gt_points'])):
                pts=img_metas[0]['gt_points'][lane_n][img_metas[0]['gtpoints_vis'][lane_n]>0]
                pts=pts[pts[:,0]>=-10]
                pts=pts[pts[:,0]<=10]
                pts=pts[pts[:,1]>=3]
                pts=pts[pts[:,1]<=103]
                
                d_pts=pts.copy()
                d_pts[:,0]=d_pts[:,0]*(-1)
                ds_pts=np.array([d_pts[:,1],d_pts[:,0],d_pts[:,2]]).T
                _3d_pts.append(ds_pts)
                # import ipdb
                # #设置断点
                # ipdb.set_trace()
                pts = np.hstack([pts, np.ones_like(pts[:, 2:3])])
                pts=pts@img_metas[0]['ego2img'].T
                pts = pts[abs(pts[:, 2]) > 1e-6]
                pts /= pts[:, 2:3]
                _2d_pts.append(pts[:, :2])
        return _2d_pts, _3d_pts

    def _get_item(self, pred_lanes, img_metas,img_file):

        img=img_metas[0]['oriimg'][0]
        lane_2d_points, lane_3d_points = self._get_gt_pts(img_metas)
        pred_2d_points, pred_3d_points = self._get_pts(pred_lanes, img_metas)
        # import ipdb
        # #设置断点
        # ipdb.set_trace()
        return dict(
            img=np.ascontiguousarray(img),
            lane_2d_points=lane_2d_points,
            lane_3d_points=lane_3d_points,
            pred_2d_points=pred_2d_points,
            pred_3d_points=pred_3d_points,
            img_metas=img_metas,
            img_file=img_file,
        )

    def load_pred_dict(self, pred_path):
        with open(pred_path, "rb") as f:
            pred_dict = pickle.load(f)
        return pred_dict

    def run_demo(self, save_dir, dataset_name, data_split, sample_size=10, **kwargs):
        img_key = self._IMG_KEYS[dataset_name]
        dataset = self._DATASETS[dataset_name](data_split=data_split)
        if "pred_path" in kwargs:
            pred_dict = self.load_pred_dict(kwargs["pred_path"])
        else:
            pred_dict = {}
        vis = Lane2DVisualizer(save_path=os.path.join(save_dir, dataset_name + ".mp4"))
        for data_idx in range(min(len(dataset), sample_size)):
            raw = dataset[data_idx]
            pred = pred_dict[data_idx] if data_idx in pred_dict else None
            vis.add_frame(**self._get_item(raw, img_key, pred))
        vis.show()


# if __name__ == "__main__":
#     lane2d_demo_manager = Lane2DDemoManager()
#     # lane2d_demo_manager.run_demo("/data/", "OpenLane", "validation", sample_size=300)
#     # lane2d_demo_manager.run_demo("/data/", "Once3DLanes", "validation", sample_size=300)
#     lane2d_demo_manager.run_demo(
#         "/data/", "PrivateLane", "validation", sample_size=300, pred_path="/data/pred_dict.pkl"
#     )
