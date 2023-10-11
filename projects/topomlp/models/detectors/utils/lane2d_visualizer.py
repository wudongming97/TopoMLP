# Lane2DVisualizer 简易使用流程
# step1: 实例化 Lane2DVisualizer
# step2: 调用 add_frame 方法添加可视化信息到某一帧
# step3: 调用 show 方法，将生成的 mp4 文件上传到 s3 用于可视化

import os
import cv2
import ffmpeg
import numpy as np
import boto3
from loguru import logger

# TODO:
# 1.. add 3D visualization
# 2. add colorful visualization


class Lane2DVisualizer:
    def __init__(
        self,
        save_path,
        vis_fov_img_shape=(760, 540),
        vis_bev_img_shape=(470, 540),
        vis_bev_range=(50.0, 120),
        fps=10,
        **kwargs,
    ):
        """
        An MP4-based lane visualizer.

        Args:
            save_path(str): path to save mp4 file
            vis_fov_img_shape(tuple): shape of fov image, (width, height)
            vis_bev_img_shape(tuple): shape of bev image, (width, height)
            vis_bev_range(tuple): range of bev image, default(30M, 120M)
            fps(int): frames per second for mp4 file
        """
        self.vis_fov_img_shape = vis_fov_img_shape
        self.vis_bev_img_shape = vis_bev_img_shape
        self.vis_bev_range = vis_bev_range
        self.fps = fps
        self.vis_img_shape = (self.vis_fov_img_shape[0] + self.vis_bev_img_shape[0], self.vis_fov_img_shape[1])
        self.s3_client = boto3.client("s3", endpoint_url="http://oss.i.brainpp.cn")

        if not save_path.endswith(".mp4"):
            save_path = "{}{}".format(save_path, ".mp4")
        if save_path.startswith("s3"):
            self.s3_path = save_path
            self.s3_temp_filename = "/tmp/s3_tempfile_{}.mp4".format(os.getpid())
            self.writer = cv2.VideoWriter(
                self.s3_temp_filename, cv2.VideoWriter_fourcc(*"mp4v"), self.fps, self.vis_img_shape
            )
        else:
            self.s3_path = None
            self.writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"mp4v"), self.fps, self.vis_img_shape)

    @property
    def bev_img(self):
        if not hasattr(self, "_bev_img"):
            self._bev_img = self._configure_bev_img()
        return self._bev_img

    def _configure_bev_img(self):
        bev_img = np.ones((self.vis_bev_img_shape[1], self.vis_bev_img_shape[0], 3)).astype("uint8")
        nr_y_bins = int(self.vis_bev_range[1] / 10.0) + 1  # 10.0m
        idx = np.linspace(0, self.vis_bev_img_shape[1], nr_y_bins)
        for i in range(nr_y_bins):
            bev_img = cv2.line(bev_img, (0, int(idx[i])), (self.vis_bev_img_shape[0], int(idx[i])), (128, 128, 0), 1)
            bev_img = cv2.putText(
                bev_img,
                "{}m".format(10 * i),
                (self.vis_bev_img_shape[0] - 40, int(idx[nr_y_bins - i - 1])),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
        return bev_img

    def add_frame(
        self,
        img,
        lane_2d_points=None,
        pred_2d_points=None,
        bev_img=None,
        lane_3d_points=None,
        pred_3d_points=None,
        img_metas=None,
        img_file=None,
    ):
        """
        Args:
            img(np.ndarray): image using BGR(CV2) color space
            lane_2d_points(np.ndarray): list of 2D points, each item is a vector of (x, y)
            pred_2d_points(np.ndarray): list of predicted 2D points, each item is a vector of (x, y)
            bev_img(np.ndarray): bev image, default is None
            lane_3d_points(np.ndarray): list of 3D points, each item is a vector of (x, y, z)
            pred_3d_points(np.ndarray): list of predicted 3D points, each item is a vector of (x, y, z)
        """
        # draw lane of fov image
        for _2d_points, color in [(lane_2d_points, (0, 0, 255)), (pred_2d_points, (0, 255, 0))]:
            if _2d_points is not None:
                for pts in _2d_points:
                    img = cv2.polylines(
                        img, np.int32([pts]), color=color, isClosed=False, thickness=2, lineType=cv2.LINE_AA
                    )

        img = cv2.resize(img, self.vis_fov_img_shape)

        # draw lane of bev image
        is_norm_bev_coord = False
        if bev_img is None:
            bev_img = self.bev_img.copy()
            is_norm_bev_coord = True

        for _3d_points, color in [(lane_3d_points, (0, 0, 255)), (pred_3d_points, (0, 255, 0))]:
            if _3d_points is not None:
                for pts in _3d_points:
                    if is_norm_bev_coord:
                        pts[:, 0] = (1.0 - pts[:, 0] / self.vis_bev_range[1]) * self.vis_bev_img_shape[1]
                        pts[:, 1] = (0.5 - pts[:, 1] / self.vis_bev_range[0]) * self.vis_bev_img_shape[0]

                    bev_img = cv2.polylines(
                        bev_img,
                        np.int32([pts[:, 1::-1]]),
                        color=color,
                        isClosed=False,
                        thickness=1,
                        lineType=cv2.LINE_AA,
                    )

        bev_img = cv2.resize(bev_img, self.vis_bev_img_shape)
        img = np.concatenate([img, bev_img], axis=1)
        cv2.imwrite('./'+img_file+'/'+str(img_metas[0]['index'])+'_res.jpg',img)
        return True

    def show(self):
        """
        Close video writer, x264 encode, and upload to OSS
        """
        self.writer.release()
        if self.s3_path is not None:
            x264_temp_filename = "/tmp/s3_tempfile_{}_x264.mp4".format(os.getpid())
            try:
                _ = (
                    # ffmpeg.input(self.s3_tempfile.name)
                    ffmpeg.input(self.s3_temp_filename)
                    .filter("fps", fps=self.fps, round="up")
                    .output(x264_temp_filename, vcodec="h264")
                    .overwrite_output()
                    .run(capture_stdout=True, capture_stderr=True)
                )
            except ffmpeg.Error as e:
                print("stdout:", e.stdout.decode("utf8"))
                print("stderr:", e.stderr.decode("utf8"))
                raise e

            bucket_name = self.s3_path.split("/")[2]
            key = "/".join(self.s3_path.split("/")[3:])
            _ = self.s3_client.upload_file(x264_temp_filename, bucket_name, key)
            logger.info("open_url:")
            show_url = f"https://oss.iap.hh-b.brainpp.cn/{self.s3_path[5:]}"
            logger.info(show_url)

            os.remove(self.s3_temp_filename)
            os.remove(x264_temp_filename)
            logger.info("Successfully upload video to OSS and remove the temp files")
