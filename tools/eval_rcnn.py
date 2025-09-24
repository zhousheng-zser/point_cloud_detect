import _init_path
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from lib.net.point_rcnn import PointRCNN
from lib.datasets.kitti_rcnn_dataset import KittiRCNNDataset, compute_3d_box_cam2
import tools.train_utils.train_utils as train_utils
from lib.utils.bbox_transform import decode_bbox_target
from tools.kitti_object_eval_python.evaluate import evaluate as kitti_evaluate
from tools.draw_kitti_util import corner_to_surfaces_3d_jit,points_in_convex_polygon_3d_jit

from lib.config import cfg, cfg_from_file, save_config_to_file, cfg_from_list
import argparse
import lib.utils.kitti_utils as kitti_utils
import lib.utils.iou3d.iou3d_utils as iou3d_utils
from datetime import datetime
import logging
import re
import glob
import time
from tensorboardX import SummaryWriter
import tqdm

np.random.seed(1024)  # set the same seed

def parse_args():
    parser = argparse.ArgumentParser(description="arg parser")
    parser.add_argument('--cfg_file', type=str, default='cfgs/default.yaml', help='specify the config for evaluation')
    parser.add_argument("--eval_mode", type=str, default='rcnn', help="specify the evaluation mode")

    parser.add_argument('--test', action='store_true', default=False, help='evaluate without ground truth')
    parser.add_argument("--ckpt", type=str, default='PointRCNN.pth', help="specify a checkpoint to be evaluated")
    parser.add_argument("--rpn_ckpt", type=str, default=None, help="specify the checkpoint of rpn if trained separated")
    parser.add_argument("--rcnn_ckpt", type=str, default=None, help="specify the checkpoint of rcnn if trained separated")

    parser.add_argument('--batch_size', type=int, default=1, help='batch size for evaluation')
    parser.add_argument('--workers', type=int, default=4, help='number of workers for dataloader')
    parser.add_argument("--extra_tag", type=str, default='default', help="extra tag for multiple evaluation")
    parser.add_argument('--output_dir', type=str, default=None, help='specify an output directory if needed')
    parser.add_argument("--ckpt_dir", type=str, default=None, help="specify a ckpt directory to be evaluated if needed")

    parser.add_argument('--save_result', action='store_true', default=False, help='save evaluation results to files')
    parser.add_argument('--save_rpn_feature', action='store_true', default=False,
                        help='save features for separately rcnn training and evaluation')

    parser.add_argument('--random_select', action='store_true', default=True, help='sample to the same number of points')
    parser.add_argument('--start_epoch', default=0, type=int, help='ignore the checkpoint smaller than this epoch')
    parser.add_argument("--rcnn_eval_roi_dir", type=str, default=None,
                        help='specify the saved rois for rcnn evaluation when using rcnn_offline mode')
    parser.add_argument("--rcnn_eval_feature_dir", type=str, default=None,
                        help='specify the saved features for rcnn evaluation when using rcnn_offline mode')
    parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER,
                        help='set extra config keys if needed')
    args = parser.parse_args()
    return args

def create_logger(log_file):
    log_format = '%(asctime)s  %(levelname)5s  %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_format, filename=log_file)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter(log_format))
    logging.getLogger(__name__).addHandler(console)
    return logging.getLogger(__name__)

def rot_y_matrix(angle):
    """标准的绕 Y 轴旋转矩阵 R_y(angle)（右手规则）"""
    c = np.cos(angle)
    s = np.sin(angle)
    return np.array([
        [ c, 0,  s],
        [ 0, 1,  0],
        [-s, 0,  c]
    ])


def compute_min_box_with_rot(points_in_box, rot_y):
    a, b, c, d = cfg.TEST.PLANES[0], cfg.TEST.PLANES[1], cfg.TEST.PLANES[2], cfg.TEST.PLANES[3]
    y_ = -(a * points_in_box[0][0] + c * points_in_box[0][2] + d) / b
    pts = points_in_box[:, :3].astype(np.float64)
    if pts.shape[0] == 0:
        return None

    # Step1: 点云转局部坐标系 (对齐 rot_y)
    R_world2local = rot_y_matrix(-rot_y)
    pts_local = (R_world2local @ pts.T).T

    x_min, y_min, z_min = pts_local.min(axis=0)
    x_max, y_max, z_max = pts_local.max(axis=0)
    y_max = y_ 

    length = float(x_max - x_min)
    height = float(y_max - y_min)
    width  = float(z_max - z_min)

    # 中心的 x,z 用 local 中心再变换回 global（几何中心）
    cx_local = (x_min + x_max) / 2.0
    cy_local = (y_min + y_max) / 2.0
    cz_local = (z_min + z_max) / 2.0

    # Step6: location (底部中心，KITTI 语义)
    center_global = (R_world2local.T @ np.array([cx_local, cy_local, cz_local]))
    kitti_loc = center_global.copy()
    kitti_loc[1] += height / 2.0
    return height, width, length, kitti_loc[0], kitti_loc[1], kitti_loc[2], rot_y


def Calib_3d_box(pts_unique, height, width, length, pos_x, pos_y, pos_z, rot_y):
    corners_3d_cam2 = compute_3d_box_cam2(height, width, length, pos_x, pos_y, pos_z, rot_y)
    corners_3d_velo = corners_3d_cam2.T
    box_surfaces = corner_to_surfaces_3d_jit(corners_3d_velo[np.newaxis, ...])
    points_mask = points_in_convex_polygon_3d_jit(pts_unique[:, :3], box_surfaces)
    mask = points_mask.reshape([-1])
    points_in_box = pts_unique[mask]  #筛选点云中属于ROI框的点
    return compute_min_box_with_rot(points_in_box,rot_y)


def get_in_box(pts_lidar, height, width, length, pos_x, pos_y, pos_z, rot_y):
    corners_3d_cam2 = compute_3d_box_cam2(height, width, length, pos_x, pos_y, pos_z, rot_y)
    corners_3d_velo = corners_3d_cam2.T
    box_surfaces = corner_to_surfaces_3d_jit(corners_3d_velo[np.newaxis, ...])
    points_mask = points_in_convex_polygon_3d_jit(pts_lidar[:, :3], box_surfaces)
    mask = points_mask.reshape([-1])
    return mask 

def save_kitti_format(points, pts_unique, bbox3d, scores, cfg_classes="Car"):
    result_lines = []
    road_list =[]
    in_box_list=[]
    for k in range(bbox3d.shape[0]):
        score = float(scores[k])
        if score < 3 :
            continue
        x, z, ry = float(bbox3d[k, 0]), float(bbox3d[k, 2]), float(bbox3d[k, 6])
        #beta = np.arctan2(z, x)
        #alpha = -np.sign(beta) * np.pi / 2 + beta + ry
        
        h, w, l = float(bbox3d[k, 3]), float(bbox3d[k, 4]), float(bbox3d[k, 5])
        x, y, z = float(bbox3d[k, 0]), float(bbox3d[k, 1]), float(bbox3d[k, 2])
        
        truncated = -1    
        occluded = -1     
        bbox_left = 0.0   
        bbox_top = 0.0    
        bbox_right = 0.0  
        bbox_bottom = 0.0 
        # car_x = x - w/2
        # car_y = y - h/2
        # car_z = z - l/2
        # car = (car_x, car_y, car_z, w, h, l)
        #line = f"{cfg_classes} {truncated} {occluded} {alpha:.4f} {bbox_left:.4f} {bbox_top:.4f} {bbox_right:.4f} {bbox_bottom:.4f} {h:.4f} {w:.4f} {l:.4f} {x:.4f} {y:.4f} {z:.4f} {ry:.4f} {score:.4f}"
        #result_lines.append(line)
        
        #获取最小外接贴地长方体
        h, w, l, x, y, z, ry = Calib_3d_box(pts_unique, h, w, l, x, y, z, ry )
        #获取在长方体内的点云
        in_box_list.append(get_in_box(points, h, w, l, x, y, z, ry))
        #生成新的KITTI数据
        beta = np.arctan2(z, x)
        alpha = -np.sign(beta) * np.pi / 2 + beta + ry
        line_new = f"{cfg_classes} {truncated} {occluded} {alpha:.4f} {bbox_left:.4f} {bbox_top:.4f} {bbox_right:.4f} {bbox_bottom:.4f} {h:.4f} {w:.4f} {l:.4f} {x:.4f} {y:.4f} {z:.4f} {ry:.4f} {score:.4f}"
        result_lines.append(line_new)
        road_id = -1
        for i, roi in enumerate(cfg.TEST.ROADS_ROI):
            left = roi[3] - roi[1] / 2
            right = roi[3] + roi[1] / 2
            if left <= x < right: 
                road_id = i
                break
        
        road_list.append(road_id)

    #标记每个长发体框内的点云   
    result_mask  = np.zeros(len(points), dtype=bool)
    for mask in in_box_list:
        result_mask = np.logical_or(result_mask, mask)

    return  result_lines,result_mask,road_list

def eval_one_epoch_joint(points, model, dataloader, epoch_id, logger, test_mode=False):
    result_lines=[]
    road_list =[]
    if points is None or len(points) == 0 or points.size == 0:
        logger.info('---- Warning points is NULL ----------------')
        return result_lines
    np.random.seed(666)
    MEAN_SIZE = torch.from_numpy(cfg.CLS_MEAN_SIZE[0]).cuda()
    mode = 'TEST' if test_mode else 'EVAL'
    model.eval()

    thresh_list = [0.1, 0.3, 0.5, 0.7, 0.9]
    total_recalled_bbox_list, total_gt_bbox = [0] * 5, 0
    total_roi_recalled_bbox_list = [0] * 5
    dataset = dataloader.dataset
    final_total = total_cls_acc = total_cls_acc_refined = total_rpn_iou = 0
    data = dataset.get_rpn_sample(points)
    #logger.info('------------ BEGIN  ------------' )
    pts_input = data['pts_input']
    # ############求地
    # import open3d as o3d
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(pts_input)
    # plane_model, inliers = pcd.segment_plane(
    #     distance_threshold=0.05,  # 内点距离阈值（单位：米）
    #     ransac_n=3,  # 每次采样3个点
    #     num_iterations=100000  # 迭代次数
    # )
    # a, b, c, d = plane_model
    # if b > 0:
    #     plane_model =  -plane_model
    #     a, b, c, d = plane_model
    # print("a, b, c, d : ", a, b, c, d)

    #batch_size = len(sample_id)
    batch_size = 1    
    inputs = torch.from_numpy(pts_input).cuda(non_blocking=True).float()
    inputs = inputs.unsqueeze(0)  #  [1, N, C]
    input_data = {'pts_input': inputs}

    # model inference
    ret_dict = model(input_data)
    pts_unique = [list(pt) for pt in set(map(tuple, pts_input))]  #车道点云去重 
    #去地面
    a, b, c, d = cfg.TEST.PLANES[0], cfg.TEST.PLANES[1], cfg.TEST.PLANES[2], cfg.TEST.PLANES[3]
    
    denom = np.sqrt(a*a + b*b + c*c)
    distances = np.abs(pts_input @ np.array([a, b, c]) + d) / denom
    # 假设距离阈值 0.2m
    mask = distances > 0.03
    pts_unique = pts_input[mask]

    roi_scores_raw = ret_dict['roi_scores_raw']  # (B, M)
    roi_boxes3d = ret_dict['rois']  # (B, M, 7)
    seg_result = ret_dict['seg_result'].long()  # (B, N)

    rcnn_cls = ret_dict['rcnn_cls'].view(batch_size, -1, ret_dict['rcnn_cls'].shape[1])
    rcnn_reg = ret_dict['rcnn_reg'].view(batch_size, -1, ret_dict['rcnn_reg'].shape[1])  # (B, M, C)

    # bounding box regression
    anchor_size = MEAN_SIZE
    if cfg.RCNN.SIZE_RES_ON_ROI:
        assert False

    pred_boxes3d = decode_bbox_target(roi_boxes3d.view(-1, 7), rcnn_reg.view(-1, rcnn_reg.shape[-1]),
                                      anchor_size=anchor_size,
                                      loc_scope=cfg.RCNN.LOC_SCOPE,
                                      loc_bin_size=cfg.RCNN.LOC_BIN_SIZE,
                                      num_head_bin=cfg.RCNN.NUM_HEAD_BIN,
                                      get_xz_fine=True, get_y_by_bin=cfg.RCNN.LOC_Y_BY_BIN,
                                      loc_y_scope=cfg.RCNN.LOC_Y_SCOPE, loc_y_bin_size=cfg.RCNN.LOC_Y_BIN_SIZE,
                                      get_ry_fine=True).view(batch_size, -1, 7)

    # scoring
    if rcnn_cls.shape[2] == 1:
        raw_scores = rcnn_cls  # (B, M, 1)

        norm_scores = torch.sigmoid(raw_scores)
        pred_classes = (norm_scores > cfg.RCNN.SCORE_THRESH).long()
    else:
        pred_classes = torch.argmax(rcnn_cls, dim=1).view(-1)
        cls_norm_scores = F.softmax(rcnn_cls, dim=1)
        raw_scores = rcnn_cls[:, pred_classes]
        norm_scores = cls_norm_scores[:, pred_classes]

    # evaluation
    recalled_num = gt_num = rpn_iou = 0
    disp_dict = {'mode': mode, 'recall': '%d/%d' % (total_recalled_bbox_list[3], total_gt_bbox)}

    # scores thresh
    inds = norm_scores > cfg.RCNN.SCORE_THRESH
    for k in range(batch_size):
        cur_inds = inds[k].view(-1)
        if cur_inds.sum() == 0:
            continue

        pred_boxes3d_selected = pred_boxes3d[k, cur_inds]
        raw_scores_selected = raw_scores[k, cur_inds]
        norm_scores_selected = norm_scores[k, cur_inds]

        # NMS thresh
        # rotated nms
        boxes_bev_selected = kitti_utils.boxes3d_to_bev_torch(pred_boxes3d_selected)
        keep_idx = iou3d_utils.nms_gpu(boxes_bev_selected, raw_scores_selected, cfg.RCNN.NMS_THRESH).view(-1)
        pred_boxes3d_selected = pred_boxes3d_selected[keep_idx]
        scores_selected = raw_scores_selected[keep_idx]
        pred_boxes3d_selected, scores_selected = pred_boxes3d_selected.cpu().numpy(), scores_selected.cpu().numpy()

        #cur_sample_id = sample_id[k]
        calib = dataset.get_calib()
        final_total += pred_boxes3d_selected.shape[0] ##  can del
        #image_shape = dataset.get_image_shape(cur_sample_id)
        result_lines ,result_mask, road_list= save_kitti_format(calib.project_velo_to_ref(points[:, 0:3]), pts_unique, pred_boxes3d_selected, scores_selected)
        #长发体框外的点云全为-3   
        points[~result_mask, 3] = -3

    return points,cfg.TEST.ROADS_ROI, result_lines, road_list


def load_part_ckpt(model, filename, logger, total_keys=-1):
    if os.path.isfile(filename):
        logger.info("==> Loading part model from checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename)
        model_state = checkpoint['model_state']

        update_model_state = {key: val for key, val in model_state.items() if key in model.state_dict()}
        state_dict = model.state_dict()
        state_dict.update(update_model_state)
        model.load_state_dict(state_dict)

        update_keys = update_model_state.keys().__len__()
        if update_keys == 0:
            raise RuntimeError
        logger.info("==> Done (loaded %d/%d)" % (update_keys, total_keys))
    else:
        raise FileNotFoundError


def load_ckpt_based_on_config(model, logger, ckpt=None, rpn_ckpt=None, rcnn_ckpt=None):
    if ckpt is not None:
        train_utils.load_checkpoint(model, filename=ckpt, logger=logger)

    total_keys = model.state_dict().keys().__len__()
    if cfg.RPN.ENABLED and rpn_ckpt is not None:
        load_part_ckpt(model, filename=rpn_ckpt, logger=logger, total_keys=total_keys)

    if cfg.RCNN.ENABLED and rcnn_ckpt is not None:
        load_part_ckpt(model, filename=rcnn_ckpt, logger=logger, total_keys=total_keys)


def create_dataloader(data_path, batch_size=1, workers=4, logger=None, test_mode=False, random_select=True,
                     rcnn_eval_roi_dir=None, rcnn_eval_feature_dir=None):
    mode = 'TEST' if test_mode else 'EVAL'

    # create dataloader
    logger.info('---- root_dir: %s ' % data_path)
    logger.info('---- npoints: %s ' % cfg.RPN.NUM_POINTS)
    logger.info('---- split: %s ' % cfg.TEST.SPLIT)
    logger.info('---- mode: %s ' % mode)
    logger.info('---- random_select: %s ' % random_select)
    logger.info('---- rcnn_eval_roi_dir: %s ' % rcnn_eval_roi_dir)
    logger.info('---- rcnn_eval_feature_dir: %s ' % rcnn_eval_feature_dir)
    logger.info('---- classes: %s ' % cfg.CLASSES)
    logger.info('---- Warning_roi: %s ' % cfg.TEST.WARNING_ROI)
    logger.info('---- Roads_roi: %s ' % cfg.TEST.ROADS_ROI)
    logger.info('---- Planes: %s ' % cfg.TEST.PLANES)
 
    test_set = KittiRCNNDataset(root_dir=data_path, npoints=cfg.RPN.NUM_POINTS, split=cfg.TEST.SPLIT, mode=mode,
                                random_select=random_select,
                                rcnn_eval_roi_dir=rcnn_eval_roi_dir,
                                rcnn_eval_feature_dir=rcnn_eval_feature_dir,
                                classes=cfg.CLASSES,
                                logger=logger)

    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, pin_memory=True,
                             num_workers=workers, collate_fn=test_set.collate_batch)
    return test_loader


def setup_config(cfg_file=None, set_cfgs=None):
    if cfg_file is not None:
        cfg_from_file(cfg_file)
    if set_cfgs is not None:
        cfg_from_list(set_cfgs)
    
    # Set default config for RCNN mode
    cfg.RCNN.ENABLED = True
    cfg.RPN.ENABLED = cfg.RPN.FIXED = True
    cfg.TAG = os.path.splitext(os.path.basename(cfg_file))[0] if cfg_file else 'default'

class PointCloudConverter:
    def __init__(self, cfg_file='cfgs/default.yaml', set_cfgs=None, ckpt='PointRCNN.pth', rpn_ckpt=None, rcnn_ckpt=None, 
                            output_dir=None, test_mode=False, batch_size=1, workers=4, 
                            data_path='./cfgs', extra_tag='default', random_select=True,
                            rcnn_eval_roi_dir=None, rcnn_eval_feature_dir=None):

        # Setup config
        setup_config(cfg_file, set_cfgs)
        
        # Setup output directory
        if output_dir is None:
            root_result_dir = os.path.join('../', 'output', 'rcnn', cfg.TAG)
        else:
            root_result_dir = output_dir
        
        # Set epoch_id based on checkpoint
        if ckpt is not None:
            num_list = re.findall(r'\d+', ckpt)
            self.epoch_id = num_list[-1] if num_list.__len__() > 0 else 'no_number'
        else:
            self.epoch_id = 'no_number'
        self.test_mode = test_mode
        # Setup logger
        log_file = os.path.join('log_eval.txt')
        self.logger = create_logger(log_file)
        self.logger.info('**********************Start logging**********************')
        
        # Log config
        save_config_to_file(cfg, logger=self.logger)
        
        # Create dataloader
        self.test_loader = create_dataloader(
            data_path=data_path,
            batch_size=batch_size,
            workers=workers,
            logger=self.logger,
            test_mode=self.test_mode,
            random_select=random_select,
            rcnn_eval_roi_dir=rcnn_eval_roi_dir,
            rcnn_eval_feature_dir=rcnn_eval_feature_dir
        )
        # self.test_loader.dataset.num_class -> Car -> 2
        # Create model
        self.model = PointRCNN(num_classes=2, use_xyz=True, mode='TEST')
        self.model.cuda()
        
        # Load checkpoint
        load_ckpt_based_on_config(self.model, self.logger, ckpt, rpn_ckpt, rcnn_ckpt)
        
        # Start evaluation
        #    eval_results = eval_one_epoch(model, test_loader, epoch_id, result_dir, logger, test_mode)

    def eval_one_epoch(self,points):
        if points.shape[0] < 10000:
            self.logger.info('**********************Warning: points<10000 **********************')
            return "", "","",""
        with torch.no_grad():
            #results = eval_one_epoch_joint(points_test, self.model, self.test_loader, self.epoch_id, self.logger, self.test_mode)
            return eval_one_epoch_joint(points, self.model, self.test_loader, self.epoch_id, self.logger, self.test_mode)


# def evaluate_point_rcnn(cfg_file='cfgs/default.yaml', set_cfgs=None, ckpt='PointRCNN.pth', rpn_ckpt=None......
def main():
    args = parse_args()
    
    # Use our API function
    with torch.no_grad():
        print(args)
        results = PointCloudConverter(
            cfg_file=args.cfg_file,
            set_cfgs=args.set_cfgs,
            ckpt=args.ckpt,
            rpn_ckpt=args.rpn_ckpt,
            rcnn_ckpt=args.rcnn_ckpt,
            output_dir=args.output_dir,
            test_mode=args.test,
            batch_size=args.batch_size,
            workers=args.workers,
            extra_tag=args.extra_tag,
            random_select=args.random_select,
            rcnn_eval_roi_dir=args.rcnn_eval_roi_dir,
            rcnn_eval_feature_dir=args.rcnn_eval_feature_dir
        )
        results.eval_one_epoch()


if __name__ == "__main__":
    main()