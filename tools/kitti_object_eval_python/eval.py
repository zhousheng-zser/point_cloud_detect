import io as sysio
import numba
import numpy as np

from .rotate_iou import rotate_iou_gpu_eval

# http://zhaoxuhui.top/blog/2019/01/17/PythonNumba.html
# https://zhuanlan.zhihu.com/p/60994299
# jit stands for Just-in-time, in numba it specifically refers to Just-in-time compilation
# There are two compilation modes for Numba's @jit: nopython and object mode
# nopython mode will fully compile the decorated function, its execution is completely independent of Python interpreter and won't call Python C API
# @njit decorator is equivalent to @jit(nopython=True)
# In object mode, the compiler will automatically identify parts like loop statements that can be compiled and accelerated, 
# while leaving the remaining parts to Python interpreter
# If nopython=True is not set, Numba will first try to use nopython mode, and fall back to object mode if not possible
# Adding nopython will force the compiler to use nopython mode, but may raise errors if the code contains types that cannot be automatically deduced
# Numba likes loops
# Numba likes NumPy functions
# Numba likes NumPy broadcasting
@numba.jit
def get_thresholds(scores: np.ndarray, num_gt, num_sample_pts=41):
    """
    Get the scores that must be evaluated for recall discretization
    Exit when r_recall-current_recall < current_recall-l_recall is no longer satisfied
    Before exiting, add 1/40 for each satisfied current_recall --> artificially set value
    Args:
        scores: Scores of all valid gt matching dt, number of tp
        num_gt: Total number of valid boxes
    Returns:
        thresholds: 41 threshold points
    """
    scores.sort()  # Sort scores in ascending order
    scores = scores[::-1]  # Reverse to descending order
    current_recall = 0
    thresholds = []
    for i, score in enumerate(scores):
        l_recall = (i + 1) / num_gt  # Calculate left recall
        if i < (len(scores) - 1):
            r_recall = (i + 2) / num_gt  # Calculate right recall
        else:
            r_recall = l_recall
        # Find 40 recall threshold points where current_recall is between l_recall and r_recall
        if (((r_recall - current_recall) < (current_recall - l_recall))
                and (i < (len(scores) - 1))):
            continue
        # recall = l_recall
        thresholds.append(score)
        current_recall += 1 / (num_sample_pts - 1.0)  # Increase by 0.25 each time
    return thresholds


def clean_data(gt_anno, dt_anno, current_class, difficulty):
    """
    Classify and label gt and dt boxes based on current_class and difficulty,
    and calculate number of valid boxes and dc_bboxes
    Args:
        gt_anno: Single frame point cloud annotation dict
        dt_anno: Single frame point cloud prediction dict
        current_class: Scalar 0
        difficulty: Scalar 0
    Returns:
        num_valid_gt: Number of valid gt
        ignored_gt: gt flag list 0=valid, 1=ignore, -1=other
        ignored_dt: dt flag list
        dc_bboxes: don't care boxes
    """
    CLASS_NAMES = ['car', 'pedestrian', 'cyclist', 'van', 'person_sitting', 'truck']  # Class names
    MIN_HEIGHT = [40, 25, 25]  # Minimum height threshold
    MAX_OCCLUSION = [0, 1, 2]  # Maximum occlusion threshold
    MAX_TRUNCATION = [0.15, 0.3, 0.5]  # Maximum truncation threshold
    dc_bboxes, ignored_gt, ignored_dt = [], [], []
    current_cls_name = CLASS_NAMES[current_class].lower()
    num_gt = len(gt_anno["name"])  # Number of gt, e.g. 7
    num_dt = len(dt_anno["name"])  # Number of dt, e.g. 13
    num_valid_gt = 0
    
    # 2. Iterate through all gt boxes
    for i in range(num_gt):
        # Get the i-th bbox, name and height info
        bbox = gt_anno["bbox"][i]
        gt_name = gt_anno["name"][i].lower()
        height = bbox[3] - bbox[1]
        valid_class = -1
        
        # 2.1 First, classify based on category: 1=valid, 0=ignore, -1=invalid
        if (gt_name == current_cls_name):
            valid_class = 1
        elif (current_cls_name == "Pedestrian".lower()
              and "Person_sitting".lower() == gt_name):
            valid_class = 0
        elif (current_cls_name == "Car".lower() and "Van".lower() == gt_name):
            valid_class = 0
        else:
            valid_class = -1
            
        ignore = False
        # 2.2 Then, determine if the box should be ignored based on occluded, truncated and height
        if ((gt_anno["occluded"][i] > MAX_OCCLUSION[difficulty])  # Severe occlusion
                or (gt_anno["truncated"][i] > MAX_TRUNCATION[difficulty])  # Severe truncation
                or (height <= MIN_HEIGHT[difficulty])):  # Too small height
            ignore = True
            
        # 2.3 Finally, classify based on valid_class and ignore status
        if valid_class == 1 and not ignore:
            ignored_gt.append(0)  # 0 means valid
            num_valid_gt += 1  # Increment valid box count
        elif (valid_class == 0 or (ignore and (valid_class == 1))):
            ignored_gt.append(1)  # 1 means ignore
        else:
            ignored_gt.append(-1)  # -1 means invalid
            
    # 2.4 If category is DontCare, add the box to dc_bboxes
    for i in range(num_gt):
        if gt_anno["name"][i] == "DontCare":
            dc_bboxes.append(gt_anno["bbox"][i])
    
    # 3. Iterate through dt boxes
    for i in range(num_dt):
        # 3.1 If detected box category matches current class, set valid flag to 1
        if (dt_anno["name"][i].lower() == current_cls_name):
            valid_class = 1
        else:
            valid_class = -1
            
        # 3.2 Calculate height
        height = abs(dt_anno["bbox"][i, 3] - dt_anno["bbox"][i, 1])
        # If detected height is less than minimum height, ignore the box
        if height < MIN_HEIGHT[difficulty]:
            ignored_dt.append(1)
        # 3.3 If valid, add 0
        elif valid_class == 1:
            ignored_dt.append(0)
        # Otherwise, add -1
        else:
            ignored_dt.append(-1)

    return num_valid_gt, ignored_gt, ignored_dt, dc_bboxes


@numba.jit(nopython=True)
def image_box_overlap(boxes, query_boxes, criterion=-1):
    """
    Calculate IoU of image boxes
    Args:
        boxes: All gt in a part, e.g. (642,4) for first part
        query_boxes: All dt in a part, e.g. (233,4) for first part
    """
    N = boxes.shape[0]  # Total number of gt_boxes
    K = query_boxes.shape[0]  # Total number of det_boxes
    # Initialize overlap matrix
    overlaps = np.zeros((N, K), dtype=boxes.dtype)
    
    # Two nested for loops to calculate IoU for each box, using jit acceleration
    for k in range(K):
        # Calculate area of k-th dt box (box is in [x1,y1,x2,y2] format)
        qbox_area = ((query_boxes[k, 2] - query_boxes[k, 0]) *
                     (query_boxes[k, 3] - query_boxes[k, 1]))
        for n in range(N):  # Iterate through gt boxes
            # Overlap width = min of right edges - max of left edges
            iw = (min(boxes[n, 2], query_boxes[k, 2]) -
                  max(boxes[n, 0], query_boxes[k, 0]))
            if iw > 0:  # If width overlap exists, calculate height
                # Overlap height = min of top edges - max of bottom edges
                ih = (min(boxes[n, 3], query_boxes[k, 3]) -
                      max(boxes[n, 1], query_boxes[k, 1]))
                if ih > 0:
                    if criterion == -1:  # Default execution with criterion = -1
                        # Calculate union of two boxes
                        ua = (
                            (boxes[n, 2] - boxes[n, 0]) *
                            (boxes[n, 3] - boxes[n, 1]) + qbox_area - iw * ih)
                    elif criterion == 0:
                        ua = ((boxes[n, 2] - boxes[n, 0]) *
                              (boxes[n, 3] - boxes[n, 1]))
                    elif criterion == 1:
                        ua = qbox_area
                    else:
                        ua = 1.0
                    
                    overlaps[n, k] = iw * ih / ua
    return overlaps


def bev_box_overlap(boxes, qboxes, criterion=-1):
    riou = rotate_iou_gpu_eval(boxes, qboxes, criterion)
    return riou

# Compile in non-python mode, compile native multithreading
# The compiler will compile a version that runs multiple native threads in parallel (no GIL)
@numba.jit(nopython=True, parallel=True)
def d3_box_overlap_kernel(boxes, qboxes, rinc, criterion=-1):
    """
    Calculate 3D IoU of boxes
    Args:
        boxes: All gt in a part, e.g. (642,7) [x,y,z,dx,dy,dz,alpha] for first part
        query_boxes: All dt in a part, e.g. (233,7) for first part
        rinc: IoU in bird's eye view (642,233)
    Returns:
        3D IoU --> rinc
    """
    # ONLY support overlap in CAMERA, not lidar.
    # Calculate in camera coordinate system: z forward, y down, x right
    # So bird's eye view takes x and z
    N, K = boxes.shape[0], qboxes.shape[0]
    
    # Iterate through gt
    for i in range(N):
        # Iterate through dt
        for j in range(K):
            # If bird's eye view overlap exists
            if rinc[i, j] > 0:
                # Here 1 is y-axis, which is height direction in camera coordinates
                # Overlap height = min of top edges - max of bottom edges
                iw = (min(boxes[i, 1], qboxes[j, 1]) - max(
                    boxes[i, 1] - boxes[i, 4], qboxes[j, 1] - qboxes[j, 4]))
                # If overlap height > 0
                if iw > 0:
                    # 1. Calculate volume of both boxes
                    area1 = boxes[i, 3] * boxes[i, 4] * boxes[i, 5]
                    area2 = qboxes[j, 3] * qboxes[j, 4] * qboxes[j, 5]
                    # 2. Calculate intersection volume
                    inc = iw * rinc[i, j]
                    # 3. Calculate union volume based on criterion
                    if criterion == -1:
                        ua = (area1 + area2 - inc)
                    elif criterion == 0:
                        ua = area1
                    elif criterion == 1:
                        ua = area2
                    else:
                        ua = inc
                    # 4. Calculate IoU
                    rinc[i, j] = inc / ua
                else:
                    rinc[i, j] = 0.0


def d3_box_overlap(boxes, qboxes, criterion=-1):
    """
    Calculate IoU of boxes in bird's eye view (with rotation angle)
    Args:
        boxes: All gt in a part, e.g. (642,7) [x,y,z,dx,dy,dz,alpha] for first part
        query_boxes: All dt in a part, e.g. (233,7) for first part
        rinc: IoU in bird's eye view (642,233)
    Returns:
        IoU --> rinc
    """
    rinc = rotate_iou_gpu_eval(boxes[:, [0, 2, 3, 5, 6]],
                               qboxes[:, [0, 2, 3, 5, 6]], 2)
    d3_box_overlap_kernel(boxes, qboxes, rinc, criterion)
    return rinc


@numba.jit(nopython=True)
def compute_statistics_jit(overlaps,
                           gt_datas,
                           dt_datas,
                           ignored_gt,
                           ignored_det,
                           dc_bboxes,
                           metric,
                           min_overlap,
                           thresh=0,
                           compute_fp=False,
                           compute_aos=False):
    """
    Calculate statistics per frame: tp, fp, fn, similarity, thresholds[:thresh_idx]
    Two modes exist based on compute_fp state
    Args:
        overlaps: IoU of single frame point cloud (N,M)
        gt_datas: (N,5) --> (x1, y1, x2, y2, alpha)
        dt_datas: (M,6) --> (x1, y1, x2, y2, alpha, score)
        ignored_gt: (N,) box status 0,1,-1
        ignored_det: (M,) box status 0,1,-1
        dc_bboxes: (k,4)
        metric: 0: bbox, 1: bev, 2: 3d
        min_overlap: Minimum IoU threshold
        thresh=0: Ignore dt with score below this, will pass 41 thresholds based on recall points
        compute_fp=False
        compute_aos=False
    Returns:
        tp: True positive - predicted true, actually true
        fp: False positive - predicted true, actually false
        fn: False negative - predicted false, actually true
        similarity: Cosine similarity
        thresholds[:thresh_idx]: Scores of dt matching valid gt
    precision = TP / (TP + FP) - proportion of TP among all predicted true
    recall = TP / (TP + FN) - proportion of TP among all actually true
    """
    # ============================ 1 Initialization ============================
    det_size = dt_datas.shape[0]  # Number of det boxes M
    gt_size = gt_datas.shape[0]  # Number of gt boxes N
    dt_scores = dt_datas[:, -1]  # dt box scores (M,)
    dt_alphas = dt_datas[:, 4]  # dt alpha scores (M,)
    gt_alphas = gt_datas[:, 4]  # gt alpha scores (N,)
    dt_bboxes = dt_datas[:, :4]  # (M,4)
    gt_bboxes = gt_datas[:, :4]  # (N,4)

    # Initialize for dt
    assigned_detection = [False] * det_size  # Track if dt matched gt
    ignored_threshold = [False] * det_size  # Mark True if dt score below threshold

    # If calculating fp: predicted true, actually false
    if compute_fp:
        # Iterate through dt
        for i in range(det_size):
            # If score below threshold
            if (dt_scores[i] < thresh):
                # Ignore this box
                ignored_threshold[i] = True

    # Initialize
    NO_DETECTION = -10000000
    tp, fp, fn, similarity = 0, 0, 0, 0
    thresholds = np.zeros((gt_size, ))  # (N,)
    thresh_idx = 0  # thresholds index, updated later
    delta = np.zeros((gt_size, ))  # (N,)
    delta_idx = 0  # delta index, updated later

    # ============================ 2 Match gt to dt, calculate tp and fn (gt-focused) ============================
    # Iterate through gt, matching to dt, skipping invalid gt
    for i in range(gt_size):
        if ignored_gt[i] == -1:  # Skip invalid gt
            continue
        det_idx = -1  # Store best matching dt index so far
        valid_detection = NO_DETECTION  # Mark if valid dt
        max_overlap = 0  # Store best overlap so far
        assigned_ignored_det = False  # Mark if matched to dt

        # Iterate through dt
        for j in range(det_size):
            if (ignored_det[j] == -1):  # Skip invalid dt
                continue
            if (assigned_detection[j]):  # Skip if already matched to gt
                continue 
            if (ignored_threshold[j]):  # Skip if dt score below threshold
                continue

            overlap = overlaps[j, i]  # Get overlap between current dt and this gt
            dt_score = dt_scores[j]  # Get current dt score

            if (not compute_fp  # compute_fp is false, no need to calculate FP
                    and (overlap > min_overlap)  # overlap > min threshold e.g. 0.7
                    and dt_score > valid_detection):  # Find highest scoring detection
                det_idx = j
                valid_detection = dt_score  # Update highest score found so far
            elif (compute_fp  # When compute_fp is true, select based on overlap
                  and (overlap > min_overlap)  # overlap > min threshold
                  and (overlap > max_overlap or assigned_ignored_det)  # If current overlap > previous max or gt already matched
                  and ignored_det[j] == 0):  # dt is valid
                max_overlap = overlap  # Update best overlap
                det_idx = j  # Update best matching dt id
                valid_detection = 1  # Mark valid dt
                assigned_ignored_det = False  # Use holdout method to indicate assigned unit
            elif (compute_fp  # compute_fp is true
                  and (overlap > min_overlap)  # If overlap sufficient
                  and (valid_detation == NO_DETECTION)  # Nothing assigned yet
                  and ignored_det[j] == 1):  # dt is ignored
                det_idx = j  # Update best matching dt id
                valid_detection = 1  # Mark valid dt
                assigned_ignored_det = True  # Mark gt matched to dt

        # If valid gt found no match, increment fn (true label not matched)
        if (valid_detection == NO_DETECTION) and ignored_gt[i] == 0:
            fn += 1
        # If gt found match and gt flag is ignore or dt flag is ignore, mark assigned_detection as True
        elif ((valid_detection != NO_DETECTION)
              and (ignored_gt[i] == 1 or ignored_det[det_idx] == 1)):
            assigned_detection[det_idx] = True
        # Otherwise valid gt found match
        elif valid_detection != NO_DETECTION:
            tp += 1
            thresholds[thresh_idx] = dt_scores[det_idx]  # Assign dt score to thresholds
            thresh_idx += 1
            if compute_aos:
                delta[delta_idx] = gt_alphas[i] - dt_alphas[det_idx]  # Difference between two alphas
                delta_idx += 1

            assigned_detection[det_idx] = True
    
    # ============================ 3 Calculate fp (dt-focused) ============================
    # After iterating through all gt and dt, if compute_fp is true, calculate fp
    if compute_fp:
        # Iterate through dt
        for i in range(det_size):
            # If all four conditions are false, increment fp
            # assigned_detection[i] == 0 --> dt not assigned to gt
            # ignored_det[i] != -1 and ignored_det[i] != 1 --> ignored_det[i] == 0 (valid dt)
            # ignored_threshold[i] == false, cannot ignore this dt box
            if (not (assigned_detection[i] or ignored_det[i] == -1
                     or ignored_det[i] == 1 or ignored_threshold[i])):
                fp += 1  # Predicted true, actually false
        
        nstuff = 0  # Number of don't care
        if metric == 0:  # If calculating bbox
            # Calculate IoU between dt and dc, with criterion=0, union is dt area
            overlaps_dt_dc = image_box_overlap(dt_bboxes, dc_bboxes, 0)
            for i in range(dc_bboxes.shape[0]):
                for j in range(det_size):
                    # Skip content not added to fp above
                    if (assigned_detection[j]):
                        continue
                    if (ignored_det[j] == -1 or ignored_det[j] == 1):
                        continue
                    if (ignored_threshold[j]):
                        continue
                    # If overlap between them > min_overlap
                    if overlaps_dt_dc[j, i] > min_overlap:
                        assigned_detection[j] = True  # Assign detection to dt
                        nstuff += 1 
        fp -= nstuff  # Remove don't care from previously calculated fp (don't care not counted)

        if compute_aos:
            # fp+tp (valid gt found match) = total predicted true boxes at this recall threshold
            tmp = np.zeros((fp + delta_idx, ))
            # Similarity calculation formula
            for i in range(delta_idx):
                tmp[i + fp] = (1.0 + np.cos(delta[i])) / 2.0
            if tp > 0 or fp > 0:
                similarity = np.sum(tmp)  # Directly sum counts, division done in eval_class
            else:
                similarity = -1
    return tp, fp, fn, similarity, thresholds[:thresh_idx]


def get_split_parts(num, num_part):  # num:3769 num_part:100
    """
    Return list of numbers dividing num into num_part parts, remainder at end
    """
    same_part = num // num_part  # 37
    remain_num = num % num_part  # 69
    if same_part == 0:
        return [num]

    if remain_num == 0:
        return [same_part] * num_part
    else:
        return [same_part] * num_part + [remain_num]  # [37]*100 + [69] --> [37 37 ... 69]


@numba.jit(nopython=True)
def fused_compute_statistics(overlaps,
                             pr,
                             gt_nums,
                             dt_nums,
                             dc_nums,
                             gt_datas,
                             dt_datas,
                             dontcares,
                             ignored_gts,
                             ignored_dets,
                             metric,
                             min_overlap,
                             thresholds,
                             compute_aos=False):
    """
    Calculate pr for a part
    Args:
        overlaps: IoU of a part (M,N) --> (642,233)
        pr: (41,4) --> tp, fp, fn, similarity
        gt_nums: Number of gt in a part (37,)
        dt_nums: Number of dt in a part (37,)
        dc_nums: Number of dc in a part (37,)
        gt_datas: gt data in a part (233,5)
        dt_datas: dt data in a part (642,5)
        dontcares: gt data in a part (79,4)
        ignored_gts: (233,)
        ignored_dets: (642,)
        metric: 0
        min_overlap: 0.7
        thresholds: (41,)
        compute_aos=False: True
    Return:
        No return value as pr is passed in as parameter, results are in parameters
    """
    gt_num = 0
    dt_num = 0
    dc_num = 0
    # Iterate through part point clouds, accumulate pr and box counts frame by frame
    for i in range(gt_nums.shape[0]):
        # Iterate through thresholds
        for t, thresh in enumerate(thresholds):
            # Extract IoU matrix for this frame
            overlap = overlaps[dt_num:dt_num + dt_nums[i], gt_num:
                               gt_num + gt_nums[i]]  # (13,7)
            # Get this frame's data
            gt_data = gt_datas[gt_num:gt_num + gt_nums[i]]  # (7,5)
            dt_data = dt_datas[dt_num:dt_num + dt_nums[i]]  # (13,6)
            ignored_gt = ignored_gts[gt_num:gt_num + gt_nums[i]]  # (7,)
            ignored_det = ignored_dets[dt_num:dt_num + dt_nums[i]]  # (13,)
            dontcare = dontcares[dc_num:dc_num + dc_nums[i]]  # (4,4)
            # Actually calculate metrics
            tp, fp, fn, similarity, _ = compute_statistics_jit(
                overlap,
                gt_data,
                dt_data,
                ignored_gt,
                ignored_det,
                dontcare,
                metric,
                min_overlap=min_overlap,
                thresh=thresh,
                compute_fp=True,
                compute_aos=compute_aos)
            # Accumulate metrics
            pr[t, 0] += tp
            pr[t, 1] += fp
            pr[t, 2] += fn
            if similarity != -1:
                pr[t, 3] += similarity
        # Accumulate box counts
        gt_num += gt_nums[i]
        dt_num += dt_nums[i]
        dc_num += dc_nums[i]


def calculate_iou_partly(gt_annos, dt_annos, metric, num_parts=50):
    """Fast IoU algorithm. Can be used independently for result analysis. 
    Must be used in CAMERA coordinate system.
    Args:
        gt_annos: dict from get_label_annos() in kitti_common.py
        dt_annos: dict from get_label_annos() in kitti_common.py
        metric: eval type 0: bbox, 1: bev, 2: 3d
        num_parts: int parameter for fast calculation
    """
    assert len(gt_annos) == len(dt_annos)
    # 1. Calculate box count per frame
    total_dt_num = np.stack([len(a["name"]) for a in dt_annos], 0)  # (3769,) [7,2,7...]
    total_gt_num = np.stack([len(a["name"]) for a in gt_annos], 0)  # (3769,) [13,13,11...]
    num_examples = len(gt_annos)  # 3769
    split_parts = get_split_parts(num_examples, num_parts)
    parted_overlaps = []
    example_idx = 0
    
    # 2. Calculate metrics by part
    for num_part in split_parts:
        gt_annos_part = gt_annos[example_idx:example_idx + num_part]  # Get 37 frame annotations
        dt_annos_part = dt_annos[example_idx:example_idx + num_part]
        # Calculate different metrics based on metric index
        if metric == 0:
            gt_boxes = np.concatenate([a["bbox"] for a in gt_annos_part], 0)  # Concatenate box info, e.g. (642,4) for first part
            dt_boxes = np.concatenate([a["bbox"] for a in dt_annos_part], 0)  # (233,4)
            overlap_part = image_box_overlap(gt_boxes, dt_boxes)
        # Calculate in camera coordinates: z forward, y down, x right
        # So bird's eye view takes x and z
        elif metric == 1:
            loc = np.concatenate(
                [a["location"][:, [0, 2]] for a in gt_annos_part], 0)
            dims = np.concatenate(
                [a["dimensions"][:, [0, 2]] for a in gt_annos_part], 0)
            rots = np.concatenate([a["rotation_y"] for a in gt_annos_part], 0)
            gt_boxes = np.concatenate(
                [loc, dims, rots[..., np.newaxis]], axis=1)  # (N,5)
            loc = np.concatenate(
                [a["location"][:, [0, 2]] for a in dt_annos_part], 0)
            dims = np.concatenate(
                [a["dimensions"][:, [0, 2]] for a in dt_annos_part], 0)
            rots = np.concatenate([a["rotation_y"] for a in dt_annos_part], 0)
            dt_boxes = np.concatenate(
                [loc, dims, rots[..., np.newaxis]], axis=1)  # (K,5)
            overlap_part = bev_box_overlap(gt_boxes, dt_boxes).astype(np.float64)
        elif metric == 2:
            loc = np.concatenate([a["location"] for a in gt_annos_part], 0)
            dims = np.concatenate([a["dimensions"] for a in gt_annos_part], 0)
            rots = np.concatenate([a["rotation_y"] for a in gt_annos_part], 0)
            gt_boxes = np.concatenate(
                [loc, dims, rots[..., np.newaxis]], axis=1)
            loc = np.concatenate([a["location"] for a in dt_annos_part], 0)
            dims = np.concatenate([a["dimensions"] for a in dt_annos_part], 0)
            rots = np.concatenate([a["rotation_y"] for a in dt_annos_part], 0)
            dt_boxes = np.concatenate(
                [loc, dims, rots[..., np.newaxis]], axis=1)
            overlap_part = d3_box_overlap(gt_boxes, dt_boxes).astype(np.float64)
        else:
            raise ValueError("unknown metric")
        parted_overlaps.append(overlap_part)  # Record part overlaps, 101 elements total
        example_idx += num_part  # Update example_idx as new part start
    
    # Extra calculations done (IoU between different point clouds in same part), now trim
    overlaps = []
    example_idx = 0
    for j, num_part in enumerate(split_parts):
        gt_annos_part = gt_annos[example_idx:example_idx + num_part]  # 37
        dt_annos_part = dt_annos[example_idx:example_idx + num_part]  # 37
        gt_num_idx, dt_num_idx = 0, 0
        # Calculate overlaps frame by frame
        for i in range(num_part):
            gt_box_num = total_gt_num[example_idx + i]
            dt_box_num = total_dt_num[example_idx + i]
            # In j-th part of overlaps, extract corresponding IoU based on gt_box_num and dt_box_num
            overlaps.append(
                parted_overlaps[j][gt_num_idx:gt_num_idx + gt_box_num,
                                   dt_num_idx:dt_num_idx + dt_box_num])
            gt_num_idx += gt_box_num
            dt_num_idx += dt_box_num
        example_idx += num_part
    
    # overlaps: 3769 elements, each represents IoU of one frame
    # parted_overlaps: 101 elements, IoU per part
    # total_gt_num: 3769 elements, gt box count per frame
    # total_dt_num: 3769 elements, det box count per frame
    return overlaps, parted_overlaps, total_gt_num, total_dt_num


def _prepare_data(gt_annos, dt_annos, current_class, difficulty):
    """
    Classify and integrate data
    Args:
        gt_annos: Point cloud annotation dict 3796
        dt_annos: Point cloud prediction dict 3796
        current_class: Scalar 0
        difficulty: Scalar 0
    Returns:
        gt_datas_list: 3769 elements, each (N,5) --> (x1,y1,x2,y2,alpha) (N varies)
        dt_datas_list: 3769 elements, each (M,6) --> (x1,y1,x2,y2,alpha,score) (M varies)
        ignored_gts: 3769 elements, each (N,) box status 0,1,-1
        ignored_dets: 3769 elements, each (M,) box status 0,1,-1
        dontcares: 3769 elements, each (k,4) (k varies)
        total_dc_num: 3769 elements, dc box count per frame
        total_num_valid_gt: Total valid boxes: 2906
    """
    # Data initialization
    gt_datas_list = []
    dt_datas_list = []
    total_dc_num = []
    ignored_gts, ignored_dets, dontcares = [], [], []
    total_num_valid_gt = 0

    # Iterate through test set point clouds
    for i in range(len(gt_annos)):
        # Clean data
        rets = clean_data(gt_annos[i], dt_annos[i], current_class, difficulty)
        # num_valid_gt: Number of valid gt
        # ignored_gt: gt flag list 0=valid, 1=ignore, -1=other
        # ignored_dt: dt flag list
        # dc_bboxes: Don't Care boxes
        num_valid_gt, ignored_gt, ignored_det, dc_bboxes = rets

        # Add ignored_gt status list to ignored_gts
        ignored_gts.append(np.array(ignored_gt, dtype=np.int64))
        ignored_dets.append(np.array(ignored_det, dtype=np.int64))

        # Process don't care boxes
        if len(dc_bboxes) == 0:
            dc_bboxes = np.zeros((0, 4)).astype(np.float64)
        else:
            dc_bboxes = np.stack(dc_bboxes, 0).astype(np.float64)
        total_dc_num.append(dc_bboxes.shape[0])  # Total don't care box count
        dontcares.append(dc_bboxes)  # Add don't care boxes to list
        total_num_valid_gt += num_valid_gt  # Calculate total valid gt count
        
        # Recombine detection results
        gt_datas = np.concatenate(
            [gt_annos[i]["bbox"], gt_annos[i]["alpha"][..., np.newaxis]], 1)  # (N,5) --> (x1,y1,x2,y2,alpha)
        dt_datas = np.concatenate([
            dt_annos[i]["bbox"], dt_annos[i]["alpha"][..., np.newaxis],
            dt_annos[i]["score"][..., np.newaxis]
        ], 1)
        # Add results to lists
        gt_datas_list.append(gt_datas)
        dt_datas_list.append(dt_datas)
    total_dc_num = np.stack(total_dc_num, axis=0)  # (3769,) List of total don't care box counts
    return (gt_datas_list, dt_datas_list, ignored_gts, ignored_dets, dontcares,
            total_dc_num, total_num_valid_gt)


def eval_class(gt_annos,
               dt_annos,
               current_classes,
               difficultys,
               metric,
               min_overlaps,
               compute_aos=False,
               num_parts=100):
    """Kitti evaluation. Supports 2d/bev/3d/aos evaluation. Supports 0.5:0.05:0.95 coco AP.
    Args:
        gt_annos: dict from get_label_annos() in kitti_common.py
        dt_annos: dict from get_label_annos() in kitti_common.py
        current_classes: list of int, 0: car, 1: pedestrian, 2: cyclist
        difficultys: list of int, eval difficulty, 0: easy, 1: normal, 2: hard
        metric: eval type 0: bbox, 1: bev, 2: 3d
        min_overlaps: float, min overlap. format: [num_overlap, metric, class]. (2,3,3)
        compute_aos: bool
        num_parts: int parameter for fast calculation
    Returns:
        dict of recall, precision and aos
    """
    assert len(gt_annos) == len(dt_annos)
    num_examples = len(gt_annos)  # 3769
    # 1. split_parts has 101 elements, 100 of 37 and one 69 --> [37,37......69]
    split_parts = get_split_parts(num_examples, num_parts)
    # 2. Calculate IoU
    rets = calculate_iou_partly(dt_annos, gt_annos, metric, num_parts)
    # overlaps: 3769 elements, IoU per frame
    # parted_overlaps: 101 elements, IoU per part
    # total_gt_num: 3769 elements, gt box count per frame
    # total_dt_num: 3769 elements, det box count per frame
    overlaps, parted_overlaps, total_dt_num, total_gt_num = rets

    # 3. Calculate required array dimensions for initialization
    N_SAMPLE_PTS = 41
    num_minoverlap = len(min_overlaps)
    num_class = len(current_classes)
    num_difficulty = len(difficultys)
    # Initialize precision, recall and aos
    precision = np.zeros([num_class, num_difficulty, num_minoverlap, N_SAMPLE_PTS])
    recall = np.zeros([num_class, num_difficulty, num_minoverlap, N_SAMPLE_PTS])
    aos = np.zeros([num_class, num_difficulty, num_minoverlap, N_SAMPLE_PTS])

    # 4. Iterate through classes 0: car, 1: pedestrian, 2: cyclist
    for m, current_class in enumerate(current_classes):
        # Iterate through difficulties 0: easy, 1: normal, 2: hard
        for l, difficulty in enumerate(difficultys):
            # 4.1 Prepare data
            rets = _prepare_data(gt_annos, dt_annos, current_class, difficulty)
            # gt_datas_list: 3769 elements, each (N,5) --> (x1,y1,x2,y2,alpha) (N varies)
            # dt_datas_list: 3769 elements, each (M,6) --> (x1,y1,x2,y2,alpha,score) (M varies)
            # ignored_gts: 3769 elements, each (N,) box status 0,1,-1
            # ignored_dets: 3769 elements, each (M,) box status 0,1,-1
            # dontcares: 3769 elements, each (k,4) (k varies)
            # total_dc_num: 3769 elements, dc box count per frame
            # total_num_valid_gt: Total valid boxes: 2906
            (gt_datas_list, dt_datas_list, ignored_gts, ignored_dets,
             dontcares, total_dc_num, total_num_valid_gt) = rets
            # 4.2 Iterate through min_overlaps
            for k, min_overlap in enumerate(min_overlaps[:, metric, m]):
                thresholdss = []
                # 4.2.1 Calculate all scores of dt matching valid gt, total 2838 box scores and 41 recall thresholds
                for i in range(len(gt_annos)):
                    # compute_statistics_jit not actually calculating metrics here, but getting thresholdss to calculate 41 recall thresholds
                    rets = compute_statistics_jit(
                        overlaps[i],  # Single frame IoU (N,M)
                        gt_datas_list[i],  # (N,5) --> (x1,y1,x2,y2,alpha)
                        dt_datas_list[i],  # (M,6) --> (x1,y1,x2,y2,alpha,score)
                        ignored_gts[i],  # (N,) box status 0,1,-1
                        ignored_dets[i],  # (M,) box status 0,1,-1
                        dontcares[i],  # (k,4)
                        metric,  # 0: bbox, 1: bev, 2: 3d
                        min_overlap=min_overlap,  # Min IoU threshold
                        thresh=0.0,  # Ignore dt with score below this
                        compute_fp=False)
                    tp, fp, fn, similarity, thresholds = rets
                    thresholdss += thresholds.tolist()  # List addition like extend (2838,)
                thresholdss = np.array(thresholdss)
                thresholds = get_thresholds(thresholdss, total_num_valid_gt)  # Get 41 recall thresholds
                thresholds = np.array(thresholds)
                pr = np.zeros([len(thresholds), 4])  # (41, 4)
                idx = 0

                # 4.2.2 Iterate through 101 parts, calculate tp, fp, fn, similarity per frame per recall_threshold in part, accumulate pr
                for j, num_part in enumerate(split_parts):
                    gt_datas_part = np.concatenate(
                        gt_datas_list[idx:idx + num_part], 0)  # (N,5)
                    dt_datas_part = np.concatenate(
                        dt_datas_list[idx:idx + num_part], 0)  # (M,6)
                    dc_datas_part = np.concatenate(
                        dontcares[idx:idx + num_part], 0)  # (K,4)
                    ignored_dets_part = np.concatenate(
                        ignored_dets[idx:idx + num_part], 0)  # (M,)
                    ignored_gts_part = np.concatenate(
                        ignored_gts[idx:idx + num_part], 0)  # (N,)
                    # Actually calculate metrics, fuse statistics
                    fused_compute_statistics(
                        parted_overlaps[j],
                        pr,
                        total_gt_num[idx:idx + num_part],
                        total_dt_num[idx:idx + num_part],
                        total_dc_num[idx:idx + num_part],
                        gt_datas_part,
                        dt_datas_part,
                        dc_datas_part,
                        ignored_gts_part,
                        ignored_dets_part,
                        metric,
                        min_overlap=min_overlap,
                        thresholds=thresholds,
                        compute_aos=compute_aos)
                    idx += num_part
                
                # 4.2.3 Calculate metrics based on class, difficulty, min IoU threshold and recall threshold
                for i in range(len(thresholds)):
                    # m:class, l:difficulty, k:min_overlap, i:threshold
                    # pr:(41,4) --> tp, fp, fn, similarity
                    recall[m, l, k, i] = pr[i, 0] / (pr[i, 0] + pr[i, 2])  # recall = tp/(tp+fn) actual value (3,3,2,41)
                    precision[m, l, k, i] = pr[i, 0] / (pr[i, 0] + pr[i, 1])  # precision = tp/(tp+fp) predicted value (3,3,2,41)
                    if compute_aos:
                        aos[m, l, k, i] = pr[i, 3] / (pr[i, 0] + pr[i, 1])  # aos = similarity/(tp+fp)
                # 4.2.4 Since PR curve is outer arc shape, take max after threshold node, equivalent to rectangular cut at node
                for i in range(len(thresholds)):
                    precision[m, l, k, i] = np.max(
                        precision[m, l, k, i:], axis=-1)
                    recall[m, l, k, i] = np.max(recall[m, l, k, i:], axis=-1)
                    if compute_aos:
                        aos[m, l, k, i] = np.max(aos[m, l, k, i:], axis=-1)
    ret_dict = {
        "recall": recall,  # (3,3,2,41)
        "precision": precision,  # (3,3,2,41)
        "orientation": aos,  # (3,3,2,41)
    }
    return ret_dict


def get_mAP(prec):
    """
    mAP: Area under PR curve, divided into mAP_R11 and mAP_R40 based on segments
    """
    sums = 0
    # Take every 4 points, assuming width=1, finally normalize
    for i in range(0, prec.shape[-1], 4):
        sums = sums + prec[..., i]
    return sums / 11 * 100 


def get_mAP_R40(prec):
    sums = 0
    for i in range(1, prec.shape[-1]):
        sums = sums + prec[..., i]
    return sums / 40 * 100


def print_str(value, *arg, sstream=None):
    "Result printing function"
    if sstream is None:
        sstream = sysio.StringIO()
    sstream.truncate(0)
    sstream.seek(0)
    print(value, *arg, file=sstream)
    return sstream.getvalue()


def do_eval(gt_annos,
            dt_annos,
            current_classes,
            min_overlaps,
            compute_aos=False,
            PR_detail_dict=None):
    """
    Calculate evaluation metrics
    Args:
        gt_annos: 3769 frame test point cloud annotation dict including {name,alpha,rotation_y,gt_boxes_lidar} etc.
        dt_annos: 3769 frame test point cloud prediction dict including {name,alpha,rotation_y,boxes_lidar} etc.
        current_classes: [0,1,2]
        min_overlaps: (2,3,3)
        compute_aos: True
        PR_detail_dict: None
    """
    # min_overlaps: [num_minoverlap, metric, num_class]
    difficultys = [0, 1, 2]  # difficultys: list of int, eval difficulty, 0: easy, 1: normal, 2: hard
    # bbox, bev, 3d and aos calculations are similar, all calling eval_class, just 5th parameter differs (0,1,2)
    # ret_dict = {
    #     "recall": (3,3,2,41)
    #     "precision": (3,3,2,41)
    #     "orientation": (3,3,2,41)
    # } m:class, l:difficulty, k:min_overlap, i:threshold
    # ret: [num_class, num_diff, num_minoverlap, num_sample_points]
    # eval_class will call calculate_iou_partly for different IoU based on metric
    ret = eval_class(gt_annos, dt_annos, current_classes, difficultys, 0,
                     min_overlaps, compute_aos)
    
    mAP_bbox = get_mAP(ret["precision"])
    mAP_bbox_R40 = get_mAP_R40(ret["precision"])

    if PR_detail_dict is not None:
        PR_detail_dict['bbox'] = ret['precision']

    mAP_aos = mAP_aos_R40 = None
    if compute_aos:
        mAP_aos = get_mAP(ret["orientation"])
        mAP_aos_R40 = get_mAP_R40(ret["orientation"])

        if PR_detail_dict is not None:
            PR_detail_dict['aos'] = ret['orientation']

    ret = eval_class(gt_annos, dt_annos, current_classes, difficultys, 1,
                     min_overlaps)
    mAP_bev = get_mAP(ret["precision"])
    mAP_bev_R40 = get_mAP_R40(ret["precision"])

    if PR_detail_dict is not None:
        PR_detail_dict['bev'] = ret['precision']

    ret = eval_class(gt_annos, dt_annos, current_classes, difficultys, 2,
                     min_overlaps)
    mAP_3d = get_mAP(ret["precision"])
    mAP_3d_R40 = get_mAP_R40(ret["precision"])
    if PR_detail_dict is not None:
        PR_detail_dict['3d'] = ret['precision']
    # Return 8 values: mAP and mAP_R40 for bbox, bev, 3d and aos etc.
    return mAP_bbox, mAP_bev, mAP_3d, mAP_aos, mAP_bbox_R40, mAP_bev_R40, mAP_3d_R40, mAP_aos_R40


def do_coco_style_eval(gt_annos, dt_annos, current_classes, overlap_ranges,
                       compute_aos):
    # overlap_ranges: [range, metric, num_class]
    min_overlaps = np.zeros([10, *overlap_ranges.shape[1:]])
    for i in range(overlap_ranges.shape[1]):
        for j in range(overlap_ranges.shape[2]):
            min_overlaps[:, i, j] = np.linspace(*overlap_ranges[:, i, j])
    mAP_bbox, mAP_bev, mAP_3d, mAP_aos = do_eval(
        gt_annos, dt_annos, current_classes, min_overlaps, compute_aos)
    # ret: [num_class, num_diff, num_minoverlap]
    mAP_bbox = mAP_bbox.mean(-1)
    mAP_bev = mAP_bev.mean(-1)
    mAP_3d = mAP_3d.mean(-1)
    if mAP_aos is not None:
        mAP_aos = mAP_aos.mean(-1)
    return mAP_bbox, mAP_bev, mAP_3d, mAP_aos


def get_official_eval_result(gt_annos, dt_annos, current_classes, PR_detail_dict=None):
    """
    Top-level metric calculation function (like main function), also responsible for printing and output
    Args:
        gt_annos: 3769 frame test point cloud annotation dict including {name,alpha,rotation_y,gt_boxes_lidar} etc.
        dt_annos: 3769 frame test point cloud prediction dict including {name,alpha,rotation_y,boxes_lidar} etc.
        current_classes: [0,1,2]
        compute_aos: True
        PR_detail_dict: None
    """
    # Set different thresholds for different classes
    # 6 classes' IoU thresholds in easy, moderate and hard
    overlap_0_7 = np.array([[0.7, 0.5, 0.5, 0.7, 0.5, 0.7], 
                            [0.7, 0.5, 0.5, 0.7, 0.5, 0.7],
                            [0.7, 0.5, 0.5, 0.7, 0.5, 0.7]])
    overlap_0_5 = np.array([[0.7, 0.5, 0.5, 0.7, 0.5, 0.5], 
                            [0.5, 0.25, 0.25, 0.5, 0.25, 0.5],
                            [0.5, 0.25, 0.25, 0.5, 0.25, 0.5]])
    # Each class outputs 4 results
    # Using AP at overlap=0.7 or 0.5
    # And AP_R40 at overlap=0.7 or 0.5
    min_overlaps = np.stack([overlap_0_7, overlap_0_5], axis=0)  
    class_to_name = {
        0: 'Car',           #[[0.7,0.7,0.7],[0.7, 0.5, 0.5]]
        1: 'Pedestrian',    #[[0.5,0.5,0.5],[0.5,0.25,0.25]]
        2: 'Cyclist',      #[[0.5,0.5,0.5],[0.5,0.25,0.25]]
        3: 'Van',          #[[0.7,0.7,0.7],[0.7, 0.5, 0.5]]
        4: 'Person_sitting',#[[0.5,0.5,0.5],[0.5,0.25,0.25]]
        5: 'Truck'         #[[0.7,0.7,0.7],[0.5, 0.5, 0.5]]
    }
    name_to_class = {v: n for n, v in class_to_name.items()}  # Map class names to numbers
    # If input classes not list, convert to list
    if not isinstance(current_classes, (list, tuple)):
        current_classes = [current_classes]
    # Convert evaluation classes to numbers
    current_classes_int = []
    for curcls in current_classes:
        if isinstance(curcls, str):
            current_classes_int.append(name_to_class[curcls])
        else:
            current_classes_int.append(curcls)
    current_classes = current_classes_int
    # Select IoU thresholds based on classes
    min_overlaps = min_overlaps[:, :, current_classes]  # Take first 3 columns
    result = ''
    # check whether alpha is valid, decide whether to calculate AOS
    compute_aos = False
    for anno in dt_annos:
        if anno['alpha'].shape[0] != 0:
            if anno['alpha'][0] != -10:
                compute_aos = True
            break
    
    # Core calculation function: pass gt and det annos, detection classes, IoU thresholds and AOS flag, PR_detail_dict=None
    mAPbbox, mAPbev, mAP3d, mAPaos, mAPbbox_R40, mAPbev_R40, mAP3d_R40, mAPaos_R40 = do_eval(
        gt_annos, dt_annos, current_classes, min_overlaps, compute_aos, PR_detail_dict=PR_detail_dict)

    # Print results
    ret_dict = {}
    for j, curcls in enumerate(current_classes):
        # mAP threshold array: [num_minoverlap, metric, class]
        # mAP result: [num_class, num_diff, num_minoverlap]
        for i in range(min_overlaps.shape[0]):
            result += print_str(
                (f"{class_to_name[curcls]} "
                 "AP@{:.2f}, {:.2f}, {:.2f}:".format(*min_overlaps[i, :, j])))  # i selects IoU 0.7 or 0.5, j selects class
            
            result += print_str((f"bbox AP:{mAPbbox[j, 0, i]:.4f}, "
                                 f"{mAPbbox[j, 1, i]:.4f}, "
                                 f"{mAPbbox[j, 2, i]:.4f}"))
            result += print_str((f"bev  AP:{mAPbev[j, 0, i]:.4f}, "
                                 f"{mAPbev[j, 1, i]:.4f}, "
                                 f"{mAPbev[j, 2, i]:.4f}"))
            result += print_str((f"3d   AP:{mAP3d[j, 0, i]:.4f}, "
                                 f"{mAP3d[j, 1, i]:.4f}, "
                                 f"{mAP3d[j, 2, i]:.4f}"))
            # Calculate AOS
            if compute_aos:
                result += print_str((f"aos  AP:{mAPaos[j, 0, i]:.2f}, "
                                     f"{mAPaos[j, 1, i]:.2f}, "
                                     f"{mAPaos[j, 2, i]:.2f}"))
                # if i == 0:
                   # ret_dict['%s_aos/easy' % class_to_name[curcls]] = mAPaos[j, 0, 0]
                   # ret_dict['%s_aos/moderate' % class_to_name[curcls]] = mAPaos[j, 1, 0]
                   # ret_dict['%s_aos/hard' % class_to_name[curcls]] = mAPaos[j, 2, 0]
            
            # AP_R40 output to console and record in ret_dict, write to file
            result += print_str(
                (f"{class_to_name[curcls]} "
                 "AP_R40@{:.2f}, {:.2f}, {:.2f}:".format(*min_overlaps[i, :, j])))
            result += print_str((f"bbox AP:{mAPbbox_R40[j, 0, i]:.4f}, "
                                 f"{mAPbbox_R40[j, 1, i]:.4f}, "
                                 f"{mAPbbox_R40[j, 2, i]:.4f}"))
            result += print_str((f"bev  AP:{mAPbev_R40[j, 0, i]:.4f}, "
                                 f"{mAPbev_R40[j, 1, i]:.4f}, "
                                 f"{mAPbev_R40[j, 2, i]:.4f}"))
            result += print_str((f"3d   AP:{mAP3d_R40[j, 0, i]:.4f}, "
                                 f"{mAP3d_R40[j, 1, i]:.4f}, "
                                 f"{mAP3d_R40[j, 2, i]:.4f}"))
            if compute_aos:
                result += print_str((f"aos  AP:{mAPaos_R40[j, 0, i]:.2f}, "
                                     f"{mAPaos_R40[j, 1, i]:.2f}, "
                                     f"{mAPaos_R40[j, 2, i]:.2f}"))
                # Only record results at IoU=0.7 (i = 0)
                if i == 0:
                   ret_dict['%s_aos/easy_R40' % class_to_name[curcls]] = mAPaos_R40[j, 0, 0]
                   ret_dict['%s_aos/moderate_R40' % class_to_name[curcls]] = mAPaos_R40[j, 1, 0]
                   ret_dict['%s_aos/hard_R40' % class_to_name[curcls]] = mAPaos_R40[j, 2, 0]

            if i == 0:
                ret_dict['%s_3d/easy_R40' % class_to_name[curcls]] = mAP3d_R40[j, 0, 0]
                ret_dict['%s_3d/moderate_R40' % class_to_name[curcls]] = mAP3d_R40[j, 1, 0]
                ret_dict['%s_3d/hard_R40' % class_to_name[curcls]] = mAP3d_R40[j, 2, 0]
                ret_dict['%s_bev/easy_R40' % class_to_name[curcls]] = mAPbev_R40[j, 0, 0]
                ret_dict['%s_bev/moderate_R40' % class_to_name[curcls]] = mAPbev_R40[j, 1, 0]
                ret_dict['%s_bev/hard_R40' % class_to_name[curcls]] = mAPbev_R40[j, 2, 0]
                ret_dict['%s_image/easy_R40' % class_to_name[curcls]] = mAPbbox_R40[j, 0, 0]
                ret_dict['%s_image/moderate_R40' % class_to_name[curcls]] = mAPbbox_R40[j, 1, 0]
                ret_dict['%s_image/hard_R40' % class_to_name[curcls]] = mAPbbox_R40[j, 2, 0]

    return result, ret_dict


def get_coco_eval_result(gt_annos, dt_annos, current_classes):
    class_to_name = {
        0: 'Car',
        1: 'Pedestrian',
        2: 'Cyclist',
        3: 'Van',
        4: 'Person_sitting',
    }
    class_to_range = {
        0: [0.5, 0.95, 10],
        1: [0.25, 0.7, 10],
        2: [0.25, 0.7, 10],
        3: [0.5, 0.95, 10],
        4: [0.25, 0.7, 10],
    }
    name_to_class = {v: n for n, v in class_to_name.items()}
    if not isinstance(current_classes, (list, tuple)):
        current_classes = [current_classes]
    current_classes_int = []
    for curcls in current_classes:
        if isinstance(curcls, str):
            current_classes_int.append(name_to_class[curcls])
        else:
            current_classes_int.append(curcls)
    current_classes = current_classes_int
    overlap_ranges = np.zeros([3, 3, len(current_classes)])
    for i, curcls in enumerate(current_classes):
        overlap_ranges[:, :, i] = np.array(
            class_to_range[curcls])[:, np.newaxis]
    result = ''
    # check whether alpha is valid
    compute_aos = False
    for anno in dt_annos:
        if anno['alpha'].shape[0] != 0:
            if anno['alpha'][0] != -10:
                compute_aos = True
            break
    mAPbbox, mAPbev, mAP3d, mAPaos = do_coco_style_eval(
        gt_annos, dt_annos, current_classes, overlap_ranges, compute_aos)
    for j, curcls in enumerate(current_classes):
        # mAP threshold array: [num_minoverlap, metric, class]
        # mAP result: [num_class, num_diff, num_minoverlap]
        o_range = np.array(class_to_range[curcls])[[0, 2, 1]]
        o_range[1] = (o_range[2] - o_range[0]) / (o_range[1] - 1)
        result += print_str((f"{class_to_name[curcls]} "
                             "coco AP@{:.2f}:{:.2f}:{:.2f}:".format(*o_range)))
        result += print_str((f"bbox AP:{mAPbbox[j, 0]:.2f}, "
                             f"{mAPbbox[j, 1]:.2f}, "
                             f"{mAPbbox[j, 2]:.2f}"))
        result += print_str((f"bev  AP:{mAPbev[j, 0]:.2f}, "
                             f"{mAPbev[j, 1]:.2f}, "
                             f"{mAPbev[j, 2]:.2f}"))
        result += print_str((f"3d   AP:{mAP3d[j, 0]:.2f}, "
                             f"{mAP3d[j, 1]:.2f}, "
                             f"{mAP3d[j, 2]:.2f}"))
        if compute_aos:
            result += print_str((f"aos  AP:{mAPaos[j, 0]:.2f}, "
                                 f"{mAPaos[j, 1]:.2f}, "
                                 f"{mAPaos[j, 2]:.2f}"))
    return result