import numpy as np


def iou(box1, box2):
    """
    计算两个边界框的IoU（交并比）
    """
    # 确定交集区域的坐标
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])

    # 计算交集区域面积
    width_inter = max(0, x2_inter - x1_inter)
    height_inter = max(0, y2_inter - y1_inter)
    area_inter = width_inter * height_inter

    # 计算各自区域面积
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # 计算并集面积
    area_union = area1 + area2 - area_inter

    # 避免除以零
    if area_union == 0:
        return 0.0
    return area_inter / area_union


def ats_nms(boxes, scores, k=9):
    """
    Adaptive Threshold Soft NMS算法实现
    :param boxes: 边界框列表，格式为[[x1,y1,x2,y2], ...]
    :param scores: 置信度得分列表
    :param k: 考虑的相邻检测框数量
    :return: 抑制后的检测框和对应得分
    """
    boxes = np.array(boxes)
    scores = np.array(scores)
    keep_boxes = []
    keep_scores = []

    while len(boxes) > 0:
        # 获取当前最高得分索引
        m = np.argmax(scores)
        best_box = boxes[m]
        best_score = scores[m]

        # 保存当前最高得分框
        keep_boxes.append(best_box.tolist())
        keep_scores.append(best_score)

        # 移除当前框
        mask = np.arange(len(boxes)) != m
        boxes = boxes[mask]
        scores = scores[mask]

        if len(boxes) == 0:
            break

        # 计算中心点距离
        centers = np.array([[(b[0] + b[2]) / 2, (b[1] + b[3]) / 2] for b in boxes])
        best_center = [(best_box[0] + best_box[2]) / 2, (best_box[1] + best_box[3]) / 2]
        distances = np.sqrt(np.sum((centers - best_center) ** 2, axis=1))

        # 选择k个最近邻
        k_selected = min(k, len(boxes))
        if k_selected == 0:
            continue

        # 获取距离最近的k个索引
        nearest_indices = np.argpartition(distances, k_selected - 1)[:k_selected]
        candidate_boxes = boxes[nearest_indices]

        # 计算IoU并统计
        ious = np.array([iou(best_box, box) for box in candidate_boxes])
        mean_iou = np.mean(ious)
        std_iou = np.std(ious)
        epsilon = mean_iou + std_iou

        # 更新候选框得分
        for i in nearest_indices:
            box = boxes[i]
            box_iou = iou(best_box, box)
            box_center = [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]
            distance = np.sqrt((box_center[0] - best_center[0]) ** 2 +
                               (box_center[1] - best_center[1]) ** 2)

            if abs(box_iou - distance) >= epsilon:
                scores[i] *= (1 - box_iou)

    return keep_boxes, keep_scores