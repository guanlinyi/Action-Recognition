# Copyright (c) OpenMMLab. All rights reserved.
import time
import argparse
import queue
import tempfile
import cv2
import mmcv
import mmengine
import numpy as np
import torch
import os
from mmengine import DictAction
from mmengine.utils import track_iter_progress

from mmaction.apis import inference_skeleton, init_recognizer
from mmaction.registry import VISUALIZERS
from mmaction.utils import frame_extract

try:
    from ultralytics import YOLO
except ImportError:
    raise ImportError('Please install ultralytics to use YOLOv26 pose model')

FONTSCALE = 1.2          # 从 0.75 增大到 1.2（1.6倍）
THICKNESS = 2            # 从 1 增大到 2

# 新增 FPS 专用大字体参数
FPS_FONTSCALE = 1.5      # FPS 更大
FPS_THICKNESS = 3        # FPS 更粗
FPS_COLOR = (0, 255, 0)  # 亮绿色

# 添加字体相关常量定义
FONTFACE = cv2.FONT_HERSHEY_SIMPLEX      # 字体类型
FONTCOLOR = (255, 255, 255)              # 白色文字
LINETYPE = cv2.LINE_AA                   # 抗锯齿线型，更平滑

# COCO 关键点连接关系（用于绘制骨架）
SKELETON = [
    [15, 13], [13, 11], [16, 14], [14, 12], [11, 12],  # 下肢
    [5, 11], [6, 12], [5, 6], [5, 7], [6, 8], [7, 9], [8, 10],  # 躯干和上肢
    [1, 2], [2, 3], [3, 4], [1, 0], [0, 4]  # 头部和面部
]
# 关键点颜色（可选）
POSE_COLOR = (0, 255, 0)  # 绿色
LINK_COLOR = (255, 0, 0)  # 蓝色


def parse_args():
    parser = argparse.ArgumentParser(description='MMAction2 real-time demo')
    input_dir= r"D:\zero_track\mmaction2\my_code\input"
    input_video_name = "fisheye_back_2026-01-29-14-13-15_30fps.mp4"
    input_video_path = input_dir + os.sep + input_video_name
    output_video_path = input_dir + os.sep + "output_" + input_video_name
    parser.add_argument('--video', help='video file/url or camera index (e.g. 0)', default=input_video_path)
    parser.add_argument('--out-filename', help='output filename (optional)', default=output_video_path)
    parser.add_argument('--no-action', action='store_true', default=False,
                    help='disable action recognition, only run pose estimation')
    parser.add_argument('--no-skeleton', action='store_true', help='disable drawing skeleton', default=True,)
    parser.add_argument('--no-bbox', action='store_true', help='disable drawing bounding boxes', default=False,)
    parser.add_argument('--inference-interval',type=int,default=24,help='每累积多少帧进行一次动作识别推理 (默认: 24)')
    parser.add_argument(
        '--config',
        default=('configs/skeleton/posec3d/'
                 'slowonly_r50_8xb16-u48-240e_ntu60-xsub-keypoint_my.py'),
        help='skeleton model config file path')
    parser.add_argument(
        '--checkpoint',
        default=r"D:\zero_track\mmaction2\work_dirs\slowonly_r50_8xb16-u48-240e_ntu60-xsub-keypoint\20260202_173400_best\best_acc_top1_epoch_13.pth",
        help='skeleton model checkpoint file/url')
    parser.add_argument(
        '--yolo-model',
        default='checkpoints/yolo26s-pose.pt',
        help='YOLOv26 pose model file path (default: yolov26s-pose.pt)')
    parser.add_argument(
        '--label-map',
        default='tools/data/skeleton/label_map_ntu60.txt',
        help='label map file')
    parser.add_argument(
        '--device', type=str, default='cuda:0', help='CPU/CUDA device option')
    parser.add_argument(
        '--short-side',
        type=int,
        default=480,
        help='resize the short side of the frame to this value')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        default={},
        help='override some settings in the used config, the key-value pair '
             'in xxx=yyy format will be merged into config file.')
    args = parser.parse_args()
    return args


def convert_yolo_to_mmaction(results, img_shape):
    """
    将 YOLOv26 姿态估计结果转换为 MMAction2 可接受的格式。

    Args:
        results: YOLOv26 预测结果对象
        img_shape: (h, w) 原始图像尺寸

    Returns:
        pose_results: 字典列表，每个字典包含 keypoints 和 keypoint_scores
    """
    pose_results = []
    # 检查是否有检测到的人体关键点
    if (results[0].keypoints is not None and 
        results[0].keypoints.data is not None and 
        len(results[0].keypoints.data) > 0):
        
        # 取置信度最高的人（或所有人，这里简化取第一个）
        kpts_data = results[0].keypoints.data[0].cpu().numpy()  # shape (17, 3)
        keypoints = kpts_data[:, :2]  # (17, 2)
        scores = kpts_data[:, 2]      # (17,)

        # 归一化坐标（如果需要，模型可能要求绝对坐标，这里保持绝对坐标）
        keypoints = keypoints.astype(np.float32)
        scores = scores.astype(np.float32)

        # 构造与 mmpose 输出类似的字典
        pose_results.append({
            'keypoints': keypoints[np.newaxis, ...],  # (1, 17, 2)
            'keypoint_scores': scores[np.newaxis, ...]  # (1, 17)
        })
    else:
        # 没有检测到人，返回空关键点（全零）
        keypoints = np.zeros((1, 17, 2), dtype=np.float32)
        scores = np.zeros((1, 17), dtype=np.float32)
        pose_results.append({
            'keypoints': keypoints,
            'keypoint_scores': scores
        })
    return pose_results


def main():
    args = parse_args()

    # 加载动作识别模型
    config = mmengine.Config.fromfile(args.config)
    config.merge_from_dict(args.cfg_options)
    model = init_recognizer(config, args.checkpoint, args.device)

    # 加载 YOLOv26 姿态模型
    yolo_model = YOLO(args.yolo_model)

    # 打开视频
    cap = cv2.VideoCapture(args.video if not args.video.isdigit() else int(args.video))
    if not cap.isOpened():
        raise RuntimeError(f'Failed to open video source: {args.video}')
    ret, frame = cap.read()
    if not ret:
        raise RuntimeError('Failed to read video stream')
    h, w = frame.shape[:2]

    # 获取原视频的FPS
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or fps is None:
        fps = 30
    print(f"原视频尺寸: {w}x{h}, FPS: {fps}")

    # 计算缩放比例（用于姿态估计，但绘制在原始帧上）
    scale = 1.0
    new_w, new_h = w, h
    if args.short_side > 0:
        scale = args.short_side / min(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        print(f"缩放后尺寸: {new_w}x{new_h} (用于姿态估计)")

    # 输出视频写入（使用原始尺寸）
    out_writer = None
    if args.out_filename:
        output_dir = os.path.dirname(args.out_filename)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"创建输出目录: {output_dir}")

        # 尝试多种编码器
        fourcc_options = ['mp4v', 'XVID', 'avc1', 'MJPG']
        for codec in fourcc_options:
            fourcc = cv2.VideoWriter_fourcc(*codec)
            out_writer = cv2.VideoWriter(args.out_filename, fourcc, fps, (w, h))
            if out_writer.isOpened():
                print(f"成功创建视频写入器，编码器: {codec}, 尺寸: {w}x{h}")
                break
        else:
            print(f"警告：无法创建视频写入器，尝试的编码器: {fourcc_options}")
            print(f"输出路径: {os.path.abspath(args.out_filename)}")

    # 加载标签映射
    with open(args.label_map, 'r') as f:
        label_map = [line.strip() for line in f.readlines()]

    # 用于跟踪的数据结构
    window_size = 48
    pose_history = {}
    action_labels = {}
    first_frame = True
    
    # 新增：推理间隔控制变量
    inference_interval = args.inference_interval
    frame_counter = {}  # 每个ID的帧计数器
    last_inference_results = {}  # 缓存上次的推理结果，避免频繁更新标签闪烁

    # FPS计算相关变量
    fps_history = []
    fps_avg = 0
    last_time = time.time()
    frame_count = 0

    print(f'动作识别推理间隔: 每 {inference_interval} 帧推理一次')
    print('Press "q" to quit.')

    current_frame = frame
    is_first_loop = True

    while True:
        if not is_first_loop:
            ret, current_frame = cap.read()
            if not ret:
                break
        is_first_loop = False

        # FPS计算（保持不变）
        current_time = time.time()
        delta_time = current_time - last_time
        if delta_time > 0:
            instant_fps = 1.0 / delta_time
            fps_history.append(instant_fps)
            if len(fps_history) > 30:
                fps_history.pop(0)
            fps_avg = sum(fps_history) / len(fps_history)
        last_time = current_time
        frame_count += 1

        # 缩放帧用于姿态估计（保持不变）
        if scale != 1.0:
            frame_resized = cv2.resize(current_frame, (new_w, new_h))
        else:
            frame_resized = current_frame

        # 使用跟踪模式进行姿态估计（保持不变）
        if first_frame:
            results = yolo_model.track(
                frame_resized,
                device=args.device,
                verbose=False,
                persist=True,
                tracker="bytetrack.yaml"
            )
            first_frame = False
        else:
            results = yolo_model.track(
                frame_resized,
                device=args.device,
                verbose=False,
                persist=True
            )

        # 解析检测结果（保持不变）
        detections = []
        if (results[0].boxes is not None and results[0].boxes.id is not None):
            boxes = results[0].boxes
            keypoints = results[0].keypoints
            ids = boxes.id.cpu().numpy().astype(int)

            for i, obj_id in enumerate(ids):
                kpts = keypoints.data[i].cpu().numpy()
                keypoints_xy = kpts[:, :2] / scale
                scores = kpts[:, 2]
                bbox = boxes.xyxy[i].cpu().numpy() / scale
                bbox = bbox.astype(int)
                conf = boxes.conf[i].item()

                detections.append({
                    'id': obj_id,
                    'keypoints': keypoints_xy[np.newaxis, ...],
                    'keypoint_scores': scores[np.newaxis, ...],
                    'bbox': bbox,
                    'conf': conf
                })

        # 更新姿态历史（保持不变）
        current_ids = set()
        for det in detections:
            obj_id = det['id']
            current_ids.add(obj_id)
            if obj_id not in pose_history:
                pose_history[obj_id] = []
                frame_counter[obj_id] = 0  # 初始化计数器
            pose_history[obj_id].append({
                'keypoints': det['keypoints'],
                'keypoint_scores': det['keypoint_scores']
            })
            if len(pose_history[obj_id]) > window_size:
                pose_history[obj_id].pop(0)

        # 修改后的动作识别逻辑：按间隔触发推理
        for obj_id in current_ids:
            # 累积帧数
            frame_counter[obj_id] = frame_counter.get(obj_id, 0) + 1
            
            # 检查是否满足推理条件：窗口满且达到间隔帧数
            if (len(pose_history[obj_id]) == window_size and 
                frame_counter[obj_id] >= inference_interval):
                
                # 执行推理
                pose_results_for_model = pose_history[obj_id]
                with torch.no_grad():
                    result = inference_skeleton(model, pose_results_for_model, (h, w))
                pred_label = result.pred_score.argmax().item()
                
                # 更新标签并缓存结果
                action_labels[obj_id] = label_map[pred_label]
                last_inference_results[obj_id] = label_map[pred_label]
                
                # 重置计数器
                frame_counter[obj_id] = 0
                
                # 调试信息（可选）
                if frame_count % 30 == 0:
                    print(f"ID {obj_id}: 执行推理，标签: {label_map[pred_label]}")
            
            # 如果该ID有缓存结果但当前没有标签（首次或丢失后恢复），使用缓存
            elif obj_id not in action_labels and obj_id in last_inference_results:
                action_labels[obj_id] = last_inference_results[obj_id]

        # 清理消失ID的计数器（防止内存无限增长）
        disappeared_ids = set(pose_history.keys()) - current_ids
        for dead_id in disappeared_ids:
            if dead_id in frame_counter:
                del frame_counter[dead_id]
            # 可选：保留历史记录一段时间或立即清理
            # del pose_history[dead_id]
            # del action_labels[dead_id]

        # 在原始帧上绘制（不是缩放帧！）
        plotted_frame = current_frame.copy()

        for det in detections:
            obj_id = det['id']
            bbox = det['bbox']
            conf = det['conf']
            keypoints = det['keypoints'][0]
            keypoint_scores = det['keypoint_scores'][0]

            if not args.no_bbox:
                x1, y1, x2, y2 = bbox
                # 边框也加粗一点
                cv2.rectangle(plotted_frame, (x1, y1), (x2, y2), (0, 255, 0), 3)  # thickness 从 2 改为 3

            if not args.no_skeleton:
                for link in SKELETON:
                    idx1, idx2 = link
                    if keypoint_scores[idx1] > 0.3 and keypoint_scores[idx2] > 0.3:
                        pt1 = tuple(keypoints[idx1].astype(int))
                        pt2 = tuple(keypoints[idx2].astype(int))
                        # 骨架连线也加粗
                        cv2.line(plotted_frame, pt1, pt2, LINK_COLOR, 3)  # thickness 从 2 改为 3

                for kpt_idx, (x, y) in enumerate(keypoints):
                    if keypoint_scores[kpt_idx] > 0.3:
                        # 关键点圆圈也稍微大一点
                        cv2.circle(plotted_frame, (int(x), int(y)), 5, POSE_COLOR, -1)  # radius 从 3 改为 5

            if not args.no_bbox:
                x1, y1, x2, y2 = bbox
                label = action_labels.get(obj_id, '')
                text = f'ID:{obj_id} {label} {conf:.2f}'

                # 计算文本大小
                (text_width, text_height), _ = cv2.getTextSize(
                    text, FONTFACE, FONTSCALE, THICKNESS
                )

                # 绘制背景矩形（放在框内顶部）
                margin = 8
                bg_y1 = y1  # 背景顶部对齐框顶部
                bg_y2 = y1 + text_height + margin * 2  # 背景底部在框内
                
                cv2.rectangle(
                    plotted_frame,
                    (x1, bg_y1),
                    (x1 + text_width + margin * 2, bg_y2),
                    (0, 0, 0),  # 黑色背景
                    -1
                )

                # 绘制文本（在背景矩形内垂直居中）
                text_x = x1 + margin
                text_y = bg_y1 + text_height + margin  # 基线位置
                cv2.putText(
                    plotted_frame, 
                    text, 
                    (text_x, text_y),
                    FONTFACE, 
                    FONTSCALE, 
                    FONTCOLOR, 
                    THICKNESS, 
                    LINETYPE
                )

        # 在左上角绘制FPS信息（更大更醒目）
        fps_text = f"FPS: {fps_avg:.1f}"
        (fps_text_width, fps_text_height), _ = cv2.getTextSize(
            fps_text, FONTFACE, FPS_FONTSCALE, FPS_THICKNESS
        )

        fps_margin = 10
        fps_x, fps_y = 20, 20  # 稍微偏移一点，不贴边

        # 绘制更大的背景矩形（半透明黑色背景，更专业）
        overlay = plotted_frame.copy()
        cv2.rectangle(
            overlay,
            (fps_x - fps_margin, fps_y - fps_margin),
            (fps_x + fps_text_width + fps_margin, fps_y + fps_text_height + fps_margin),
            (0, 0, 0),
            -1
        )
        # 应用透明度（0.6透明度）
        cv2.addWeighted(overlay, 0.6, plotted_frame, 0.4, 0, plotted_frame)

        # 绘制FPS文本（亮绿色，更大更粗）
        cv2.putText(
            plotted_frame,
            fps_text,
            (fps_x, fps_y + fps_text_height),  # 基线位置
            FONTFACE,
            FPS_FONTSCALE,
            FPS_COLOR,
            FPS_THICKNESS,
            cv2.LINE_AA  # 使用抗锯齿，字体更平滑
        )

        # 显示画面（可以显示缩放后的版本以提高显示性能）
        if scale != 1.0:
            display_frame = cv2.resize(plotted_frame, (new_w, new_h))
        else:
            display_frame = plotted_frame
        cv2.imshow('MMAction2 Real-time Demo', display_frame)

        # 写入输出文件（使用原始尺寸的帧）
        if out_writer:
            write_ret = out_writer.write(plotted_frame)
            if frame_count % 30 == 0:
                status = "成功" if write_ret is True else f"返回{write_ret}"
                print(f"写入帧 {frame_count}, 状态: {status}, 帧尺寸: {plotted_frame.shape}")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 释放资源
    cap.release()
    if out_writer:
        out_writer.release()
        file_size = os.path.getsize(args.out_filename) / (1024*1024)  # MB
        print(f"视频已保存至: {args.out_filename} (大小: {file_size:.2f} MB)")
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
