import os
import pickle
import cv2
import numpy as np

# ==================== 配置参数 ====================
PKL_PATH = r"D:\zero_track\mmaction2\my_code\input\my_ntu60_2d.pkl"   # 生成的 pkl 文件路径
VIDEO_SRC_DIR = r"D:\zero_track\mmaction2\my_code\input\fisheye_back_2026-01-29-14-13-15_30fps\ok"                        # 视频存放根目录（递归搜索所有子文件夹）
OUTPUT_VIS_DIR = VIDEO_SRC_DIR + os.sep + "vis"                   # 可视化视频输出根目录

# ---------- 合并控制 ----------
CONCAT_TRAIN = True   # 是否合并 train 目录下的所有可视化视频
CONCAT_VAL   = True   # 是否合并 val 目录下的所有可视化视频

# ---------- 关键点骨架连接（COCO 17点示例）----------
SKELETON = [
    [0, 1], [0, 2], [1, 3], [2, 4],        # 面部
    [5, 6], [5, 7], [7, 9], [6, 8], [8, 10],  # 上肢
    [11, 12], [11, 13], [13, 15], [12, 14], [14, 16]  # 下肢
]

def build_video_map(root_dir):
    """
    递归遍历 root_dir，构建 {视频名（不含扩展名）: 完整路径} 的映射
    """
    video_map = {}
    for dirpath, _, filenames in os.walk(root_dir):
        for f in filenames:
            if f.lower().endswith('.mp4'):
                name = os.path.splitext(f)[0]
                full_path = os.path.join(dirpath, f)
                if name in video_map:
                    print(f"警告：发现重复的视频名 {name}，将使用第一个找到的路径")
                else:
                    video_map[name] = full_path
    return video_map

def draw_skeleton(frame, keypoints, scores, threshold=0.3):
    """
    在帧上绘制关键点（圆圈）和骨架（线段）
    """
    h, w = frame.shape[:2]
    pts = []
    for i, (x, y) in enumerate(keypoints):
        if scores[i] > threshold:
            # cx, cy = int(x * w), int(y * h)  # 归一化
            cx, cy = int(x), int(y)     
            pts.append((cx, cy))
            cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)
            # 可选：显示关键点索引（调试用）
            # cv2.putText(frame, str(i), (cx+5, cy-5),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
        else:
            pts.append(None)

    for (i, j) in SKELETON:
        if pts[i] is not None and pts[j] is not None:
            cv2.line(frame, pts[i], pts[j], (0, 255, 0), 2)

def process_split(split_name, frame_dirs, anno_dict, video_map, output_base):
    """
    处理一个 split（train 或 val）的所有视频
    """
    out_dir = os.path.join(output_base, split_name)
    os.makedirs(out_dir, exist_ok=True)

    for frame_dir in frame_dirs:
        print(f"处理 [{split_name}] {frame_dir}")

        # 从映射中获取视频路径
        video_path = video_map.get(frame_dir)
        if video_path is None:
            print(f"  警告：未找到视频 {frame_dir}.mp4，跳过")
            continue

        # 获取对应的标注
        anno = anno_dict.get(frame_dir)
        if anno is None:
            print(f"  警告：pkl 中无对应标注 {frame_dir}，跳过")
            continue

        # 提取关键点数据，去除可能的 batch 维度 (1, T, 17, 2) -> (T, 17, 2)
        keypoints = np.array(anno['keypoint'])
        scores    = np.array(anno['keypoint_score'])
        if keypoints.ndim == 4 and keypoints.shape[0] == 1:
            keypoints = keypoints[0]
        if scores.ndim == 3 and scores.shape[0] == 1:
            scores = scores[0]
        T = keypoints.shape[0]

        # 打开视频
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"  无法打开视频: {video_path}")
            continue

        fps = cap.get(cv2.CAP_PROP_FPS)
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # 输出视频路径
        out_path = os.path.join(out_dir, frame_dir + '_vis.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx < T:
                kps = keypoints[frame_idx]   # (17,2)
                scs = scores[frame_idx]      # (17,)
                draw_skeleton(frame, kps, scs)
            out.write(frame)
            frame_idx += 1

        cap.release()
        out.release()
        print(f"  已生成: {out_path}")

def concat_videos(input_dir, output_path, split_name):
    """
    将 input_dir 下所有以 '_vis.mp4' 结尾的视频合并为一个视频，
    并在每个小视频的每一帧左上角显示该小视频的名称（去掉 '_vis' 后缀）。
    所有视频会被缩放至统一尺寸（以第一个视频的尺寸为准）。
    """
    # 获取所有待合并的视频文件，并按文件名排序
    video_files = [f for f in os.listdir(input_dir) if f.endswith('_vis.mp4')]
    if not video_files:
        print(f"警告：{input_dir} 中没有找到 *_vis.mp4 文件，跳过合并 {split_name}")
        return
    video_files.sort()

    print(f"开始合并 {split_name} 集视频，共 {len(video_files)} 个文件")

    # 打开第一个视频，获取其参数作为输出视频的基准
    first_video = os.path.join(input_dir, video_files[0])
    cap = cv2.VideoCapture(first_video)
    if not cap.isOpened():
        print(f"错误：无法打开 {first_video}，合并终止")
        return
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    # 创建 VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # 逐个处理视频
    for vfile in video_files:
        video_name = vfile.replace('_vis.mp4', '')  # 用于显示的名称
        vpath = os.path.join(input_dir, vfile)
        cap = cv2.VideoCapture(vpath)
        if not cap.isOpened():
            print(f"  警告：无法打开 {vfile}，跳过")
            continue

        print(f"  正在合并 {vfile} ...")
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 如果当前帧尺寸与基准不同，则缩放
            if frame.shape[1] != width or frame.shape[0] != height:
                frame = cv2.resize(frame, (width, height))

            # 在左上角绘制视频名称（使用红色、稍大的字体）
            cv2.putText(frame, video_name, (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
            out.write(frame)

        cap.release()

    out.release()
    print(f"合并完成：{output_path}")

def main():
    # 1. 加载 pkl
    print("正在加载 pkl 文件...")
    with open(PKL_PATH, 'rb') as f:
        data = pickle.load(f)

    # 2. 提取 split 和 annotations
    split = data.get('split', {})
    annotations = data.get('annotations', [])
    if not split or not annotations:
        print("错误：pkl 中缺少 split 或 annotations 数据")
        return

    # 3. 构建 frame_dir -> annotation 映射
    anno_dict = {anno['frame_dir']: anno for anno in annotations}

    # 4. 构建视频文件映射（递归搜索 VIDEO_SRC_DIR）
    print("正在搜索视频文件...")
    video_map = build_video_map(VIDEO_SRC_DIR)
    print(f"找到 {len(video_map)} 个视频文件")

    # 5. 处理 train 和 val 集
    for split_name in ['train', 'val']:
        frame_list = split.get(split_name, [])
        if frame_list:
            process_split(split_name, frame_list, anno_dict,
                          video_map, OUTPUT_VIS_DIR)
        else:
            print(f"警告：split '{split_name}' 为空")

    # 6. 根据配置合并视频
    if CONCAT_TRAIN:
        train_dir = os.path.join(OUTPUT_VIS_DIR, 'train')
        train_out = os.path.join(OUTPUT_VIS_DIR, 'train_syn.mp4')
        concat_videos(train_dir, train_out, 'train')

    if CONCAT_VAL:
        val_dir = os.path.join(OUTPUT_VIS_DIR, 'val')
        val_out = os.path.join(OUTPUT_VIS_DIR, 'val_syn.mp4')
        concat_videos(val_dir, val_out, 'val')

    print("\n所有可视化视频已生成！")

if __name__ == "__main__":
    main()
