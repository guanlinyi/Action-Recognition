#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# generate_ntu60_pkl.py
 
"""
生成 MMAction2 风格的关键点动作识别数据集 pkl 文件
用法：
    python generate_pkl.py --root /path/to/annotations_root --output my_ntu60_2d.pkl
"""
 
import os
import json
import pickle
import numpy as np
import argparse
from glob import glob
from sklearn.model_selection import train_test_split
 
def find_all_annotations(root_path):
    """递归查找 root_path 下所有的 annotations.json 文件"""
    pattern = os.path.join(root_path, "**", "annotations.json")
    return glob(pattern, recursive=True)
 
def load_and_merge_samples(json_files):
    """加载多个 annotations.json 文件，合并所有 samples 到一个列表"""
    all_samples = []
    for f in json_files:
        try:
            with open(f, 'r', encoding='utf-8') as fp:
                data = json.load(fp)
                samples = data.get('samples', [])
                print(f"  {f}: {len(samples)} 个样本")
                all_samples.extend(samples)
        except Exception as e:
            print(f"  警告: 无法加载 {f} - {e}")
    return all_samples
 
def convert_to_numpy(sample):
    """
    将 sample 中的 keypoint 和 keypoint_score 从 list 转为 numpy 数组，
    并 reshape 为 (1, T, 17, 2) 和 (1, T, 17)
    """
    kp = np.array(sample['keypoint'], dtype=np.float32)          # (T, 17, 2)
    score = np.array(sample['keypoint_score'], dtype=np.float32) # (T, 17)
    T = kp.shape[0]
 
    # 添加人物维度 (1, T, 17, 2)
    kp = kp.reshape(1, T, 17, 2)
    score = score.reshape(1, T, 17)
 
    return {
        'frame_dir': sample['frame_dir'],
        'label': sample['label'],
        'img_shape': tuple(sample['img_shape']),
        'original_shape': tuple(sample['original_shape']),
        'total_frames': sample['total_frames'],
        'keypoint': kp,
        'keypoint_score': score
    }
 
def main():
    parser = argparse.ArgumentParser(description='生成动作识别 pkl 文件')
    input_path = r"D:\zero_track\mmaction2\my_code\input"
    parser.add_argument('--root', type=str, default=input_path, 
                        help='标注工具根目录（会递归查找所有 annotations.json）')
    parser.add_argument('--output', type=str, default=input_path + os.sep + 'my_ntu60_2d.pkl', help='输出 pkl 文件名')
    parser.add_argument('--test-size', type=float, default=0.2,
                        help='验证集比例，默认 0.2')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    args = parser.parse_args()
 
    print(f"查找根目录下的 annotations.json: {args.root}")
    json_files = find_all_annotations(args.root)
    if not json_files:
        print("错误: 未找到任何 annotations.json 文件")
        return
 
    print(f"找到 {len(json_files)} 个 annotations.json 文件，开始合并...")
    samples = load_and_merge_samples(json_files)
    print(f"共合并 {len(samples)} 个样本")
 
    if len(samples) == 0:
        print("错误: 没有样本数据")
        return
 
    # 提取所有 frame_dir 用于划分
    frame_dirs = [s['frame_dir'] for s in samples]
 
    # 随机划分训练集/验证集
    train_names, val_names = train_test_split(
        frame_dirs,
        test_size=args.test_size,
        random_state=args.seed
    )
    print(f"训练集样本数: {len(train_names)}")
    print(f"验证集样本数: {len(val_names)}")
 
    # 构建 split 字典
    split = {
        'train': train_names,
        'val': val_names
    }
 
    # 转换所有样本为 numpy 格式
    annotations = []
    for s in samples:
        try:
            ann = convert_to_numpy(s)
            annotations.append(ann)
        except Exception as e:
            print(f"  跳过样本 {s.get('frame_dir', 'unknown')}: {e}")
 
    # 最终数据结构
    output_data = {
        'split': split,
        'annotations': annotations
    }
 
    # 保存 pkl
    with open(args.output, 'wb') as f:
        pickle.dump(output_data, f)
 
    print(f"\n成功生成 pkl 文件: {args.output}")
    print(f"  split 包含训练集 {len(train_names)} 个, 验证集 {len(val_names)} 个")
    print(f"  annotations 包含 {len(annotations)} 个样本")
 
if __name__ == '__main__':
    main()
