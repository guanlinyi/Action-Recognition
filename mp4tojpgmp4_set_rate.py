#!/usr/bin/env python3
import cv2
import os
import argparse
import numpy as np
 
# ---------- 可自定义 ----------
VIDEO_EXTS = ('.mp4', '.avi', '.mov', '.mkv')   # 支持的视频扩展名
# ----------------------------
 
def interpolate_frame(frame1, frame2, alpha):
    """线性插值生成中间帧"""
    return cv2.addWeighted(frame1, 1-alpha, frame2, alpha, 0)
 
def extract_frames_with_fps(video_path, target_fps, gen_video=False, gen_images=True):
    """把单个视频拆帧成图片文件夹，支持指定输出帧率"""
    base_dir = os.path.dirname(video_path)
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    
    # 初始化变量
    output_dir = None
    video_writer = None
    
    # 准备图片输出目录
    if gen_images:
        output_dir = os.path.join(base_dir, f"{video_name}_{target_fps}fps")
        os.makedirs(output_dir, exist_ok=True)
 
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f'[WARN] 无法打开视频: {video_path}')
        return 0
 
    # 获取视频属性
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / original_fps
    
    # 计算目标帧数
    target_frame_count = int(duration * target_fps)
    
    # 根据生成选项显示不同信息
    if gen_images and gen_video:
        print(f'[INFO] 视频: {video_name}')
        print(f'     时长: {duration:.2f}s, 原始帧率: {original_fps:.2f}fps, 目标帧率: {target_fps}fps')
        print(f'     将生成 {target_frame_count} 张图片 + MP4视频')
    elif gen_images:
        print(f'[INFO] 视频: {video_name}')
        print(f'     时长: {duration:.2f}s, 原始帧率: {original_fps:.2f}fps, 目标帧率: {target_fps}fps')
        print(f'     将生成 {target_frame_count} 张图片')
    elif gen_video:
        print(f'[INFO] 视频: {video_name}')
        print(f'     时长: {duration:.2f}s, 原始帧率: {original_fps:.2f}fps, 目标帧率: {target_fps}fps')
        print(f'     将生成 {target_frame_count} 帧的MP4视频')
 
    # 准备视频写入器（如果需要生成视频）
    if gen_video:
        # 获取第一帧的尺寸来初始化视频写入器
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, first_frame = cap.read()
        if ret:
            height, width = first_frame.shape[:2]
            video_output_path = os.path.join(base_dir, f"{video_name}_{target_fps}fps.mp4")
            
            # 使用MP4V编码器（兼容性较好）
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(
                video_output_path, 
                fourcc, 
                target_fps, 
                (width, height)
            )
            print(f'[INFO] 视频文件将保存为: {video_output_path}')
        else:
            print(f'[WARN] 无法读取第一帧，跳过生成视频')
            gen_video = False
 
    # 按目标帧率计算采样时间点
    frame_positions = np.linspace(0, total_frames - 1, target_frame_count)
    
    frame_idx_output = 0
    for pos in frame_positions:
        # 定位帧位置
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        ret, frame = cap.read()
        
        if not ret:
            # 如果读取失败，尝试插值
            prev_pos = int(np.floor(pos))
            next_pos = min(prev_pos + 1, total_frames - 1)
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, prev_pos)
            ret1, frame1 = cap.read()
            cap.set(cv2.CAP_PROP_POS_FRAMES, next_pos)
            ret2, frame2 = cap.read()
            
            if ret1 and ret2:
                alpha = pos - prev_pos
                frame = interpolate_frame(frame1, frame2, alpha)
            else:
                print(f'[WARN] 无法读取帧位置: {pos}')
                continue
        
        # 保存图片（如果需要）
        if gen_images and output_dir is not None:
            frame_path = os.path.join(output_dir, f'{frame_idx_output:06d}.jpg')
            cv2.imwrite(frame_path, frame)
        
        # 写入视频帧（如果需要）
        if gen_video and video_writer is not None:
            video_writer.write(frame)
            
        frame_idx_output += 1
 
    cap.release()
    
    # 释放视频写入器
    if gen_video and video_writer is not None:
        video_writer.release()
        print(f'[INFO] 视频文件已保存')
    
    # 输出总结信息
    if gen_images:
        print(f'[INFO] 图片提取完成: {frame_idx_output} 张 -> {output_dir}')
    if gen_video:
        print(f'[INFO] 视频生成完成: {frame_idx_output} 帧, {target_fps}fps')
    
    return frame_idx_output
 
def walk_and_extract(root_dir, target_fps, gen_video=False, gen_images=True):
    """递归遍历目录，对所有视频拆帧"""
    count = 0
    for dirpath, _, filenames in os.walk(root_dir):
        for file in filenames:
            if file.lower().endswith(VIDEO_EXTS):
                video_path = os.path.join(dirpath, file)
                extract_frames_with_fps(video_path, target_fps, gen_video, gen_images)
                count += 1
    print(f'[INFO] 批量提取完成，共处理 {count} 个视频')
 
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='视频逐帧提取为图片（支持指定输出帧率）')
    parser.add_argument('--input', help='单个视频路径')
    parser.add_argument('--dir', help='批量提取目录')
    parser.add_argument('--fps', type=float, default=30, help='目标输出帧率 (默认: 30)')
    parser.add_argument('--gen_video', action='store_true', default=True,
                       help='是否生成MP4视频文件 (默认: 是)')
    parser.add_argument('--no_images', action='store_true', default=False,
                       help='是否不生成图片文件夹 (默认: 生成图片)')
    
    args = parser.parse_args()
 
    # 缺省行为：如果两个都没给，就默认用指定目录
    if args.input is None and args.dir is None:
        # 修改这里为你的视频文件夹目录
        args.dir = r'D:\zero_track\mmaction2\input_videos'
 
    # 互斥检查
    if args.input is not None and args.dir is not None:
        parser.error('不能同时指定 --input 和 --dir，请二选一')
 
    # 检查帧率有效性
    if args.fps <= 0:
        parser.error('帧率必须大于0')
        
    # 检查至少选择一种输出格式
    if args.no_images and not args.gen_video:
        parser.error('必须至少选择一种输出格式（图片或视频），请指定 --gen_video 或移除 --no_images')
 
    # 计算是否生成图片
    gen_images = not args.no_images
 
    # 执行
    if args.input:
        extract_frames_with_fps(args.input, args.fps, args.gen_video, gen_images)
    else:
        walk_and_extract(args.dir, args.fps, args.gen_video, gen_images)
