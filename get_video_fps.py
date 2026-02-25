import cv2
import sys
import argparse
 
def get_video_fps(video_path):
    """
    获取视频的帧率（FPS）
    
    参数:
        video_path (str): 视频文件的路径
        
    返回:
        fps (float): 视频的帧率，如果失败则返回None
    """
    try:
        # 打开视频文件
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"错误：无法打开视频文件 '{video_path}'")
            return None
        
        # 获取帧率
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # 释放视频对象
        cap.release()
        
        return fps
        
    except Exception as e:
        print(f"读取视频时发生错误: {str(e)}")
        return None
 
 
def main():
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='获取视频文件的帧率 (FPS)')
    parser.add_argument('--video_path', default=r'D:\zero_track\mmaction2\tools\data\skeleton\S001C001P001R001A001_rgb.avi', help='视频文件的路径')
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 获取帧率
    fps = get_video_fps(args.video_path)
    
    if fps is not None:
        print(f"视频帧率: {fps:.2f} FPS")
        print(f"每帧时长: {1000/fps:.2f} ms")
    else:
        sys.exit(1)
 
 
if __name__ == "__main__":
    main()
