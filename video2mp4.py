import os
import sys
import subprocess
import argparse
from pathlib import Path
 
# 支持的视频扩展名
VIDEO_EXTENSIONS = {
    '.avi', '.mp4', '.mkv', '.mov', '.wmv', '.flv', '.webm', '.m4v',
    '.mpg', '.mpeg', '.3gp', '.ts', '.mts', '.m2ts', '.rm', '.rmvb',
    '.asf', '.divx', '.vob', '.ogv', '.f4v', '.swf'
}
 
def check_ffmpeg():
    """检查FFmpeg是否安装"""
    try:
        subprocess.run(['ffmpeg', '-version'], stdout=subprocess.DEVNULL, 
                      stderr=subprocess.DEVNULL, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False
 
def convert_video(input_path, output_path, quality='medium', delete_original=False):
    """
    转换视频为MP4格式
    
    参数:
        input_path: 输入视频路径
        output_path: 输出视频路径
        quality: 质量设置 (low, medium, high, lossless)
        delete_original: 是否删除原始文件
    """
    # 根据质量设置编码参数
    quality_presets = {
        'low': {
            'video_codec': 'libx264',
            'crf': 28,  # 较高的CRF值，更低的文件大小
            'preset': 'faster',
            'audio_bitrate': '96k'
        },
        'medium': {
            'video_codec': 'libx264',
            'crf': 23,  # 平衡质量和大小
            'preset': 'medium',
            'audio_bitrate': '128k'
        },
        'high': {
            'video_codec': 'libx264',
            'crf': 18,  # 高质量
            'preset': 'slow',
            'audio_bitrate': '192k'
        },
        'lossless': {
            'video_codec': 'libx264',
            'crf': 0,   # 无损（文件会很大）
            'preset': 'veryslow',
            'audio_bitrate': '320k'
        }
    }
    
    if quality not in quality_presets:
        quality = 'medium'
    
    preset = quality_presets[quality]
    
    # 构建FFmpeg命令
    # 关键参数说明：
    # -vsync 0: 保持原始时间戳，避免帧率问题
    # -c:v 指定视频编码器
    # -crf: 恒定质量因子（0-51，越低质量越好）
    # -preset: 编码速度/压缩率平衡
    # -pix_fmt yuv420p: 确保兼容性
    # -c:a aac: 使用AAC音频编码
    cmd = [
        'ffmpeg',
        '-i', input_path,
        '-c:v', preset['video_codec'],
        '-crf', str(preset['crf']),
        '-preset', preset['preset'],
        '-pix_fmt', 'yuv420p',  # 确保兼容所有设备
        '-vsync', '0',  # 保持原始时间戳，避免帧同步问题
        '-movflags', '+faststart',  # 使视频支持流媒体播放
        '-c:a', 'aac',
        '-b:a', preset['audio_bitrate'],
        '-y',  # 覆盖已存在的文件
        output_path
    ]
    
    try:
        print(f"正在转换: {input_path}")
        print(f"输出到: {output_path}")
        
        # 运行FFmpeg命令
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        if delete_original and os.path.exists(output_path):
            # 验证转换后的文件大小不为0
            if os.path.getsize(output_path) > 0:
                os.remove(input_path)
                print(f"已删除原始文件: {input_path}")
        
        print(f"✓ 转换完成: {input_path}")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"✗ 转换失败: {input_path}")
        print(f"错误信息: {e.stderr}")
        return False
    except Exception as e:
        print(f"✗ 转换过程中出现错误: {input_path}")
        print(f"错误: {str(e)}")
        return False
 
def find_video_files(root_dir):
    """查找目录下的所有视频文件"""
    video_files = []
    
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            ext = Path(filename).suffix.lower()
            
            if ext in VIDEO_EXTENSIONS:
                video_files.append(filepath)
    
    return video_files
 
def batch_convert_videos(root_dir, quality='medium', delete_original=False, 
                        skip_existing=True, recursive=True):
    """
    批量转换视频文件
    
    参数:
        root_dir: 根目录路径
        quality: 视频质量
        delete_original: 是否删除原始文件
        skip_existing: 是否跳过已存在的MP4文件
        recursive: 是否递归查找子目录
    """
    # 检查FFmpeg
    if not check_ffmpeg():
        print("错误: 未找到FFmpeg。请先安装FFmpeg:")
        print("Windows: 从 https://ffmpeg.org/download.html 下载")
        print("macOS: brew install ffmpeg")
        print("Ubuntu/Debian: sudo apt install ffmpeg")
        return
    
    # 确保目录存在
    if not os.path.exists(root_dir):
        print(f"错误: 路径不存在: {root_dir}")
        return
    
    print(f"开始扫描目录: {root_dir}")
    print(f"质量设置: {quality}")
    print(f"删除原始文件: {'是' if delete_original else '否'}")
    print("-" * 50)
    
    # 查找视频文件
    if recursive:
        video_files = find_video_files(root_dir)
    else:
        # 只处理当前目录
        video_files = []
        for filename in os.listdir(root_dir):
            filepath = os.path.join(root_dir, filename)
            if os.path.isfile(filepath):
                ext = Path(filename).suffix.lower()
                if ext in VIDEO_EXTENSIONS:
                    video_files.append(filepath)
    
    if not video_files:
        print("未找到视频文件")
        return
    
    print(f"找到 {len(video_files)} 个视频文件")
    
    # 统计转换结果
    success_count = 0
    fail_count = 0
    skip_count = 0
    
    for video_file in video_files:
        # 生成输出文件名
        video_path = Path(video_file)
        output_path = video_path.with_suffix('.mp4')
        
        # 跳过已经是MP4格式的文件
        if video_path.suffix.lower() == '.mp4':
            print(f"跳过已为MP4格式的文件: {video_file}")
            skip_count += 1
            continue
        
        # 如果设置了跳过已存在的文件，检查是否存在
        if skip_existing and output_path.exists():
            print(f"跳过已存在的文件: {output_path}")
            skip_count += 1
            continue
        
        # 转换视频
        if convert_video(str(video_file), str(output_path), quality, delete_original):
            success_count += 1
        else:
            fail_count += 1
        
        print()  # 空行分隔
    
    # 输出统计结果
    print("=" * 50)
    print("转换完成!")
    print(f"成功: {success_count}")
    print(f"失败: {fail_count}")
    print(f"跳过: {skip_count}")
 
def main():
    parser = argparse.ArgumentParser(description='批量转换视频为MP4格式')
    parser.add_argument('--path', default=r'D:\zero_track\mmaction2\demo\input_video', help='要处理的目录路径')
    parser.add_argument('-q', '--quality', choices=['low', 'medium', 'high', 'lossless'],
                       default='high', help='视频质量 (默认: medium)')
    parser.add_argument('-d', '--delete', action='store_true', default=False,
                       help='转换后删除原始文件')
    parser.add_argument('-f', '--force', action='store_true', default=False,
                       help='强制覆盖已存在的MP4文件')
    parser.add_argument('-r', '--no-recursive', action='store_true', default=True,
                       help='不递归处理子目录')
    
    args = parser.parse_args()
    
    # 调用批量转换函数
    batch_convert_videos(
        root_dir=args.path,
        quality=args.quality,
        delete_original=args.delete,
        skip_existing=not args.force,
        recursive=not args.no_recursive
    )
 
if __name__ == "__main__":
    main()
 
    '''
    # 以下代码注释的原因是我们直接在上面parser.add_argument改default的内容就可以了，不需要那么多交互，交互会导致每次运行都要输入一堆参数，还不方便调试
    # 如果没有命令行参数，提示用户输入
    if len(sys.argv) > 1:
        main()
    else:
        print("视频转MP4批量处理工具")
        print("=" * 50)
        
        # 交互式输入路径
        path = input("请输入要处理的目录路径: ").strip()
        
        if not path:
            print("错误: 请输入有效的路径")
            sys.exit(1)
        
        print("\n请选择视频质量:")
        print("1. 低质量 (文件小，适合快速分享)")
        print("2. 中等质量 (平衡质量和大小，推荐)")
        print("3. 高质量 (文件较大，画质好)")
        print("4. 无损质量 (文件非常大，适合存档)")
        
        quality_map = {'1': 'low', '2': 'medium', '3': 'high', '4': 'lossless'}
        quality_choice = input("请选择 (1-4，默认2): ").strip() or '2'
        quality = quality_map.get(quality_choice, 'medium')
        
        delete_choice = input("转换后删除原始文件? (y/N): ").strip().lower()
        delete_original = delete_choice == 'y'
        
        force_choice = input("覆盖已存在的MP4文件? (y/N): ").strip().lower()
        force = force_choice == 'y'
        
        recursive_choice = input("递归处理子目录? (Y/n): ").strip().lower()
        recursive = recursive_choice != 'n'
        
        print("\n" + "=" * 50)
        
        # 调用批量转换函数
        batch_convert_videos(
            root_dir=path,
            quality=quality,
            delete_original=delete_original,
            skip_existing=not force,
            recursive=recursive
        )
    '''
