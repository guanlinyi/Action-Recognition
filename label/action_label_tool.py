import os
import pygame
import sys
import shutil
import time
import json
import subprocess
import numpy as np
from pygame.locals import *
import cv2
 
ROOT_PATH = r"D:\zero_track\mmaction2\my_code\input\fisheye_back_2026-01-29-14-13-15_30fps"
 
# 初始化pygame
pygame.init()
 
# 配置参数
SCREEN_WIDTH, SCREEN_HEIGHT = pygame.display.Info().current_w, pygame.display.Info().current_h   
WINDOW_WIDTH, WINDOW_HEIGHT = SCREEN_WIDTH - 100, SCREEN_HEIGHT - 100
BG_COLOR = (40, 44, 52)
TEXT_COLOR = (220, 220, 220)
HIGHLIGHT_COLOR = (97, 175, 239)
BUTTON_COLOR = (56, 58, 66)
BUTTON_HOVER_COLOR = (72, 74, 82)
WARNING_COLOR = (255, 152, 0)
CONFIRM_COLOR = (76, 175, 80)
BOX_COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), 
              (255, 0, 255), (0, 255, 255), (255, 128, 0), (128, 255, 0)]
 
# 创建窗口
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("图像分类标注工具 - YOLO-Pose 多人标注")
 
# 字体
font = pygame.font.SysFont("SimHei", 24)
small_font = pygame.font.SysFont("SimHei", 18)
 
def run_yolo_pose_inference(image_folder, output_json_path, device='cuda', conf_threshold=0.5):
    """
    运行YOLO-pose推理，逐帧处理并使用跟踪模式保持ID一致性
    返回：处理后的图片列表和推理结果
    """
    print(f"开始对文件夹 {image_folder} 进行YOLO-pose逐帧跟踪推理...")
    print(f"设备: {device}, 置信度阈值: {conf_threshold}")
    
    # 定义检查连续性的辅助函数
    def check_continuity(frame_numbers):
        """检查帧号是否连续"""
        if not frame_numbers or len(frame_numbers) <= 1:
            return True
        
        sorted_frames = sorted(frame_numbers)
        for i in range(len(sorted_frames) - 1):
            if sorted_frames[i] + 1 != sorted_frames[i + 1]:
                return False
        return True
    
    try:
        from ultralytics import YOLO
        import torch
        
        # 检查CUDA是否可用
        if device == 'cuda' and not torch.cuda.is_available():
            print("CUDA不可用，将使用CPU进行推理")
            device = 'cpu'
        
        # 加载YOLO-pose模型
        print("正在加载YOLO-pose模型...")
        model_path = "checkpoints/yolo26s-pose.pt"  # 请确保模型路径正确
        
        # 检查模型文件是否存在
        if not os.path.exists(model_path):
            print(f"警告: 模型文件不存在: {model_path}")
            # 尝试使用默认的yolo-pose模型
            model_path = 'yolov8s-pose.pt'
            print(f"尝试使用默认模型: {model_path}")
        
        model = YOLO(model_path)
        model.to(device)
        
        # 获取文件夹中所有图片并排序
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
        image_files = []
        for f in os.listdir(image_folder):
            if f.lower().endswith(image_extensions):
                image_files.append(f)
        image_files.sort()  # 确保按文件名顺序处理
        
        if not image_files:
            print(f"文件夹 {image_folder} 中没有找到图片")
            return [], {}
        
        inference_results = {}
        valid_images = []
        deleted_count = 0
        
        print(f"开始逐帧处理 {len(image_files)} 张图片...")
        
        # 逐帧处理，使用persist=True保持跟踪状态
        for i, img_file in enumerate(image_files):
            if (i + 1) % 10 == 0 or (i + 1) == len(image_files):
                print(f"处理进度: {i+1}/{len(image_files)}...")
            
            img_path = os.path.join(image_folder, img_file)
            
            try:
                # 使用track模式，persist=True保持跟踪状态
                # 这是关键：逐帧处理，但persist=True会保持跟踪器状态
                if i == 0:
                    # 第一帧，开始新的跟踪
                    results = model.track(
                        source=img_path,
                        conf=conf_threshold,
                        persist=True,  # 开始跟踪并保持状态
                        verbose=False,
                        tracker="bytetrack.yaml" if i == 0 else None  # 第一帧初始化跟踪器
                    )
                else:
                    # 后续帧，继续之前的跟踪
                    results = model.track(
                        source=img_path,
                        conf=conf_threshold,
                        persist=True,  # 继续跟踪，使用之前的状态
                        verbose=False
                    )
                
                if not results or len(results) == 0:
                    # 没有检测结果，删除图片
                    os.remove(img_path)
                    deleted_count += 1
                    continue
                
                result = results[0]  # 单张图片，只有一个结果
                
                # 获取关键点和边界框
                keypoints = result.keypoints
                boxes = result.boxes
                
                # 检查是否检测到人物
                if (keypoints is not None and len(keypoints) > 0 and 
                    boxes is not None and len(boxes) > 0):
                    
                    persons = []
                    
                    # 获取跟踪ID（如果有）
                    track_ids = None
                    if hasattr(boxes, 'id') and boxes.id is not None:
                        track_ids = boxes.id.cpu().numpy().astype(int)
                    
                    # 处理每个检测到的人物
                    for person_idx in range(len(keypoints)):
                        kpt_data = keypoints.data[person_idx].cpu().numpy()
                        box_data = boxes.data[person_idx].cpu().numpy()
                        
                        # 过滤低置信度的检测
                        if box_data[5] < conf_threshold:
                            continue
                        
                        # 获取边界框
                        bbox = box_data[:4].tolist()
                        
                        # 获取跟踪ID
                        if track_ids is not None and person_idx < len(track_ids):
                            person_id = int(track_ids[person_idx])
                        else:
                            # 如果没有跟踪ID，使用索引
                            person_id = person_idx
                        
                        # 处理关键点数据
                        keypoints_list = []
                        for point in kpt_data:
                            x, y, v = float(point[0]), float(point[1]), float(point[2])
                            # 只保留置信度足够高的关键点
                            if v > 0.3:  # 关键点可见性阈值
                                keypoints_list.extend([x, y, v])
                            else:
                                keypoints_list.extend([0.0, 0.0, 0.0])
                        
                        person_info = {
                            "class_id": int(boxes.cls[person_idx].item()),
                            "id": person_id,
                            "bbox": bbox,
                            "keypoints": keypoints_list,
                            "confidence": float(box_data[5]),
                        }
                        persons.append(person_info)
                    
                    if persons:
                        # 按ID排序，便于查看
                        persons.sort(key=lambda x: x["id"])
                        inference_results[img_file] = persons
                        valid_images.append(img_path)
                        
                        # 可选：显示当前帧的跟踪信息
                        if (i + 1) % 20 == 0:
                            current_ids = [p["id"] for p in persons]
                            print(f"  帧 {i+1}: 检测到 {len(persons)} 人, IDs: {sorted(current_ids)}")
                    else:
                        # 没有检测到有效人物，删除图片
                        os.remove(img_path)
                        deleted_count += 1
                else:
                    # 没有检测到人物，删除图片
                    os.remove(img_path)
                    deleted_count += 1
                    
            except torch.cuda.OutOfMemoryError:
                print(f"警告: 处理 {img_file} 时显存不足，跳过此图片")
                # 清理CUDA缓存
                torch.cuda.empty_cache()
                # 保留图片，不删除
                valid_images.append(img_path)
                continue
            except Exception as e:
                print(f"处理图片 {img_file} 时出错: {e}")
                # 出错时保留图片，不删除
                valid_images.append(img_path)
        
        # 推理完成后清理显存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print(f"YOLO-pose逐帧跟踪推理完成。")
        print(f"原始图片数: {len(image_files)}")
        print(f"保留有人的图片: {len(valid_images)}")
        print(f"删除无人的图片: {deleted_count}")
        
        # 分析ID变化情况
        if inference_results:
            all_ids = set()
            id_transitions = {}
            
            # 收集所有ID
            for img_file, persons in inference_results.items():
                for person in persons:
                    all_ids.add(person["id"])
            
            # 分析每个ID的出现情况
            for person_id in sorted(all_ids):
                frames_with_id = []
                for img_file, persons in inference_results.items():
                    if any(p["id"] == person_id for p in persons):
                        frames_with_id.append(img_file)
                
                if frames_with_id:
                    # 提取帧号（假设文件名中有数字）
                    frame_numbers = []
                    for f in frames_with_id:
                        import re
                        numbers = re.findall(r'\d+', f)
                        if numbers:
                            frame_numbers.append(int(numbers[-1]))
                        else:
                            # 如果没有数字，使用索引
                            frame_numbers.append(frames_with_id.index(f) + 1)
                    
                    if len(frame_numbers) > 1:
                        id_transitions[person_id] = {
                            "start_frame": min(frame_numbers),
                            "end_frame": max(frame_numbers),
                            "frame_count": len(frame_numbers),
                            "frames": sorted(frame_numbers)
                        }
            
            print(f"检测到 {len(all_ids)} 个不同的人物ID")
            print("ID追踪情况:")
            for person_id, info in sorted(id_transitions.items()):
                continuity = "连续" if check_continuity(info["frames"]) else "不连续"
                print(f"  ID {person_id}: 出现在 {info['frame_count']} 帧中 ({continuity}), "
                      f"从帧 {info['start_frame']} 到 {info['end_frame']}")
        
        # 保存推理结果到JSON
        if inference_results:
            # 转换为可序列化的格式
            json_data = {}
            for img_file, persons in inference_results.items():
                json_persons = []
                for person in persons:
                    # 转换numpy类型为Python原生类型
                    json_person = {
                        "id": int(person["id"]),
                        "bbox": [float(coord) for coord in person["bbox"]],
                        "keypoints": [float(val) for val in person["keypoints"]],
                        "confidence": float(person["confidence"]),
                        "class_id": int(person["class_id"])
                    }
                    json_persons.append(json_person)
                json_data[img_file] = json_persons
            
            # 确保输出目录存在
            os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
            
            with open(output_json_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=2, ensure_ascii=False)
            
            print(f"推理结果已保存到: {output_json_path}")
            
            # 保存统计信息
            stats = {
                "total_images": len(image_files),
                "valid_images": len(valid_images),
                "deleted_images": deleted_count,
                "detected_persons_total": sum(len(persons) for persons in inference_results.values()),
                "unique_person_ids": len(all_ids) if inference_results else 0,
                "tracking_mode": True,
                "id_transitions": id_transitions
            }
            
            stats_path = output_json_path.replace(".json", "_stats.json")
            with open(stats_path, 'w', encoding='utf-8') as f:
                json.dump(stats, f, indent=2, ensure_ascii=False)
            print(f"统计信息已保存到: {stats_path}")
        else:
            print("警告: 没有检测到任何人物")
        
        return valid_images, inference_results
        
    except ImportError as e:
        print(f"无法导入ultralytics库: {e}")
        print("请安装: pip install ultralytics")
        print("正在使用备用方案...")
        
        # 备用方案：返回所有图片，不进行推理
        all_images = [os.path.join(image_folder, f) for f in os.listdir(image_folder) 
                     if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        all_images.sort()
        return all_images, {}
        
    except Exception as e:
        print(f"YOLO-pose推理失败: {e}")
        import traceback
        traceback.print_exc()
        
        # 出错时返回所有图片作为备用
        all_images = [os.path.join(image_folder, f) for f in os.listdir(image_folder) 
                     if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        all_images.sort()
        return all_images, {}
 
class ImageLabelingTool:
    def __init__(self, root_path):
        self.root_path = root_path
        self.folders = []               # 所有含图片的文件夹绝对路径
        self.current_folder_index = 0   # 当前文件夹索引
        self.images = []                # 当前文件夹内所有图片绝对路径
        self.current_image_index = 0    # 当前图片索引
        self.labels = {}                # 路径 -> {person_id: 'positive'/'negative'}
        self.inference_data = {}        # 图片名称 -> [人物列表]
        self.selected_person_id = 0     # 当前选定的人物ID
        self.person_id_list = []        # 当前图片中所有人物ID列表
        
        # 显示设置
        self.show_bboxes = True         # 是否显示边界框
        self.show_keypoints = True      # 是否显示关键点
        
        self.convert_to_video = True    # 是否启用转视频模式
        self.video_fps = 30             # 视频帧率
        
        # 自动播放相关
        self.playing = False
        self.play_direction = 1
        self.last_play_tick = 0
        self.play_interval = 100
        
        # 标记状态
        self.continuous_mode = False
        self.continuous_label = None
        self.continuous_start_index = None
        
        # 键盘长按状态
        self.key_pressed = {"left": False, "right": False}
        self.last_key_time = 0
        self.key_repeat_delay = 0.8
        self.key_repeat_interval = 0.15
        
        # 操作历史
        self.undo_stack = []
        self.max_undo_steps = 50
        
        # 不再删除原图，改为移动到ok文件夹
        self.discard_before_last_action = False  # 禁用删除原图
        
        # 确认对话框状态
        self.show_confirm_dialog = False
        self.confirm_message = ""
        self.confirm_action = ""
        
        # 获取所有包含图片的文件夹
        self.find_image_folders()
        
        # 加载当前文件夹的图片和推理结果
        if self.folders:
            self.load_current_folder_images()
        
        # 加载保存的标记状态
        self.load_labels()
 
        # 新增：annotations 文件路径及数据存储
        self.annotations_path = os.path.join(self.root_path, "annotations.json")
        self.annotations_data = {"samples": []}
        self.load_annotations()
 
    def cancel_current_person_label(self):
        """取消当前图片中选定人物的标记"""
        current_image = self.get_current_image()
        if current_image and current_image in self.labels:
            if self.selected_person_id in self.labels[current_image]:
                self.save_state()  # 保存状态以便撤销
                del self.labels[current_image][self.selected_person_id]
                self.save_labels()
                return True
        return False
 
    def images_to_video(self, image_paths, output_path, fps=10):
        """将图片序列转为视频"""
        if not image_paths:
            return False
        
        print(f"正在合成视频: {output_path}")
        print(f"图片数量: {len(image_paths)}，帧率: {fps}")
        
        try:
            # 读取第一张图获取尺寸
            first_image_path = image_paths[0]
            if not os.path.exists(first_image_path):
                print(f"第一张图片不存在: {first_image_path}")
                return False
            
            frame = cv2.imread(first_image_path)
            if frame is None:
                print(f"无法读取第一张图片: {first_image_path}")
                return False
            
            h, w, _ = frame.shape
            
            # 初始化视频写入器
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
            
            if not out.isOpened():
                print(f"无法创建视频写入器: {output_path}")
                return False
            
            # 逐帧写入图片
            success_count = 0
            for i, img_path in enumerate(image_paths):
                if not os.path.exists(img_path):
                    print(f"警告: 图片不存在，跳过: {img_path}")
                    continue
                
                frame = cv2.imread(img_path)
                if frame is None:
                    print(f"警告: 无法读取图片，跳过: {img_path}")
                    continue
                
                # 如果图片尺寸与第一张不同，调整尺寸
                if frame.shape[0] != h or frame.shape[1] != w:
                    print(f"警告: 图片尺寸不匹配，调整尺寸: {img_path}")
                    frame = cv2.resize(frame, (w, h))
                
                out.write(frame)
                success_count += 1
                
                if (i + 1) % 50 == 0:
                    print(f"  已处理 {i + 1}/{len(image_paths)} 张图片")
            
            out.release()
            cv2.destroyAllWindows()
            
            if success_count > 0:
                print(f"视频合成成功: {output_path} ({success_count} 张图片)")
                return True
            else:
                print(f"视频合成失败: 没有成功处理的图片")
                if os.path.exists(output_path):
                    os.remove(output_path)
                return False
                
        except Exception as e:
            print(f"视频合成失败: {e}")
            import traceback
            traceback.print_exc()
            
            # 清理未完成的视频文件
            if os.path.exists(output_path):
                try:
                    os.remove(output_path)
                except:
                    pass
            return False
 
    def analyze_id_tracking(self):
        """分析ID跟踪情况"""
        if not self.inference_data:
            print("没有推理数据，无法分析ID跟踪")
            return
        
        id_frames = {}  # ID -> [出现的图片列表]
        
        # 收集每个ID出现的图片
        for img_name, persons in self.inference_data.items():
            for person in persons:
                person_id = person.get("id", 0)
                if person_id not in id_frames:
                    id_frames[person_id] = []
                id_frames[person_id].append(img_name)
        
        print("ID跟踪分析:")
        print("-" * 50)
        
        # 按ID出现的频率排序
        sorted_ids = sorted(id_frames.items(), key=lambda x: len(x[1]), reverse=True)
        
        for person_id, frames in sorted_ids:
            frame_numbers = []
            for f in frames:
                # 提取帧号
                import re
                numbers = re.findall(r'\d+', f)
                if numbers:
                    frame_numbers.append(int(numbers[-1]))
            
            if frame_numbers:
                continuity = "连续" if self._check_continuity(frame_numbers) else "不连续"
                
                print(f"ID {person_id}: 出现在 {len(frames)} 帧中, {continuity}")
                print(f"      帧范围: {min(frame_numbers)} - {max(frame_numbers)}")
        
        print("-" * 50)
        print(f"总共检测到 {len(sorted_ids)} 个不同ID")
 
    def preprocess_folder_with_yolo(self, folder_path):
        """对文件夹进行YOLO-pose预处理"""
        json_path = os.path.join(folder_path, "yolo_pose_results.json")
        
        # 检查是否已有预处理结果
        if os.path.exists(json_path):
            print(f"加载已有的YOLO-pose推理结果: {json_path}")
            with open(json_path, 'r') as f:
                inference_data = json.load(f)
            
            # 获取有效的图片列表
            valid_images = []
            for img_name in os.listdir(folder_path):
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    if img_name in inference_data:
                        valid_images.append(os.path.join(folder_path, img_name))
            
            valid_images.sort()
            return valid_images, inference_data
        else:
            # 运行YOLO-pose推理
            return run_yolo_pose_inference(folder_path, json_path)
    
    def find_image_folders(self):
        """查找所有包含图片的文件夹"""
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
        for root, dirs, files in os.walk(self.root_path):
            has_images = any(file.lower().endswith(image_extensions) for file in files)
            if has_images:
                self.folders.append(root)
    
    def load_current_folder_images(self):
        """加载当前文件夹中的所有图片和YOLO-pose推理结果"""
        folder_path = self.folders[self.current_folder_index]
        
        # 预处理文件夹（YOLO-pose推理）
        self.images, self.inference_data = self.preprocess_folder_with_yolo(folder_path)
        
        # 重置指针
        self.current_image_index = 0 if self.images else -1
        self.selected_person_id = 0
        self.update_person_id_list()
    
    def update_person_id_list(self):
        """更新当前图片的人物ID列表"""
        self.person_id_list = []
        current_image = self.get_current_image()
        if current_image:
            img_name = os.path.basename(current_image)
            if img_name in self.inference_data:
                persons = self.inference_data[img_name]
                self.person_id_list = [person.get("id", i) for i, person in enumerate(persons)]
                if self.person_id_list and self.selected_person_id not in self.person_id_list:
                    self.selected_person_id = self.person_id_list[0]
    
    def get_current_image(self):
        """获取当前图片"""
        if not self.images or self.current_image_index < 0:
            return None
        return self.images[self.current_image_index]
    
    def next_image(self):
        """切换到下一张图片"""
        if self.current_image_index < len(self.images) - 1:
            self.save_state()
            self.current_image_index += 1
            self.update_person_id_list()
            return True
        return False
    
    def prev_image(self):
        """切换到上一张图片"""
        if self.current_image_index > 0:
            self.current_image_index -= 1
            self.update_person_id_list()
            return True
        return False
    
    def label_current_image(self, label):
        """标记当前图片中的选定人物"""
        current_image = self.get_current_image()
        if current_image:
            self.save_state()
        
            # 确保labels字典中有当前图片的条目
            if current_image not in self.labels:
                self.labels[current_image] = {}
        
            # 标记选定的人物
            self.labels[current_image][self.selected_person_id] = label
        
            # 自动保存标记状态
            self.save_labels()
    
    def start_continuous_labeling(self):
        """开始连续标记"""
        current_image = self.get_current_image()
        if current_image:
            self.save_state()
            
            # 如果当前图片的选定人物已经有标签，使用该标签
            if (current_image in self.labels and 
                self.selected_person_id in self.labels[current_image]):
                self.continuous_label = self.labels[current_image][self.selected_person_id]
            else:
                # 如果没有标签，默认为正样本
                self.continuous_label = "positive"
                if current_image not in self.labels:
                    self.labels[current_image] = {}
                self.labels[current_image][self.selected_person_id] = self.continuous_label
            
            self.continuous_mode = True
            self.continuous_start_index = self.current_image_index
            self.save_labels()
            return True
        return False
    
    def end_continuous_labeling(self):
        """结束连续标记"""
        if self.continuous_mode and self.continuous_start_index is not None:
            self.save_state()
            start = min(self.continuous_start_index, self.current_image_index)
            end = max(self.continuous_start_index, self.current_image_index)
            
            # 为范围内的所有图片的选定人物标记相同的标签
            for i in range(start, end + 1):
                img_path = self.images[i]
                if img_path not in self.labels:
                    self.labels[img_path] = {}
                self.labels[img_path][self.selected_person_id] = self.continuous_label
            
            self.continuous_mode = False
            self.continuous_start_index = None
            self.save_labels()
            return True
        return False
    
    def get_current_label(self):
        """获取当前图片选定人物的标签"""
        current_image = self.get_current_image()
        if current_image and current_image in self.labels:
            person_labels = self.labels[current_image]
            if isinstance(person_labels, dict):
                return person_labels.get(self.selected_person_id)
            else:
                # 旧格式，假设人物ID为0
                return person_labels if self.selected_person_id == 0 else None
        return None
 
    def get_next_unprocessed_index(self, start_idx):
        """从 start_idx 开始向后查找第一个未标记为 __processed__ 的图片索引，找不到返回 None"""
        for i in range(start_idx, len(self.images)):
            img_path = self.images[i]
            if img_path in self.labels and "__processed__" in self.labels[img_path]:
                continue
            return i
        return None
 
    def get_prev_unprocessed_index(self, start_idx):
        """从 start_idx 开始向前查找第一个未标记为 __processed__ 的图片索引，找不到返回 None"""
        for i in range(start_idx, -1, -1):
            img_path = self.images[i]
            if img_path in self.labels and "__processed__" in self.labels[img_path]:
                continue
            return i
        return None
 
    def move_labeled_files(self, positive_dir, negative_dir):
 
        """
        移动已标记的文件到 ok 文件夹，并按连续片段生成视频 + annotation
        修改后：不复制原图，仅生成视频，并在 labels 中标记已处理帧
        """
        if not self.folders:
            return "无文件夹可处理"
 
        current_folder = self.folders[self.current_folder_index]
        ok_folder = os.path.join(current_folder, "ok")
        positive_ok = os.path.join(ok_folder, "positive")
        negative_ok = os.path.join(ok_folder, "negative")
 
        # 创建输出目录（视频目录）
        if self.convert_to_video:
            for folder in [positive_ok, negative_ok]:
                os.makedirs(folder, exist_ok=True)
 
        from collections import defaultdict
        groups = defaultdict(list)   # (label, person_id) -> list of img_paths
 
        # 收集所有正负标记
        for img_path, person_labels in list(self.labels.items()):
            if not os.path.exists(img_path):
                continue
            for person_id, label in person_labels.items():
                if person_id == "__processed__":  # 跳过已处理标记
                    continue
                if label in ["positive", "negative"]:
                    groups[(label, person_id)].append(img_path)
 
        if not groups:
            return "没有找到已标记的文件"
 
        video_count = 0
        processed_frames = 0
        new_samples = []
        last_processed_frame_index = -1   # 记录最后一个被处理的帧的索引
 
        # 处理每个 (标签, 人物ID) 组
        for (label, person_id), img_paths in groups.items():
            # 按帧号排序
            img_paths.sort(key=lambda p: self.extract_frame_number(p))
 
            # 切分成连续帧段
            segments = []
            current_segment = [img_paths[0]]
            for i in range(1, len(img_paths)):
                prev_frame = self.extract_frame_number(img_paths[i-1])
                curr_frame = self.extract_frame_number(img_paths[i])
                if curr_frame == prev_frame + 1:
                    current_segment.append(img_paths[i])
                else:
                    segments.append(current_segment)
                    current_segment = [img_paths[i]]
            segments.append(current_segment)
 
            # 处理每个连续片段
            for seg_paths in segments:
                if len(seg_paths) < 2:
                    continue   # 单帧忽略，也可根据需要保留
 
                # --- 生成视频（仅当开启转视频模式）---
                video_name = self.build_video_name(seg_paths, person_id, label)
                video_path = None
                if self.convert_to_video:
                    dest_dir = positive_ok if label == "positive" else negative_ok
                    video_path = os.path.join(dest_dir, video_name + ".mp4")
                    if self.images_to_video(seg_paths, video_path, fps=self.video_fps):
                        video_count += 1
                        processed_frames += len(seg_paths)
 
                        # --- 生成 annotation ---
                        try:
                            annotation = self.build_annotation(
                                seg_paths, person_id, label, video_name
                            )
                            self.annotations_data["samples"].append(annotation)
                            new_samples.append(annotation)
                            print(f"  生成 annotation: {video_name} 共 {len(seg_paths)} 帧")
                        except Exception as e:
                            print(f"  生成 annotation 失败 {video_name}: {e}")
 
                # --- 标记已处理帧（无论是否生成视频，都标记为已处理）---
                for img_path in seg_paths:
                    # 添加 __processed__ 标记，记录视频信息
                    if img_path not in self.labels:
                        self.labels[img_path] = {}
                    if "__processed__" not in self.labels[img_path]:
                        self.labels[img_path]["__processed__"] = []
                    self.labels[img_path]["__processed__"].append({
                        "person_id": person_id,
                        "label": label,
                        "video": video_name + ".mp4" if video_path else None,
                        "timestamp": time.time()
                    })
 
                    # 删除该人物 ID 的原标记（已完成）
                    # if person_id in self.labels[img_path]:
                    #     del self.labels[img_path][person_id]
                        # 如果该图片没有其他标记且没有其他 ID 待处理，则保留空字典，不删除整个条目
                        # 也可以选择清理空字典，但保留方便下次判断
                    # 记录最后处理的帧索引，用于跳转
                    try:
                        idx = self.images.index(img_path)
                        if idx > last_processed_frame_index:
                            last_processed_frame_index = idx
                    except ValueError:
                        pass
 
        # 保存更新后的 labels 和 annotations
        self.save_labels()
        if new_samples:
            self.save_annotations()
 
        # 构造状态消息
        result_msg = f"已生成 {video_count} 个视频，处理 {processed_frames} 帧"
        if new_samples:
            result_msg += f"，新增 {len(new_samples)} 个 annotation"
        self.status_message = result_msg
        self.status_message_timer = pygame.time.get_ticks()  # 记录显示起始时间
 
        print(result_msg)
 
        # 自动跳转到最后一个被处理帧的下一帧（未处理）
        if last_processed_frame_index != -1:
            next_idx = self.get_next_unprocessed_index(last_processed_frame_index + 1)
            if next_idx is not None:
                self.current_image_index = next_idx
                self.update_person_id_list()
                print(f"已跳转到下一未处理帧: 索引 {next_idx+1}")
            else:
                # 如果没有下一帧，留在当前帧或第一帧
                print("所有帧均已处理")
 
        return result_msg
 
    def draw_detections(self, screen, img_rect, current_image_path):
        """在图片上绘制检测结果"""
        if not current_image_path or not self.show_bboxes:
            return
        
        img_name = os.path.basename(current_image_path)
        if img_name not in self.inference_data:
            return
        
        persons = self.inference_data[img_name]
        if not persons:
            return
        
        # 获取图片显示的实际尺寸和位置
        img_display_rect = img_rect
        
        # 读取图片原始尺寸
        original_img = pygame.image.load(current_image_path)
        orig_width, orig_height = original_img.get_size()
        
        # 计算缩放比例
        scale_x = img_display_rect.width / orig_width
        scale_y = img_display_rect.height / orig_height
        
        for i, person in enumerate(persons):
            person_id = person.get("id", i)
            bbox = person.get("bbox", [])
            
            if len(bbox) >= 4:
                # 缩放边界框坐标
                x1 = img_display_rect.x + bbox[0] * scale_x
                y1 = img_display_rect.y + bbox[1] * scale_y
                x2 = img_display_rect.x + bbox[2] * scale_x
                y2 = img_display_rect.y + bbox[3] * scale_y
                
                width = x2 - x1
                height = y2 - y1
                
                # 选择颜色
                color_idx = person_id % len(BOX_COLORS)
                color = BOX_COLORS[color_idx]
                
                # 绘制边界框
                if person_id == self.selected_person_id:
                    # 选中的边界框加粗
                    pygame.draw.rect(screen, color, (x1, y1, width, height), 3)
                    
                    # 在框上方显示ID
                    id_text = f"ID: {person_id}"
                    text_surface = small_font.render(id_text, True, color)
                    screen.blit(text_surface, (x1, y1 - 25))
                    
                    # 显示置信度
                    conf = person.get("confidence", 0)
                    conf_text = f"{conf:.2f}"
                    conf_surface = small_font.render(conf_text, True, color)
                    screen.blit(conf_surface, (x1, y1 - 45))
                else:
                    pygame.draw.rect(screen, color, (x1, y1, width, height), 2)
                
                # 绘制关键点
                if self.show_keypoints:
                    keypoints = person.get("keypoints", [])
                    for j in range(0, len(keypoints), 3):
                        if j + 2 < len(keypoints):
                            kx = img_display_rect.x + keypoints[j] * scale_x
                            ky = img_display_rect.y + keypoints[j+1] * scale_y
                            confidence = keypoints[j+2]
                            
                            if confidence > 0.5:  # 只显示置信度高的关键点
                                pygame.draw.circle(screen, color, (int(kx), int(ky)), 3)
    
    def select_person_by_click(self, mouse_pos, img_rect):
        """通过鼠标点击选择人物"""
        current_image = self.get_current_image()
        if not current_image:
            return False
        
        img_name = os.path.basename(current_image)
        if img_name not in self.inference_data:
            return False
        
        persons = self.inference_data[img_name]
        if not persons:
            return False
        
        # 读取图片原始尺寸
        original_img = pygame.image.load(current_image)
        orig_width, orig_height = original_img.get_size()
        
        # 计算缩放比例
        scale_x = img_rect.width / orig_width
        scale_y = img_rect.height / orig_height
        
        # 检查鼠标是否在某个边界框内
        for i, person in enumerate(persons):
            person_id = person.get("id", i)
            bbox = person.get("bbox", [])
            
            if len(bbox) >= 4:
                # 计算缩放后的边界框
                x1 = img_rect.x + bbox[0] * scale_x
                y1 = img_rect.y + bbox[1] * scale_y
                x2 = img_rect.x + bbox[2] * scale_x
                y2 = img_rect.y + bbox[3] * scale_y
                
                if (x1 <= mouse_pos[0] <= x2 and y1 <= mouse_pos[1] <= y2):
                    self.selected_person_id = person_id
                    print(f"选中人物 ID: {person_id}")
                    return True
        
        return False
    
    def next_person_id(self):
        """切换到下一个可选的人物ID"""
        if self.person_id_list:
            current_idx = self.person_id_list.index(self.selected_person_id) if self.selected_person_id in self.person_id_list else -1
            if current_idx < len(self.person_id_list) - 1:
                self.selected_person_id = self.person_id_list[current_idx + 1]
                return True
        return False
    
    def prev_person_id(self):
        """切换到上一个可选的人物ID"""
        if self.person_id_list:
            current_idx = self.person_id_list.index(self.selected_person_id) if self.selected_person_id in self.person_id_list else -1
            if current_idx > 0:
                self.selected_person_id = self.person_id_list[current_idx - 1]
                return True
        return False
    
    def handle_key_repeats(self):
        """处理方向键长按"""   
        current_time = time.time() 
        
        if any(self.key_pressed.values()):
            if self.last_key_time == 0:
                if current_time - self.key_pressed_time > self.key_repeat_delay:
                    if self.key_pressed["left"]:
                        self.prev_image()
                    elif self.key_pressed["right"]:
                        self.next_image()
                    self.last_key_time = current_time
            elif current_time - self.last_key_time > self.key_repeat_interval:
                if self.key_pressed["left"]:
                    self.prev_image()
                elif self.key_pressed["right"]:  
                    self.next_image()
                self.last_key_time = current_time
    
    def save_state(self):
        """保存当前状态以便撤销"""
        if len(self.undo_stack) >= self.max_undo_steps:
            self.undo_stack.pop(0)
    
        # 深度复制labels，确保能够正确处理嵌套结构
        labels_copy = {}
        for img_path, person_labels in self.labels.items():
            if isinstance(person_labels, dict):
                # 新格式：{图片路径: {人物ID: 标签}}
                labels_copy[img_path] = person_labels.copy()
            else:
                # 旧格式：{图片路径: 标签字符串} - 转换为新格式
                labels_copy[img_path] = {0: person_labels}
    
        state = {
            "current_image_index": self.current_image_index,
            "labels": labels_copy,
            "selected_person_id": self.selected_person_id,
            "continuous_mode": self.continuous_mode,
            "continuous_start_index": self.continuous_start_index,
            "continuous_label": self.continuous_label
        }
    
        self.undo_stack.append(state)
    
    def undo(self):
        """撤销上一次操作"""
        if self.undo_stack:
            state = self.undo_stack.pop()
            self.current_image_index = state["current_image_index"]
            self.labels = state["labels"]
            self.selected_person_id = state["selected_person_id"]
            self.continuous_mode = state["continuous_mode"]
            self.continuous_start_index = state["continuous_start_index"]
            self.continuous_label = state["continuous_label"]
            self.update_person_id_list()
            return True
        return False
    
    def load_labels(self):
        """从文件加载标记状态，支持新旧格式"""
        labels_file = os.path.join(self.root_path, "labels_backup.json")
        if os.path.exists(labels_file):
            try:
                with open(labels_file, 'r') as f:
                    loaded = json.load(f)
                
                # 转换旧格式到新格式
                self.labels = {}
                for img_path, label_data in loaded.items():
                    if isinstance(label_data, dict):
                        # 新格式：已经是 {人物ID: 标签}
                        self.labels[img_path] = label_data
                    else:
                        # 旧格式：标签字符串，转换为新格式，默认人物ID为0
                        self.labels[img_path] = {0: label_data}
                    
                print(f"成功加载标签，共 {len(self.labels)} 张图片的标签")
            except Exception as e:
                print(f"加载标记状态失败: {e}")
                self.labels = {}
        else:
            print(f"标签文件不存在: {labels_file}")
            self.labels = {}
 
    def save_labels(self):
        """保存标记状态到文件"""
        labels_file = os.path.join(self.root_path, "labels_backup.json")
        try:
            existing_labels = {}
            for img_path, person_labels in self.labels.items():
                if os.path.exists(img_path) and person_labels:
                    # 确保保存新格式
                    existing_labels[img_path] = person_labels
        
            with open(labels_file, 'w') as f:
                json.dump(existing_labels, f, indent=2)
        
            print(f"标签已保存到: {labels_file}")
        except Exception as e:
            print(f"保存标记状态失败: {e}")
 
    def load_annotations(self):
        """加载全局 annotations.json"""
        if os.path.exists(self.annotations_path):
            try:
                with open(self.annotations_path, 'r', encoding='utf-8') as f:
                    self.annotations_data = json.load(f)
                print(f"已加载 {len(self.annotations_data.get('samples', []))} 个标注片段")
            except Exception as e:
                print(f"加载 annotations 失败: {e}")
                self.annotations_data = {"samples": []}
        else:
            self.annotations_data = {"samples": []}
 
    def save_annotations(self):
        """保存 annotations.json"""
        try:
            with open(self.annotations_path, 'w', encoding='utf-8') as f:
                json.dump(self.annotations_data, f, indent=2, ensure_ascii=False)
            print(f"annotations 已保存到: {self.annotations_path}")
        except Exception as e:
            print(f"保存 annotations 失败: {e}")
 
    @staticmethod
    def extract_frame_number(img_path):
        """从文件名中提取最后一个数字串作为帧号"""
        import re
        name = os.path.splitext(os.path.basename(img_path))[0]
        numbers = re.findall(r'\d+', name)
        return int(numbers[-1]) if numbers else 0
 
    def build_video_name(self, segment_paths, person_id, label):
        """构建视频文件名（不含扩展名）"""
        folder_name = os.path.basename(self.folders[self.current_folder_index])
        start_frame = self.extract_frame_number(segment_paths[0])
        end_frame = self.extract_frame_number(segment_paths[-1])
        range_str = f"{start_frame:06d}to{end_frame:06d}"
        suffix = "A001" if label == "positive" else "A002"
        return f"{folder_name}_{range_str}_P{person_id}_{suffix}"
 
    def build_annotation(self, segment_paths, person_id, label, video_name):
        """
        根据连续帧片段构建 annotation 字典
        segment_paths : list of absolute image paths (已排序)
        person_id     : int
        label         : str ('positive' / 'negative')
        video_name    : str (不含扩展名)
        """
        # 1. 读取第一帧获取图像尺寸
        first_img = cv2.imread(segment_paths[0])
        if first_img is None:
            raise ValueError(f"无法读取图片: {segment_paths[0]}")
        h, w = first_img.shape[:2]
 
        # 2. 逐帧提取该人物的关键点
        keypoint_seq = []   # T x 17 x 2
        score_seq = []      # T x 17
 
        for img_path in segment_paths:
            img_name = os.path.basename(img_path)
            persons = self.inference_data.get(img_name, [])
            target_person = None
            for p in persons:
                if p.get("id") == person_id:
                    target_person = p
                    break
 
            if target_person is None:
                # 理论上不应该发生，但若缺失则用全零填充
                keypoint_seq.append([[0.0, 0.0]] * 17)
                score_seq.append([0.0] * 17)
                continue
 
            kps = target_person.get("keypoints", [])
            # 解析 keypoints 列表 (x,y,v) 共 51 个值
            coords = []
            scores = []
            for i in range(0, len(kps), 3):
                x = kps[i] if i < len(kps) else 0.0
                y = kps[i+1] if i+1 < len(kps) else 0.0
                v = kps[i+2] if i+2 < len(kps) else 0.0
                coords.append([x, y])
                scores.append(v)
            # 如果某些帧点数不足17，补齐
            while len(coords) < 17:
                coords.append([0.0, 0.0])
                scores.append(0.0)
            keypoint_seq.append(coords[:17])
            score_seq.append(scores[:17])
 
        # 3. 构建 annotation 字典
        annotation = {
            "frame_dir": video_name,
            "label": 0 if label == "positive" else 1,   # 正样本→0，负样本→1
            "img_shape": [h, w],
            "original_shape": [h, w],
            "total_frames": len(segment_paths),
            "keypoint": keypoint_seq,      # T x 17 x 2
            "keypoint_score": score_seq    # T x 17
        }
        return annotation
 
def draw_button(screen, text, rect, hover=False, color=None):
    """绘制按钮"""
    if color is None:
        color = BUTTON_HOVER_COLOR if hover else BUTTON_COLOR
    
    pygame.draw.rect(screen, color, rect, border_radius=5)
    pygame.draw.rect(screen, (100, 100, 100), rect, 2, border_radius=5)
    
    text_surface = small_font.render(text, True, TEXT_COLOR)
    txt_rect = text_surface.get_rect(center=rect.center)
    screen.blit(text_surface, txt_rect)
 
def draw_confirm_dialog(screen, message, width=400, height=200):
    """绘制确认对话框"""
    dialog_rect = pygame.Rect(
        (WINDOW_WIDTH - width) // 2,
        (WINDOW_HEIGHT - height) // 2, 
        width, height
    )
    
    pygame.draw.rect(screen, BG_COLOR, dialog_rect, border_radius=10)
    pygame.draw.rect(screen, TEXT_COLOR, dialog_rect, 2, border_radius=10)
    
    lines = []
    words = message.split()
    current_line = ""
    
    for word in words:
        test_line = current_line + word + " "
        if small_font.size(test_line)[0] < width - 40:
            current_line = test_line
        else:
            lines.append(current_line)
            current_line = word + " "
    
    if current_line:
        lines.append(current_line)
    
    for i, line in enumerate(lines):
        text_surface = small_font.render(line, True, TEXT_COLOR)
        screen.blit(text_surface, (dialog_rect.x + 20, dialog_rect.y + 30 + i * 25))
    
    yes_button = pygame.Rect(dialog_rect.x + width // 2 - 100, dialog_rect.y + height - 50, 80, 30)
    no_button = pygame.Rect(dialog_rect.x + width // 2 + 20, dialog_rect.y + height - 50, 80, 30)
    
    draw_button(screen, "是", yes_button, color=CONFIRM_COLOR)
    draw_button(screen, "否", no_button, color=WARNING_COLOR)
    
    return dialog_rect, yes_button, no_button
 
def main():
    root_path = ROOT_PATH
    
    tool = ImageLabelingTool(root_path)
    
    positive_dir = os.path.join(root_path, "1")
    negative_dir = os.path.join(root_path, "0")
    
    running = True
    clock = pygame.time.Clock()
    
    # 按钮布局
    button_height = 40
    button_width = 140
    button_margin = 15
    
    # 行坐标（从底部向上数）
    row1_y = WINDOW_HEIGHT - button_height - button_margin                     # 最下面一行（第0行）
    row2_y = WINDOW_HEIGHT - 2*button_height - 2*button_margin                # 第1行
    row3_y = WINDOW_HEIGHT - 3*button_height - 3*button_margin                # 第2行
    row4_y = WINDOW_HEIGHT - 4*button_height - 4*button_margin                # 第3行
 
    # 分析按钮（第4行）
    analysis_buttons = {
        "analyze_ids": pygame.Rect(button_margin, row4_y, button_width, button_height),
    }
 
    # 人物选择按钮（第3行）
    person_buttons = {
        "prev_person": pygame.Rect(button_margin, row3_y, button_width, button_height),
        "next_person": pygame.Rect(button_margin * 2 + button_width, row3_y, button_width, button_height),
        "toggle_bbox": pygame.Rect(button_margin * 3 + button_width * 2, row3_y, button_width, button_height),
        "toggle_kpts": pygame.Rect(button_margin * 4 + button_width * 3, row3_y, button_width, button_height),
        "cancel_label": pygame.Rect(button_margin * 5 + button_width * 4, row3_y, button_width, button_height),
    }
 
    # 导航按钮（第2行）
    nav_buttons = {
        "prev": pygame.Rect(button_margin, row2_y, button_width, button_height),
        "next": pygame.Rect(button_margin * 2 + button_width, row2_y, button_width, button_height),
        "prev_folder": pygame.Rect(button_margin * 3 + button_width * 2, row2_y, button_width, button_height),
        "next_folder": pygame.Rect(button_margin * 4 + button_width * 3, row2_y, button_width, button_height),
        "undo": pygame.Rect(button_margin * 5 + button_width * 4, row2_y, button_width, button_height),
    }
 
    # 标注按钮（第1行）
    label_buttons = {
        "positive": pygame.Rect(button_margin, row1_y, button_width, button_height),
        "negative": pygame.Rect(button_margin * 2 + button_width, row1_y, button_width, button_height),
        "continuous_start": pygame.Rect(button_margin * 3 + button_width * 2, row1_y, button_width, button_height),
        "continuous_end": pygame.Rect(button_margin * 4 + button_width * 3, row1_y, button_width, button_height),
        "move_files": pygame.Rect(button_margin * 5 + button_width * 4, row1_y, button_width, button_height),
    }
 
    # 图片显示区域
    image_area = pygame.Rect(50, 80, WINDOW_WIDTH - 100, WINDOW_HEIGHT - 340)
    
    tool.key_pressed_time = 0
    
    while running:
        mouse_pos = pygame.mouse.get_pos()
        
        tool.handle_key_repeats()
        
        if tool.playing:
            now = pygame.time.get_ticks()
            if now - tool.last_play_tick > tool.play_interval:
                if tool.play_direction == 1:
                    tool.next_image()
                else:
                    tool.prev_image()
                tool.last_play_tick = now
        
        for event in pygame.event.get():
            if event.type == QUIT:
                running = False
            elif event.type == KEYDOWN:
                if event.key == K_d:
                    tool.key_pressed["right"] = True
                    tool.key_pressed["left"] = False
                    tool.key_pressed_time = time.time()
                    tool.next_image()
                elif event.key == K_a:
                    tool.key_pressed["left"] = True
                    tool.key_pressed["right"] = False
                    tool.key_pressed_time = time.time()
                    tool.prev_image()
                elif event.key == K_q:  # 切换到上一个人物
                    tool.prev_person_id()
                elif event.key == K_e:  # 切换到下一个人物
                    tool.next_person_id()
                elif event.key == K_b:  # 切换边界框显示
                    tool.show_bboxes = not tool.show_bboxes
                elif event.key == K_k:  # 切换关键点显示
                    tool.show_keypoints = not tool.show_keypoints
                elif event.key == K_RIGHT:
                    tool.play_direction = 1
                    tool.playing = True
                    tool.last_play_tick = pygame.time.get_ticks()
                elif event.key == K_LEFT:
                    tool.play_direction = -1
                    tool.playing = True
                    tool.last_play_tick = pygame.time.get_ticks()
                elif event.key == K_SPACE:
                    tool.playing = not tool.playing
                    if tool.playing:
                        tool.last_play_tick = pygame.time.get_ticks()
                elif event.key == K_v:
                    tool.convert_to_video = not tool.convert_to_video
                    print("转视频模式：" + ("开启" if tool.convert_to_video else "关闭"))
                elif event.key == K_w:  # 标记选定人物为正样本
                    tool.label_current_image("positive")
                elif event.key == K_s:  # 标记选定人物为负样本
                    tool.label_current_image("negative")
                elif event.key == K_UP:
                    if not tool.start_continuous_labeling():
                        print("无法开始连续标记")
                elif event.key == K_DOWN:
                    if not tool.end_continuous_labeling():
                        print("没有激活的连续标记")
                elif event.key == K_x:
                    tool.move_labeled_files(positive_dir, negative_dir)
                elif event.key == K_c:
                    tool.next_folder()
                elif event.key == K_r:  # 按 R 取消当前选定人物的标记
                    if tool.cancel_current_person_label():
                        print(f"已取消人物 {tool.selected_person_id} 的标记")
                    else:
                        print("无法取消标记（没有标记或图片）")
                elif event.key == K_z:
                    if pygame.key.get_mods() & KMOD_CTRL:
                        # Ctrl+Z: 撤销
                        if tool.undo():
                            print("已撤销上一次操作")
                        else:
                            print("没有可撤销的操作")
                    else:
                        # 单独按 Z: 上一个文件夹
                        tool.prev_folder()
                elif event.key == K_ESCAPE:
                    if tool.show_confirm_dialog:
                        tool.show_confirm_dialog = False
            
            elif event.type == KEYUP:
                if event.key == K_d:
                    tool.key_pressed["right"] = False
                    tool.last_key_time = 0
                elif event.key == K_a:
                    tool.key_pressed["left"] = False
                    tool.last_key_time = 0
            
            elif event.type == MOUSEBUTTONDOWN:
                if event.button == 1:
                    if tool.show_confirm_dialog:
                        dialog_rect, yes_button, no_button = draw_confirm_dialog(screen, tool.confirm_message)
                        if yes_button.collidepoint(mouse_pos):
                            tool.show_confirm_dialog = False
                            if tool.confirm_action == "next_folder":
                                tool.current_folder_index += 1
                                tool.load_current_folder_images()
                        elif no_button.collidepoint(mouse_pos):
                            tool.show_confirm_dialog = False
                    else:
                        # 分析按钮
                        if analysis_buttons["analyze_ids"].collidepoint(mouse_pos):
                            tool.analyze_id_tracking()
 
                        # 人物选择按钮
                        if person_buttons["prev_person"].collidepoint(mouse_pos):
                            tool.prev_person_id()
                        elif person_buttons["next_person"].collidepoint(mouse_pos):
                            tool.next_person_id()
                        elif person_buttons["toggle_bbox"].collidepoint(mouse_pos):
                            tool.show_bboxes = not tool.show_bboxes
                        elif person_buttons["toggle_kpts"].collidepoint(mouse_pos):
                            tool.show_keypoints = not tool.show_keypoints
                        elif person_buttons["cancel_label"].collidepoint(mouse_pos):
                            if tool.cancel_current_person_label():
                                print(f"已取消人物 {tool.selected_person_id} 的标记")
                            else:
                                print("无法取消标记")
                        
                        # 导航按钮
                        elif nav_buttons["prev"].collidepoint(mouse_pos):
                            tool.prev_image()
                        elif nav_buttons["next"].collidepoint(mouse_pos):
                            tool.next_image()
                        elif nav_buttons["prev_folder"].collidepoint(mouse_pos):
                            tool.prev_folder()
                        elif nav_buttons["next_folder"].collidepoint(mouse_pos):
                            tool.next_folder()
                        elif nav_buttons["undo"].collidepoint(mouse_pos):
                            if tool.undo():
                                print("已撤销上一次操作")
                            else:
                                print("没有可撤销的操作")
                        
                        # 标注按钮
                        elif label_buttons["positive"].collidepoint(mouse_pos):
                            tool.label_current_image("positive")
                        elif label_buttons["negative"].collidepoint(mouse_pos):
                            tool.label_current_image("negative")
                        elif label_buttons["continuous_start"].collidepoint(mouse_pos):
                            if not tool.start_continuous_labeling():
                                print("无法开始连续标记")
                        elif label_buttons["continuous_end"].collidepoint(mouse_pos):
                            if not tool.end_continuous_labeling():
                                print("没有激活的连续标记")
                        elif label_buttons["move_files"].collidepoint(mouse_pos):
                            tool.move_labeled_files(positive_dir, negative_dir)
                        
                        # 检查是否点击了图片区域来选择人物
                        elif image_area.collidepoint(mouse_pos):
                            tool.select_person_by_click(mouse_pos, image_area)
        
        # 清屏
        screen.fill(BG_COLOR)
        
        # 显示文件信息
        if tool.folders:
            folder_text = f"当前文件夹: {os.path.basename(tool.folders[tool.current_folder_index])} ({tool.current_folder_index + 1}/{len(tool.folders)})"
            text_surface = small_font.render(folder_text, True, TEXT_COLOR)
            screen.blit(text_surface, (20, 20))
        
        # 显示当前图片
        current_image_path = tool.get_current_image()
        if current_image_path and os.path.exists(current_image_path):
            try:
                img = pygame.image.load(current_image_path)
                img_rect = img.get_rect()
                
                scale = min(image_area.width / img_rect.width, image_area.height / img_rect.height)
                new_size = (int(img_rect.width * scale), int(img_rect.height * scale))
                img = pygame.transform.smoothscale(img, new_size)
                img_rect = img.get_rect(center=image_area.center)
                
                screen.blit(img, img_rect)
                
                # 显示检测结果
                tool.draw_detections(screen, img_rect, current_image_path)
                
                # 显示图片信息
                info_text = f"{os.path.basename(current_image_path)} ({tool.current_image_index + 1}/{len(tool.images)})"
 
                # 显示当前图片所有已标记人物的动作类别（正样本 → 0，负样本 → 1）
                if current_image_path in tool.labels:
                    person_labels = tool.labels[current_image_path]
                    # 过滤出真正的人物ID（排除 __processed__ 等非数字键）
                    valid_labels = {pid: label for pid, label in person_labels.items()
                                    if isinstance(pid, int) or (isinstance(pid, str) and pid.isdigit())}
                    if valid_labels:
                        label_map = {'positive': 0, 'negative': 1}
                        labeled_dict = {int(pid): label_map[label] for pid, label in valid_labels.items()
                                        if label in label_map}
                        if labeled_dict:
                            # 格式化为 {1:0,2:1} 紧凑形式
                            dict_str = str(labeled_dict).replace(' ', '')
                            info_text += f" - 已标记人物动作{dict_str}"
 
                text_surface = font.render(info_text, True, TEXT_COLOR)
                text_rect = text_surface.get_rect(center=(WINDOW_WIDTH // 2, image_area.y - 20))
                screen.blit(text_surface, text_rect)
                
                # 显示选定人物ID
                person_text = f"选定人物ID: {tool.selected_person_id} (Q/E切换, 点击框选择)"
                person_surface = small_font.render(person_text, True, HIGHLIGHT_COLOR)
                screen.blit(person_surface, (20, 50))
                
                # 显示连续标记范围
                if tool.continuous_mode and tool.continuous_start_index is not None:
                    start_idx = min(tool.continuous_start_index, tool.current_image_index)
                    end_idx = max(tool.continuous_start_index, tool.current_image_index)
                    frame_diff = abs(tool.current_image_index - tool.continuous_start_index)
                    
                    # 第一行：连续标记范围
                    range_text1 = f"连续标记范围: {start_idx + 1} - {end_idx + 1}"
                    range_surface1 = small_font.render(range_text1, True, HIGHLIGHT_COLOR)
                    screen.blit(range_surface1, (20, 75))
                    
                    # 第二行：已过帧数
                    range_text2 = f"已过 {frame_diff} 帧 (人物 {tool.selected_person_id})"
                    range_surface2 = small_font.render(range_text2, True, HIGHLIGHT_COLOR)
                    screen.blit(range_surface2, (20, 100))
                    
                    marker_width = image_area.width / len(tool.images)
                    start_x = image_area.x + start_idx * marker_width
                    end_x = image_area.x + (end_idx + 1) * marker_width
                    
                    pygame.draw.rect(screen, HIGHLIGHT_COLOR,
                                    (start_x, image_area.y + image_area.height + 5,
                                    end_x - start_x, 5))
            
            except Exception as e:
                error_text = f"无法加载图片: {e}"
                text_surface = font.render(error_text, True, (255, 0, 0))
                screen.blit(text_surface, (image_area.centerx - text_surface.get_width() // 2, image_area.centery - text_surface.get_height() // 2))
        
        else:
            no_image_text = "没有图片可显示"
            text_surface = font.render(no_image_text, True, TEXT_COLOR)
            screen.blit(text_surface, (image_area.centerx - text_surface.get_width() // 2, image_area.centery - text_surface.get_height() // 2))
        
        # 显示显示设置状态
        display_status = f"显示: 边界框({'开' if tool.show_bboxes else '关'}/B) 关键点({'开' if tool.show_keypoints else '关'}/K)"
        status_surface = small_font.render(display_status, True, TEXT_COLOR)
        screen.blit(status_surface, (WINDOW_WIDTH - status_surface.get_width() - 20, 50))
 
        # 绘制分析按钮
        draw_button(screen, "分析ID跟踪", analysis_buttons["analyze_ids"], analysis_buttons["analyze_ids"].collidepoint(mouse_pos))
        
        # 绘制人物选择按钮
        draw_button(screen, "上个ID (Q)", person_buttons["prev_person"], person_buttons["prev_person"].collidepoint(mouse_pos))
        draw_button(screen, "下个ID (E)", person_buttons["next_person"], person_buttons["next_person"].collidepoint(mouse_pos))
        draw_button(screen, f"边界框(B):{'开' if tool.show_bboxes else '关'}", person_buttons["toggle_bbox"], person_buttons["toggle_bbox"].collidepoint(mouse_pos))
        draw_button(screen, f"关键点(K):{'开' if tool.show_keypoints else '关'}", person_buttons["toggle_kpts"], person_buttons["toggle_kpts"].collidepoint(mouse_pos))
        draw_button(screen, "取消标记 (R)", person_buttons["cancel_label"], person_buttons["cancel_label"].collidepoint(mouse_pos)) 
 
        # 绘制导航按钮
        draw_button(screen, "上一张 (a)", nav_buttons["prev"], nav_buttons["prev"].collidepoint(mouse_pos))
        draw_button(screen, "下一张 (d)", nav_buttons["next"], nav_buttons["next"].collidepoint(mouse_pos))
        draw_button(screen, "上个文件夹 (z)", nav_buttons["prev_folder"], nav_buttons["prev_folder"].collidepoint(mouse_pos))
        draw_button(screen, "下个文件夹 (c)", nav_buttons["next_folder"], nav_buttons["next_folder"].collidepoint(mouse_pos))
        draw_button(screen, "撤销 (Ctrl+Z)", nav_buttons["undo"], nav_buttons["undo"].collidepoint(mouse_pos))
        
        # 绘制标注按钮
        draw_button(screen, "正样本 (w)", label_buttons["positive"], label_buttons["positive"].collidepoint(mouse_pos))
        draw_button(screen, "负样本 (s)", label_buttons["negative"], label_buttons["negative"].collidepoint(mouse_pos))
        draw_button(screen, "开始连续标(↑)", label_buttons["continuous_start"], label_buttons["continuous_start"].collidepoint(mouse_pos))
        draw_button(screen, "结束连续标(↓)", label_buttons["continuous_end"], label_buttons["continuous_end"].collidepoint(mouse_pos))
        draw_button(screen, "移动文件 (x)", label_buttons["move_files"], label_buttons["move_files"].collidepoint(mouse_pos))
        
 
        # 显示确认对话框
        if tool.show_confirm_dialog:
            draw_confirm_dialog(screen, tool.confirm_message)
        
        # 更新屏幕
        pygame.display.flip()
        clock.tick(30)
    
    # 退出前保存
    tool.save_annotations()   # 若已有修改，确保保存
    # 退出前保存标记状态
    tool.save_labels()
    pygame.quit()
    sys.exit()
 
if __name__ == "__main__":
    main()
