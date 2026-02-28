# Action-Recognition

1.关键环境库版本:
（1）conda create -n action_rec python=3.10.16
（2）pip install pygame==2.6.1

2.脚本使用顺序
（1）get_video_fps.py   获取视频目标帧率
（2）video2mp4.py       转mp4
（3）mp4tojpgmp4_set_rate.py   转视频目标帧率
（4）action_label_tool.py    动作分类标注工具
（5）generate_ntu60_pkl.py     生成pkl文件
（6）visualize_pkl_keypoints.py   可视化pkl检查
 (7) yolo26pose_PoseC3D.py  实时动作识别

3. 标注工具action_label_tool.py生成的文件结果如下:
<img width="629" height="401" alt="image" src="https://github.com/user-attachments/assets/f1964b3b-7a88-451c-baaa-8f6c587469e8" />

标注工具生成的annotations.pkl格式如下：
<img width="610" height="539" alt="image" src="https://github.com/user-attachments/assets/7959e20f-0d4f-4b60-bc7f-c7d380e0728c" />

4.generate_ntu60_pkl.py根据action_label_tool.py生成的annotations.pkl生成可用于训练的my_ntu60_2d.pkl，格式如下：
<img width="611" height="646" alt="image" src="https://github.com/user-attachments/assets/71f1b19b-a12a-467f-b58c-145e62e0e5dc" />

5.yolo26pose_PoseC3D.py调用opencv进行实时动作识别
