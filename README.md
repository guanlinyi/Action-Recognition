# Action-Recognition

1.环境:  
（1）需要拷贝mmaction2仓库：git clone https://github.com/open-mmlab/mmaction2  
     然后在mmaction2中新建文件夹my_tools，将本仓库代码拷贝进my_tools  
     然后创建conda环境：conda create -n action_rec python=3.10.16  
     在conda环境中安装mmaction2环境，可阅读官方文档：https://mmpose.readthedocs.io/zh-cn/latest/installation.html 或者 博客https://blog.csdn.net/shimingwang/article/details/154687372的4.1和4.2  
     然后安装依赖库，关键库：pip install pygame==2.6.1  
     然后下载yolo26s-pose.pt和slowonly_r50_8xb16-u48-240e_ntu60-xsub-keypoint_20220815-38db104b.pth 放到mmaction2/checkpoints  
     https://docs.ultralytics.com/tasks/pose/#models 下载yolo26s-pose.pt,放到mmaction2\checkpoints， 后面第（4）步动作分类标注工具和第（8）步yolo26pose_PoseC3D.py实时动作识别的代码会加载该模型  
     https://mmaction2.readthedocs.io/zh-cn/latest/model_zoo/skeleton.html#posec3d 找到NTU60_XSub，下载第一行的ckpt,放到mmaction2\checkpoints。 （后面第（7）步在配置文件slowonly_r50_8xb16-u48-240e_ntu60-xsub-keypoint.py最后一行会加载该模型）  

2.脚本使用顺序
 注：脚本提供了从数据转换到数据标注到训练到实时动作识别的指导
（1）get_video_fps.py   获取视频目标帧率  
（2）video2mp4.py       转mp4  
（3）mp4tojpgmp4_set_rate.py   转视频目标帧率  
 注：前三步的目的是将视频转为30fps的mp4，如果你的输入视频帧率本身就是30fps的mp4则不需要转换  
（4）action_label_tool.py    动作分类标注工具  
 注：目前只支持动作二分类，即分正样本视频和负样本视频  
（5）PoseC3D_count_action.py  计算正样本视频和负样本视频数  
 注：首次训练正样本和负样本=1:1较好，因此需要计算正样本视频和负样本视频数  
（5）generate_ntu60_pkl.py     生成pkl文件  
 注：action_label_tool.py只能生成annotations.pkl，这个文件不能直接用于训练，因为还没有将训练集和验证集划分成8:2，该脚本用于将annotations.pkl转为可训练的pkl格式，即训练集和验证集划分成8:2  
（6）visualize_pkl_keypoints.py   可视化pkl检查  
 注：可选择可视化pkl以确保自己生成的pkl文件中的骨架序列是正确的。  
 (7) 注：训练使用mmaction2的训练配置和训练脚本，但训练配置要修改  
     训练配置文件：将slowonly_r50_8xb16-u48-240e_ntu60-xsub-keypoint.py替换mmaction2/configs/skeleton/posec3d/slowonly_r50_8xb16-u48-240e_ntu60-xsub-keypoint.py  
     训练脚本：mmaction2/tools/train.py   
     训练命令（cmd终端）：  
     python tools/train.py configs/skeleton/posec3d/slowonly_r50_8xb16-u48-240e_ntu60-xsub-keypoint.py ^  
     --work-dir work_dirs/slowonly_r50_8xb16-u48-240e_ntu60-xsub-keypoint ^  
     --seed 42  
 (8) yolo26pose_PoseC3D.py  实时动作识别  
  
3. 标注工具action_label_tool.py生成的文件结果如下:  
<img width="629" height="401" alt="image" src="https://github.com/user-attachments/assets/f1964b3b-7a88-451c-baaa-8f6c587469e8" />  
  
标注工具生成的annotations.pkl格式如下：  
<img width="610" height="539" alt="image" src="https://github.com/user-attachments/assets/7959e20f-0d4f-4b60-bc7f-c7d380e0728c" />  
  
4.generate_ntu60_pkl.py根据action_label_tool.py生成的annotations.pkl生成可用于训练的my_ntu60_2d.pkl，格式如下：  
<img width="611" height="646" alt="image" src="https://github.com/user-attachments/assets/71f1b19b-a12a-467f-b58c-145e62e0e5dc" />  
  
5.yolo26pose_PoseC3D.py调用opencv进行实时动作识别  
