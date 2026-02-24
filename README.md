# Action-Recognition

1.环境
conda create -n action_rec python=3.10.16
pip install pygame==2.6.1

1. 标注工具action_label_tool.py生成的文件结果如下:

long_video1
├──ok/
│  ├── positive/         # 存放所有裁剪出来的正样本动作视频
│      ├── xxx_P1_A001.mp4
│      └── ..
│  ├── negative/         # 存放所有裁剪出来的负样本动作视频
│      ├── xxx_P2_A002.mp4
│      └── ...
├──000000.jpg         # 原始的视频帧
├──000001.jpg
├──......
├──yolo_pose_results.json         # 记录yolo推理结果的文件
├──yolo_pose_results_stats.json   # 记录yolo推理结果状态的文件
├──labels_backup.json             # 记录标注工具标注进度的文件
└── annotations.pkl               # 记录标注信息的文件，可用于转成输入训练的pkl文件
 
long_video2同理
├──...

标注工具生成的annotations.pkl格式如下：
{
    "samples": \[
        {   # P1的1是指用户1
            # A001的001是指用户标记真实的动作类别是第1个动作，A后面的数字是动作id
            'frame_dir': 'fisheye_back_2026-01-29-14-16-15_358to389_P1_A001',
            'label': 0,  # 真实的动作类别索引，第1个动作类别对应索引0
            'img_shape': (1080, 1920),
            'original_shape': (1080, 1920),
            'total_frames': 103,
            'keypoint': (1, 103, 17, 2),
            'keypoint_score': (1, 103, 17),
        },
        { 
            'frame_dir': 'fisheye_back_2026-01-29-14-16-15_358to389_P2_A002',
            'label': 1,  # 真实的动作类别索引，第2个动作类别对应索引1
            'img_shape': (1080, 1920),
            'original_shape': (1080, 1920),
            'total_frames': 158,
            'keypoint': (1, 158, 17, 2),
            'keypoint_score': (1, 158, 17),
        },
        ...... 
     ]
}
  
2.generate_ntu60_pkl.py根据action_label_tool.py生成的annotations.pkl生成可用于训练的my_ntu60_2d.pkl  

data:{
    'split':{
                 'train': \['frame_dir名称', 'xxx', ...],  
                 'val':  \['xxx', 'xxx', ...],  
     }
 
    'annotations':{  
        {   # P1的1是指用户1
            # A001的001是指用户标记真实的动作类别是第1个动作，A后面的数字是动作id
            'frame_dir': 'fisheye_back_2026-01-29-14-16-15_358to389_P1_A001',
            'label': 0,  # 真实的动作类别索引，第1个动作类别对应索引0
            'img_shape': (1080, 1920),
            'original_shape': (1080, 1920),
            'total_frames': 103,
            'keypoint': (1, 103, 17, 2),
            'keypoint_score': (1, 103, 17),
        },
        { 
            'frame_dir': 'fisheye_back_2026-01-29-14-16-15_358to389_P2_A002',
            'label': 1,  # 真实的动作类别索引，第2个动作类别对应索引1
            'img_shape': (1080, 1920),
            'original_shape': (1080, 1920),
            'total_frames': 158,
            'keypoint': (1, 158, 17, 2),
            'keypoint_score': (1, 158, 17),
        },
        ......
    }
}
 
