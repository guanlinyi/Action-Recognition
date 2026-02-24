# Action-Recognition

1.环境
conda create -n action_rec python=3.10.16
pip install pygame==2.6.1

1. 标注工具action_label_tool.py生成的文件结果如下:
<img width="629" height="401" alt="image" src="https://github.com/user-attachments/assets/f1964b3b-7a88-451c-baaa-8f6c587469e8" />

标注工具生成的annotations.pkl格式如下：
<img width="610" height="539" alt="image" src="https://github.com/user-attachments/assets/7959e20f-0d4f-4b60-bc7f-c7d380e0728c" />

2.generate_ntu60_pkl.py根据action_label_tool.py生成的annotations.pkl生成可用于训练的my_ntu60_2d.pkl，格式如下：
<img width="611" height="646" alt="image" src="https://github.com/user-attachments/assets/71f1b19b-a12a-467f-b58c-145e62e0e5dc" />
