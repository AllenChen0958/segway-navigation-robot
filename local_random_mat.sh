#!/bin/bash

source ~/catkin_ws/devel/setup.sh

roslaunch zed_wrapper zed.launch &
sleep 5
roslaunch rmp_base rmp_base.launch &
sleep 5

python local.py --path="${HOME}/Indoor-segmentation/robot_project/train_log/train-unity3d-random_material/model-42000.index" --use_seg=False
