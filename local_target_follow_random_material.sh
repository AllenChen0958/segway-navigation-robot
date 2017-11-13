#!/bin/bash

USE_SEG=$1
source ~/catkin_ws/devel/setup.sh
roslaunch jetson_csi_cam jetson_csi_cam.launch width:=640 height:=480 &
#roslaunch zed_wrapper zed.launch &
sleep 5
roslaunch rmp_base rmp_base.launch &
sleep 5

python local.py --path="${HOME}/Indoor-segmentation-essential/robot_project/train_log/train-unity3d-follow_target_random_material/model-45000.index" --use_seg=False
