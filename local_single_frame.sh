#!/bin/bash

USE_SEG=$1
source ~/catkin_ws/devel/setup.sh
roslaunch jetson_csi_cam jetson_csi_cam.launch width:=640 height:=480 &
#roslaunch zed_wrapper zed.launch &
sleep 5
#roslaunch rmp_base rmp_base.launch &
#sleep 5

python local.py --path="${HOME}/Indoor-segmentation-essential/robot_project/train_log/a3c-seg-frame-1/model-182400.index" --use_seg=${USE_SEG} 
