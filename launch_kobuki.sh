#!/bin/bash

USE_SEG=$1

BASE="${HOME}/Indoor-segmentation-essential/robot_project/train_log"
#A3C_SEG_SINGLE_FRAME_FOUR_COLOR="a3c-seg-frame-1/model-182400.index"
A3C_SEG_SINGLE_FRAME_FOUR_COLOR="a3c-timescale-1-slim/model-52200"
A3C_RGB_SINGLE_FRAME_FOUR_COLOR="a3c-rgb-frame-1/model-126600.index"
A3C_SEG_FOUR_FRAME_THREE_COLOR="train-unity3d-hallway_four_movingObs_3color/model-81000.index"

MODEL=$A3C_SEG_SINGLE_FRAME_FOUR_COLOR

H=480
W=640

source ~/catkin_ws/devel/setup.sh
roslaunch jetson_csi_cam jetson_csi_cam.launch width:=$W height:=$H fps:=30 &

#roslaunch zed_wrapper zed.launch &
sleep 5
#roslaunch rmp_base rmp_base.launch &
roslaunch kobuki_node minimal.launch --screen &
sleep 5

python local.py --path="$BASE/$MODEL" --use_seg=${USE_SEG} --height $H --width $W 
