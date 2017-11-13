#!/bin/bash

USE_SEG=$1

BASE="${HOME}/Indoor-segmentation-essential/robot_project/train_log"
A3C_SEG_SINGLE_FRAME_FOUR_COLOR="A3C-SEG-HIST-1/model-51000.index"
A3C_SEG_FOUR_FRAME_FOUR_COLOR="A3C-SEG-HIST-4/model-51000.index"
A3C_RGB_SINGLE_FRAME_FOUR_COLOR="A3C-DR-RAW-HIST-1/model-45000.index"
A3C_RGB_FOUR_FRAME_FOUR_COLOR="A3C-DR-RAW-HIST-4/model-36000.index"
A3C_SEG_SINGLE_FRAME_THREE_COLOR="train-unity3d-follow_target_endNoR_his1/model-60000.index"
A3C_RGB_SINGLE_FRAME_THREE_COLOR="train_log/train-unity3d-follow_random_material_his1/model-33000.index"
#A3C_SEG_FOUR_FRAME_THREE_COLOR="train-unity3d-hallway_four_movingObs_3color/model-81000.index"

MODEL=$A3C_SEG_SINGLE_FRAME_THREE_COLOR

H=480
W=640

source ~/catkin_ws/devel/setup.sh
roslaunch jetson_csi_cam jetson_csi_cam.launch width:=$W height:=$H fps:=30 &

#roslaunch zed_wrapper zed.launch &
sleep 5
roslaunch rmp_base rmp_base.launch &
sleep 5

python local_icnet_tar.py --path="$BASE/$MODEL" --use_seg=${USE_SEG} --height $H --width $W 
