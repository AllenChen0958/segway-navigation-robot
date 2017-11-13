#!/bin/bash

USE_SEG=$1
Q_SIZE=$2
LIN=$3
ANG=$4

BASE="${HOME}/Indoor-segmentation-essential/robot_project/train_log"

MODEL="a3c-dr-resnet-hist-1/model-5760630"

H=480
W=640

source ~/catkin_ws/devel/setup.sh

# Camera node
roslaunch jetson_csi_cam jetson_csi_cam.launch width:=$W height:=$H fps:=30 &
#roslaunch zed_wrapper zed.launch &
sleep 5

# RMP node
#roslaunch rmp_base rmp_base.launch &
roslaunch kobuki_node minimal.launch --screen &
sleep 5

python local_resnet.py --lin ${LIN} --ang ${ANG} --path="$BASE/$MODEL" --use_seg=${USE_SEG} --height $H --width $W --qsize=${Q_SIZE}
