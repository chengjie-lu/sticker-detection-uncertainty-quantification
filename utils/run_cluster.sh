#!/bin/bash


# models="fasterrcnn_resnet50_fpn fasterrcnn_resnet50_fpn_v2 ssd300_vgg16 ssdlite320_mobilenet_v3_large retinanet_resnet50_fpn retinanet_resnet50_fpn_v2"
# models="fasterrcnn_resnet50_fpn fasterrcnn_resnet50_fpn_v2 ssd300_vgg16 ssdlite320_mobilenet_v3_large"
models="retinanet_resnet50_fpn_v2"

dropout=0.1
datasets="sdimg/adv_run_1 sdimg/adv_run_2 sdimg/adv_run_3 sdimg/adv_run_4 sdimg/adv_run_5"

# datasets="origimg/org origimg/adv_run_1 origimg/adv_run_2 origimg/adv_run_3 origimg/adv_run_4 origimg/adv_run_5 origimg/adv_run_6 origimg/adv_run_7 origimg/adv_run_8 origimg/adv_run_9 origimg/adv_run_10"


for m in ${models[*]}
do
	for d in ${datasets[*]}
	do
		python run_cluster.py --model_n=$m --dataset_p=$d --dropout=$dropout
	done
done
