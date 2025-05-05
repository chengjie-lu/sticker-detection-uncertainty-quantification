#!/bin/bash

uq_method=$1
models=$2
drop_rates=$3
datasets=$4
save_path=$5
block_sizes=$6

# models="fasterrcnn_resnet50_fpn fasterrcnn_resnet50_fpn_v2 ssd300_vgg16 ssdlite320_mobilenet_v3_large retinanet_resnet50_fpn retinanet_resnet50_fpn_v2"
# models="ssd300_vgg16 ssdlite320_mobilenet_v3_large"

# models="retinanet_resnet50_fpn"
# models="fasterrcnn_resnet50_fpn fasterrcnn_resnet50_fpn_v2 ssdlite320_mobilenet_v3_large"
# models="retinanet_resnet50_fpn retinanet_resnet50_fpn_v2 ssd300_vgg16"
# dropouts="0.3 0.4 0.5"

# dropouts="0.1 0.2 0.3 0.4 0.5"

# dropouts="0.45"
# datasets="sdimg/adv_run_8 sdimg/adv_run_9 sdimg/adv_run_10"

# datasets="sdimg/org sdimg/adv_run_1 sdimg/adv_run_2 sdimg/adv_run_3 sdimg/adv_run_4 sdimg/adv_run_5 sdimg/adv_run_6 sdimg/adv_run_7 sdimg/adv_run_8 sdimg/adv_run_9 sdimg/adv_run_10"

# datasets="origimg/org origimg/adv_run_1 origimg/adv_run_2 origimg/adv_run_3 origimg/adv_run_4 origimg/adv_run_5 origimg/adv_run_6 origimg/adv_run_7 origimg/adv_run_8 origimg/adv_run_9 origimg/adv_run_10"

# datasets="dalleimg/org dalleimg/adv_run_1 dalleimg/adv_run_2 dalleimg/adv_run_3 dalleimg/adv_run_4 dalleimg/adv_run_5 dalleimg/adv_run_6 dalleimg/adv_run_7 dalleimg/adv_run_8 dalleimg/adv_run_9 dalleimg/adv_run_10"

echo "$uq_method"
echo "$models"
echo "$drop_rates"
echo "$datasets"
echo "$save_path"
echo "$block_sizes"
echo "=================="


for model in ${models[*]}
do
	
	for drop_rate in ${drop_rates[*]}
	do
		#if [[ "$dropout" == '0.1' ]]; then
		#	echo "0.1 dropout"
    		#	continue
  		#fi
        	for d in ${datasets[*]}
        	do
        	  for block_size in ${block_sizes[*]}
        	  do
                	python run_cluster.py --uq_method="$uq_method" --model_n="$model" --dataset_p="$d" --drop_rate="$drop_rate" --save_folder="$save_path" --block_size="$block_size"
            done
        	done
	done
done

