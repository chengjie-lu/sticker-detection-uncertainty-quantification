#!/bin/bash
#SBATCH -p hgx2q
#SBATCH --job-name="mc_d"
#SBATCH -n 1
#SBATCH -c 32      #   256cores / 8 GPUs = 32 cores per GPU  Usually you do not even need that many
#SBATCH --gres=gpu:1
#SBATCH --time 6-00:00:00    # time (D-HH:MM:SS)

ulimit -s 10240
module purge
# module use /cm/shared/ex3-modules/latest/modulefiles
module load slurm/slurm/21.08.8
#module load cuda11.8/toolkit/11.8.0
module load cuda12.2
#module load cuda11.4
#module load cuda10.1/toolkit/10.1.243
source /home/chengjielu/anaconda3/bin/activate sticker-detector
export PYTHONPATH=root_path:$PYTHONPATH

#srun bash run.sh "mc_dropblock" "fasterrcnn_resnet50_fpn fasterrcnn_resnet50_fpn_v2" "0 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.50" "origimg/org origimg/adv_run_1 origimg/adv_run_2 origimg/adv_run_3 origimg/adv_run_4 origimg/adv_run_5 origimg/adv_run_6 origimg/adv_run_7 origimg/adv_run_8 origimg/adv_run_9 origimg/adv_run_10" "experiment_results_mc_dropblock" "1 3 5 7 9"
#srun bash run.sh "mc_dropblock" "retinanet_resnet50_fpn retinanet_resnet50_fpn_v2" "0 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.50" "origimg/org origimg/adv_run_1 origimg/adv_run_2 origimg/adv_run_3 origimg/adv_run_4 origimg/adv_run_5 origimg/adv_run_6 origimg/adv_run_7 origimg/adv_run_8 origimg/adv_run_9 origimg/adv_run_10" "experiment_results_mc_dropblock" "1 3 5 7 9"

#srun bash run.sh "mc_dropblock" "fasterrcnn_resnet50_fpn fasterrcnn_resnet50_fpn_v2" "0 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.50" "dalleimg/org dalleimg/adv_run_1 dalleimg/adv_run_2 dalleimg/adv_run_3 dalleimg/adv_run_4 dalleimg/adv_run_5 dalleimg/adv_run_6 dalleimg/adv_run_7 dalleimg/adv_run_8 dalleimg/adv_run_9 dalleimg/adv_run_10" "experiment_results_mc_dropblock" "1 3 5 7 9"
#srun bash run.sh "mc_dropblock" "retinanet_resnet50_fpn retinanet_resnet50_fpn_v2" "0 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.50" "dalleimg/org dalleimg/adv_run_1 dalleimg/adv_run_2 dalleimg/adv_run_3 dalleimg/adv_run_4 dalleimg/adv_run_5 dalleimg/adv_run_6 dalleimg/adv_run_7 dalleimg/adv_run_8 dalleimg/adv_run_9 dalleimg/adv_run_10" "experiment_results_mc_dropblock" "1 3 5 7 9"

srun bash run.sh "mc_dropblock" "fasterrcnn_resnet50_fpn fasterrcnn_resnet50_fpn_v2" "0 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.50" "sdimg/org sdimg/adv_run_1 sdimg/adv_run_2 sdimg/adv_run_3 sdimg/adv_run_4 sdimg/adv_run_5 sdimg/adv_run_6 sdimg/adv_run_7 sdimg/adv_run_8 sdimg/adv_run_9 sdimg/adv_run_10" "experiment_results_mc_dropblock" "1 3 5 7 9"
#srun bash run.sh "mc_dropblock" "retinanet_resnet50_fpn retinanet_resnet50_fpn_v2" "0 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.50" "sdimg/org sdimg/adv_run_1 sdimg/adv_run_2 sdimg/adv_run_3 sdimg/adv_run_4 sdimg/adv_run_5 sdimg/adv_run_6 sdimg/adv_run_7 sdimg/adv_run_8 sdimg/adv_run_9 sdimg/adv_run_10" "experiment_results_mc_dropblock" "1 3 5 7 9"