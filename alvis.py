#!/usr/bin/env bash
#SBATCH -A NAISS2023-22-310 -p alvis
#SBATCH -N 1 --gpus-per-node=A40:1
#SBATCH -t 7-00:00:00

#module load TensorFlow/2.5.0-fosscuda-2020b
#pwd
#ls
#python app.py

#module load torchtext/0.10.0-fosscuda-2020b-PyTorch-1.9.0  
module load PyTorch-bundle/1.13.1-foss-2022a-CUDA-11.7.0
#source ../../torch_test/bin/activate
pip install --upgrade pip
pip install transformers tokenizers
pip install nltk
pip install networkx
pip install colorlog
pip install omegconf
pip install torch
pip install typing
pip install datasets




CUDA_LAUNCH_BLOCKING=1 python pretraining_scratch.py --lr_type 'linear' # to add when we wan to resume training --model_checkpoint scratch_pretraining/checkpoint_    --trainer_checkpoint scratch_pretraining/checkpoint_