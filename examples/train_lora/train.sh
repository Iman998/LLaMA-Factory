export CUDA_VISIBLE_DEVICES=6,7
# export MLFLOW_EXPERIMENT_NAME="/data/models/DeepSeek-R1-Distill-Llama-70B_full"
# export MLFLOW_TRACKING_URI="http://37.228.138.179:7891"
# export MLFLOW_FLATTEN_PARAMS="1"
# export MLFLOW_TAGS='{"dataset":"dolphin_r1_translated(30k)"}'
export DISABLE_VERSION_CHECK=1
export FORCE_TORCHRUN=1

python -m llamafactory.cli train /data/Iman/LLM/training/LLaMA-Factory/examples/train_lora/gemma3_lora_sft_test.yaml