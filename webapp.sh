#
# first build the virtualenv using the virtualenv.sh script 
#
# gradio webapp.py
torchrun --nproc_per_node $MP webapp.py
#
# or use CUDA_VISIBLE_DEVICES if you want to target a specific gpu device
# CUDA_VISIBLE_DEVICES=1 torchrun --nproc_per_node $MP webapp.py
#
