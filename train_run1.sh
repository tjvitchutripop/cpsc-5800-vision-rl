CUDA_VISIBLE_DEVICES=0 uv run src/train_ppo.py --depth --include_state
CUDA_VISIBLE_DEVICES=0 uv run src/train_ppo.py --rgb --depth --segmentation --include_state