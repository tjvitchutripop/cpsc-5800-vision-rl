CUDA_VISIBLE_DEVICES=0 uv run src/train_ppo.py --rgb 
CUDA_VISIBLE_DEVICES=0 uv run src/train_ppo.py --segmentation 
CUDA_VISIBLE_DEVICES=0 uv run src/train_ppo.py --rgb --segmentation 
CUDA_VISIBLE_DEVICES=0 uv run src/train_ppo.py --depth 
CUDA_VISIBLE_DEVICES=0 uv run src/train_ppo.py --rgb --depth 
CUDA_VISIBLE_DEVICES=0 uv run src/train_ppo.py --segmentation --depth 
CUDA_VISIBLE_DEVICES=0 uv run src/train_ppo.py --rgb --segmentation --depth 