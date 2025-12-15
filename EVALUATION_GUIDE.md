# Model Evaluation Guide

This guide explains how to use the `evaluate_model.py` script to test trained PPO models.

## Basic Usage

```bash
python src/evaluate_model.py --model-path <path_to_model> --task <task_name> [options]
```

## Examples

### 1. Evaluate a PickCube model with RGB only
```bash
python src/evaluate_model.py \
    --model-path models/pickCube/pickCube_rgb.pt \
    --task PickCube-v1 \
    --rgb \
    --num-episodes 10
```

### 2. Evaluate a PickCube model with RGB + Depth
```bash
python src/evaluate_model.py \
    --model-path models/pickCube/pickCube_rgb_depth.pt \
    --task PickCube-v1 \
    --rgb \
    --depth \
    --num-episodes 10
```

### 3. Evaluate with RGB + Depth + Segmentation and save video
```bash
python src/evaluate_model.py \
    --model-path models/pickCube/pickCube_rgb_seg_depth.pt \
    --task PickCube-v1 \
    --rgb \
    --depth \
    --segmentation \
    --num-episodes 10 \
    --save-video \
    --video-dir eval_videos/pickCube_all_modalities
```

### 4. Evaluate a PushCube model
```bash
python src/evaluate_model.py \
    --model-path models/pushCube/pushCube_rgb.pt \
    --task PushCube-v1 \
    --rgb \
    --num-episodes 10
```

### 5. Evaluate with different feature extractor
```bash
python src/evaluate_model.py \
    --model-path models/pickCube/pickCube_rgb.pt \
    --task PickCube-v1 \
    --rgb \
    --feature-extractor resnet50 \
    --num-episodes 10
```

## Available Arguments

### Required Arguments
- `--model-path`: Path to the model checkpoint file (e.g., `models/pickCube/pickCube_rgb.pt`)
- At least one modality flag must be set (`--rgb`, `--depth`, or `--segmentation`)

### Task Selection
- `--task`: Environment to evaluate on
  - Options: `PickCube-v1`, `PushCube-v1`, `PushT-v1`
  - Default: `PickCube-v1`

### Input Modalities
- `--rgb`: Include RGB images
- `--depth`: Include depth images
- `--segmentation`: Include segmentation maps
- `--state`: Include state information (default: True)

**Note**: The modalities you select should match what the model was trained with!

### Model Architecture
- `--feature-extractor`: Feature extraction architecture
  - Options: `nature_cnn`, `theia`, `resnet50`, `densenet121`, `efficientnet_b0`
  - Default: `nature_cnn`

### Evaluation Settings
- `--num-episodes`: Number of episodes to run (default: 10)
- `--max-steps`: Maximum steps per episode (default: 200)
- `--seed`: Random seed for reproducibility (default: 42)
- `--deterministic`: Use deterministic actions (default: True)
- `--device`: Device to run on (`cuda` or `cpu`, auto-detected by default)

### Visualization
- `--render`: Render the environment during evaluation
- `--save-video`: Save evaluation videos
- `--video-dir`: Directory to save videos (default: `eval_videos`)

## Model File Naming Convention

Based on the models in your workspace:
- `pickCube_rgb.pt` - RGB only
- `pickCube_depth.pt` - Depth only
- `pickCube_seg.pt` - Segmentation only
- `pickCube_rgb_depth.pt` - RGB + Depth
- `pickCube_rgb_seg.pt` - RGB + Segmentation
- `pickCube_seg_depth.pt` - Segmentation + Depth
- `pickCube_rgb_seg_depth.pt` - All modalities

Make sure to use the appropriate flags that match your model's training configuration!

## Output

The script will print:
- Progress for each episode (reward and length)
- Summary statistics:
  - Mean reward ± standard deviation
  - Mean episode length ± standard deviation
  - Min/Max rewards
  - Success rate (if applicable)
