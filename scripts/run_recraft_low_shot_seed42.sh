#!/usr/bin/env bash
set -euo pipefail

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate aide

AIDE_ROOT="$HOME/projects/AIDE"
cd "$AIDE_ROOT"

for SHOT in 0025 0050 0100 0250 0500; do
    RUN="results/recraft_v3_lowshot/shot_${SHOT}_seed42"
    TRAIN_DATA="dataset/adaptation/recraft_v3_seed42/low_shot_equal_budget/shot_${SHOT}"
    PREDICTIONS="$RUN/source_test/_RecraftV3_predictions.csv"

    echo
    echo "Starting shot_${SHOT}"

    if [ ! -f "$RUN/checkpoint-best.pth" ]; then
        mkdir -p "$RUN/tensorboard"

        CUDA_VISIBLE_DEVICES=0 \
        systemd-inhibit --what=sleep \
          --why="Training Recraft low-shot model" \
          python main_finetune.py \
            --model AIDE \
            --data_path "$TRAIN_DATA" \
            --eval_data_path dataset/adaptation/recraft_v3_seed42/aide_input/val \
            --init_checkpoint pretrained_ckpts/GenImage_train.pth \
            --trainable_modules fc \
            --batch_size 2 \
            --epochs 1 \
            --lr 1e-4 \
            --min_lr 1e-6 \
            --num_workers 0 \
            --auto_resume False \
            --dist_eval False \
            --output_dir "$RUN" \
            --log_dir "$RUN/tensorboard" \
          2>&1 | tee "$RUN/training.log"
    else
        echo "Training checkpoint already exists; skipping training."
    fi

    if [ ! -f "$PREDICTIONS" ]; then
        mkdir -p "$RUN/source_test"

        CUDA_VISIBLE_DEVICES=0 \
        systemd-inhibit --what=sleep \
          --why="Evaluating Recraft low-shot model" \
          python main_finetune.py \
            --model AIDE \
            --batch_size 2 \
            --data_path "$TRAIN_DATA" \
            --eval_data_path dataset/adaptation/recraft_v3_seed42/aide_eval/source_test_clean \
            --init_checkpoint "$RUN/checkpoint-best.pth" \
            --trainable_modules fc \
            --eval True \
            --dist_eval False \
            --num_workers 0 \
            --auto_resume False \
            --output_dir "$RUN/source_test" \
          2>&1 | tee "$RUN/source_test/evaluation.log"
    else
        echo "Test predictions already exist; skipping evaluation."
    fi

    echo "Finished shot_${SHOT}"
done

echo
echo "All requested runs completed."