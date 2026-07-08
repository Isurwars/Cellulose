export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
uv run python train_electronic.py \
  --data_path cellulose_finetuning.db \
  --base_model orb_v3_direct_omol \
  --custom_reference_energies refs.json \
  --energy_loss_weight 0.0 \
  --stress_loss_weight 0.0 \
  --forces_loss_weight 1.0 \
  --eigenvalue_loss_weight 0.05 \
  --weight_loss_weight 1.0 \
  --pdos_peak_boost 15.0 \
  --pdos_active_threshold 0.05 \
  --use_force_residual \
  --force_loss_type mse \
  --scheduler cosine \
  --unfreeze_epoch 10 \
  --backbone_lr 5e-5 \
  --lr 1e-3 \
  --min_lr 1e-5 \
  --batch_size 4 \
  --num_steps 0 \
  --eval_every_x_epochs 1 \
  --max_epochs 201 \
  --warmup_epochs 1 \
  --normalize_eigenvalues \
  --normalize_forces
  # Optional: add --use_uncertainty_weights to enable learnable multi-task loss scaling