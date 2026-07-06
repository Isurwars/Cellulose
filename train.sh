export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
uv run python train_electronic.py \
  --data_path cellulose_finetuning.db \
  --base_model orb_v3_direct_omol \
  --custom_reference_energies refs.json \
  --energy_loss_weight 0.01 \
  --forces_loss_weight 1.0 \
  --stress_loss_weight 0.0 \
  --eigenvalue_loss_weight 0.1 \
  --weight_loss_weight 20.0 \
  --scheduler cosine \
  --unfreeze_epoch 5 \
  --backbone_lr 1e-3 \
  --min_lr 5e-6 \
  --batch_size 1 \
  --max_epochs 201