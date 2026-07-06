export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
uv run python train_electronic.py \
  --data_path cellulose_finetuning.db \
  --base_model orb_v3_direct_omol \
  --custom_reference_energies refs.json \
  --energy_loss_weight 0.01 \
  --forces_loss_weight 20.0 \
  --stress_loss_weight 0.0 \
  --eigenvalue_loss_weight 0.05 \
  --weight_loss_weight 50.0 \
  --scheduler flat_cosine \
  --unfreeze_epoch 10 \
  --backbone_lr 3e-4 \
  --lr 1e-3 \
  --min_lr 1e-5 \
  --weight_head_noise_std 5e-5 \
  --weight_head_noise_interval 2 \
  --batch_size 1 \
  --num_steps 0 \
  --max_epochs 51 