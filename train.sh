export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python finetune.py \
  --data_path cellulose_finetuning.db \
  --base_model orb_v3_conservative_omol \
  --custom_reference_energies refs.json \
  --energy_loss_weight 0.1 \
  --forces_loss_weight 1.0 \
  --stress_loss_weight 0.0 \
  --batch_size 2 \
  --max_epochs 50