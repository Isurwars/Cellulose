# Optimize PyTorch CPU performance on many-core systems (avoid thread oversubscription thrashing)
export OMP_NUM_THREADS=32
export MKL_NUM_THREADS=32
export OPENBLAS_NUM_THREADS=32
export VECLIB_MAXIMUM_THREADS=32
export NUMEXPR_NUM_THREADS=32

# CPU thread pinning optimizations (improves cache locality)
export KMP_AFFINITY="granularity=fine,compact,1,0"

# Redirect Warp cache directory to avoid Permission denied on /home/isurwars
export WARP_CACHE_PATH="$(pwd)/.cache/warp"

# Redirect all other common machine learning cache directories to the project folder
export CACHED_PATH_CACHE_ROOT="$(pwd)/.cache/cached_path"
export HF_HOME="$(pwd)/.cache/huggingface"
export TORCH_HOME="$(pwd)/.cache/torch"
export MPLCONFIGDIR="$(pwd)/.cache/matplotlib"


uv run python run_finetune.py \
  --data_path cellulose.db \
  --base_model orb_v3_direct_omol \
  --custom_reference_energies refs.json \
  --energy_loss_weight 0.1 \
  --stress_loss_weight 0.0 \
  --forces_loss_weight 0.1 \
  --eigenvalue_loss_weight 0.01 \
  --weight_loss_weight 1.0 \
  --use_force_residual \
  --scheduler flat_cosine \
  --unfreeze_epoch 0 \
  --backbone_lr 1e-4 \
  --lr 1e-3 \
  --min_lr 1e-5 \
  --eval_every_x_epochs 1 \
  --max_epochs 21 \
  --warmup_epochs 3 \
  --normalize_eigenvalues \
  --normalize_forces \
  --use_uncertainty_weights \
  --batch_size 16 \
  --accumulation_steps 2 \
  --num_workers 16 \
  "$@"