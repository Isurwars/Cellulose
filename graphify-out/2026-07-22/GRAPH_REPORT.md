# Graph Report - /home/isurwars/Projects/Cellulose  (2026-07-22)

## Corpus Check
- cluster-only mode — file stats not available

## Summary
- 167 nodes · 296 edges · 12 communities (10 shown, 2 thin omitted)
- Extraction: 99% EXTRACTED · 1% INFERRED · 0% AMBIGUOUS · INFERRED: 4 edges (avg confidence: 0.5)
- Token cost: 710 input · 30 output

## Graph Freshness
- Built from commit: `51c39bc6`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- Training Execution and Logging
- Model Fine-tuning and Evaluation
- build_train_loader
- build_heads
- losses.py
- train.sh
- extract_pdos.py
- Global Agent Directives
- run_finetune.py
- Contributing Guide
- Security Policy
- cellulose

## God Nodes (most connected - your core abstractions)
1. `run()` - 21 edges
2. `finetune()` - 17 edges
3. `evaluate_model()` - 14 edges
4. `AttentionPool` - 12 edges
5. `ForceResidualHead` - 12 edges
6. `resume_checkpoint()` - 12 edges
7. `save_checkpoint()` - 11 edges
8. `build_heads()` - 10 edges
9. `UncertaintyLossWeighting` - 10 edges
10. `build_optimizer()` - 10 edges

## Surprising Connections (you probably didn't know these)
- `UncertaintyLossWeighting` --uses--> `AttentionPool`  [INFERRED]
  trainer.py → models.py
- `UncertaintyLossWeighting` --uses--> `ForceResidualHead`  [INFERRED]
  trainer.py → models.py
- `UncertaintyLossWeighting` --uses--> `WeightHead`  [INFERRED]
  trainer.py → models.py
- `run()` --calls--> `load_custom_reference_energies()`  [EXTRACTED]
  train.py → data.py
- `run()` --calls--> `LocalSubgraphsDataset`  [EXTRACTED]
  train.py → data.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Agent Governance & Discovery Framework** — agents_agents, agents_rules_caveman_navigation, agents_skills_caveman_communication, agents_skills_graphify_maintenance [EXTRACTED 0.90]
- **PyTorch Development & Validation Suite** — agents_skills_python_pytorch_standards, agents_skills_pytorch_cuda_sanitizer, agents_skills_test_and_tea_loop, agents_rules_code_style_guide [EXTRACTED 0.90]
- **Community Contribution & Reporting Flow** — doc_contributing, github_pull_request_template, github_issue_template_bug_report, github_issue_template_feature_request [EXTRACTED 0.85]

## Communities (12 total, 2 thin omitted)

### Community 0 - "Training Execution and Logging"
Cohesion: 0.09
Nodes (28): _format_lr_summary(), FsyncFileHandler, init_wandb_from_config(), main(), Any, Namespace, Optimizer, Return a compact LR string: single value if uniform, backbone/heads split otherw (+20 more)

### Community 1 - "Model Fine-tuning and Evaluation"
Cohesion: 0.13
Nodes (30): _LRScheduler, ModelMixin, AttentionPool, ForceResidualHead, Softmax attention pooling to aggregate node representations into graph-level fea, Learns a domain-specific correction to pretrained force predictions.      Zero-i, ndarray, build_loss_weights() (+22 more)

### Community 2 - "build_train_loader"
Cohesion: 0.15
Nodes (17): AseSqliteDataset, build_train_loader(), cache_eval_frames(), extract_eigenvalues(), extract_weights(), load_custom_reference_energies(), LocalSubgraphsDataset, AbstractAtomsAdapter (+9 more)

### Community 3 - "build_heads"
Cohesion: 0.14
Nodes (10): build_heads(), device, Module, Tensor, Helper to instantiate prediction heads and place them on device.      Returns ``, Return per-atom force corrections [N_atoms, 3]., PDOS Weight Prediction Head with Group Normalisation.      Applies LayerNorm to, Simple linear-residual block with layer normalisation, SiLU activation, and drop (+2 more)

### Community 4 - "losses.py"
Cohesion: 0.22
Nodes (16): compute_eigenvalue_loss(), compute_electronic_losses(), compute_energy_loss(), compute_force_loss(), compute_r2(), compute_stress_loss(), compute_weight_loss(), device (+8 more)

### Community 5 - "train.sh"
Cohesion: 0.14
Nodes (13): CACHED_PATH_CACHE_ROOT, HF_HOME, KMP_AFFINITY, MKL_NUM_THREADS, MPLCONFIGDIR, NUMEXPR_NUM_THREADS, OMP_NUM_THREADS, OPENBLAS_NUM_THREADS (+5 more)

### Community 6 - "extract_pdos.py"
Cohesion: 0.31
Nodes (9): main(), create_orb_database(), main(), parse_manual_bands(), parse_manual_castep(), Writes a multi-frame dataset to Extended XYZ format., Creates the ASE SQLite database., transform_target() (+1 more)

### Community 7 - "Global Agent Directives"
Cohesion: 0.32
Nodes (8): Global Agent Directives, Caveman Navigation Protocol, Python & PyTorch Code Style Guide, Caveman Communication Protocol, Graphify Maintenance Skill, Python & PyTorch Standards, PyTorch CUDA Sanitizer, Test-and-Tea Loop

### Community 8 - "run_finetune.py"
Cohesion: 0.47
Nodes (5): get_gpu_vram_gb(), get_system_ram_gb(), main(), Return total system RAM in GB by parsing /proc/meminfo on Linux., Return GPU VRAM in GB for the specified device ID.

### Community 9 - "Contributing Guide"
Cohesion: 0.40
Nodes (5): Code of Conduct, Contributing Guide, Bug Report Template, Feature Request Template, Pull Request Template

## Knowledge Gaps
- **22 isolated node(s):** `cellulose`, `train.sh script`, `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS` (+17 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **2 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `run()` connect `Training Execution and Logging` to `Model Fine-tuning and Evaluation`, `build_train_loader`, `build_heads`?**
  _High betweenness centrality (0.160) - this node is a cross-community bridge._
- **Why does `finetune()` connect `Model Fine-tuning and Evaluation` to `Training Execution and Logging`, `losses.py`?**
  _High betweenness centrality (0.084) - this node is a cross-community bridge._
- **Why does `build_heads()` connect `build_heads` to `Training Execution and Logging`, `Model Fine-tuning and Evaluation`?**
  _High betweenness centrality (0.049) - this node is a cross-community bridge._
- **What connects `cellulose`, `train.sh script`, `OMP_NUM_THREADS` to the rest of the system?**
  _22 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Training Execution and Logging` be split into smaller, more focused modules?**
  _Cohesion score 0.08712121212121213 - nodes in this community are weakly interconnected._
- **Should `Model Fine-tuning and Evaluation` be split into smaller, more focused modules?**
  _Cohesion score 0.12903225806451613 - nodes in this community are weakly interconnected._
- **Should `build_train_loader` be split into smaller, more focused modules?**
  _Cohesion score 0.14761904761904762 - nodes in this community are weakly interconnected._