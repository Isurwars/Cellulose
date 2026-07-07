import json
import logging
from collections.abc import Callable
from typing import Any
import ase
import ase.data
import numpy as np
import torch
from torch.utils.data import BatchSampler, DataLoader, RandomSampler, SubsetRandomSampler

from orb_models.common.atoms.abstract_atoms_adapter import AbstractAtomsAdapter
from orb_models.common.dataset import augmentations, property_definitions
from orb_models.common.dataset.ase_sqlite_dataset import AseSqliteDataset
from orb_models.common.dataset.loaders import worker_init_fn
from orb_models.common.dataset.property_definitions import PROPERTIES, PropertyDefinition

NUM_BANDS = 250

def extract_eigenvalues(row: Any, dataset: str | None = None) -> torch.Tensor:
    """Return the 250 CASTEP Kohn-Sham eigenvalues for a structure as float32."""
    return torch.tensor(row.data["eigenvalues"], dtype=torch.float32)


def extract_weights(row: Any, dataset: str | None = None) -> torch.Tensor:
    """Return the per-atom 250-band PDOS weight vector for a structure as float32."""
    return torch.tensor(row.data["weights"], dtype=torch.float32)


# Register CASTEP electronic-structure properties in the Orb property registry.
PROPERTIES["eigenvalues"] = PropertyDefinition(
    name="eigenvalues",
    dim=NUM_BANDS,
    domain="graph",
    row_to_property_fn=extract_eigenvalues,
)
PROPERTIES["weights"] = PropertyDefinition(
    name="weights",
    dim=NUM_BANDS,
    domain="node",
    row_to_property_fn=extract_weights,
)


def load_custom_reference_energies(filepath: str) -> torch.Tensor:
    """Load custom reference energies from a file.

    Supports two formats:
      1. JSON: ``{"1": -13.6, "6": -1030.5, ...}`` or
               ``{"H": -13.6, "C": -1030.5, ...}``
      2. Text: One line per element — ``element_number energy`` or
               ``element_symbol energy``
    """
    atomic_numbers: dict[str, int] = ase.data.atomic_numbers  # type: ignore[assignment]
    ref_energies = torch.zeros(118)

    def _set_ref(key: str, value: float) -> None:
        try:
            z = int(key)
            if 1 <= z <= 118:
                ref_energies[z] = value
            else:
                logging.warning(f"Atomic number out of range: {key}")
        except ValueError:
            z = atomic_numbers.get(key, 0)
            if z:
                ref_energies[z] = value
            else:
                logging.warning(f"Unknown element symbol or invalid atomic number: {key}")

    # Try to load as JSON first
    try:
        with open(filepath) as f:
            data = json.load(f)

        for key, value in data.items():
            _set_ref(key, float(value))
        logging.info(f"Loaded reference energies from JSON file: {filepath}")

    except json.JSONDecodeError:
        with open(filepath) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                parts = line.split()
                if len(parts) != 2:
                    logging.warning(f"Skipping invalid line: {line}")
                    continue

                _set_ref(parts[0], float(parts[1]))
        logging.info(f"Loaded reference energies from text file: {filepath}")

    return ref_energies


def build_train_loader(
    dataset_name: str,
    dataset_path: str,
    num_workers: int,
    batch_size: int,
    atoms_adapter: AbstractAtomsAdapter,
    augmentation: bool | None = True,
    target_config: dict[str, list[str]] | None = None,
    train_indices: list[int] | None = None,
    **kwargs: Any,
) -> DataLoader:
    """Build the training DataLoader from an ASE SQLite database."""
    log_train = "Loading train datasets:\n"
    aug: list[Callable[[ase.Atoms], None]] = []
    if augmentation:
        aug = [augmentations.rotate_randomly]

    target_property_config = property_definitions.instantiate_property_config(target_config)
    dataset = AseSqliteDataset(
        dataset_name,
        dataset_path,
        atoms_adapter=atoms_adapter,
        target_config=target_property_config,
        augmentations=aug,
        **kwargs,
    )

    log_train += f"Total train dataset size: {len(dataset)} samples"
    logging.info(log_train)

    if train_indices is not None:
        sampler = SubsetRandomSampler(train_indices)
    else:
        sampler = RandomSampler(dataset)

    batch_sampler = BatchSampler(
        sampler,
        batch_size=batch_size,
        drop_last=False,
    )

    train_loader: DataLoader = DataLoader(
        dataset,
        num_workers=num_workers,
        worker_init_fn=worker_init_fn,
        collate_fn=atoms_adapter.batch,
        batch_sampler=batch_sampler,
        timeout=10 * 60 if num_workers > 0 else 0,
    )
    return train_loader


def cache_eval_frames(
    dataset: AseSqliteDataset,
    val_indices: set[int] | None = None,
) -> list[tuple[Any, dict[str, Any]]]:
    """Preprocess and cache all database frames for evaluation."""
    logging.info("Preprocessing and caching database frames for evaluation...")
    eval_frames: list[tuple[Any, dict[str, Any]]] = []

    indices = list(val_indices) if val_indices is not None else list(range(len(dataset)))
    for idx in indices:
        single_graph = dataset[idx]
        gt: dict[str, Any] = {
            "energy": single_graph.system_targets.get("energy").item() if "energy" in single_graph.system_targets else None,
            "forces": single_graph.node_targets.get("forces").cpu().numpy() if "forces" in single_graph.node_targets else None,
            "eigenvalues": single_graph.system_targets.get("eigenvalues").cpu().numpy() if "eigenvalues" in single_graph.system_targets else None,
            "weights": single_graph.node_targets.get("weights").cpu().numpy() if "weights" in single_graph.node_targets else None,
            "cell": single_graph.cell.cpu().numpy() if single_graph.cell is not None else None,
        }
        eval_frames.append((single_graph, gt))

    logging.info(f"Cached {len(eval_frames)} frames for evaluation.")
    return eval_frames
