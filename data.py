import json
import logging
import copy
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


class LocalSubgraphsDataset(torch.utils.data.Dataset):
    """Wraps an AseSqliteDataset to extract local neighbor environments (subgraphs) 
    centered around each atom of the parent structures.
    """
    def __init__(
        self,
        base_dataset: AseSqliteDataset,
        cutoff: float = 6.0,
        ref_energies: torch.Tensor | None = None,
        eigenvalue_mode: str = "filtered",
        eigenvalue_threshold: float = 0.15,
        local_pbc: bool = True,
        indices: list[int] | None = None,
    ):
        self.base_dataset = base_dataset
        self.cutoff = cutoff
        self.ref_energies = ref_energies
        self.eigenvalue_mode = eigenvalue_mode
        self.eigenvalue_threshold = eigenvalue_threshold
        self.local_pbc = local_pbc
        
        self.name = base_dataset.name + "_local"
        self.dtype = base_dataset.dtype
        self.db = base_dataset.db
        
        self.mapping = []
        target_indices = indices if indices is not None else range(len(self.base_dataset))
        for frame_idx in target_indices:
            row = self.base_dataset.db.get(frame_idx + 1)
            num_atoms = len(row.toatoms())
            for atom_idx in range(num_atoms):
                self.mapping.append((frame_idx, atom_idx))
                
    def __len__(self) -> int:
        return len(self.mapping)
        
    def __getitem__(self, idx: int) -> Any:
        frame_idx, center_atom_idx = self.mapping[idx]
        
        row = self.base_dataset.db.get(frame_idx + 1)
        parent_atoms = row.toatoms()
        
        parent_atoms.info = {}
        parent_atoms.info.update(self.base_dataset.feature_config.extract(row, self.base_dataset.name, "features"))
        parent_atoms.info.update(self.base_dataset.target_config.extract(row, self.base_dataset.name, "targets"))
        
        # Calculate MIC distance vectors and norms
        vecs = parent_atoms.get_distances(center_atom_idx, indices=None, mic=True, vector=True)
        dists = np.linalg.norm(vecs, axis=1)
        neighbor_indices = np.where(dists <= self.cutoff)[0]
        
        # Symmetrize/reorder neighbors so that the center atom is index 0
        center_pos_in_neighbors = np.where(neighbor_indices == center_atom_idx)[0][0]
        neighbor_indices = np.concatenate([[center_atom_idx], np.delete(neighbor_indices, center_pos_in_neighbors)])
        
        # Build the local subgraph atoms object depending on local_pbc mode
        sub_symbols = [parent_atoms.symbols[j] for j in neighbor_indices]
        
        if self.local_pbc:
            # Retain parent cell, PBC, and absolute wrapped coordinates
            sub_positions = parent_atoms.positions[neighbor_indices]
            sub_atoms = ase.Atoms(
                symbols=sub_symbols,
                positions=sub_positions,
                cell=parent_atoms.cell,
                pbc=parent_atoms.pbc,
            )
        else:
            # Cluster mode: remove PBC and cell, center center-atom at the origin
            sub_positions = vecs[neighbor_indices]
            sub_atoms = ase.Atoms(
                symbols=sub_symbols,
                positions=sub_positions,
                pbc=False,
            )
            
        # Reconstruct targets for the subgraph
        node_targets = {}
        graph_targets = {}
        
        # 1. Forces
        if "node_targets" in parent_atoms.info and "forces" in parent_atoms.info["node_targets"]:
            parent_forces = parent_atoms.info["node_targets"]["forces"]
            if isinstance(parent_forces, torch.Tensor):
                parent_forces = parent_forces.cpu().numpy()
            node_targets["forces"] = torch.tensor(parent_forces[neighbor_indices], dtype=torch.float32)
            
        # 2. PDOS Weights
        if "node_targets" in parent_atoms.info and "weights" in parent_atoms.info["node_targets"]:
            parent_weights = parent_atoms.info["node_targets"]["weights"]
            if isinstance(parent_weights, torch.Tensor):
                parent_weights_np = parent_weights.cpu().numpy()
            else:
                parent_weights_np = parent_weights
            node_targets["weights"] = torch.tensor(parent_weights_np[neighbor_indices], dtype=torch.float32)
            
            # Extract LDoS
            raw_weights = 1.0 / (1.0 + np.exp(-parent_weights_np))
            sub_raw_weights = raw_weights[neighbor_indices]
            ldos = np.sum(sub_raw_weights, axis=0)
            graph_targets["ldos"] = torch.tensor(ldos, dtype=torch.float32)
        else:
            ldos = None
            
        # 3. Eigenvalues
        if "graph_targets" in parent_atoms.info and "eigenvalues" in parent_atoms.info["graph_targets"]:
            parent_eigenvalues = parent_atoms.info["graph_targets"]["eigenvalues"]
            if isinstance(parent_eigenvalues, torch.Tensor):
                parent_eigenvalues = parent_eigenvalues.cpu().numpy()
                
            if ldos is not None:
                if self.eigenvalue_mode == "raw":
                    sub_eigenvalues = parent_eigenvalues
                elif self.eigenvalue_mode == "weighted":
                    norm_ldos = ldos / np.clip(ldos.max(), 1e-5, None)
                    sub_eigenvalues = parent_eigenvalues * norm_ldos
                elif self.eigenvalue_mode == "filtered":
                    sub_eigenvalues = np.where(ldos >= self.eigenvalue_threshold, parent_eigenvalues, 0.0)
                else:
                    sub_eigenvalues = parent_eigenvalues
            else:
                sub_eigenvalues = parent_eigenvalues
                
            graph_targets["eigenvalues"] = torch.tensor(sub_eigenvalues, dtype=torch.float32)
            
        # 4. Energy & Energy per Atom
        if "graph_targets" in parent_atoms.info and "energy" in parent_atoms.info["graph_targets"]:
            parent_energy = parent_atoms.info["graph_targets"]["energy"]
            if isinstance(parent_energy, torch.Tensor):
                parent_energy = parent_energy.item()
                
            def get_ref_energy(numbers, ref):
                if ref is None:
                    return 0.0
                return sum(ref[z].item() for z in numbers)
                
            parent_numbers = parent_atoms.numbers
            sub_numbers = sub_atoms.numbers
            
            e_ref_parent = get_ref_energy(parent_numbers, self.ref_energies)
            e_ref_sub = get_ref_energy(sub_numbers, self.ref_energies)
            
            e_excess = parent_energy - e_ref_parent
            e_avg_excess = e_excess / len(parent_atoms)
            
            sub_energy = e_avg_excess * len(sub_atoms) + e_ref_sub
            
            graph_targets["energy"] = torch.tensor([sub_energy], dtype=torch.float32)
            graph_targets["energy_per_atom"] = torch.tensor([e_avg_excess], dtype=torch.float32)
            
        sub_atoms.info = {
            "node_targets": node_targets,
            "graph_targets": graph_targets,
        }
        
        for augmentation in self.base_dataset.augmentations:
            augmentation(sub_atoms)
            
        return self.base_dataset._from_ase_atoms(
            atoms=sub_atoms,
            device=torch.device("cpu"),
            output_dtype=self.base_dataset.dtype,
            system_id=idx,
            edge_method="knn_alchemi",
            half_supercell=False,
            graph_construction_dtype=self.base_dataset.dtype,
        )


def build_train_loader(
    dataset_name: str,
    dataset_path: str,
    num_workers: int,
    batch_size: int,
    atoms_adapter: AbstractAtomsAdapter,
    augmentation: bool | None = True,
    target_config: dict[str, list[str]] | None = None,
    train_indices: list[int] | None = None,
    use_local_graphs: bool = False,
    local_cutoff: float = 6.0,
    ref_energies: torch.Tensor | None = None,
    eigenvalue_mode: str = "filtered",
    eigenvalue_threshold: float = 0.15,
    local_pbc: bool = True,
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

    if use_local_graphs:
        dataset = LocalSubgraphsDataset(
            dataset,
            cutoff=local_cutoff,
            ref_energies=ref_energies,
            eigenvalue_mode=eigenvalue_mode,
            eigenvalue_threshold=eigenvalue_threshold,
            local_pbc=local_pbc,
            indices=train_indices,
        )
        log_train += f"Total train dataset size (local subgraphs): {len(dataset)} samples"
        sampler = RandomSampler(dataset)
    else:
        log_train += f"Total train dataset size: {len(dataset)} samples"
        if train_indices is not None:
            sampler = SubsetRandomSampler(train_indices)
        else:
            sampler = RandomSampler(dataset)

    logging.info(log_train)

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
