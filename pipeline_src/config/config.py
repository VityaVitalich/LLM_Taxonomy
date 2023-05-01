from dataclasses import dataclass, field
import torch
from typing import Dict, Any


@dataclass
class TaskConfig:
    gen_args: Dict[str, Any] = field(
        default_factory=lambda: {
            "num_beams": 3,
            "early_stopping": True,
            "max_new_tokens": 5,
        }
    )
    device: torch.device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu"
    )
    batch_size: int = 64
    dict_size: int = 51

    n_epochs: int = 20
    lr: float = 1e-4
    project_name: str = "taxonomy"
    show_every: int = 5
    save_every: int = 100
    validation: int = 0
    weight_decay: float = 1e-4
    base_factor: int = 48
    exp_name: str = "small_t5_debug"
    compute_metrics_every: int = 100
    full_log: int = 50
    warmup: int = 4000
    model_checkpoint: str = "google/flan-t5-small"
    max_length: int = 100
    block_size: int = 64
    mode: str = "train"
    data_path: str = "./"
    gold_path: str = "./"
    test_data_path: str = "./"
    test_gold_path: str = "./"
