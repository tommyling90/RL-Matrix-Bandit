from dataclasses import dataclass

@dataclass
class PickleContext:
    game_idx: int
    run_idx: int
    iter_idx: int
    save_every: int
    metrics: list
    cp_file: str