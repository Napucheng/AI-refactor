"""Train and evaluate a Push-T imitation policy."""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import tyro
import wandb
from torch.utils.data import DataLoader

from hw1_imitation.data import (
    Normalizer,
    PushtChunkDataset,
    download_pusht,
    load_pusht_zarr,
)
from hw1_imitation.model import build_policy, PolicyType
from hw1_imitation.evaluation import Logger

LOGDIR_PREFIX = "exp"


@dataclass
class TrainConfig:
    # The path to download the Push-T dataset to.
    data_dir: Path = Path("data")

    # The policy type -- either MSE or flow.
    policy_type: PolicyType = "mse"
    # The number of denoising steps to use for the flow policy (has no effect for the MSE policy).
    flow_num_steps: int = 10
    # The action chunk size.
    chunk_size: int = 8

    batch_size: int = 128
    lr: float = 3e-4
    weight_decay: float = 0.0
    hidden_dims: tuple[int, ...] = (256, 256, 256)
    # The number of epochs to train for.
    num_epochs: int = 400
    # How often to run evaluation, measured in training steps.
    eval_interval: int = 10_000
    num_video_episodes: int = 5
    video_size: tuple[int, int] = (256, 256)
    # How often to log training metrics, measured in training steps.
    log_interval: int = 100
    # Random seed.
    seed: int = 42
    # WandB project name.
    wandb_project: str = "hw1-imitation"
    # Experiment name suffix for logging and WandB.
    exp_name: str | None = None


def parse_train_config(
    args: list[str] | None = None,
    *,
    defaults: TrainConfig | None = None,
    description: str = "Train a Push-T MLP policy.",
) -> TrainConfig:
    defaults = defaults or TrainConfig()
    return tyro.cli(
        TrainConfig,
        args=args,
        default=defaults,
        description=description,
    )


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def config_to_dict(config: TrainConfig) -> dict[str, Any]:
    data = asdict(config)
    for key, value in data.items():
        if isinstance(value, Path):
            data[key] = str(value)
    return data


def run_training(config: TrainConfig) -> None:
    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    zarr_path = download_pusht(config.data_dir)
    states, actions, episode_ends = load_pusht_zarr(zarr_path)
    normalizer = Normalizer.from_data(states, actions)

    dataset = PushtChunkDataset(
        states,
        actions,
        episode_ends,
        chunk_size=config.chunk_size,
        normalizer=normalizer,
    )

    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
    )

    model = build_policy(
        config.policy_type,
        state_dim=states.shape[1],
        action_dim=actions.shape[1],
        chunk_size=config.chunk_size,
        hidden_dims=config.hidden_dims,
    ).to(device)

    exp_name = f"seed_{config.seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if config.exp_name is not None:
        exp_name += f"_{config.exp_name}"
    log_dir = Path(LOGDIR_PREFIX) / exp_name
    wandb.init(
        project=config.wandb_project, config=config_to_dict(config), name=exp_name
    )
    logger = Logger(log_dir)

    ### TODO: PUT YOUR MAIN TRAINING LOOP HERE ###
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )

    if device.type == "cuda":
        try:
            model = torch.compile(model)
            print("✅ torch.compile 已启用")
        except Exception as e:
            print(f"⚠️ torch.compile 不可用：{e}")

    total_steps = 0
    best_eval_reward = -float("inf")
    
    # 导入评估函数
    from hw1_imitation.evaluation import evaluate_policy

    for epoch in range(config.num_epochs):
        for batch in loader:
            obs, actions = batch
            obs, actions = obs.to(device), actions.to(device)
            
            loss = model.compute_loss(obs, actions)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if total_steps % config.log_interval == 0:
                wandb.log({
                    "train/loss": loss.item(),
                    "train/epoch": epoch,
                    "train/step": total_steps,
                })
                print(f"Step {total_steps} | Epoch {epoch} | Loss: {loss.item():.4f}")
            
            if total_steps > 0 and total_steps % config.eval_interval == 0:
                print(f"Evaluating at step {total_steps}...")
                
                evaluate_policy(
    model=model,
    normalizer=normalizer,
    device=device,
    chunk_size=config.chunk_size,
    video_size=config.video_size,
    num_video_episodes=config.num_video_episodes,
    flow_num_steps=config.flow_num_steps if config.policy_type == "flow" else 1,
    step=total_steps,
    logger=logger,
)
                
                
            
            total_steps += 1
        
        print(f"Epoch {epoch + 1}/{config.num_epochs} completed")
    
    logger.save_model(model.state_dict(), "model_final.pt")
    print("Training completed!")

    logger.dump_for_grading()


def main() -> None:
    config = parse_train_config()
    run_training(config)


if __name__ == "__main__":
    main()
