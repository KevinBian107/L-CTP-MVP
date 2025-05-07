import os
import yaml
import argparse

from toy_tda.train_topo import generate_and_train
from toy_tda.train_rl import train_hopper_with_topology_analysis


def run_topological_training():
    print("\n=== Starting Topological Dataset Training ===")
    config_path = os.path.join("toy_tda", "configs", "configs_original.yml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    generate_and_train(config)


def run_rl_training():
    print("\n=== Starting RL Training with Topology ===")
    config_path = os.path.join("toy_tda", "configs", "configs_rl.yml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    train_hopper_with_topology_analysis(**config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run training scripts for L-CTP")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["topo", "rl"],
        required=True,
        help="Select which training to run: 'topo' for dataset training, 'rl' for RL training",
    )
    args = parser.parse_args()

    if args.mode == "topo":
        run_topological_training()
    elif args.mode == "rl":
        run_rl_training()
