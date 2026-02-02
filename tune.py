import argparse
import subprocess
import optuna
import os
import re
import sys
from pathlib import Path


def parse_ndcg10(log_path):
    if not log_path.exists():
        raise RuntimeError(f"Log file not found: {log_path}")

    ndcg10 = None
    pattern = re.compile(r"NDCG @5\|10:\s*([0-9.]+)\s+([0-9.]+)")

    with open(log_path, "r") as f:
        for line in reversed(f.readlines()):
            m = pattern.search(line)
            if m:
                ndcg10 = float(m.group(2))
                break

    if ndcg10 is None:
        raise RuntimeError("Failed to parse NDCG@10 from log")

    return ndcg10


def objective(trial, args):
    # ===== sample hyperparameters =====
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    mu = trial.suggest_float("mu", 0.0, 1.0)
    frac = trial.suggest_float("frac", 0.1, 1.0)
    temperature = trial.suggest_float("temperature", 0.05, 1.0, log=True)

    trial_id = str(trial.number)

    # ===== build command =====
    cmd = [
        sys.executable, "-u", "main.py",
        "--method", args.method,
        "--epochs", str(args.epochs),
        "--local_epoch", str(args.local_epoch),
        "--batch_size", str(args.batch_size),
        "--optimizer", args.optimizer,
        "--lr", str(lr),
        "--weight_decay", str(weight_decay),
        "--mu", str(mu),
        "--frac", str(frac),
        "--temperature", str(temperature),
        "--gpu", str(args.gpu),
        "--id", trial_id,
    ]

    if args.do_eval:
        cmd.append("--do_eval")

    # positional domains
    cmd.extend(args.domains)

    # ===== run training =====
    subprocess.run(cmd, check=True)

    # ===== locate log file =====
    domain_tag = "".join(d[0] for d in args.domains)
    log_dir = Path(args.log_dir) / f"domain_{domain_tag}"
    log_file = log_dir / f"{args.method}_{int(trial_id):02d}.log"

    ndcg10 = parse_ndcg10(log_file)
    return ndcg10


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True)
    parser.add_argument("--domains", nargs="+", required=True)
    parser.add_argument("--log_dir", default="./log")
    parser.add_argument("--gpu", default=0, type=int)

    # fixed training args
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--local_epoch", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--optimizer", default="adam")

    # flags
    parser.add_argument("--load_prep", action="store_true")
    parser.add_argument("--do_eval", action="store_true")

    # optuna
    parser.add_argument("--n_trials", type=int, default=50)
    parser.add_argument("--storage", default='sqlite:///optuna.db')

    args = parser.parse_args()

    study_name = f"{args.method}_{'-'.join(args.domains)}"

    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        storage=args.storage,
        load_if_exists=True,
    )

    study.optimize(lambda t: objective(t, args), n_trials=args.n_trials)

    print("Best trial:")
    print(study.best_trial.value)
    print(study.best_trial.params)


if __name__ == "__main__":
    main()
