# Optuna Hyperparameter Tuning for `main.py`

This script (`tune.py`) performs **hyperparameter tuning using Optuna** for models trained via `main.py`.  
Each Optuna trial launches a full training run, parses the resulting log file, and uses **NDCG@10** as the optimization objective.

---

## Overview

- Training is executed via:
```bash
  python -u main.py ...
```

* Hyperparameters are sampled by Optuna
* Each trial:

  * Is assigned a unique `--id` equal to the Optuna trial number
  * Writes logs to the standard logging directory
  * Produces a log file ending with test metrics
* The tuner extracts **NDCG@10** from the log and maximizes it

---

## Objective Metric

The objective value is **NDCG@10**, parsed from the final test output in the log file:

```
| NDCG @5|10: 0.0614     0.0702
```

In this example, the objective value is:

```
NDCG@10 = 0.0702
```

---

## Log File Convention

The tuner expects log files at:

```
./log/domain_<first_letters>/<method>_<id>.log
```

Example:

```
./log/domain_BM/FedDCSR_03.log
```

Where:

* `BMG` = first letters of domain names (`Books Movies`)
* `03` = zero-padded Optuna trial ID

---

## Tuned Hyperparameters

The following parameters are tuned by Optuna:

| Parameter      | Range                |
| -------------- | -------------------- |
| `lr`           | `[1e-4, 1e-2]` (log) |
| `weight_decay` | `[1e-6, 1e-3]` (log) |
| `mu`           | `[0.0, 1.0]`         |
| `frac`         | `[0.1, 1.0]`         |
| `temperature`  | `[0.05, 1.0]` (log)  |

All other arguments (batch size, epochs, domains, etc.) are fixed and passed as input arguments to the tuning script.

---

## Usage

### Basic Example

```bash
python tune.py \
  --method FedDCSR \
  --domains book movie\
  --batch_size 128 \
  --epochs 40 \
  --local_epoch 3 \
  --gpu 0 \
  --n_trials 50
```

---

## Command-Line Arguments

### Required

| Argument    | Description                               |
| ----------- | ----------------------------------------- |
| `--method`  | Model name (e.g. `FedDCSR`, `FedSASRec`)  |
| `--domains` | List of domains (positional in `main.py`) |

### Training Configuration

| Argument        | Default | Description            |
| --------------- | ------- | ---------------------- |
| `--epochs`      | 40     | Total training epochs  |
| `--local_epoch` | 3      | Local epochs per round |
| `--batch_size`  | 128     | Training batch size    |
| `--optimizer`   | `adam`  | Optimizer              |
| `--gpu`         | 0       | GPU ID                 |
| `--log_dir`     | `./log` | Log directory          |

### Optuna Settings

| Argument       | Default  | Description                     |
| -------------- | -------- | ------------------------------- |
| `--n_trials`   | 50       | Number of Optuna trials         |
| `--study_name` | `tuning` | Optuna study name               |
| `--storage`    | `sqlite:///optuna.db`   | Optional Optuna storage backend |

---


## Output

After completion, the script prints:

* Best NDCG@10 score
* Best hyperparameter configuration


Example:

```
Best trial:
0.0741
{'lr': 0.0012, 'weight_decay': 1e-5, 'mu': 0.3, 'frac': 0.8, 'temperature': 0.2}
```
* All trials results can be monitored on optuna-dashboard, excute this:
```
optuna-dashboard sqlite:///optuna.db --port=8889
```
then open the study name: `<METHOD>_<DOMAIN1>-<DOMAIN2>`, for example: `FedSASRec_book-movie`


---

## Dependencies

* Python ≥ 3.8
* `optuna`
* Same environment required to run `main.py`


