# QUBE-Servo 2 Balance RL

This project trains and evaluates reinforcement-learning controllers for the
QUBE-Servo 2 rotary inverted pendulum. It includes a voltage-driven simulator,
Soft Actor-Critic (SAC) training, sim-to-real reference profiles, direct
hardware rollout scripts, upright-balance training, and report-style analysis
plots.

The main workflow is SAC training in simulation followed by direct testing on
the real QUBE hardware. For smoother upright behavior, the simulator includes
an upright-balance reward profile that trains from near-upright starts and
penalizes aggressive velocity and voltage changes.

## Setup

Install the dependencies:

```bash
python3 -m pip install -r requirements.txt
```

## Train SAC

Run training with the default swing-up simulator settings:

```bash
python3 scripts/train/train_sac.py
```

For the sim-to-real reference profile with recovery disturbances:

```bash
python3 scripts/train/train_sac.py \
  --reference-profile \
  --recovery-disturbances \
  --smooth-balance \
  --model-out models/sac_qube_servo2_reference
```

For a smoother direct upright-balance policy:

```bash
python3 scripts/train/train_sac.py \
  --upright-balance-profile \
  --timesteps 100000 \
  --model-out models/sac_qube_upright_balance
```


Useful training parameters:

| Parameter | Default | Description |
| --- | --- | --- |
| `--timesteps` | `500000` | Number of training steps. When resuming, this means additional steps. |
| `--log-dir` | `runs/sac_qube` | TensorBoard log and checkpoint directory. |
| `--model-out` | `models/sac_qube_servo2` | Output path for the trained model. Stable-Baselines3 adds `.zip`. |
| `--seed` | `7` | Random seed. |
| `--reference-profile` | `False` | Use the 6 V sim-to-real parameter profile. |
| `--upright-balance-profile` | `False` | Train from upright with balance-focused reward shaping. |
| `--recovery-disturbances` | `False` | Train with random external torque pulses. |
| `--smooth-balance` | `False` | Penalize rapid voltage changes near upright in the swing-up profile. |
| `--resume-from` | `None` | Resume from a saved SAC `.zip` checkpoint. |
| `--replay-buffer` | `None` | Optional replay buffer `.pkl` to load when resuming. |


## Evaluate In Simulation

Evaluate a trained SAC policy:

```bash
python3 scripts/evaluate/evaluate_sac.py \
  --reference-profile \
  --model models/with_recovery_disturbances/sac_qube_reference_with_recovery_disturbances_2m.zip \
  --episodes 3 \
  --seconds 20
```

Evaluate an upright-balance policy:

```bash
python3 scripts/evaluate/evaluate_sac.py \
  --upright-balance-profile \
  --model models/sac_qube_upright_balance.zip \
  --episodes 3 \
  --seconds 20 \
  --csv runs/sac_upright_balance_eval.csv
```

Save a rollout CSV for plotting:

```bash
python3 scripts/evaluate/evaluate_sac.py \
  --reference-profile \
  --model models/with_recovery_disturbances/sac_qube_reference_with_recovery_disturbances_2m.zip \
  --no-render \
  --csv runs/sac_eval.csv
```

Validate the classical swing-up/balance baseline:

```bash
python3 scripts/evaluate/validate_sim.py --controller ais --csv runs/classic_validation.csv
```

The original simulator baseline is still available:

```bash
python3 scripts/evaluate/validate_sim.py --controller servo --csv runs/classic_validation_servo.csv
```

## Run SAC On Real QUBE

Flash `teensy_qube_serial.ino` to the Teensy first.

Run a trained SAC policy directly on the hardware:

```bash
python3 scripts/train/run_sac_on_qube.py \
  --reference-profile \
  --model models/with_recovery_disturbances/sac_qube_reference_with_recovery_disturbances_2m.zip \
  --out runs/real_sac_rollout.csv
```

Run a trained upright-balance policy directly on the hardware:

```bash
python3 scripts/train/run_sac_on_qube.py \
  --reference-profile \
  --model models/sac_qube_upright_balance.zip \
  --out runs/real_sac_upright_balance.csv
```

Parameters:

| Parameter | Default | Description |
| --- | --- | --- |
| `--model` | `models/sac_qube_servo2.zip` | Trained SAC model to load. |
| `--calibration` | `runs/qube_calibration.json` | Calibration JSON for observation scaling and voltage limits. |
| `--reference-profile` | `False` | Use the built-in 6 V sim-to-real profile. |
| `--port` | `auto` | Serial port for the QUBE. |
| `--seconds` | `10.0` | Run duration. |
| `--rate` | `300.0` | Control loop rate in Hz. |
| `--max-voltage` | Profile voltage limit | Optional sent-voltage clipping limit. |
| `--out` | `runs/real_sac_rollout.csv` | CSV output log path. |
| `--dry-run` | `False` | Predict actions but send zero motor voltage. |

## Analysis And Plots

Generate plots and a metrics CSV from available rollout logs:

```bash
python3 scripts/analysis/generate_report_graphs.py
```

Summarize a hardware or simulator CSV:

```bash
python3 scripts/analysis/analyze_hardware_log.py runs/real_sac_rollout.csv
```

Compare two rollout CSVs:

```bash
python3 scripts/analysis/compare_rollouts.py \
  --left runs/classic_validation.csv \
  --right runs/sac_upright_balance_eval.csv \
  --left-label classical \
  --right-label sac_upright_balance \
  --output results/report_figures/classical_vs_sac_balance.png
```

Generated figures are written to `results/report_figures/` by default.
