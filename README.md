# STAR-MASK

STAR-MASK is a Python implementation of a masked Dueling Double-DQN scheduler for CSQF routing and time-sensitive deterministic flow scheduling under topology dynamics and anomaly events.

The codebase contains:

- `main.py`: centralized training with parameter sharing.
- `data_define.py`: training environment, flow model, CSQF executor, Dueling-DQN agent, and masked action selection.
- `ReplayBuffer.py`: prioritized replay buffer used by training.
- `Kuiper_Shell.py`: Walker-Delta LEO snapshot generation, including the 10 x 12 evaluation constellation.
- `evaluate_T.py`: LEO evaluation environment and model definitions.
- `LEO_evaluate.py`: checkpoint evaluation and scenario JSON generation.

## Method Summary

Each episode builds flow instances and anomaly settings, constructs port-centric observations and valid-port masks, selects valid actions with masked Dueling-DQN and exploration, executes earliest-legal continuous-block CSQF allocation, and stores transitions in a prioritized replay buffer. Training uses Double-DQN targets and an importance-weighted Huber loss with a periodically updated target network.

## Experiment Settings

The LEO evaluation configuration follows the paper settings:

- Satellites: `120`
- Orbital planes: `10`
- Satellites per plane: `12`
- Walker phasing parameter: `3`
- Inclination: `42 deg`
- Altitude: `610 km`
- ISL maximum length: `6000 km`
- CSQF queues per port: `4`
- Queue length: `10`
- Evaluation window / flow lifetime: `60 s`
- Deadline: `[150, 200] ms`
- Packet period: `{6, 12, 18} ms`
- Packets per period: `{1, 2, 3}`

The training entry point uses Adam with learning rate `5e-5`, discount factor `0.99`, batch size `128`, PER capacity `60000`, and traffic stages of `200` flows for episodes `1-2000` and `700` flows from episode `2001`.

## Installation

Python 3.12 is recommended.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

For CUDA training, install the PyTorch build matching your GPU and CUDA runtime from the official PyTorch instructions.

## Training

```bash
python main.py
```

By default, checkpoints and `train_metrics.csv` are written to `D:\Agent_over` as configured in `main.py`. Adjust `ckpt_dir` and `save_dir` before long runs if you want artifacts inside the repository or another output directory.

## Evaluation

```bash
python LEO_evaluate.py --ckpt D:\Agent_over\csqf_agent_ep10000.pth --runs 5 --seed 1000
```

This evaluates with exploration disabled and writes generated scenario JSON files under `eval_suites/`.

## Notes

- `main.py` trains on the static graph in `data_define.py`; the LEO evaluation uses the dynamic Walker-Delta snapshotter in `Kuiper_Shell.py`.
- Bandwidth modeling is intentionally not emphasized in this release.
- Large checkpoints and generated evaluation suites are ignored by Git.
