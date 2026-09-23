"""Suite-wide settings for paper/fixtures.

train_wm.py streams to Weights & Biases by default, and several tests run the trainer in-process.
`WANDB_MODE=disabled` makes a real `wandb` package, where one is installed, open a local no-op run,
so no test can reach the network or Rohan's W&B project; the W&B tests use a stub module instead.
"""
import os

os.environ["WANDB_MODE"] = "disabled"
