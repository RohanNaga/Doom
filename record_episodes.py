"""
Record lossless per-tic ViZDoom episodes with the public gameNgen-repro PPO agent.

One process = one ViZDoom instance = a range of episodes. Every engine tic is
recorded (35 per second), with the action the agent holds during it, so any
frame stride can be chosen later. Frames are stored as PNG bytes inside a
per-episode parquet file, which is also the Hugging Face release layout.

Semantics of a row: `frame` is the screen at tic `tic`; `action` is the action
applied from this tic to the next (the agent decides every ACTION_REPEAT tics
and the environment holds the choice). Game variables are read at the same tic.

Usage (from anywhere; the repro checkout is needed for the agent and wad):
    python record_episodes.py --repro-dir /sata2/data/rnagabhi/doom/gameNgen-repro \
        --out-dir /sata2/data/rnagabhi/doom/raw --worker-id 0 --num-workers 16 \
        --episodes 60 --seed 0
"""
import argparse
import io
import json
import os
import sys
import time

import numpy as np
import torch


ACTION_REPEAT = 4          # agent decides every 4 tics, as in the PPO training and the HF dataset
SCENARIO = "deathmatch_simple"
MODEL_REL = "logs/models/deathmatch_simple/best_model.zip"


def png_bytes(frame_u8, compress_level):
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(frame_u8).save(buf, format="PNG", compress_level=compress_level)
    return buf.getvalue()


def build_env_and_agent(repro_dir):
    """Replicates load_model_generate_dataset.py's construction, level fixed at 5."""
    ppo_dir = os.path.join(repro_dir, "ViZDoomPPO")
    os.chdir(ppo_dir)                 # scenarios/ and bots.cfg are resolved relative to cwd
    sys.path.insert(0, ppo_dir)
    from common import envs
    from train_ppo_parallel import DoomWithBotsCurriculum, game_instance
    from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage

    kwargs = dict(frame_skip=1, frame_processor=envs.default_frame_processor, n_bots=8,
                  shaping=True, initial_level=5, max_level=5, rolling_mean_length=10)
    env = VecTransposeImage(DummyVecEnv([lambda: DoomWithBotsCurriculum(game_instance(SCENARIO), **kwargs)]))
    agent = envs.load_model(MODEL_REL, env)
    game = env.venv.envs[0].game
    return env, agent, game


def record_episode(env, agent, game, episode_id, compress_level):
    from vizdoom import GameVariable
    import pyarrow as pa

    obs = env.reset()
    rows = {k: [] for k in ["tic", "action", "health", "ammo", "frags", "pos_x", "pos_y", "angle", "frame"]}
    action = None
    tic = 0
    t0 = time.time()
    while True:
        state = game.get_state()
        if state is None or game.is_episode_finished():
            break
        if tic % ACTION_REPEAT == 0:
            action, _ = agent.predict(obs)
        rows["tic"].append(tic)
        rows["action"].append(int(np.asarray(action).item()))
        rows["health"].append(int(game.get_game_variable(GameVariable.HEALTH)))
        rows["ammo"].append(int(game.get_game_variable(GameVariable.SELECTED_WEAPON_AMMO)))
        rows["frags"].append(int(game.get_game_variable(GameVariable.FRAGCOUNT)))
        rows["pos_x"].append(float(game.get_game_variable(GameVariable.POSITION_X)))
        rows["pos_y"].append(float(game.get_game_variable(GameVariable.POSITION_Y)))
        rows["angle"].append(float(game.get_game_variable(GameVariable.ANGLE)))
        rows["frame"].append(png_bytes(np.ascontiguousarray(state.screen_buffer), compress_level))
        obs, _, done, _ = env.step(action)   # DummyVecEnv auto-resets on done; the loop exits above
        tic += 1
        if done[0]:
            break
    n = len(rows["tic"])
    table = pa.table({
        "episode_id": pa.array([episode_id] * n, pa.int32()),
        "tic": pa.array(rows["tic"], pa.int32()),
        "action": pa.array(rows["action"], pa.int8()),
        "health": pa.array(rows["health"], pa.int16()),
        "ammo": pa.array(rows["ammo"], pa.int16()),
        "frags": pa.array(rows["frags"], pa.int16()),
        "pos_x": pa.array(rows["pos_x"], pa.float32()),
        "pos_y": pa.array(rows["pos_y"], pa.float32()),
        "angle": pa.array(rows["angle"], pa.float32()),
        "frame": pa.array(rows["frame"], pa.binary()),
    })
    stats = dict(episode_id=episode_id, tics=n, seconds=time.time() - t0,
                 png_bytes_mean=float(np.mean([len(b) for b in rows["frame"]])) if n else 0.0,
                 frags=rows["frags"][-1] if n else 0)
    return table, stats


def main(args):
    import pyarrow.parquet as pq
    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(args.seed * 1000 + args.worker_id)
    np.random.seed(args.seed * 1000 + args.worker_id)
    env, agent, game = build_env_and_agent(args.repro_dir)
    game.set_seed(args.seed * 1000 + args.worker_id)
    w, h, c = game.get_screen_width(), game.get_screen_height(), game.get_screen_channels()
    print(f"worker {args.worker_id}: screen {w}x{h}x{c}, actions {env.action_space.n}", flush=True)
    assert (w, h, c) == (320, 240, 3), (w, h, c)

    first = args.start_id + args.worker_id * args.episodes
    log_path = os.path.join(args.out_dir, f"worker_{args.worker_id:02d}.jsonl")
    for k in range(args.episodes):
        ep = first + k
        out = os.path.join(args.out_dir, f"ep_{ep:05d}.parquet")
        if os.path.exists(out):
            continue
        table, stats = record_episode(env, agent, game, ep, args.compress_level)
        tmp = out + ".tmp"
        pq.write_table(table, tmp, compression=None, row_group_size=256)
        os.replace(tmp, out)
        with open(log_path, "a") as f:
            f.write(json.dumps(stats) + "\n")
        print(f"ep {ep}: {stats['tics']} tics in {stats['seconds']:.0f}s "
              f"({stats['tics'] / max(stats['seconds'], 1e-6):.0f} tics/s), "
              f"{stats['png_bytes_mean'] / 1024:.1f} KB/frame, frags {stats['frags']}", flush=True)
    env.close()
    print("DONE", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--repro-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--worker-id", type=int, default=0)
    p.add_argument("--num-workers", type=int, default=1)
    p.add_argument("--episodes", type=int, default=1, help="episodes per worker")
    p.add_argument("--start-id", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--compress-level", type=int, default=6)
    main(p.parse_args())
