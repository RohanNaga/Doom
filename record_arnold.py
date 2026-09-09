"""
Record lossless per-tic ViZDoom episodes with the Arnold agent (Lample and Chaplot, 2017),
the agent MultiGen (2026) used for its Doom data.

Hooks Arnold's own entry point: `src.args.parse_game_args` builds params, the game, and the
network exactly as `arnold.py --evaluate 1` does, then calls the scenario's
`evaluate_deathmatch`, which this script replaces with a recording loop. Arnold's decision
preamble (favourite-weapon switching, anti-stuck manual control) is replicated verbatim from
`Game.make_action`; the only change is that the engine advances one tic at a time so every
tic is captured.

Row semantics: `frame` is the screen at `tic`; `action` (Arnold action id) and `buttons`
(0/1 string over the game's available buttons) are what is applied from this tic to the next.
Game variables are read at the same tic. Frames are 320x240 RGB with HUD, weapon, crosshair.

Usage (run with Arnold's usual flags after the script's own):
    python record_arnold.py --arnold-dir /sata2/data/rnagabhi/doom/Arnold \
        --out-dir /sata2/data/rnagabhi/doom/raw_arnold --map-ids 1-17 --episodes 850 \
        --episode-time 150 --worker-id 0 --num-workers 20 -- \
        --frame_skip 4 --action_combinations "move_fb+move_lr;turn_lr;attack" ... --evaluate 1
"""
import argparse
import functools
import io
import json
import os
import sys
import time

import numpy as np


def png_bytes(frame_u8, level):
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(frame_u8).save(buf, format="PNG", compress_level=level)
    return buf.getvalue()


def parse_map_ids(s):
    out = []
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def decision_buttons(game, action_id):
    """Arnold's `Game.make_action` preamble: weapon preference and anti-stuck manual control.
    Returns (buttons, n_tics) to apply."""
    from src.doom.game import WEAPONS_PREFERENCES
    action = game.action_builder.get_action(action_id)
    for weapon_name, weapon_ammo, weapon_id in WEAPONS_PREFERENCES:
        min_ammo = 40 if weapon_name == "bfg9000" else 1
        if game.properties[weapon_name] > 0 and game.properties[weapon_ammo] >= min_ammo:
            if game.properties["sel_weapon"] != weapon_id:
                switch_action = ([False] * game.mapping["SELECT_WEAPON%i" % weapon_id]) + [True]
                action = action + switch_action[len(action):]
            break
    if action[game.mapping["MOVE_FORWARD"]]:
        game.count_non_forward_actions = 0
    else:
        game.count_non_forward_actions += 1
    if action[game.mapping["TURN_LEFT"]] or action[game.mapping["TURN_RIGHT"]]:
        game.count_non_turn_actions = 0
    else:
        game.count_non_turn_actions += 1
    if game.manual_control and (game.count_non_forward_actions >= 30 or game.count_non_turn_actions >= 60):
        manual = [False] * len(action)
        manual[game.mapping["TURN_RIGHT"]] = True
        manual[game.mapping["SPEED"]] = True
        if game.count_non_forward_actions >= 30:
            manual[game.mapping["MOVE_FORWARD"]] = True
        game.count_non_forward_actions = 0
        game.count_non_turn_actions = 0
        return manual, 40, action
    return action, REC.frame_skip, action


class Rec:
    pass


REC = Rec()


def record_episode(game, network, params, map_id, episode_id):
    from vizdoom import GameVariable
    import pyarrow as pa

    game.start(map_id=map_id, episode_time=REC.episode_time, log_events=False, manual_control=True)
    game.randomize_textures(False)
    game.init_bots_health(100)
    network.reset()
    network.module.eval()
    dg = game.game
    gv = dg.get_game_variable
    cols = {k: [] for k in ["tic", "action", "buttons", "health", "ammo", "kills", "deaths", "frags",
                            "pos_x", "pos_y", "angle", "frame"]}
    last_states, tic, t0 = [], 0, time.time()
    while not game.is_episode_finished():
        if game.is_player_dead():
            game.respawn_player()
            network.reset()
            if game.is_player_dead() or game.is_episode_finished():
                continue
        game.observe_state(params, last_states)
        action_id = network.next_action(last_states)
        buttons, n_tics, logical = decision_buttons(game, action_id)
        bstr = "".join("1" if b else "0" for b in buttons)
        for _ in range(n_tics):
            st = dg.get_state()
            if st is None:
                break
            frame = np.asarray(st.screen_buffer)
            if frame.ndim == 3 and frame.shape[0] == 3:      # CRCGCB -> HWC
                frame = np.ascontiguousarray(frame.transpose(1, 2, 0))
            cols["tic"].append(tic); cols["action"].append(int(action_id)); cols["buttons"].append(bstr)
            cols["health"].append(int(gv(GameVariable.HEALTH)))
            cols["ammo"].append(int(gv(GameVariable.SELECTED_WEAPON_AMMO)))
            cols["kills"].append(int(gv(GameVariable.KILLCOUNT)))
            cols["deaths"].append(int(gv(GameVariable.DEATHCOUNT)))
            cols["frags"].append(int(gv(GameVariable.FRAGCOUNT)))
            cols["pos_x"].append(float(gv(GameVariable.POSITION_X)))
            cols["pos_y"].append(float(gv(GameVariable.POSITION_Y)))
            cols["angle"].append(float(gv(GameVariable.ANGLE)))
            cols["frame"].append(png_bytes(frame, REC.compress_level))
            dg.make_action(buttons, 1)
            tic += 1
            if game.is_player_dead() or game.is_episode_finished():
                break
        gs = dg.get_state()
        if gs is not None:
            game._screen_buffer = gs.screen_buffer
            game._depth_buffer = gs.depth_buffer
            game._labels_buffer = gs.labels_buffer
            game._labels = gs.labels
        game.update_game_variables()
        game.update_statistics_and_reward(logical)
    stats = {k: game.statistics[map_id].get(k, 0) for k in ["kills", "deaths", "suicides", "frags"]}
    game.close()
    n = len(cols["tic"])
    table = pa.table({
        "episode_id": pa.array([episode_id] * n, pa.int32()),
        "map_id": pa.array([map_id] * n, pa.int8()),
        "tic": pa.array(cols["tic"], pa.int32()),
        "action": pa.array(cols["action"], pa.int16()),
        "buttons": pa.array(cols["buttons"], pa.string()),
        "health": pa.array(cols["health"], pa.int16()),
        "ammo": pa.array(cols["ammo"], pa.int16()),
        "kills": pa.array(cols["kills"], pa.int16()),
        "deaths": pa.array(cols["deaths"], pa.int16()),
        "frags": pa.array(cols["frags"], pa.int16()),
        "pos_x": pa.array(cols["pos_x"], pa.float32()),
        "pos_y": pa.array(cols["pos_y"], pa.float32()),
        "angle": pa.array(cols["angle"], pa.float32()),
        "frame": pa.array(cols["frame"], pa.binary()),
    })
    stats.update(episode_id=episode_id, map_id=map_id, tics=n, seconds=time.time() - t0,
                 png_bytes_mean=float(np.mean([len(b) for b in cols["frame"]])) if n else 0.0)
    return table, stats


def record_all(game, network, params):
    """Replaces deathmatch.evaluate_deathmatch. Called by Arnold's main with built objects."""
    import pyarrow.parquet as pq
    REC.frame_skip = params.frame_skip
    os.makedirs(REC.out_dir, exist_ok=True)
    buttons = [str(b).split(".")[-1] for b in game.game.get_available_buttons()] if hasattr(game, "game") and game.game else None
    log_path = os.path.join(REC.out_dir, f"worker_{REC.worker_id:02d}.jsonl")
    maps = REC.map_ids
    for e in range(REC.worker_id, REC.episodes, REC.num_workers):
        map_id = maps[e % len(maps)]
        out = os.path.join(REC.out_dir, f"ep_{e:05d}.parquet")
        if os.path.exists(out):
            continue
        table, stats = record_episode(game, network, params, map_id, e)
        if buttons is None:
            buttons = [str(b).split(".")[-1] for b in game.game.get_available_buttons()]
        tmp = out + ".tmp"
        pq.write_table(table, tmp, compression=None, row_group_size=256)
        os.replace(tmp, out)
        with open(log_path, "a") as f:
            f.write(json.dumps(stats) + "\n")
        print(f"ep {e} map {map_id}: {stats['tics']} tics in {stats['seconds']:.0f}s "
              f"({stats['tics'] / max(stats['seconds'], 1e-6):.0f} tics/s), {stats['png_bytes_mean'] / 1024:.1f} KB/frame, "
              f"kills {stats['kills']} deaths {stats['deaths']}", flush=True)
    meta_path = os.path.join(REC.out_dir, "buttons.json")
    if buttons and not os.path.exists(meta_path):
        with open(meta_path, "w") as f:
            json.dump({"available_buttons": buttons, "action_combinations": params.action_combinations,
                       "n_actions": game.action_builder.n_actions, "frame_skip": params.frame_skip,
                       "screen": "RES_320X240 RGB HUD on weapon on crosshair on", "agent": "Arnold vizdoom_2017_track2"}, f, indent=1)
    print("DONE", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arnold-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--map-ids", default="1-17")
    p.add_argument("--episodes", type=int, default=1, help="total episodes across all workers")
    p.add_argument("--episode-time", type=int, default=150, help="game seconds per episode")
    p.add_argument("--worker-id", type=int, default=0)
    p.add_argument("--num-workers", type=int, default=1)
    p.add_argument("--compress-level", type=int, default=6)
    mine, arnold_args = p.parse_known_args()
    if arnold_args and arnold_args[0] == "--":
        arnold_args = arnold_args[1:]
    REC.out_dir, REC.map_ids, REC.episodes = mine.out_dir, parse_map_ids(mine.map_ids), mine.episodes
    REC.episode_time, REC.worker_id, REC.num_workers = mine.episode_time, mine.worker_id, mine.num_workers
    REC.compress_level = mine.compress_level

    os.chdir(mine.arnold_dir)
    sys.path.insert(0, mine.arnold_dir)
    from src.logger import get_logger
    from src.utils import get_dump_path
    dump_path = get_dump_path(os.path.join(mine.out_dir, "arnold_dump"), f"worker_{mine.worker_id:02d}")
    os.makedirs(dump_path, exist_ok=True)
    get_logger(filepath=os.path.join(dump_path, "train.log"))
    from src.doom.scenarios import deathmatch
    from src.doom.game import Game
    deathmatch.evaluate_deathmatch = record_all
    deathmatch.Game = functools.partial(Game, screen_resolution="RES_320X240")
    from src.args import parse_game_args
    parse_game_args(arnold_args + ["--dump_path", dump_path, "--render_hud", "1"])


if __name__ == "__main__":
    main()
