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

The episode metadata (parquet schema key `doomdit_episode`) records the seeds, the WAD, the map id
as stored and as the engine saw it, and any console commands sent at spawn. Readers must tolerate
keys being absent: corpora recorded before a key existed simply do not carry it.

`--decision-only` stores one row per agent decision instead of one per tic: the engine advances
with the agent's native frame skip in a single `make_action(buttons, frame_skip)` call, so the
three tics in between are never rendered, never converted and never PNG-encoded. Only training
frames survive either way (`encode_parquet.py` keeps decision tics), so the rows are the same rows;
the file records `stored_tic_stride` in its episode metadata and `action`/`buttons` then describe
the next `stored_tic_stride` tics rather than the next one. An absent key means a per-tic file.
An anti-stuck override runs 40 tics rather than the skip, so the `tic` gap after that row is 40;
the alignment rejects override rows in either mode, so no training frame is ever mislabelled by it.
The engine is not stepped identically in the two modes, so the same seed does not replay the same
episode: a death inside a skip is seen up to `frame_skip - 1` tics later here, and the episode
diverges from there. Same agent, same seeds, same distribution; a different draw from it.

Usage (run with Arnold's usual flags after the script's own):
    python record_arnold.py --arnold-dir /sata2/data/rnagabhi/doom/Arnold \
        --out-dir /sata2/data/rnagabhi/doom/raw_arnold --map-ids 1-17 --episodes 850 \
        --episode-time 150 --worker-id 0 --num-workers 20 -- \
        --frame_skip 4 --action_combinations "move_fb+move_lr;turn_lr;attack" ... --evaluate 1
"""
import argparse
import functools
import hashlib
import io
import json
import os
import random
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
SEED_SCHEME = "doomdit-episode-v1"
MAP_ID_LIMIT = 127          # the stored map_id column is int8


def stored_map_id(map_id, offset):
    """The `map_id` written to the corpus, which is not always the map the engine was told to load.

    Arnold addresses every scenario's levels as MAP01 upward, so `deathmatch_simple`'s only map and
    `full_deathmatch`'s arena 1 are both map 1 to the engine. Recording a second WAD under an offset
    keeps the labels apart once the directories are merged at encode time; the WAD name goes into the
    episode metadata alongside it, so the label is decodable rather than merely unique.
    """
    out = int(map_id) + int(offset)
    if not 0 < out <= MAP_ID_LIMIT:
        raise ValueError(f"map_id {map_id} + offset {offset} = {out}, outside 1..{MAP_ID_LIMIT} (int8 column)")
    return out


def episode_seeds(corpus_id, episode_id):
    """Independent 32-bit seeds per RNG stream, a pure function of (corpus, episode); no worker, pid, or time."""
    def derive(stream):
        payload = json.dumps([SEED_SCHEME, corpus_id, int(episode_id), stream], separators=(",", ":")).encode()
        return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little")
    return {stream: derive(stream) for stream in ("python", "numpy", "torch", "vizdoom")}


def start_seeded_episode(game, map_id, episode_id):
    """Seed every RNG stream, then start the episode with the ViZDoom instance that Arnold's Game.start creates
    seeded before init. Returns the seeds for the provenance record."""
    import torch
    import src.doom.game as arnold_game
    seeds = episode_seeds(REC.corpus_id, episode_id)
    random.seed(seeds["python"]); np.random.seed(seeds["numpy"]); torch.manual_seed(seeds["torch"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seeds["torch"])
    # the anti-stuck counters otherwise carry over from the previous episode of the same worker
    game.count_non_forward_actions = 0
    game.count_non_turn_actions = 0
    factory = arnold_game.DoomGame

    def seeded_factory(*a, **k):
        engine = factory(*a, **k)
        engine.set_seed(seeds["vizdoom"])
        return engine
    arnold_game.DoomGame = seeded_factory   # one episode at a time per worker process, so this is safe
    try:
        game.start(map_id=map_id, episode_time=REC.episode_time, log_events=False, manual_control=True)
    finally:
        arnold_game.DoomGame = factory
    send_init_commands(game)
    return seeds


def send_init_commands(game):
    """Console commands the scenario needs to be playable, sent after every spawn and respawn.

    `deathmatch_simple` has no bots and keeps its monsters behind an ACS difficulty script, so
    without `pukename change_difficulty 5` the agent records an empty map. The GameNGen
    reproductions send exactly this, on reset, which is where Arnold's `initialize_game` leaves us.
    """
    for command in REC.init_game_commands:
        game.game.send_game_command(command)


def episode_provenance(episode_id, map_id, label, wad, seeds):
    """The `doomdit_episode` record written into each episode's parquet metadata.

    Only the first five keys are unconditional. Every other key appears exactly when the option that
    needs it was used, so a default per-tic run writes byte-for-byte what the recorder wrote before
    these options existed, and a segment already being recorded can be resumed by a newer build
    without the files disagreeing. A reader defaults a missing key: offset 0, no commands, stride 1.
    """
    prov = {"seed_scheme": SEED_SCHEME, "corpus_id": REC.corpus_id, "episode_id": int(episode_id),
            "map_id": int(label), "seeds": seeds}
    if REC.map_id_offset:
        # the stored label is no longer the map the engine loaded, so both, and the WAD, become facts
        prov.update(engine_map_id=int(map_id), map_id_offset=int(REC.map_id_offset), wad=wad)
    if REC.init_game_commands:
        prov["init_game_commands"] = list(REC.init_game_commands)
    if REC.zdoom_bots:
        prov["bots"] = "zdoom addbot"
    if REC.decision_only:
        prov["stored_tic_stride"] = int(REC.frame_skip)
    return prov


def record_episode(game, network, params, map_id, episode_id):
    from vizdoom import GameVariable
    import pyarrow as pa

    seeds = start_seeded_episode(game, map_id, episode_id)
    game.randomize_textures(False)
    game.init_bots_health(100)
    network.reset()
    network.module.eval()
    dg = game.game
    gv = dg.get_game_variable
    cols = {k: [] for k in ["tic", "action", "buttons", "health", "ammo", "kills", "deaths", "frags",
                            "pos_x", "pos_y", "angle", "frame"]}

    def store(st, tic, action_id, bstr):
        """One row: the screen at `tic`, and the game variables read at the same tic."""
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

    last_states, tic, t0 = [], 0, time.time()
    decisions = 0
    while not game.is_episode_finished():
        if game.is_player_dead():
            # a death on the tic the episode clock expires leaves no state to respawn into
            if game.is_episode_finished() or dg.get_state() is None:
                break
            try:
                game.respawn_player()
            except AttributeError:
                break
            send_init_commands(game)
            network.reset()
            if game.is_player_dead() or game.is_episode_finished():
                continue
        game.observe_state(params, last_states)
        action_id = network.next_action(last_states)
        buttons, n_tics, logical = decision_buttons(game, action_id)
        bstr = "".join("1" if b else "0" for b in buttons)
        decisions += 1
        if REC.decision_only:
            # the whole skip in one engine call: the tics in between are never rendered or stored.
            # A death inside the skip is seen after it, and the respawn row that follows carries the
            # incremented `deaths`, so the life ends here exactly as it does per tic and this row is
            # rejected as a transition source rather than pointing across the death.
            st = dg.get_state()
            if st is not None:
                store(st, tic, action_id, bstr)
            dg.make_action(buttons, n_tics)
            tic += n_tics
        else:
            for _ in range(n_tics):
                st = dg.get_state()
                if st is None:
                    break
                store(st, tic, action_id, bstr)
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
    wad = os.path.basename(game.scenario_path)
    game.close()
    n = len(cols["tic"])
    label = stored_map_id(map_id, REC.map_id_offset)
    table = pa.table({
        "episode_id": pa.array([episode_id] * n, pa.int32()),
        "map_id": pa.array([label] * n, pa.int8()),
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
    provenance = episode_provenance(episode_id, map_id, label, wad, seeds)
    table = table.replace_schema_metadata({**(table.schema.metadata or {}), b"doomdit_episode": json.dumps(provenance, sort_keys=True).encode()})
    stats.update(rows=n, tics=tic, decisions=decisions, seconds=time.time() - t0,
                 png_bytes_mean=float(np.mean([len(b) for b in cols["frame"]])) if n else 0.0)
    stats.update(provenance)   # carries episode_id and map_id
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
        try:
            table, stats = record_episode(game, network, params, map_id, e)
        except Exception as ex:  # one bad episode must not kill the worker
            print(f"ep {e} map {map_id}: FAILED {type(ex).__name__}: {ex}", flush=True)
            try:
                game.close()
            except Exception:
                pass
            continue
        if buttons is None:
            buttons = [str(b).split(".")[-1] for b in game.game.get_available_buttons()]
        tmp = out + ".tmp"
        pq.write_table(table, tmp, compression=None, row_group_size=256)
        os.replace(tmp, out)
        with open(log_path, "a") as f:
            f.write(json.dumps(stats) + "\n")
        secs = max(stats["seconds"], 1e-6)
        print(f"ep {e} map {map_id}: {stats['rows']} rows / {stats['tics']} tics in {stats['seconds']:.0f}s "
              f"({stats['tics'] / secs:.0f} tics/s, {stats['decisions'] / secs:.0f} decisions/s), "
              f"{stats['png_bytes_mean'] / 1024:.1f} KB/frame, "
              f"kills {stats['kills']} deaths {stats['deaths']}", flush=True)
    meta_path = os.path.join(REC.out_dir, "buttons.json")
    if buttons and not os.path.exists(meta_path):
        meta = {"available_buttons": buttons, "action_combinations": params.action_combinations,
                "n_actions": game.action_builder.n_actions, "frame_skip": params.frame_skip,
                "screen": "RES_320X240 RGB HUD on weapon on crosshair on", "agent": "Arnold vizdoom_2017_track2",
                "wad": os.path.basename(game.scenario_path)}
        if REC.map_id_offset:
            meta["map_id_offset"] = REC.map_id_offset
        if REC.init_game_commands:
            meta["init_game_commands"] = list(REC.init_game_commands)
        if REC.zdoom_bots:
            meta["bots"] = "zdoom addbot"
        if REC.decision_only:
            meta["stored_tic_stride"] = int(params.frame_skip)
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=1)
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
    p.add_argument("--decision-only", action="store_true",
                   help="store one row per agent decision and advance the engine with the agent's frame skip "
                        "(several times faster; the file records stored_tic_stride)")
    p.add_argument("--map-id-offset", type=int, default=0,
                   help="added to the stored map_id only; the engine still loads --map-ids. Use it when a "
                        "second WAD's MAP01 would collide with the first WAD's arena 1 (e.g. 100 -> map_id 101)")
    p.add_argument("--init-game-command", action="append", default=[], metavar="CMD",
                   help="console command sent after every spawn and respawn, repeatable; an escape hatch for "
                        "scenarios that need ACS setup")
    p.add_argument("--zdoom-bots", action="store_true",
                   help="fill the game with ZDoom's own addbot instead of Arnold's scripted marines. Arnold's "
                        "deathmatch scenario hardcodes scripted marines, whose ACS script lives in "
                        "full_deathmatch.wad, so on any other WAD no opponent ever spawns")
    p.add_argument("--corpus-id", required=True, help="immutable corpus name; with the episode id it determines every seed")
    mine, arnold_args = p.parse_known_args()
    if arnold_args and arnold_args[0] == "--":
        arnold_args = arnold_args[1:]
    REC.out_dir, REC.map_ids, REC.episodes = mine.out_dir, parse_map_ids(mine.map_ids), mine.episodes
    REC.episode_time, REC.worker_id, REC.num_workers = mine.episode_time, mine.worker_id, mine.num_workers
    REC.compress_level = mine.compress_level
    REC.decision_only = mine.decision_only
    REC.map_id_offset = mine.map_id_offset
    REC.init_game_commands = mine.init_game_command
    REC.zdoom_bots = mine.zdoom_bots
    REC.corpus_id = mine.corpus_id
    stored_map_id(max(REC.map_ids), REC.map_id_offset)      # fail now, not after an episode of recording

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
    game_kwargs = {"screen_resolution": "RES_320X240"}
    if mine.zdoom_bots:
        # `src/doom/scenarios/deathmatch.py` hardcodes use_scripted_marines=True, and that ACS script is part
        # of full_deathmatch.wad. On deathmatch_simple it silently adds nobody, so the agent records an empty
        # map; ZDoom's own addbot fills it (measured: 4 to 5 opponents visible).
        game_kwargs["use_scripted_marines"] = False
    deathmatch.Game = functools.partial(Game, **game_kwargs)
    from src.args import parse_game_args
    parse_game_args(arnold_args + ["--dump_path", dump_path, "--render_hud", "1"])


if __name__ == "__main__":
    main()
