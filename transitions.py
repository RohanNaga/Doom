"""
Verified fixed-duration transitions from a per-tic recording (pure numpy, no torch).

The agent decides every `repeat` tics and the engine holds the buttons in between, but a death
cuts a decision short and the next one starts at the respawn row, so the decision phase shifts
within an episode. Rather than reconstruct decisions, we accept only transitions that are
provably one control interval: source row s and target row s + repeat inside one continuous
life, with the executed button vector constant over rows s .. s+repeat-1 and, when a canonical
table is given, equal to the canonical vector of the requested action (so anti-stuck overrides
are excluded rather than mislabeled). Accepted transitions are chained: consecutive accepted
sources form a chain, and a training window or rollout never crosses a chain boundary.

Row semantics (record_arnold.py): `action`/`buttons` on row t are applied from t to t+1;
`deaths` increments on the respawn row.

A `record_arnold.py --decision-only` recording stores only the tics the agent decides on, so its
rows are already `frame_skip` tics apart and `repeat` becomes `stride // stored_tic_stride`. Every
row-level rule above is unchanged: a death still cuts the life at the respawn row, so a decision
whose skip was cut short is still the last row of its life and is still rejected as a source.
"""
import json

import numpy as np

CONTROL_BITS = 9   # MOVE_FORWARD, MOVE_BACKWARD, TURN_LEFT, TURN_RIGHT, MOVE_LEFT, MOVE_RIGHT, ATTACK, SPEED, CROUCH; weapon-select bits ignored
EPISODE_META_KEY = b"doomdit_episode"

# The engine's whole button list: the nine above plus SELECT_WEAPON0..9. ViZDoom's `setAction` loops
# over `availableButtons.size()` and reads `actions[i]` only for `i < 19`, zero-filling what is
# missing and never looking at what is beyond (ViZDoom 1.2.4 `src/lib/ViZDoomGame.cpp:151-158`), so
# the EXECUTED control of a recorded row is exactly its first 19 characters, right-padded with 0.
EXECUTED_BUTTONS = 19

# `SELECT_WEAPONj` sits at index CONTROL_BITS + 10*k + j after k `Game.start()` calls in one
# recorder process (Arnold `src/doom/actions.py:197-199` re-appends the ten names to the SHARED
# `available_buttons` list on every start, while ViZDoom deduplicates its own list). The widest
# string in an episode therefore pins k, which is why the report can infer it.
WEAPON_SLOTS = 10


def inferred_starts(width):
    """How many `Game.start()` calls a raw string of this width implies, or None for a plain row.

    A press of `SELECT_WEAPONj` after k starts sits at index `CONTROL_BITS + WEAPON_SLOTS*k + j`
    with `0 <= j < WEAPON_SLOTS`, so the string is that long plus one and integer division recovers
    k whichever weapon was asked for.
    """
    return (width - 1 - CONTROL_BITS) // WEAPON_SLOTS if width > CONTROL_BITS else None


def normalize_buttons(s):
    """One recorded button string as the EXECUTED control: 19 characters, binary, 0-padded.

    `record_arnold.py:255` renders Arnold's requested control list and line 273 hands the SAME list
    to `make_action`, so nothing is lost by truncating at what the engine reads. A 1 beyond index 18
    is a weapon-select press the engine never executed, not a control input: it is dropped, and
    `switch_request_indices` keeps its index so the request stays recoverable from the sidecar.

    Only the first `EXECUTED_BUTTONS` characters are validated, because only they become control
    values; a stray character further out cannot reach the model.
    """
    s = str(s)
    if not s:
        raise ValueError("empty button string: this row holds no executed control")
    head = s[:EXECUTED_BUTTONS]
    bad = sorted(set(head) - {"0", "1"})
    if bad:
        raise ValueError(f"button string {s[:24]!r} is not binary: {bad[:4]} appear within its first "
                         f"{EXECUTED_BUTTONS} characters, so it is not the recorder's per-button 0/1 flags")
    return head.ljust(EXECUTED_BUTTONS, "0")


def normalize_button_column(buttons):
    """A whole recorded column as a fixed `<U19` array of executed control vectors.

    Fixed width is the point: `np.array(raw)` over a corpus whose widest row is 2,506 characters
    makes a `<U2506` column, which is a 38 to 49 MB sidecar per episode instead of about 200 KB.
    """
    return np.array([normalize_buttons(s) for s in np.asarray(buttons).tolist()],
                    dtype=f"<U{EXECUTED_BUTTONS}")


def raw_button_lengths(buttons):
    """Per-row length of the RAW recorded string, so the sidecar records what was requested."""
    return np.array([len(str(s)) for s in np.asarray(buttons).tolist()], dtype=np.int16)


def switch_request_indices(buttons):
    """Per-row index of the highest 1 beyond the nine control bits, or -1 when there is none.

    Below `EXECUTED_BUTTONS` this is a weapon switch the engine performed; at or above it, one it
    was asked for and never performed. Keeping the index means the unexecuted requests are
    recoverable from the sidecar alone, without the raw parquet.
    """
    out = []
    for s in np.asarray(buttons).tolist():
        s = str(s)
        out.append(s.rfind("1", CONTROL_BITS))
    return np.array(out, dtype=np.int32)


def button_width_report(buttons):
    """What the raw widths of one episode's column say, without refusing any of them.

    `rows_over_executed` is the share of rows whose request ran past the engine's button list;
    `inferred_starts` is the k those widths imply.

    `within_episode_growth` is True when one episode's rows imply two different k. The width itself
    is NOT constant inside an episode -- it moves with the weapon id j, which changes from row to
    row -- so the invariant is the start count, constant up to j. A True here would contradict "the
    list grows once per `Game.start()`" and means the explanation is wrong.
    """
    raw = [str(s) for s in np.asarray(buttons).tolist()]
    lens = [len(s) for s in raw]
    idx = switch_request_indices(raw).tolist()
    over = [i for i in idx if i >= EXECUTED_BUTTONS]
    tails = sorted({n for n in lens if n > CONTROL_BITS})
    starts = sorted({inferred_starts(n) for n in tails})
    widest = max(lens) if lens else 0
    return {"rows": len(raw), "width": EXECUTED_BUTTONS,
            "raw_max_width": widest, "raw_min_width": min(lens) if lens else 0,
            "rows_over_executed": len(over),
            "fraction_over_executed": len(over) / len(raw) if raw else 0.0,
            "executed_switch_rows": sum(1 for i in idx if 0 <= i < EXECUTED_BUTTONS),
            "unexecuted_switch_rows": len(over),
            "raw_widths_over_control_bits": tails,
            "inferred_starts_seen": starts,
            "within_episode_growth": len(starts) > 1,
            "inferred_starts": inferred_starts(widest)}


def stored_tic_stride(schema_metadata):
    """Tics between consecutive stored rows, read from the parquet schema metadata.

    1 for a per-tic recording, which is what an absent key means: `record_arnold.py` writes the key
    only when `--decision-only` stored one row per agent decision, and every corpus recorded before
    that flag existed is per-tic.
    """
    blob = (schema_metadata or {}).get(EPISODE_META_KEY)
    return int(json.loads(blob).get("stored_tic_stride", 1)) if blob else 1


def decision_rows(action, buttons, deaths, stride=4, canonical=None, stored=1):
    """Rows to encode and their chain ids, for a recording whose rows are `stored` tics apart.

    The single place the stride arithmetic lives: a control interval of `stride` tics spans
    `stride // stored` rows, and everything else is `valid_transitions` unchanged.
    """
    if stride % stored:
        raise ValueError(f"stride {stride} is not a multiple of the stored tic stride {stored}")
    repeat = stride // stored
    src, ch = valid_transitions(action, buttons, deaths, repeat, canonical)
    return chain_frames(src, ch, repeat)


def life_segments(deaths):
    """[(start, end)) row ranges of continuous lives; a new life starts where `deaths` increments."""
    d = np.asarray(deaths)
    cuts = np.flatnonzero(np.diff(d) != 0) + 1
    bounds = np.concatenate([[0], cuts, [len(d)]])
    return [(int(a), int(b)) for a, b in zip(bounds[:-1], bounds[1:]) if b > a]


def control_counters():
    """The empty accumulator `count_controls` folds episodes into: action id -> Counter of control bits."""
    from collections import Counter, defaultdict
    return defaultdict(Counter)


def count_controls(counters, action, buttons):
    """Fold one episode's rows into `counters` in place, and return it.

    This is the streaming half of `canonical_table`. The counter keys are the first
    `CONTROL_BITS` characters of the executed button string, so at most a few dozen keys per
    action id survive however many rows are folded in; the caller can therefore drop each
    episode's `action`/`buttons` arrays as soon as it has been counted instead of concatenating
    the whole corpus first. A whole-corpus concatenation of 40M `<U19` strings costs about 6 GB
    of arrays plus another 2.7 GB of Python strings, which is what made six concurrent encoder
    shards a host-memory risk.
    """
    for a, b in zip(np.asarray(action).tolist(), list(buttons)):
        counters[int(a)][str(b)[:CONTROL_BITS]] += 1
    return counters


def canonical_from_counters(counters):
    """Modal control bits per action id, from an accumulator built by `count_controls`."""
    return {a: cnt.most_common(1)[0][0] for a, cnt in counters.items()}


def canonical_table(action, buttons, num_actions=None):
    """Modal control bits per requested action id, over the whole recording (overrides are a minority).

    The one-shot form, for a caller that already holds both columns of a single recording.
    `count_controls` / `canonical_from_counters` are the streaming form a multi-episode corpus
    should use.
    """
    return canonical_from_counters(count_controls(control_counters(), action, np.asarray(buttons).tolist()))


def valid_transitions(action, buttons, deaths, repeat=4, canonical=None):
    """Return (sources, chain_ids): accepted source rows and the chain id of each.

    Within a life, the phase is anchored at the life start and re-anchored at any control change
    that falls off the grid (an interrupted decision). A transition is accepted when rows
    s .. s+repeat-1 share one (action, control bits) pair that matches the canonical vector for
    the action, and s+repeat is still inside the life. Consecutive accepted sources (spacing exactly
    `repeat`) share a chain id.
    """
    action = np.asarray(action); buttons = np.asarray(buttons).astype(str); n = len(action)
    ctrl = np.array([b[:CONTROL_BITS] for b in buttons])
    sources, chains = [], []
    chain = -1
    for a0, a1 in life_segments(deaths):
        s = a0
        prev_src = None
        while s + repeat < a1:
            seg_a = action[s:s + repeat]; seg_c = ctrl[s:s + repeat]
            change = np.flatnonzero((seg_a[1:] != seg_a[:-1]) | (seg_c[1:] != seg_c[:-1]))
            if len(change):
                s = s + int(change[0]) + 1        # re-anchor the phase at the observed change point
                prev_src = None
                continue
            ok = True
            if canonical is not None:
                ok = canonical.get(int(seg_a[0])) == seg_c[0]
            if ok:
                if prev_src is None or s - prev_src != repeat:
                    chain += 1
                sources.append(s); chains.append(chain); prev_src = s
            else:
                prev_src = None
            s += repeat
    return np.array(sources, dtype=np.int64), np.array(chains, dtype=np.int64)


def chain_frames(sources, chains, repeat=4):
    """Frames to encode: every source plus the final target of each chain; returns (rows, chain_id_per_row)."""
    rows, cid = [], []
    for k, (s, c) in enumerate(zip(sources.tolist(), chains.tolist())):
        rows.append(s); cid.append(c)
        last_of_chain = (k == len(sources) - 1) or (chains[k + 1] != c)
        if last_of_chain:
            rows.append(s + repeat); cid.append(c)
    return np.array(rows, dtype=np.int64), np.array(cid, dtype=np.int64)
