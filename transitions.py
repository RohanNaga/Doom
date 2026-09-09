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
"""
import numpy as np

CONTROL_BITS = 9   # MOVE_FORWARD, MOVE_BACKWARD, TURN_LEFT, TURN_RIGHT, MOVE_LEFT, MOVE_RIGHT, ATTACK, SPEED, CROUCH; weapon-select bits ignored


def life_segments(deaths):
    """[(start, end)) row ranges of continuous lives; a new life starts where `deaths` increments."""
    d = np.asarray(deaths)
    cuts = np.flatnonzero(np.diff(d) != 0) + 1
    bounds = np.concatenate([[0], cuts, [len(d)]])
    return [(int(a), int(b)) for a, b in zip(bounds[:-1], bounds[1:]) if b > a]


def canonical_table(action, buttons, num_actions=None):
    """Modal control bits per requested action id, over the whole recording (overrides are a minority)."""
    from collections import Counter, defaultdict
    c = defaultdict(Counter)
    for a, b in zip(np.asarray(action).tolist(), np.asarray(buttons).tolist()):
        c[int(a)][str(b)[:CONTROL_BITS]] += 1
    return {a: cnt.most_common(1)[0][0] for a, cnt in c.items()}


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
