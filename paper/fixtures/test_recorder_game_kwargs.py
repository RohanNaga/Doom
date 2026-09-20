"""The recorder's game settings must beat Arnold's own, not merely default underneath them.

`record_arnold.py` does not construct Arnold's `Game`; Arnold's `deathmatch.py` does, and it passes
its choices as explicit keywords. The recorder injects its own by replacing the name `Game` in that
module. A `functools.partial` is the wrong tool for that, because partial keywords are defaults the
call site overrides: `partial(G, use_scripted_marines=False)(use_scripted_marines=True)` builds a
`Game` with scripted marines. That silently cancelled `--zdoom-bots`, which is why
`deathmatch_simple` recorded with no opponent (release/DENSE_CORPUS.md).

Nothing here needs ViZDoom, Arnold or torch: the call site is reproduced with a stand-in class.

    python -m pytest paper/fixtures/test_recorder_game_kwargs.py -q
"""
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)

from record_arnold import forced_game  # noqa: E402


class FakeGame:
    """Stand-in for Arnold's `Game`, which records only what it was constructed with."""

    def __init__(self, scenario, **kwargs):
        self.scenario = scenario
        self.kwargs = kwargs


def arnold_call_site(GameCls):
    """`src/doom/scenarios/deathmatch.py:86-107`: every choice passed as an explicit keyword."""
    return GameCls(scenario="deathmatch_simple", n_bots=8, use_scripted_marines=True, render_hud=1)


def test_forced_keyword_overrides_the_call_site():
    game = arnold_call_site(forced_game(FakeGame, use_scripted_marines=False))
    assert game.kwargs["use_scripted_marines"] is False


def test_partial_does_not_override_and_is_why_the_bug_existed():
    import functools
    game = arnold_call_site(functools.partial(FakeGame, use_scripted_marines=False))
    assert game.kwargs["use_scripted_marines"] is True


def test_call_site_keywords_the_recorder_does_not_force_are_left_alone():
    game = arnold_call_site(forced_game(FakeGame, use_scripted_marines=False))
    assert game.kwargs["n_bots"] == 8 and game.kwargs["render_hud"] == 1
    assert game.scenario == "deathmatch_simple"


def test_forcing_a_keyword_the_call_site_omits_still_applies():
    # screen_resolution is commented out at Arnold's call site, which is the only reason the
    # recorder's 320x240 ever took effect under the partial
    game = arnold_call_site(forced_game(FakeGame, screen_resolution="RES_320X240"))
    assert game.kwargs["screen_resolution"] == "RES_320X240"


def test_positional_arguments_still_reach_the_class():
    game = forced_game(FakeGame, use_scripted_marines=False)("full_deathmatch", n_bots=8)
    assert game.scenario == "full_deathmatch" and game.kwargs["use_scripted_marines"] is False


def test_no_forced_keywords_is_the_plain_class():
    game = arnold_call_site(forced_game(FakeGame))
    assert game.kwargs["use_scripted_marines"] is True and game.kwargs["n_bots"] == 8


@pytest.mark.parametrize("zdoom_bots", [False, True])
def test_scripted_marines_are_forced_off_exactly_under_zdoom_bots(zdoom_bots):
    """The flag's whole contract: off means Arnold's ACS marines, on means ZDoom's own addbot."""
    forced = {"screen_resolution": "RES_320X240"}
    if zdoom_bots:
        forced["use_scripted_marines"] = False
    game = arnold_call_site(forced_game(FakeGame, **forced))
    assert game.kwargs["use_scripted_marines"] is (not zdoom_bots)
