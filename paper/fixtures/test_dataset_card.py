"""The dataset card says what the published files actually hold (docs/REVIEW_2026-09-22.md L1).

Five statements were checked against the files by the review and found wrong:

  * `index_arenas.parquet` / `index_arenas_678.parquet` have kills = deaths = -1 in every row; the
    real per-episode values are in `worker_*.jsonl`;
  * a default per-tic file's `doomdit_episode` metadata carries seed scheme, corpus id, episode id,
    map id and seeds, not WAD, kills or deaths;
  * `tic` is the recorder's counter, not the engine tic: it steps by exactly 1 across a respawn and
    the roughly 39 respawn tics per death are never recorded;
  * the unseen split is a 60-episode scoring subset, not all 3,000 episodes;
  * "bit-for-bit reproducible" from the corpus and episode id holds only given the recorder's start
    history, because a worker's first episode is the only one whose weapon selects execute.

    python -m pytest paper/fixtures/test_dataset_card.py -q
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))


def card():
    with open(os.path.join(REPO, "release", "DATASET_CARD.md")) as f:
        return f.read()


def dense():
    with open(os.path.join(REPO, "release", "DENSE_CORPUS.md")) as f:
        return f.read()


def test_the_index_files_kills_and_deaths_are_called_placeholders():
    text = card()
    i = text.index("- `index_arenas.parquet`")
    bullet = text[i:text.index("\n- ", i + 1)]
    assert "-1" in bullet and "worker_*.jsonl" in bullet, bullet


def test_the_episode_metadata_keys_are_the_ones_a_default_file_has():
    text = card()
    assert "map id, WAD, seeds, kills, deaths" not in text
    for key in ("seed_scheme", "corpus_id", "episode_id", "map_id", "seeds"):
        assert f"`{key}`" in text, key


def test_tic_is_the_recorders_counter():
    text = card()
    assert "engine tic inside the episode" not in text
    assert "recorder's own counter" in text and "39" in text


def test_the_unseen_split_row_names_the_scoring_subset():
    text = card()
    assert "| unseen maps | `arenas_678/` (all) | 3,000 |" not in text
    assert "60 to 119" in text


def test_reproducibility_is_stated_with_its_condition():
    for text in (card(), dense()):
        assert "start history" in text


def test_the_card_does_not_promise_unconditional_reproduction():
    text = card()
    i = text.index("bit-for-bit reproducible")
    assert "start history" in text[i:i + 600]
