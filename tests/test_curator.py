"""Unit tests for the dataset curation pipeline (src/load/curator.py).

No model is involved. Covers quality filters, stratification, candidate freezing,
hand-vetting decision application, and balanced selection of the clean-gold set.
"""

import json
from types import SimpleNamespace

from src.load.curator import (
    DatasetCurator, CurationReport, DEFAULTS,
    content_word_count, _normalized_key, _token_set, _jaccard,
)
from src.load.dataset_loader import Instance


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def make_instance(idx, text, label, genre="", dataset="t", split="s"):
    return Instance(
        instance_id=f"{dataset}_{split}_{idx:06d}",
        text=text, label=label, dataset=dataset, split=split,
        metadata={"genre": genre, "content_words": content_word_count(text), "char_len": len(text)},
    )


def sst2_cfg(**over):
    base = dict(
        name="sst2", split="validation", huggingface_id="x",
        text_field="sentence", label_field="label", secondary_text_field=None,
        labels=["negative", "positive"],
    )
    base.update(over)
    return SimpleNamespace(**base)


def cfg_dict(**over):
    c = dict(DEFAULTS)
    c.update(over)
    return c


# --------------------------------------------------------------------------- #
# pure helpers
# --------------------------------------------------------------------------- #
def test_content_word_count_ignores_pure_punctuation():
    assert content_word_count("a great movie !") == 3
    assert content_word_count("...") == 0


def test_normalized_key_and_jaccard():
    assert _normalized_key("The Movie!") == _normalized_key("the movie")
    a, b = _token_set("the quick brown fox"), _token_set("the quick brown cat")
    assert 0.5 < _jaccard(a, b) < 1.0
    assert _jaccard(_token_set("x y"), _token_set("x y")) == 1.0


# --------------------------------------------------------------------------- #
# Stage 1: build_instances
# --------------------------------------------------------------------------- #
def test_build_instances_maps_labels_and_drops_unmapped():
    curator = DatasetCurator(seed=1)
    rows = [
        {"sentence": "good film", "label": 1},
        {"sentence": "bad film", "label": 0},
        {"sentence": "no gold", "label": -1},   # must be dropped
    ]
    out = curator.build_instances(rows, sst2_cfg())
    assert [i.label for i in out] == ["positive", "negative"]
    assert out[0].instance_id == "sst2_validation_000000"


def test_build_instances_mnli_floor_uses_hypothesis():
    curator = DatasetCurator(seed=1)
    cfg = sst2_cfg(name="mnli", text_field="premise", secondary_text_field="hypothesis",
                   labels=["entailment", "neutral", "contradiction"])
    rows = [{"premise": "A long premise with many words here.", "hypothesis": "Short.",
             "label": 0, "genre": "fiction"}]
    out = curator.build_instances(rows, cfg)
    assert out[0].text.startswith("Premise:")
    assert out[0].metadata["genre"] == "fiction"
    assert out[0].metadata["content_words"] == 1  # counted on hypothesis "Short."


# --------------------------------------------------------------------------- #
# Stage 2: quality_filter
# --------------------------------------------------------------------------- #
def test_quality_filter_gates():
    curator = DatasetCurator(seed=1)
    cfg = cfg_dict(min_content_words=3, max_chars=40)
    insts = [
        make_instance(0, "this is a perfectly fine sentence", "positive"),  # keep
        make_instance(1, "too short", "positive"),                          # too_short (2 words)
        make_instance(2, "x " * 40, "positive"),                            # too_long (>40 chars)
        make_instance(3, "click <b>here</b> now please", "positive"),       # junk (markup)
        make_instance(4, "this is a perfectly fine sentence", "positive"),  # exact dup of #0
        make_instance(5, "", "positive"),                                   # empty
    ]
    kept, drops = curator.quality_filter(insts, cfg)
    assert len(kept) == 1
    assert drops["too_short"] == 1
    assert drops["too_long"] == 1
    assert drops["junk"] == 1
    assert drops["exact_dup"] == 1
    assert drops["empty"] == 1


# --------------------------------------------------------------------------- #
# Stage 3: stratification
# --------------------------------------------------------------------------- #
def test_stratified_pool_balances_labels():
    curator = DatasetCurator(seed=42)
    insts = []
    for i in range(100):
        label = "positive" if i % 2 == 0 else "negative"
        # distinct vocabulary per instance; vary length so buckets populate
        text = " ".join(f"w{i}t{j}" for j in range(3 + (i % 9)))
        insts.append(make_instance(i, text, label))
    curator.assign_length_buckets(insts)
    cfg = cfg_dict(target=20, oversample_factor=1.5, stratify_by=["label", "length"],
                   near_dup_jaccard=0.95)
    pool, n_near = curator.stratified_pool(insts, cfg)
    assert len(pool) == 30  # 20 * 1.5
    labels = [i.label for i in pool]
    assert abs(labels.count("positive") - labels.count("negative")) <= 4


# --------------------------------------------------------------------------- #
# Stage: freeze candidates round-trip
# --------------------------------------------------------------------------- #
def test_freeze_candidates_roundtrip(tmp_path):
    curator = DatasetCurator(seed=1)
    insts = [make_instance(i, f"sentence number {i} here", "positive", genre="fiction")
             for i in range(5)]
    for inst in insts:
        inst.metadata["length_bucket"] = "medium"
    path = curator.freeze_candidates(insts, tmp_path / "c.jsonl")
    rows = [json.loads(l) for l in path.read_text().splitlines()]
    assert len(rows) == 5
    assert {"instance_id", "text", "label", "genre", "length_bucket", "content_words"} <= rows[0].keys()
    # no model/correctness fields leak into the candidate file
    assert "predicted_label" not in rows[0] and "correct" not in rows[0]


# --------------------------------------------------------------------------- #
# Stage 4: apply_decisions
# --------------------------------------------------------------------------- #
def test_apply_decisions_keep_drop_undecided():
    curator = DatasetCurator(seed=1)
    pool = [make_instance(i, f"s {i}", "positive") for i in range(4)]
    decisions = {
        pool[0].instance_id: {"instance_id": pool[0].instance_id, "decision": "keep", "reason": "clear"},
        pool[1].instance_id: {"instance_id": pool[1].instance_id, "decision": "drop", "reason": "ambiguous"},
        pool[2].instance_id: {"instance_id": pool[2].instance_id, "decision": "drop", "reason": "mislabeled"},
        # pool[3] has no decision → undecided drop
    }
    kept, dropped = curator.apply_decisions(pool, decisions)
    assert [i.instance_id for i in kept] == [pool[0].instance_id]
    assert dropped == {"ambiguous": 1, "mislabeled": 1, "undecided": 1}


def test_load_decisions_roundtrip(tmp_path):
    curator = DatasetCurator(seed=1)
    path = tmp_path / "d.jsonl"
    path.write_text(
        '{"instance_id": "a", "decision": "keep", "reason": "x"}\n'
        '\n'  # blank line tolerated
        '{"instance_id": "b", "decision": "drop", "reason": "y"}\n',
        encoding="utf-8")
    d = curator.load_decisions(path)
    assert set(d) == {"a", "b"}
    assert d["a"]["decision"] == "keep"


# --------------------------------------------------------------------------- #
# Stage 5: select_balanced
# --------------------------------------------------------------------------- #
def test_select_balanced_hits_target_and_balances():
    curator = DatasetCurator(seed=7)
    insts = []
    for i in range(80):
        label = "positive" if i % 2 == 0 else "negative"
        insts.append(make_instance(i, "a fine sentence here " + str(i), label))
    curator.assign_length_buckets(insts)
    cfg = cfg_dict(target=40, stratify_by=["label", "length"])
    final, shortfalls = curator.select_balanced(insts, cfg)
    assert len(final) == 40
    labels = [i.label for i in final]
    # label is the primary axis -> exactly balanced for an even target
    assert labels.count("positive") == 20 and labels.count("negative") == 20
    assert shortfalls == []


def test_select_balanced_three_labels_near_even():
    curator = DatasetCurator(seed=11)
    insts = []
    for i in range(90):
        label = ["entailment", "neutral", "contradiction"][i % 3]
        insts.append(make_instance(i, "a fine sentence here " + str(i), label, genre=["a", "b"][i % 2]))
    curator.assign_length_buckets(insts)
    cfg = cfg_dict(target=50, stratify_by=["label", "genre", "length"])
    final, _ = curator.select_balanced(insts, cfg)
    assert len(final) == 50
    counts = sorted(__import__("collections").Counter(i.label for i in final).values())
    # 50 across 3 labels -> 16/17/17, balanced within 1
    assert counts[-1] - counts[0] <= 1


def test_select_balanced_shortfall_when_too_few():
    curator = DatasetCurator(seed=7)
    insts = [make_instance(i, "a fine sentence here " + str(i),
                           "positive" if i % 2 == 0 else "negative") for i in range(25)]
    curator.assign_length_buckets(insts)
    cfg = cfg_dict(target=40, stratify_by=["label"])
    final, shortfalls = curator.select_balanced(insts, cfg)
    assert len(final) == 25  # capped at what's available
    assert any("available for target" in s for s in shortfalls)


# --------------------------------------------------------------------------- #
# Stage 6: output / datasheet
# --------------------------------------------------------------------------- #
def test_write_outputs_emits_curated_and_datasheet(tmp_path):
    curator = DatasetCurator(seed=1)
    insts = []
    for i in range(6):
        inst = make_instance(i, "a fine sentence here " + str(i),
                             "positive" if i % 2 == 0 else "negative", genre="fiction")
        inst.metadata["length_bucket"] = "medium"
        insts.append(inst)
    report = CurationReport(dataset="sst2", huggingface_id="x", split="validation",
                            seed=1, target=6)
    report.vetted_kept = 6
    report.vetted_dropped = {"ambiguous": 2}
    curated, datasheet = curator.write_outputs(insts, report, tmp_path)
    lines = [json.loads(l) for l in curated.read_text().splitlines()]
    assert len(lines) == 6
    assert {"instance_id", "text", "label", "length_bucket", "genre"} <= lines[0].keys()
    assert "correct" not in lines[0] and "predicted_label" not in lines[0]
    sheet = json.loads(datasheet.read_text())
    assert sheet["final_count"] == 6
    assert sheet["label_distribution"] == {"negative": 3, "positive": 3}
    assert sheet["vetted_dropped"] == {"ambiguous": 2}
    assert "correctness_distribution" not in sheet
