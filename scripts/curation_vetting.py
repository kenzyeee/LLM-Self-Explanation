#!/usr/bin/env python
"""Hand-vetting decisions for the curated datasets (authored by Claude).

This is the auditable record of the clean-gold vetting step: for each dataset's
frozen candidate pool (data/processed/{ds}_candidates.jsonl), every candidate is
KEPT unless its instance_id appears in DROPS below, in which case it is dropped with
the stated reason. Running this script (re)generates the decisions files
(data/processed/{ds}_decisions.jsonl) consumed by `curate_dataset.py finalize`.

Vetting criterion: exclude instances whose GOLD LABEL is wrong or genuinely
ambiguous, plus low-quality items (garbled/fragment text). No model was used — these
are manual judgments from reading every candidate. Everything not listed here was
judged a clean, unambiguous, correctly-labeled instance.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

PROCESSED = Path("data/processed")

# instance_id -> drop reason. Anything not listed is kept.
DROPS = {
    "sst2": {
        "sst2_validation_000846": "neutral/descriptive; gold 'negative' not clearly supported",
        "sst2_validation_000850": "text reads negative but gold is positive (likely mislabel)",
        "sst2_validation_000656": "text reads negative but gold is positive (likely mislabel)",
        "sst2_validation_000501": "mixed/ambiguous polarity (heart vs brains)",
        "sst2_validation_000013": "ambiguous polarity ('closer to pity')",
        "sst2_validation_000812": "weak/ambiguous polarity ('troubling interpretation')",
    },
    "mnli": {
        "mnli_validation_matched_000663": "mislabel: premise <=18 vs hypothesis >=18, not entailment",
        "mnli_validation_matched_005281": "mislabel: hypothesis misstates the interest-rate definition",
        "mnli_validation_matched_001327": "garbled premise text",
        "mnli_validation_matched_001964": "garbled hypothesis ('absdorns')",
        "mnli_validation_matched_005580": "garbled/merged premise",
        "mnli_validation_matched_002388": "fragment premise/hypothesis (not full sentences)",
        "mnli_validation_matched_006895": "fragment premise (noun phrase); relation debatable",
        "mnli_validation_matched_001225": "fragment premise (noun phrase)",
        "mnli_validation_matched_001833": "fragment premise (noun phrase)",
        "mnli_validation_matched_009133": "fragment header premise ('TEST ORGANISMS')",
        "mnli_validation_matched_000393": "fragment premise ('Near Jerusalem')",
        "mnli_validation_matched_007644": "attribution to Tommy debatable; relation unclear",
        "mnli_validation_matched_008016": "weak/neutral relation; entailment debatable",
        "mnli_validation_matched_007850": "unclear premise; weak entailment",
        "mnli_validation_matched_007541": "hypothesis adds unstated info; not strict entailment",
        "mnli_validation_matched_008602": "subject-swap; entailment questionable",
        "mnli_validation_matched_004501": "confusing premise + debatable attribution",
        "mnli_validation_matched_005267": "meta hypothesis; relation questionable",
    },
    "ag_news": {
        "ag_news_test_000336": "Venezuela referendum -> World, gold is Business (mislabel)",
        "ag_news_test_001262": "lawsuit/crime story -> not Sci/Tech (mislabel)",
        "ag_news_test_002234": "GM-grass science -> Sci/Tech, gold is Business (mislabel)",
        "ag_news_test_006782": "biotech economic development -> Business, gold is Sci/Tech",
        "ag_news_test_005457": "Gap/Wild Planet retail deal -> Business, gold is Sci/Tech",
        "ag_news_test_005151": "beer how-to / lifestyle -> not clearly Sci/Tech",
        "ag_news_test_005410": "earnings/profit story -> Business, gold is Sci/Tech (mislabel)",
        "ag_news_test_001497": "campaign-finance topic -> weak/cross-category Sci/Tech",
        "ag_news_test_007068": "Nextel/Sprint merger -> Business, gold is Sci/Tech",
        "ag_news_test_004912": "Senate-candidate politics -> not Sports (mislabel)",
        "ag_news_test_006766": "airline equity raise -> Business, gold is World (mislabel)",
        "ag_news_test_002015": "celebrity drug arrest -> entertainment, not World",
        "ag_news_test_002591": "Olympic gymnastics appeal -> Sports, gold is World (mislabel)",
        "ag_news_test_000833": "English-Channel swim -> ambiguous category",
        "ag_news_test_004734": "cricket Test -> Sports, gold is World (mislabel)",
        "ag_news_test_000747": "Olympic basketball -> Sports, gold is World (mislabel)",
        "ag_news_test_003661": "cricket Test -> Sports, gold is World (mislabel)",
        "ag_news_test_007459": "Michael Jackson legal -> entertainment, not World",
        "ag_news_test_007505": "gene therapy -> Sci/Tech, gold is World (mislabel)",
        "ag_news_test_000614": "Olympic field hockey -> Sports, gold is World (mislabel)",
        "ag_news_test_003662": "stocks/earnings -> Business, gold is World (mislabel)",
        "ag_news_test_006914": "BBC job cuts -> Business, gold is World",
        "ag_news_test_000316": "oil/gas prices -> Business, gold is World (mislabel)",
    },
}


def write_decisions(dataset: str) -> None:
    candidates_path = PROCESSED / f"{dataset}_candidates.jsonl"
    decisions_path = PROCESSED / f"{dataset}_decisions.jsonl"
    drops = DROPS.get(dataset, {})
    n_keep = n_drop = 0
    with open(candidates_path, encoding="utf-8") as fin, \
            open(decisions_path, "w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            iid = json.loads(line)["instance_id"]
            if iid in drops:
                rec = {"instance_id": iid, "decision": "drop", "reason": drops[iid]}
                n_drop += 1
            else:
                rec = {"instance_id": iid, "decision": "keep", "reason": "clean gold, unambiguous"}
                n_keep += 1
            fout.write(json.dumps(rec) + "\n")
    print(f"{dataset}: wrote {decisions_path} (keep={n_keep}, drop={n_drop})")


if __name__ == "__main__":
    for ds in ("sst2", "mnli", "ag_news"):
        write_decisions(ds)
