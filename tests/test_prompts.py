"""Tests for prompt rendering and parser validation."""
from pathlib import Path


def load_prompt(filepath: str) -> str:
    path = Path(__file__).parent.parent / filepath
    return path.read_text(encoding='utf-8').strip()


def format_prompt(template: str, input_text: str, label_set) -> str:
    return template.format(input_text=input_text, label_set=", ".join(label_set))


def format_explain_prompt(template: str, predicted_label: str, other_labels: str = "World, Sports, Business") -> str:
    return template.format(predicted_label=predicted_label, other_labels=other_labels)


def test_classification_prompt_no_unrendered_placeholders():
    text = "This movie was great."
    label_set = ["positive", "negative"]
    prompt = load_prompt("prompts/classification.txt")
    rendered = format_prompt(prompt, text, label_set)
    assert "{label_set}" not in rendered
    assert "{input_text}" not in rendered
    assert text in rendered


def test_classification_sst2_prompt_renders():
    text = "Great movie."
    label_set = ["positive", "negative"]
    prompt = load_prompt("prompts/classification_sst2.txt")
    rendered = format_prompt(prompt, text, label_set)
    assert "{input_text}" not in rendered
    assert text in rendered


def test_classification_mnli_prompt_renders():
    text = "Premise [SEP] Hypothesis"
    label_set = ["entailment", "neutral", "contradiction"]
    prompt = load_prompt("prompts/classification_mnli.txt")
    rendered = format_prompt(prompt, text, label_set)
    assert "{input_text}" not in rendered
    assert text in rendered


def test_classification_ag_news_prompt_renders():
    text = "Some news article text."
    label_set = ["World", "Sports", "Business", "Sci/Tech"]
    prompt = load_prompt("prompts/classification_ag_news.txt")
    rendered = format_prompt(prompt, text, label_set)
    assert "{input_text}" not in rendered


def test_highlighting_explain_prompt_renders():
    prompt = load_prompt("prompts/highlighting_explain.txt")
    rendered = format_explain_prompt(prompt, "positive")
    assert "{predicted_label}" not in rendered, f"Unrendered placeholder: {rendered}"
    assert "positive" in rendered


def test_rationale_explain_prompt_renders():
    prompt = load_prompt("prompts/rationale_explain.txt")
    rendered = format_explain_prompt(prompt, "entailment")
    assert "{predicted_label}" not in rendered
    assert "entailment" in rendered


def test_counterfactual_explain_prompt_renders():
    prompt = load_prompt("prompts/counterfactual_explain.txt")
    rendered = format_explain_prompt(prompt, "Sci/Tech")
    assert "{predicted_label}" not in rendered


def test_rank_ordering_explain_prompt_renders():
    prompt = load_prompt("prompts/rank_ordering_explain.txt")
    rendered = format_explain_prompt(prompt, "positive")
    assert "{predicted_label}" not in rendered
    assert "positive" in rendered


def test_all_explain_prompts_have_no_blank_labels():
    for name in ["highlighting_explain.txt", "rationale_explain.txt",
                  "counterfactual_explain.txt", "rank_ordering_explain.txt"]:
        prompt = load_prompt(f"prompts/{name}")
        rendered = format_explain_prompt(prompt, "positive")
        assert "{predicted_label}" not in rendered, f"Blank label in {name}"
        assert "positive" in rendered, f"Label not injected in {name}"


def test_classification_json_format_wanted():
    prompt = load_prompt("prompts/classification.txt")
    assert "JSON" in prompt


def test_highlighting_json_format_wanted():
    prompt = load_prompt("prompts/highlighting_explain.txt")
    assert "JSON" in prompt
    assert "highlights" in prompt


def test_rationale_json_format_wanted():
    prompt = load_prompt("prompts/rationale_explain.txt")
    assert "JSON" in prompt
    assert "evidence" in prompt


def test_counterfactual_json_format_wanted():
    prompt = load_prompt("prompts/counterfactual_explain.txt")
    assert "JSON" in prompt
    assert "counterfactual_text" in prompt or "new_prediction" in prompt


def test_rank_ordering_json_format_wanted():
    prompt = load_prompt("prompts/rank_ordering_explain.txt")
    assert "JSON" in prompt
    assert "ranking" in prompt
