from src.utils.pretty_printer import PrettyPrinter


class TestPrettyPrinter:
    def setup_method(self):
        self.printer = PrettyPrinter()

    def test_format_instance_basic(self):
        instance = {
            'instance_id': '1',
            'text': 'great movie',
            'ground_truth_label': 'positive',
            'predicted_label': 'positive',
            'confidence': 0.95,
            'correct': True,
            'raw_highlighting': 'some highlighted text',
            'highlighting_parsed': True,
            'raw_rationale': 'rationale explanation',
            'rationale_parsed': True,
            'raw_counterfactual': 'counterfactual text',
            'counterfactual_parsed': False,
            'raw_rank_ordering': 'ordered tokens',
            'rank_ordering_parsed': True,
        }
        result = self.printer.format_instance(instance)
        assert 'Instance ID: 1' in result
        assert 'great movie' in result
        assert 'positive' in result
        assert '[OK]' in result
        assert '[FAILED]' in result

    def test_format_instance_missing_keys(self):
        instance = {}
        result = self.printer.format_instance(instance)
        assert 'N/A' in result
        assert '[FAILED]' in result

    def test_format_raw_strategy_truncation(self):
        result = self.printer._format_raw_strategy("Test", "x" * 200, True)
        assert "[OK]" in result
        assert "..." in result
        assert len(result) < 200

    def test_format_raw_strategy_short(self):
        result = self.printer._format_raw_strategy("Test", "short", False)
        assert "[FAILED]" in result
        assert "short" in result

    def test_format_normalized_tokens(self):
        instance = {
            'highlighting_tokens': {'good', 'great'},
            'rationale_tokens': {'good', 'nice'},
            'counterfactual_tokens': {'bad'},
            'rank_ordering_tokens': [('good', 1), ('bad', 2)],
        }
        result = self.printer.format_normalized_tokens(instance)
        assert 'H:' in result
        assert 'R:' in result
        assert 'CF:' in result
        assert 'RO:' in result
        assert 'good' in result

    def test_format_normalized_tokens_empty_ro(self):
        instance = {
            'highlighting_tokens': {'good'},
            'rationale_tokens': {'nice'},
            'counterfactual_tokens': {'bad'},
            'rank_ordering_tokens': [],
        }
        result = self.printer.format_normalized_tokens(instance)
        assert 'RO: []' in result or 'RO:' in result

    def test_format_normalized_tokens_missing_keys(self):
        instance = {}
        result = self.printer.format_normalized_tokens(instance)
        # Should not crash
        assert result is not None

    def test_format_pairwise_agreements(self):
        instance = {
            'jaccard_H_R': 0.5,
            'jaccard_H_CF': 0.3,
            'jaccard_H_RO': 0.7,
            'jaccard_R_CF': 0.2,
            'jaccard_R_RO': 0.6,
            'jaccard_CF_RO': 0.4,
            'kendall_H_RO': 0.8,
            'ecs': 0.5,
        }
        result = self.printer.format_pairwise_agreements(instance)
        assert 'H-R' in result
        assert '0.5000' in result
        assert 'ECS' in result

    def test_format_pairwise_agreements_none_values(self):
        instance = {}
        result = self.printer.format_pairwise_agreements(instance)
        assert 'N/A' in result

    def test_highlight_consensus_core_with_tokens(self):
        text = "the quick brown fox"
        cc_tokens = {"quick", "fox"}
        result = self.printer.highlight_consensus_core(text, cc_tokens)
        assert '\x1b[93mquick\x1b[0m' in result
        assert '\x1b[93mfox\x1b[0m' in result

    def test_highlight_consensus_core_empty(self):
        text = "the quick brown fox"
        result = self.printer.highlight_consensus_core(text, set())
        assert result == text

    def test_highlight_consensus_core_longer_first(self):
        text = "the quick brown fox"
        cc_tokens = {"fox", "brown fox"}
        result = self.printer.highlight_consensus_core(text, cc_tokens)
        # "brown fox" is replaced first (longer), then "fox" within the ANSI text
        assert '\x1b[93m' in result
        assert 'brown' in result
