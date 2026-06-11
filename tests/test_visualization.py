import pytest
import pandas as pd
from pathlib import Path
from src.plots.visualization_generator import VisualizationGenerator
from src.statistics.statistical_tests import CorrelationResult


class TestVisualizationGenerator:
    @pytest.fixture
    def viz(self, tmp_path):
        return VisualizationGenerator(tmp_path, dpi=72)

    def test_plot_agreement_heatmap(self, viz):
        data = pd.DataFrame({
            'H-R': [0.5, 0.3, 0.4],
            'H-CF': [0.2, 0.4, 0.3],
            'H-RO': [0.6, 0.5, 0.7],
        }, index=['SST-2', 'MNLI', 'AG News'])
        viz.plot_agreement_heatmap(data)
        assert (viz.output_dir / 'agreement_heatmap.pdf').exists()
        assert (viz.output_dir / 'agreement_heatmap.png').exists()

    def test_plot_ecs_distributions(self, viz):
        ecs_by_dataset = {'SST-2': [0.3, 0.5, 0.7], 'MNLI': [0.2, 0.4, 0.6]}
        ecs_by_model = {'Model A': [0.4, 0.5], 'Model B': [0.3, 0.6]}
        viz.plot_ecs_distributions(ecs_by_dataset, ecs_by_model)
        assert (viz.output_dir / 'ecs_distributions.pdf').exists()
        assert (viz.output_dir / 'ecs_distributions.png').exists()

    def test_plot_confidence_ecs_scatter(self, viz):
        confidences = [0.5, 0.6, 0.7, 0.8, 0.9]
        ecs_values = [0.3, 0.4, 0.5, 0.6, 0.7]
        corr = CorrelationResult(rho=0.8, p_value=0.01, ci_lower=0.5, ci_upper=0.95)
        viz.plot_confidence_ecs_scatter(confidences, ecs_values, corr)
        assert (viz.output_dir / 'confidence_ecs_scatter.pdf').exists()
        assert (viz.output_dir / 'confidence_ecs_scatter.png').exists()

    def test_plot_flip_rate_comparison(self, viz):
        cc_rates = {'SST-2': 0.6, 'MNLI': 0.5}
        random_rates = {'SST-2': 0.2, 'MNLI': 0.3}
        viz.plot_flip_rate_comparison(cc_rates, random_rates)
        assert (viz.output_dir / 'flip_rate_comparison.pdf').exists()
        assert (viz.output_dir / 'flip_rate_comparison.png').exists()

    def test_plot_robustness_analysis_empty(self, viz):
        viz.plot_robustness_analysis(pd.DataFrame())

    def test_plot_robustness_analysis(self, viz):
        data = pd.DataFrame({
            'Variation': ['A', 'A', 'B', 'B'],
            'ECS': [0.3, 0.4, 0.5, 0.6],
        })
        viz.plot_robustness_analysis(data)
        assert (viz.output_dir / 'robustness_analysis.pdf').exists()
        assert (viz.output_dir / 'robustness_analysis.png').exists()

    def test_output_directory_creation(self, tmp_path):
        viz = VisualizationGenerator(tmp_path / "new_dir", dpi=72)
        assert (tmp_path / "new_dir").exists()
