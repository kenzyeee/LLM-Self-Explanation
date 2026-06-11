import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
from typing import Dict, List
from src.statistics.statistical_tests import CorrelationResult


class VisualizationGenerator:
    def __init__(self, output_dir, dpi=72):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.figure_dpi = dpi

    def _save(self, fig, filename):
        filepath_pdf = self.output_dir / f"{filename}.pdf"
        filepath_png = self.output_dir / f"{filename}.png"
        fig.savefig(filepath_pdf, dpi=self.figure_dpi, bbox_inches="tight")
        fig.savefig(filepath_png, dpi=self.figure_dpi, bbox_inches="tight")
        plt.close(fig)

    def plot_agreement_heatmap(self, data: pd.DataFrame):
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(data, annot=True, fmt=".3f", cmap="YlOrRd", vmin=0, vmax=1, ax=ax)
        ax.set_title("Mean Pairwise Jaccard Similarity")
        self._save(fig, "agreement_heatmap")

    def plot_ecs_distributions(self, ecs_by_dataset: Dict[str, List[float]],
                                ecs_by_model: Dict[str, List[float]]):
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for idx, (data, title) in enumerate([
            (ecs_by_dataset, "ECS by Dataset"),
            (ecs_by_model, "ECS by Model")
        ]):
            rows = []
            for group, values in data.items():
                for v in values:
                    rows.append({"Group": group, "ECS": v})
            df = pd.DataFrame(rows)
            if not df.empty:
                sns.boxplot(data=df, x="Group", y="ECS", ax=axes[idx])
                sns.stripplot(data=df, x="Group", y="ECS", color="black", alpha=0.3, size=3, ax=axes[idx])
            axes[idx].set_title(title)
        plt.tight_layout()
        self._save(fig, "ecs_distributions")

    def plot_confidence_ecs_scatter(self, confidences: List[float], ecs_values: List[float],
                                     correlation: CorrelationResult):
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(confidences, ecs_values, alpha=0.5)
        ax.set_xlabel("Confidence")
        ax.set_ylabel("ECS")
        ax.set_title(f"Confidence vs ECS (rho={correlation.rho:.3f}, p={correlation.p_value:.4f})")
        self._save(fig, "confidence_ecs_scatter")

    def plot_flip_rate_comparison(self, cc_rates: Dict[str, float], random_rates: Dict[str, float]):
        fig, ax = plt.subplots(figsize=(6, 5))
        categories = list(cc_rates.keys())
        cc_values = [cc_rates[k] for k in categories]
        random_values = [random_rates.get(k, 0) for k in categories]
        x = range(len(categories))
        width = 0.35
        ax.bar([i - width / 2 for i in x], cc_values, width, label="CC", color="#e74c3c", alpha=0.8)
        ax.bar([i + width / 2 for i in x], random_values, width, label="Random", color="#3498db", alpha=0.8)
        ax.set_ylabel("Flip Rate")
        ax.set_title("Prediction Flip Rates After Token Removal")
        ax.set_xticks(list(x))
        ax.set_xticklabels(categories)
        ax.legend()
        self._save(fig, "flip_rate_comparison")

    def plot_robustness_analysis(self, data: pd.DataFrame):
        fig, ax = plt.subplots(figsize=(8, 5))
        if not data.empty:
            sns.boxplot(data=data, x="Variation", y="ECS", ax=ax)
            sns.stripplot(data=data, x="Variation", y="ECS", color="black", alpha=0.3, size=3, ax=ax)
        ax.set_title("Robustness Analysis")
        self._save(fig, "robustness_analysis")
