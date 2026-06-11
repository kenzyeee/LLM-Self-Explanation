from pathlib import Path
from typing import Dict, Any


class PaperGenerator:
    def __init__(self, results: Dict[str, Any], config: Dict[str, Any]):
        self.results = results
        self.config = config
        self.mean_ecs = results.get('mean_ecs', 0.0)
        self.datasets = results.get('datasets', [])
        self.experiment_name = config.get('experiment', {}).get('name', 'LLM Explanation Agreement Study')

    def generate_paper(self, output_path):
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        doc = self._build_latex()
        output_path.write_text(doc, encoding='utf-8')

    def _build_latex(self):
        return (
            r"\documentclass{article}" + "\n"
            + r"\usepackage{geometry}" + "\n"
            + r"\geometry{margin=1in}" + "\n"
            + r"\begin{document}" + "\n\n"
            + self._title()
            + "\n\n"
            + self._abstract()
            + "\n\n"
            + self._methodology()
            + "\n\n"
            + self._results()
            + "\n\n"
            + r"\end{document}"
        )

    def _title(self):
        return (
            r"\title{" + self.experiment_name + "}" + "\n"
            + r"\author{Research Team}" + "\n"
            + r"\date{\today}" + "\n"
            + r"\maketitle"
        )

    def _abstract(self):
        return (
            r"\begin{abstract}" + "\n"
            + "This study investigates cross-strategy agreement among LLM self-explanations. "
            + f"We analyze {len(self.datasets)} datasets with mean ECS = {self.mean_ecs:.4f}. "
            + r"\end{abstract}"
        )

    def _methodology(self):
        return (
            r"\section{Methodology}" + "\n\n"
            + "We employ four explanation strategies: Highlighting (H), Rationale (R), "
            + "Counterfactual (CF), and Rank Ordering (RO). "
            + "Agreement is measured via Jaccard similarity on normalized token sets. "
            + "The Explanation Consensus Score (ECS) averages six pairwise Jaccard values."
        )

    def _results(self):
        datasets_str = ", ".join(self.datasets) if self.datasets else "N/A"
        return (
            r"\section{Results}" + "\n\n"
            + f"Datasets analyzed: {datasets_str} \\\\" + "\n"
            + f"Mean ECS: {self.mean_ecs:.4f}" + "\n"
        )
