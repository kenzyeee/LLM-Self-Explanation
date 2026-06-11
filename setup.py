"""Setup script for LLM Explanation Agreement Study package."""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file for long description
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

# Read requirements from requirements.txt
requirements_file = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_file.exists():
    with open(requirements_file, "r", encoding="utf-8") as f:
        requirements = [
            line.strip()
            for line in f
            if line.strip() and not line.startswith("#")
        ]

setup(
    name="llm-explanation-agreement",
    version="0.1.0",
    description="Research pipeline for investigating cross-strategy agreement among LLM self-explanations",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/yourusername/llm-explanation-agreement",
    license="MIT",
    
    # Package configuration
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    
    # Dependencies
    install_requires=requirements,
    
    # Optional dependencies for development
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "hypothesis>=6.82.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.4.0",
            "isort>=5.12.0",
        ],
        "docs": [
            "sphinx>=6.0.0",
            "sphinx-rtd-theme>=1.2.0",
        ],
    },
    
    # Entry points for command-line scripts
    entry_points={
        "console_scripts": [
            "llm-explain=scripts.run_experiment:main",
            "llm-prepare-data=scripts.prepare_data:main",
            "llm-inference=scripts.run_inference:main",
            "llm-compute-metrics=scripts.compute_metrics:main",
            "llm-validity-tests=scripts.run_validity_tests:main",
            "llm-statistics=scripts.compute_statistics:main",
            "llm-plots=scripts.generate_plots:main",
            "llm-paper=scripts.generate_paper:main",
        ],
    },
    
    # Package metadata
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
    ],
    
    # Include non-Python files
    include_package_data=True,
    package_data={
        "": [
            "config/*.yaml",
            "prompts/*.txt",
        ],
    },
    
    # Project URLs
    project_urls={
        "Bug Reports": "https://github.com/yourusername/llm-explanation-agreement/issues",
        "Source": "https://github.com/yourusername/llm-explanation-agreement",
        "Documentation": "https://llm-explanation-agreement.readthedocs.io",
    },
    
    # Keywords for PyPI
    keywords=[
        "nlp",
        "explainability",
        "interpretability",
        "large language models",
        "llm",
        "self-explanation",
        "research",
    ],
)
