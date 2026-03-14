# Systematic Study of Structural Prompting in Arithmetic LLMs
Hannah Zhao
STAT 359: LLM Reasoning & Mechanics

## Project Overview
This project investigates the limits of mathematical reasoning in small-scale Transformers. I specifically looked at how a model trained on 1-to-20 arithmetic performs on Out-of-Distribution (OOD) tasks such as multi-digit addition, negative numbers, and unseen operators like division. The goal was to analyze where the model maintains correct formatting (syntax) versus where it fails to perform the actual calculation (semantics).

## Directory Structure
The project is organized within the student/ folder:

01_training_and_analysis.ipynb: Contains the full data pipeline, including 100k sample generation, BPE tokenizer training, and logs for the instruction fine-tuning process.

02_final_visualizations.ipynb: The analysis notebook that loads saved checkpoints to generate Tokenizer X-rays, performance metrics, and visualization charts.

data/: Includes the foundational and instruction corpora, plus .jsonl files for OOD stress tests.

models/: Stores .pt checkpoints for both the baseline and instruction-tuned models.

evaluation_results/: Contains the JSON and text logs for all evaluation runs.

### Note: Assignment folders 1-3 and the general final_project folder are from previous coursework and are not part of this specific study.

## How to Reproduce & Evaluate
1. Environment Setup
This project uses poetry. Run poetry install and poetry shell to set up the dependencies.

2. Viewing the Training Pipeline (No Execution Required)
You can open student/01_training_and_analysis.ipynb to read through the data generation steps and view the preserved training logs. The final models are already saved in the models/ directory, so you do not need to rerun this notebook.

3. Running the Analysis (Interactive)
To evaluate the model and regenerate the findings:

Open student/02_final_visualizations.ipynb.

You can safely select "Run All". This notebook will load the pre-trained .pt checkpoints from the models/ folder, execute the OOD stress tests, and generate the final charts and Tokenizer X-Ray diagnostics instantly.

## Key Findings
Physical Failure: Tokenizer X-rays show the model cannot "see" certain operators (like /) because they were not included in the BPE training.

Reasoning Compression: For 4-digit and 5-digit math, the model attempts to fit the logic into a template length meant for 2-digit numbers, causing reasoning to fail.

Prompt Limits: Few-shot prompting helped the model maintain the "thought" format but did not recover the underlying arithmetic logic for OOD categories.
