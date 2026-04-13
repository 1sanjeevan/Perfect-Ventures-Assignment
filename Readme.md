# Vexoo Labs – AI Engineer Assignment

**Author:** Sanjeevan
**Date:** April 2026

## Files
- `part1_ingestion.py` → Sliding Window + Knowledge Pyramid
- `part2_training.py` → GSM8K LoRA Fine-Tuning (Simulation Mode)
- `bonus_adapter.py` → Reasoning-Aware Routing Adapter

## How to Run

Install dependencies:
pip install transformers datasets peft accelerate torch tqdm

Run Part 1:
python part1_ingestion.py

Run Bonus Adapter:
python bonus_adapter.py

Run Part 2 (Simulation - no GPU needed):
python part2_training.py

## Notes
- part2_training.py runs in SIMULATE_MODE=True by default (no GPU needed)
- For real training, set SIMULATE_MODE=False and run on Google Colab with GPU
- Part 1 uses pure Python stdlib, no extra installs needed