# Handwritten Assignment Assessment via OCR Noise Profiling & NLP Grading

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?style=flat-square&logo=pytorch)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow?style=flat-square&logo=huggingface)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

> **Can automated grading survive real-world OCR noise from handwritten scripts?**
> This project quantifies exactly how much — and shows how to fix it.

---

## Overview

Most automated grading systems assume clean typed text. In reality paper-based examinations require OCR to first convert handwriting to text, introducing character-level noise that can fundamentally alter meaning before a single grading decision is made.

This project presents an end-to-end pipeline that:
1. **Benchmarks three OCR engines** on 10,373 real handwriting samples
2. **Extracts a data-driven noise profile** from 221,818 character-level edit operations
3. **Injects realistic OCR noise** into student answer datasets at controlled levels
4. **Evaluates three NLP grading models** under five training conditions
5. **Demonstrates** that mixed noise-augmented training recovers up to **+0.272 F1** with negligible clean-text penalty

---

## Key Results

### OCR Benchmarking (IAM Handwriting Database, n = 10,373)

| Engine | Mean CER | Mean WER | Notes |
|--------|----------|----------|-------|
| EasyOCR | 0.5239 ± 0.1639 | 1.0153 | Primary engine for analysis |
| TrOCR (Microsoft) | ≈ 0.00 | ≈ 0.00 | Excluded — pre-trained on IAM (data leakage) |
| Tesseract v5 | ≈ 0.65 | ≈ 1.15 | Weakest on cursive handwriting |

> Mean WER > 1.0 means more than one word-level error per reference word on average. Only 5 of 10,373 samples achieve WER below 0.20.

### Character-Level Error Profile

| Error Type | Count | Share |
|------------|-------|-------|
| Substitutions | 144,097 | 64.96% |
| Insertions | 61,910 | 27.91% |
| Deletions | 15,811 | 7.13% |
| **Total** | **221,818** | **100%** |

**Top substitution confusions:**

| Rank | Pair | Count | Why |
|------|------|-------|-----|
| 1 | l → t | 3,323 | Ascending stroke ambiguity |
| 2 | l → h | 2,721 | Ascending stroke ambiguity |
| 3 | u → n | 1,977 | Mirror-image shapes |
| 4 | o → a | 1,788 | Open/closed curve confusion |
| 5 | space → t | 1,318 | Word boundary hallucination |

**Word-length vulnerability:**

| Word Length | Mean CER | Example Words |
|-------------|----------|---------------|
| 1 | 2.587 | a, I |
| 2 | 1.149 | no, is, to |
| 3 | 0.893 | not, the, and |
| 4 | 0.772 | salt, heat |
| 6-8 | ≈ 0.70 | stable range |

> Short function words like "not", "no", "is" that carry high semantic weight are disproportionately destroyed by OCR.

**Positional error distribution:**

| Position | Share |
|----------|-------|
| Middle of word | 53.3% |
| Beginning of word | 36.1% |
| End of word | 10.6% |

### Grading Model Results (SemEval-2013 Task 7, 3-way classification)

**Experiment 1 — Clean training, noisy test (Weighted F1):**

| Model | Clean | 10% Noise | 20% Noise | 30% Noise | Drop |
|-------|-------|-----------|-----------|-----------|------|
| BERT | 0.6881 | 0.4672 | 0.3694 | 0.3360 | −51.2% |
| RoBERTa | 0.7031 | 0.5277 | 0.4418 | 0.4113 | −41.5% |
| SBERT | 0.5594 | 0.4173 | 0.3101 | 0.2910 | −48.0% |

**Experiment 2 — Mixed noise training recovery (Δ = Mixed − Clean):**

| Test Condition | BERT Δ | RoBERTa Δ | SBERT Δ |
|----------------|--------|-----------|---------|
| Clean test | −0.020 | −0.004 | +0.020 |
| 10% noise | +0.152 | +0.132 | +0.103 |
| 20% noise | +0.252 | +0.177 | +0.154 |
| 30% noise | **+0.272** | **+0.220** | +0.070 |

**Full result matrix — RoBERTa (best model):**

| Train \ Test | Clean | 10% | 20% | 30% |
|---|---|---|---|---|
| Clean | 0.7031 | 0.5277 | 0.4418 | 0.4113 |
| Noisy 10% | 0.5066 | **0.6173** | 0.6087 | 0.5741 |
| Noisy 20% | 0.5226 | 0.5308 | 0.5273 | 0.5356 |
| Noisy 30% | 0.5147 | 0.5371 | 0.5270 | 0.5236 |
| **Mixed** | **0.6987** | **0.6597** | **0.6188** | **0.6309** |

---

## Pipeline

```
IAM Handwriting Images (10,373)
        ↓
EasyOCR (RGB convert + left-right spatial sort)
        ↓
CER / WER Measurement (jiwer)
        ↓
Character-Level Error Analysis (python-Levenshtein editops)
        ↓
error_profile.json
(2,299 substitution pairs · position weights · word-length CER)
        ↓
Noise Injector N(x, α)
(position weighted + word-length weighted + space bias)
        ↓
Noisy SemEval Variants
(6 files: train+test × 3 noise levels)
        ↓
NLP Grading Models
(BERT / RoBERTa fine-tuned · SBERT bi-encoder)
        ↓
Weighted F1 + Macro F1
(5 training conditions × 4 test conditions)
```

---

## Noise Injection Design

Our injector is grounded in real OCR behavior, not random noise.

**Error type sampling from IAM-derived rates:**

| Type | Probability |
|------|-------------|
| Substitution | 64.96% |
| Insertion | 27.91% |
| Deletion | 7.13% |

**Position weighting — errors are not uniform across a word:**

| Position | Real Error Share | Injection Weight |
|----------|-----------------|-----------------|
| Middle | 53.3% | 0.53 |
| Beginning | 36.1% | 0.36 |
| End | 10.6% | 0.11 |

**Word-length weighting — short words are hit harder:**

| Word Length | Corruption Multiplier |
|-------------|----------------------|
| 1–2 chars | 1.8× base rate |
| 3–4 chars | 1.3× base rate |
| 5–8 chars | 1.0× (baseline) |
| 9+ chars | 0.85× base rate |

**Space deletion bias — word merging is a real OCR artifact:**

Space characters receive 3.0× higher deletion probability, causing adjacent words to merge — exactly as observed in real EasyOCR output.

**Example output at each noise level:**

| Level | Text |
|-------|------|
| Original | Let the water evaporate and the salt is left behind. |
| 10% noise | Lat thenwater evaborate and the salt is teft be ind. |
| 20% noise | Lht thbsuater baaaorate and the sat is teft behiind. |
| 30% noise | Lyi tttater etraiorate andtthe saht as teft beh iid. |

---

## Datasets

| Dataset | Purpose | Size |
|---------|---------|------|
| [IAM Handwriting Database](https://fki.tic.heia-fr.ch/databases/iam-handwriting-database) | OCR benchmarking + error profiling | 10,373 line images, 657 writers |
| [SemEval-2013 Task 7](https://github.com/myrosia/semeval-2013-task7) | Short answer grading evaluation | 4,969 train / 540 test |

**SemEval label distribution:**

| Label | Train | Test | Share |
|-------|-------|------|-------|
| Incorrect | 2,462 | 249 | 49.5% |
| Correct | 2,008 | 233 | 40.4% |
| Contradictory | 499 | 58 | 10.0% |

---

## Models

| Model | Architecture | Training Setup |
|-------|-------------|----------------|
| BERT | bert-base-uncased + classification head | Fine-tuned, lr=2e-5, batch=16, 8 epochs |
| RoBERTa | roberta-base + classification head | Fine-tuned, lr=2e-5, batch=16, 8 epochs |
| SBERT | all-MiniLM-L6-v2 bi-encoder | Cosine sim features + LogisticRegression |

**Input format for all models:**

```
Question: [Q]  Reference: [R]  Answer: [A]
```

**Five training conditions per model:**

| Condition | Training Data |
|-----------|--------------|
| Clean | Original clean student answers |
| Noisy 10% | 10% character corruption |
| Noisy 20% | 20% character corruption |
| Noisy 30% | 30% character corruption |
| Mixed | Clean + Noisy 10% + Noisy 20% + Noisy 30% |

---

## Project Structure

```
automated_assessment/
├── data/
│   ├── semeval_train.csv
│   ├── semeval_test.csv
│   └── noisy/
│       ├── semeval_train_noisy_10.csv
│       ├── semeval_train_noisy_20.csv
│       ├── semeval_train_noisy_30.csv
│       ├── semeval_test_noisy_10.csv
│       ├── semeval_test_noisy_20.csv
│       └── semeval_test_noisy_30.csv
├── notebooks/
│   ├── error_analysis.ipynb
│   ├── noise_injection.ipynb
│   └── scoring_pipeline.ipynb
├── outputs/
│   ├── error_profile.json
│   ├── figures/
│   └── results/
└── models/
    ├── bert_ft_*/
    └── roberta_ft_*/
```

---

## Key Findings

**F1.** RoBERTa is the most noise-robust model — retains highest F1 across all noise levels under both clean and mixed training.

**F2.** Clean training is brittle — all models lose 41–51% weighted F1 by 30% noise. The sharpest drop occurs between clean and just 10% noise.

**F3.** Mixed training is the practical solution — recovers up to +0.272 F1 with less than 0.02 clean-text penalty. Recommended for real deployment.

**F4.** Matched noise training overfits — models trained at one noise level underperform at others, making them impractical when deployment noise level is unknown.

**F5.** Short function words are most vulnerable — single-character words achieve CER of 2.587, meaning "not", "no", "is" are reliably destroyed by OCR, directly damaging semantic scoring.

---

## Tech Stack

| Category | Tools |
|----------|-------|
| OCR | EasyOCR, TrOCR (HuggingFace), Tesseract v5 |
| Error Analysis | python-Levenshtein, jiwer |
| NLP Models | HuggingFace Transformers, Sentence-Transformers |
| Training | PyTorch, AdamW, linear warmup scheduler |
| Data | pandas, numpy |
| Visualisation | matplotlib (PDF vector figures) |
| Platform | Google Colab (GPU), VS Code (local) |

---



---

## Authors

Nikhil Chaudhary · Ashutosh Ranjan · Pankaj · Nikhil Kumar · Aryan Sachan · Bhupathi SaiCharan

*Computer Science & Engineering, Faculty of Technology, University of Delhi*
