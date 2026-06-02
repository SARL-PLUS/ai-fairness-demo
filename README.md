<p align="center">
  <strong>🐱 AI Fairness Demo 🐶</strong><br>
  <em>Can an algorithm be fair to everyone? Spoiler: it's mathematically impossible.</em>
</p>

<p align="center">
  <a href="https://sarl-plus.github.io/ai-fairness-demo/fairness_demo.html">🚀 Try the Live Demo</a>
</p>

---

## What Is This?

An **interactive teaching tool** that lets you explore one of the most important results in AI ethics: the **fairness impossibility theorem**.

Using a playful scenario — an AI that decides which shelter animals (🐱 cats vs. 🐶 dogs) get premium website placement — you'll discover that **three intuitive definitions of fairness cannot all be satisfied at the same time** when group base rates differ.

This isn't just a toy example. The same mathematical tension underlies real-world debates:
- **COMPAS** — criminal recidivism prediction (ProPublica vs. Northpointe)
- **Hiring algorithms** — equal selection rates vs. equal accuracy
- **Medical AI** — equal diagnostic quality across patient groups

## The Three Fairness Criteria

### ⚖️ Independence — *"Equal chances for everyone"*

The AI should give **the same percentage** of cats and dogs the premium spot.

> **Example:** If 60% of dogs get premium placement, then 60% of cats should too — regardless of whether they actually get adopted. It's about equal opportunity.

### 🔬 Separation — *"Equal mistakes for everyone"*

The AI should make the **same types of errors** for both groups.

> **Example:** If the AI correctly identifies 80% of dogs that *will* be adopted (and misses 20%), it should also correctly identify 80% of adoptable cats — not just 50%. Same accuracy for both, not just overall.

### 📊 Sufficiency — *"Same score means the same thing"*

A score of "70" should mean the **same probability of adoption** whether it's a cat or a dog.

> **Example:** If 75% of dogs who score 70+ actually get adopted, then 75% of cats who score 70+ should also get adopted. The score should be equally trustworthy for both groups.

### 🚫 The Catch — You Can't Have All Three

When the groups have **different base rates** (dogs: 70% adopted, cats: 40% adopted), satisfying all three criteria at the same time is **mathematically impossible**. You always have to choose — and that choice is an ethical one, not a technical one.

**The impossibility theorem** (Chouldechova 2017; Kleinberg, Mullainathan & Raghavan 2016): When the base rates differ between groups, you **cannot satisfy all three criteria simultaneously** — except in trivial cases (perfect prediction or equal base rates).


## How to Use

### Interactive Fairness Demo

Simply open [`fairness_demo.html`](fairness_demo.html) in any modern browser — **no installation required**.

- 🎛️ **Adjust thresholds** for cats and dogs independently
- 🎯 **Click goal presets** to see how different loss functions trade off fairness criteria
- ⚖️ **Try to make all three fairness criteria green** — you can't!

### Cat/Dog Classifier (Optional Hands-On)

A companion deep learning project that trains a real image classifier using PyTorch and transfer learning.

```bash
cd classifier

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate    # Linux/Mac
# venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Option A: Jupyter Notebook (recommended for workshops)
jupyter notebook cat_dog_training.ipynb

# Option B: Python script
python train.py
```

**No GPU required** — transfer learning (ResNet-18) makes it work on any laptop in minutes.

## Project Structure

```
ai-fairness-demo/
├── README.md                           ← You are here
├── fairness_demo.html                  ← Interactive fairness impossibility demo
│                                         (open in browser, no setup needed)
└── classifier/                         ← Hands-on deep learning project
    ├── cat_dog_training.ipynb          ← Jupyter notebook (workshop version)
    ├── train.py                        ← Training script
    ├── download_data.py                ← Dataset download & preview
    └── requirements.txt                ← Python dependencies
```

## For Instructors

This demo is designed for **university courses and workshops** on AI foundations, ethics, or machine learning. Suggested flow:

1. **Show the interactive demo** — let students try to "beat" the impossibility theorem
2. **Discuss real-world parallels** — COMPAS, hiring, medical AI
3. **Hands-on coding** (optional) — train the classifier, then connect back to the fairness questions: *What happens when we deploy this model across different populations?*

### Discussion Questions
- If you can only satisfy 2 out of 3 fairness criteria, which one do you sacrifice — and who decides?
- Is a "fair" algorithm that treats both groups equally always better than an "unfair" one that produces better outcomes overall?
- Should the choice of fairness criterion be a **technical** or **political/ethical** decision?

## References

- Hardt, M. & Recht, B. (2022). [*Patterns, Predictions, and Actions: Foundations of Machine Learning*](https://mlstory.org). Chapter 2.
- Chouldechova, A. (2017). Fair prediction with disparate impact. *Big Data*, 5(2).
- Kleinberg, J., Mullainathan, S. & Raghavan, M. (2016). Inherent trade-offs in the fair determination of risk scores. *ITCS 2017*.

## License

MIT License — feel free to use, modify, and share for educational purposes.

---

<p align="center">
  Built for the <strong>KI Grundlagen und Anwendung</strong> course at the University of Salzburg<br>
  <a href="https://github.com/SARL-PLUS">SARL-PLUS</a>
</p>
