# CMPT 423 Machine Learning — Complete Study Notes

**Course:** CMPT 423 Machine Learning · **Instructor:** Lucas Lehnert · **Term:** Fall 2026 · **University of Saskatchewan** · **Language:** Python
**Textbooks:** Murphy, *Probabilistic Machine Learning: An Introduction* (MIT Press, 2022; free at probml.github.io/pml-book/book1.html) and Bishop, *Pattern Recognition and Machine Learning* (2006).
**Covers:** Lectures 02–19 (there is no Lecture 01 in the materials; L19 is the midterm example exercise and reference sheet). New lectures get added in the same format.

---

## How to use these notes

Every lecture follows the same pattern:

1. **Numbered subsections** that mirror the slides, in slide order. For each idea you get: *plain words first → the formula or algorithm with every symbol defined → a worked example with real numbers or a traced run.*
2. **📖 Textbook deep dive** — the extra explanations, examples and numbers in Murphy and Bishop, plus worked answers to their conceptual exercises. Items the textbook covers but the slides do not are tagged **➕ Textbook extra**.
3. **✅ Key takeaways** — the five-line version to re-read before an exam.

Page numbers are **printed page numbers**: Murphy "p.57" is the number printed on the page (PDF page − 30); Bishop "p.206" is the printed page number. Section numbers are given too, so you can find them in either edition of the file.

### The one running example (used in every lecture)

Ten students each reported **hours studied** $h$ and then wrote a test. We record the **score** $s$ (out of 100) and whether they **passed** ($t=1$ if $s \ge 65$, else $0$). A 3-class label $y_3$ splits scores into fail ($<65$) → class 0, pass ($65$–$74$) → class 1, distinction ($\ge 75$) → class 2.

| Student | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| hours $h$ | 4 | 2 | 6 | 1 | 3 | 5 | 7 | 3 | 8 | 2 |
| score $s$ | 66 | 48 | 71 | 45 | 57 | 62 | 77 | 59 | 83 | 56 |
| passed $t$ | 1 | 0 | 1 | 0 | 0 | 0 | 1 | 0 | 1 | 0 |
| class $y_3$ | 1 | 0 | 1 | 0 | 0 | 0 | 2 | 0 | 2 | 0 |

Class counts: 6 fail, 2 pass, 2 distinction. Four of ten passed, so the pass rate is $0.4$.

```python
import numpy as np
h  = np.array([4, 2, 6, 1, 3, 5, 7, 3, 8, 2], dtype=float)   # hours
s  = np.array([66, 48, 71, 45, 57, 62, 77, 59, 83, 56], dtype=float)  # score
t  = (s >= 65).astype(int)                                    # pass / fail
y3 = np.digitize(s, [65, 75])                                 # 0 fail, 1 pass, 2 distinction
print(t, y3)   # [1 0 1 0 0 0 1 0 1 0] [0 ...] -> see table
```

> Note: `np.digitize(s,[65,75])` gives 0 for $s<65$, 1 for $65\le s<75$, 2 for $s\ge 75$ — exactly the table above.

---

## Part A. Course overview (from the syllabus)

### A.1 Who, when, where

| Item | Detail |
|---|---|
| Course | CMPT 423.3 Machine Learning, Fall 2026 |
| Instructor | Lucas Lehnert |
| Class times | Mon / Wed / Fri 11:30–12:20, ENG 2B53 |
| Contact | lucas.lehnert@usask.ca (preferred method) |
| Course sites | Canvas and the course GitLab: `git.cs.usask.ca/cmpt-423-ml-f26` |
| Prerequisites | CMPT 317.3; one of STAT 242.3 (preferred), STAT 245.3 or EE 216.3; and MATH 164.3 |
| Recording | Recording of the course is **not permitted**, and the course is not recorded |

### A.2 Grading

| Component | Weight | Notes |
|---|---|---|
| Assignment 0 (A0) | 5% | Warm-up |
| Assignment 1 (A1) | 14% (+1% team formation) | Gaussian / GMM on MNIST digit 1 |
| Assignment 2 (A2) | 14% (+1%) | |
| Assignment 3 (A3) | 14% (+1%) | |
| Midterm | 10% | In class, on paper, **in teams of two** |
| Final exam | 40% | In person, on paper, **individual** |

The assignment weights plus the exams add up to $5+14+14+14+10+40 = 97\%$; the three "+1%" team-formation bonuses make up the remaining $3\%$, giving $100\%$.

**Check:** $5 + 14 + 14 + 14 = 47$ for A0–A3, plus $10 + 40 = 50$ for exams gives $97$; plus $3 \times 1 = 3$ gives $100$. ✔

### A.3 Deadlines (tentative)

| Item | Due |
|---|---|
| A0 | Thu Sep 17 |
| A1 | Thu Oct 1 |
| A2 | Thu Nov 5 |
| Midterm | Mon Nov 16 (in class) |
| A3 | Thu Dec 3 |

All assignments are due at **4 pm CST**. (Weekdays computed from the calendar and verified.)

### A.4 Rules you must not break

* **Exams:** no electronic or wearable devices except a **simple calculator**. Questions are multiple choice or long answer. **All assessments must be completed to pass the course.**
* **GenAI policy:** generative AI may be used **only to improve the clarity, wording or structure of your writing**. Any other use is academic misconduct.
* **Reports:** use the provided LaTeX template **unmodified**, and also submit separately on Canvas.

### A.5 Weekly schedule

| Week | Topics |
|---|---|
| 1 | Intro, Git, tensors |
| 2 | Probability distributions, linear algebra |
| 3 | Expected values, Monte Carlo |
| 4 | Linear regression, bias–variance |
| 5 | Logistic and softmax regression |
| 6 | Regularization, MAP |
| 7 | Neural networks, basis functions, feature selection |
| 8 | Sequence models, LLMs |
| 9 | Gradient descent |
| 10 | Review (midterm Nov 16) |
| 11 | Constrained optimization, SVMs |
| 12 | Dimensionality reduction, PCA |
| 13 | Review |

The uploaded lectures (02–19) cover weeks 1–7 of this list: tensors, probability, MLE, Git, linear algebra, linear/logistic/softmax regression, generalization, experiment reporting, statistics, **Bayes' law and MAP, ridge and lasso regularization, and the multilayer perceptron** (Lectures 13–15; Lecture 15 covers the first part of week 7). Later topics (feature selection details, sequence models/LLMs, gradient descent in depth, constrained optimization/SVMs, PCA…) will be added as lectures arrive; where the textbooks already touch them, a **➕ Textbook extra** mentions them.

---

## Part B. The big picture

Machine learning in one sentence: **choose a model with adjustable numbers (parameters), define a score for how badly it fits data, and adjust the parameters to make that score as good as possible — then check honestly that it works on data it has never seen.**

Every lecture is one piece of that sentence:

| Piece of the sentence | Lectures |
|---|---|
| **The tools** to hold and move data (arrays, tensors, vectors, matrices) | 02 Tensors, 06 Linear algebra |
| **The language** of uncertainty (probability, distributions, expectation) | 03 Probability |
| **The principle** for scoring a model: *how likely is the data under these parameters?* | 04 Maximum likelihood |
| **The workflow** for keeping your code and results safe | 05 Git |
| **The models**: predict a number | 07 Linear regression |
| **The models**: predict yes/no | 08 Logistic regression |
| **The models**: predict one of many classes | 09 Softmax regression |
| **The honesty check**: does it work on new data? | 10 Generalization |
| **The prior belief**: which parameters were plausible *before* seeing data? | 13 Bayes' law and MAP |
| **The brake on overfitting**: penalize big weights (ridge) or unneeded features (lasso) | 14 Lasso and ridge |
| **The learned features**: let the model discover its own basis functions | 15 Multilayer perceptron |
| **The reporting**: how sure are we about a measured number? | 11 Experiment reporting, 12 Statistics |

### The chain of ideas

```text
 data (tensors, 02/06)  →  random variable + distribution (03)
        ↓                              ↓
   model with parameters θ   →   likelihood p(data | θ)  (04)
        ↓                              ↓
  Gaussian noise  → linear regression  (07)   [MLE = least squares]
  Bernoulli label → logistic regression (08)  [MLE = cross-entropy]
  Categorical     → softmax regression  (09)  [MLE = cross-entropy]
        ↓
  fit by solving equations or by gradient descent
        ↓
  does it generalize? bias–variance, validation, cross-validation (10)
        ↓
  how sure are we of the measured score? CLT, SEM, error bars (11, 12)
        ↓
  add a prior p(θ): MAP = likelihood × prior (13)
  Gaussian prior → ridge (L2);  Laplace prior → lasso (L1, zeros)   (14)
        ↓
  learn the basis functions too: MLP = g(W h(V x)), backprop,
  and regularize it (weight decay, dropout, early stopping)        (15)
        ↓
  sequences: Markov chains, hidden state, RNN, attention, LLM = MLE  (16, 17)
        ↓
  minimize the loss: gradient descent, SGD, momentum, autodiff      (18)
        ↓
  midterm: the whole recipe on a new problem (Normal with mean AND variance) (19)
```

The **single most important idea** to carry through the course: *a loss function is a negative log-likelihood.* Squared error is what you get from a Gaussian assumption; binary cross-entropy is what you get from a Bernoulli assumption; softmax cross-entropy comes from a categorical assumption. Once you see that, three "different" models are one recipe: **write down $p(\text{data}\mid\theta)$, take $-\log$, minimize.**

---

## Part C. Lecture → textbook map

Page numbers are printed page numbers. "§" is the section number.

| Lecture | Topic | Murphy (2022) | Bishop (2006) |
|---|---|---|---|
| 02 | Tensor arithmetic (NumPy) | §7.2 Matrix multiplication p.242 (for the shape/axis ideas) | — |
| 03 | Probability | §2.2 Random variables p.35; §2.2.4 Independence p.39; §2.2.5 Moments p.40; §2.3 Bayes' rule p.44; §2.4 Bernoulli p.49; §2.5 Categorical p.53; §2.6 Gaussian p.57; §2.8.7 Monte Carlo p.73; exercises p.73 | §1.2 Probability theory p.12; §1.2.4 Gaussian p.24 |
| 04 | Maximum likelihood | §4.2 MLE p.107; §4.2.3 Bernoulli MLE p.110; §4.2.5 Gaussian MLE p.111; §4.2.6 MVN MLE p.112 | §1.2.4 p.24; §1.2.5 Curve fitting revisited p.28 |
| 05 | Git / GitLab | — (not a textbook topic) | — |
| 06 | Linear algebra | §7.1.3 Norms p.234; §7.2 Matrix multiplication p.242; §7.3 Inverse p.249; §7.5.3 Pseudo-inverse p.261; §7.6.3 Cholesky p.266; §7.7.3 Least squares p.268; §7.8 Matrix calculus p.269; exercises p.274 | — |
| 07 | Linear regression | §4.2.7 Linear regression MLE p.114; §7.7.3 p.268; §8.4.2 SGD for linear regression p.293; §11.2 Least squares p.371; §11.2.2.2 Geometric interpretation p.373; §11.2.4 R² p.380; §11.3 Ridge p.381; exercises p.411 | §3.1 Linear basis function models p.138; §3.1.1 ML and least squares p.140; §3.1.2 Geometry p.143 |
| 08 | Logistic regression | §2.4.2 Sigmoid p.50; §2.4.3 Binary logistic regression p.52; §5.1.2 Classification p.169; §5.1.3 ROC p.171; §5.1.4 PR curves p.174; §8.2 First-order methods p.282; §8.4 SGD p.292; §10.2 Binary logistic regression p.339; §10.2.3 MLE p.342; §10.2.4 SGD p.345; §10.2.6 IRLS p.346; §10.2.8 Standardization p.350 | §4.3 Probabilistic discriminative models p.203; §4.3.2 Logistic regression p.205; §4.3.3 IRLS p.207 |
| 09 | Softmax regression | §2.5 Categorical p.53; §2.5.2 Softmax p.54; §2.5.4 Log-sum-exp p.56; §10.3 Multinomial logistic regression p.350; exercises p.367 | §4.3.4 Multiclass logistic regression p.209 |
| 10 | Generalization | §1.2.3 Overfitting p.12; §1.2.4 No free lunch p.13; §4.3 ERM p.115; §4.5 Regularization p.120; §4.5.4 Validation set p.124; §4.5.5 Cross-validation p.125; §4.7.6 Bias–variance p.159; §5.1 Decision theory p.169 | §1.1 Curve fitting p.4; §1.3 Model selection p.32; §3.2 Bias–variance p.147 |
| 11 | Experiment reporting | §2.8.6 CLT p.72; §4.7 Frequentist statistics p.154; §5.4.3 Cross-validation p.196; §5.5 Hypothesis testing p.198 | — |
| 12 | Statistics | §4.7 p.154; §6.1 Entropy p.207; §6.1.2 Cross entropy p.209; §6.1.3 Joint entropy p.209; §6.1.4 Conditional entropy p.210; §6.3 Mutual information p.217; exercises p.226 | §1.6 Information theory p.48; §1.6.1 KL / mutual information p.55 |
| 13 | Bayes' law and MAP | §2.3 Bayes' rule p.44; §2.7.4 Beta distribution p.63; §4.5 Regularization (MAP) p.120; §4.5.1 MAP for Bernoulli p.121; §4.6.2 Beta–binomial p.130 | §1.1 Curve fitting p.4; §1.2.3 Bayesian probabilities p.21; §2.1.1 Beta distribution p.71 |
| 14 | Lasso and ridge regression | §2.7.3 Laplace p.63; §4.5.3 Weight decay p.123; §11.3 Ridge p.381; §11.4 Lasso p.385; §11.4.4 Regularization path p.389; §11.4.5 Comparison p.390; §13.4.5 Initialization p.452 | §3.1.4 Regularized least squares p.144; §3.3 Bayesian linear regression p.152 |
| 15 | Multilayer perceptron | §13.2 MLPs p.426 (XOR p.426, activations p.428, depth p.434); §13.3 Backprop p.438; §13.4 Training p.446 (vanishing gradients p.447, initialization p.452); §13.5 Regularization p.455 (dropout p.455) | §4.1.7 Perceptron p.192; §5.1 Network functions p.227; §5.2 Training p.232; §5.3 Backprop p.241; §5.5 Regularization p.256 |
| 16 | Hidden-state models | §3.6.1.2 p.101; §15.2 Sequence models p.503–505 | Bishop §13.1–13.2 Markov and hidden Markov models |
| 17 | Sequence models (RNN, attention, LLM) | §15.2–15.7 p.503–545 | Bishop ch.13 (background) |
| 18 | Gradient descent | §8.1.3 convexity p.277; §8.2 first-order p.282–289; §8.2.4 momentum p.287; §8.4 SGD p.292–302 | §5.2.4 p.240 |
| 19 | Midterm example and sheet | §13.2.4.4 heteroskedastic MLP p.433; §2.2 distributions | p.273 heteroscedastic |

---


## Lecture 02 — Tensor Arithmetic

**Why this lecture exists.** Machine learning does millions of small calculations. A Python `for` loop does them one at a time and is slow. **NumPy** does the same work in fast compiled code, on *whole arrays at once*. Learning to think in whole-array operations ("tensor arithmetic") is the first skill of the course — Assignment 1 literally asks for code with no `for` loops and no `if/else`.

### 2.1 What is a tensor?

**Plain words.** A tensor is a box of numbers arranged along some number of directions. A single number is a 0-direction box, a list is 1-direction, a table is 2-direction, a stack of tables is 3-direction.

**Precisely.** A tensor is an array of numbers with $k$ indices. The number of indices $k$ is its **rank** (NumPy calls it `ndim`). The **shape** is the tuple of sizes along each index.

| Rank | Name | Shape example | In our running example |
|---|---|---|---|
| 0 | scalar | `()` | the mean hours, `4.1` |
| 1 | vector | `(10,)` | all ten `hours` |
| 2 | matrix | `(10, 2)` | one row per student: [1, hours] |
| 3 | 3-D tensor | `(6700, 28, 28)` | (Assignment 1) a stack of MNIST images |

NumPy's class for tensors is `np.ndarray`. **Habit #1: print `.shape` all the time.** Most bugs in this course are shape bugs.

**Worked example.** `h = np.array([4,2,6,1,3,5,7,3,8,2.])` has `h.shape == (10,)` and `h.ndim == 1`. The matrix `X = np.stack([np.ones(10), h], axis=1)` has shape `(10, 2)`: ten rows (students), two columns (a constant 1 and the hours).

### 2.2 `dtype`: what kind of number?

Every array stores one kind of number, its **dtype**: `float64` (decimals), `int64` (whole numbers), `bool` (True/False). Choose it deliberately — e.g. Assignment 1 requires an integer type for cluster labels. A wrong dtype gives *silent* bugs: integer division, truncated values, a "mask" that is actually floats.

```python
np.array([1, 2, 3]).dtype            # int64
np.array([1, 2, 3.0]).dtype          # float64   (one float promotes the whole array)
(h >= 5).dtype                       # bool
(h >= 5).astype(np.int32).dtype      # int32     (explicit cast)
```

### 2.3 Element-wise operations

**Plain words.** `+ - * / **` between arrays (or an array and a number) act on **every element independently** and return an array of the same shape. `@` is *not* element-wise (see 2.9).

| Symbol | Meaning |
|---|---|
| `*` | element-wise product |
| `**` | element-wise power |
| `@` | matrix multiplication |

**Worked example.** `h * 2` = `[8, 4, 12, 2, 6, 10, 14, 6, 16, 4]`. `h ** 2` = `[16, 4, 36, 1, 9, 25, 49, 9, 64, 4]`.

### 2.4 Masking (comparisons replace `if`)

**Plain words.** A comparison such as `h >= 5` is also element-wise and produces an array of `True/False` — a **mask**. A mask replaces a whole `for … if …` loop.

**Worked example (running example).** Which students studied at least 5 hours?

```python
m = h >= 5
# [False False  True False False  True  True False  True False]
m.sum()          # 4         (True counts as 1)  -> 4 students
h[m]             # [6. 5. 7. 8.]                  -> their hours
h[m].mean()      # 6.5
```

The sum of a boolean mask is a **count**; the sum of mask × values is a **conditional total**. This is exactly what Assignment 1 does with `self.x[assign == j]`: pick all images assigned to cluster $j$.

*Detail from the lecture notebook.* A mask has dtype `bool`. `(x >= 2).astype(np.int32).sum()` casts to 0/1 first and then sums; `(x >= 2).sum()` does that cast **implicitly**. Both work, but the explicit form is easier to read and safer. For $10^8$ items a Python `for/if` loop takes ages while the NumPy line finishes almost instantly — that is the point of the lecture.

### 2.5 Reduce operations and `axis`

**Plain words.** A **reduce** squashes an array along one direction into fewer numbers: `sum`, `mean`, `max`, `min`, `argmax`, `argmin`. The `axis` argument says *which direction disappears*.

> **Memory trick:** `axis=k` means "dimension $k$ disappears from the shape."

**Worked example.** For $X$ of shape `(10, 2)`:

| Call | Result shape | Value |
|---|---|---|
| `X.mean(axis=0)` | `(2,)` | `[1.0, 4.1]` — the average of each column |
| `X.sum(axis=1)` | `(10,)` | `[5,3,7,2,4,6,8,4,9,3]` — row totals (1 + hours) |
| `X.mean()` | `()` | 2.55 — average of all 20 entries |

`argmax` returns the *position* of the maximum, not the value: `s.argmax()` is `8` (student 9, the best score, 83), `s.argmin()` is `3` (student 4, score 45). Positions start at 0.

### 2.6 `reshape`

**Plain words.** Re-arrange the *same* numbers into a different shape. The total count of numbers must stay the same. `-1` in one slot means "work this one out for me."

```python
h.reshape(2, 5)
# [[4. 2. 6. 1. 3.]
#  [5. 7. 3. 8. 2.]]
h.reshape(-1, 2).shape    # (5, 2)
```

For Assignment 1: `images.reshape(-1, 784)` turns `(N, 28, 28)` into `(N, 784)` — each image flattened into one row of 784 pixels.

### 2.7 Broadcasting

**Plain words.** When two arrays have different shapes, NumPy *virtually stretches* any dimension of size 1 (or a missing leading dimension) to match the other — **without copying data**. It lets you subtract a mean from every row, or build a table of all pairs, in one line.

**The rule.** Line the shapes up from the **right**. For each pair of dimensions they are compatible if they are **equal** or **one of them is 1**. The result's dimension is the larger one. If any pair fails, you get an error.

| Shape A | Shape B | Result | Meaning |
|---|---|---|---|
| (5, 1) | (1, 7) | (5, 7) ✓ | every combination (the lecture's example: `arange(5).reshape(-1,1) * ones(7).reshape(1,-1)`) |
| (10,) | () | (10,) ✓ | `h - h.mean()` |
| (6700, 784) | (784,) | (6700, 784) ✓ | subtract the mean image from every image |
| (6700, 784) | (6700,) | ✗ error | 784 ≠ 6700 — reshape to `(6700, 1)` first |
| (10, 1) | (1, 10) | (10, 10) ✓ | all pairs of students |

**Worked example 1 — centering.** `h - h.mean()` where `h.mean() = 4.1`:
`[-0.1, -2.1, 1.9, -3.1, -1.1, 0.9, 2.9, -1.1, 3.9, -2.1]`. These values always sum to 0.

**Worked example 2 — all pairs.** `D = h[:, None] - h[None, :]` has shape `(10, 10)` with `D[i, j] = h[i] - h[j]`. Row 0 (student 1, 4 hours) is `[0, 2, -2, 3, 1, -1, -3, 1, -4, 2]`. `h[:, None]` is a trick to insert a size-1 dimension (same as `reshape(-1, 1)`).

### 2.8 Slicing

**Plain words.** `a[start:stop:step]` picks a regular run of elements. `start` is included, `stop` is excluded, indexing starts at 0, negative numbers count from the end.

```python
ar = np.arange(5)     # [0 1 2 3 4]
ar[1:-1]              # [1 2 3]      drop first and last
ar[:-1]               # [0 1 2 3]
ar[::-1]              # [4 3 2 1 0]  reverse
ar[::-2]              # [4 2 0]
ar2d[0, 0]            # 2-D: row 0, column 0
```

### 2.9 Matrix multiplication with `@`

`A @ B` has entries $c_{ij} = \sum_\ell a_{i\ell}\, b_{\ell j}$ — a sum of products, which is why one `@` can replace a double loop. The inner dimensions must match: `(n, m) @ (m, k)` → `(n, k)`.

**Worked example.** With $X$ of shape `(10, 2)` and parameters $\theta=[41.8078, 5.0225]^\top$ (they appear in Lecture 07), `X @ theta` has shape `(10,)` and gives all ten predicted scores at once.

### 2.10 Views vs copies (the sneaky bug)

**Plain words.** A **slice** (and `reshape`) does **not** copy; it is a *window onto the same memory*. Changing the window changes the original. Use `.copy()` when you want independence.

```python
a = h.copy()
v = a[1:-1]          # a view onto a[1..8]
v += 100
print(a)             # [4. 102. 106. 101. 103. 105. 107. 103. 108. 2.]   <- 'a' changed!
```

The lecture's notebook shows the same with `reshape`: changing `y[0]` changes `y_1` because `y_1` is only a new view of `y`.

### 2.11 Fancy indexing (index lists) — copies

```python
h[[2, 4, 1]]         # [6. 3. 2.]   students 3, 5, 2
```

Indexing with a **list of positions** (or a boolean mask) builds a **copy**, so editing the result leaves the original untouched:

```python
b = h.copy(); f = b[[1, 2]]; f += 100
b[:4]                # [4. 2. 6. 1.]    unchanged
```

It is also **not** a fast operation (NumPy iterates internally), so use it sparingly. Summary: *slice → view; list or mask → copy.*

### 2.12 Mini-checklist for writing NumPy

1. Know the shape of every array (print it).
2. Replace loops with element-wise ops, masks, reduces, broadcasting, `@`.
3. Set dtypes explicitly.
4. Never modify a slice unless you intend to modify the original.

#### 📖 Textbook deep dive

Murphy treats arrays as vectors and matrices rather than "tensors" and never teaches NumPy, so the useful overlap is the *meaning* of the operations.

##### Murphy §7.2 (p.242) — four ways to see $C = AB$

The slide formula $c_{ij}=\sum_\ell a_{i\ell}b_{\ell j}$ is the "entry" view. Murphy also shows:

* **Row view:** each row of $C$ is a row of $A$ times $B$ — $c_{i\cdot} = a_{i\cdot}B$.
* **Column view:** each column of $C$ is $A$ times a column of $B$ — $c_{\cdot j} = A\,b_{\cdot j}$.
* **Sum of outer products:** $C=\sum_\ell a_{\cdot \ell}\, b_{\ell\cdot}$ (column $\ell$ of $A$ times row $\ell$ of $B$).

*Worked check.* $A=\begin{bmatrix}1&2\\3&4\end{bmatrix}$, $B=\begin{bmatrix}5&6\\7&8\end{bmatrix}$. Entry $c_{11}=1\cdot5+2\cdot7=19$. Column view: column 1 of $B$ is $(5,7)$, and $A(5,7)^\top=(1\cdot5+2\cdot7,\,3\cdot5+4\cdot7)=(19,43)$ — the first column of $C$. ✔ The code view is the same: `A @ B[:, 0]`.

##### Murphy §7.1 — vectors vs arrays
Murphy writes data as an $N\times D$ **design matrix**: one row per example, one column per feature. This is exactly the `(N, D)` convention you use in NumPy (and in Assignment 1: `(6700, 784)`).

#### ➕ Textbook extra — Memory layout and cost
Murphy does not cover views vs copies; this is a pure NumPy/software topic. The tip to remember: a view is $O(1)$ (no data copied), a copy is $O(n)$ in time and memory.

#### ✅ Key takeaways

* A **tensor** = array with a **shape**; print shapes constantly.
* **Masks** (`x >= 2`) replace `if`; `mask.sum()` counts; `x[mask]` selects.
* `axis=k` in a reduce means "dimension $k$ disappears."
* **Broadcasting:** align shapes from the right; dimensions must be equal or 1.
* **Slice = view; list/mask indexing = copy.** Use `.copy()` to be safe.
* `*` is element-wise, `@` is matrix multiplication.

---


## Lecture 03 — Probability

**Why this lecture exists.** Data is noisy and models are never perfect. Probability is the language for saying *how uncertain* something is. Every model in this course (regression, classification, GMM) is a **probability distribution with adjustable parameters**.

### 3.1 Random events and Schrödinger's cat

**Plain words.** A random event is something whose outcome you cannot predict in advance — a coin flip, a die roll, whether it rains. We cannot predict one outcome, but we *can* talk about how often each outcome happens over many repeats.

**The lecture's story (Schrödinger's cat).** A cat sits in a box with a poison hammer that a Geiger counter triggers at a *random time*. We cannot predict *when*, but we can describe how likely each trigger time is (the slide draws a skewed bump of "frequency of Geiger counter triggering" against time in seconds).

**Running example.** "Will a randomly chosen student from our ten pass?" We cannot predict one student, but 4 of 10 passed, so we say the pass frequency is $0.4$.

### 3.2 The event space $\Omega$ and σ-algebras

**Plain words.** Before talking about probability, list everything that *could* happen. That list is the **event space** $\Omega$ (omega). One possible outcome is written $\omega\in\Omega$ ("omega is an element of Omega").

| Experiment | $\Omega$ |
|---|---|
| Coin flip | {heads, tails} |
| Die roll | {1,2,3,4,5,6} |
| A student's score | any number from 0 to 100 |
| A student's result (running example) | {fail, pass} |

**The formal part (from the slides).** We do not ask about the probability of *every* weird subset; we only ask about a chosen collection $\Sigma$ of subsets of $\Omega$ called a **σ-algebra** ("sigma-algebra") if

1. $\Omega\in\Sigma$ (the whole set is allowed),
2. if $A\in\Sigma$ then the complement $\Omega\setminus A\in\Sigma$ (if you can ask about "A", you can ask about "not A"),
3. if $A_1,A_2,\dots\in\Sigma$ then their union $A_1\cup A_2\cup\cdots\in\Sigma$.

*Worked example.* For a die, $\Sigma$ = all subsets of $\{1,\dots,6\}$ (including $\emptyset$ and the whole set). Check: $\{6\}\in\Sigma\Rightarrow\{1,2,3,4,5\}\in\Sigma$; and $\{1\},\{2\},\{3\}\in\Sigma\Rightarrow\{1,2,3\}\in\Sigma$. You will not be asked to construct one in the assignment.

A **probability measure** is a function $p:\Sigma\to[0,1]$ with $p(\emptyset)=0$, $p(\Omega)=1$, and $p(A_1\cup A_2\cup\cdots)=p(A_1)+p(A_2)+\cdots$ whenever the $A_i$ don't overlap. A **random variable** is a function from $\Omega$ to another set (also with a σ-algebra); for a coin it maps {heads, tails} → {1, 0}.

### 3.3 Probability rules

* $0\le p\le 1$: 0 means impossible, 1 means certain.
* Something in $\Omega$ happens: $p(\Omega)=1$.
* Events that **cannot happen together** (mutually exclusive) add: $p(A\cup B)=p(A)+p(B)$.

*Worked example.* Fair die, each face $1/6$:
$$p(\text{even})=p(2)+p(4)+p(6)=\tfrac16+\tfrac16+\tfrac16=\tfrac12.$$

*Running example.* Using the 3-class labels, $p(\text{class }0)=6/10$, $p(\text{class }1)=2/10$, $p(\text{class }2)=2/10$, and $p(\text{class }1\text{ or }2)=0.2+0.2=0.4=$ pass rate. ✔

### 3.4 Random variables

**Plain words.** Computers need numbers, not words. A **random variable** (rv) is a rule that turns each outcome into a number. By convention **capital** $X$ is the rule/variable and **lowercase** $x$ is a particular value it takes. "$X=1$" means "the coin came up heads".

*Running example.* $T$ = 1 if the student passed, 0 if not. Then $T=1$ for students 1, 3, 7, 9.

### 3.5 Distributions

A **distribution** says how likely each value is. For a discrete rv the list of probabilities is called the **pmf** (probability mass function), $p(x)=\Pr(X=x)$, with $0\le p(x)\le1$ and $\sum_x p(x)=1$. For a continuous rv the function is a **density** (pdf).

#### 3.5.1 Bernoulli — one yes/no event

**Plain words.** Two outcomes (1 or 0). One parameter $\theta$ ("theta") = the probability of a 1.

| $x$ | probability |
|---|---|
| 1 | $\theta$ |
| 0 | $1-\theta$ |

**One-line formula (the lecture's trick).**
$$\text{bernoulli}(x;\theta)=\theta^{x}(1-\theta)^{1-x}.$$
Check: $x=1\Rightarrow\theta^1(1-\theta)^0=\theta$; $x=0\Rightarrow\theta^0(1-\theta)^1=1-\theta$. (Anything to the power 0 is 1.)

*Worked example.* Pass/fail with $\theta=0.4$: $p(T=1)=0.4$, $p(T=0)=0.6$.

#### 3.5.2 Categorical — one of $K$ outcomes

Like Bernoulli but with $K$ outcomes; the **list of $K$ probabilities that sums to 1 is the model**. A fair die is [1/6,…,1/6]. Our 3-class result has $\boldsymbol\theta=[0.6,\,0.2,\,0.2]$.

#### 3.5.3 Normal (Gaussian) — the bell curve

**Plain words.** For measurements like height or score. Most values sit near the centre; far-away values are rare. Two parameters: $\mu$ (mean, the centre) and $\sigma$ (standard deviation, the width); $\sigma^2$ is the **variance**. Written $\mathcal N(\mu,\sigma^2)$. The **standard normal** is $\mathcal N(0,1)$.

**Density formula.**
$$f(x;\mu,\sigma)=\frac{1}{\sqrt{2\pi\sigma^2}}\exp\!\Big(-\frac{(x-\mu)^2}{2\sigma^2}\Big).$$

How to read it: $(x-\mu)^2$ is the squared distance from the centre; $\exp(-\text{something})$ shrinks as "something" grows; so density falls as $x$ moves away from $\mu$. The factor $1/\sqrt{2\pi\sigma^2}$ only makes the total area 1.

*Worked example.* Heights $\mathcal N(170,10^2)$:
$f(170)=1/(10\sqrt{2\pi})=0.03989$; $f(190)=0.03989\cdot e^{-2}=0.005399$ (two standard deviations away, 7.4 times lower). The probability of landing within $\pm1\sigma$ (160–180) is $0.6827$; a height above 210 (4σ) has probability $3.2\times10^{-5}$.

**Probability vs density.** For a continuous variable the probability of *exactly* 170.0000 is zero. Probability comes from **area**: $\Pr(X\in[a,b])=\int_a^b f(x)\,dx$ (the lecture notes it is a Lebesgue integral, beyond the course). Higher density = more likely neighbourhood.

### 3.6 Independence and IID

Two events are **independent** if one does not affect the other; for independent events probabilities **multiply**: $p(A\text{ and }B)=p(A)\,p(B)$.

**IID = Independent and Identically Distributed**: every data point (i) is independent of the others and (ii) comes from the same distribution. This is assumed for essentially every model in the course because it lets us **multiply** per-sample probabilities (Lecture 04).

*Worked example.* Biased coin $\theta=0.7$: $p(\text{H,H,T})=0.7\cdot0.7\cdot0.3=0.147$.
*Running example.* If each student passes independently with $\theta=0.4$, the probability of exactly the observed sequence $1,0,1,0,0,0,1,0,1,0$ is $0.4^4\cdot0.6^6=0.0011944$. (The probability of *some* sequence with exactly 4 passes is $\binom{10}{4}\cdot0.0011944=0.2508$.)

### 3.7 Models versus reality

The real world (the Geiger counter and the radiation) is the true random process on $\Omega$. We only see **measurements** (cat alive / dead), which is a random variable. We *choose* a **probabilistic model** — a distribution or density — to describe them. Two consequences stated in the lecture:

* every probabilistic model rests on **modelling assumptions** and can never be certain to match the true process;
* a model is designed by an expert and then **validated and tested on data**; a **mismatch** between model and reality is always possible. This is why Lecture 10 (generalization) exists.

#### 📖 Textbook deep dive

##### Murphy §2.2.1–2.2.2 (p.35–38) — pmf, cdf, pdf, quantiles

* **cdf:** $P(x)=\Pr(X\le x)$ (capital $P$). Interval probability: $\Pr(a<X\le b)=P(b)-P(a)$. The cdf never decreases.
* **pdf:** the derivative of the cdf, $p(x)=\frac{d}{dx}P(x)$. Intuition: $\Pr(x<X\le x+dx)\approx p(x)\,dx$ — *density × width*.
* **Quantiles** invert the cdf: $P^{-1}(q)=x_q$ satisfies $\Pr(X\le x_q)=q$. $P^{-1}(0.5)$ is the **median**; 0.25 and 0.75 give the **quartiles**. For the standard normal, $(\Phi^{-1}(0.025),\Phi^{-1}(0.975))=(-1.96,\,1.96)$, so the central 95% of $\mathcal N(\mu,\sigma^2)$ is $\mu\pm1.96\sigma$, often rounded to $\mu\pm2\sigma$. (Verified: $\Phi^{-1}(0.975)=1.95996$.) **This 1.96 returns in Lectures 11–12.**
* A pdf **can exceed 1** — only its *area* must be 1. E.g. $\mathcal N(\mu,0.1^2)$ has peak height $1/(0.1\sqrt{2\pi})=3.989$.

##### Murphy §2.2.3 (p.38–39) — joint, marginal, conditional, chain rule

Joint tables sum to one. Murphy's example: $p(X,Y)$ = [[0.2, 0.3],[0.3, 0.2]]. **Marginal** (sum rule): $p(X=0)=0.2+0.3=0.5$. **Conditional:** $p(y|x)=p(x,y)/p(x)$, rearranged to the **product rule** $p(x,y)=p(x)p(y|x)$, extended to the **chain rule** $p(x_{1:D})=p(x_1)p(x_2|x_1)\cdots p(x_D|x_{1:D-1})$.

*Why independence matters (Fig. 2.3, p.39):* a general joint over a 6-state and a 5-state variable needs $6\cdot5-1=29$ parameters; if independent it needs only $(6-1)+(5-1)=9$. Independence assumptions make models *small* — the same reason IID makes likelihoods easy.

*Running example.* Split hours into low ($\le4$) and high ($\ge5$). The joint of (hours, result) is

| | fail | pass |
|---|---|---|
| low | 0.5 | 0.1 |
| high | 0.1 | 0.3 |

Marginals: $p(\text{low})=0.6$, $p(\text{pass})=0.4$. If hours and result were independent the (low, pass) cell would be $0.6\cdot0.4=0.24$, but it is $0.1$ — so they are **dependent**: studying more goes with passing. (Lecture 12 puts a number on this dependence: mutual information 0.2564 bits.)

##### Murphy §2.2.4 (p.39) — independence and conditional independence

$X\perp Y\iff p(X,Y)=p(X)p(Y)$. *Mutual* independence of $X_1,\dots,X_n$ requires the factorization for **every subset**. $X\perp Y\mid Z\iff p(X,Y|Z)=p(X|Z)p(Y|Z)$. Murphy's remark: unconditional independence is rare, but variables are often independent *given* the right intermediate variable.

##### Murphy §2.2.5 (p.40) — moments

* Mean: $\mathbb E[X]=\sum_x x\,p(x)$ or $\int x\,p(x)\,dx$. **Linearity:** $\mathbb E[aX+b]=a\mathbb E[X]+b$ and $\mathbb E[\sum X_i]=\sum\mathbb E[X_i]$ (always); $\mathbb E[\prod X_i]=\prod\mathbb E[X_i]$ only if independent.
* Variance: $\mathbb V[X]=\mathbb E[(X-\mu)^2]=\mathbb E[X^2]-\mu^2$.

*Check on the students' hours:* $\mathbb E[h]=4.1$; $\mathbb E[h^2]=(16+4+36+1+9+25+49+9+64+4)/10=21.7$; $21.7-4.1^2=4.89$ ✔ (matches the direct variance 4.89).

##### Murphy §2.3 (p.44–48) — Bayes' rule

$$p(H=h\mid Y=y)=\frac{p(h)\,p(y\mid h)}{p(y)},\qquad p(y)=\sum_{h'}p(h')p(y\mid h').$$
Vocabulary: **prior** $p(h)$ (belief before data), **likelihood** $p(y|h)$ as a function of $h$ (it does *not* sum to one over $h$), **marginal likelihood** $p(y)$ (the normalizer), **posterior** $p(h|y)$. In words: **posterior ∝ prior × likelihood**.

*Murphy's COVID test example (p.46).* Sensitivity $p(Y{=}1|H{=}1)=0.875$, specificity $p(Y{=}0|H{=}0)=0.975$, prior prevalence 0.1. Then $p(Y{=}1)=0.875\cdot0.1+0.025\cdot0.9=0.11$ and the posterior is
$$p(H{=}1|Y{=}1)=\frac{0.0875}{0.11}=0.795.$$
A positive test of a good test lifts a 10% prior to about 80%, not to 99%. With a rarer disease it would be much lower (Exercise 2.9 below).

*Running example.* Prior $p(\text{pass})=0.4$. Observe a student who studied $\ge5$ hours. $p(\text{high}|\text{pass})=3/4$, $p(\text{high})=4/10$. Posterior $=0.75\cdot0.4/0.4=0.75$ ✔ — consistent with the table (0.3/0.4).

##### Murphy §2.4–2.5 (p.49–56) — the two classification distributions

* **Bernoulli** as a conditional model uses the **sigmoid** $\sigma(a)=1/(1+e^{-a})$ to squash a real score $a$ into $(0,1)$. Identities (p.50): $\sigma'(a)=\sigma(a)(1-\sigma(a))$, $1-\sigma(a)=\sigma(-a)$, and the inverse is the **logit** $\log\frac{p}{1-p}$. (Lecture 08.)
* **Categorical** with the **softmax**; includes the *temperature* form $\text{softmax}(\mathbf a/T)$ and the **log-sum-exp** trick (p.56). (Lecture 09.)

##### Murphy §2.6 (p.57–60) — Gaussian

The pdf matches the slide. Murphy stresses the Gaussian is popular because (i) it has two interpretable parameters, (ii) the central limit theorem justifies it for sums, (iii) it assumes the least structure for a given mean and variance (maximum entropy), and (iv) sums of independent Gaussians are Gaussian: $\mathcal N(\mu_1,\sigma_1^2)\otimes\mathcal N(\mu_2,\sigma_2^2)=\mathcal N(\mu_1+\mu_2,\sigma_1^2+\sigma_2^2)$ (p.72, Exercise 2.4). Linear regression with Gaussian noise is §2.6.3 (p.59) — see Lecture 07.

##### Murphy §2.8.6 (p.72) — Central limit theorem (preview of Lectures 11–12)

For iid $X_n$ with mean $\mu$ and variance $\sigma^2$, the standardized sum
$$Z_N=\frac{S_N-N\mu}{\sigma\sqrt N}=\frac{\bar X-\mu}{\sigma/\sqrt N}\ \longrightarrow\ \mathcal N(0,1).$$
Murphy's Fig. 2.23 shows means of Beta(1,5) draws becoming bell-shaped already at $N=5$.

##### Murphy §2.8.7 (p.73) — Monte Carlo approximation

If $y=f(x)$ and $p(y)$ is hard to compute analytically, **draw many samples of $x$, apply $f$, and use the histogram of the results.** Murphy's example: $x\sim\mathrm{Unif}(-1,1)$, $y=x^2$. *Check:* the mean of 100,000 such $y$ values was $0.333$ (exact value $1/3$) — Monte Carlo also estimates expectations: $\mathbb E[f(x)]\approx\frac1N\sum_s f(x_s)$.

```python
rng = np.random.default_rng(0)
x = rng.uniform(-1, 1, 100_000)
(x**2).mean()        # ≈ 0.333   (exact 1/3)
```

##### Solved exercises (Murphy §2.9, p.73–76)

**Ex 2.1 (conditional independence).** Bayes: $P(H|e_1,e_2)=P(e_1,e_2|H)P(H)/P(e_1,e_2)$. (a) With no assumption, sets (i) and (iii) lack the joint $P(e_1,e_2|H)$, so only **(ii)** suffices. (b) If $E_1\perp E_2\mid H$ then $P(e_1,e_2|H)=P(e_1|H)P(e_2|H)$, and now **(i), (ii) and (iii) all suffice** ((iii) works because the normalizer is the sum over $H$ of the numerator).

**Ex 2.2 (pairwise ≠ mutual independence).** Let $X,Y$ be independent fair bits and $Z=X\oplus Y$ (XOR). Any two of them are independent: e.g. $\Pr(X{=}1,Z{=}1)=\Pr(X{=}1,Y{=}0)=\tfrac14=\tfrac12\cdot\tfrac12$. But all three are *not* mutually independent: $\Pr(X{=}1,Y{=}1,Z{=}1)=0\ne\tfrac18$ (indeed $Z$ is determined by $X,Y$).

**Ex 2.5 (expected minimum of two uniforms).** $\Pr(\min>m)=(1-m)^2$ so $\mathbb E[\min]=\int_0^1(1-m)^2dm=\tfrac13$.

**Ex 2.6 (variance of a sum).** $\mathbb V[X+Y]=\mathbb E[(X+Y)^2]-(\mathbb E X+\mathbb E Y)^2=\mathbb V[X]+\mathbb V[Y]+2(\mathbb E[XY]-\mathbb E X\,\mathbb E Y)=\mathbb V[X]+\mathbb V[Y]+2\,\mathrm{Cov}[X,Y]$. For independent variables the covariance is 0 — the fact behind the SEM $\sigma/\sqrt n$ in Lecture 11.

**Ex 2.9 (medical diagnosis).** $p(\text{disease}\mid +)=\dfrac{0.99\cdot10^{-4}}{0.99\cdot10^{-4}+0.01\cdot(1-10^{-4})}=\dfrac{0.000099}{0.010098}\approx0.0098$ — about **1%**, despite a "99% accurate" test, because the disease is so rare that false positives (1% of 9,999 healthy people ≈ 100) outnumber true positives (≈ 0.99 per 10,000).

**Ex 2.10 (legal reasoning).** (a) *Prosecutor's fallacy:* the 1% is $P(\text{evidence}\mid\text{innocent})$, but the claim "99% guilty" treats it as $P(\text{innocent}\mid\text{evidence})$. These are different quantities; converting needs the prior (Bayes). (b) *Defender's fallacy:* the 1-in-8000 ignores that the evidence **shrinks the pool of plausible suspects from 800,000 to about 8,000** — a 100-fold increase in the probability the defendant is guilty compared with a random citizen. So the evidence is highly relevant, just not conclusive alone.

**Ex 2.11 (two children).** (a) "Has at least one boy": outcomes BB, BG, GB equally likely → $P(\text{a girl})=2/3$. (b) "I *see* one child and it is a boy": the unseen child is independent of which child ran by → $P(\text{other is a girl})=1/2$. Lesson: the answer depends on **how the information was obtained**.

**Ex 2.12 (Gaussian normalization).** $Z^2=\int_0^{2\pi}\!d\theta\int_0^\infty r\,e^{-r^2/2\sigma^2}dr=2\pi\cdot\sigma^2$ (substitute $u=e^{-r^2/2\sigma^2}$, $\int_0^\infty r e^{-r^2/2\sigma^2}dr=\sigma^2$). So $Z=\sigma\sqrt{2\pi}$ — this is where the $1/\sqrt{2\pi\sigma^2}$ in the density comes from.

##### Bishop §1.2 (p.12–24) — same material, more intuition

Bishop develops the sum rule $p(X)=\sum_Yp(X,Y)$, product rule, Bayes, and expectation/variance with the "red/blue box, apple/orange" fruit example, and defines the Gaussian (§1.2.4, p.24) as the distribution that arises as sums of many things. Use it as a second explanation if Murphy's is too terse.

#### ➕ Textbook extra — cdf/quantiles, chain rule, Bayes' rule
The slides never state the **cdf**, **quantile**, **chain rule**, **Bayes' rule** or **linearity of expectation**. You need all of them later: quantiles for confidence intervals (Lecture 11), Bayes' rule for MAP (week 6), linearity of expectation for the unbiasedness proofs (Lecture 12).

#### ✅ Key takeaways

* $\Omega$ = all outcomes; a **random variable** maps outcomes to numbers; a **distribution** says how likely each value is.
* **Bernoulli** $\theta^x(1-\theta)^{1-x}$ for yes/no; **categorical** for $K$ classes; **normal** $\mathcal N(\mu,\sigma^2)$ for continuous values.
* Density ≠ probability; probabilities of continuous events are **areas**.
* **IID** ⇒ probabilities multiply. This is the engine of the likelihood.
* A model is an assumption: it can mismatch reality, so test on data.
* Bayes: posterior ∝ prior × likelihood; CLT: averages become bell-shaped.

---


## Lecture 04 — Maximum Likelihood Estimation (MLE)

**Why this lecture exists.** This is the single most important idea in the course. It answers: *given data and a model with adjustable parameters, which parameter values should I pick?* Answer: **the ones that make the data you actually saw as probable as possible.** Linear regression, logistic regression, softmax regression and the Gaussian/GMM of Assignment 1 are all fitted this way.

### 4.1 The question

We have a **dataset** $\mathcal D=\{x_1,\dots,x_D\}$ with $D$ data points. Our running example's pass/fail results:

$$\mathcal D=\{1,0,1,0,0,0,1,0,1,0\}\qquad(1=\text{passed},\;0=\text{failed}),\ D=10.$$

Count: $c_1=4$ passes, $c_0=6$ fails. The model is $\text{Bernoulli}(\theta)$. **Which $\theta$ best explains this data?** Your gut says $4/10=0.4$. MLE proves it — and, more importantly, shows a *method* that works when your gut cannot help.

### 4.2 The likelihood

**Plain words.** Pick a guess for $\theta$. Ask: "if the world really worked like this $\theta$, how probable would *exactly my dataset* be?" That probability, viewed as a function of $\theta$, is the **likelihood**.

Because the data points are **IID** (Lecture 03) their probabilities multiply:
$$L(\theta;\mathcal D)=p(\mathcal D\mid\theta)=\prod_{i=1}^{D}\text{bernoulli}(x_i;\theta)=\prod_{i=1}^{D}\theta^{x_i}(1-\theta)^{1-x_i}=\theta^{c_1}(1-\theta)^{c_0}.$$

Symbols: $\prod$ (capital pi) = "multiply all of these", just as $\sum$ = "add all of these"; $c_1,c_0$ = number of ones and zeros.

For our data $L(\theta)=\theta^4(1-\theta)^6$. Try values:

| guess $\theta$ | $\theta^4(1-\theta)^6$ |
|---|---|
| 0.1 | 0.0000531 |
| 0.2 | 0.000419 |
| 0.3 | 0.000953 |
| **0.4** | **0.001194** ← biggest |
| 0.5 | 0.000977 |
| 0.6 | 0.000531 |
| 0.7 | 0.000175 |

The value that maximizes it is the **maximum likelihood estimate**:
$$\theta^*=\arg\max_\theta L(\theta;\mathcal D).$$
($\arg\max$ = "the input that gives the biggest output", not the output itself; the star means "best".)

*Important distinction:* the likelihood is a function of **$\theta$**, with the data fixed. It is **not** a probability distribution over $\theta$ — the column above does not sum to one.

### 4.3 Why the numbers get tiny

A product of many numbers below 1 collapses toward 0. With 100 flips at 40% heads the likelihood at $\theta=0.4$ is $5.9\times10^{-30}$; with 1000 flips it is $5.2\times10^{-293}$ (the lecture shows the same pattern, "10⁻²⁹" and "10⁻²⁸⁹"-sized values). A computer's `float64` bottoms out near $10^{-308}$; beyond that it becomes 0 (underflow) and the maximization breaks.

### 4.4 The log trick

Take the logarithm. Two facts make this legal and useful:

1. $\log(ab)=\log a+\log b$ and $\log(a^n)=n\log a$ — **products become sums**, and sums of moderate numbers are numerically safe.
2. $\log$ is **strictly increasing** ($A>B\Rightarrow\log A>\log B$), so the $\theta$ that maximizes $L$ also maximizes $\log L$. The peak does not move.

For the pass/fail data:
$$\log L(\theta)=4\log\theta+6\log(1-\theta).$$
At $\theta=0.4$: $4\log0.4+6\log0.6=4(-0.9163)+6(-0.5108)=-6.730$. (Natural log throughout.)

### 4.5 Find the peak: derivative $=0$

Rules needed: $\frac{d}{d\theta}\log\theta=\frac1\theta$ and $\frac{d}{d\theta}\log(1-\theta)=-\frac{1}{1-\theta}$ (chain rule).
$$\frac{d}{d\theta}\log L=\frac4\theta-\frac6{1-\theta}=0\;\Rightarrow\;4(1-\theta)=6\theta\;\Rightarrow\;\theta^*=\frac{4}{10}=0.4.$$
General case: $\theta^*=c_1/(c_1+c_0)=c_1/D$ — the fraction of ones.

**Check that it is a maximum, not a minimum:** the second derivative $-\frac4{\theta^2}-\frac6{(1-\theta)^2}$ at $\theta=0.4$ is $-25-16.67=-41.67<0$ (a downward-curving hill) ✔. And at $\theta=0.5$ the slope is $4/0.5-6/0.5=-4<0$: the curve is already going down, so the peak is to the left of 0.5 ✔.

### 4.6 From likelihood to a loss

Machine learning prefers to *minimize*. Since $\arg\max f=\arg\min(-f)$:
$$\theta^*=\arg\min_\theta\ \Big(-\frac1D\log L(\theta;\mathcal D)\Big).$$
This is the **negative average log-likelihood (NLL/D)**. It is a **loss**: lower is better. Dividing by $D$ does not move the minimum; it only makes numbers comparable across datasets of different size (in code: `.mean()` instead of `.sum()`).

| $\theta$ | NLL/D $=-\frac1{10}(4\log\theta+6\log(1-\theta))$ |
|---|---|
| 0.4 | **0.6730** |
| 0.5 | 0.6931 ($=\log 2$) |

A fair-coin guess loses slightly to the best guess, as it should. Note $0.6931$ is also the loss of a model that always says "50/50" — a useful reference point for logistic regression (Lecture 08).

### 4.7 More data ⇒ a sharper peak

The lecture plots the likelihood for 10, 100 and 1000 samples: with 10 the hill is wide (many $\theta$ look plausible); with 1000 it is a narrow spike. The values also shrink ($\sim10^{-3}\to10^{-29}\to10^{-289}$), but the **log-likelihood averaged over $D$** shows a clear peak even for 10,000 samples. So more data means a more *certain* estimate.

### 4.8 What did we assume?

* The data follow a **Bernoulli** model (two outcomes).
* The points are **IID**.

If either is wrong the answer can be meaningless even if the algebra is perfect. Also: **MLE gives a point estimate** — one value of $\theta$, not a range of plausible values (Lectures 11–12 deal with how much to trust it).

### 4.9 Notation details from the lecture

* $\theta\in[0,1]$ includes the endpoints; $(0,1)$ would exclude them. The $\log$ needs inputs $>0$; the lecture notebook adds a tiny `1e-8` inside the log to avoid $\log 0$.
* Write $P(X=x_i)$, not $P(X=\text{heads})$: $X$ takes the values 0 and 1.
* Counting form: $L=\theta^{c_1}(1-\theta)^{c_0}$, so $\theta^*=c_1/D$.

### 4.10 The lecture's NumPy `bernoulli` function, traced

```python
def bernoulli(theta, x):
    assert ((theta >= 0.) & (theta <= 1.)).all()
    assert ((x == 0) | (x == 1)).all()
    assert len(theta.shape) == 1
    assert len(x.shape) == 1
    if theta.shape[0] > 1 and x.shape[0] > 1:
        theta = theta.reshape(-1, 1)      # (n_theta, 1)
        x = x.reshape(1, -1)              # (1, n_x)
    return (theta ** x) * ((1 - theta) ** (1 - x))
```

*Trace 1.* `theta = np.array([0.4])` (shape `(1,)`) and `x = np.array([0,0,1])` (shape `(3,)`): the `if` is skipped; shapes `(1,)` and `(3,)` broadcast to `(3,)`: $[0.6,\,0.6,\,0.4]$ (up to float round-off).

*Trace 2 (our dataset on a grid of $\theta$ values).*

```python
x = np.array([1,0,1,0,0,0,1,0,1,0])
theta = np.linspace(0, 1, 101)               # shape (101,)
p = bernoulli(theta, x)                      # reshape to (101,1) and (1,10) -> (101, 10)
L = p.prod(axis=-1)                          # likelihood per theta, shape (101,)
logL_mean = np.log(p + 1e-8).mean(axis=-1)   # (1/D) log L per theta
theta[L.argmax()]                            # 0.4
theta[(-logL_mean).argmin()]                 # 0.4  — same answer
```

Broadcasting is triggered by combining `(…, 1, …)` with `(…, n, …)` at the same position (Lecture 02).

#### 📖 Textbook deep dive

##### Murphy §4.2 (p.107) — MLE defined and why log

$\hat{\boldsymbol\theta}_{\rm mle}=\arg\max_\theta p(\mathcal D|\theta)$ and, for iid data, $=\arg\min_\theta\ \text{NLL}(\theta)$ with $\text{NLL}(\theta)=-\sum_n\log p(y_n|x_n,\theta)$. Murphy lists the same reason to take logs: products of small numbers underflow, and sums are easier to differentiate.

*Same recipe for the three standard cases (all at p.110–112):*

| Model | Data | MLE |
|---|---|---|
| Bernoulli (§4.2.3, p.110) | $N_1$ ones out of $N$ | $\hat\theta=N_1/N$ |
| Categorical (§4.2.4) | counts $N_k$ | $\hat\theta_k=N_k/N$ |
| Gaussian (§4.2.5, p.111) | $y_1..y_N$ | $\hat\mu=\bar y$, $\hat\sigma^2=\frac1N\sum(y_n-\bar y)^2$ |
| Multivariate normal (§4.2.6, p.112) | $\mathbf x_1..\mathbf x_N\in\mathbb R^D$ | $\hat{\boldsymbol\mu}=\bar{\mathbf x}$, $\hat{\boldsymbol\Sigma}=\frac1N\sum(\mathbf x_n-\bar{\mathbf x})(\mathbf x_n-\bar{\mathbf x})^\top$ |

*Running example, categorical:* $y_3$ counts $[6,2,2]$, so $\hat{\boldsymbol\theta}=[0.6,0.2,0.2]$.
*Running example, Gaussian on the scores:* $\hat\mu=62.4$, $\hat\sigma^2=131.64$, $\hat\sigma=11.473$. The log-likelihood at the MLE is $-\frac N2\log(2\pi\hat\sigma^2)-\frac N2=-38.59$, i.e. NLL/D $=3.859$ nats per student. (Gaussian log-likelihoods *can* be positive for tiny $\sigma$ because densities can exceed 1.)

##### Murphy §4.2.2 — MLE = minimizing KL divergence to the data

Define the **empirical distribution** $p_{\rm emp}(y)=\frac1N\sum_n\delta(y-y_n)$ (a spike of mass $1/N$ on each data point). Then
$$\text{KL}(p_{\rm emp}\,\|\,q_\theta)=-\mathbb H(p_{\rm emp})-\frac1N\sum_n\log q_\theta(y_n),$$
so minimizing KL to the data is *exactly* minimizing NLL/D. **Interpretation:** MLE picks the model distribution closest to the data distribution. Cross-entropy (Lecture 08–09) is this same quantity.

##### Murphy §4.2.6 — a warning about the MVN MLE (Assignment 1!)

For $D$-dimensional data with $N$ samples, $\hat{\boldsymbol\Sigma}$ has rank at most $N-1$, so if $N<D$ it is **singular** (not invertible). That is why Assignment 1 adds a tiny $\epsilon I$ to $\boldsymbol\Sigma$ for MNIST ($D=784$ pixels, many always-zero pixels): the covariance of constant pixels is exactly 0.

##### Bishop §1.2.4 (p.24–28) — ML is biased for the variance

Bishop derives $\mu_{\rm ML}=\frac1N\sum x_n$, $\sigma^2_{\rm ML}=\frac1N\sum(x_n-\mu_{\rm ML})^2$ and then shows
$$\mathbb E[\mu_{\rm ML}]=\mu,\qquad \mathbb E[\sigma^2_{\rm ML}]=\Big(\frac{N-1}{N}\Big)\sigma^2.$$
So the ML variance **systematically underestimates** the true variance (by a factor $\frac{N-1}{N}$), and the unbiased estimate multiplies by $\frac N{N-1}$. (Bishop's explanation: the sample mean is fitted to the same data, which makes the points look closer to their mean than to the true mean.) *Check by simulation (σ=1):* $\mathbb E[\hat\sigma^2_{\rm ML}]=0.50,\,0.80,\,0.90$ for $n=2,5,10$ — matching $(n-1)/n$. Lecture 12 returns to this.

##### Bishop §1.2.5 (p.28) — curve fitting revisited

Assuming Gaussian noise $t=y(x,\mathbf w)+\epsilon$, $\epsilon\sim\mathcal N(0,\beta^{-1})$, the log-likelihood is $-\frac\beta2\sum(y(x_n,\mathbf w)-t_n)^2+\text{const}$, so **maximizing likelihood = minimizing sum of squares.** This is the bridge to Lecture 07.

##### Solved exercises

**Murphy Ex 4.1 (Gaussian MLE).** $\log L=-\frac N2\log(2\pi\sigma^2)-\frac1{2\sigma^2}\sum(y_n-\mu)^2$. $\partial_\mu=\frac1{\sigma^2}\sum(y_n-\mu)=0\Rightarrow\hat\mu=\bar y$. $\partial_{\sigma^2}=-\frac N{2\sigma^2}+\frac1{2\sigma^4}\sum(y_n-\mu)^2=0\Rightarrow\hat\sigma^2=\frac1N\sum(y_n-\hat\mu)^2$.

**Murphy Ex 4.7 (bias of the variance MLE).** $\mathbb E[\hat\sigma^2]=\frac1N\mathbb E\big[\sum(y_n-\bar y)^2\big]=\frac1N\big(\sum\mathbb E[y_n^2]-N\mathbb E[\bar y^2]\big)=\frac1N\big(N(\sigma^2+\mu^2)-N(\mu^2+\sigma^2/N)\big)=\frac{N-1}{N}\sigma^2.$

**Murphy Ex 4.8 (when $\mu$ is known).** Then $\hat\sigma^2=\frac1N\sum(y_n-\mu)^2$ and $\mathbb E[(y_n-\mu)^2]=\sigma^2$ so $\mathbb E[\hat\sigma^2]=\sigma^2$: **unbiased**. The bias only appears because we also estimated $\mu$ from the same data (we "used up" one degree of freedom).

#### ➕ Textbook extra — MLE vs MAP and "add-one" smoothing
Murphy §4.5 (p.120): if you have 0 ones in 3 trials the MLE says $\theta=0$ ("it can never happen"), which is dangerous. Adding a prior (MAP) with a "pseudo-count" gives $\hat\theta=(N_1+1)/(N+2)$ (*add-one* or Laplace smoothing). For our data this is $(4+1)/(10+2)=0.4167$. Regularization/MAP is week 6 of the schedule.

#### ✅ Key takeaways

* Likelihood $L(\theta;\mathcal D)=\prod_i p(x_i|\theta)$ — "how probable is *my* data if the parameters were $\theta$".
* **Use $\log$:** products → sums, no underflow, same maximizer.
* $\theta^*=\arg\max\log L=\arg\min -\frac1D\log L$ (the NLL/D loss).
* For Bernoulli, $\theta^*=c_1/D$: the observed frequency (our data: $0.4$, NLL/D $0.673$).
* More data ⇒ sharper peak; but MLE is still a **point estimate** resting on Bernoulli + IID assumptions.
* Gaussian MLE: sample mean and the $1/N$ variance, which is **biased** by $(N-1)/N$.

---


## Lecture 05 — Git and GitLab

**Why this lecture exists.** Assignments are handed in through Git, and the grader looks at **one specific snapshot** of your code identified by a *commit hash*. Doing this wrong can mean the grader sees the wrong (or no) code. This lecture is also the only one with **no textbook chapter** — the books are about maths, not tooling.

### 5.1 Why use a version control system?

The slides give three reasons:

1. Software development is hard, especially in machine learning.
2. To keep things simple, complex projects are implemented in **many small steps**.
3. Those steps are **tracked with a version control system** such as Git, the de-facto standard.

**Plain words.** Git is a "save history" for a folder of code. Each save is a **commit** (a snapshot plus a message). You can look back at any snapshot, compare versions, undo mistakes, and work on a risky idea on a separate **branch** without breaking what works.

### 5.2 How Assignment 1 is set up

* Every student has a **private GitLab group** on the department server. Your NSID (university username) is in the URL:
  `https://git.cs.usask.ca/cmpt-423-f26-private-abc123` (`abc123` stands for your NSID). Only you and the instructor should have access.
* Inside it is a **private repository** for the assignment, e.g. `…/cmpt-423-820-f26-a1`.

*A wording inconsistency in the slides.* Slide 3 says each of you has a repository; the title of slide 4 says "One Git Repository per Team". Interpretation: individual assignments use per-student repositories, team assignments use one repository per team. Check the assignment handout if unsure.

### 5.3 The branch structure

| Branch | Who can write | What it holds |
|---|---|---|
| **`main`** | Instructor: read+write. **Student: read only.** | The starter code. Cannot be modified by you. **You cannot merge into `main`.** |
| **`solution`** | Instructor and student: read+write | **Your work.** You solve the project here through commits 1, 2, 3, … |

Commands from the slide:

```bash
git branch solution          # create the branch
git checkout solution        # switch to it
# or in one command:
git checkout -b solution
```

You may open a **merge request** on GitLab, which simply tracks every change you made compared with `main` — but **do not merge it**. "Simply leave your project solution in the `solution` branch."

### 5.4 A typical work session (traced)

```bash
git clone https://git.cs.usask.ca/cmpt-423-f26-private-abc123/cmpt-423-820-f26-a1.git
cd cmpt-423-820-f26-a1
git checkout -b solution          # work on the solution branch, never main
# ... edit the notebook / Python files ...
git status                        # what changed?
git add a1.py                     # stage the file(s) you want in the snapshot
git commit -m "Implement Gaussian MLE"   # snapshot 1
# ... more edits ...
git commit -am "Vectorize log-likelihood"  # snapshot 2 (-a stages tracked files)
git push -u origin solution       # upload so GitLab (and the grader) can see it
```

*Meaning of each word:* **clone** = copy the remote repository to your computer; **add** = choose what goes in the next commit; **commit** = save a snapshot; **push** = upload your commits to GitLab; `origin` = the nickname for the GitLab copy.

### 5.5 Submitting: the commit hash

You submit the **commit hash** that should be graded. A commit hash is a long hexadecimal fingerprint of one snapshot, e.g. `81de7dd8034e500d402762a23e347357415ed79d` (the slide example; real hashes are 40 characters, "often very long").

1. **Step 1 — check** that you are looking at the **`solution`** branch on the GitLab website and that **all files to be submitted appear there**.
2. **Step 2 — copy the commit hash** of the latest (or intended) commit and submit it (together with the PDF report on Canvas).

```bash
git log --oneline -3      # shows short hashes of recent commits
git rev-parse HEAD        # prints the full hash of the current commit
```

**Why a hash?** A hash identifies *exactly* one version. If you keep pushing after submitting, the grader still grades the snapshot you named.

#### 📖 Textbook deep dive

Neither Murphy nor Bishop covers Git. The nearest idea in the course is **reproducibility of experiments**, discussed in Lecture 11 (experiment reporting): to re-create a result you need to know exactly which code produced it. A commit hash is that "which code" label.

#### ➕ Textbook extra — none
(No textbook coverage.)

#### ✅ Key takeaways

* Work **only** on the `solution` branch; `main` is read-only starter code.
* Make **small commits** with clear messages; `git push` so GitLab has them.
* Open a merge request if you like, but **never merge it**.
* **Submit the commit hash** of the snapshot to grade, after checking all files are on `solution`.
* Git tracks history so you can undo mistakes and compare versions.

---


## Lecture 06 — Linear Algebra

**Why this lecture exists.** A single measurement is a number, but real data has many numbers per example (an MNIST image has 784 pixels; a student has hours, sleep, …). Linear algebra is the language for handling **lists of numbers (vectors)** and **tables of numbers (matrices)** with a few short formulas instead of loops. Everything from Lecture 07 on is written this way.

### 6.1 Vectors

**Plain words.** A vector is an ordered list of numbers. You can picture a 2-number vector as an arrow (or a point): "3 to the right, 2 up".

**Notation (from the slides).** Vectors are written as **lowercase bold** letters, matrices as **uppercase bold** letters:
$$\mathbf v=\begin{bmatrix}x_1\\ \vdots\\ x_n\end{bmatrix}\in\mathbb R^n.$$
"$\in\mathbb R^n$" reads "lives in $n$-dimensional real space" — it is a list of $n$ real numbers.

**Worked example.** $\mathbf v=[3,2]^\top\in\mathbb R^2$. Our ten students' hours form $\mathbf h\in\mathbb R^{10}$. A $28\times28$ image flattened to 784 pixel values is a vector in $\mathbb R^{784}$: **one image = one point in 784-dimensional space.** You cannot draw 784 dimensions, but the rules are the same as in 2-D.

**Column vs row; transpose.** A column vector is $n\times1$; its **transpose** $\mathbf v^\top$ is the $1\times n$ row $[3,2]$. Maths texts use columns, NumPy code usually stores each example as a **row** — a classic source of shape bugs.

### 6.2 Matrices are 2-D tensors

A **matrix** is a finite two-dimensional tensor (Lecture 02): $m$ rows and $n$ columns, written $\mathbf M\in\mathbb R^{m\times n}$. A dataset of $D$ examples with $n$ features is a $D\times n$ matrix — one example per row. Our student design matrix (a column of 1s and the hours) is $\mathbf X\in\mathbb R^{10\times 2}$.

### 6.3 Norms: size and distance

**Plain words.** A **norm** $\|\mathbf v\|$ measures the size of a vector; applied to a *difference* of two vectors it gives the **distance** between two points.

Slide example: $\mathbf p_1=(1,2)$, $\mathbf p_2=(4,4)$, so $\mathbf v=\mathbf p_2-\mathbf p_1=(3,2)$.

| Norm | Formula | Value for $(3,2)$ | Picture |
|---|---|---|---|
| **L1** | $\sum_i\lvert v_i\rvert$ | $3+2=\mathbf 5$ | walking along city blocks |
| **L2** (Euclidean) | $\sqrt{\sum_i v_i^2}$ | $\sqrt{9+4}=\sqrt{13}\approx\mathbf{3.61}$ | straight line |
| **L∞** | $\max_i\lvert v_i\rvert$ | $\max(3,2)=\mathbf 3$ | largest single difference |

**L2 is the one used most.** Its **square** $\|\mathbf x-\boldsymbol\mu\|^2=\sum_i(x_i-\mu_i)^2$ is the sum of squared differences — it is what MSE (Lecture 07) and k-means (Assignment 1) are built from.

*Running example.* $\mathbf h=(4,2,6,1,3,5,7,3,8,2)$: $\|\mathbf h\|_1=41$, $\|\mathbf h\|_2=\sqrt{217}=14.731$, $\|\mathbf h\|_\infty=8$. Distance between student 1 (4 h, 66 pts) and student 2 (2 h, 48 pts): difference $(2,18)$, so L1 $=20$ and L2 $=\sqrt{328}=18.11$.

### 6.4 Dot product and matrix multiplication

**Dot product.** Multiply two vectors entry by entry and add: $\mathbf a^\top\mathbf b=\sum_i a_ib_i$. E.g. $[1,2,3]\cdot[4,5,6]=4+10+18=32$. Special case: $\mathbf v^\top\mathbf v=\|\mathbf v\|_2^2$.

**Matrix multiplication.** For $\mathbf A\in\mathbb R^{m\times n}$ and $\mathbf B\in\mathbb R^{n\times k}$ the product $\mathbf C=\mathbf A\mathbf B\in\mathbb R^{m\times k}$ has entries
$$c_{ij}=\sum_{\ell=1}^{n}a_{i\ell}\,b_{\ell j}\qquad(\text{row }i\text{ of }\mathbf A\ \cdot\ \text{column }j\text{ of }\mathbf B).$$
**Shape rule:** $(m\times\mathbf n)\cdot(\mathbf n\times k)\to(m\times k)$ — inner sizes must match, outer sizes give the result.

*Worked example.*
$$\begin{bmatrix}1&2\\3&4\end{bmatrix}\begin{bmatrix}5\\6\end{bmatrix}=\begin{bmatrix}1\cdot5+2\cdot6\\3\cdot5+4\cdot6\end{bmatrix}=\begin{bmatrix}17\\39\end{bmatrix}.$$
In NumPy: `np.array([[1,2],[3,4]]) @ np.array([5,6])` → `[17 39]`.

*Running example (this is the key product of Lecture 07).* With $\mathbf X$ the $10\times2$ design matrix:
$$\mathbf X^\top\mathbf X=\begin{bmatrix}10&41\\41&217\end{bmatrix},\qquad \mathbf X^\top\mathbf y=\begin{bmatrix}624\\2804\end{bmatrix}.$$
Check: top-left $=\sum 1\cdot1=10$; off-diagonal $=\sum h_i=41$; bottom-right $=\sum h_i^2=217$; and $624=\sum s_i$, $2804=\sum h_is_i$.

### 6.5 A matrix is a transformation (change of basis)

**Plain words.** Multiplying a vector by a matrix **moves it to a new place**: it re-expresses the same vector using new "basic arrows". The slides call $\mathbf M\mathbf v$ a **linear projection into a new basis**.

**Slide example.** $\mathbf M=\begin{bmatrix}1&-1\\1&0\end{bmatrix}$, whose **columns** are $\mathbf x_1'=(1,1)$ and $\mathbf x_2'=(-1,0)$.
$$\mathbf M\begin{bmatrix}1\\0\end{bmatrix}=\begin{bmatrix}1\\1\end{bmatrix},\qquad \mathbf M\begin{bmatrix}0\\1\end{bmatrix}=\begin{bmatrix}-1\\0\end{bmatrix}.$$
**The columns of $\mathbf M$ are where the basic arrows $(1,0)$ and $(0,1)$ land**; every other vector follows by linearity. Check with $\mathbf v=(3,2)$: $\mathbf M\mathbf v=3\cdot(1,1)+2\cdot(-1,0)=(1,3)$ ✔.

*Why this matters:* Assignment 1 turns random noise into image-like vectors by multiplying by a matrix (the Cholesky factor, §6.7 below). Also, in Lecture 07 the design matrix times $\boldsymbol\theta$ combines *columns* of $\mathbf X$ with weights from $\boldsymbol\theta$.

### 6.6 Tools you need for regression: identity, inverse, pseudo-inverse

(These are used in Lecture 07; the facts below come from the Lecture 07 slides and the textbooks rather than the Lecture 06 deck.)

| Name | Definition | Role |
|---|---|---|
| **Identity** $\mathbf I$ | 1s on the diagonal, 0s elsewhere; $\mathbf I\mathbf v=\mathbf v$ | "multiply by 1" |
| **Inverse** $\mathbf M^{-1}$ | $\mathbf M^{-1}\mathbf M=\mathbf I$ | "divide by $\mathbf M$"; only for square, non-degenerate matrices |
| **Symmetric** | $\mathbf M^\top=\mathbf M$ | covariance matrices; $\mathbf X^\top\mathbf X$ |
| **Lower triangular** | zeros above the diagonal | cheap to solve; the Cholesky factor |
| **Outer product** $\mathbf v\mathbf v^\top$ | $(n\times1)(1\times n)=n\times n$ | appears in covariance formula |

**Singular matrices (no inverse).** A matrix with a zero column, or a column that is a copy/combination of others, "squashes space flat" and loses information; it cannot be undone. Linear regression's pseudo-inverse $(\mathbf X^\top\mathbf X)^{-1}\mathbf X^\top$ needs the columns of $\mathbf X$ to be **linearly independent**. In our example $\mathbf X^\top\mathbf X$ has determinant $10\cdot217-41^2=489\ne0$, eigenvalues $2.175$ and $224.82$ (condition number $103.4$), so it is invertible:
$$(\mathbf X^\top\mathbf X)^{-1}=\frac1{489}\begin{bmatrix}217&-41\\-41&10\end{bmatrix}.$$

### 6.7 Cholesky factor: turning noise into correlated noise (Assignment 1)

Any symmetric positive-definite matrix $\boldsymbol\Sigma$ can be written $\boldsymbol\Sigma=\mathbf L\mathbf L^\top$ with $\mathbf L$ lower triangular. To sample $\mathbf y\sim\mathcal N(\boldsymbol\mu,\boldsymbol\Sigma)$: draw $\mathbf z\sim\mathcal N(\mathbf 0,\mathbf I)$ (independent 1-D normals) and set $\mathbf y=\mathbf L\mathbf z+\boldsymbol\mu$.

*Worked example.* $\boldsymbol\Sigma=\begin{bmatrix}4&2\\2&3\end{bmatrix}$ has $\mathbf L=\begin{bmatrix}2&0\\1&1.4142\end{bmatrix}$ and indeed $\mathbf L\mathbf L^\top=\begin{bmatrix}4&2\\2&3\end{bmatrix}$ ✔. With $\mathbf z=(1,1)$ and $\boldsymbol\mu=(10,20)$: $\mathbf y=(2+0+10,\ 1+1.4142+20)=(12,\ 22.414)$.

#### 📖 Textbook deep dive

##### Murphy §7.1.3 (p.234) — norms in general
* **$p$-norm:** $\|\mathbf x\|_p=(\sum|x_i|^p)^{1/p}$; $p=1,2$ give L1, L2 and $p\to\infty$ gives max-norm. A norm must satisfy: $\|\mathbf x\|\ge0$ with equality only at $\mathbf 0$; $\|a\mathbf x\|=|a|\|\mathbf x\|$; triangle inequality $\|\mathbf x+\mathbf y\|\le\|\mathbf x\|+\|\mathbf y\|$.
* **Frobenius norm** of a matrix: $\|\mathbf A\|_F=\sqrt{\sum_{ij}a_{ij}^2}$ (the L2 norm of the flattened matrix).

*Check:* for $(3,2)$: L1 $5\ge$ L2 $3.61\ge$ L∞ $3$ — always true.

##### Murphy §7.2 (p.242–249) — products
* Vector–vector: inner product $\mathbf x^\top\mathbf y$ (scalar) and outer product $\mathbf x\mathbf y^\top$ (matrix).
* Matrix–vector $\mathbf A\mathbf x$ is a **linear combination of the columns** of $\mathbf A$ with weights $x_j$ — the same as §6.5's "columns land here".
* **Data-matrix tricks (§7.2.4):** with $\mathbf X$ being $N\times D$, the *sum-of-squares matrix* is $\mathbf X^\top\mathbf X=\sum_n\mathbf x_n\mathbf x_n^\top$; the *scatter matrix* is the same on mean-centred data (divide by $N$ to get the MLE covariance — Lecture 04/Assignment 1). The **distance matrix** of squared pairwise distances is $D_{ij}=\|\mathbf x_i\|^2-2\mathbf x_i^\top\mathbf y_j+\|\mathbf y_j\|^2$, which Murphy notes is "much faster than for loops" (the k-means distance step).

##### Murphy §7.3 (p.249) — inverse
A square $\mathbf A$ is invertible iff it has full rank (det $\ne0$). In §7.7.1 (p.267) Murphy shows how to solve $\mathbf A\mathbf x=\mathbf b$ **without taking inverses**: factor $\mathbf A=\mathbf L\mathbf U$ (two triangular matrices) and use back-substitution. In NumPy this is what `np.linalg.solve(A, b)` does, and it is the preferred way to "apply $\mathbf A^{-1}$".

##### Murphy §7.5.3 (p.261) — pseudo-inverse
Defined by four properties; for a tall full-column-rank $\mathbf A$: $\mathbf A^\dagger=(\mathbf A^\top\mathbf A)^{-1}\mathbf A^\top$, a **left inverse** ($\mathbf A^\dagger\mathbf A=\mathbf I$) but *not* a right inverse ($\mathbf A\mathbf A^\dagger$ has rank $n<m$). For a short full-row-rank $\mathbf A$: $\mathbf A^\dagger=\mathbf A^\top(\mathbf A\mathbf A^\top)^{-1}$, a right inverse. In general, via the SVD $\mathbf A=\mathbf U\mathbf S\mathbf V^\top$: $\mathbf A^\dagger=\mathbf V\,\mathrm{diag}(1/\sigma_1,\dots,1/\sigma_r,0,\dots,0)\,\mathbf U^\top$ — this is what `np.linalg.pinv` computes, and it works even when columns are dependent.

*Check on the students:* `np.linalg.pinv(X) @ s` gives $[41.8078,\ 5.0225]$, matching the closed form.

##### Murphy §7.6.3 (p.266) — Cholesky and sampling an MVN
Exactly the recipe of §6.7: $\mathrm{Cov}[\mathbf y]=\mathbf L\,\mathrm{Cov}[\mathbf z]\,\mathbf L^\top=\mathbf L\mathbf I\mathbf L^\top=\boldsymbol\Sigma$. `np.linalg.cholesky` returns the *lower* factor $\mathbf L$. Cost $O(V^3)$ for $V$ variables.

##### Murphy §7.7.3 (p.268) and §7.8 (p.269) — least squares and gradients
Over-determined systems ($\mathbf X\boldsymbol\theta\approx\mathbf y$, more rows than unknowns) are solved by minimizing $\|\mathbf X\boldsymbol\theta-\mathbf y\|^2$, giving the **normal equations** $\mathbf X^\top\mathbf X\boldsymbol\theta=\mathbf X^\top\mathbf y$ (Lecture 07). The gradient of $f(\boldsymbol\theta)=\tfrac12\|\mathbf X\boldsymbol\theta-\mathbf y\|^2$ is $\mathbf X^\top(\mathbf X\boldsymbol\theta-\mathbf y)$, and its Hessian $\mathbf X^\top\mathbf X$ is positive definite when $\mathbf X$ has full column rank (so the minimum is unique).

##### Solved exercises (Murphy §7.9, p.274)
**Ex 7.1.** (a) The columns of $\mathbf R(\alpha)$ are $(\cos\alpha,\sin\alpha,0)$, $(-\sin\alpha,\cos\alpha,0)$, $(0,0,1)$. Each has length $\sqrt{\cos^2+\sin^2}=1$, and the dot product of the first two is $-\cos\alpha\sin\alpha+\sin\alpha\cos\alpha=0$, the third is perpendicular to both. So the columns are orthonormal, i.e. $\mathbf R^\top\mathbf R=\mathbf I$. (b) A rotation about the $z$-axis leaves the $z$-axis fixed: the unit eigenvector with eigenvalue 1 is $\mathbf v=(0,0,1)^\top$ (up to sign), for every $\alpha$.

**Ex 7.2.** $\mathbf A=\mathrm{diag}(2,3)$: $\det(\mathbf A-\lambda\mathbf I)=(2-\lambda)(3-\lambda)=0\Rightarrow\lambda=2,3$ with eigenvectors $(1,0)$ and $(0,1)$ (`np.linalg.eig(np.diag([2.,3.]))` agrees).

#### ➕ Textbook extra — SVD, eigen-decomposition, matrix calculus
The slides skip SVD, eigenvalues and matrix derivatives. They return for PCA (week 12) and gradient descent (week 9). The one fact to keep now: **singular values of $\mathbf X$ tell you how close $\mathbf X^\top\mathbf X$ is to being singular** (condition number = largest/smallest eigenvalue, here $224.8/2.175=103.4$).

#### ✅ Key takeaways

* Vectors = lowercase bold, matrices = uppercase bold; a matrix is a 2-D tensor.
* Norms: L1 (sum |·|), L2 (Euclid), L∞ (max). For $(3,2)$: 5, $\sqrt{13}$, 3.
* $c_{ij}=\sum_\ell a_{i\ell}b_{\ell j}$; shapes $(m\times n)(n\times k)\to(m\times k)$.
* $\mathbf M\mathbf v$ **re-expresses** $\mathbf v$ in new basis arrows — the columns of $\mathbf M$.
* Pseudo-inverse $(\mathbf X^\top\mathbf X)^{-1}\mathbf X^\top$ needs independent columns; otherwise use SVD/`pinv`.
* Cholesky $\boldsymbol\Sigma=\mathbf L\mathbf L^\top$ turns white noise into correlated samples.

---


## Lecture 07 — Linear Regression

**Why this lecture exists.** First real model: **predict a number** (here: a test score from hours studied). It is also the first time you see the full recipe — *assume a distribution → write the likelihood → take the log → optimize* — turn into a loss function (squared error), and into a closed-form solution you can compute in one line.

### 7.1 The problem

Data are pairs $(x_i,y_i)$: an input and the number we want to predict. In the running example $x_i$ = hours studied and $y_i$ = test score $s_i$ for ten students. We want a rule that predicts the score from the hours.

### 7.2 Two ways to see the same line

**View 1 — draw the best line.** $\hat y=\theta_0+\theta_1x$, where $\theta_0$ is the intercept and $\theta_1$ the slope.

**View 2 — the lecture's view: a bell curve around the line.** Two students with the same hours do not get exactly the same score. So the model says: *for hours $x$, the score is a normal random variable whose **mean** is $\theta_0+\theta_1x$:*
$$p(y_i\mid x_i;\theta_0,\theta_1)=f\big(y_i;\ \theta_0+\theta_1x_i,\ 1\big)\qquad(\text{normal density, mean = the line, }\sigma=1).$$
Both views give *exactly the same line*; View 2 explains **why**.

### 7.3 Applying the recipe (assume → likelihood → log → drop constants)

1. **Assume:** each $y_i\sim\mathcal N(\theta_0+\theta_1x_i,\,1)$, IID.
2. **Likelihood:** multiply the densities
   $$\prod_{i=1}^{D}\frac1{\sqrt{2\pi}}\exp\!\Big(-\tfrac12\big(y_i-(\theta_0+\theta_1x_i)\big)^2\Big).$$
3. **Log** (products → sums, and $\log\exp(u)=u$):
   $$\sum_{i=1}^{D}\Big[\log\tfrac1{\sqrt{2\pi}}-\tfrac12\big(y_i-(\theta_0+\theta_1x_i)\big)^2\Big].$$
4. **Drop constants.** $\log\frac1{\sqrt{2\pi}}$ and the factor $\frac12$ do not depend on $\theta$; *"adding a constant to a function does not change its optimum"* (the lecture draws two vertically shifted curves with the same peak). What remains: maximize $-\sum_i(y_i-\hat y_i)^2$, i.e. **minimize the Mean Squared Error**
   $$\boxed{\ \text{MSE}(\boldsymbol\theta)=\frac1D\sum_{i=1}^{D}\big(y_i-(\theta_0+\theta_1x_i)\big)^2\ }$$

$y_i-\hat y_i$ is the **prediction error** (residual). We square it so negative and positive errors do not cancel, then average.

> **The big insight of Lecture 07:** *"Every time a machine learning model is optimized using the MSE, this loss assumes $y\sim\mathcal N(\mu(x),\sigma=1)$."* Squared error **is** a Gaussian assumption in disguise. If the data do not spread like a bell curve around the prediction, MSE is the wrong tool. The lecture's counter-example: $y$ taking only the values 1–6 (like die rolls).

*Running example: the numbers.* Under the fitted model with $\sigma=1$, the NLL per student is $\tfrac12\log(2\pi)+\tfrac12\,\text{MSE}=0.9189+\tfrac12(8.2875)=5.0627$.

### 7.4 Solving it: the normal equations

**Step 1 — set the gradient to zero.** Differentiating the MSE with respect to $\theta_0$ and $\theta_1$ (chain rule; each derivative brings down a factor 2 and the inner derivative $-1$ or $-x_i$; constants like $2/D$ vanish against the right-hand side 0):
$$0=\sum_i\big(y_i-\theta_0-\theta_1x_i\big)\quad(E),\qquad 0=\sum_i\big(y_i-\theta_0-\theta_1x_i\big)x_i\quad(F).$$

**Step 2 — stack into matrices.** Let $\boldsymbol\theta=[\theta_0,\theta_1]^\top\in\mathbb R^2$, each input become $\mathbf x_i=[1,x_i]^\top$, and let $\mathbf X\in\mathbb R^{D\times2}$ have the $\mathbf x_i^\top$ as rows (a "tall" matrix), $\mathbf y\in\mathbb R^D$, and $\mathbf 1$ a vector of $D$ ones. Then $\boldsymbol\theta^\top\mathbf x_i=\theta_0\cdot1+\theta_1x_i$, and (E), (F) say
$$0=(\mathbf y-\mathbf X\boldsymbol\theta)^\top\mathbf 1,\qquad 0=(\mathbf y-\mathbf X\boldsymbol\theta)^\top\mathbf x,$$
where $\mathbf x$ is the column of inputs. Both hold when the **error vector is zero or orthogonal to every column of $\mathbf X$**; the clean statement is $\mathbf X^\top(\mathbf y-\mathbf X\boldsymbol\theta)=\mathbf 0$ — the **normal equations** $\mathbf X^\top\mathbf X\boldsymbol\theta=\mathbf X^\top\mathbf y$.

**Step 3 — undo $\mathbf X$.** We want a matrix $\mathbf X^+$ with $\mathbf X^+\mathbf X=\mathbf I$; then $\boldsymbol\theta=\mathbf X^+\mathbf y$. The Moore–Penrose **left pseudo-inverse** is
$$\mathbf X^+=(\mathbf X^\top\mathbf X)^{-1}\mathbf X^\top,$$
which exists iff the columns of $\mathbf X$ are **linearly independent**. In code:

```python
theta = np.linalg.pinv(X) @ y        # (n,D)@(D,) -> (n,)
```

More generally data can be $(\mathbf x,\mathbf y)$ with $\mathbf x\in\mathbb R^n$ and $\mathbf y\in\mathbb R^K$ (several outputs).

#### 7.4.1 Full worked example (running example)

$\mathbf X^\top\mathbf X=\begin{bmatrix}10&41\\41&217\end{bmatrix}$, $\ \mathbf X^\top\mathbf y=\begin{bmatrix}624\\2804\end{bmatrix}$, $\ \det=10\cdot217-41^2=489$.
$$\boldsymbol\theta=\frac1{489}\begin{bmatrix}217&-41\\-41&10\end{bmatrix}\begin{bmatrix}624\\2804\end{bmatrix}=\frac1{489}\begin{bmatrix}217\cdot624-41\cdot2804\\-41\cdot624+10\cdot2804\end{bmatrix}=\frac1{489}\begin{bmatrix}20444\\2456\end{bmatrix}=\begin{bmatrix}41.8078\\5.0225\end{bmatrix}.$$

**Reading the answer:** a student who studies 0 hours is predicted 41.8; each extra hour adds about 5.02 points. Cross-check with the "textbook" slope formula: $\theta_1=S_{xy}/S_{xx}=245.6/48.9=5.0225$ and $\theta_0=\bar s-\theta_1\bar h=62.4-5.0225\cdot4.1=41.8078$ ✔.

| student | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| hours $h$ | 4 | 2 | 6 | 1 | 3 | 5 | 7 | 3 | 8 | 2 |
| score $s$ | 66 | 48 | 71 | 45 | 57 | 62 | 77 | 59 | 83 | 56 |
| prediction $\hat s$ | 61.90 | 51.85 | 71.94 | 46.83 | 56.88 | 66.92 | 76.97 | 56.88 | 81.99 | 51.85 |
| residual $s-\hat s$ | 4.10 | −3.85 | −0.94 | −1.83 | 0.12 | −4.92 | 0.03 | 2.12 | 1.01 | 4.15 |

* $\text{RSS}=\sum(s-\hat s)^2=82.875$, so **MSE $=8.2875$** (RMSE $=2.879$ points).
* The residuals sum to **0** and are orthogonal to $h$ (this *is* the normal equations).
* Prediction for a new student with 5.5 hours: $41.8078+5.0225\cdot5.5=69.43$.
* **$R^2=1-\text{RSS}/\text{TSS}=1-82.875/1316.4=0.9370$.** (TSS $=\sum(s-\bar s)^2=1316.4$.)

### 7.5 Curves with a "linear" model: basis functions

**Plain words.** If the data are not a straight cloud, replace $x$ by a list of features $\boldsymbol\varphi(x)$ computed from it, then fit $y\approx\boldsymbol\varphi(x)^\top\boldsymbol\theta$.

$$\boldsymbol\varphi(x)=[1,\,x,\,x^2,\,x^3,\dots]^\top,\qquad \hat y=\boldsymbol\varphi(x)^\top\boldsymbol\theta=\theta_0+\theta_1x+\theta_2x^2+\cdots$$

The model is still called **linear** regression because $\hat y$ is **linear in the parameters $\boldsymbol\theta$**, even though it is not linear in $x$. The same pseudo-inverse formula works with the new design matrix $\boldsymbol\Phi$ (rows $\boldsymbol\varphi(x_i)^\top$).

A basis function can be *any* map $\mathbb R\to\mathbb R^K$. The lecture's second example is the **staircase**: (1) split the input range into $K$ intervals; (2) map $x$ to a **one-hot vector** with a 1 for the interval it falls in. The model $\boldsymbol\varphi(x)^\top\boldsymbol\theta$ then learns one number per interval — each $\theta_k$ ends up as the average $y$ in interval $k$.

*Running example, polynomial of degree $d$ (fit by least squares):*

| degree | parameters | training MSE |
|---|---|---|
| 1 (line) | $[41.81,\ 5.02]$ | 8.288 |
| 2 | $[41.36,\ 5.288,\ -0.0294]$ | 8.272 |
| 3 | $[33.67,\ 12.86,\ -2.014,\ 0.148]$ | 6.711 |

Higher degree always lowers **training** error, but that does not mean it predicts new students better (Lecture 10 shows it does not).

*Running example, staircase with $K=3$ intervals* (low $\le3$ h, mid 4–6 h, high $\ge7$ h): $\boldsymbol\theta=[53,\ 66.33,\ 80]$ — the mean score in each group (e.g. low group scores 45, 48, 56, 57, 59 → mean 53). Training MSE $=20.87$: worse than the line (8.29) because the true pattern is smoothly increasing.

```python
h = np.array([4,2,6,1,3,5,7,3,8,2.]);  s = np.array([66,48,71,45,57,62,77,59,83,56.])
X = np.stack([np.ones_like(h), h], axis=1)         # (10, 2) design matrix
theta = np.linalg.pinv(X) @ s                       # [41.8078, 5.0225]
resid = s - X @ theta
mse = (resid**2).mean()                             # 8.2875
r2  = 1 - (resid**2).sum() / ((s - s.mean())**2).sum()   # 0.9370
# staircase basis
k = np.digitize(h, [3.5, 6.5])                      # 0,1,2
Phi = np.eye(3)[k]                                  # one-hot, (10, 3)
theta_step = np.linalg.pinv(Phi) @ s                # [53., 66.33, 80.]
```

#### 📖 Textbook deep dive

##### Murphy §11.2.1–11.2.2 (p.371–373) — the model, the NLL and the RSS

Murphy writes $p(y|\mathbf x,\boldsymbol\theta)=\mathcal N(y|\mathbf w^\top\mathbf x,\sigma^2)$, so
$$\text{NLL}(\mathbf w,\sigma^2)=\frac1{2\sigma^2}\sum_n(y_n-\hat y_n)^2+\frac N2\log(2\pi\sigma^2).$$
Up to constants this is the **residual sum of squares** $\text{RSS}(\mathbf w)=\tfrac12\|\mathbf X\mathbf w-\mathbf y\|_2^2$. (Note Murphy puts a $\tfrac12$ and no $1/D$; the slides use $1/D$. They differ by a constant factor, so the minimizer is the same.) Because the weights $\mathbf w$ enter first, **σ is irrelevant for finding $\mathbf w$** — the slide's $\sigma=1$ is a convenience, not a restriction. Afterwards the MLE of the noise is $\hat\sigma^2=\text{MSE}$: here $\hat\sigma^2=8.2875$, $\hat\sigma=2.88$.

*Gradient and normal equations (Eqs. 11.7–11.9):* $\nabla\text{RSS}=\mathbf X^\top\mathbf X\mathbf w-\mathbf X^\top\mathbf y$, so $\mathbf X^\top\mathbf X\mathbf w=\mathbf X^\top\mathbf y$ — "normal" because at the optimum $\mathbf y-\mathbf X\mathbf w$ is *normal (orthogonal)* to the range of $\mathbf X$.

*Uniqueness (Eq. 11.10–11.11):* the Hessian is $\mathbf X^\top\mathbf X$; it is positive definite if $\mathbf X$ has independent columns, because $\mathbf v^\top\mathbf X^\top\mathbf X\mathbf v=\|\mathbf X\mathbf v\|^2>0$ for every $\mathbf v\neq\mathbf 0$ (Murphy's text says "for any $\mathbf v>0$", a typo — see Errata). Hence a *unique global minimum*. Our Hessian $\mathbf X^\top\mathbf X$ has eigenvalues $2.175$ and $224.8$, both positive ✔.

##### Murphy §11.2.2.2 (p.373) — geometry: least squares is a projection

Think of the $D$-dimensional vector $\mathbf y$ and the plane spanned by the columns of $\mathbf X$ (the vector of ones and the vector of hours). The prediction $\hat{\mathbf y}=\mathbf X\boldsymbol\theta$ is the **orthogonal projection** of $\mathbf y$ onto that plane:
$$\hat{\mathbf y}=\mathbf X(\mathbf X^\top\mathbf X)^{-1}\mathbf X^\top\mathbf y=\mathbf H\mathbf y\qquad(\mathbf H=\text{"hat matrix"}).$$
The residual $\mathbf y-\hat{\mathbf y}$ is perpendicular to the plane. *Check:* for the students, $\sum(s-\hat s)\cdot1=0$ and $\sum(s-\hat s)h=0$ (both verified numerically). Pythagoras then gives $\text{TSS}=\text{explained}+\text{RSS}$ (when an intercept is included).

##### Murphy §11.2.2.3–11.2.2.4 (p.375–376) — how to compute it in practice

Forming $(\mathbf X^\top\mathbf X)^{-1}$ squares the condition number of $\mathbf X$ (here $\mathbf X^\top\mathbf X$ has condition number 103.4, so $\mathbf X$ itself has about $\sqrt{103.4}=10.2$) and is numerically risky. Better: **QR decomposition** or **SVD** (what `np.linalg.lstsq` / `pinv` do). Murphy's advice: prefer SVD-based solvers to explicit normal equations. **Weighted least squares** (§11.2.2.4) multiplies each squared error by a weight, so noisier points count less.

##### Murphy §11.2.3 (p.376–379) — other ways to get the same answer

* **§11.2.3.1–11.2.3.2, "simple linear regression" (1-D input, p.377):** slope $=\sum(x_n-\bar x)(y_n-\bar y)\big/\sum(x_n-\bar x)^2$ and intercept $=\bar y-\text{slope}\cdot\bar x$ — exactly the $S_{xy}/S_{xx}$ check used above.
* **§11.2.3.4 (p.377)** computes the MLE *recursively*, one example at a time (a preview of SGD).
* **§11.2.3.6 (p.379)** derives $\hat\sigma^2_{\rm mle}=\text{RSS}/N$: the noise variance is the mean squared residual.

##### Murphy §11.2.4 (p.380–381) — how good is the fit?

* **RMSE** $=\sqrt{\text{RSS}/N}$ (units of $y$): $\sqrt{8.2875}=2.879$ points.
* **$R^2=1-\text{RSS}/\text{TSS}$**, the fraction of variance explained. For a single input it equals the squared correlation $r^2$ between $x$ and $y$: here $r^2=0.9370$ ✔ (computed independently with `np.corrcoef`). A model that predicts $\bar y$ everywhere has $R^2=0$.

##### Murphy §11.3 (p.381) — ridge regression (preview of regularization, week 6)

Add a penalty $\lambda\|\mathbf w\|^2$ to shrink weights: $\hat{\mathbf w}=(\mathbf X^\top\mathbf X+\lambda\mathbf I)^{-1}\mathbf X^\top\mathbf y$. The matrix $\mathbf X^\top\mathbf X+\lambda\mathbf I$ is always invertible for $\lambda>0$, which also cures singular cases. The intercept $w_0$ is *not* penalized (otherwise adding a constant to all $y$ would change the slope), so one centres the data first (Ex 11.2).

*Running example (hours centred, $\lambda=10$):* slope $=S_{xy}/(S_{xx}+\lambda)=245.6/(48.9+10)=4.170$ (vs OLS 5.022: shrunk toward 0), intercept $\bar s-4.170\cdot4.1=45.30$.

##### Murphy §8.4.2 (p.293) — least squares by SGD ("LMS" / delta rule)

Instead of solving, update one example at a time: $\boldsymbol\theta\leftarrow\boldsymbol\theta+\alpha\,(y_n-\boldsymbol\theta^\top\mathbf x_n)\,\mathbf x_n$. *Traced step:* start $\boldsymbol\theta=\mathbf 0$, student 1 ($\mathbf x=[1,4]$, $y=66$), $\alpha=0.01$: error $=66$, update $=0.01\cdot66\cdot[1,4]=[0.66,\,2.64]$. This is the same "(error) × (input)" structure that reappears in logistic regression (Lecture 08).

**Step-size warning.** For gradient descent on the MSE the step must satisfy $\alpha<2/\lambda_{\max}(\mathbf H)$, where $\mathbf H=\tfrac2D\mathbf X^\top\mathbf X$ has eigenvalues $0.435$ and $44.96$; so $\alpha<0.0445$. The huge ratio (103.4) is why plain GD is slow here — standardizing the input fixes it (Murphy §10.2.8).

##### Bishop §3.1 (p.138–146) — linear basis function models

* **Basis functions** $\phi_j(\mathbf x)$: polynomial, Gaussian bumps $\exp(-\frac{(x-\mu_j)^2}{2s^2})$ and sigmoidal $\sigma(\frac{x-\mu_j}{s})$ are the three standard families (Fig. 3.1). The slide's staircase is a fourth.
* **Design matrix** $\boldsymbol\Phi$ with $\Phi_{nj}=\phi_j(\mathbf x_n)$ and **normal equations** $\mathbf w_{\rm ML}=(\boldsymbol\Phi^\top\boldsymbol\Phi)^{-1}\boldsymbol\Phi^\top\mathbf t$ (§3.1.1, p.140–142). Bishop shows the bias parameter $w_0$ simply compensates for "the difference between the average of the target values and the weighted sum of the average basis function values", and that the ML noise precision satisfies $1/\beta_{\rm ML}=\frac1N\sum(t_n-\mathbf w^\top\boldsymbol\phi(\mathbf x_n))^2$ — the residual variance, same as Murphy.
* **Geometry (§3.1.2, p.143):** $\mathbf y=\boldsymbol\Phi\mathbf w_{\rm ML}$ is the projection of $\mathbf t$ onto the subspace spanned by the columns of $\boldsymbol\Phi$ (same picture as Murphy).

##### Solved exercises (Murphy §11.8, p.411–412)

**Ex 11.1 (multi-output regression).** $\boldsymbol\varphi(0)=(1,0)$, $\boldsymbol\varphi(1)=(0,1)$ so $\boldsymbol\Phi^\top\boldsymbol\Phi=\mathrm{diag}(3,3)$ and $\boldsymbol\Phi^\top\mathbf Y$ sums the $y$ vectors of each group: $(-4,-4)$ for $x=0$ and $(4,4)$ for $x=1$. Hence
$$\hat{\mathbf W}=\begin{bmatrix}-4/3&-4/3\\4/3&4/3\end{bmatrix}.$$
Each row is just the group mean — exactly the staircase result above.

**Ex 11.2 (centering and ridge).** Set $\partial J/\partial w_0=-2\mathbf 1^\top(\mathbf y-\mathbf X\mathbf w-w_0\mathbf 1)=0\Rightarrow w_0=\bar y-\bar{\mathbf x}^\top\mathbf w=\bar y$ because the inputs are centred ($\bar{\mathbf x}=0$, and so $\mathbf X^\top\mathbf 1=\mathbf 0$). Then $\partial J/\partial\mathbf w=-2\mathbf X^\top(\mathbf y-\mathbf X\mathbf w-w_0\mathbf 1)+2\lambda\mathbf w=0\Rightarrow(\mathbf X^\top\mathbf X+\lambda\mathbf I)\mathbf w=\mathbf X^\top\mathbf y$.

**Ex 11.3 (partial derivative of RSS).** Write $\text{RSS}=\sum_i\big(y_i-\mathbf w_{-k}^\top\mathbf x_{i,-k}-w_kx_{ik}\big)^2$. Then $\frac{\partial}{\partial w_k}=-2\sum_ix_{ik}(y_i-\mathbf w_{-k}^\top\mathbf x_{i,-k}-w_kx_{ik})=a_kw_k-c_k$ with $a_k=2\sum_ix_{ik}^2=2\|\mathbf x_{:,k}\|^2$ and $c_k=2\sum_ix_{ik}(y_i-\mathbf w_{-k}^\top\mathbf x_{i,-k})=2\mathbf x_{:,k}^\top\mathbf r_k$. Setting it to zero: $\hat w_k=\mathbf x_{:,k}^\top\mathbf r_k/\|\mathbf x_{:,k}\|^2$ — the **orthogonal projection of feature $k$ onto the residual** left by all other features.

(Exercises 11.4–11.6 on elastic net/lasso, splines and EM belong to later weeks.)

#### ➕ Textbook extra — weighted least squares, lasso, splines, the Hessian check
Not in the slides: weighted LS (Murphy §11.2.2.4, p.376), lasso and elastic net (§11.4, p.385), regression splines (§11.5, p.399). Of these only **ridge** is on the course schedule (week 6, regularization).

#### ✅ Key takeaways

* Linear regression = **Gaussian MLE with fixed σ**; the loss is **MSE** $=\frac1D\sum(y-\hat y)^2$.
* **Normal equations** $\mathbf X^\top\mathbf X\boldsymbol\theta=\mathbf X^\top\mathbf y$ ⇒ $\boldsymbol\theta=(\mathbf X^\top\mathbf X)^{-1}\mathbf X^\top\mathbf y=$ `pinv(X) @ y` (needs independent columns). Our result: $\boldsymbol\theta=[41.81,\,5.02]$, MSE $8.29$, $R^2=0.937$.
* Add a **column of 1s** for the intercept.
* **Basis functions** give curves while staying linear *in θ*.
* Geometrically $\hat{\mathbf y}$ is the **projection** of $\mathbf y$ onto the column space; residuals are orthogonal to the inputs.
* Training error always falls with a richer model — but that is not generalization (Lecture 10).

---


## Lecture 08 — Logistic Regression

**Why this lecture exists.** Linear regression predicts a *number*. Often the answer is **yes/no** — spam or not, pass or fail. We need a model that outputs a *probability between 0 and 1* and a way to fit it. The recipe is identical (assume → likelihood → log → optimize) but there is **no closed-form solution**, so this is where **gradient descent** enters the course.

### 8.1 The problem: classification

The dataset is $\mathcal D=\{(\mathbf x,y)\mid \mathbf x\in\mathbb R^n,\ y\in\{0,1\}\}$. Running example: $\mathbf x=[1,h]^\top$ (a constant 1 and hours studied) and $y=t$ (1 = passed). A straight line can output $-3$ or $47$; a probability must live in $[0,1]$.

### 8.2 The sigmoid

$$\sigma(z)=\frac{1}{1+e^{-z}}$$

| $z$ | $\sigma(z)$ |
|---|---|
| $-10$ | $4.5\times10^{-5}\approx0$ |
| $-1$ | 0.269 |
| 0 | 0.5 |
| 1 | 0.731 |
| $+10$ | $0.99995\approx1$ |

It is S-shaped, squashing any real number into $(0,1)$. (This $\sigma$ is a *function*, unrelated to the standard deviation $\sigma$; same Greek letter, different job.)

### 8.3 The model

$$p(y=1\mid\mathbf x;\boldsymbol\theta)=\sigma(\boldsymbol\theta^\top\mathbf x).$$

Compute a score $\boldsymbol\theta^\top\mathbf x$ (a dot product), squash it into a probability, then flip a **biased coin** with that probability. **This is the coin of Lecture 04, except the bias now depends on the input.** The lecture draws the pipeline as $\mathbb R^n\to[0,1]\to\{0,1\}$: a model turns $\mathbf x$ into a probability (the part we must build), then a **Bernoulli trial** is sampled. $\boldsymbol\theta$ is an $n$-dimensional parameter vector; with one scalar input, $\boldsymbol\theta^\top\mathbf x=\theta_0\cdot1+\theta_1x$, which is why the design matrix again has a column of ones.

### 8.4 Applying the recipe

1. **Assume:** $y_i\sim\text{Bernoulli}(\sigma(\boldsymbol\theta^\top\mathbf x_i))$, independent.
2. **Likelihood:** $\prod_ip(y_i\mid\mathbf x_i;\boldsymbol\theta)$.
3. **Log:** $\sum_i\log p(y_i\mid\mathbf x_i;\boldsymbol\theta)$.
4. **Optimize:** no neat formula → walk uphill step by step (§8.6).

**Both cases as one formula (no `if`).** $p(y_i|\mathbf x_i;\boldsymbol\theta)=\sigma(\boldsymbol\theta^\top\mathbf x_i)$ if $y_i=1$ and $1-\sigma(\boldsymbol\theta^\top\mathbf x_i)$ if $y_i=0$. The lecture first shows an `if y == 1` Python function, then the tensor-arithmetic version:

```python
sigmoid_res     = sigmoid(theta @ x)             # σ(θᵀx)
sigmoid_res_inv = 1 - sigmoid_res                # 1 − σ(θᵀx)
p = y * sigmoid_res + (1 - y) * sigmoid_res_inv  # one term always vanishes
```

Taking the log:
$$\log p(y_i\mid\mathbf x_i;\boldsymbol\theta)=y_i\log\sigma(\boldsymbol\theta^\top\mathbf x_i)+(1-y_i)\log\big(1-\sigma(\boldsymbol\theta^\top\mathbf x_i)\big).$$

### 8.5 The loss: binary cross-entropy (BCE)

Flip the sign and average (as in Lecture 04):
$$\boxed{\,J(\boldsymbol\theta)=-\frac1D\sum_{i=1}^D\Big[y_i\log\sigma(\boldsymbol\theta^\top\mathbf x_i)+(1-y_i)\log\big(1-\sigma(\boldsymbol\theta^\top\mathbf x_i)\big)\Big]\,}$$

For each example only one term is "on": if $y_i=1$ the loss is $-\log p$, where $p$ is the predicted probability of 1; if $y_i=0$ it is $-\log(1-p)$. The loss measures **how surprised the model was by the truth**:

| truth is 1, model says $p=$ | loss $-\log p$ |
|---|---|
| 0.9 (confident, right) | 0.105 |
| 0.7 | 0.357 |
| 0.5 (shrug) | 0.693 |
| 0.1 (confident, wrong) | 2.303 |
| 0.01 (very confident, wrong) | 4.605 |

Being confidently wrong is punished hardest.

*Running example, a baseline.* With $\boldsymbol\theta=\mathbf 0$ every student gets $p=0.5$, so $J=\log2=0.6931$ — the loss of "I have no idea". Any useful model must beat this.

### 8.6 Gradient descent

**Plain words.** Imagine the loss as a landscape and $\boldsymbol\theta$ as your position. The **gradient** $\nabla J$ points *uphill*. Take a small step the opposite way, repeat.

$$\boldsymbol\theta\leftarrow\boldsymbol\theta-\alpha\,\nabla J(\boldsymbol\theta)\quad\text{repeated until a termination condition.}$$

$\nabla J=[\partial J/\partial\theta_0,\ \partial J/\partial\theta_1]^\top$ collects one slope per parameter; $\alpha$ is the **learning rate** (step size): too big overshoots the valley, too small takes forever. The minus sign is there because we *minimize* $J$.

**The gradient of BCE is pleasantly simple:**
$$\nabla J(\boldsymbol\theta)=\frac1D\sum_{i=1}^D\big(\sigma(\boldsymbol\theta^\top\mathbf x_i)-y_i\big)\,\mathbf x_i .$$
*(prediction − truth) × input, averaged.* $\sigma(\boldsymbol\theta^\top\mathbf x_i)-y_i$ is a single number per example (the slide's **scalar prediction error**); $\mathbf x_i$ is the feature vector. Where the model is already right the term is $\approx0$.

**Derivation (one example).** With $z=\boldsymbol\theta^\top\mathbf x$ and $\sigma'(z)=\sigma(z)(1-\sigma(z))$:
$\frac{\partial}{\partial\boldsymbol\theta}\big[y\log\sigma+(1-y)\log(1-\sigma)\big]=\big[y(1-\sigma)-(1-y)\sigma\big]\mathbf x=(y-\sigma)\mathbf x$. Negate and average to get the formula above.

In NumPy, with `X` of shape `(D, n)` and `y` of shape `(D,)`, there is no loop:

```python
p    = 1 / (1 + np.exp(-X @ theta))   # (D,)  predicted probabilities
grad = X.T @ (p - y) / D              # (n,)  the gradient
theta = theta - alpha * grad          # one step downhill
```

The lecture's pictures: a **contour map of $J(\theta_0,\theta_1)$** (dark = low loss) with the path of $\boldsymbol\theta$ sliding downhill along $-\nabla J$, and a plot of the sigmoid curve steepening toward the data at steps 0, 100, 1000, 10,000 and 100,000. It is an *approximation algorithm*, so many steps may be needed.

#### 8.6.1 Traced run on the students

Data: $\mathbf x_i=[1,h_i]$, targets $t=[1,0,1,0,0,0,1,0,1,0]$, start $\boldsymbol\theta=\mathbf 0$, $\alpha=0.1$.

* **Step 1.** All $p_i=0.5$. $\sum_i(0.5-t_i)=5-4=1$ and $\sum_i(0.5-t_i)h_i=0.5\cdot41-(4+6+7+8)=-4.5$. Gradient $=[0.1,\,-0.45]$. Update: $\boldsymbol\theta=[0,0]-0.1\cdot[0.1,-0.45]=[-0.01,\,0.045]$. New loss $0.6769$.

| step | $\boldsymbol\theta$ | loss $J$ |
|---|---|---|
| 0 | [0, 0] | 0.6931 |
| 1 | [−0.0100, 0.0450] | 0.6769 |
| 2 | [−0.0243, 0.0668] | 0.6711 |
| 3 | [−0.0405, 0.0785] | 0.6673 |
| 4 | [−0.0574, 0.0857] | 0.6640 |
| 5 | [−0.0747, 0.0909] | 0.6607 |
| … 100,000 | [−6.6836, 1.4495] | **0.26438** (gradient = 0) |

Progress is slow at $\alpha=0.1$ (the landscape is a long, narrow valley — see the deep dive). But the direction is right: the weight on hours grows positive ("more hours → more likely to pass").

* **Final model:** $\sigma(-6.6836+1.4495\,h)$. The **decision boundary** where $p=0.5$ is at $h^*=6.6836/1.4495=4.611$ hours.

| student | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| hours | 4 | 2 | 6 | 1 | 3 | 5 | 7 | 3 | 8 | 2 |
| passed $t$ | 1 | 0 | 1 | 0 | 0 | 0 | 1 | 0 | 1 | 0 |
| $p$ | 0.292 | 0.022 | 0.882 | 0.005 | 0.088 | 0.637 | 0.970 | 0.088 | 0.993 | 0.022 |
| loss | 1.231 | 0.022 | 0.125 | 0.005 | 0.092 | 1.014 | 0.031 | 0.092 | 0.007 | 0.022 |

The two surprises are exactly the two misclassified students: student 1 (studied 4 h, passed, model said 29%) and student 6 (5 h, failed, model said 64%). Their losses (1.23, 1.01) make up 85% of the total loss $2.644$.

### 8.7 From probability to decision: the threshold $\tau$

The model outputs a probability; to answer yes/no choose a **threshold** $\tau\in[0,1]$: predict 1 if $\sigma(\boldsymbol\theta^\top\mathbf x)\ge\tau$. $\tau=0.5$ is the default but a *choice*.

**The confusion matrix** (what the test set counts):

| | predicted 1 | predicted 0 |
|---|---|---|
| **actually 1** | TP | FN |
| **actually 0** | FP | TN |

* **Accuracy** $=\dfrac{TP+TN}{TP+TN+FP+FN}$ — misleading when one class is rare.
* **Precision** $=\dfrac{TP}{TP+FP}$ — *when I say yes, how often am I right?*
* **Recall** (true-positive rate, TPR) $=\dfrac{TP}{TP+FN}$ — *of all real yeses, how many did I catch?*
* **FPR** $=\dfrac{FP}{FP+TN}$ — fraction of real noes that I wrongly flagged.

*Running example (the fitted model, 4 real passes and 6 fails):*

| $\tau$ | TP | FP | FN | TN | accuracy | precision | recall | FPR |
|---|---|---|---|---|---|---|---|---|
| 0.2 | 4 | 1 | 0 | 5 | 0.9 | 0.80 | 1.00 | 0.167 |
| 0.35 | 3 | 1 | 1 | 5 | 0.8 | 0.75 | 0.75 | 0.167 |
| 0.5 | 3 | 1 | 1 | 5 | 0.8 | 0.75 | 0.75 | 0.167 |
| 0.7 | 3 | 0 | 1 | 6 | 0.9 | 1.00 | 0.75 | 0 |
| 0.9 | 2 | 0 | 2 | 6 | 0.8 | 1.00 | 0.50 | 0 |

Lowering $\tau$ → you say "yes" more often → recall up, precision down. Raising it does the reverse. A baseline "everyone fails" has accuracy $0.6$ but recall $0$.

**ROC and PR curves.** Sweep $\tau$ over every value and plot **TPR against FPR** (the **ROC curve**) or **precision against recall** (the **PR curve**). A perfect classifier hugs the top-left of the ROC plot (the best $\tau^*$ sits at that corner); a random classifier lies on the diagonal. For the PR curve we want it to stay near the top-right. Which $\tau$ is right depends on which error costs more (a missed cancer vs a false alarm). *Running example:* ranking the 4 passes against the 6 fails, 23 of the 24 (pass, fail) pairs are ordered correctly, so $\text{AUC}=23/24=0.958$ (only student 1 is ranked below student 6).

### 8.8 How good is the model? (honest evaluation)

The slide recalls the picture from Lecture 03: measurements are random variables from an underlying process we cannot see, and our model may mismatch it. So evaluate like this:

1. Obtain a **test dataset**, independently sampled from the same process.
2. **Predict labels** for the test set.
3. **Score**: the **log-likelihood** (the same quantity we trained on), and/or **accuracy, precision and recall**.

(The slide's precision/recall picture shows relevant vs retrieved elements: green = true positives, red = false positives; precision = fraction of retrieved items that are relevant, recall = fraction of relevant items that are retrieved.) Splitting data into train/test in several ways is **cross-validation** — Lecture 10.

> ⚠️ Our ten-student numbers above are scored on the **training** data, so they are optimistic. Lecture 10 shows what honest scoring looks like.

#### 📖 Textbook deep dive

##### Murphy §2.4.2 (p.50) — facts about the sigmoid
* $\sigma'(a)=\sigma(a)\,(1-\sigma(a))$ — the derivative reuses the output (that is why the gradient is so tidy).
* $1-\sigma(a)=\sigma(-a)$ — flipping the sign swaps the classes.
* The inverse of $\sigma$ is the **logit** (log-odds): $\sigma^{-1}(p)=\log\frac p{1-p}$.
*Check:* $\log(0.8/0.2)=1.3863$ and $\sigma(1.3863)=0.8000$.

So the model says: **the log-odds of passing is a linear function of hours**: $\log\frac{p}{1-p}=-6.6836+1.4495\,h$. Each extra hour multiplies the odds by $e^{1.4495}=4.26$.

##### Murphy §2.4.3 (p.52) and §10.2 (p.339) — binary logistic regression
$p(y|\mathbf x,\mathbf w)=\text{Ber}(y\mid\sigma(\mathbf w^\top\mathbf x+b))$. The decision boundary $\mathbf w^\top\mathbf x+b=0$ is a straight line/plane (**linear classifier**); the sigmoid only controls how fast the probability changes across it.

##### Murphy §10.2.3 (p.342) — MLE, gradient, Hessian, convexity
With $\mu_n=\sigma(\mathbf w^\top\mathbf x_n)$, the average NLL is the BCE of §8.5. Gradient (Murphy Eq. 10.21, derived step by step on p.343) and Hessian (§10.2.3.4, p.344):
$$\nabla_{\mathbf w}\text{NLL}=\frac1N\sum_n(\mu_n-y_n)\mathbf x_n,\qquad \mathbf H=\frac1N\mathbf X^\top\mathbf S\mathbf X,\ \ \mathbf S=\text{diag}\big(\mu_n(1-\mu_n)\big).$$
Because $\mu_n\in(0,1)$ every entry of $\mathbf S$ is positive, so $\mathbf v^\top\mathbf H\mathbf v=\frac1N\|\mathbf S^{1/2}\mathbf X\mathbf v\|^2\ge0$: the NLL is **convex** (a bowl with no local minima that are not global). That is why gradient descent is guaranteed to find the best answer rather than a bad local one. Murphy adds that if $\mu_n\to0$ or $1$ the Hessian becomes nearly singular, which $\ell_2$ regularization (MAP estimation, §10.2.7, p.348) fixes.

*Running example:* at $\mathbf w^*$ the Hessian has eigenvalues $0.006$ and $1.709$, both positive ✔ — but their ratio ($\approx285$) explains why GD at $\alpha=0.1$ crawls: stability needs $\alpha<2/\lambda_{\max}=1.17$, yet progress along the flat direction scales with $\lambda_{\min}=0.006$.

##### Murphy §10.2.4 (p.345) — SGD
Use one example (or a small minibatch) per step instead of all $D$: $\mathbf w\leftarrow\mathbf w-\alpha(\mu_n-y_n)\mathbf x_n$. *Traced:* from $\mathbf w=\mathbf 0$, student 1 ($\mathbf x=[1,4]$, $y=1$, $\mu=0.5$): $\mathbf w=-0.1\cdot(0.5-1)\cdot[1,4]=[0.05,\,0.2]$. Cheap and noisy; the preferred method for huge datasets.

##### Murphy §10.2.5 (p.346) — the perceptron
Replace the sigmoid by a hard threshold $\hat y=\mathbb I(\mathbf w^\top\mathbf x+b>0)$ and update: $\mathbf w\leftarrow\mathbf w-\eta(\hat y_n-y_n)\mathbf x_n$ (Murphy Eq. 10.30). If the prediction is right nothing changes; if $y_n=1,\hat y_n=0$ then $\mathbf w\leftarrow\mathbf w+\mathbf x_n$; if $y_n=0,\hat y_n=1$ then $\mathbf w\leftarrow\mathbf w-\mathbf x_n$. It is exactly the SGD update of logistic regression with the soft probability $\mu_n$ replaced by the hard label $\hat y_n$. Advantage: no probabilities needed. Disadvantage: it **converges only if the data are linearly separable**, whereas SGD on the logistic NLL always converges to the global optimum.

##### Murphy §10.2.6 (p.346) — Newton's method / IRLS
Use curvature: $\mathbf w\leftarrow\mathbf w-\mathbf H^{-1}\mathbf g$. Each step solves a *weighted least-squares* problem with weights $\mu_n(1-\mu_n)$, hence **iteratively reweighted least squares (IRLS)**. Bishop §4.3.3 (p.207) derives the same update.

*Traced on the students* (start at $\mathbf 0$): first step $\mathbf H=0.025\,\mathbf X^\top\mathbf X$, $\mathbf g=[0.1,-0.45]$ ⇒ $\Delta=-\mathbf H^{-1}\mathbf g$ gives $\mathbf w=[-3.2843,\,0.7035]$ (loss 0.3268).

| Newton step | $\mathbf w$ | loss |
|---|---|---|
| 1 | [−3.2843, 0.7035] | 0.32684 |
| 2 | [−5.0938, 1.0978] | 0.27461 |
| 3 | [−6.2557, 1.3545] | 0.26501 |
| 4 | [−6.6487, 1.4417] | 0.26439 |
| 5 | [−6.6833, 1.4494] | 0.26438 |
| 6 | [−6.6836, 1.4495] | 0.26438 |

Six Newton steps do what 100,000 GD steps do — at the price of forming and solving with a Hessian each step (expensive for large models).

##### Murphy §10.2.8 (p.350) — standardize your inputs
Subtract the mean and divide by the standard deviation of each feature before fitting. It makes the loss landscape rounder (smaller condition number), so GD with one learning rate works well.

##### Murphy §5.1.2–5.1.4 (p.169–176) — decision theory for classifiers
* **Cost-sensitive threshold (p.169–170).** With losses $\ell_{ij}$, choose $\hat y=0$ iff $\ell_{00}p_0+\ell_{10}p_1<\ell_{01}p_0+\ell_{11}p_1$. If $\ell_{00}=\ell_{11}=0$ and a false negative costs $c$ times a false positive, predict 1 iff $p_1\ge\frac{1}{1+c}$. For $c=2$ the threshold is $\tfrac13$ (Murphy's example). *On the students:* with $\tau=1/3$ the decisions equal those at $\tau=0.5$ (student 1's $p=0.292<1/3$ stays "fail"); with $c=4$ ($\tau=0.2$) student 1 is flagged, giving the first row of the table above: recall 1.0.
* **ROC (§5.1.3, p.171):** TPR vs FPR; **AUC** = area under it = probability a random positive is ranked above a random negative; **EER** = point where FPR = FNR (our $\text{AUC}=0.958$).
* **PR curves and F-scores (§5.1.4, p.174):** useful with **class imbalance**, since FPR hides how many false positives there are relative to few positives. $F_1=\dfrac{2PR}{P+R}$ is the **harmonic mean** of precision $P$ and recall $R$ — it is low unless *both* are high. $F_\beta$ weights recall $\beta$ times as much as precision. *Check:* $\tau=0.5$: $F_1=2(0.75)(0.75)/1.5=0.75$; $\tau=0.7$: $2(1)(0.75)/1.75=0.857$.

##### Bishop §4.3 (p.203–209) — discriminative vs generative
* **Logistic regression (§4.3.2, p.205)** has only $M$ adjustable parameters for $M$ features, whereas the generative alternative (two Gaussians sharing one covariance, fitted by MLE, as in Assignment 1's Gaussian) needs $2M$ for the means, $M(M+1)/2$ for the covariance and 1 for the prior: $M(M+5)/2+1$ — quadratic in $M$. For $M=784$ pixels that is $784\cdot789/2+1=309{,}289$ parameters versus only 784 for logistic regression.
* **Separable data (p.206).** If a line separates the classes perfectly, ML pushes $\|\mathbf w\|\to\infty$: the sigmoid becomes a step, reaching likelihood 1 at the cost of absurd weights, and the "MLE" doesn't exist. Remedy: regularization. *(While designing this running example, an earlier version of the hours data was almost separable and Newton's weights grew without bound; the data above overlap, so the MLE is finite.)*
* **IRLS (§4.3.3, p.207):** Newton–Raphson on the cross-entropy; the weighted-least-squares form is the same as Murphy's.
* Bishop's concavity wording on p.206 ("the error function is concave… unique minimum") is a slip — see Errata; the error function is **convex**.

##### Solved exercise (Murphy §10.6, Ex 10.3, p.368–369)
Compare training conditional log-likelihoods $L(M)=\frac1n\sum\log p(y_i|\mathbf x_i,\hat\theta,M)$:

* (a) **GaussI vs LinLog:** $L(\text{GaussI})\le L(\text{LinLog})$. GaussI (two Gaussians, identity covariance) implies a posterior of logistic-of-linear form, i.e. it is one particular setting of LinLog's parameters. LinLog maximizes the conditional likelihood over *all* linear settings, so it can do no worse on the training set.
* (b) **GaussX vs QuadLog:** $L(\text{GaussX})\le L(\text{QuadLog})$ for the same reason (Gaussians with free covariances give a logistic of a quadratic).
* (c) **LinLog vs QuadLog:** $\le$ — linear features are a special case of linear+quadratic features.
* (d) **GaussI vs QuadLog:** $\le$, by chaining (a) with (c).
* (e) **Is $L(M)>L(M')\Rightarrow R(M)<R(M')$?** **No.** Log-likelihood rewards calibrated *probabilities*; misclassification rate only looks at which side of 0.5 you are on. A model can raise likelihood (by being more confident on cases it gets right) while making the same or more errors.

#### ➕ Textbook extra — Bayesian logistic regression and the probit approximation
Murphy §10.5 (p.363–367) shows predictions averaged over a posterior on $\mathbf w$ give *less extreme* probabilities: $p(y=1|\mathbf x)\approx\sigma(\kappa(v)m)$ with $\kappa(v)=(1+\pi v/8)^{-1/2}<1$, but with the **same decision boundary**. Beyond the course schedule; mentioned because "confidence" vs "correctness" returns in Lecture 10.

#### ✅ Key takeaways

* **Sigmoid** squashes a score to $(0,1)$; $p(y{=}1|\mathbf x)=\sigma(\boldsymbol\theta^\top\mathbf x)$ — a Bernoulli whose bias depends on $\mathbf x$.
* The loss is **binary cross-entropy** = negative average log-likelihood; confident wrong answers cost the most.
* Gradient $\frac1D\sum(\sigma(\boldsymbol\theta^\top\mathbf x_i)-y_i)\mathbf x_i$; update $\boldsymbol\theta\leftarrow\boldsymbol\theta-\alpha\nabla J$. BCE is **convex**, so GD reaches the global optimum (here $J=0.264$, boundary at 4.61 h).
* The probability becomes a decision via a **threshold $\tau$**; **precision/recall/ROC/PR** describe the trade-off; accuracy alone misleads under imbalance.
* Score on a **held-out test set**, not the training set.

---


## Lecture 09 — Softmax Regression

**Why this lecture exists.** Logistic regression answers yes/no. Many questions have *several* answers: which digit (0–9) is in this image? Softmax regression extends logistic regression to $C$ classes and introduces **cross-entropy**, the loss used by almost every classifier (including large neural networks).

### 9.1 The problem

Dataset $\mathcal D=\{(\mathbf x,y)\mid y\in\{0,\dots,C-1\}\}$: $\mathbf x$ is the input (the lecture's example: an image, a matrix of pixels) and $y$ its **class label**; the lecture's picture shows handwritten digits labelled 0,…,9, so $C=10$. As always the examples are IID, so
$$P(y_1,\dots,y_D\mid\mathbf x_1,\dots,\mathbf x_D;\boldsymbol\theta)=\prod_{i=1}^{D}P(y_i\mid\mathbf x_i;\boldsymbol\theta).$$

*Running example.* The 3-class label $y_3$: 0 = fail ($s<65$), 1 = pass (65–74), 2 = distinction ($\ge75$). Labels $[1,0,1,0,0,0,2,0,2,0]$; counts $[6,2,2]$. Features $\mathbf x=[1,h]$ as before.

### 9.2 The big idea: one score per class

Suppose a model $f_{\boldsymbol\theta}$ (the lecture's example is a **convolutional neural network**) maps an input to a **vector of scores**
$$\mathbf z=f_{\boldsymbol\theta}(\mathbf x)\in\mathbb R^{C},$$
one number per class (10 numbers for digits). Scores can be any real numbers ($-3$, $12$, …), but a categorical distribution needs a **probability vector**: all entries $\ge0$ and adding up to 1. How do we turn scores into probabilities?

For a linear softmax regression $\mathbf z=\mathbf W\mathbf x$ (or $\mathbf W^\top\mathbf x$) with one weight vector per class; the CNN is just a more powerful $f_{\boldsymbol\theta}$.

### 9.3 The probability simplex

The set of all valid probability vectors has a name:
$$\Delta_n=\Big\{\mathbf p\in\mathbb R^n\ \Big|\ p[i]\ge0\ \forall i,\ \sum_ip[i]=1\Big\}.$$
For three classes $(0.5,0.3,0.2)\in\Delta_3$; $(0.5,0.9,-0.4)\notin\Delta_3$ (negative entry and the sum is 1.0 only by accident). To use $f_{\boldsymbol\theta}$ inside the likelihood its output must land in $\Delta_C$.

### 9.4 The softmax function

$$s(\mathbf x)[i]=\frac{e^{x[i]}}{\sum_{j=1}^{C}e^{x[j]}},\qquad s:\mathbb R^C\to\Delta_C .$$

Two steps: (1) $e^{x}$ makes every score **positive** and exaggerates the leader (the lecture plots $e^x$); (2) dividing by the total makes the entries **sum to 1**. In NumPy, as on the slide:

```python
x = ...                              # shape (C,)
p_exp = np.exp(x)
p_exp = p_exp / np.sum(p_exp)        # this is softmax
```

**Worked example, scores $(2,1,0)$:**

| class | $e^{\text{score}}$ | ÷ total ($11.107$) |
|---|---|---|
| 0 | 7.389 | **0.6652** |
| 1 | 2.718 | **0.2447** |
| 2 | 1.000 | **0.0900** |

They sum to 1 and the largest score gets the largest probability. **Only score differences matter:** adding 100 to every score changes nothing, and equal scores give equal probabilities — softmax$(5,5,5)=(\tfrac13,\tfrac13,\tfrac13)$.

> **Connection to Lecture 08.** With two classes, $\dfrac{e^a}{e^a+e^b}=\sigma(a-b)$. Check: $a=1.3,\ b=-0.4$ gives $\sigma(1.7)=0.8455$ both ways. Logistic regression *is* softmax regression with $C=2$.

### 9.5 The likelihood of one label

Reuse the Bernoulli trick with an **indicator**:
$$P(y_i\mid\mathbf x_i;\boldsymbol\theta)=\prod_{c=1}^{C}\Big(s\big(f_{\boldsymbol\theta}(\mathbf x_i)\big)[c]\Big)^{\mathbb 1[c=y_i]},\qquad \mathbb 1[c=y_i]=\begin{cases}1&c=y_i\\0&\text{otherwise.}\end{cases}$$
(The slide writes the Bernoulli formula $\theta^x(1-\theta)^{1-x}$ beside it.) Every wrong class is raised to the power 0 (it becomes 1 and vanishes); only the **probability of the correct class** survives — a "pick the right entry" switch.

Taking logs (with $\log(ab)=\log a+\log b$ and $\log a^c=c\log a$):
$$\log P(y_1..y_D\mid\mathbf x_1..\mathbf x_D;\boldsymbol\theta)=\sum_{i}\sum_{c}\mathbb 1[c=y_i]\,\log s\big(f_{\boldsymbol\theta}(\mathbf x_i)\big)[c].$$
The slide notes that $\log s(f_{\boldsymbol\theta}(\mathbf x))[c]$ is the **log probability predicted by the model**, and the whole expression **computes the negative cross-entropy** between labels and predicted distribution.

### 9.6 Cross-entropy

For two probability vectors $\mathbf p$ (truth) and $\mathbf q$ (prediction):
$$H(\mathbf p,\mathbf q)=-\sum_ip[i]\log q[i].$$
When $\mathbf p$ is **one-hot** (all mass on the correct class) this is just $-\log q[y]$. The **cross-entropy loss** is
$$\boxed{\,J(\boldsymbol\theta)=-\frac1D\sum_{i=1}^{D}\sum_{c}\mathbb 1[c=y_i]\log s\big(f_{\boldsymbol\theta}(\mathbf x_i)\big)[c]\,}$$
*In words:* **loss = $-\log$(probability the model gave the correct class)**, averaged over the data — the same "surprise" idea as Lecture 08. Binary cross-entropy is the $C=2$ case.

**Worked example.** Scores $\mathbf z=[0.5,\,1.5,\,-1.0]$ ⇒ $\mathbf p=[0.2537,\,0.6897,\,0.0566]$. If the true class is 1 the loss is $-\log0.6897=0.3715$; if the true class is 0 it is $-\log0.2537=1.3715$ (exactly 1 more).

**Reference points on the running example** (3 classes, counts $[6,2,2]$):
* Always predicting uniform $(\tfrac13,\tfrac13,\tfrac13)$: $J=\log3=1.0986$.
* Always predicting the class frequencies $(0.6,0.2,0.2)$: $J=0.9503$ nats ($=1.3710$ bits, the entropy of the labels — Lecture 12).
Any useful model must beat these.

### 9.7 Finding $\boldsymbol\theta$

The MLE is $\boldsymbol\theta^*\in\arg\min_{\boldsymbol\theta}J(\boldsymbol\theta)$. As the slide says, it can be found with **gradient descent** *if $f_{\boldsymbol\theta}$ is differentiable*. To predict a class, take the largest probability: $\hat y=\arg\max_c p_c$.

**Gradient for linear softmax regression** (derived in the deep dive): for $\mathbf z=\mathbf W^\top\mathbf x$,
$$\nabla_{\mathbf w_c}J=\frac1D\sum_i\big(p_{ic}-\mathbb 1[c=y_i]\big)\mathbf x_i,$$
the same "(prediction − truth) × input" as logistic regression, one weight vector per class.

*Traced first step on the students* ($\mathbf W=\mathbf 0$ so all $p=\tfrac13$, $\alpha=0.1$):
$$\mathbf G=\frac1{10}\mathbf X^\top(\mathbf P-\mathbf Y)=\begin{bmatrix}-0.2667&0.1333&0.1333\\-0.2333&0.3667&-0.1333\end{bmatrix}\ \ (\text{rows: intercept, hours; columns: class 0,1,2}),$$
$$\mathbf W_1=-0.1\,\mathbf G=\begin{bmatrix}0.0267&-0.0133&-0.0133\\0.0233&-0.0367&0.0133\end{bmatrix},\qquad J: 1.0986\to1.0759 .$$
Reading it: more hours push class 2 (distinction) up and class 0 (fail) down — the pattern in the data.

**A warning about this dataset.** Both distinction students (7 h and 8 h) are *separated* from all others by hours alone, so the **unregularized MLE does not exist** — the weights can grow forever while the loss keeps shrinking (BFGS ran to weights of size 150). (This is the same "separable data ⇒ $\mathbf w\to\infty$" effect as in Lecture 08.) With a tiny $\ell_2$ penalty ($10^{-4}\|\mathbf W\|^2$ — regularization, week 6 of the schedule) we get a sensible fit, $\mathbf W=\begin{bmatrix}10.36&4.50&-14.86\\-1.83&-0.59&2.42\end{bmatrix}$:

| student | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| hours | 4 | 2 | 6 | 1 | 3 | 5 | 7 | 3 | 8 | 2 |
| true class | 1 | 0 | 1 | 0 | 0 | 0 | 2 | 0 | 2 | 0 |
| $p_0$ | 0.709 | 0.967 | 0.138 | 0.990 | 0.894 | 0.411 | 0.009 | 0.894 | 0.000 | 0.967 |
| $p_1$ | 0.290 | 0.033 | 0.675 | 0.010 | 0.106 | 0.581 | 0.149 | 0.106 | 0.009 | 0.033 |
| $p_2$ | 0.000 | 0.000 | 0.188 | 0.000 | 0.000 | 0.008 | 0.843 | 0.000 | 0.991 | 0.000 |
| predicted | 0 ✗ | 0 | 1 | 0 | 0 | 1 ✗ | 2 | 0 | 2 | 0 |

8 of 10 correct; the two errors are the same students (1 and 6) that logistic regression got wrong — they are the "inconsistent" ones in the data.

### 9.8 Numerical safety: log-sum-exp

$e^{1000}$ overflows to `inf`, so `np.exp(x)/np.exp(x).sum()` returns `nan` for large scores. Because softmax is unchanged by subtracting a constant, **subtract the maximum first**:
```python
def softmax(x):
    z = x - x.max()          # largest entry becomes 0 -> e^0 = 1, no overflow
    e = np.exp(z)
    return e / e.sum()
softmax(np.array([1000., 1001., 1000.]))     # [0.2119, 0.5761, 0.2119]
```
(The slide's three-line version is fine for small scores.)

#### 📖 Textbook deep dive

##### Murphy §2.5 (p.53–56) — categorical distribution and softmax
* $\text{Cat}(y|\boldsymbol\theta)=\prod_c\theta_c^{\mathbb I(y=c)}$ — the slide's indicator formula. Written in one-hot form $\mathbf y$, $\prod_c\theta_c^{y_c}$.
* **Softmax** $\mathcal S(\mathbf a)_c=e^{a_c}/\sum_{c'}e^{a_{c'}}$; the inputs $\mathbf a$ are called **logits**. It is a "soft" argmax: it is smooth, positive, and differentiable.
* **Temperature (§2.5.2, p.54).** Dividing the logits by $T$ gives $\mathcal S(\mathbf a/T)$. As $T\to\infty$ the output tends to uniform; as $T\to0$ it tends to one-hot (hard argmax). *Check with $\mathbf a=(3,0,1)$:* $T=100\to[0.339,0.329,0.332]$; $T=2\to[0.629,0.140,0.231]$; $T=1\to[0.844,0.042,0.114]$.
* **Log-sum-exp (§2.5.4, p.56).** $\log\mathcal S(\mathbf a)_c=a_c-\text{lse}(\mathbf a)$ with $\text{lse}(\mathbf a)=\log\sum_ce^{a_c}=m+\log\sum_ce^{a_c-m}$, $m=\max_ca_c$. *Check:* $\text{lse}(0,1,0)=\log4.7183=1.5514$; and for $\mathbf a=(1000,1001,1000)$ the shifted form gives $1001+\log(0.3679+1+0.3679)=1001.55$. Software exposes this as "cross-entropy with logits" (e.g. `torch.nn.CrossEntropyLoss`) to avoid ever forming the probabilities (p.354, Eq. 10.73).

##### Murphy §10.3 (p.350–359) — multinomial logistic regression
* Model $p(y=c|\mathbf x,\mathbf W)=\mathcal S(\mathbf W\mathbf x)_c$; the NLL is exactly the cross-entropy of §9.6 (Murphy writes it $-\frac1N\sum_n\sum_cy_{nc}\log\mu_{nc}$).
* **Redundancy.** Adding the same vector to all class weight vectors leaves the probabilities unchanged (softmax ignores a common shift). So the weights are not unique; one can pin one class to $\mathbf 0$ (as logistic regression implicitly does, with $C=2$ and a single weight vector).
* **Gradient (§10.3.2.3):** $\nabla_{\mathbf w_c}\text{NLL}=\frac1N\sum_n(\mu_{nc}-y_{nc})\mathbf x_n$.
* **Hessian (§10.3.2.4, p.353):** $\mathbf H=\frac1N\sum_n\big(\text{diag}(\boldsymbol\mu_n)-\boldsymbol\mu_n\boldsymbol\mu_n^\top\big)\otimes(\mathbf x_n\mathbf x_n^\top)$, a $CD\times CD$ matrix ($\otimes$ = Kronecker product). It is positive semi-definite, so the NLL is **convex** (Murphy says "positive definite"; strictly it is only semi-definite until the redundancy above is removed — see Errata).
* §10.3.3 (p.354): SGD works as before; second-order methods are expensive, so quasi-Newton (L-BFGS) is common. §10.3.4: "bound optimization" with a fixed matrix instead of the Hessian.

##### Bishop §4.3.4 (p.209) — multiclass logistic regression
Bishop derives softmax ("normalized exponential") from Bayes' rule on class-conditional densities, uses **1-of-$K$ coding** for targets, and obtains the gradient $\sum_n(y_{nj}-t_{nj})\boldsymbol\phi_n$ — identical in form to the binary case — with Hessian blocks $\sum_ny_{nk}(I_{kj}-y_{nj})\boldsymbol\phi_n\boldsymbol\phi_n^\top$.

##### Solved exercise (Murphy Ex 10.1, p.367–368)
(a) **Jacobian.** $\mu_k=e^{\eta_k}/\sum_je^{\eta_j}$. For $j=k$: $\partial\mu_k/\partial\eta_k=\mu_k-\mu_k^2=\mu_k(1-\mu_k)$; for $j\ne k$: $\partial\mu_k/\partial\eta_j=-\mu_k\mu_j$. Together $\ \partial\mu_{k}/\partial\eta_{j}=\mu_k(\delta_{kj}-\mu_j)$.

(b) **Gradient.** $\text{NLL}=-\sum_i\sum_ky_{ik}\log\mu_{ik}$. Then $\dfrac{\partial\text{NLL}}{\partial\eta_{ij}}=-\sum_k\dfrac{y_{ik}}{\mu_{ik}}\mu_{ik}(\delta_{kj}-\mu_{ij})=-\Big(y_{ij}-\mu_{ij}\sum_ky_{ik}\Big)=\mu_{ij}-y_{ij}$, using $\sum_ky_{ik}=1$. With $\eta_{ij}=\mathbf w_j^\top\mathbf x_i$ the chain rule gives $\nabla_{\mathbf w_c}\text{NLL}=\sum_i(\mu_{ic}-y_{ic})\mathbf x_i$. (Murphy's printed gradient $\sum_i(y_{ic}-\mu_{ic})\mathbf x_i$ is the gradient of the *log-likelihood*, i.e. the negative of this.)

(c) **Hessian.** Differentiating again: block $(c,c')=\sum_i\mu_{ic}(\delta_{cc'}-\mu_{ic'})\mathbf x_i\mathbf x_i^\top$. For any stacked vector $\mathbf v=(\mathbf v_1..\mathbf v_C)$ with $a_c=\mathbf v_c^\top\mathbf x_i$: $\mathbf v^\top\mathbf H\mathbf v=\sum_i\big[\sum_c\mu_{ic}a_c^2-(\sum_c\mu_{ic}a_c)^2\big]=\sum_i\text{Var}_{c\sim\boldsymbol\mu_i}[a_c]\ge0$ — a variance, hence non-negative. It is zero when all $a_c$ coincide (e.g. $\mathbf v_1=\dots=\mathbf v_C$, the redundant direction), so the Hessian is positive **semi**-definite; once one class is pinned to zero it is positive definite (if $\mathbf X$ has full column rank). Either way the NLL is convex.

#### ➕ Textbook extra — generalized probit approximation and "bound optimization"
Murphy §10.5 (p.363–368) gives multiclass Bayesian predictions with $\exp(\kappa(v_c)m_c)$ in place of softmax — less confident but with the same arg-max. Not examined here; included only so you recognize "Bayesian logistic regression" when it appears in later weeks.

#### ✅ Key takeaways

* A model outputs **one score per class**; **softmax** turns scores into a probability vector in the simplex $\Delta_C$.
* The likelihood of one label is $\prod_c p_c^{\mathbb 1[c=y]}$ — only the correct class survives.
* **Cross-entropy loss** $=-\frac1D\sum_i\log p_{i,y_i}$; it is the negative log-likelihood. Binary cross-entropy = the $C=2$ case; softmax with 2 classes = sigmoid of the difference.
* Gradient $=(\text{prediction}-\text{one-hot})\times\text{input}$; fit with gradient descent whenever $f_{\boldsymbol\theta}$ is differentiable.
* Use **log-sum-exp / subtract the max** for numerical safety.
* Separable classes ⇒ the unregularized MLE does not exist; regularize.

---


## Lecture 10 — Generalization

**Why this lecture exists.** Training loss only says how well the model fits data it has *already seen*. What we care about is **new** data. This ability is **generalization**. Think of studying from past exams: scoring 100% on the questions you memorized proves nothing about the real exam.

### 10.1 Bias–variance trade-off: controlling model complexity

The lecture's first pictures (from Murphy 2022, Fig. 1.7 and Fig. 11.5) fit curves of different flexibility to the same points:

| | Too simple | Too complex |
|---|---|---|
| Flexibility | low | high |
| Behaviour | **high bias, low variance** (underfits) | **low bias, high variance** (overfits) |

* **Bias:** error from wrong or too-simple assumptions (a straight line forced through curved data).
* **Variance:** how much the fitted model would *change* if a different training set were drawn.
* The **error residual** is $e=f_{\boldsymbol\theta}(\mathbf x)-y$ (prediction minus truth). If residuals show a *pattern* instead of random scatter, the model is missing structure.

The gap between performance on training and on test data is the **generalization gap**.

**Worked example on the students (polynomial of degree $d$ in hours).**
Train on the first 7 students, test on the last 3 (hours 3, 8, 2; scores 59, 83, 56):

| degree $d$ | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| train MSE (7 students) | 7.30 | 7.10 | 6.71 | 4.58 | 2.49 |
| **test MSE** (3 students) | **14.89** | 16.07 | 10.62 | 177.7 | 392.1 |

Training error falls steadily as the model grows, but test error explodes for degrees 4–5: **overfitting**. (With 7 training points a degree-5 polynomial has 6 parameters — it nearly memorizes the training set.) Only 3 test points is a noisy measurement, which is exactly why we use cross-validation (§10.5).

#### 10.1.1 Making bias and variance concrete (simulation)

Suppose the truth is $f(h)=40+5h$ and each score is $f(h)$ plus Gaussian noise with $\sigma=3$. Generate 20,000 fresh sets of ten scores at the same ten hours, fit a degree-$d$ polynomial each time, and look at the prediction at $h=4.5$ (true value $62.5$):

| degree | bias² | variance | bias² + variance |
|---|---|---|---|
| 0 (constant) | 4.03 | 0.88 | 4.91 |
| 1 (line) | ≈ 0 | 0.91 | **0.91** |
| 2 | ≈ 0 | 2.26 | 2.26 |
| 5 | ≈ 0 | 4.08 | 4.08 |

Checks against theory: for a constant fit the variance is $\sigma^2/n=9/10=0.90$ (simulated 0.88) and the bias is $(40+5\bar h-62.5)^2=(60.5-62.5)^2=4.0$ (simulated 4.03); for a line the variance is $\sigma^2\big(\tfrac1n+\tfrac{(4.5-\bar h)^2}{S_{xx}}\big)=9(0.1+0.0033)=0.93$ (simulated 0.91). **The line matches the truth's shape, so it has no bias and the lowest variance; extra flexibility adds only variance.** Expected squared error at a point = bias² + variance + noise ($\sigma^2=9$ here).

### 10.2 Model selection

1. How can we select the best possible model?
2. Many ML algorithms have settings: the learning rate, the choice of basis functions, the polynomial degree, … (**hyperparameters**).
3. Usually many combinations are tried in parallel — **grid search**.
4. Model-selection algorithms exist but are often cumbersome to use.

### 10.3 The MLE is a point estimate: does it carry over to new data?

1. A dataset $\mathcal D$ is sampled from an overall data distribution $p_{\text{data}}(\mathcal D)$.
2. The MLE is computed: $\boldsymbol\theta_{\mathcal D}\in\arg\max_{\boldsymbol\theta}\mathcal L(\mathcal D\mid\boldsymbol\theta)$.
3. **Inference:** use $\boldsymbol\theta_{\mathcal D}$ to predict on a test dataset $\mathcal D_{\text{test}}$.

*Is $\boldsymbol\theta_{\mathcal D}$ also the best parameter for the test data, i.e. is $\boldsymbol\theta_{\mathcal D}\in\arg\max_{\boldsymbol\theta}\mathcal L(\mathcal D_{\text{test}}\mid\boldsymbol\theta)$?* Not necessarily — the MLE is best for the data it saw. So we evaluate on an **independently sampled test dataset**.

**Same-process requirement.** Training and test data must stem from the same sampling process: $\mathcal D_{\text{train}},\mathcal D_{\text{test}}\sim p_{\text{data}}$, all points IID. If the test data come from a *different* distribution, what we learned does not necessarily inform predictions: this is **out-of-distribution (OOD) generalization**. *Example:* a pass/fail model fitted on one course's students, applied to another course with a different exam.

### 10.4 Model evaluation: the quantity we would like

After training, $\boldsymbol\theta^*\in\arg\max$ (equivalently $\arg\min$ of the average loss over $\mathcal D_{\text{train}}$). We would like to know
$$\mathbb E_{\mathcal D_{\text{test}}\sim p_{\text{data}}}\Big[\frac1{|\mathcal D_{\text{test}}|}\sum_{(\mathbf x_i,y_i)\in\mathcal D_{\text{test}}}l\big(f_{\boldsymbol\theta}(\mathbf x_i),y_i\big)\Big],$$
an expectation over *all test datasets that could exist*. There are infinitely many, so we cannot compute it. Instead we **approximate by sampling different test datasets and averaging.**

### 10.5 Cross-validation (CV)

**Idea.** Split one dataset $\mathcal D$ into $K$ pieces (the lecture uses 5: $\mathcal D_1\dots\mathcal D_5$). Run $K$ experiments; in experiment $k$ train on the other four pieces and test on piece $k$, giving error $e_k$. **Report the average** $\bar e=\frac1K\sum_ke_k$. Compare models or hyperparameters by $\bar e$ and pick the best. **Finally retrain the best model with its best hyperparameters on the entire dataset $\mathcal D$** and use that for inference.

**Worked example on the students.** Five folds of two students each (students 1–2, 3–4, 5–6, 7–8, 9–10); per-fold test MSE for the line (degree 1): $17.24,\ 3.88,\ 15.76,\ 2.98,\ 13.28$ ⇒ mean $\bar e=10.63$.

| degree | 5-fold CV MSE | leave-one-out (LOO) CV MSE | train MSE |
|---|---|---|---|
| 1 | **10.63** | **11.76** | 8.29 |
| 2 | 15.27 | 16.14 | 8.27 |
| 3 | 12.86 | 14.59 | 6.71 |
| 4 | — | 48.56 | 6.70 |
| 5 | — | 81.67 | 5.81 |

**Decision:** the straight line is the best model (lowest CV error under both schemes) — even though degree 5 has the best *training* error. Having chosen degree 1, **retrain it on all ten students**: $\boldsymbol\theta=[41.81,\,5.02]$. The honest expected error on a new student is about $11$–$12$ (MSE), i.e. an RMSE of $\approx3.3$–$3.4$ points — notably more than the optimistic training $8.29$.

```python
def poly_fit(h, s, d):                       # least squares on [1, h, ..., h^d]
    P = np.vander(h, d + 1, increasing=True)
    return np.linalg.lstsq(P, s, rcond=None)[0]

def cv_mse(h, s, d, folds):
    errs = []
    for f in folds:                          # f = indices of the test fold
        m = np.ones(len(h), bool); m[f] = False          # boolean mask = training set
        th = poly_fit(h[m], s[m], d)
        pred = np.vander(h[f], d + 1, increasing=True) @ th
        errs.append(((s[f] - pred) ** 2).mean())
    return np.mean(errs)

folds = [[0,1],[2,3],[4,5],[6,7],[8,9]]
[round(cv_mse(h, s, d, folds), 2) for d in (1, 2, 3)]   # [10.63, 15.27, 12.86]
```

### 10.6 How to evaluate a model

1. It depends on the model and the experiment.
2. With maximum likelihood, **the loss evaluated on the test set is always a good start** (including the likelihood of generating the test set, $p(\mathcal D_{\text{test}}\mid\boldsymbol\theta)$).
3. The best model is chosen by **matching training and test errors as closely as possible**; a training error far below test error means **overfitting** (the lecture's high-degree polynomial).
4. In cross-validation the values are simply **averaged**.

#### 10.6.1 Models trained with MSE (regression)
With $\text{MSE}(\boldsymbol\theta|\mathcal D)=\frac1D\sum(y_i-f_{\boldsymbol\theta}(\mathbf x_i))^2$ we can inspect the **error residuals** and *how they are distributed*. They should look like the Gaussian noise the model assumed (Lecture 07). *Students:* the residuals $[4.10,-3.85,-0.94,-1.83,0.12,-4.92,0.03,2.12,1.01,4.15]$ are small and balanced around 0 — consistent with the assumption.

#### 10.6.2 Models trained with binary cross-entropy (logistic regression)
Outputs are 0/1. The slides show the precision/recall picture and say "Accuracy is the same as precision in this case" (this is imprecise — see Errata: accuracy is $(TP+TN)/\text{all}$, precision is $TP/(TP+FP)$). Other measures on the slide:

* $F_1=2\,\dfrac{\text{Precision}\times\text{Recall}}{\text{Precision}+\text{Recall}}$.
* $F_\beta=(1+\beta^2)\dfrac{\text{Precision}\times\text{Recall}}{\beta^2\,\text{Precision}+\text{Recall}}$ (the slide prints "$(1-\beta)$" — a typo, see Errata). $\beta>1$ weights recall more, $\beta<1$ precision more.
* **Fallout** (false-positive rate) $=\dfrac{FP}{FP+TN}$ (the slide's "False Positives / True Negatives" omits the $FP$ in the denominator — see Errata).
* **Sensitivity** $=\dfrac{TP}{TP+FN}$ (= recall; the slide calls these "less common").

**ROC and precision–recall curves.** The model outputs $p(y=1|\mathbf x,\boldsymbol\theta)$; sliding the threshold $\tau$ "artificially" makes you predict $y=1$ more or less often. *The purpose of the threshold is to control the false-positive and true-positive rates.* Plot TPR vs FPR (ROC) or precision vs recall (PR) over all $\tau$. (Numbers for the students: Lecture 08, §8.7.)

#### 10.6.3 Models trained with cross-entropy (softmax regression)
* **Accuracy** $=\dfrac{\#\text{ correct predictions}}{\text{test set size}}$.
* **Confusion matrix** (rows = actual class, columns = predicted class). The **normalized** version divides each row by the number of test examples in that class, so each row sums to 1 (slide example for 3 classes: row 0 = [0.9, 0.08, 0.02], row 1 = [0.1, 0.8, 0.1], row 2 = [0.05, 0.1, 0.85]). The **count** version reports raw counts (the lecture shows one for the **CIFAR-10** image dataset). It shows *which* classes get confused.

*Students (3 classes, regularized fit from Lecture 09):* true class 0 (6 students): 5 predicted 0, 1 predicted 1 → row $[0.833,0.167,0]$; true class 1 (students 1, 3): 1 predicted 0, 1 predicted 1 → $[0.5,0.5,0]$; true class 2 (students 7, 9): both predicted 2 → $[0,0,1]$. Overall accuracy $8/10$.

### 10.7 Bias and fairness

ML fits a dataset sampled from a distribution of datasets $\mathcal D\sim p_{\text{data}}(\cdot)$, so the model **learns the idiosyncrasies of that data**. The slides give four real examples (each with a link):

1. Amazon's hiring algorithm decided not to invite women to interview.
2. Google's online ad algorithm showed high-income jobs to men much more often than to women.
3. A machine-learning algorithm denied credit based on race or gender.
4. A health-care algorithm offered less care to Black patients.

*How can we factor this into the evaluation of our models?* A practical first step: report performance **per group**, not only on average. A model can have high average accuracy while failing a small group badly (e.g. overall accuracy $0.9$ when a group making up 10% of the data is 100% wrong).

### 10.8 Inductive bias

Every ML algorithm makes implicit assumptions — its **inductive bias**.

* **Occam's razor:** always use the simplest explanation of the data.
* Inductive biases can be good: our world is not random (the sun will rise tomorrow).
* Ask: *what are the inductive biases of the algorithm I currently use? Are they all clearly understood?*

*Examples from this course:* linear regression assumes the mean is a *line* in the features and noise is Gaussian; logistic regression assumes a *linear* decision boundary; IID sampling assumes train and test come from the same process.

### 10.9 No Free Lunch (Wolpert and Macready, 1997)

Consider the binary classification task (inputs $x_1,x_2,x_3\in\{0,1\}$, label $y$):

| $x_1$ | $x_2$ | $x_3$ | $y$ |
|---|---|---|---|
| 0 | 1 | 0 | 0 |
| 0 | 1 | 1 | 1 |
| 1 | 0 | 0 | 0 |
| 1 | 0 | 1 | 0 |

There are many logic functions that fit this data. Counting: there are $2^3=8$ possible inputs, so $2^8=256$ Boolean functions; the four rows pin down four outputs, leaving the other four inputs free, so **$2^4=16$ functions fit the data perfectly**. For example $f_1=x_2\wedge x_3$ and $f_2=\neg x_1\wedge x_3$ both fit, yet they disagree on the unseen input $(0,0,1)$: $f_1=0$, $f_2=1$. Which one a learner picks depends on its inductive bias. One learning algorithm may find a function fitting this data well while another may not; every ML (or AI) algorithm may do well on one dataset or problem and poorly on another:

> **There is no single ML/AI algorithm that performs well on all datasets.**

A method wins only when its assumptions match the problem.

### 10.10 Lecture summary (from the slides)

* Complex models can have very different training and test performance — the **generalization gap**.
* The **bias–variance trade-off**.
* What we care about is generalization: **overfitting** (good on training doesn't mean good on test) and **underfitting** (not even good on training).
* Performance is estimated with a **validation set** or **cross-validation**; analyze how well the model captures the data-generation process and whether its implicit bias limits performance.

#### 📖 Textbook deep dive

##### Murphy §1.2.3–1.2.4 (p.12–14) — overfitting, generalization, no free lunch
Murphy's polynomial example (Fig. 1.7) tells the same story as our students: degree $D=1$ underfits, $D=2$ is "just right", and as $D$ grows to $N-1$ (one parameter per data point) the model **interpolates the training data with 0 MSE but is too "wiggly"** to predict new inputs. His definitions: the **population risk** $\mathcal L(\boldsymbol\theta;p^*)=\mathbb E_{p^*(\mathbf x,y)}[\ell(y,f(\mathbf x;\boldsymbol\theta))]$ is the expected loss under the true (unknown) distribution; the **generalization gap** is $\mathcal L(\boldsymbol\theta;p^*)-\mathcal L(\boldsymbol\theta;\mathcal D_{\rm train})$. In practice we approximate the population risk by the **test risk** on held-out data; test error is U-shaped in model complexity (Fig. 1.7d). Important rule from p.13: *if you use the training set to compare models you will always pick the most complex one*, so partition the data into **three** sets — training (fit), validation (model selection) and test (estimating future performance; never used for fitting or selection).
§1.2.4 "No free lunch theorem" (p.13): there is no single best model for all kinds of problems, because a set of assumptions (the **inductive bias**) that works well in one domain may work poorly in another; the best way to choose is domain knowledge and/or trial and error such as cross-validation. (Murphy quotes George Box: *all models are wrong, but some models are useful.*)

##### Murphy §4.3 (p.115) — empirical risk minimization
Replace the log-loss in MLE by any loss $\ell$: $\mathcal L(\boldsymbol\theta)=\frac1N\sum_n\ell(y_n,\boldsymbol\theta;\mathbf x_n)$. MLE is ERM with $\ell=-\log p$; MSE is ERM with squared error; BCE with the Bernoulli log-loss.

##### Murphy §4.5.4–4.5.5 (p.124–126) — validation set and cross-validation
* **Validation set:** split into $\mathcal D_{\rm train}$ and $\mathcal D_{\rm valid}$ (often 80/20). For each candidate hyperparameter $\lambda$ fit on train, compute the **validation risk**, pick the best $\lambda^*$, then *refit on all data* — exactly the lecture's final step.
* **$K$-fold CV:** $R^{\rm cv}=\frac1K\sum_kR(\hat\theta(\mathcal D_{-k}),\mathcal D_k)$ (Fig. 4.6 shows $K=5$). If $K=N$ it is **leave-one-out CV**, used when data are scarce (our $N=10$: LOO).
* **One-standard-error rule (§4.5.5.1):** CV gives a point estimate but no uncertainty. Compute the mean $\hat\mu$ of per-example CV losses and its standard error $se=\hat\sigma/\sqrt N$; then pick the **simplest model whose CV risk is within one standard error of the best**. (Link to Lecture 11: the same SEM.) On the students the line is both the simplest and the best, so the rule changes nothing.
* **Early stopping (§4.5.6):** stop gradient descent when validation error starts to rise — another regularizer.

##### Murphy §4.7.6 (p.159) — bias–variance
For an estimator $\hat\theta$ of a true value $\theta^*$: $\text{bias}=\mathbb E[\hat\theta]-\theta^*$, variance $=\mathbb V[\hat\theta]$, and
$$\text{MSE}=\mathbb E[(\hat\theta-\theta^*)^2]=\text{variance}+\text{bias}^2 .$$
*Worked check — the variance MLE (Lecture 12):* for $N=5$, $\mathbb E[\hat\sigma^2_{\rm mle}]=\tfrac45\sigma^2$, so its bias is $-0.2\sigma^2$. Murphy's example (§4.7.6.4, p.161) estimates a Gaussian mean with a MAP estimate $\tilde x=w\bar x+(1-w)\theta_0$, $0\le w\le1$: its bias is $(1-w)(\theta_0-\theta^*)$ but its variance is only $w^2\sigma^2/N$, smaller than the MLE's $\sigma^2/N$ — so *accepting a little bias* can **lower** the total MSE.

##### Murphy §5.1.2–5.1.4 (p.169–176) and §5.4.3 (p.196) — metrics and model comparison
Confusion matrices, ROC/AUC, PR/F-scores: see Lecture 08 deep dive. Murphy's formula for $F_\beta$ is the standard $(1+\beta^2)PR/(\beta^2P+R)$, which confirms the slide's $(1-\beta)$ is a typo. §5.4.3 repeats the CV recipe.

##### Bishop §1.1 (p.4–12) — the polynomial curve-fitting story
Bishop fits a polynomial of order $M$ to 10 noisy points from $\sin(2\pi x)$. $M=0,1$ underfit; $M=3$ fits well; **$M=9$ passes through every point (zero training error) but oscillates wildly**. His Table 1.1 shows the coefficients exploding as $M$ grows: for $M=9$, $w_1^*=232.37$, $w_3^*=48{,}568.31$, $w_5^*=640{,}042.26$, $w_9^*=125{,}201.43$ — large opposite-sign weights that cancel at the training points. Remedies: more data (the same $M=9$ model is fine with $N=100$) or **regularization** (penalize $\|\mathbf w\|^2$; with $\ln\lambda=-18$ the weights drop to magnitudes under 100). He uses the **RMS error** $E_{\rm RMS}=\sqrt{2E(\mathbf w^*)/N}$ on separate training and test sets to see the U-shaped test curve.

##### Bishop §1.3 (p.32–33) — model selection
Cross-validation is "S-fold" ($S=N$ is leave-one-out). Drawbacks: the number of training runs grows by $S$, and with several hyperparameters the number of combinations explodes — Bishop's motivation for information criteria such as **AIC**, which choose the model maximizing $\ln p(\mathcal D|\mathbf w_{\rm ML})-M$ (log-likelihood minus the number of parameters $M$).

##### Bishop §3.2 (p.147–152) — bias–variance decomposition
Bishop shows the expected squared error splits into $(\text{bias})^2+\text{variance}+\text{noise}$, and illustrates with **100 data sets of 25 points** each from $\sin(2\pi x)$, fitted with 24 Gaussian basis functions and regularization $\ln\lambda=2.6,-0.31,-2.4$: large $\lambda$ gives low variance/high bias (all fits look alike but are off), small $\lambda$ gives high variance/low bias (fits scatter widely but average to the truth). Our §10.1.1 table is the same experiment in miniature.

##### Murphy exercises touching this lecture
The chapter 4 exercises (p.164) on the MLE bias (Ex 4.7, 4.8) are solved in Lectures 04 and 12. There is no separate exercise on cross-validation in Murphy's chapters 1 or 5 that is not figure-based.

#### ➕ Textbook extra — regularization, early stopping, AIC/BIC
Regularization ($\ell_2$ penalty = MAP with a Gaussian prior; Murphy §4.5.3 "weight decay") and information criteria are on the course schedule only in week 6 (regularization, MAP), so they are *extras* here. One sentence version: a penalty $\lambda\|\mathbf w\|^2$ adds bias but cuts variance — a knob that moves you along the bias–variance curve.

#### ✅ Key takeaways

* **Generalization** = performance on new data; training loss alone is optimistic.
* Simple models: **high bias, low variance** (underfit). Complex models: **low bias, high variance** (overfit). Expected error = bias² + variance + noise.
* Evaluate on **independent test data from the same distribution**; otherwise it is OOD.
* **Cross-validation** averages errors over several train/test splits; select hyperparameters by CV, then **retrain on all data**.
* Metrics: MSE & residuals (regression); precision/recall/$F_\beta$/ROC/PR (binary); accuracy & confusion matrix (multiclass). **Check per-group performance** for fairness.
* **Inductive bias** is unavoidable; **no free lunch** — no algorithm wins on every problem.
* On our students: train MSE favours degree 5, but CV picks the **line** ($\text{CV MSE}\approx11$).

---


## Lecture 11 — Experiment Reporting

**Why this lecture exists.** You train model A and model B and A scores higher. Is A *really* better, or did you just get lucky this run? Every ML experiment has randomness (initial weights, data shuffling, sampling), so a single number is not evidence. This lecture gives the standard tool for honest reporting: **run it several times, report the mean with its standard error.**

### 11.1 The question

> *Given methods $A$ and $B$, how can one determine whether one model outperforms the other in a statistically significant manner?*

Every time we run an ML experiment we simulate a random process on the computer that generates a measurement of some performance statistic (accuracy, prediction error, …). Think of each run as **sampling a random variable $X_i$**. $X_i$ follows some probability distribution that **we do not know** (because we are running a new ML experiment). We only see samples $x_1,\dots,x_n$.

### 11.2 The Central Limit Theorem (CLT)

Let $X_1,X_2,X_3,\dots$ be a sequence of **IID** random variables with $\mathbb E[X_i]=\mu$ and $\text{Var}[X_i]=\sigma^2<\infty$ (the same *unknown, arbitrary* distribution). Define the sample average $\bar X_n=\frac1n\sum_{i=1}^nX_i$. Then as $n\to\infty$
$$\sqrt n\,(\bar X_n-\mu)\ \xrightarrow{d}\ \mathcal N(0,\sigma^2).$$

"Converges in distribution" means: if $Y=\lim_{n\to\infty}\sqrt n(\bar X_n-\mu)$ then $Y\sim\mathcal N(0,\sigma^2)$. **In words: an average (or sum) of IID random variables is approximately normally distributed, whatever the distribution of the individual $X_i$, as long as the mean and variance are finite.**

The lecture's illustration: $X_i\in\{0,1,2,3,4,5\}$ drawn from a lumpy discrete distribution; yet the histogram of $\bar X_n$ is a smooth bell curve.

*Running example (CLT with pass/fail).* Each $X_i$ is a single pass/fail result with pass probability $0.4$ — about as un-bell-shaped as it gets (two spikes). Averages of $n$ of them have standard deviation $\sqrt{0.4\cdot0.6/n}$:

| $n$ | sd of $\bar X_n$ | 68% of averages fall in |
|---|---|---|
| 10 | 0.155 | $0.4\pm0.155$ |
| 100 | 0.049 | $0.4\pm0.049$ |
| 1000 | 0.0155 | $0.4\pm0.0155$ |

*(Slide typo: the first CLT slide defines $\bar X_n=\sum_iX_i$ without the factor $1/n$; the later slides correctly use $\bar X=\frac1n\sum X_i$. See Errata.)*

### 11.3 The CLT in ML experiments

Repeat the experiment and record measurements $X_1,\dots,X_n$. Their average $\bar X=\frac1n\sum_{i=1}^nX_i$ is approximately normal, and the slide writes $\bar X\sim\mathcal N(\mu,\sigma)$, where
$$\sigma=\frac{\hat\sigma_X}{\sqrt n},\qquad \hat\sigma_X=\sqrt{\frac1{n-1}\sum_{i=1}^n(X_i-\bar X)^2}.$$
($\mathcal N(\mu,\sigma)$ means $\mathcal N(\mu,\sigma^2)$ in the lecture's notation — here $\sigma$ is the standard deviation *of the average*.)

For a normal variable the chance of falling within $\pm\sigma$ of the true mean is $2\cdot34.1\%=68.2\%$ (the slide shows the bell with 34.1% / 13.6% / 2.1% / 0.1% bands at $1\sigma,2\sigma,3\sigma$; exact value $68.27\%$).

### 11.4 The standard error of the mean (SEM)

Therefore the true mean $\mu$ lies in
$$\Big[\ \bar X-\frac{\hat\sigma_X}{\sqrt n},\ \ \bar X+\frac{\hat\sigma_X}{\sqrt n}\ \Big]\quad\text{with probability }68.2\%.$$
The quantity $\dfrac{\hat\sigma_X}{\sqrt n}$ is the **standard error of the mean (SEM)**. In Python: `scipy.stats.sem`.

**The $\sqrt n$ rule.** To halve the uncertainty you need **four times** as many runs: with $\hat\sigma=0.05$, 25 runs give $0.05/5=0.010$ and 100 runs give $0.05/10=0.005$.

*Running example, a SEM of the students' scores.* $n=10$, $\bar x=62.4$, $\hat\sigma=12.094$ (with $n-1$), so $\text{SEM}=12.094/\sqrt{10}=3.824$ and the 68% interval is $[58.58,\ 66.22]$. (`scipy.stats.sem(s)` returns 3.8245; with `ddof=0` it would be 3.628.)

### 11.5 Plotting and stating results

For models $A$ and $B$ plot $\bar X_A$ and $\bar X_B$ with error bars $\bar X\pm\text{sem}$. **If there is a gap between the error bars** (the lecture's drawing: $\bar X_A+\text{sem}(A)$ below $\bar X_B-\text{sem}(B)$) **the performance difference is statistically significant.** Results may also be written as "model A has performance $\bar X_A\pm\text{err}$". Always state in the caption whether the bars are SEM or standard deviation.

**Worked example (5 runs each).**

| Model | accuracies | mean | $\hat\sigma$ | SEM | $\bar X\pm$ SEM |
|---|---|---|---|---|---|
| A | .91 .93 .92 .90 .94 | 0.92 | 0.01581 | 0.00707 | [0.9129, 0.9271] |
| B | .86 .88 .87 .85 .89 | 0.87 | 0.01581 | 0.00707 | [0.8629, 0.8771] |

The intervals do not touch (B's upper end is $0.8771$, A's lower end is $0.9129$: a gap of $0.0358$), so A is credibly better. In this case a formal test agrees emphatically: Welch's t-test on the same 10 numbers gives $t=5.00$, $p=0.0011$.

```python
import numpy as np
from scipy import stats
A = np.array([.91, .93, .92, .90, .94]); B = np.array([.86, .88, .87, .85, .89])
A.mean(), stats.sem(A)            # 0.92, 0.007071
B.mean(), stats.sem(B)            # 0.87, 0.007071
stats.ttest_ind(A, B, equal_var=False)   # t = 5.0, p = 0.00105
```

**A caution about the "gap" rule.** With equal SEMs, two ±1-SEM bars that *just touch* mean a difference of 2 SEM; the standard error of the *difference* is $\sqrt2\,\text{SEM}$, so $z=2/\sqrt2=1.41$ and the two-sided $p\approx0.157$ — not 0.05. To reach 5% you need a difference of about $1.96\sqrt2=2.77$ SEM (a gap of ~0.77 SEM between the bars). So a gap between ±1-SEM bars is a *rule of thumb* for "probably real"; for borderline cases use a proper test or ±2 SEM / 95% intervals. Also with only $n=5$ runs the 68.2% is not exact: $\Pr(|T_4|<1)=62.6\%$ for the $t$-distribution (67.3% at $n=25$).

### 11.6 Setting seeds and reproducibility

* You can often set a **seed** for deterministic execution…
* …but reproducible, deterministic execution is only guaranteed if the **exact same code, libraries and OS** are used, and it often also depends on **hardware** (e.g. GPUs).
* More advanced performance features (e.g. distributed training in PyTorch) often break determinism; so do small library version upgrades.
* In practice, seeding is often not possible/enough. So many ML experiments are **not exactly reproducible**, but more general statements such as *"Model A tends to significantly outperform model B"* **should be**.

**Reproducibility in ML.** Ensuring it through seeding alone is difficult; seeds are still used for **debugging**. Instead, experiments are **repeated, results are stated with errors, and conclusions rest on proper hypothesis testing** (the SEM argument above). When repeating is too expensive (e.g. training LLMs) compute constraints are documented and one argues from performance patterns (one model outperforms another; performance increases with a variable).

```python
rng = np.random.default_rng(423)      # seed it for debugging...
x = rng.normal(size=3)                # ...same numbers on re-run (on the same setup)
```

### 11.7 Hyperparameter selection and "deflating a baseline"

* Experiments often have many **hyperparameters** (e.g. learning rates).
* To support "method A outperforms method B", **different hyperparameter combinations must be tried for both** models.
* Then the **best-performing A and best-performing B** are selected for the final comparison.
* This also means **every hyperparameter combination must be repeated** (several runs each).
* Not testing hyperparameters properly for the baseline is a form of **"deflating a baseline"**.

Choose hyperparameters on validation data or by cross-validation (Lecture 10), **never on the final test data**.

### 11.8 Reporting checklist (for assignments)

* Repeat experiments over several runs; report mean ± SEM and **say which error you plot**.
* Tune both methods equally; repeat every combination.
* Don't treat a seed as proof of reproducibility.
* Document compute limits when you cannot repeat. Use the provided LaTeX template unmodified for reports (syllabus).

#### 📖 Textbook deep dive

##### Murphy §2.8.6 (p.72) — CLT
For IID $X_n$ with mean $\mu$, variance $\sigma^2$ and sum $S_N$: $Z_N=\dfrac{S_N-N\mu}{\sigma\sqrt N}=\dfrac{\bar X-\mu}{\sigma/\sqrt N}\to\mathcal N(0,1)$. This is the slide's statement rewritten: $\bar X\approx\mathcal N(\mu,\sigma^2/N)$, whose standard deviation $\sigma/\sqrt N$ is the SEM. Murphy's Fig. 2.23 averages Beta(1,5) samples (a skewed distribution): $N=1$ is skewed, while the means of $N=5$ samples are already much closer to bell-shaped.

##### Murphy §4.7.1–4.7.2 (p.154–156) — sampling distributions and standard error
The **frequentist** view: the *data* are random and the true parameter $\theta^*$ is a fixed constant, so an estimate $\hat\theta=\hat\Theta(\mathcal D)$ is itself a random variable. Its distribution is the **sampling distribution** — exactly "the distribution of $\bar X$ over repeated experiments". For the MLE with large $N$ it is Gaussian (asymptotic normality, Theorem 4.7.1) with variance given by the inverse **Fisher information** — high curvature of the log-likelihood means a precisely determined parameter. The square root of the variance is the **standard error** $\text{se}$. Murphy's Bernoulli example: $\hat\theta=\frac1N\sum X_n$, $\text{se}=\sqrt{\theta^*(1-\theta^*)/N}$, estimated by $\widehat{\text{se}}=\sqrt{\hat\theta(1-\hat\theta)/N}$. *On the students' pass rate:* $\hat\theta=0.4$, $N=10$ ⇒ $\widehat{\text{se}}=\sqrt{0.24/10}=0.155$, so the 68% interval is $0.4\pm0.155$ — the same reasoning as the slide, with $\sigma^2=\theta(1-\theta)$.

##### Murphy §4.7.3 (p.156) — bootstrap
When the estimator is complicated or $N$ is small, **resample the data with replacement** to make many fake datasets, compute the estimate on each, and use the spread as the sampling distribution. (A bootstrap sample contains on average only $63.2\%=1-e^{-1}$ of the distinct original points.) *Students' mean score:* 20,000 bootstrap means have standard deviation $3.63$ (the analytic SEM is $3.82$ with $1/(n-1)$ and $3.63$ with $1/n$ — the bootstrap uses the $1/n$ version) and a 95% percentile interval $[55.5,\ 69.7]$.
For $N=10$ the bootstrap sampling distribution of a Bernoulli MLE is visibly asymmetric (Fig. 4.23); at $N=100$ it looks Gaussian — the CLT in action. Bootstrap ≈ "poor man's posterior".

##### Murphy §4.7.4 (p.157) — confidence intervals
A $100(1-\alpha)\%$ **confidence interval** is a procedure whose interval captures the true parameter with probability at least $1-\alpha$ over repeated datasets. With the Gaussian approximation: $\hat\theta\pm z_{\alpha/2}\,\widehat{\text{se}}$; for $\alpha=0.05$, $z=1.96$ ("$\hat\theta\pm2\widehat{\text{se}}$"). **The slide's $\bar X\pm\text{SEM}$ is this with $z=1$, i.e. $\alpha=0.318$ (68.2% confidence).** *Students' mean score:* $62.4\pm1.96(3.82)=[54.9,\ 69.9]$ with $z$, or $[53.7,\ 71.1]$ with the more accurate $t_{9}$ multiplier $2.262$ (small $n$ needs wider intervals).

##### Murphy §4.7.5 (p.158) — confidence intervals are not credible intervals
A "95% confidence interval" does **not** mean "the true value lies in *this* interval with probability 0.95". It means *the procedure* captures the truth 95% of the time over repeated datasets; for the one interval you computed, the truth is either inside or not. The statement people usually want, $p(\theta^*\in I\mid\mathcal D)$, is a Bayesian **credible interval**. So read the slide's "with a probability of 68.2%" as "this procedure covers the true mean in 68.2% of repeated experiments".

##### Murphy §5.5 (p.198–203) — hypothesis testing, p-values (what "significant" really means)
* **Null hypothesis significance testing (§5.5.3, p.200):** pick a null hypothesis $H_0$ (e.g. "A and B have equal mean accuracy") and a test statistic; the **p-value** is $\Pr(\text{test statistic as large or larger than observed}\mid H_0)$. Reject $H_0$ if $p<\alpha=0.05$ (the **significance level** = the Type I error rate). The **Wald statistic** is $(\hat\theta-\theta_0)/\widehat{\text{se}}$ with two-sided $p\approx2\Phi(-|z|)$. *Check with our A vs B:* $z=0.05/0.01=5$ ⇒ $p=5.7\times10^{-7}$ (normal) or $0.0011$ (Welch $t$, 8 d.f. — the right one for 5 runs each).
* A **permutation test** (p.201) tests "A and B have the same distribution" by shuffling labels; the **bootstrap** can also be used when the sampling distribution is unknown.
* **Type I / Type II errors and power (§5.5.2, p.199):** Type I = rejecting a true $H_0$ (false positive); Type II = failing to reject a false $H_0$; power $=1-\beta$.
* **p-values considered harmful (§5.5.4, p.201):** a p-value is *not* the probability that $H_0$ is true. Murphy's clinical-trial table: 200 trials, 20 truly effective drugs; with $\alpha=0.05$ and Type II rate $\beta=0.2$, if 90% of drugs are ineffective then $p(H_0\mid\text{"significant"})=\dfrac{0.05\cdot0.9}{0.05\cdot0.9+0.8\cdot0.1}=0.36$ — **36%, not 5%**. Moral: distrust a "significant" claim that contradicts prior knowledge, and report effect sizes and uncertainty rather than just p < 0.05.

##### Murphy §4.5.5.1 (p.126) and §5.4.3 (p.196) — SEM inside cross-validation
CV gives a *mean* test loss; the SEM $\hat\sigma/\sqrt N$ of the per-example losses tells you how uncertain that mean is, which leads to the **one-standard-error rule** (Lecture 10). Same ingredient, different use.

##### Solved conceptual check (no Murphy exercise covers this lecture directly)
*How many runs are needed so that the SEM of an accuracy with $\hat\sigma=0.05$ falls below 0.005?* $0.05/\sqrt n<0.005\Rightarrow\sqrt n>10\Rightarrow n>100$, i.e. 101 runs (100 gives exactly 0.005).

#### ➕ Textbook extra — bootstrap, Wald/permutation tests, Bayesian view
The slides stop at "intervals apart ⇒ significant". Murphy adds: the bootstrap for small $N$, the exact meaning (and misuse) of confidence intervals and p-values, permutation tests, and the Bayesian alternative (posterior $p(H|\mathcal D)$, Bayes factors/likelihood ratio, §5.5.4–5.5.5). Worth reading before writing the report for any assignment.

#### ✅ Key takeaways

* Each experiment run = one sample of an unknown random variable; **average several runs**.
* **CLT:** $\sqrt n(\bar X-\mu)\to\mathcal N(0,\sigma^2)$ — averages are bell-shaped whatever the original distribution.
* **SEM** $=\hat\sigma/\sqrt n$ with $\hat\sigma$ using $1/(n-1)$; $\bar X\pm\text{SEM}$ covers the true mean ≈68% of the time. Halving the error needs **4× the runs**.
* Report **mean ± SEM** and **say what the bars are**; non-overlapping bars suggest a real difference (but ±1-SEM bars that merely avoid touching correspond to only $p\approx0.16$).
* **Seeds ≠ reproducibility** (code, libraries, OS, hardware); use them for debugging; claims should be statistical.
* **Tune the baseline as much as your method** (every combination repeated) — otherwise you are deflating a baseline.

---


## Lecture 12 — Statistics

**About the slides.** This lecture is a single page of handwritten notes (formulas at the top, then Schrödinger's cat, the CLT, the derivation of the standard error, the $n$ vs $n-1$ discussion, and reporting with error bars). It ties together the whole first third of the course: **information measures** (entropy, mutual information), **model vs reality**, and the **derivation of the SEM** used in Lecture 11. Items are listed in slide order.

### 12.1 Formula box: entropy, mutual information, conditional entropy, standard deviation

The slide's top-right corner collects four formulas:

1. **Entropy:** the slide writes $H(X)=\mathbb E[\log X]$ — the correct definition is $H(X)=-\mathbb E[\log p(X)]=-\sum_xp(x)\log p(x)$ (see Errata for the missing minus sign and the $p$).
2. **Mutual information:** $I(X,Y)=H(X)+H(Y)-H(X,Y)$.
3. **Conditional entropy:** $H(Y|X)=-\sum_{x,y}p(x,y)\log\dfrac{p(x,y)}{p(x)}$.
4. **Sample standard deviation:** $\sigma=\sqrt{\dfrac1{N-1}\sum_{i=1}^N(x_i-\bar x)^2}$.

Each deserves plain words.

#### 12.1.1 Entropy — "how surprised, on average?"

**Plain words.** Entropy measures the uncertainty (unpredictability) of a random variable. A coin that always lands heads has 0 uncertainty. A fair coin has the most uncertainty a two-outcome variable can have. The unit with $\log_2$ is the **bit**: the number of yes/no questions needed, on average, to learn the outcome.

$$H(X)=-\sum_{k}p(k)\log_2p(k)\qquad\text{(with }0\log0:=0).$$
$-\log_2p(k)$ is the **surprise** of outcome $k$ (rare outcome ⇒ big surprise); $H$ is the *average* surprise.

*Running example.* The pass rate is $\theta=0.4$:
$$H(0.4)=-(0.4\log_20.4+0.6\log_20.6)=0.4(1.3219)+0.6(0.7370)=0.5288+0.4422=\mathbf{0.971}\text{ bits}.$$
$H(0.5)=1$ bit (maximum), $H(0.9)=0.469$ bits (a 90/10 event is quite predictable), and the 3-class label with probabilities $(0.6,0.2,0.2)$ has $H=1.371$ bits (the maximum for 3 classes is $\log_23=1.585$ bits).

#### 12.1.2 Conditional entropy and joint entropy

* **Joint entropy** $H(X,Y)=-\sum_{x,y}p(x,y)\log p(x,y)$ — uncertainty of the pair.
* **Conditional entropy** $H(Y|X)$ = *the uncertainty left in $Y$ after you learn $X$*. Equivalently $H(Y|X)=H(X,Y)-H(X)$ (chain rule).

*Running example.* Let $S$ = studied low ($\le4$ h) or high ($\ge5$ h) and $T$ = failed/passed, with the joint distribution

| | fail | pass | row total |
|---|---|---|---|
| low | 0.5 | 0.1 | 0.6 |
| high | 0.1 | 0.3 | 0.4 |
| total | 0.6 | 0.4 | 1 |

$H(S)=H(0.4)=0.971$, $H(T)=0.971$, and $H(S,T)=-\big(0.5\log_20.5+2\cdot0.1\log_20.1+0.3\log_20.3\big)=0.5+0.6644+0.5211=1.6855$ bits. Then
$$H(T|S)=H(S,T)-H(S)=1.6855-0.971=\mathbf{0.7145}\text{ bits}.$$
Direct check with the slide's formula: given *low* hours $P(\text{pass})=\tfrac16$, uncertainty $H(\tfrac16)=0.650$ bits; given *high* hours $P(\text{pass})=\tfrac34$, $H(\tfrac34)=0.811$ bits; weighting by $P(S)$: $0.6(0.650)+0.4(0.811)=0.390+0.325=0.7145$ ✔.

#### 12.1.3 Mutual information — "how much does knowing $X$ tell me about $Y$?"
$$I(X,Y)=H(X)+H(Y)-H(X,Y)=H(Y)-H(Y|X)=H(X)-H(X|Y)\ \ge0,$$
and $I=0$ exactly when $X$ and $Y$ are independent.

*Running example:* $I(S,T)=0.971+0.971-1.6855=\mathbf{0.2564}$ bits. Learning whether a student studied a lot removes 0.256 bits of the 0.971 bits of uncertainty about passing (26%). Same answer as $H(T)-H(T|S)=0.971-0.7145=0.2564$ ✔. This puts a *number* on the dependence noticed in Lecture 03 (the joint table is not the product of its marginals).

### 12.2 Schrödinger's cat again: model versus reality

The slide redraws the Lecture 03 picture:

* **Random variables** — "cat alive" / "cat poisoned" — are the **measurements we collect in this experiment** (the dotted line).
* Underneath is the **random process** on $\Omega$ — "we don't have insight into this process".
* There is a **potential mismatch between the designed model and the underlying probabilistic process**.
* Sentence on the slide: *"Any model is designed by an expert and validated (tested) by collecting data."*

*Takeaway:* every number computed later (a mean, a SEM, an interval) describes *the model's world*. The validation step — test data, cross-validation, repeated experiments — is what tells us whether the model's world resembles the real one.

### 12.3 Central Limit Theorem (from the notes)

For random variables $X_1,\dots,X_n$ that are IID with mean $\mu$ and variance $\sigma^2$ (they "can have an arbitrary distribution") define $\bar X_n=\frac1n\sum_{i=1}^nX_i$. Then
$$\sqrt n\,(\bar X_n-\mu)\ \overset{\text{app.}}{\sim}\ \mathcal N(0,\sigma^2)\qquad(\text{A})$$
(the same statement as Lecture 11, written with "$\approx$" for large $n$).

**The question posed.** For two different models $A$ and $B$ we sample statistics $e_A$ and $e_B$. *How can we tell, given $n$ samples, that the true mean lies within a certain range?*

### 12.4 Deriving the interval

**Step 1 — a standard normal fact.** For $S\sim\mathcal N(0,\sigma^2)$:
$$\Pr\big(S\in[-\sigma,\sigma]\big)=0.682,\ \text{i.e.}\ \Pr(-\sigma\le S\le\sigma)=0.682\qquad(\text{B}).$$
(The picture: 34.1% on each side of the mean inside $\pm1\sigma$.)

**Step 2 — substitute (A) into (B).** With $S=\sqrt n(\bar X_n-\mu)$:
$$\Pr\big(-\sigma\le\sqrt n(\bar X_n-\mu)\le\sigma\big)=0.682.$$
Divide through by $\sqrt n$:
$$\Pr\Big(-\frac{\sigma}{\sqrt n}\le\bar X_n-\mu\le\frac{\sigma}{\sqrt n}\Big)=0.682\qquad(\text{C}).$$
Here $\bar X_n$ is **calculated from the data** and $\mu$ is the **theoretical mean we are trying to measure**.

**Step 3 — we do not know $\sigma$, so estimate it.** Use
$$\hat\sigma_n=\sqrt{\frac1{n-1}\sum_{i=1}^n(x_i-\bar x)^2},\qquad\bar x=\frac1n\sum_{i=1}^nx_i\qquad(\text{D}).$$

**Step 4 — substitute (D) into (C):**
$$\Pr\Big(-\frac{\hat\sigma_n}{\sqrt n}\le\bar X_n-\mu\le\frac{\hat\sigma_n}{\sqrt n}\Big)=0.682,$$
and $\dfrac{\hat\sigma_n}{\sqrt n}$ **is defined as the standard error of the mean (SEM)**.

*Worked example (students' scores).* $n=10$, $\bar x=62.4$, $\hat\sigma=12.094$: $\text{SEM}=12.094/\sqrt{10}=3.824$, so
$62.4-3.824\le\mu\le62.4+3.824$, i.e. $\mu\in[58.58,\,66.22]$ with 68.2% confidence. (We are estimating the mean score of the population these ten students represent.)

### 12.5 Why $1/(n-1)$ and not $1/n$?

Right side of the slide:

* The **variance** is $\sigma^2=\text{Var}[X]=\mathbb E[(X-\mu)^2]$.
* Natural estimate: replace the expectation by an average and $\mu$ by the sample average:
$$\hat\sigma^2=\frac1n\sum_{i=1}^n(x_i-\bar x)^2,\qquad\hat\sigma=\sqrt{\frac1n\sum_{i=1}^n(x_i-\bar x)^2}.$$
Here $x_i$ is an *instantiation of the random variable* and $\bar x$ is the *sample average*, which only *approximates* $\mu$.
* The slide says: **we can prove that $\hat\sigma=\sqrt{\frac1{n-1}\sum(x_i-\bar x)^2}$ has lower bias** — "$\mathbb E[|\hat\sigma-\sigma|]$ is smaller with an $n-1$ term".

**Numerical check (simulation, true $\sigma=1$, 200,000 repetitions each):**

| $n$ | $\mathbb E[\hat\sigma^2]$, $1/n$ | $\mathbb E[\hat\sigma^2]$, $1/(n-1)$ | $\mathbb E|\hat\sigma-1|$, $1/n$ | $\mathbb E|\hat\sigma-1|$, $1/(n-1)$ |
|---|---|---|---|---|
| 2 | 0.503 | **1.006** | 0.5353 | 0.5350 |
| 5 | 0.799 | **0.998** | 0.2844 | 0.2811 |
| 10 | 0.899 | **0.999** | 0.1899 | 0.1885 |

* The $1/n$ **variance** is biased: $\mathbb E[\hat\sigma^2_{1/n}]=\frac{n-1}{n}\sigma^2$ (0.5, 0.8, 0.9 for $n=2,5,10$); the $1/(n-1)$ version is exactly **unbiased** for the variance.
* For the **standard deviation** (what the slide uses), the $n-1$ version is *closer on average* ($\mathbb E|\hat\sigma-\sigma|$ is smaller), exactly as the slide states — but the improvement is small, and the $n-1$ standard deviation is still slightly low on average (0.80, 0.94, 0.97 for $n=2,5,10$) because the square root of an unbiased variance is not unbiased.
* **Why it happens:** $\bar x$ is computed from the same data, so the points sit closer to $\bar x$ than to the true $\mu$; one "degree of freedom" is used up. (The $n-1$ is called **Bessel's correction**.)
* **A subtlety the notes do not mention:** "unbiased" does not mean "smaller error". For $n=5$ the mean squared error of the variance estimate is $0.362$ with $1/n$ but $0.503$ with $1/(n-1)$ — the biased estimator has smaller variance and wins in MSE (the bias–variance trade-off of Lecture 10). For **reporting a SEM** the lecture uses $n-1$, which is the standard convention.

In NumPy: `x.std(ddof=1)` gives the $1/(n-1)$ version; `x.std()` (ddof=0) gives $1/n$. `scipy.stats.sem(x)` uses $ddof=1$.

### 12.6 Reporting two models with the SEM

If we have two models with statistics $\bar X_n$ and $\bar Y_n$, report both with the SEM: draw each average as a dot with a bar of one SEM on each side. **If the intervals are apart, we can state that $Y$ performs statistically significantly better than $X$.** (The slide's sketch shows the two intervals separated by a gap.) The numeric version is in Lecture 11: with $A=0.92\pm0.0071$ and $B=0.87\pm0.0071$ the gap is $0.0358$; but remember the caution there — merely non-overlapping ±1-SEM bars correspond to $p\approx0.16$, not 0.05.

```python
import numpy as np
from scipy import stats
s = np.array([66,48,71,45,57,62,77,59,83,56.])
mean, sd, sem = s.mean(), s.std(ddof=1), stats.sem(s)    # 62.4, 12.094, 3.824
lo, hi = mean - sem, mean + sem                           # 58.58, 66.22  (68%)

def H(p):                                                # entropy in bits
    p = np.asarray(p, float); p = p[p > 0]
    return -(p * np.log2(p)).sum()

J = np.array([[.5, .1], [.1, .3]])                       # joint of (hours low/high, fail/pass)
I = H(J.sum(1)) + H(J.sum(0)) - H(J.ravel())             # 0.2564 bits
```

#### 📖 Textbook deep dive

##### Murphy §6.1 (p.207–213) — entropy and relatives
* **Entropy** $H(X)=-\sum_kp(k)\log_2p(k)=-\mathbb E[\log p(X)]$ (Eq. 6.1). The unit is *bits* ($\log_2$) or *nats* ($\ln$). Murphy's example: $p=[0.25,0.25,0.2,0.15,0.15]$ has $H=2.29$ bits (verified $2.2855$). **Max** entropy is the uniform distribution, $H=\log_2K$; **min** (0) is a delta function. For a Bernoulli, $H(\theta)=-[\theta\log_2\theta+(1-\theta)\log_2(1-\theta)]$, maximal (1 bit) at $\theta=0.5$ (Fig. 6.1: the binary entropy curve).
* **Cross entropy** $H_{ce}(p,q)=-\sum_kp_k\log q_k$ (§6.1.2, p.209): the expected number of bits to encode data from $p$ with a code designed for $q$; minimized at $q=p$, where it equals $H(p)$ (Shannon's source-coding theorem). **This is the loss of Lectures 08–09.** *Students:* $p=(0.4,0.6)$ against a model $q=(0.5,0.5)$: $H_{ce}=1$ bit vs $H(p)=0.971$ — the extra 0.029 bits is the KL divergence, and in nats $0.029\ln2=0.0201=0.6931-0.6730$ (the gap between the NLL/D of $\theta=0.5$ and $\theta^*=0.4$ in Lecture 04 ✔).
* **Joint entropy** (§6.1.3, p.209), Murphy's example: pick $n\in\{1..8\}$; $X$ = even, $Y$ = prime. The four combinations have probabilities $\frac18,\frac38,\frac38,\frac18$ so $H(X,Y)=1.811$ bits, while $H(X)=H(Y)=1$; hence $I(X;Y)=2-1.811=0.189$ bits.
* **Conditional entropy** (§6.1.4, p.210): $H(Y|X)=\sum_xp(x)H(Y|X{=}x)=H(X,Y)-H(X)$, and $H(Y|X)\le H(Y)$ — *on average, conditioning never increases uncertainty* (though a particular observation can; see Ex 6.3 below).
* **Perplexity** (§6.1.5, p.211): $2^{H(p)}$ — the "effective number of equally likely outcomes". *Students:* pass/fail has perplexity $2^{0.971}=1.96$ (almost a fair coin); the 3-class labels $2^{1.371}=2.59$.
* **Differential entropy** (§6.1.6, p.212): for continuous variables, $h(X)=-\int p\log p$; for a Gaussian $h=\tfrac12\log(2\pi e\sigma^2)$. *Students' scores:* with $\hat\sigma^2=131.64$, $h=\tfrac12\ln(2\pi e\cdot131.64)=3.859$ nats — exactly the NLL/D of the fitted Gaussian found in Lecture 04. (MLE at the optimum has loss equal to the entropy of the fitted model.) Unlike discrete entropy, $h$ can be negative.
* **KL divergence** (§6.2, p.213–216): $D_{KL}(p\|q)=\sum_kp_k\log\frac{p_k}{q_k}=H_{ce}(p,q)-H(p)\ge0$, zero iff $p=q$. §6.2.5 (p.215): **maximum likelihood = minimizing KL to the empirical distribution** (the result noted in Lecture 04).

##### Murphy §6.3 (p.217–225) — mutual information
* Definition $I(X;Y)=D_{KL}\big(p(x,y)\,\|\,p(x)p(y)\big)=\sum_{x,y}p(x,y)\log\frac{p(x,y)}{p(x)p(y)}\ge0$, with equality iff independent — "how far is the joint from the product of marginals". Equivalent forms: $H(X)-H(X|Y)=H(Y)-H(Y|X)$ (the slide's $H(X)+H(Y)-H(X,Y)$).
* *Interpretation:* the reduction in uncertainty about one variable from observing the other (§6.3.2). §6.3.5 views MI as a "generalized correlation coefficient" that also detects non-linear dependence. §6.3.8 **data processing inequality**: processing $Y$ can't create information about $X$.

##### Murphy §4.7.6 (p.159) — bias of an estimator
$\text{bias}(\hat\theta)=\mathbb E[\hat\theta]-\theta^*$; an estimator is **unbiased** if this is 0. The MLE of a Gaussian *mean* is unbiased but the MLE of the *variance* is not: $\mathbb E[\hat\sigma^2_{\rm mle}]=\frac{N-1}{N}\sigma^2$ (Exercise 4.7, solved in Lecture 04). Murphy also notes $\text{MSE}=\text{variance}+\text{bias}^2$, which explains why the biased $1/n$ estimator can still have the smaller MSE (our $n=5$ check: 0.362 vs 0.503).

##### Murphy §4.7.1–4.7.4 and Bishop §1.2.4 (p.24–28)
Murphy §4.7.2 (p.155): $\text{se}=\sqrt{\mathbb V[\hat\theta]}$; the Gaussian approximation gives the interval $\hat\theta\pm z\cdot\widehat{\text{se}}$ (Lecture 11). Bishop §1.2.4 derives the ML mean and variance and the bias factor $\frac{N-1}{N}$, adding that "the bias of the maximum likelihood solution becomes less significant as $N$ increases" — in our table the gap shrinks from 50% at $n=2$ to 10% at $n=10$.

##### Bishop §1.6 (p.48–58) — entropy and mutual information
* **Entropy examples (p.49–50):** a variable with 8 equally likely states needs 3 bits ($H=3$); with probabilities $(\frac12,\frac14,\frac18,\frac1{16},\frac1{64},\frac1{64},\frac1{64},\frac1{64})$ the entropy is only 2 bits (both verified), so a *non-uniform* distribution has lower entropy and can be encoded with a shorter average code. This ties entropy to the optimal average code length (the noiseless coding theorem).
* **Relative entropy / KL divergence (§1.6.1, p.55)** $\text{KL}(p\|q)=-\int p(x)\ln\frac{q(x)}{p(x)}dx\ge0$ (by convexity/Jensen), and the **mutual information** $I[x,y]=\text{KL}(p(x,y)\|p(x)p(y))=H[x]-H[x|y]=H[y]-H[y|x]$. Bishop's reading: "the reduction in uncertainty about $x$ by virtue of being told the value of $y$". Minimizing KL between the data distribution and the model is equivalent to maximizing likelihood.

##### Solved exercises (Murphy §6.4, p.226–227)

**Ex 6.1 (MI identities).** $I(X;Y)=\sum p(x,y)\log\frac{p(x,y)}{p(x)p(y)}=\sum p(x,y)\log\frac{p(x|y)}{p(x)}=-\sum p(x,y)\log p(x)+\sum p(x,y)\log p(x|y)=H(X)-H(X|Y)$; by symmetry $=H(Y)-H(Y|X)$. Then $H(X,Y)=H(X)+H(Y|X)=H(X|Y)+H(Y|X)+I(X;Y)$, since $H(X)=H(X|Y)+I(X;Y)$.

**Ex 6.3 (fun with entropies).** For the $4\times4$ joint with rows $y$ and columns $x$ (all $y$ have $p(y)=\frac14$): 
(a) $H(X,Y)=\mathbf{27/8=3.375}$ bits. 
(b) $H(X)=\mathbf{7/4=1.75}$ bits (marginals $\frac12,\frac14,\frac18,\frac18$) and $H(Y)=\mathbf{2}$ bits (uniform over 4). 
(c) $H(X|y=1)=\frac74$, $H(X|y=2)=\frac74$, $H(X|y=3)=2$, $H(X|y=4)=0$. **Yes, the posterior entropy can go up for a particular observation** ($y=3$ gives $2>1.75$). 
(d) $H(X|Y)=\frac14(\frac74+\frac74+2+0)=\mathbf{11/8=1.375}$ — **on average it decreases** (from 1.75). 
(e) $I(X;Y)=H(X)-H(X|Y)=1.75-1.375=\mathbf{3/8=0.375}$ bits (also $=1.75+2-3.375$ ✔).

**Ex 4.7 / 4.8.** (bias of the variance MLE) are in Lecture 04.

#### ➕ Textbook extra — KL divergence, perplexity, data-processing inequality, normalized MI
Murphy §6.1.5 (perplexity, used to evaluate language models — week 8 "sequence models, LLMs"), §6.2 (KL; forward vs reverse KL, §6.2.6), §6.3.6 (normalized MI) and §6.3.8 (data-processing inequality) go beyond the slide's three formulas. Perplexity and cross-entropy will matter when the course reaches language models.

#### ✅ Key takeaways

* **Entropy** $H(X)=-\mathbb E[\log p(X)]$ = average surprise; max for the uniform distribution. Pass rate 0.4 ⇒ 0.971 bits.
* **Conditional entropy** $H(Y|X)=H(X,Y)-H(X)$; **mutual information** $I=H(X)+H(Y)-H(X,Y)\ge0$, zero iff independent (our hours/pass: 0.256 bits).
* Every model is an assumption about a hidden random process — validate with data.
* **CLT ⇒ interval:** $\Pr(|\bar X-\mu|\le\hat\sigma/\sqrt n)\approx0.682$; $\hat\sigma/\sqrt n$ is the **SEM**; estimate $\hat\sigma$ with $1/(n-1)$.
* $1/n$ variance is biased by $\frac{n-1}{n}$; $1/(n-1)$ is unbiased for the variance and slightly better for the standard deviation.
* Report two models with **mean ± SEM**; separated intervals suggest a real difference.

---


## Lecture 13 — Bayes' Law and Maximum A Posteriori (MAP) Estimation

**About the slides.** This lecture is a single long page of handwritten notes with five blocks, in this order: (1) **Bayes' law** drawn with areas, (2) the **degenerate cases of maximum likelihood** (the polynomial-basis regression picture), (3) the **Bayesian approach** (likelihood, prior, posterior), (4) the **MAP estimate**, and (5) a full **coin-flip example** with a Beta prior and six small plots (prior, likelihood, posterior for two data sets). Lecture 14 then applies the same idea to regression (ridge and lasso). Everything below follows that order.

**Big idea in one sentence.** Maximum likelihood asks "which parameters make the data most probable?"; MAP asks "which parameters are most probable *after seeing the data*, given what I believed *before* seeing it?" — the belief is called the **prior**, and it keeps the answer from going to silly extremes when data are scarce.

### 13.1 Bayes' law, drawn with areas

**Plain words.** Draw a big square for *everything that could happen* ($\Omega$, the event space). Draw a green circle for event $A$ and a blue circle for event $B$; they overlap. A probability is just a **share of the square's area**:

* $p(A)=\dfrac{\text{area of green circle}}{\text{area of square}}$, and likewise $p(B)$ for the blue circle.
* $p(A,B)=p(A\wedge B)=\dfrac{\text{area of the overlap}}{\text{area of the square}}$ — the share of the square that is *both* green and blue.
* $p(A|B)=\dfrac{\text{overlap}}{\text{blue circle}}$ — "pretend we are *inside* the blue circle; what fraction of it is also green?" (the new "universe" is $B$, not the whole square).
* $p(B|A)=\dfrac{\text{overlap}}{\text{green circle}}$.

The slide then multiplies fractions so the circle areas cancel (the picture with $\frac{\text{overlap}}{\text{blue}}\cdot\frac{\text{blue}}{\text{square}}=\frac{\text{overlap}}{\text{square}}$):
$$p(A,B)=p(A|B)\,p(B),\qquad p(A,B)=p(B|A)\,p(A).$$
Both right-hand sides equal the same overlap, so they equal each other:
$$p(A|B)\,p(B)=p(B|A)\,p(A)\ \Longrightarrow\ \boxed{\,p(A|B)=\frac{p(B|A)\,p(A)}{p(B)}\,}\qquad(p(B)\ne0).$$
This is **Bayes' law**. It lets us **flip a conditional probability**: if we know how likely $B$ is when $A$ holds, we can compute how likely $A$ is when $B$ holds.

**Symbols.** $\Omega$ = set of all outcomes; $A,B$ = events (subsets of $\Omega$); $p(A,B)$ = probability both happen; $p(A|B)$ = probability of $A$ *given that* $B$ happened.

*Running example.* Let $A$ = "the student passed" ($t=1$) and $B$ = "the student studied at least 4 hours". From the ten students ($h$ = 4, 2, 6, 1, 3, 5, 7, 3, 8, 2; passed = students 1, 3, 7, 9):

* 4 of 10 passed ⇒ $p(A)=0.4$.
* 5 of 10 studied $\ge4$ h (hours 4, 6, 5, 7, 8) ⇒ $p(B)=0.5$.
* The students who did both: hours 4, 6, 7, 8 passed (and $h=5$ did *not*) ⇒ 4 of 10 ⇒ $p(A,B)=0.4$.

Direct conditionals: $p(A|B)=0.4/0.5=0.8$ ("of the five who studied $\ge4$ h, four passed") and $p(B|A)=0.4/0.4=1.0$ ("every student who passed studied $\ge4$ h"). **Bayes check:**
$$p(A|B)=\frac{p(B|A)p(A)}{p(B)}=\frac{1.0\times0.4}{0.5}=0.8\ \checkmark$$

### 13.2 Recap: maximum likelihood and its degenerate cases

The slide restates the **maximum likelihood method**: find the parameters that make *generating the dataset* most likely,
$$\theta^*\in\arg\max_\theta\ p(\mathcal D\,|\,\theta),$$
where $p(\mathcal D|\theta)$, viewed as a function of $\theta$, is the **likelihood function** (Lecture 04).

**The solution may lead to degenerate cases.** The slide's example is *linear regression with basis functions*: the same 1-D data set is fitted with a polynomial of degree 3, degree 5 and degree 50. The degree-3 and degree-5 curves (orange and green) follow the points sensibly; the **degree-50 curve (red)** passes through the training points but is flat or violently oscillating in between and shoots to $\pm\infty$ at the right edge. The side table summarizes:

| | Training error | Test error |
|---|---|---|
| **Overfitting** | low | high |
| **Underfitting** | high | high |

**Why does ML overfit?** The model has so much freedom that "make the training data most probable" is solved by memorizing noise. The optimisation is **under-constrained**: many parameter vectors fit the training data almost equally well, and ML has no way to prefer a sensible one.

*Running example (numbers).* Fit our ten students with a **degree-7 polynomial** in standardized hours $x=(h-\bar h)/\mathrm{std}(h)$ (8 coefficients for only 8 distinct hour values). The fit is almost perfect on the training data — training MSE $=3.40$, and that remaining error is *only* the unavoidable spread between the two students with $h=2$ (scores 48, 56) and the two with $h=3$ (scores 57, 59): $(2\cdot4^2+2\cdot1^2)/10=3.4$. But leave-one-out cross-validation (Lecture 10) gives an error of about **$10^4$** (the exact digits depend on rounding because the problem is nearly singular), versus **11.76** for the plain straight line. Perfect on training data, hopeless on a new student.

```python
# continues from the running-example setup (Part F.1)
x = (h - h.mean()) / h.std()
def ridge_fit(Phi, y, lam):            # lam = 0 -> plain least squares; intercept (column 0) not penalised
    P = lam * np.eye(Phi.shape[1]); P[0, 0] = 0
    return np.linalg.solve(Phi.T @ Phi + P, Phi.T @ y)
Phi = np.vander(x, 8, increasing=True)             # degree-7 polynomial features, shape (10, 8)
w = ridge_fit(Phi, s, 1e-8)                        # lam tiny = "almost" maximum likelihood
print(round(((s - Phi @ w) ** 2).mean(), 2))       # 3.4  (training MSE)
loo = []
for i in range(len(s)):
    m = np.arange(len(s)) != i
    wi = ridge_fit(Phi[m], s[m], 1e-8)
    loo.append((s[i] - Phi[i] @ wi) ** 2)
print(round(np.mean(loo)))                         # about 11438: huge (digits depend on rounding)
```

### 13.3 The Bayesian approach

The slide's idea: instead of treating $\theta$ as an unknown *constant*, treat it as a **random variable** with its own probability distribution. Then data and parameters have a **joint distribution** that can be factored in two ways:
$$p(\mathcal D,\theta)=\underbrace{p(\mathcal D|\theta)}_{\text{likelihood function}}\ \underbrace{p(\theta)}_{\text{prior}}\;=\;p(\theta|\mathcal D)\ \underbrace{p(\mathcal D)}_{\text{evidence}}.$$
Dividing by $p(\mathcal D)$ is Bayes' law with $A=\theta$ and $B=\mathcal D$:
$$\boxed{\,p(\theta|\mathcal D)=\frac{p(\mathcal D|\theta)\,p(\theta)}{p(\mathcal D)}\,}\qquad\text{posterior}=\frac{\text{likelihood}\times\text{prior}}{\text{evidence}}.$$

| Name | Meaning | In words |
|---|---|---|
| **Prior** $p(\theta)$ | belief about $\theta$ *before* the data | "what values of $\theta$ do I find plausible?" |
| **Likelihood** $p(\mathcal D\mid \theta)$ | how probable the data are for a given $\theta$ | the same function Lecture 04 maximized |
| **Evidence** $p(\mathcal D)$ | probability of the data under the whole model | "determined by the probabilistic process we wish to model" (a normalizing constant; the slide marks it as an unknown quantity) |
| **Posterior** $p(\theta\mid \mathcal D)$ | belief about $\theta$ *after* the data | the compromise between prior and data |

### 13.4 The MAP estimate

**Maximum a posteriori (MAP) estimate** = the most probable parameter value *under the posterior*:
$$\theta^*\in\arg\max_\theta\ p(\theta|\mathcal D).$$
Plug in Bayes' law:
$$p(\theta|\mathcal D)=\frac{p(\mathcal D|\theta)\,p(\theta)}{p(\mathcal D)}\ \propto\ p(\mathcal D|\theta)\,p(\theta)\qquad(\text{“}\propto\text{” = proportional to, because }p(\mathcal D)\text{ is constant with respect to }\theta).$$
A constant factor cannot change *where* the maximum is, so
$$\boxed{\,\theta^*\in\arg\max_\theta\ p(\theta|\mathcal D)=\arg\max_\theta\ p(\mathcal D|\theta)\,p(\theta)\,}\qquad\text{(slide label (A)).}$$
We never have to compute the difficult evidence $p(\mathcal D)$. Taking logs (as in Lecture 04) turns this into a sum: $\arg\max_\theta[\log p(\mathcal D|\theta)+\log p(\theta)]$ — "log-likelihood plus log-prior" — which is exactly what Lecture 14 uses.

**MLE vs MAP:** if the prior is flat ($p(\theta)$ the same for every $\theta$) the prior factor is a constant and MAP *equals* MLE.

### 13.5 Worked example: Bernoulli coin flips with a Beta prior

**Data.** $\mathcal D=\{1,0,1,0,0,0,1,0,1,0\}$ — ten Bernoulli trials ("1" = heads). This is the same pattern as our pass/fail column. $D=10$ data points, $H=\sum_ix_i=4$ heads.

**Likelihood** (slide label (B)). Each point has probability $p(x_i|\theta)=\theta^{x_i}(1-\theta)^{1-x_i}$ and the points are independent, so
$$L(\theta;\mathcal D)=p(\mathcal D|\theta)=\prod_{i=1}^{D}\theta^{x_i}(1-\theta)^{1-x_i}=\theta^{H}(1-\theta)^{D-H}.$$
Its maximizer is the MLE $\theta^*=H/D=0.4$ (the slide's first plot: a hill peaking at $\theta=0.4$ with height $0.4^4\cdot0.6^6=0.00119$, matching the $\approx0.0012$ on the axis).

**The problem we want to fix.** With so little data the MLE can be extreme. If all 10 flips were heads the MLE would be $\theta=1$: "the coin *always* lands heads", which nobody believes. We want to *constrain* the optimisation with a **prior** that prefers solutions **close to 0.5** (a fair coin).

**The Beta distribution** (slide label (C)). A probability distribution over a number $\theta\in[0,1]$:
$$p(\theta)\propto\theta^{\alpha-1}(1-\theta)^{\beta-1}.$$
$\alpha,\beta>0$ are *hyperparameters* (settings of the prior). With $\alpha=\beta=2$ the curve is a smooth arch centred at 0.5 ($p(\theta)\propto\theta(1-\theta)$, zero at 0 and 1). Larger equal values (e.g. 6, 6) give a taller, narrower arch: a *stronger* belief that the coin is fair.

**Posterior.** Multiply the likelihood by the prior (slide: "by (A), (B), (C)"), keeping only $\theta$-dependent factors:
$$p(\theta|\mathcal D)\ \propto\ \underbrace{\theta^{H}(1-\theta)^{D-H}}_{\text{likelihood}}\cdot\underbrace{\theta^{1}(1-\theta)^{1}}_{\text{Beta}(2,2)\text{ prior}}=\theta^{H+1}(1-\theta)^{D-H+1}.$$
The slide's summary line: $p(\theta|\mathcal D)\propto\theta^{H+1}(1-\theta)^{D-H+1}$. **The prior acted like one extra imaginary head and one extra imaginary tail.**

**MAP estimate (finishing the slide's derivation).** Maximize the log of the posterior:
$$\frac{d}{d\theta}\Big[(H{+}1)\ln\theta+(D{-}H{+}1)\ln(1-\theta)\Big]=\frac{H+1}{\theta}-\frac{D-H+1}{1-\theta}=0\ \Longrightarrow\ \theta_{\rm MAP}=\frac{H+1}{D+2}.$$
For our data: $\theta_{\rm MAP}=\dfrac{4+1}{10+2}=\dfrac5{12}=0.4167$, slightly pulled from the MLE $0.4$ toward $0.5$. For general Beta$(\alpha,\beta)$:
$$\boxed{\theta_{\rm MAP}=\frac{H+\alpha-1}{D+\alpha+\beta-2}}$$

| Prior | $\alpha,\beta$ | $\theta_{\rm MAP}$ for $H=4,D=10$ | $\theta_{\rm MAP}$ for $H=10,D=10$ |
|---|---|---|---|
| Flat (no belief) | 1, 1 | $4/10=0.400$ (= MLE) | $10/10=1.000$ (= MLE) |
| Weak | 2, 2 | $5/12=0.417$ | $11/12=0.917$ |
| Medium | 3, 3 | $6/14=0.429$ | $12/14=0.857$ |
| Strong | 6, 6 | $9/20=0.450$ | $15/20=\mathbf{0.750}$ |

**The slide's six plots (Prior · Likelihood · Posterior).** Each plot's vertical axis is "$\propto$" (unnormalized, so only *shapes* and the location of the peak matter).

* **Example 1: $H=4,\ D=10$** (4 heads, 6 tails). The likelihood peaks at the **MLE = 0.4** (height $\approx0.0012$). The prior is an arch centred at 0.5. The posterior peaks at the **MAP**, just to the right of 0.4 (the slide writes "0.4" under the dashed line; the value is $0.43$ — see the note below).
* **Example 2: $H=10,\ D=10$** (10 heads, 0 tails). The likelihood $\theta^{10}$ keeps rising to **MLE = 1.0** ("every flip was heads ⇒ $\theta=1$"). The prior still pulls toward 0.5, so the posterior peaks at the **MAP $\approx0.75$**: a sensible compromise between "the coin is always heads" and "the coin is fair".

*Checking which prior the plots used.* The prior's peak height at $\theta=0.5$ is $0.5^{2(\alpha-1)}$. In example 1 the peak reads $\approx0.06$, i.e. $0.5^4=0.0625$, so $\alpha=\beta=3$; the posterior then peaks at $6/14=0.4286$ with height $0.4286^6\cdot0.5714^8=7.04\times10^{-5}\approx70\,\mu$, matching the "60μ–70μ" axis. In example 2 the prior peak reads $\approx0.001=0.5^{10}$, so $\alpha=\beta=6$; the posterior peaks at $15/20=0.75$ with height $0.75^{15}\cdot0.25^5=1.3\times10^{-5}$, matching "10μ". (This is my reading of the plotted heights; the handwritten derivation itself uses $\alpha=\beta=2$. With $\alpha=\beta=2$ and $H=10$ the MAP would be $0.917$, not the $0.75$ shown. See Errata.)

```python
# continues from the running-example setup (Part F.1)
import numpy as np
D_ = np.array([1, 0, 1, 0, 0, 0, 1, 0, 1, 0]); H_ = D_.sum(); N_ = len(D_)   # H = 4, D = 10
theta = np.linspace(0, 1, 100001)
lik = theta ** H_ * (1 - theta) ** (N_ - H_)
print(theta[lik.argmax()], lik.max())                      # 0.4  0.001194  (the MLE)
for a, b in [(1, 1), (2, 2), (3, 3), (6, 6)]:
    post = lik * theta ** (a - 1) * (1 - theta) ** (b - 1)  # unnormalised posterior
    print((a, b), round(theta[post.argmax()], 4), round((H_ + a - 1) / (N_ + a + b - 2), 4))
# (1,1) 0.4 0.4 | (2,2) 0.4167 0.4167 | (3,3) 0.4286 0.4286 | (6,6) 0.45 0.45
H2 = 10
lik2 = theta ** H2 * (1 - theta) ** (N_ - H2)               # ten heads in ten flips
post2 = lik2 * theta ** 5 * (1 - theta) ** 5                # Beta(6, 6) prior
print(theta[lik2.argmax()], round(theta[post2.argmax()], 2))   # 1.0 (MLE)  0.75 (MAP)
```

#### 📖 Textbook deep dive

##### Murphy §4.5 (p.120–122) — regularization is MAP
* **The problem (p.120–121).** Toss a coin 3 times and see 3 heads: $\hat\theta_{\rm mle}=3/3=1$, so the model predicts *every future toss is heads*. "The core of the problem is that the model has enough parameters to perfectly fit the observed training data, so it can perfectly match the empirical distribution … putting all the probability mass on the observed set of $N$ examples will not leave over any probability for novel data."
* **Regularized objective (Eq. 4.89–4.92).** $L(\theta;\lambda)=\frac1N\sum_n\ell(y_n,\theta;x_n)+\lambda C(\theta)$. Choosing the penalty $C(\theta)=-\log p(\theta)$ and $\lambda=1$ gives $-[\log p(\mathcal D|\theta)+\log p(\theta)]$, so *minimizing the penalized loss = maximizing the log posterior* = **MAP**. **This is the bridge to Lecture 14:** every regularizer is a prior.
* **Beta prior for a Bernoulli (§4.5.1, Eq. 4.93–4.96, p.121–122).** With $p(\theta)=\text{Beta}(\theta|a,b)$: $\theta_{\rm MAP}=\dfrac{N_1+a-1}{N_1+N_0+a+b-2}$ (the slide's formula with $a=\alpha$, $b=\beta$). For $a=b=2$ this is **add-one smoothing** $\dfrac{N_1+1}{N+2}$ (Eq. 4.96), "a simple but widely used technique to avoid the zero count problem." *Students:* 3 heads, 0 tails: MLE $=1$ but add-one gives $4/5=0.8$.
* **Black swans (p.122).** Murphy relates this to the "black swan paradox": induction from past observations is never certain; "the best we can do is to make plausible guesses about what the future might hold, by combining the empirical data with prior knowledge."
* **MAP for a covariance matrix (§4.5.2, p.122–123).** In high dimensions the MLE of $\Sigma$ can be singular; a MAP estimate with an inverse-Wishart prior "shrinks" the off-diagonal entries by a factor $(1-\lambda)$ and is always well conditioned (Fig. 4.4). *Connection to Lecture 06:* this is the principled version of adding $\epsilon I$ to a covariance matrix.
* **Choosing $\lambda$ (§4.5.4–4.5.5, p.124–126).** Split into train and validation sets, fit for each $\lambda$, pick the $\lambda$ with the lowest *validation* risk, then refit on all data (Eq. 4.101–4.105); if data are scarce use $K$-fold cross-validation (Lecture 10). The **one-standard-error rule** picks the *simplest* model whose cross-validated risk is within one standard error of the best, $\text{se}(\hat\mu)=\hat\sigma/\sqrt N$ (Lecture 11).
* **Early stopping and more data (§4.5.6–4.5.7, p.126–127).** Stopping an iterative optimizer when validation error rises is a form of regularization; and overfitting shrinks as $N$ grows (learning curves, Fig. 4.9).

##### Murphy §4.6.2 (p.130–133) — the beta–binomial model in detail
* **Conjugate prior (p.130–131).** A prior is *conjugate* to a likelihood if the posterior is in the same family. For a Bernoulli/binomial likelihood the Beta prior is conjugate: $p(\theta|\mathcal D)\propto\theta^{N_1+a-1}(1-\theta)^{N_0+b-1}=\text{Beta}(a+N_1,\,b+N_0)$ (Eq. 4.111–4.114). The slide's "posterior $\propto\theta^{H+1}(1-\theta)^{D-H+1}$" is exactly Beta$(H{+}2,\,D{-}H{+}2)$.
* **Pseudo-counts (p.131).** "The hyper-parameters play a role analogous to the sufficient statistics; they are therefore often called pseudo counts … we can compute the posterior by simply adding the observed counts (from the likelihood) to the pseudo counts (from the prior)." $a=b=2$ means "we have already seen two heads and two tails" (a *very weak* preference for 0.5); the strength $\breve N=a+b$ is the *equivalent sample size*. $a=b=1$ is the uniform prior and the posterior then has exactly the shape of the likelihood (Fig. 4.10).
* **Mode, mean, variance (p.132–133; Eq. 2.139 on p.64).** Beta$(a,b)$: mean $\frac a{a+b}$, **mode** $\frac{a-1}{a+b-2}$ (for $a,b>1$), variance $\frac{ab}{(a+b)^2(a+b+1)}$. MAP = the posterior **mode**; the posterior **mean** $\frac{a+N_1}{a+b+N}$ is a more robust summary "since it integrates over the whole space". *Students:* with $a=b=2$, $H=4$, $D=10$: posterior Beta(6, 8), MAP $=5/12=0.4167$, posterior mean $=6/14=0.4286$, variance $\frac{6\cdot8}{14^2\cdot15}=0.0163$ (sd $0.128$).
* **Posterior mean = blend of prior and data (Eq. 4.124).** $\mathbb E[\theta|\mathcal D]=\lambda m+(1-\lambda)\hat\theta_{\rm mle}$ with prior mean $m=a/(a+b)$ and $\lambda=\frac{a+b}{a+b+N}$ — "the weaker the prior, the smaller is $\lambda$, and hence the closer the posterior mean is to the MLE." *Students:* $\lambda=4/14=0.286$, $m=0.5$, $\hat\theta_{\rm mle}=0.4$ ⇒ $0.286(0.5)+0.714(0.4)=0.4286$ ✔. With more data $\lambda\to0$ and **the prior washes out** (compare the slide's weak prior on 10 flips).
* **Uncertainty (Eq. 4.127–4.128).** For large $N$ the posterior sd is $\approx\sqrt{\hat\theta(1-\hat\theta)/N}$, shrinking as $1/\sqrt N$ — the same $1/\sqrt n$ law as the SEM of Lecture 11.
* **Posterior predictive (p.133).** Plugging in a point estimate (MLE or MAP) can overfit; a fully Bayesian prediction averages over the posterior instead: for Beta$(\breve a,\breve b)$ the probability that the next toss is heads is the posterior mean. (Beyond the slides.)

##### Murphy §2.7.4 (p.63–64) and Bishop §2.1.1 (p.71–72) — the Beta distribution
$\text{Beta}(\theta|a,b)=\frac1{B(a,b)}\theta^{a-1}(1-\theta)^{b-1}$ with $B(a,b)=\frac{\Gamma(a)\Gamma(b)}{\Gamma(a+b)}$. Shapes (Murphy Fig. 2.17a): $a=b=1$ uniform; $a,b>1$ unimodal; $a,b<1$ "spikes" at 0 and 1. Bishop motivates it differently: because the likelihood has the form $\mu^x(1-\mu)^{1-x}$, a prior of the same form keeps the posterior in the same family — *conjugacy* — and "the effect of observing $m$ ones and $l$ zeros has been to increase $a$ by $m$ and $b$ by $l$"; the hyperparameters are "an effective number of observations". The posterior can serve as the prior for the next batch of data (sequential learning).

##### Bishop §1.1 (p.4–11) and §1.2.3 (p.21–23) — overfitting and the Bayesian view
* **Polynomial curve fitting (p.4–11).** Fitting sine data with polynomials of order $M=0,1,3,9$: $M=9$ has 10 coefficients for 10 points and fits them exactly ("training error goes to zero") but "exhibits wild oscillations"; the test error becomes very large (Fig. 1.5). **Table 1.1 (p.8)** shows the cause: the $M=9$ coefficients grow to the order of $10^6$ ($-1061800.52$, $1042400.18$, …) — "the coefficients have become finely tuned to the data by developing large positive and negative values". *Same phenomenon as our degree-7 fit above.* With more data the overfitting shrinks (Fig. 1.6, $N=15$ vs $N=100$); a rough rule of thumb is to have 5–10 times as many data points as parameters.
* **Regularization (p.10, Eq. 1.4).** Adding $\frac\lambda2\|\mathbf w\|^2$ to the error suppresses the problem ($\ln\lambda=-18$ works; $\ln\lambda=0$ is too much and underfits again). "The particular case of a quadratic regularizer is called ridge regression … In the context of neural networks, this approach is known as weight decay." Bishop adds that "by adopting a Bayesian approach, the over-fitting problem can be avoided … the effective number of parameters adapts automatically to the size of the data set."
* **Bayesian probabilities (§1.2.3, p.21–23).** Probability can express *degrees of belief*, not only long-run frequencies. Bayes' theorem $p(\mathbf w|\mathcal D)=\frac{p(\mathcal D|\mathbf w)p(\mathbf w)}{p(\mathcal D)}$ converts a prior $p(\mathbf w)$ into a posterior after observing $\mathcal D$; "posterior $\propto$ likelihood $\times$ prior". A well-known criticism, which the slides share, is that the prior is a modelling choice — it injects an **inductive bias** (Lecture 10).

##### Solved conceptual checks
**(a) Why is MAP not affected by the evidence $p(\mathcal D)$?** $p(\mathcal D)=\int p(\mathcal D|\theta)p(\theta)d\theta$ does not contain $\theta$ — it is a single number for a fixed data set. Dividing a function by a positive constant does not move its peak, so $\arg\max_\theta p(\theta|\mathcal D)=\arg\max_\theta p(\mathcal D|\theta)p(\theta)$.

**(b) Show that $\theta_{\rm MAP}=\frac{H+\alpha-1}{D+\alpha+\beta-2}$.** Log-posterior $=(H+\alpha-1)\ln\theta+(D-H+\beta-1)\ln(1-\theta)$ + const. Derivative $\frac{H+\alpha-1}\theta-\frac{D-H+\beta-1}{1-\theta}=0$ ⇒ $(H+\alpha-1)(1-\theta)=(D-H+\beta-1)\theta$ ⇒ $\theta=\frac{H+\alpha-1}{D+\alpha+\beta-2}$ ✔ (check $\alpha=\beta=2$: $\frac{H+1}{D+2}$ ✔).

**(c) Does MAP always lie between the prior's peak and the MLE?** For the Beta prior yes, in these examples: prior mode $=0.5$, MLE $=0.4$, MAP $=0.4167\in[0.4,0.5]$; MLE $=1$, MAP $=0.75\in[0.5,1]$. The posterior is a compromise whose position depends on the *relative strength* (pseudo-counts $\alpha+\beta-2=2$ versus $D=10$).

**(d) Bayes' law with other numbers.** If a disease affects 1% of people, and a test is positive for 99% of the sick and 5% of the healthy, then $p(\text{sick}|+)=\frac{0.99\cdot0.01}{0.99\cdot0.01+0.05\cdot0.99}=\frac{0.0099}{0.0594}=0.1667$ — only about 17% despite a "99% accurate" test, because the **prior** (1%) is small. (Lecture 03's Exercise 2.9 solution shows an even rarer disease giving about 1%. The point here is that the **prior** changes the answer.)

#### ➕ Textbook extra — Bayesian prediction and empirical Bayes
Murphy §4.6.2.9 (posterior predictive: average predictions over the posterior instead of trusting one point estimate), §4.6.2.10 (marginal likelihood/evidence $p(\mathcal D)$ — the quantity the slides call "difficult to compute" and drop), and Bishop §3.3 (Bayesian linear regression, where the prior on $\mathbf w$ is Gaussian and the posterior is Gaussian in closed form, with the MAP equal to the posterior mean) go beyond the slides. The evidence can also be used to *choose* the prior strength ("empirical Bayes", Murphy §11.3.3 p.384) instead of cross-validation.

#### ✅ Key takeaways

* **Bayes' law:** $p(A|B)=p(B|A)\,p(A)/p(B)$ — flip a conditional by multiplying by the prior ratio. Probabilities are shares of area.
* **Maximum likelihood** can overfit/degenerate when the model is flexible or data are scarce (degree-50 polynomial; MLE $=1$ after 10 heads).
* **Bayesian view:** posterior $p(\theta|\mathcal D)=\dfrac{\text{likelihood}\times\text{prior}}{\text{evidence}}$, written $p(\theta|\mathcal D)\propto p(\mathcal D|\theta)\,p(\theta)$.
* **MAP** $=\arg\max_\theta p(\mathcal D|\theta)p(\theta)$: the evidence is constant in $\theta$ so it is ignored; with a flat prior MAP $=$ MLE.
* **Coin example:** Beta$(\alpha,\beta)$ prior ⇒ posterior $\propto\theta^{H+\alpha-1}(1-\theta)^{D-H+\beta-1}$, $\theta_{\rm MAP}=\frac{H+\alpha-1}{D+\alpha+\beta-2}$; Beta$(2,2)$ gives add-one smoothing $\frac{H+1}{D+2}$ ($5/12$ for 4 heads in 10; $0.75$ with the stronger prior in the 10-heads plot).
* The prior is an **inductive bias**; its influence fades as the data set grows.

---

## Lecture 14 — Lasso and Ridge Regression

**About the slides.** Seventeen slides (numbered 2–4, 7–17; slides 5–6 are not in the PDF). They cover: the Bayesian approach (recap of Lecture 13), basis functions in linear regression, the family of priors $p(\theta)\propto e^{-\lambda\sum|\theta_i|^p}$, how prior and likelihood combine into a loss, **ridge** ($L^2$) and **lasso** ($L^1$) regression, regularization in general, a definition of MAP estimation, and two real-data examples (prostate cancer data, taken from Murphy's Figure 11.10). Handwritten red notes on the slides are included below.

**Big idea in one sentence.** *A regularizer is a prior:* penalizing large weights is the same as believing, before seeing data, that small weights are more probable — and the **shape** of the penalty ($L^2$ round, $L^1$ pointy) decides whether the learned weights are merely *small* (ridge) or exactly *zero* (lasso).

### 14.1 The Bayesian approach to regression (slides 2–4)

The slides repeat Lecture 13 in three steps.

1. **Bayes' theorem:** for events $A,B$ with $p(B)\ne0$, $\ p(A|B)=\dfrac{p(B|A)\,p(A)}{p(B)}$.
2. **Maximum likelihood** finds $\theta^*\in\arg\max_\theta p(\mathcal D|\theta)$. The slide asks the key question: *"How could we constrain which $\theta$ we want to find?"*
3. **Bayesian approach:** assume a **prior** distribution on $\theta$; then the probability of the parameters after seeing the data $\mathcal D$ is
$$p(\theta|\mathcal D)=\frac{\overbrace{p(\mathcal D|\theta)}^{\text{likelihood}}\ \overbrace{p(\theta)}^{\text{prior}}}{\underbrace{p(\mathcal D)}_{\text{evidence}}}\qquad(\text{left side: the posterior distribution}).$$
 "Because [the evidence] is difficult to compute and we only want to maximize the posterior, we can write" $p(\theta|\mathcal D)\propto p(\mathcal D|\theta)\,p(\theta)$.

(Full derivation and the coin example are in Lecture 13.)

### 14.2 Basis functions in linear regression (slides 7–9)

**Setup (handwritten notes on slide 7).** The data set is $\mathcal D$: pairs $(x,y)$ with $x,y\in\mathbb R$ (one number in, one number out). A **basis expansion** turns the single number $x$ into a feature vector $\boldsymbol\phi(x)\in\mathbb R^{50}$ whose $i$-th entry is $x^{i-1}$ (polynomial basis). The model is still linear in the parameters:
$$y=\boldsymbol\theta^\top\boldsymbol\phi(x,n)+\varepsilon\qquad(n=\text{number of basis functions},\ \varepsilon=\text{noise}).$$
**"The linear model does fit this dataset, because the inputs are represented differently"** — a straight line in $x$ cannot fit a curve, but a straight line in the *features* $(1,x,x^2,\dots)$ can. The slide-7 picture is the degree-3/5/50 plot from Lecture 13.

**The trouble.** *"If we pick $n$ too large, we start to fit dataset noise into the model parameter $\boldsymbol\theta$. The MLE is not providing us with a good estimate because the optimization process is under-constrained. How can we fix this?"* Handwritten answer on the slide: **"Lasso regression would perform feature selection."**

**Common basis functions (slide 8, from Bishop 2006, Fig. 3.1).** Each $\phi_k$ is one "building block"; $\mu_k$ is a centre (location) and $s$ a width:

| Family | Formula | Shape |
|---|---|---|
| **Polynomial** | $\phi_k(x)=x^k$ | global: every term affects all $x$ |
| **Gaussian** | $\phi_k(x)=\exp\!\Big(-\dfrac{(x-\mu_k)^2}{s^2}\Big)$ | a bump around $\mu_k$ (local) |
| **Sigmoid** | $\phi_k(x)=\sigma\!\Big(\dfrac{x-\mu_k}{s}\Big)$ | a smooth step at $\mu_k$ |

*Running example (real numbers).* Evaluate the Gaussian and sigmoid bases at $h=4$ hours with centres $\mu=(2,5,8)$ and width $s=2$ (Gaussian) / $s=1$ (sigmoid):
$$\phi^{\rm Gauss}(4)=\big(e^{-(4-2)^2/4},e^{-(4-5)^2/4},e^{-(4-8)^2/4}\big)=(e^{-1},e^{-0.25},e^{-4})=(0.368,\ 0.779,\ 0.018);$$
$$\phi^{\rm sigm}(4)=\big(\sigma(2),\sigma(-1),\sigma(-4)\big)=(0.881,\ 0.269,\ 0.018).$$
A 4-hour student is "closest" to the centre at 5 hours (Gaussian bump 0.779), has passed the step at 2 (0.881) and is far below the step at 8 (0.018). (The slide's Gaussian has no factor 2 in the denominator, $e^{-(x-\mu)^2/s^2}$; Bishop's (3.4) writes $e^{-(x-\mu)^2/(2s^2)}$ — both are fine; they just differ by how "$s$" is defined.)

**Recap of basis selection (slide 9).** With $\boldsymbol\phi(x)=[\phi_1(x),\dots,\phi_n(x)]^\top$ the model is $y=\boldsymbol\theta^\top\boldsymbol\phi(x)=\theta_1\phi_1(x)+\dots+\theta_n\phi_n(x)$. *"Picking too many basis functions leads to overfitting. But in linear regression we can interpret the weights $\theta_i$ to determine how much a basis function contributes to the output."* *"Instead of computing the MLE, the MAP estimate can be used to select which basis functions should be included in the model by setting $\theta_i=0$ for different $i$."* — a weight of exactly 0 deletes that feature from the model (**feature selection**).

### 14.3 Prior distributions (slide 10)

The slide says: *suppose we make up our own custom prior*
$$p(\boldsymbol\theta)\ \propto\ e^{-\lambda\sum_i|\boldsymbol\theta(i)|^p}.$$
**Plain words.** The exponent is a *penalty* for large weights: the larger $\sum_i|\theta_i|^p$, the smaller the prior probability. $\lambda>0$ is the **strength** of the belief ($\lambda$ large ⇒ "weights are surely small"), and $p>0$ controls the **shape**. $\theta(i)$ is the $i$-th entry of the parameter vector.

**The four contour plots** (2-D, $\theta=[\theta_1,\theta_2]$; each curve is a set of points with the *same* prior probability, i.e. $\sum|\theta_i|^p=$ constant):

| $p$ | Shape of contours | Name of the prior | Behaviour |
|---|---|---|---|
| $0.5$ | star with sharp, curved-in spikes on the axes | (very sparse prior; not convex) | strongly prefers weights *exactly* 0 |
| $1$ | diamond (square rotated 45°) | **Laplace** prior ($L^1$) | corners on the axes ⇒ favours zeros |
| $2$ | circle | **Gaussian** prior ($L^2$) | round: shrinks all weights evenly |
| $4$ | rounded square | (flat in the middle, steep at the edge) | allows all weights up to a limit |

*Numbers.* With $\lambda=1$ one weight $\theta$, the *unnormalized* prior is $e^{-|\theta|^p}$:

| $\theta$ | $p=1$ (Laplace) | $p=2$ (Gaussian) |
|---|---|---|
| 0 | 1 | 1 |
| 0.5 | 0.607 | 0.779 |
| 1 | 0.368 | 0.368 |
| 2 | 0.135 | 0.018 |

The Laplace density has a *sharp peak* at 0 (and falls off like a tent), the Gaussian a *rounded* top; far from zero the Gaussian decays much faster (heavier tails for Laplace — Murphy §2.7.3). For $p=2$: $e^{-\lambda\theta^2}=e^{-\theta^2/(2\tau^2)}$ with variance $\tau^2=1/(2\lambda)$; for $p=1$: Laplace with scale $b=1/\lambda$ and variance $2b^2=2/\lambda^2$.

### 14.4 Interaction between prior and likelihood (slides 11–12)

Apply MAP to linear regression and take logs (a log is increasing, so it keeps the same maximizer):
$$\arg\max_\theta p(\theta|\mathcal D)=\arg\max_\theta\log p(\theta|\mathcal D)=\arg\max_\theta\Big[\log p(\mathcal D|\theta)+\log p(\theta)\Big].$$
The slide splits this into two bullets:

* **Negative log-likelihood:** $-\log p(\mathcal D|\theta)=\text{MSE}(\theta)$ (Lecture 07: Gaussian noise ⇒ the loss is the squared error).
* **Negative log-prior:** $-\log e^{-\lambda\sum_i|\theta(i)|^p}=\lambda\sum_i|\theta(i)|^p$.

Maximizing a sum of logs = minimizing the sum of their negatives, so
$$\boxed{\ \arg\max_\theta p(\theta|\mathcal D)=\arg\min_\theta\ \underbrace{\text{MSE}(\theta)}_{\text{fit the data}}+\underbrace{\lambda\sum_i|\theta(i)|^p}_{\text{keep the weights small}}\ }\qquad J(\theta)=\text{MSE}(\theta)+\lambda\sum_i|\theta(i)|^p.$$

**A precision note** (the slide is deliberately loose). With Gaussian noise of variance $\sigma^2$, $-\log p(\mathcal D|\theta)=\dfrac{\text{RSS}}{2\sigma^2}+\dfrac N2\log(2\pi\sigma^2)$ with RSS $=\sum_n(y_n-\theta^\top\phi_n)^2$, i.e. *equal to the MSE up to a positive scale and an additive constant*. Neither changes the minimizer, but the scale changes what $\lambda$ means. If the prior is $p(\theta)\propto e^{-\lambda'\sum|\theta_i|^p}$, then
$$\frac{\text{RSS}}{2\sigma^2}+\lambda'\sum_i|\theta_i|^p\ \ \text{has the same minimizer as}\ \ \text{MSE}+\underbrace{\frac{2\sigma^2\lambda'}{N}}_{=\ \lambda\text{ of the slide}}\sum_i|\theta_i|^p\quad(\text{MSE}=\text{RSS}/N).$$
For ridge ($p=2$, Gaussian prior $\mathcal N(0,\tau^2)$ so $\lambda'=1/(2\tau^2)$) and the *RSS* form of the loss used in the examples later, the penalty weight is $\lambda_{\rm RSS}=\sigma^2/\tau^2$ (Murphy Eq. 11.53): a **noisy-data / confident-prior** ratio. *Students:* with $\sigma^2=1$ and prior variance $\tau^2=0.02$ the penalty weight is $\lambda=50$ (used in the ridge numbers below).

**The picture on slide 12.** Two copies of the plane of $(w_1,w_2)$, where $\mathbf w=[w_1,w_2]^\top$ is the parameter vector.

* **MSE contribution to the loss surface** = concentric ellipses (blue) around the MLE (red dot): the minimum of the data-fit term. Every ellipse is a set of weights with the *same* training error.
* **Prior contribution to the loss surface** = the filled region (orange) around the origin. Left: a **circle** ($p=2$, the slide writes $p\le2$/"$p=2$" for the circle). Right: a **diamond** ($p=1$).
* **MAP estimate $\mathbf w^\star$** = the point where the *smallest* blue ellipse that still reaches the orange region just **touches** it. On the circle the touching point is generic: *both* coordinates are non-zero. On the diamond the touching point is a **corner, on an axis**, so one coordinate is **exactly zero**: *sparsity*.

Why is "touching the region" the right picture? A penalty $\lambda\|\mathbf w\|_p^p$ is the Lagrange form of the *constraint* "$\|\mathbf w\|_p^p\le B$" (Murphy Eq. 11.78–11.80): a large $\lambda$ ⇔ a small region (tight bound $B$). Minimizing the error subject to being inside the region means growing the ellipses until they first touch it.

### 14.5 Ridge and lasso regression (slides 13–14)

$$\textbf{Ridge:}\ \ J(\theta)=\text{MSE}(\theta)+\lambda\sum_i|\theta(i)|^2\ (L^2\text{ norm, Gaussian prior});\qquad \textbf{Lasso:}\ \ J(\theta)=\text{MSE}(\theta)+\lambda\sum_i|\theta(i)|^1\ (L^1\text{ norm, Laplace prior}).$$
**"Lasso results in sparse models."** *Sparse* = many parameters are exactly 0.

**Regularization (slide 14).** *"Regularization is a method for reducing model complexity by constraining the parameter space."* In regression, $L^p$-regularization adds a second term, $J(\theta)=\text{MSE}(\theta)+\lambda\sum_i|\theta(i)|^p$, which "constrains the solution of the optimization problem to a parameter vector with lower $L^p$-norm". *"In deep learning, other methods also exist, such as dropout, batch normalization, etc."* (dropout comes in Lecture 15).

**Computing ridge.** Write the loss in the RSS form used by the examples (slides 16–17) and Murphy: $J(\theta)=\|\mathbf y-\mathbf X\theta\|^2+\lambda\|\theta\|^2$. Setting the gradient to zero (Lecture 07 gradient plus $2\lambda\theta$) gives
$$\nabla J=-2\mathbf X^\top(\mathbf y-\mathbf X\theta)+2\lambda\theta=0\ \Longrightarrow\ \boxed{\ \hat\theta_{\rm ridge}=(\mathbf X^\top\mathbf X+\lambda\mathbf I)^{-1}\mathbf X^\top\mathbf y\ }$$
(Murphy Eq. 11.57). If the loss is written with the *mean* squared error, $\text{MSE}+\lambda\|\theta\|^2$ with $\text{MSE}=\text{RSS}/N$, replace $\lambda$ by $N\lambda$. Adding $\lambda\mathbf I$ to $\mathbf X^\top\mathbf X$ also makes it invertible even when features are dependent — the "$\epsilon$ on the diagonal" trick of Lecture 06 now has a probabilistic meaning. **Lasso** has no such closed form (the $|\cdot|$ has a kink at 0); it is solved iteratively (coordinate descent below).

**Don't penalize the intercept.** The offset $\theta_0$ only shifts all predictions; penalizing it would make the answer depend on the origin of $y$ (Murphy p.381). Below, all intercepts are left free.

*Running example: one feature (hours).* Centre the data: $S_{xx}=\sum(h_i-\bar h)^2=48.9$ and $S_{xy}=\sum(h_i-\bar h)(s_i-\bar s)=245.6$, so the OLS slope is $245.6/48.9=5.0225$ (the Lecture 07 answer, intercept $41.808$). With a free intercept the one-feature solutions are
$$\hat w_{\rm ridge}=\frac{S_{xy}}{S_{xx}+\lambda},\qquad \hat w_{\rm lasso}=\frac{\text{sign}(S_{xy})\,\big(|S_{xy}|-\lambda/2\big)_+}{S_{xx}}\quad(\text{for }\text{RSS}+\lambda|w|),$$
where $(z)_+=\max(z,0)$.

| $\lambda$ | ridge slope | ridge RSS | lasso slope | lasso RSS |
|---|---|---|---|---|
| 0 | 5.0225 | 82.88 | 5.0225 | 82.88 |
| 10 | 4.1698 | 118.4 | – | – |
| 50 | 2.4833 | 398.2 | 4.5112 | 95.66 |
| 100 | 1.6494 | 639.2 | 4.0000 | 134.0 |
| 200 | – | – | 2.9775 | 287.4 |
| 400 | – | – | 0.9325 | 900.9 |
| 500 | – | – | **0** (since $2\lvert S_{xy}\rvert=491.2<500$) | 1316.4 |
| 1000 | 0.2342 | 1204 | 0 | 1316.4 |

*Reading the table.* **Ridge** divides by a larger number, so the slope shrinks *smoothly toward* 0 but never reaches it (even at $\lambda=1000$ it is $0.234$). **Lasso** subtracts $\lambda/2$ from the numerator, so it reaches **exactly 0** as soon as $\lambda\ge2|S_{xy}|=491.2$ (the "$\lambda_{\max}$" beyond which the model predicts the mean score 62.4 for everyone, RSS $=1316.4=$ TSS). For $\lambda=100$ the lasso slope is exactly $4.000$ ($=5.0225-50/48.9\cdot\ldots$; check: $(245.6-50)/48.9=4.000$ ✔) and the intercept is $46.0$.

*Feature selection with junk features (the point of lasso).* Add two **irrelevant** columns of random numbers next to hours and standardize everything (code below):

| $\lambda$ | lasso: [hours, junk1, junk2] | ridge: [hours, junk1, junk2] |
|---|---|---|
| 0 (OLS) | [10.78, −0.51, 0.07] | [10.78, −0.51, 0.07] |
| 1 | – | [9.24, −1.48, 0.19] |
| 10 | – | [4.77, −2.16, −0.34] |
| 20 | **[10.11, 0, 0]** | – |
| 100 | [6.11, 0, 0] | [0.96, −0.63, −0.25] |
| 300 | [0, 0, 0] | – |

At $\lambda=20$ **lasso keeps only the real feature and sets both junk weights to exactly 0** (and the hours weight is still large, 10.11). At every $\lambda$ **ridge keeps all three weights non-zero**, so the junk features keep contributing noise (and ridge coefficients are not even monotone in $\lambda$: the first junk weight goes from $-0.51$ to $-1.48$ at $\lambda=1$ before shrinking; Murphy's Figure 11.10(a) shows the same non-monotone paths). For a *standardized* design, lasso switches off the last feature when $\lambda$ exceeds $2\|\mathbf X^\top\mathbf y\|_\infty=222.1$ (at $\lambda=300$ everything is 0).

```python
# continues from the running-example setup (Part F.1)
import numpy as np
hc, sc = h - h.mean(), s - s.mean()
Sxx, Sxy = (hc ** 2).sum(), (hc * sc).sum()          # 48.9, 245.6
soft = lambda z, d: np.sign(z) * max(abs(z) - d, 0)   # soft-thresholding
for lam in [0, 10, 50, 100, 1000]:
    w = Sxy / (Sxx + lam); b = s.mean() - w * h.mean()
    print('ridge', lam, round(w, 4), round(((s - b - w * h) ** 2).sum(), 1))
for lam in [0, 50, 100, 200, 400, 500]:
    w = soft(Sxy, lam / 2) / Sxx; b = s.mean() - w * h.mean()
    print('lasso', lam, round(w, 4), round(((s - b - w * h) ** 2).sum(), 1))
# ridge with a free intercept, matrix form: (X^T X + lam*P)^-1 X^T y with P = diag(0, lam)
print(np.linalg.solve(X.T @ X + np.diag([0, 50.]), X.T @ s))     # [52.2184  2.4833]

def lasso_cd(X, y, lam, iters=2000):                  # minimises ||y - Xw||^2 + lam*||w||_1
    w = np.zeros(X.shape[1]); a = (X ** 2).sum(0)
    for _ in range(iters):
        for d in range(X.shape[1]):
            r = y - X @ w + X[:, d] * w[d]            # residual without feature d
            c = X[:, d] @ r                           # c_d in Murphy Eq. 11.83
            w[d] = soft(c, lam / 2) / a[d]            # soft-threshold, Eq. 11.88 (lam/2 for this RSS form)
    return w

J = np.random.default_rng(0).normal(size=(10, 2))     # two columns of pure noise
F = np.column_stack([h, J]); Fc = (F - F.mean(0)) / F.std(0); yc = s - s.mean()
for lam in [0, 20, 100, 300]:
    print('lasso', lam, np.round(lasso_cd(Fc, yc, lam), 4))      # lam=20 -> [10.1064 -0. -0.]
for lam in [0, 1, 10, 100]:
    print('ridge', lam, np.round(np.linalg.solve(Fc.T @ Fc + lam * np.eye(3), Fc.T @ yc), 4))
```

### 14.6 Maximum a posteriori estimation (slide 15)

**Definition.** *"In maximum a posteriori estimation, we assume a parameter prior $p(\theta)$ and solve the optimization problem*
$$\theta^*\in\arg\max_\theta\ p(\mathcal D|\theta)\,p(\theta).$$
*In contrast to using the maximum likelihood method, maximum a posteriori estimation extends the probabilistic model with a prior $p(\theta)$ on the parameters we seek to find. This prior is a form of an **inductive bias** on our machine learning model."* (Inductive bias: the built-in assumptions that make an algorithm prefer some solutions over others — Lecture 10.) The slide's two examples:

1. **$L^p$-norm regularization** for both linear regression and also **deep learning** ("weight decay").
2. **Initialization heuristics** used in deep learning (e.g. **Xavier initialization**). Starting the weights from $\mathcal N(0,\sigma^2)$ with a carefully chosen variance is like putting a prior on the weights *before* training; Murphy §13.4.5 (p.452–453) gives $\sigma^2=\frac2{n_{\rm in}+n_{\rm out}}$ (Xavier/Glorot) so that signals neither explode nor vanish. (Details in Lecture 15's deep dive.)

### 14.7 Example: ridge regression (slide 16)

*Loss objective: $\|\mathbf y-\mathbf X\theta\|_2+\alpha\|\theta\|_2^2$. "Predict lpsa given other features."* (Adopted from Murphy'22, Fig. 11.10(a).) The data: 8 clinical measurements of prostate-cancer patients (lcavol, lweight, age, lbph, svi, lcp, gleason, pgg45) used to predict the log prostate-specific antigen (lpsa). The plot shows each feature's **coefficient** (vertical axis) as a function of the regularization strength $\alpha$ (horizontal axis, log scale from 1 to $>10^4$).

* At small $\alpha$ (left) the coefficients are the least-squares ones: lcavol $\approx0.67$ is the largest, lweight $\approx0.22$, svi $\approx0.31$, age and lcp $\approx-0.14$.
* As $\alpha$ grows **every** coefficient shrinks toward 0, smoothly — "none of them is ever exactly zero".
* The thin pink vertical line is the value of $\alpha$ chosen by cross-validation (Murphy: 5-fold CV with the one-standard-error rule).

### 14.8 Example: lasso regression (slide 17)

*Loss objective: $\|\mathbf y-\mathbf X\theta\|_2+\alpha\|\theta\|_1$. "Predict lpsa given other features."* (Murphy'22, Fig. 11.10(b).) The horizontal axis is now the **$L^1$ norm** $\|\theta\|_1$ of the coefficient vector; a small norm means a *large* $\alpha$. The red notes on the slide: **"high $\alpha$"** on the left, **"low $\alpha$"** on the right; **"$x^\top\theta\approx$ lpsa", "$\sum_{i=1}^8x_i\theta_i\approx$ lpsa"** (the model is a sum of 8 weighted features); and **"By picking a larger $\alpha$, lasso regression performs feature selection, because certain $\theta_i=0$."**

At $\|\theta\|_1=0$ every coefficient is 0. As the budget grows the coefficients "turn on" **one at a time**: first lcavol, then lweight, svi, … while the least-important features (age, lcp, gleason) stay at exactly 0 until the end (the flat lines on the zero axis). Murphy's Table 11.1 gives the actual numbers (first row all zeros, last row = least squares):

| lcavol | lweight | age | lbph | svi | lcp | gleason | pgg45 |
|---|---|---|---|---|---|---|---|
| 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 0.4279 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 0.5015 | 0.0735 | 0 | 0 | 0 | 0 | 0 | 0 |
| 0.5610 | 0.1878 | 0 | 0 | 0.0930 | 0 | 0 | 0 |
| 0.5797 | 0.2456 | 0 | 0.1435 | 0.2003 | 0 | 0 | 0.0901 |
| 0.7164 | 0.2926 | −0.1425 | 0.2120 | 0.3096 | −0.2890 | −0.0209 | 0.2773 |

(I omitted some intermediate rows; see Murphy p.390.) With a larger $\alpha$ (the red dashed line) only the first few features have non-zero weights: the **ridge** curve keeps all 8, the **lasso** keeps a few — the same contrast as our hours/junk example above.

#### 📖 Textbook deep dive

##### Murphy §11.3 (p.381–385) — ridge regression
* **MAP with a Gaussian prior (p.381, Eq. 11.52–11.54).** $\hat{\mathbf w}_{\rm map}=\arg\min\frac1{2\sigma^2}(\mathbf y-\mathbf X\mathbf w)^\top(\mathbf y-\mathbf X\mathbf w)+\frac1{2\tau^2}\mathbf w^\top\mathbf w=\arg\min\text{RSS}+\lambda\|\mathbf w\|_2^2$ with $\lambda=\sigma^2/\tau^2$ "proportional to the strength of the prior". *"Note that we do not penalize the offset term $w_0$ … see Exercise 11.2."*
* **Closed form (§11.3.1, p.382, Eq. 11.55–11.57).** $J=(\mathbf y-\mathbf X\mathbf w)^\top(\mathbf y-\mathbf X\mathbf w)+\lambda\|\mathbf w\|^2$, $\nabla J=2(\mathbf X^\top\mathbf X\mathbf w-\mathbf X^\top\mathbf y+\lambda\mathbf w)$, $\hat{\mathbf w}=(\mathbf X^\top\mathbf X+\lambda\mathbf I_D)^{-1}\mathbf X^\top\mathbf y$ (matches the formula above).
* **Don't invert; use QR (§11.3.1.1, p.382, Eq. 11.58–11.64).** "Naively computing the primal estimate … using matrix inversion is a bad idea, since it can be slow and numerically unstable." Trick: **append "virtual data"** — add $D$ extra rows $\sqrt\Lambda$ with target $0$ to $\mathbf X$ and $\mathbf y$ (scaled by $1/\sigma$) so the penalty becomes ordinary squared error on the extra rows, then solve an ordinary least-squares problem by QR. *Interpretation:* the prior acts like $D$ pseudo-observations saying "this weight should be about 0". *(Lecture 06/07 link: Lecture 06's Cholesky factor of the precision $\Lambda$.)*
* **SVD view (§11.3.1.2–11.3.2, p.383–384, Eq. 11.65–11.71).** With $\mathbf X=\mathbf U\mathbf S\mathbf V^\top$ and singular values $\sigma_j$, least squares predicts $\hat{\mathbf y}=\sum_j\mathbf u_j\mathbf u_j^\top\mathbf y$, while ridge predicts $\hat{\mathbf y}=\sum_j\mathbf u_j\,\dfrac{\sigma_j^2}{\sigma_j^2+\lambda}\,\mathbf u_j^\top\mathbf y$. Directions with **large** $\sigma_j^2$ (well determined by the data) are kept ($\frac{\sigma^2}{\sigma^2+\lambda}\approx1$); directions with **small** $\sigma_j^2$ (poorly determined, noise-dominated) are **shrunk toward 0**. The **effective degrees of freedom** $\text{dof}(\lambda)=\sum_{j=1}^D\frac{\sigma_j^2}{\sigma_j^2+\lambda}$ goes from $D$ at $\lambda=0$ to $0$ as $\lambda\to\infty$ (Fig. 11.7 shows the geometry: the poorly determined parameter is pulled strongly toward the prior mean; the well-determined one barely moves — "shrinkage"). *Students:* for one standardized feature with $\sigma^2=S_{xx}=48.9$ and $\lambda=50$, the shrink factor is $48.9/98.9=0.494$ — compare the slope ratio $2.4833/5.0225=0.4944$ ✔.
* **Ridge vs principal-components regression (p.384).** PCA regression keeps the top $K$ directions and *discards* the rest; ridge "uses a soft weighting of all the dimensions" and predicts better ([HTF01]).
* **Choosing $\lambda$ (§11.3.3, p.384–385).** Cross-validation over a grid with **warm starts** (start from the strong-regularizer solution and relax); the set of solutions is the **regularization path** (Fig. 11.10a). Alternatively maximize the **marginal likelihood** $\log p(\mathcal D|\lambda)$ ("empirical Bayes"): one model fit instead of $K$, and a smooth function of $\lambda$.

##### Murphy §11.4 (p.385–397) — lasso regression
* **Why sparsity, and the $L^0$ norm (p.385).** To remove a feature we want $w_d=0$ exactly: minimize $\|\mathbf w\|_0=\sum_d\mathbb I(|w_d|>0)$ (the number of non-zeros). That is combinatorial, so we use the $L^1$ norm instead.
* **Laplace prior (§11.4.1, Eq. 11.73–11.75).** $p(\mathbf w|\lambda)=\prod_d\text{Laplace}(w_d|0,1/\lambda)\propto\prod_de^{-\lambda|w_d|}$ and $\text{PNLL}(\mathbf w)=\|\mathbf X\mathbf w-\mathbf y\|_2^2+\lambda\|\mathbf w\|_1$. The name **lasso** = "least absolute shrinkage and selection operator". $\text{Laplace}(w|\mu,b)=\frac1{2b}e^{-|w-\mu|/b}$ has mean $=$ mode $=\mu$ and variance $2b^2$ (§2.7.3, p.63): "more density on 0 than $\mathcal N(0,\sigma^2)$, even when we fix the variance to be the same."
* **$q$-norms (Eq. 11.76, p.386).** $\|\mathbf w\|_q=(\sum|w_d|^q)^{1/q}$; for $q<1$ even sparser but the problem becomes **non-convex**. "The $\ell_1$-norm is the tightest convex relaxation of the $\ell_0$-norm." (That is why the slide's $p=0.5$ star is shown but not used.)
* **Why $L^1$ gives zeros (§11.4.2, p.386–387, Fig. 11.8).** Penalty form and *bound* form are equivalent: $\min\text{NLL}+\lambda\|\mathbf w\|_1\ \Leftrightarrow\ \min\text{NLL}\ \text{s.t.}\ \|\mathbf w\|_1\le B$ (small $B$ ⇔ big $\lambda$). "The corners of the ball are more likely to intersect the ellipse than one of the sides, especially in high dimensions, because the corners 'stick out' more. The corners correspond to sparse solutions, which lie on the coordinate axes. By contrast, when we grow the $\ell_2$ ball … there are no 'corners', so there is no preference for sparsity." (= the slide 12 picture.)
* **Soft thresholding (§11.4.3, p.387–389, Eq. 11.81–11.89).** For coordinate $d$ with all other weights fixed: $a_d=\sum_nx_{nd}^2$ and $c_d=\sum_nx_{nd}(y_n-\mathbf w_{-d}^\top\mathbf x_{n,-d})$ ("proportional to the correlation between feature $d$ and the residual of the other features"). The subgradient condition has three cases:
$$\hat w_d=\begin{cases}(c_d+\lambda)/a_d&c_d<-\lambda\\0&c_d\in[-\lambda,\lambda]\\(c_d-\lambda)/a_d&c_d>\lambda\end{cases}\ =\ \text{SoftThreshold}\!\Big(\frac{c_d}{a_d},\frac\lambda{a_d}\Big),\quad\text{SoftThreshold}(x,\delta)=\text{sign}(x)(|x|-\delta)_+.$$
**Hard thresholding** (subset selection) sets small coefficients to 0 but leaves the big ones untouched (Fig. 11.9b); soft thresholding *also shrinks the big ones* by $\lambda$, so **lasso is biased**. *Debiasing:* run lasso to find which weights are non-zero, then refit those by ordinary least squares (Fig. 11.13). The coordinate-descent loop in our code is exactly this update, cycled over $d$ (§11.4.9.1, p.397).
* **Regularization path and $\lambda_{\max}$ (§11.4.4, p.389–390, Eq. 11.90).** At $\lambda=0$ the solution is OLS (dense); as $\lambda$ grows it gets sparser; above $\lambda_{\max}$ all weights are 0, where $\lambda_{\max}=\max_d|\nabla_{w_d}\text{NLL}(\mathbf 0)|=\|\mathbf X^\top\mathbf y\|_\infty$. The path is **piecewise linear** in $\lambda$ (Table 11.1). If $D>N$ the lasso selects at most $N$ variables (the **elastic net**, ridge+lasso, §11.4.8, removes that limit).
* **Orthonormal design: OLS vs ridge vs lasso vs subset (§11.4.5, p.390–391, Eq. 11.93–11.96).** With $\mathbf X^\top\mathbf X=\mathbf I$: $\hat w^{\rm ols}=\mathbf x_{:d}^\top\mathbf y$, $\hat w^{\rm ridge}=\hat w^{\rm ols}/(1+\lambda)$, $\hat w^{\rm lasso}=\text{sign}(\hat w^{\rm ols})(|\hat w^{\rm ols}|-\lambda)_+$, $\hat w^{\rm subset}=\hat w^{\rm ols}$ if it ranks in the top $K$ else 0. **Prostate comparison (Fig. 11.11):** OLS test error 0.521, best subset 0.492, ridge 0.487, **lasso 0.457** (with standard errors 0.176 / 0.141 / 0.157 / 0.146); subset and lasso give *sparse* solutions (age, lcp, gleason → 0), ridge does not (age $-0.089$, lcp $-0.095$ remain).
* **Variable-selection consistency and group lasso (§11.4.6–11.4.7, p.392–396).** CV picks $\lambda$ for *prediction*, which "is not usually the same value as the one that is likely to recover the 'true' model", so the lasso with a CV-chosen $\lambda$ tends to keep some irrelevant variables (false positives). Group lasso zeroes out whole groups of weights together.

##### Murphy §4.5.3 (p.123–124) and Bishop §1.1 (p.10) / §3.1.4 (p.144–146)
* Murphy's "weight decay" example (Fig. 4.5): a **degree-14 polynomial** on 21 points. With $\lambda=0$ the curve is "very wiggly"; a tiny $\lambda=0.00019$ already tames it; too large ($\lambda=1.1$) underfits; the train error rises monotonically with $\lambda$ while the test error is **U-shaped**.
* **Bishop §3.1.4 "Regularized least squares" (p.144–146).** Adding $\frac\lambda2\mathbf w^\top\mathbf w$ gives the closed form $\mathbf w=(\lambda\mathbf I+\boldsymbol\Phi^\top\boldsymbol\Phi)^{-1}\boldsymbol\Phi^\top\mathbf t$; Bishop generalizes to $\frac\lambda2\sum_j|w_j|^q$ (his Eq. 3.29) and shows that $q=1$ is the lasso, with the same constraint-region picture as slide 12: "if $\lambda$ is sufficiently large, some of the coefficients $w_j$ are driven to zero, leading to a sparse model". (Bishop Figs. 3.3–3.4.)

##### Solved exercises and conceptual checks

**(a) A Gaussian prior gives ridge.** $p(\mathbf w)=\mathcal N(\mathbf 0,\tau^2\mathbf I)$ and $p(\mathbf y|\mathbf X,\mathbf w)=\mathcal N(\mathbf X\mathbf w,\sigma^2\mathbf I)$. Then $-\log p(\mathbf w|\mathcal D)=\frac1{2\sigma^2}\|\mathbf y-\mathbf X\mathbf w\|^2+\frac1{2\tau^2}\|\mathbf w\|^2+\text{const}$. Multiplying by $2\sigma^2$ (does not move the minimizer): $\|\mathbf y-\mathbf X\mathbf w\|^2+\frac{\sigma^2}{\tau^2}\|\mathbf w\|^2$ ✔ (Murphy Eq. 11.53).

**(b) Orthonormal case with numbers (Murphy Eq. 11.93–11.95; convention $\frac12\text{RSS}+\lambda\|\mathbf w\|_1$).** Suppose the OLS weight is $0.8$ and $\lambda=0.5$. Ridge: $0.8/(1+0.5)=0.533$. Lasso: $\text{sign}(0.8)(0.8-0.5)_+=0.30$. If instead OLS $=0.4$: ridge $0.267$, **lasso $=(0.4-0.5)_+=0$**, so lasso removes weak features and ridge only shrinks them. Subset selection with $K=1$ keeps the largest *unshrunk*.

**(c) Is the lasso solution unique/does a larger $\lambda$ always give a sparser answer?** Sparser *in general*, "although not necessarily monotonically" (Murphy p.389): the set of non-zeros can change both ways along the path, but the shrinking trend is clear.

**(d) Murphy Ex 11.5 (shrinkage with an orthonormal design).** For an orthonormal design $\hat w_k=\mathbf y^\top\mathbf x_{:,k}=c_k/2$, so OLS is a straight line of slope $\frac12$ against $c_k=2\mathbf y^\top\mathbf x_{:,k}$; ridge is a straight line with a *smaller slope* $\frac1{2(1+\lambda_2)}$; lasso is **flat at 0 for $|c_k|\le\lambda_1$** and then a line parallel to OLS but shifted by $\lambda_1/2$. These three shapes are what the exercise asks you to identify (the figure itself is not reproduced in my extract, so I cannot give the numbers $\lambda_1,\lambda_2$ read from it).

**(e) Why standardize features before lasso/ridge?** The penalty $\sum|\theta_i|^p$ treats all weights alike; if one feature is in metres and another in millimetres the weights have very different natural sizes, so the penalty would unfairly hit one of them. Standardizing (mean 0, std 1) puts all features on one scale — which is what the junk-feature example above does.

#### ➕ Textbook extra — elastic net, group lasso, and "Bayesian" regularization paths
Murphy §11.4.8 (**elastic net**: both $L^1$ and $L^2$ penalties; keeps groups of correlated features and can select more than $N$ variables), §11.4.7 (group lasso), §11.4.9 (algorithms: coordinate descent, projected gradient, proximal gradient, LARS), §11.4.6 (variable-selection consistency) and §11.7 (Bayesian linear regression, "automatic relevance determination") go beyond the slides. The slide's $p<1$ star and $p=4$ rounded square are special cases of the same $L^p$ family but are rarely used in practice.

#### ✅ Key takeaways

* **A regularizer is a prior:** $-\log p(\theta)$ is the penalty. $L^2$ penalty ⇔ Gaussian prior ($\lambda=\sigma^2/\tau^2$); $L^1$ penalty ⇔ Laplace prior.
* **MAP loss:** $J(\theta)=\text{MSE}(\theta)+\lambda\sum_i|\theta_i|^p$; the first term fits the data, the second keeps weights small. Larger $\lambda$ ⇒ simpler model (more bias, less variance).
* **Ridge** has the closed form $(\mathbf X^\top\mathbf X+\lambda\mathbf I)^{-1}\mathbf X^\top\mathbf y$; weights shrink smoothly, **never exactly zero**. Our slope: $5.02\to2.48$ at $\lambda=50$.
* **Lasso** has no closed form (use soft-thresholding / coordinate descent); weights shrink and become **exactly zero** ⇒ **feature selection** (hours kept, both junk features dropped at $\lambda=20$).
* **Geometry:** MSE contours are ellipses; the $L^2$ region is a round disc, the $L^1$ region is a diamond with corners on the axes where the ellipse typically touches ⇒ zeros.
* Don't penalize the intercept; standardize features; choose $\lambda$ by cross-validation.
* **MAP** = MLE with a prior; the prior is an **inductive bias**. Other examples: weight decay in deep learning and weight-initialization heuristics (Xavier).

---

## Lecture 15 — Multilayer Perceptron (MLP)

**About the slides.** About forty slides (the PDF has 34 pages; slide numbers run to 39 because the PDF skips a few). Order: recap of MAP and lasso → "can we *learn* the basis functions?" → neurons and the Nobel Prize 2024 (physics, for foundational neural-network work) → the **perceptron** and its algorithm (Iris examples) → why it is not expressive enough → **adaptive sigmoid bases** → the **MLP** $\hat{\mathbf y}=\mathbf g(\mathbf W\mathbf h(\mathbf V\mathbf x))$ → **probabilistic modelling** (output activation + loss) → **activation functions** → **network architecture** → the **universal approximation theorem** → depth vs width → **regularization for neural networks** (weight decay, data augmentation, noise robustness, early stopping, dropout) → summary.

**Big idea in one sentence.** In linear regression *we* choose the basis functions $\phi_k(x)$; a neural network **learns them from the data** — each hidden unit is a small logistic-regression-like model whose own weights are trained together with the output weights by gradient descent.

### 15.1 Where we are: from fixed bases to learned bases (slides 2–4)

* **Last time (Lectures 13–14):** the MAP estimate maximizes the posterior $p(\theta|\mathcal D)\propto p(\mathcal D|\theta)\,p(\theta)$ (posterior ∝ likelihood × prior).
* **Lasso selects basis functions.** The model $y=\theta^\top\phi(x,n)+\varepsilon$ fits non-linear data because the *inputs are represented differently* (basis expansion). If $n$ is too large we fit noise; the lasso sets some $\theta_i$ exactly to 0, so those features leave the model.
* **The new question (slide 4):** *"Are there methods for learning basis functions directly from data?"* — yes: neural networks.

*Running example.* So far, for "hours studied $h\to$ score $s$", we *chose* features such as $(1,h,h^2)$ or Gaussian bumps at $h=2,5,8$. A network will instead *discover* bumps/steps in $h$ that work well.

### 15.2 Neurons and the perceptron (slides 5–8)

A **neural network** is a network of many simple units ("neurons"); the idea is *biologically motivated* — a biological neuron adds up the signals arriving on its inputs and fires if the total is large enough. The slide shows a neuron diagram and the 2024 Nobel Prize in Physics announcement (awarded to John Hopfield and Geoffrey Hinton for foundational work on artificial neural networks).

**The perceptron** is the simplest building block:
$$f(\mathbf x)=\text{sign}(\mathbf w^\top\mathbf x+b).$$
**Plain words:** multiply each input by a weight, add them with a bias $b$, and output $+1$ if the total is positive, $-1$ if it is negative. Compare the three models on the slide:

| Model | Formula | Output |
|---|---|---|
| Linear regression | $f(\mathbf x)=\mathbf w^\top\mathbf x+b$ | any real number |
| Logistic regression | $f(\mathbf x)=\sigma(\mathbf w^\top\mathbf x+b)$ | probability in $(0,1)$ |
| **Perceptron** | $f(\mathbf x)=\text{sign}(\mathbf w^\top\mathbf x+b)$ | $+1$ or $-1$ (hard decision, like an SVM) |

*Symbols:* $\mathbf x\in\mathbb R^D$ inputs, $\mathbf w\in\mathbb R^D$ weights, $b$ bias, $\text{sign}(z)=+1$ if $z>0$ else $-1$, $\sigma(z)=1/(1+e^{-z})$.

*Running example.* With $w=0.8$ per hour and $b=-2.4$ a student who studies 4 h gets $0.8\cdot4-2.4=0.8>0\Rightarrow+1$ ("pass"); 2 h gives $-0.8\Rightarrow-1$ ("fail"). The decision boundary is $h=3$.

**History (slide 8).** Networks of perceptrons were the start of *connectionist AI*; the learning algorithm minimizes an "energy" (like probabilistic modelling). The book *Perceptrons: An Introduction to Computational Geometry* by **Minsky and Papert** (1969) showed that one perceptron cannot compute simple functions like XOR and was "a leading factor in the **first AI winter**" (years of reduced funding). Murphy §13.2.1 (p.427) calls XOR "one of the most famous examples from the *Perceptrons* book".

### 15.3 The perceptron algorithm (slide 9)

Targets are $y_i\in\{-1,+1\}$ and the prediction is $\hat y_i=\text{sign}(\mathbf w^\top\mathbf x_i+b)$. A point is **misclassified** exactly when the true and predicted signs differ:
$$y_i\hat y_i<0 .$$
For a misclassified point the number $y_i(\mathbf w^\top\mathbf x_i+b)$ is *negative*, so its negative is positive. Summing over the set $\mathcal M$ of misclassified points gives the **perceptron criterion** to minimize:
$$\mathbf w,w_0\in\arg\min_{\mathbf w,w_0}\ -\sum_{(\mathbf x_i,y_i)\in\mathcal M}y_i\,(\mathbf w^\top\mathbf x_i+w_0).$$
This resembles SVM optimization (Lecture 19 later). Slide note: *how this optimization problem is solved is beyond the scope of this course*. The classic rule (Bishop §4.1.7, p.192–196) is, for each misclassified point, $\mathbf w\leftarrow\mathbf w+y_i\mathbf x_i$ (with $\mathbf x_i$ augmented by a constant 1 for the bias): "push the boundary toward the point that was wrong".

*Worked example (4 points, bias as an extra input 1).* Points $(1,3),(2,2)$ have label $+1$; $(2,0),(3,1)$ have label $-1$. Start with $\mathbf w=(b,w_1,w_2)=(0,0,0)$ and sweep through the points in order:

| epoch | point | label | $y(\mathbf w^\top\tilde{\mathbf x})$ before | update | new $\mathbf w=(b,w_1,w_2)$ |
|---|---|---|---|---|---|
| 1 | $(1,3)$ | $+1$ | $0$ (not $>0$) | $+(1,1,3)$ | $(1,1,3)$ |
| 1 | $(2,2)$ | $+1$ | $1+2+6=9>0$ ✔ | none | $(1,1,3)$ |
| 1 | $(2,0)$ | $-1$ | $-(1+2)=-3<0$ ✘ | $-(1,2,0)$ | $(0,-1,3)$ |
| 1 | $(3,1)$ | $-1$ | $-(0-3+3)=0$ ✘ | $-(1,3,1)$ | $(-1,-4,2)$ |
| 2 | $(1,3)$ | $+1$ | $-1-4+6=1>0$ ✔ | none | |
| 2 | $(2,2)$ | $+1$ | $-1-8+4=-5<0$ ✘ | $+(1,2,2)$ | $(0,-2,4)$ |
| 3 | all four | | $10,\ 4,\ 4,\ 2>0$ ✔ | none | **converged** |

Final classifier: $\text{sign}(-2x_1+4x_2)$, which labels $(1,3)\to+1$, $(2,2)\to+1$, $(2,0)\to-1$, $(3,1)\to-1$ ✔. Four updates in total.

**Convergence (slides 13–14, Iris).**

* *Linearly separable* (Iris **Setosa** vs rest using petal length and petal width): the line moves a little each iteration (iterations 1, 2, 10 on the slide) and then correctly detects Setosa as $+1$. The **perceptron convergence theorem** (Bishop p.194) says this always happens in a finite number of steps when a separating line exists.
* *Not linearly separable* (Iris **Virginica** vs rest in the same two features): the algorithm "will **oscillate** (and not converge)" — some point is always wrong, so the updates cycle forever.

### 15.4 Towards a more expressive model (slide 15)

The perceptron "cannot model data that is not linearly separable. In that case the algorithm gets stuck in cyclic updates." *The XOR function* is the standard example (Murphy Table 13.1, p.426): $y=1$ exactly when $x_1\ne x_2$:

| $x_1$ | $x_2$ | $y$ |
|---|---|---|
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 0 |

No straight line separates $\{(0,1),(1,0)\}$ from $\{(0,0),(1,1)\}$. Running the algorithm above on XOR from $\mathbf w=0$ makes **4 mistakes in every one of the 12 epochs** and returns to $\mathbf w=0$ at the end of each epoch (verified in the code below): a perfect cycle.

**Two fixes the slide proposes.**

1. **A set of fixed basis functions** $\boldsymbol\phi(\mathbf x)$ — as in linear regression with a polynomial basis. (For XOR, adding the feature $x_1x_2$ makes it separable: $x_1+x_2-2x_1x_2$ equals XOR exactly.)
2. **Adaptive bases:** *learn the parameters of the bases as well.* This is the neural network.

### 15.5 Adaptive sigmoid bases = a two-layer network (slide 18)

Each basis function $m$ is itself a logistic-regression model:
$$\phi_m(\mathbf x)=\sigma(\mathbf v_m^\top\mathbf x+b_m),\qquad \hat y=\sum_m w_m\,\phi_m(\mathbf x).$$
**Plain words:** instead of fixing sigmoid steps at hand-picked positions $\mu_k$ (Lecture 14), let each step have its own direction $\mathbf v_m$ and offset $b_m$ — and **optimize $\mathbf w$ and all $\mathbf v_m,b_m$ with gradient descent**. The slide draws this for input dimension 1 ($x$ and a constant 1 feeding hidden nodes $\phi_1\dots\phi_M$, which feed $\hat y$ through weights $w_1\dots w_M$) and for input dimension $D$ ($x_1,\dots,x_D$ and 1). It is "a neural network with two layers" (two layers of *weights*).

*Running example with numbers.* Take two hidden units on $h$: $\phi_1(h)=\sigma(2(h-3))$ (step near 3 h) and $\phi_2(h)=\sigma(2(h-6))$ (step near 6 h), output weights $w=(40,\,20)$ and bias $30$. A student with $h=4$: $\phi_1=\sigma(2)=0.881$, $\phi_2=\sigma(-4)=0.018$, so $\hat s=30+40(0.881)+20(0.018)=65.6$; with $h=7$: $\phi_1=\sigma(8)\approx1.000$, $\phi_2=\sigma(2)=0.881$, $\hat s=30+40+17.6=87.6$. Training would move the steps (the $v,b$) and the heights (the $w$) to fit the scores — the model *chooses its own basis functions*.

### 15.6 The multilayer perceptron (slide 19)

Suppose

* $D$ inputs $x_1,\dots,x_D$ (plus a constant 1 for the bias),
* $M$ hidden units $z_1,\dots,z_M$ (plus a constant 1),
* $C$ outputs $\hat y_1,\dots,\hat y_C$.

The model is
$$\hat y_c=g\Big(\sum_m W_{c,m}\,h\Big(\sum_d V_{m,d}\,x_d\Big)\Big)\qquad\text{or compactly}\qquad \hat{\mathbf y}=\mathbf g\big(\mathbf W\,\mathbf h(\mathbf V\mathbf x)\big),$$
where $\mathbf V\in\mathbb R^{M\times D}$ (first-layer weights, *input→hidden*), $\mathbf W\in\mathbb R^{C\times M}$ (second-layer weights, *hidden→output*), $h$ is the **hidden activation function** applied element-wise, and $g$ is the **output activation**. (Biases are folded in by appending a 1 to $\mathbf x$ and to $\mathbf h$.)

*Step-by-step with real numbers.* Let $D=2$, $M=2$, $C=1$, $h=$ ReLU, $g=$ identity and
$$\mathbf V=\begin{pmatrix}0.5&-1\\1&0.5\end{pmatrix},\quad \mathbf W=\begin{pmatrix}1&0.5\end{pmatrix},\quad \mathbf x=\begin{pmatrix}1\\2\end{pmatrix},\quad y=3 .$$
Forward pass:

1. Pre-activations $\mathbf a=\mathbf V\mathbf x=(0.5\cdot1-1\cdot2,\ 1\cdot1+0.5\cdot2)=(-1.5,\ 2)$.
2. Hidden activations $\mathbf z=\text{ReLU}(\mathbf a)=(0,\ 2)$.
3. Output $\hat y=\mathbf W\mathbf z=1\cdot0+0.5\cdot2=1$.
4. Loss $L=\tfrac12(\hat y-y)^2=\tfrac12(1-3)^2=2$.

Backward pass (this is **backpropagation** = the chain rule applied layer by layer from the output back; Murphy §13.3, Bishop §5.3 p.241–245):

* $\partial L/\partial\hat y=\hat y-y=-2$.
* $\partial L/\partial\mathbf W=(-2)\,\mathbf z^\top=(0,\ -4)$.
* $\partial L/\partial\mathbf z=\mathbf W^\top(-2)=(-2,\ -1)$.
* Through the ReLU: multiply by $\mathbb I(a>0)=(0,1)$ ⇒ $\partial L/\partial\mathbf a=(0,\ -1)$ (unit 1 is *inactive*, so it receives no gradient).
* $\partial L/\partial\mathbf V=(0,-1)^\top\mathbf x^\top=\begin{pmatrix}0&0\\-1&-2\end{pmatrix}$.

(A numerical finite-difference check gives the same numbers.) One gradient-descent step with learning rate $\alpha=0.1$: $\mathbf V\leftarrow\begin{pmatrix}0.5&-1\\1.1&0.7\end{pmatrix}$, $\mathbf W\leftarrow(1,\ 0.9)$. New forward pass: $\mathbf a=(-1.5,\ 2.5)$, $\mathbf z=(0,\ 2.5)$, $\hat y=2.25$, $L=0.28125$ — the loss dropped from $2$ to $0.28$ and the prediction moved from $1$ toward the target $3$.

### 15.7 Probabilistic modelling with neural networks (slides 20–21)

The output dimension and the output activation $g$ depend on the task; they define an **output distribution**, and the loss is its negative log-likelihood (Lectures 07–09):

| Task | Output activation $g$ | Likelihood | Loss |
|---|---|---|---|
| Regression | identity | $\mathcal N(\mathbf y;\hat{\mathbf y},\mathbf I)$ | $L=\tfrac12\lVert\mathbf y-\hat{\mathbf y}\rVert_2^2=-\log\mathcal N(\mathbf y;\hat{\mathbf y},\mathbf I)$ + const (MSE) |
| Binary classification | sigmoid $\sigma$ | Bernoulli | **binary cross-entropy** $-\frac1D\sum_i\big[y_i\log\sigma(f_\theta(\mathbf x_i))+(1-y_i)\log(1-\sigma(f_\theta(\mathbf x_i)))\big]$ |
| Multiclass ($C$ classes) | softmax | Categorical | **cross-entropy** $-\frac1D\sum_i\sum_{c=1}^C\mathbb I(c=y_i)\log\text{softmax}(\mathbf f_\theta(\mathbf x_i))_c$ |

More generally the network can output the **parameters of any distribution** (e.g., a mean *and* a variance for a normal; Murphy §13.2.4.4 "heteroskedastic regression", p.433–434) and the loss is $L(\theta;\mathcal D)=-\sum_{(\mathbf x,y)\in\mathcal D}\log p\big(y\mid f_\theta(\mathbf x)\big)$.

*Numbers.* Binary: a student with pass-logit $f=1.2$ and true label 1: $\sigma(1.2)=0.769$, loss $=-\log0.769=0.263$; if the label were 0 the loss would be $-\log(1-0.769)=1.46$. Three classes, logits $(2,1,0)$, true class = the first: $\text{softmax}=(0.665,0.245,0.090)$, loss $=-\log0.665=0.408$.

### 15.8 Activation functions (slides 22–26)

Inside the network we can choose any activation $h$. Writing $\hat{\mathbf y}=\mathbf g(\mathbf W\mathbf h(\mathbf V\mathbf x))$:

* **Identity** $h(x)=x$ — nothing is gained, because
$$\mathbf W\mathbf V\mathbf x=\mathbf W'\mathbf x\qquad(\underbrace{c\times m}_{\mathbf W}\cdot\underbrace{m\times d}_{\mathbf V}=\underbrace{c\times d}_{\mathbf W'}),$$
the product of two matrices is just one matrix. *Unless $m<\min(d,c)$*: then the hidden layer **compresses** the data and $\mathbf W'$ has **low rank** ($\le m$). Check: random $\mathbf W\in\mathbb R^{3\times2}$, $\mathbf V\in\mathbb R^{2\times4}$ gives a $3\times4$ matrix of rank 2. (Murphy Eq. 13.10, p.428: with linear activations "the whole model reduces to a regular linear model".)
* **Sigmoid** $h(x)=\sigma(x)=\dfrac1{1+e^{-x}}$. Away from zero the function changes slowly, so its derivative is small — **vanishing gradients**. The derivative is $\sigma'(x)=\sigma(x)(1-\sigma(x))\le0.25$.
* **Hyperbolic tangent** $h(x)=2\sigma(2x)-1=\tanh(x)$. (The slide writes $2\sigma(x)-1$; that equals $\tanh(x/2)$, **not** $\tanh(x)$ — see Errata.) Symmetric around the origin, where it is near linear; its derivative $1-\tanh^2(x)$ is again small away from zero.
* **ReLU** $h(x)=\max(0,x)$. "Usually improves the training of deep networks over sigmoid." Derivative is 1 for $x>0$ and 0 for $x<0$: **zero derivative if the unit is inactive** ("dead ReLU", Murphy p.450); initialization should make units output values above but close to zero.
* **Leaky ReLU** $h(x)=\max(0,x)+\gamma\min(0,x)$: fixes the zero-gradient problem but adds a hyper-parameter $\gamma$.
* **Softplus** $h(x)=\log(1+e^x)$: differentiable everywhere (a smooth ReLU) but "does not perform that well in practice".

Values (the numbers to remember):

| $x$ | $-2$ | $-1$ | $0$ | $1$ | $2$ |
|---|---|---|---|---|---|
| sigmoid $\sigma(x)$ | 0.119 | 0.269 | 0.5 | 0.731 | 0.881 |
| $\sigma'(x)$ | 0.105 | 0.197 | 0.25 | 0.197 | 0.105 |
| $\tanh(x)$ | −0.964 | −0.762 | 0 | 0.762 | 0.964 |
| $\tanh'(x)$ | 0.071 | 0.420 | 1 | 0.420 | 0.071 |
| ReLU | 0 | 0 | 0 | 1 | 2 |
| leaky ReLU ($\gamma=0.1$) | −0.2 | −0.1 | 0 | 1 | 2 |
| softplus | 0.127 | 0.313 | 0.693 | 1.313 | 2.127 |

**Why gradients vanish.** Backprop multiplies one derivative per layer. With sigmoids each factor is at most $0.25$, so after $L$ layers the signal is at most $0.25^L$: $0.25,\ 0.0625,\ 0.00098$ (5 layers), $9.5\times10^{-7}$ (10 layers). The early layers then barely learn. ReLU has derivative exactly 1 on the active side, so the product does not shrink (Murphy §13.4.2, p.447–448: the gradient behaviour depends on the spectral radius of the layer Jacobians; $<1$ vanishes, $>1$ explodes; exploding gradients are tamed by **gradient clipping**, vanishing ones by ReLU-type activations and residual connections).

### 15.9 Network architecture (slides 27–29)

The **architecture** is the overall structure of the network.

* **Depth** = number of layers; **width** = number of units in a layer. We can increase either.
* **Fully connected** layers connect every unit to every unit of the next layer; **sparsely connected** layers keep only some connections (this is what convolutional networks do).
* **Skip connections** let a layer feed a layer several steps later; different units can have different activations.
* **Feed-forward** architectures are trained with **backprop**. In general a feed-forward architecture is a **directed acyclic graph (DAG)**; cycles would make backprop loop forever (Murphy p.425–426: DNNs "compose differentiable functions into any kind of DAG"; a chain is the simplest DAG).

**Parameter counting (a useful check).** A fully connected MLP with $D$ inputs, $M$ hidden units and $C$ outputs, with biases, has $M(D+1)+C(M+1)$ parameters. For MNIST ($D=784$, $M=100$, $C=10$): $100\cdot785+10\cdot101=78{,}500+1{,}010=79{,}510$.

**Interactive demo (slide 30).** The slide links to an online interactive demonstration of a small network; Murphy §13.2.4.1 (p.431) shows the same kind of example, "MLP for classifying 2d data into 2 categories".

### 15.10 Universal approximation (slide 31)

**Theorem (informal).** An MLP with a **single hidden layer** can approximate any continuous function to arbitrary accuracy. (Bishop p.230; Murphy §13.2.5 p.434: "given enough hidden units, to any desired level of accuracy"; the proofs are by Cybenko 1989 and Hornik et al.) Intuition: each hidden unit makes a smooth step; sums of steps of different heights and positions can build any bump, and bumps can build any shape.

*Experiment (code below).* Fit $y=\sin(2\pi x)$ on $[0,1]$ using $M$ random sigmoid hidden units (fix random $\mathbf v,b$; fit only the output weights by least squares). Maximum error over 200 points:

| hidden units $M$ | 2 | 5 | 10 | 20 | 50 |
|---|---|---|---|---|---|
| max error | 1.13 | 0.037 | 0.0004 | $10^{-8}$ | $10^{-10}$ |

With 2 units the fit is useless (error 1.13 is as large as the sine itself), with 5 it is already close (0.037), and by 20 it is numerically exact — "enough hidden units" really does work in 1-D. (The exact numbers depend on the random seed; the trend does not.)

**Limitations (slide).**

1. For $D$ inputs "the number of basis functions needed grows exponentially with the input dimension" — the **curse of dimensionality**: a grid with 10 points per axis has $10,\ 100,\ 1000$ points for $D=1,2,3$ and $10^{10}$ for $D=10$.
2. "We may need a **very wide network**" (a large basis function set).
3. The theorem only says the function **can be fitted** (training error). "The fitted function is not guaranteed to generalize to an independent test set." Having the power to fit anything is exactly why overfitting is a danger (Section 15.12).

### 15.11 Depth versus width (slide 32)

* Deep ReLU networks of *bounded width* are also universal.
* **Empirically, increasing depth is often more effective than increasing width** (the slide's picture, Goodfellow Fig. 6.5, shows *why*: each ReLU layer **folds** the input space onto itself like a sheet of paper — three snapshots of one fold — so a few layers create exponentially many linear pieces, and one picture of the data is reused on every piece).
* A deep network gives a **compositional functional form** — each layer builds on the features from the previous layer — and that is "a useful inductive bias". Murphy's example (p.434–435): to detect the DNA pattern `*AA??CGCG??AA*` it is easier for layer 1 to detect the "AA" and "CG" motifs and layer 2 to combine them, "analogously to how we solved the XOR problem". Summary slide: *"More complex bases at higher layers (a.k.a. more 'abstract' representations)."*

### 15.12 Regularization strategies for neural networks (slides 33–38)

"Similar to previous regression examples, neural networks may overfit." The list on the slide: **L1 and L2 regularization (weight decay)**, **data augmentation**, **noise robustness**, **early stopping**, **dropout**, sparse representations (e.g., an L1 penalty on hidden-unit activations), and semi-supervised and multi-task learning. The five main ones:

**1. Weight decay ($L^2$/$L^1$).** Exactly Lecture 14: add $\lambda\sum|\theta_i|^p$ to the loss. Murphy §13.5.2 (p.455): a Gaussian prior $\mathcal N(\mathbf w|0,\alpha^2\mathbf I)$ on the weights "is equivalent to $\ell_2$ regularization … called *weight decay* since it encourages small weights". The name comes from the update: with penalty $\frac\lambda2\lVert\mathbf w\rVert^2$ each gradient step multiplies $\mathbf w$ by $(1-\alpha\lambda)$ before the usual step, so weights *decay* geometrically ($w=1$ with $\alpha\lambda=0.01$ becomes $0.99$ after one step).

**2. Data augmentation (slide 34).** More data generalizes better (we sample the "data manifold" more densely). Create new training examples by transforming old ones — for images: (1) **add noise to the input**; (2) **add noise to hidden units** (noise at higher levels of abstraction); (3) **learn a generative model** that produces image variants. The slide shows one photo (labelled "hamster" on the slide; it is the sample animal photo from the imgaug README) and a grid of its flipped, rotated, cropped and re-coloured variants (the imgaug library). Bishop §5.5.3 "Invariances" (p.261) is the textbook version: the model should give the same output under shifts, rotations, etc.

**3. Noise robustness / flat minima (slide 35).** "Flat minima typically generalize better." SGD (minibatches) is used not only to save compute but to *intentionally inject noise* into the optimization; smaller batches mean more gradient noise, so the algorithm tends to *step over sharp minima* and settle in flat ones (the slide cites Keskar et al. 2017, arXiv:1609.04836). Picture: if the test loss surface is slightly shifted from the training one, a sharp valley moves a lot (large test error) while a flat valley barely changes.

**4. Early stopping (slide 36).** Training too long makes the test error rise again: test loss versus time step is "U" shaped (Goodfellow Fig. 7.3). Keep the parameters from the epoch with the lowest **validation** loss. It "bounds the region of parameter space reachable in $T$ time steps" — starting from small weights, $T$ small steps cannot go far — and "can be thought of as a form of $L^2$ regularization" (Goodfellow Fig. 7.4: the early-stopping path stops before the unregularized optimum, like the weight-decay solution). Textbook: Murphy §13.5.1 p.455; Bishop §5.5.2 p.259–262 ("early stopping can give similar results to weight decay").

**5. Dropout (slides 37–38).** "Dropout randomly removes a subset of units during training. This is one of the most effective regularization schemes for MLPs."

* **During training:** for each data point $(\mathbf x_i,y_i)$ randomly drop each unit with probability $p$ (e.g. $p=0.5$); only the remaining *sub-network* participates in that step.
* **Test time:** in theory we would average the predictions of all possible sub-networks — computationally infeasible. Instead: (1) **Monte Carlo dropout:** average the predictions of several forward passes *with dropout on*; (2) **weight scaling:** rescale the weights to compensate for the missing units.
* **Careful with the scale** (slide says "scale weights by $p$"): if $p$ is the probability of *dropping*, each unit is present with probability $1-p$, so the expected input to the next layer during training is $(1-p)$ times the full one, and the weights must be multiplied by the **keep probability $1-p$** (Murphy p.456: "multiply the weights by the keep probability, $1-p$"). With $p=0.5$ both coincide, which is why the slide's example works.

*Numbers.* Hidden activations $\mathbf h=(1,2,3,4)$, all outgoing weights $0.5$, so the full output is $0.5\cdot10=5$. Simulating $200{,}000$ random masks: with drop probability $p=0.5$ the training-time average output is $2.50$ ($=(1-p)\cdot5$); with $p=0.2$ it is $4.00$ ($=0.8\cdot5$). Scaling by $p=0.2$ would give $1.0$ — wrong — while scaling by $1-p=0.8$ gives $4.0$ ✔. Why it helps: no unit can rely on another being present ("prevents complex co-adaptation of the hidden units", Murphy p.455), like training an ensemble of exponentially many thinned networks that share weights.

### 15.13 Summary (slide 39)

* Deep feed-forward networks **learn adaptive bases**; higher layers give more complex, more **abstract** representations.
* **Increasing depth** is often preferable to increasing width.
* They have **universal approximation power** — and that power **necessitates using regularization**.

```python
# continues from the running-example setup (Part F.1)
import numpy as np
sg = lambda z: 1 / (1 + np.exp(-z))

# --- perceptron (bias as an extra input 1): w <- w + y*x for each misclassified point
def perceptron(X, y, epochs=50):
    Xa = np.c_[np.ones(len(X)), X]; w = np.zeros(3); hist = []
    for ep in range(epochs):
        mis = 0
        for i in range(len(X)):
            if y[i] * (Xa[i] @ w) <= 0:
                w = w + y[i] * Xa[i]; mis += 1
        hist.append(mis)
        if mis == 0: break
    return w, hist
P = np.array([[1, 3], [2, 2], [2, 0], [3, 1.]]); yp = np.array([1, 1, -1, -1])
print(perceptron(P, yp))                    # (array([ 0., -2.,  4.]), [3, 1, 0]) -> 4 updates, converged
XOR = np.array([[0, 0], [0, 1], [1, 0], [1, 1.]]); yx = np.array([-1, 1, 1, -1])
print(perceptron(XOR, yx, 12)[1])           # [4, 4, 4, 4, ...]  never converges

# --- XOR with one hidden layer (Murphy 13.2.1): h1 = AND, h2 = OR, y = (not h1) AND h2
H = lambda a: (a > 0).astype(int)
for x in XOR:
    h1 = H(x @ [1, 1] - 1.5); h2 = H(x @ [1, 1] - 0.5); print(x, H(-h1 + h2 - 0.5))   # 0 1 1 0

# --- forward and backward pass of the worked example (ReLU hidden layer, identity output, L = 1/2 (yhat-y)^2)
x = np.array([1., 2.]); V = np.array([[.5, -1], [1, .5]]); W = np.array([[1, .5]]); y = np.array([3.])
a = V @ x; z = np.maximum(a, 0); yhat = W @ z; L = 0.5 * ((yhat - y) ** 2).sum()
print(a, z, yhat, L)                         # [-1.5  2.] [0. 2.] [1.] 2.0
dy = yhat - y; dW = np.outer(dy, z); dz = W.T @ dy; da = dz * (a > 0); dV = np.outer(da, x)
print(dW, dV)                                # dW = [[0, -4]],  dV = [[0, 0], [-1, -2]]
V2, W2 = V - 0.1 * dV, W - 0.1 * dW
print(0.5 * ((W2 @ np.maximum(V2 @ x, 0) - y) ** 2).sum())    # 0.28125

# --- linear activations collapse to one matrix of rank <= m
rng = np.random.default_rng(0)
Wm, Vm = rng.normal(size=(3, 2)), rng.normal(size=(2, 4))
print((Wm @ Vm).shape, np.linalg.matrix_rank(Wm @ Vm))        # (3, 4) 2

# --- universal approximation: random sigmoid hidden units, fit output weights by least squares
xg = np.linspace(0, 1, 200); target = np.sin(2 * np.pi * xg)
for M in (2, 5, 10, 20, 50):
    r = np.random.default_rng(1); v = r.normal(0, 10, M); b = -v * r.uniform(0, 1, M)
    Phi = np.c_[sg(np.outer(xg, v) + b), np.ones_like(xg)]
    w, *_ = np.linalg.lstsq(Phi, target, rcond=None)
    print(M, round(np.abs(Phi @ w - target).max(), 4))        # 1.1298 0.037 0.0004 0.0 0.0

# --- dropout: keep each unit with probability 1-p; test time multiplies weights by (1-p)
hv = np.array([1, 2, 3, 4.]); wv = np.full(4, .5); rng = np.random.default_rng(0)
for p in (0.5, 0.2):
    masks = (rng.random((200000, 4)) >= p).astype(float)
    print(p, (masks * hv * wv).sum(1).mean().round(3), (1 - p) * (hv * wv).sum())   # 2.499 2.5 / 3.999 4.0

# --- one MC-dropout prediction for a tiny net (average of S noisy forward passes)
def mc_dropout(x, V, W, p=0.5, S=1000, seed=0):
    r = np.random.default_rng(seed); out = []
    for _ in range(S):
        m = (r.random(V.shape[0]) >= p)               # drop hidden units
        out.append(W @ (np.maximum(V @ x, 0) * m))    # no rescaling: we average S samples instead
    return np.mean(out, 0)
```

#### 📖 Textbook deep dive

##### Murphy Chapter 13 (p.425–458) — neural networks for tabular data
* **§13.1 Introduction (p.425).** Linear models assume a linear input→output map. A basis expansion $f(\mathbf x;\boldsymbol\theta)=\mathbf W\boldsymbol\phi(\mathbf x)+\mathbf b$ (13.1) is still linear in $\boldsymbol\theta=(\mathbf W,\mathbf b)$ but "having to specify the feature transformation by hand is very limiting". Fix: give the feature extractor its own parameters, $f=\mathbf W\boldsymbol\phi(\mathbf x;\boldsymbol\theta_2)+\mathbf b$ (13.2), and repeat recursively: $f(\mathbf x;\boldsymbol\theta)=f_L(f_{L-1}(\cdots f_1(\mathbf x)\cdots))$ (13.3) — "the key idea behind deep neural networks". This is exactly the slide's "learn the bases". A chain of functions is a **feedforward network (FFNN) = multilayer perceptron (MLP)**; MLPs expect a fixed-length input vector (tabular data: an $N\times D$ design matrix); images and text are handled by CNNs/RNNs (later chapters).
* **§13.2.1 XOR (p.426–428).** Table 13.1 truth table; one perceptron cannot do it. An MLP with three perceptrons does: $h_1=x_1\wedge x_2$ (weights 1, 1, bias $-1.5$; check $[1,1]\cdot[1,1]-1.5=0.5>0$), $h_2=x_1\vee x_2$, and $y=\bar h_1\wedge h_2=\neg(x_1\wedge x_2)\wedge(x_1\vee x_2)$, which equals XOR. "By generalizing this example, we can show that an MLP can represent any logical function." Footnote: unlike a CNN, an MLP "makes no assumptions about its inputs".
* **§13.2.2 Differentiable MLPs (p.428).** The Heaviside step makes the network non-differentiable and hard to train ("why they were never widely used"); replace $H(a)=\mathbb I(a>0)$ by a smooth activation (e.g. $\sigma$) and use gradient-based training.
* **§13.2.3 Activations (p.428–430, Fig. 13.2).** A linear activation $\varphi_\ell(a)=c_\ell a$ collapses the model to $\mathbf W'\mathbf x$ (13.10). Sigmoid and tanh saturate ⇒ vanishing gradients; ReLU $=\max(a,0)$ is the standard non-saturating choice.
* **§13.2.5 The importance of depth (p.434–435).** One hidden layer is a universal approximator "given enough hidden units"; intuitively each hidden unit specifies a half-plane and a large enough combination can "carve up any region of space" (Fig. 13.7 shows the linear regions of a ReLU network). But deep networks work better "because later layers can leverage the features that are learned by earlier layers; the function is defined in a compositional or hierarchical way".
* **§13.4.2 Vanishing and exploding gradients (p.447–448).** $\partial L/\partial\mathbf z_l=\mathbf J_l\mathbf g_{l+1}$ (13.69). If the spectral radius of the Jacobian is $>1$ gradients explode, $<1$ they vanish. Cure for exploding: **gradient clipping** $\mathbf g'=\min(1,c/\lVert\mathbf g\rVert)\mathbf g$ (13.70). Vanishing is harder: non-saturating activations, **residual connections** (§13.4.4), good initialization.
* **§13.4.3 Non-saturating activations (p.449–451).** ReLU $\max(a,0)$ is 0 for $a<0$, so a unit whose input is always negative has zero gradient forever — the **dead ReLU problem**. **Leaky ReLU** $\max(a,0)+\alpha\min(a,0)$ (the slide's $\gamma$) fixes it; **ELU/SELU/Swish/GELU** are smooth relatives.
* **§13.4.5 Initialization (p.452–453).** For a unit with $n_{\rm in}$ inputs of variance 1 and weights of variance $\sigma^2$, the output variance is $n_{\rm in}\sigma^2$, so signals stay stable if $\sigma^2=1/n_{\rm in}$ (forward pass) and $\sigma^2=1/n_{\rm out}$ (backward). Compromise: $\sigma^2=\dfrac2{n_{\rm in}+n_{\rm out}}$ (13.95), **Xavier/Glorot initialization**. $\sigma^2=1/n_{\rm in}$ is LeCun; $\sigma^2=2/n_{\rm in}$ is **He** (recommended with ReLU). A uniform alternative: $w\sim\text{Unif}(-a,a)$ with $a=\sqrt{6/(n_{\rm in}+n_{\rm out})}$ (since $\text{var}=a^2/3$). *Students:* a layer with $n_{\rm in}=784,\ n_{\rm out}=100$ gets $\sigma^2=2/884=0.00226$, $\sigma=0.0476$. Tie to Lecture 14: choosing this distribution *is* a prior on the initial weights.
* **§13.5 Regularization (p.455–456).** Early stopping (13.5.1), weight decay as MAP with a Gaussian prior (13.5.2), sparse DNNs (13.5.3), **dropout** (13.5.4): drop each neuron with probability $p$ per example; "prevents complex co-adaptation of the hidden units"; at test time multiply weights by the keep probability $1-p$, or use **Monte Carlo dropout** $p(y|\mathbf x,\mathcal D)\approx\frac1S\sum_{s=1}^Sp(y|\mathbf x,\hat{\mathbf W}\boldsymbol\epsilon_s+\hat{\mathbf b})$ (13.96), which approximates the Bayesian posterior predictive. Also (13.5.6) the implicit regularization of SGD (the "flat minima" idea) and (13.5.7) over-parameterized models.

##### Bishop Chapter 5 (p.225–290)
* **§5.1 Feed-forward network functions (p.227–231).** Bishop's two-layer network is $y_k(\mathbf x,\mathbf w)=\sigma\big(\sum_jw^{(2)}_{kj}\,h(\sum_iw^{(1)}_{ji}x_i+w^{(1)}_{j0})+w^{(2)}_{k0}\big)$ — exactly the slide's $\mathbf g(\mathbf W\mathbf h(\mathbf V\mathbf x))$ with biases written out. *Approximation capability (p.230):* networks with one hidden layer are "universal approximators". *Weight-space symmetries (§5.1.1, p.231):* swapping two hidden units (and their weights) or flipping the sign of all weights into a $\tanh$ unit gives the same function, so a network with $M$ hidden units has $M!\,2^M$ equivalent weight vectors — which is why the loss surface has many equal minima.
* **§5.2 Network training (p.232–236).** Likelihood-derived error functions: sum-of-squares for regression, cross-entropy for classification, with the **canonical link** property that the output gradient is always $\partial E/\partial a_k=y_k-t_k$ — check with our numbers: $\hat y-y=1-3=-2$ ✔.
* **§5.3 Error backpropagation (p.241–245).** For a unit $j$: $\delta_j\equiv\partial E_n/\partial a_j=h'(a_j)\sum_kw_{kj}\delta_k$ (Bishop 5.56) — the "backward" recursion we used: $\delta_{\rm hidden}=\mathbb I(a>0)\cdot(\mathbf W^\top\delta_{\rm out})$. Algorithm: forward-propagate, compute output $\delta$'s, back-propagate $\delta$'s, then $\partial E_n/\partial w_{ji}=\delta_jz_i$.
* **§5.5 Regularization (p.256–265).** Weight decay (5.5), consistent Gaussian priors that respect scaling of inputs/outputs (§5.5.1), **early stopping** (§5.5.2: "training error decreases monotonically … while the validation error shows a minimum" — p.259–262), and **invariances** by augmenting the training set with transformed copies (§5.5.3, p.261).

##### Solved exercises and conceptual checks

**(a) Prove that a network with identity hidden activation is a linear model.** $\hat{\mathbf y}=\mathbf W(\mathbf V\mathbf x)=(\mathbf W\mathbf V)\mathbf x$, a single matrix $\mathbf W'=\mathbf W\mathbf V$; its rank is at most $\min(\text{rank}\,\mathbf W,\text{rank}\,\mathbf V)\le m$, so it can only map into an $m$-dimensional subspace of the output space.

**(b) Why can't one perceptron do XOR? (algebra).** We would need $b<0$ (for $(0,0)\to$ class 0), $w_1+b>0$ and $w_2+b>0$ (for $(1,0),(0,1)\to$ class 1), but $w_1+w_2+b<0$ (for $(1,1)\to$ class 0). Adding the middle two: $w_1+w_2+2b>0$; subtracting $b<0$ gives $w_1+w_2+b>-b>0$, which contradicts $w_1+w_2+b<0$. ∎

**(c) XOR with ReLU units (two-line version).** $h_1=\max(0,x_1+x_2)$, $h_2=\max(0,x_1+x_2-1)$, $y=h_1-2h_2$. Check: $(0,0)\to0$; $(0,1),(1,0)\to1-0=1$; $(1,1)\to2-2\cdot1=0$ ✔ (verified in code).

**(d) How many parameters? and why does initialization matter?** MNIST $784\to100\to10$ has $79{,}510$ parameters (above). If all weights were initialized to the same value, every hidden unit would receive identical gradients and stay identical forever (the symmetry problem), so initialization must be random — which Xavier/He initialization does with the right *scale*.

**(e) Sigmoid vs ReLU after 10 layers.** Best case sigmoid gradient factor $0.25^{10}\approx9.5\times10^{-7}$; ReLU on the active side keeps factor 1. That is why ReLU "usually improves the training of deep networks".

**(f) Weight decay in the update.** Loss $L+\frac\lambda2\lVert\mathbf w\rVert^2$, step $\alpha$: $\mathbf w\leftarrow\mathbf w-\alpha(\nabla L+\lambda\mathbf w)=(1-\alpha\lambda)\mathbf w-\alpha\nabla L$. With $\alpha=0.1,\lambda=0.1$ a weight $w=2$ with $\nabla L=0$ becomes $2\cdot0.99=1.98$. Ties directly to ridge (Lecture 14).

**(g) Is early stopping really like $L^2$?** In a quadratic loss, gradient descent started at 0 shrinks the components with small curvature the least (they are learned slowest), which looks like ridge's shrink factor $\sigma^2/(\sigma^2+\lambda)$: directions with a large singular value are fitted first, those with a small one are barely moved if training stops early (Goodfellow §7.8, shown on the slide).

#### ➕ Textbook extra
Murphy §13.4.4 **residual connections** (the skip connections of the architecture slide, the basis of ResNets), §13.4.6 parallel training, §13.5.5 **Bayesian neural networks**, §13.5.7 over-parameterized models ("double descent"), §13.6 other feed-forward nets (radial basis function networks, mixtures of experts), and Murphy Chapters 14–15 (CNNs, RNNs) go beyond this lecture; Bishop §5.4–5.7 (Hessian, Bayesian networks, mixture density networks) likewise. Batch normalization (mentioned in Lecture 14) is Murphy §14.2.4.1.

#### ✅ Key takeaways
* **Perceptron** $=\text{sign}(\mathbf w^\top\mathbf x+b)$; learns by $\mathbf w\leftarrow\mathbf w+y\mathbf x$ on mistakes; **converges only if the data are linearly separable** (our 4-point example: 4 updates), otherwise cycles (XOR: 4 mistakes every epoch).
* **MLP** $\hat{\mathbf y}=\mathbf g(\mathbf W\mathbf h(\mathbf V\mathbf x))$ = **learned basis functions**; train all weights by gradient descent with **backprop** (chain rule backward through the DAG; our example: loss $2\to0.28$ in one step).
* **Loss = negative log-likelihood:** identity+MSE (Gaussian), sigmoid+binary cross-entropy (Bernoulli), softmax+cross-entropy (Categorical).
* **Hidden activations:** identity collapses to one (low-rank) matrix; sigmoid/tanh saturate ⇒ vanishing gradients ($0.25^L$); **ReLU** is the default (but units can "die"); leaky ReLU fixes that; softplus is rarely used.
* **Universal approximation:** one wide enough hidden layer fits any continuous function — but width may explode with dimension, and fitting ≠ generalizing. **Depth** often beats width.
* **Regularize:** weight decay, data augmentation, noise (small-batch SGD ⇒ flat minima), early stopping, **dropout** (test time: multiply weights by the keep probability $1-p$, or MC-dropout).

---

## Lecture 16 — Markov Assumption and Hidden State Models

**About the slides.** This lecture is **one long handwritten page** (no slide numbers), so the sections below follow the order of the page. Order: languages and **deterministic finite automata (DFA)** → the **Markov chain** and the **Markov assumption** → **sequence modelling** and language models → the model with **hidden states** $h_t$ (and the sum over all hidden sequences) → why we must **learn a hidden representation**.

**Big idea in one sentence.** To model a sequence $x_0,x_1,\dots,x_T$ you cannot condition on the whole history (it keeps growing), so you invent a **hidden state** $h_t$ that summarizes the past; if $h_t$ is *Markovian* (it is all you need to predict the future), the model needs only a fixed number of parameters.

### 16.1 Languages and finite automata

**A language is a set of symbol sequences.**

* A finite **alphabet** $\Sigma=\{\sigma_1,\dots,\sigma_n\}$.
* The set of *all* sequences over it is $\Sigma^*=\{(\sigma_1\sigma_2\sigma_3\cdots)\mid\sigma_i\in\Sigma\}$.
* A **language** is a subset $L\subseteq\Sigma^*$.

To describe $L$ we can define a yes/no function $\delta^*:\Sigma^*\to\{0,1\}$: if $\ell\in L$ then $\delta^*(\ell)=1$, if $\ell\notin L$ then $\delta^*(\ell)=0$.

**A DFA implements $\delta^*$.** A **deterministic finite automaton** is a 5-tuple $M=(Q,\Sigma,\delta,q_0,F)$:

| Symbol | Meaning |
|---|---|
| $Q$ | a finite set of **states** |
| $\Sigma$ | the finite alphabet |
| $\delta:Q\times\Sigma\to Q$ | the **transition function** (current state + symbol → next state) |
| $q_0\in Q$ | the **start state** |
| $F\subseteq Q$ | the **accept states** |

**The page's example:** binary sequences with an **even number of 1s**. States $Q=\{q_{\rm even},q_{\rm odd}\}$, alphabet $\Sigma=\{0,1\}$, start $q_0=q_{\rm even}$, accept $F=\{q_{\rm even}\}$. Reading a **0** keeps the state; reading a **1** flips it. The page lists four sequences ($\ell_1=11$, $\ell_2=11101$, $\ell_3=\varnothing$ (the empty sequence), $\ell_4=010101111$, as I read the handwriting) — each has an even number of 1s (2, 4, 0, 6), so all are **accepted**.

*Trace of $\ell_2=11101$:* start $q_{\rm even}\xrightarrow{1}q_{\rm odd}\xrightarrow{1}q_{\rm even}\xrightarrow{1}q_{\rm odd}\xrightarrow{0}q_{\rm odd}\xrightarrow{1}q_{\rm even}$ ✔ accepted.

**Why this matters.** The automaton needs **only one bit of memory** (even/odd), not the whole sequence. The current state *summarizes* everything about the past that matters. That is exactly what a Markov state is.

### 16.2 Markov chains and the Markov assumption

A **Markov chain** is a sequence of random variables $X_0,X_1,X_2,\dots$ that take values in a set of states $Q$ and satisfy
$$p(X_{t+1}\mid X_0,X_1,\dots,X_t)=p(X_{t+1}\mid X_t).$$
Read: "given the present, the future does not depend on the past."

* This definition gives a **probabilistic transition function** $p(X_{t+1}|X_t)$ (a DFA whose transitions are random).
* $p(X_0)$ is the **start-state distribution**.
* **The Markov assumption** (the page's wording): *the state of a system is sufficient to make any predictions about future outcomes.*

**Running example (the pass/fail coin, $\theta=0.4$).** Let each student's result be a 0 or 1 with $p(1)=0.4$, independently, and let $X_t$ be "even or odd number of passes so far". This *is* the DFA above made random. The transition matrix (row = current state, column = next state, each row sums to 1) is
$$\mathbf A=\begin{bmatrix}0.6&0.4\\0.4&0.6\end{bmatrix}\quad(\text{rows/columns: even, odd}),\qquad\text{start}=(1,0).$$
After $t$ steps the distribution is (start)$\cdot\mathbf A^t$:

| $t$ | $P(\text{even})$ | $P(\text{odd})$ |
|---|---|---|
| 1 | 0.6 | 0.4 |
| 2 | 0.52 | 0.48 |
| 3 | 0.504 | 0.496 |
| 4 | 0.5008 | 0.4992 |

*Closed form:* $P(\text{even after }t)=\frac12\big(1+0.2^t\big)$, because the eigenvalues of $\mathbf A$ are $1$ and $1-2(0.4)=0.2$. The chain **forgets its start** exponentially fast and tends to $(0.5,0.5)$.

### 16.3 Sequence modelling and language models

*So far in the class* we predicted an outcome from a particular context: datasets of pairs $(x,y)$ where we predict $y$ given $x$. (The page writes "predict $x$ given $y$"; the order is swapped, see Part G.)

A **language model** is a probability distribution over **sequences** of arbitrary length $T$:
$$p(x_0,x_1,\dots,x_T)\quad\text{with }T\text{ variable.}$$
The **chain rule** of probability always gives
$$p(x_0,\dots,x_T)=p(x_0)\prod_{t=1}^{T}p(x_t\mid x_0,\dots,x_{t-1}).$$
The factor $p(x_t|x_0,\dots,x_{t-1})$ depends on a history that **grows with $t$**. *Approximating this probability casts the sequence modelling task as a regression problem* (the page's remark): predict the next symbol from the past, as in Lectures 07–09.

### 16.4 The hidden-state model

Introduce a hidden state $h_t$ at each step, with the **graphical model** $h_0\to h_1\to\cdots\to h_t\to\cdots\to h_T$, where each $h_t$ also emits an observation $x_t$ (and, as on the page, $x_{t-1}$ feeds into $h_t$). Then the page writes
$$p(x_0,\dots,x_T)=\prod_{t=1}^{T}p(x_t\mid x_0\dots x_{t-1})\,p(x_0)\;=\;\sum_{h_1,\dots,h_T}\ \prod_{t=1}^{T}\underbrace{p(x_t\mid h_t)}_{\text{prediction “head”}}\ \underbrace{p(h_t\mid h_{t-1},x_{t-1})}_{\text{hidden representation model}}\ p(x_0)\,\underbrace{p(h_0)}_{\text{initial hidden state distribution}}.$$

* **$p(x_t|h_t)$ — the prediction "head":** the same idea as the output layer of a neural network (Lecture 15).
* **$p(h_t|h_{t-1},x_{t-1})$ — the hidden representation model:** how the summary is updated when a new symbol arrives.
* **$p(h_0)$** — the initial hidden-state distribution; $p(x_0)$ comes from the data.
* **The sum over all hidden sequences.** The page's green note: *we have to compute the sum — the marginal — of all possible hidden state sequences here, because this probabilistic model can assign even small probabilities (or zero probabilities) to some hidden state sequences. The probability of generating a sequence $x_0,\dots,x_T$ has to consider all these sequences.* This is **marginalization** (Lecture 17, first slides).

**Two important cautions from the page.**

1. The architecture **requires a Markovian state $h_t$**. *If $h_t$ is not Markovian, the system cannot generate sequences correctly.* The equality in the formula above is a **modelling assumption** (the hidden state is a sufficient summary), not an identity.
2. *Sequence prediction tasks require learning of a hidden representation.* We do not hand-design $h_t$; we **learn** it (an RNN in Lecture 17 does exactly that).

### 16.5 A tiny hidden Markov model, worked by hand

Two hidden states $\{0,1\}$ ("low-study day" and "high-study day"), observations $x_t\in\{0,1\}$ (fail / pass):

* start $\boldsymbol\pi=(0.6,\,0.4)$
* transitions $\mathbf T=\begin{bmatrix}0.7&0.3\\0.4&0.6\end{bmatrix}$ (row = current, column = next)
* emissions $p(x{=}1\mid h{=}0)=0.2$ and $p(x{=}1\mid h{=}1)=0.8$.

We observe $x=(1,0,1)$. We want $p(x)$. **Brute force:** sum over all $2^3=8$ hidden sequences, each term $=\pi_{h_1}p(x_1|h_1)\,T_{h_1h_2}p(x_2|h_2)\,T_{h_2h_3}p(x_3|h_3)$:

| hidden sequence | term |
|---|---|
| 000 | 0.00941 |
| 001 | 0.01613 |
| 010 | 0.00058 |
| 011 | 0.00346 |
| 100 | 0.01434 |
| 101 | 0.02458 |
| 110 | 0.00307 |
| 111 | 0.01843 |
| **sum** | **0.08998** |

*Example term (hidden sequence 101):* $\pi_1\,p(x{=}1|h{=}1)=0.4\cdot0.8=0.32$; then $T_{10}\,p(x{=}0|h{=}0)=0.4\cdot0.8=0.32$; then $T_{01}\,p(x{=}1|h{=}1)=0.3\cdot0.8=0.24$. The product is $0.32\cdot0.32\cdot0.24=0.02458$ ✔.

**A smarter way (the forward algorithm, previewed).** Keep, for each state, the probability of the observations so far *and* being in that state: $\alpha_1=(0.12,\,0.32)$, then $\alpha_2=(0.1696,\,0.0456)$, then $\alpha_3=(0.0274,\,0.0626)$; the total is $0.0274+0.0626=\mathbf{0.0900}$, the same number with only $T K^2$ multiplications instead of $K^T$ terms (for $T=100$, $K=10$: about $10^4$ versus $10^{100}$). The textbooks cover it (Bishop §13.2.2).

```python
import numpy as np, itertools
pi = np.array([.6, .4]); T = np.array([[.7, .3], [.4, .6]]); B1 = np.array([.2, .8])   # P(x=1 | state)
em = lambda s, x: B1[s] if x == 1 else 1 - B1[s]
x = [1, 0, 1]

# brute force: marginalize over all hidden sequences
total = 0
for hs in itertools.product([0, 1], repeat=len(x)):
    p = pi[hs[0]] * em(hs[0], x[0])
    for t in range(1, len(x)):
        p *= T[hs[t-1], hs[t]] * em(hs[t], x[t])
    total += p
print(total)                                        # 0.089984

# forward algorithm: same answer in O(T K^2)
alpha = pi * np.array([em(s, x[0]) for s in (0, 1)])
for t in range(1, len(x)):
    alpha = (alpha @ T) * np.array([em(s, x[t]) for s in (0, 1)])
print(alpha.sum())                                  # 0.089984

# the DFA of section 16.1 and its Markov-chain version
def even_ones(s):
    q = 0
    for c in s: q ^= (c == '1')
    return q == 0
print([even_ones(s) for s in ['11', '11101', '', '010101111', '1']])   # [True, True, True, True, False]
A = np.array([[.6, .4], [.4, .6]]); p = np.array([1, 0.])
print([(p @ np.linalg.matrix_power(A, t))[0].round(4) for t in range(1, 5)])   # [0.6, 0.52, 0.504, 0.5008]
```

#### 📖 Textbook deep dive

##### Murphy §3.6.1.2 "Example: Markov chain" (p.101) and Chapter 15 (p.503–505)
* **§3.6.1.2 (p.101).** For a distribution over variable-length sequences $p(y_{1:T})$ (a **language model** if each $y_t$ is a word), the chain rule gives $p(y_{1:T})=\prod_tp(y_t|y_{1:t-1})$ (3.107). The number of parameters of $p(y_t|y_{1:t-1})$ **grows exponentially with $t$**. Make the **first-order Markov** assumption (the future is independent of the past given the present) and $p(y_{1:T})=p(y_1)\prod_{t=2}^Tp(y_t|y_{t-1})$ (3.108): a **Markov chain** (also "autoregressive model of order 1"). $p(y_t|y_{t-1})$ is the **transition function / kernel**; as a stochastic matrix $A_{jk}=p(y_t=k|y_{t-1}=j)$ each **row sums to 1** (the *state transition matrix*). Using the same matrix at every step is **homogeneous / stationary / time-invariant** and is an example of **parameter tying**: "we can model an arbitrary number of variables using a fixed number of parameters." The model generalizes to depend on the last $M$ observations (order $M$), at the cost of a table that grows like $K^M$.
* **§15.2 (p.503).** An **RNN** "maps from an input space of sequences to an output space of sequences in a stateful way": the prediction of $y_t$ depends on $x_t$ **and the hidden state $h_t$**, which is updated as the sequence is processed.
* **§15.2.1.1 (p.504, eq. 15.1, 15.4).** The RNN is the conditional generative model $p(y_{1:T}|\mathbf x)=\sum_{h_{1:T}}\prod_tp(y_t|h_t)p(h_t|h_{t-1},y_{t-1},\mathbf x)$, with the hidden state computed **deterministically**, $p(h_t|h_{t-1},y_{t-1},\mathbf x)=\mathbb I\big(h_t=f(h_{t-1},y_{t-1},\mathbf x)\big)$ — "overcomes the limitations of standard Markov models, in that [RNNs] can have **unbounded memory**," which makes them "theoretically as powerful as a Turing machine." This is the page's formula with the sum collapsed to one term. (Details: Lecture 17.)

##### Bishop Chapter 13 "Sequential data" (p.605–)
* **§13.1 Markov models.** The product rule gives $p(\mathbf x_1,\dots,\mathbf x_N)=\prod_np(\mathbf x_n|\mathbf x_1,\dots,\mathbf x_{n-1})$; assuming each observation depends only on the previous one gives the first-order Markov chain. The trouble with **fixed-order** Markov models: an order-$M$ model over $K$-state variables needs $K^M(K-1)$ parameters, growing exponentially in $M$. The fix is to introduce **latent (hidden) variables**.
* **§13.2 Hidden Markov models (p.610–612).** Joint distribution of the latent-variable chain (13.6): $p(\mathbf x_1,\dots,\mathbf x_N,\mathbf z_1,\dots,\mathbf z_N)=p(\mathbf z_1)\prod_{n=2}^Np(\mathbf z_n|\mathbf z_{n-1})\prod_{n=1}^Np(\mathbf x_n|\mathbf z_n)$. With discrete latent variables this is the **hidden Markov model (HMM)**; the observations can be discrete or continuous. Using d-separation, "the **observed variables do not satisfy the Markov property at any order**": the predictive $p(\mathbf x_{n+1}|\mathbf x_1,\dots,\mathbf x_n)$ depends on *all* previous observations — exactly why a hidden state buys unbounded memory with a fixed-size model. **Transition matrix** $A_{jk}=p(z_{nk}=1|z_{n-1,j}=1)$ (13.7) has $K(K-1)$ independent parameters; the initial node has $\boldsymbol\pi$ (13.8); **emissions** $p(\mathbf x_n|\mathbf z_n,\boldsymbol\phi)$ (13.9). In a **homogeneous** model all transitions share $\mathbf A$ and all emissions share $\boldsymbol\phi$. A **mixture model for i.i.d. data is the special case** where every row of $\mathbf A$ is the same (the horizontal links are deleted).
* **§13.2.1–13.2.5 (p.615–631).** Maximum likelihood for an HMM is done with **EM** (the Baum–Welch algorithm); the **forward–backward algorithm** computes the needed posteriors efficiently; **scaling factors** avoid underflow; the **Viterbi algorithm** finds the most probable hidden sequence. (Only the idea, "sum over hidden sequences efficiently", is used in this course.)
* **§13.3 Linear dynamical systems.** The continuous-latent-variable cousin (Kalman filter).

##### Solved exercises and conceptual checks

**(a) Is "number of 1s is even" a Markov process on the raw sequence?** Look at the raw bits alone: the next *parity* depends on **all** past bits. But the **state** $q\in\{\text{even},\text{odd}\}$ is sufficient: $p(q_{t+1}|q_0,\dots,q_t)=p(q_{t+1}|q_t)$. Moral: whether something is "Markov" depends on **what you call the state**.

**(b) Derive $P(\text{even after }t)=\frac12(1+0.2^t)$.** Let $e_t=P(\text{even after }t)$. Then $e_{t+1}=0.6\,e_t+0.4\,(1-e_t)=0.4+0.2\,e_t$, with $e_0=1$. Fixed point $e^*=0.5$; $e_t-0.5=0.2^t\,(e_0-0.5)$, so $e_t=0.5+0.5\cdot0.2^t$ ✔ ($t=1$: 0.6; $t=3$: 0.504).

**(c) Parameter counts.** An order-$M$ Markov model on $K=10$ symbols with $M=5$: $K^M(K-1)=10^5\cdot9=900{,}000$ numbers. An HMM with $K_h=10$ hidden states and $K=10$ symbols: $K_h(K_h-1)+(K_h-1)+K_h(K-1)=90+9+90=189$. A hidden state is a *compression* of the history.

**(d) Why must we sum over hidden sequences?** The hidden states are never observed, so the probability of the *observed* sequence is the sum over all ways the hidden states could have produced it (marginal distribution, Lecture 17 slides 2–4). Skipping the sum and using only the most likely hidden sequence gives the **Viterbi** path, which is a *different* quantity (the largest single term, 0.02458 here, not the total 0.08998).

**(e) Why is the page's "predict $x$ given $y$" a slip?** In Lectures 07–09 the dataset is $\{(x_i,y_i)\}$ with $x$ the input and $y$ the label, and the model is $p(y|x;\theta)$: "predict $y$ given $x$."

#### ➕ Textbook extra
Murphy §15.2.7–15.2.8 (GRU/LSTM, beam search) and Murphy's book 2 chapter on HMMs/state-space models go beyond this lecture; Bishop §13.2.2 derives the forward–backward algorithm and §13.2.5 the Viterbi algorithm. Formal-language theory (DFA, regular languages, Turing machines) is covered in a theory-of-computation course; here it only motivates "state = memory".

#### ✅ Key takeaways
* A **language** is a set of sequences; a **DFA** $(Q,\Sigma,\delta,q_0,F)$ decides membership with a finite state as memory (even number of 1s: two states).
* **Markov chain:** $p(X_{t+1}|X_0,\dots,X_t)=p(X_{t+1}|X_t)$. **Markov assumption:** the state is sufficient to predict the future. (Pass/fail parity chain: $\frac12(1+0.2^t)$.)
* A **language model** is a distribution over variable-length sequences; the chain rule gives $\prod_tp(x_t|x_{0:t-1})$, whose history grows — that is why we need a hidden state.
* **Hidden-state model:** $p(x_{0:T})=\sum_{h}\prod_tp(x_t|h_t)\,p(h_t|h_{t-1},x_{t-1})\cdots$: a **head** $p(x_t|h_t)$, a **hidden representation model**, an initial distribution; we must **marginalize** over hidden sequences (our tiny HMM: 0.08998).
* If $h_t$ is **not** Markovian the model cannot generate sequences correctly; sequence tasks therefore require **learning a hidden representation** (RNNs, Lecture 17).

---

## Lecture 17 — Sequence Modelling (RNNs, Attention, Transformers and LLMs)

**About the slides.** 41 slides (PDF page = slide number). Order: **marginal distributions** (slides 2–4) → **vec2seq** RNN and caption generation (5–10) → **seq2vec** classification, including bidirectional (11–13) → **seq2seq** translation (14–16) → **gradients for RNNs** and RNN blocks (17–18) → **attention** (19–23) → **Transformers** and **LLMs** (24–29) → **autoregressive inference** (30–36) → **teacher forcing** and **LLM = maximum likelihood** (37–38) → **guardrails and safety** (40) → summary (41).

**Big idea in one sentence.** A sequence model is a probabilistic model with a **hidden state**: an RNN learns a fixed-size summary $h_t$ that it updates at each step; **attention** replaces that single summary with a **key–value store of the whole history**; and an **LLM** is just such a model trained by **maximum likelihood** to predict the next token.

### 17.1 Marginal distributions (slides 2–4)

If two random variables have the joint distribution $p(x,y)$, then the distribution of one alone is obtained by **summing the other out**:
$$p(x)=\sum_yp(x,y)\qquad\text{(sum over all possible values of }y).$$

**With a hidden variable.** If $y$ depends on a hidden $h$ with a prior $p_{\rm prior}(h)$ and a model $p(y|h)$:
$$p(y)=\sum_hp(y|h)\,p_{\rm prior}(h)=\mathbb E_{h\sim p_{\rm prior}}\big[p(y|h)\big].$$
*In words: consider every possible $h$, weight each by how likely it is, and average.*

**Conditioning on a known input $x$** (slide 4). By assumption $y$ depends on $h$ only, so
$$p(y|x)=\sum_hp(y|h,x)\,p_{\rm prior}(h|x)=\sum_hp(y|h)\,p_{\rm prior}(h|x)=\mathbb E_{h\sim p_{\rm prior}(\cdot|x)}\big[p(y|h)\big].$$

*Running example.* A student's study day is **low** or **high** with prior $(0.6,\,0.4)$, and $p(\text{pass}\mid\text{low})=0.2$, $p(\text{pass}\mid\text{high})=0.8$. Then
$$p(\text{pass})=0.6\cdot0.2+0.4\cdot0.8=0.12+0.32=\mathbf{0.44}.$$
This is exactly the sum over hidden states that Lecture 16 said a hidden-state model must compute.

### 17.2 Sequence generation: vec2seq with an RNN (slides 5–8)

**Task.** Given an input **vector** $\mathbf x$, generate a **sequence** $y_{1:T}$ of arbitrary length $T$. At each step the network produces $y_t$ from a **hidden state** $h_t$. A recurrent architecture implements a **conditional generative model**:
$$p(y_{1:T}|\mathbf x)=\sum_{h_{1:T}}p(y_{1:T},h_{1:T}|\mathbf x)=\sum_{h_{1:T}}\prod_{t=1}^{T}p(y_t|h_t)\;p(h_t|h_{t-1},y_{t-1},\mathbf x),$$
where for the initial state $p(h_1|h_0,y_0,\mathbf x)=p(h_1|\mathbf x)$.

* **Output (head):** $p(y_t|h_t)$ is predicted from the hidden state "similar to the feed-forward architectures we have seen before" (Lecture 15).
* **Hidden state:** usually **deterministic**, written with an **indicator function** $\mathbb I$ (1 if true, 0 if false):
$$p(h_t|h_{t-1},y_{t-1},\mathbf x)=\mathbb I\big(h_t=f(h_{t-1},y_{t-1},\mathbf x)\big).$$
Then the sum over $h_{1:T}$ has a **single non-zero term**, so no marginalization is needed.
* (Slide 6 notes that one could also let $h_t\sim\mathcal N(\mu=f_\theta(h_{t-1}),\sigma=1)$: "in principle we can assume any arbitrary distribution.")

**The RNN model (slide 7).**
$$\mathbf y_t=a\big(\mathbf W_{hy}\mathbf h_t\big),\qquad \mathbf h_t=b\big(\mathbf W_{xh}[\mathbf x,\mathbf y_{t-1}]+\mathbf W_{hh}\mathbf h_{t-1}\big).$$
$a,b$ are activation functions (e.g. softmax and $\tanh$), $\mathbf W_{xh}$ maps input to hidden, $\mathbf W_{hh}$ maps hidden to hidden (the **recurrence**), $\mathbf W_{hy}$ maps hidden to output. **The same weights are used at every time step** (parameter tying), so the parameter count does **not** depend on $T$.

**Slide 8: RNNs are powerful.** In theory an RNN has **unbounded memory** and is as expressive as a **Turing machine**. In practice the memory is limited by the **size of the latent space** and the **strength of the parameters**.

*Running example (a tiny scalar RNN).* $h_t=\tanh(0.5\,x_t+0.8\,h_{t-1})$, $y_t=h_t$, $h_0=0$, inputs $x=(1,\,0,\,-1)$:
$$h_1=\tanh(0.5)=0.4621,\quad h_2=\tanh(0+0.8\cdot0.4621)=0.3537,\quad h_3=\tanh(-0.5+0.8\cdot0.3537)=-0.2137.$$
The output at step 3 still "remembers" the first input (through $h_2$).

**Counting parameters.** For input size $d_x=10$, hidden size $H=64$, output size $d_y=10$: $d_xH+H^2+H+Hd_y+d_y=640+4096+64+640+10=\mathbf{5450}$, whether the sequence has length 5 or 5000.

### 17.3 Caption generation and tokenization (slides 9–10)

* Build a **dictionary** mapping every word to an index, including a **beginning-of-sequence** token `<bos>` and an **end-of-sequence** token `<eos>`.
* **Image captioning:** a **CNN** turns the image into the input vector $\mathbf x$; an RNN then generates `This is a Puppy . <eos>` one token at a time, feeding each generated token back in as the next input (the bottom row of the slide: `<bos> This is a Puppy .`).
* **Sentence-piece tokenization** (Sennrich et al. 2016) splits words into **word fragments** ("syllabi" in the slide) and builds the dictionary over those fragments, so rare words never fall outside the vocabulary.
* Slide 10's **alternative vec2seq model** uses "slightly different probabilistic modelling assumptions … implemented using different connectivity" (the image vector is fed in differently).

### 17.4 Sequence classification: seq2vec (slides 11–13)

**Task:** map a **variable-length input sequence** to a **single fixed-length output vector** (e.g. sentiment).
$$\mathbf y=a\big(\mathbf W_{hy}\mathbf h_T\big),\qquad \mathbf h_t=b\big(\mathbf W_{xh}\mathbf x_t+\mathbf W_{hh}\mathbf h_{t-1}\big).$$
Train by **maximum likelihood** (the same recipe as Lectures 04–09):
$$p(\mathcal D|\boldsymbol\theta)=\prod_{(\mathbf x_{1:T},\mathbf y)\in\mathcal D}p\big(\mathbf y\mid\mathbf x_{1:T},\boldsymbol\theta\big).$$

**Bidirectional models** (slides 12–13) read the sequence **forward and backward** and combine both:
$$\mathbf h^\rightarrow_t=c\big(\mathbf W_{xh}\mathbf x_t+\mathbf W_{hh}\mathbf h^\rightarrow_{t-1}\big),\qquad\mathbf h^\leftarrow_t=c\big(\mathbf W_{xh}\mathbf x_t+\mathbf W_{hh}\mathbf h^\leftarrow_{t+1}\big),\qquad\mathbf h_t=b\big(\mathbf W[\mathbf h^\rightarrow_t,\mathbf h^\leftarrow_t]\big).$$
**Example (slide 13):** sentiment classification of IMDB movie reviews with a word-level bidirectional **LSTM**: "this movie is so great" → *positive*, "this movie is so bad" → *negative*. A backward pass helps because the meaning of a word can depend on words that come *after* it.

### 17.5 Sequence translation: seq2seq (slides 14–16)

**Task:** map a sequence to another sequence (aligned case: one output per input):
$$p(y_{1:T}|x_{1:T})=\sum_{h_{1:T}}\prod_{t=1}^{T}p(y_t|h_t)\,p(h_t|h_{t-1},\mathbf x_t)=\sum_{h_{1:T}}\prod_{t=1}^Tp(y_t|h_t)\,\mathbb I\big(h_t=f(h_{t-1},x_t)\big).$$
Slide 15: we can **stack several hidden layers** ($h^1_t,h^2_t,\dots$) to increase model complexity.

**Slide 16 (e.g. English → French translation).** The input sequence (or **prompt**) is **encoded** into the model and the response is then **auto-regressively decoded** to produce the output sequence.

### 17.6 Gradients for RNNs, and the RNN blocks (slides 17–18)

* **Backpropagation through time:** *unroll* the recurrent model for the length of the sequence (copies of the same module with the same $\boldsymbol\theta$, as in the slide's picture) and compute the gradient of the unrolled model (Lecture 15's backprop).
* **The problem:** for long sequences this can be unstable, because the gradient passes through the recurrence many times: a factor $<1$ per step vanishes, a factor $>1$ explodes. *Numbers:* $0.9^{50}=0.0052$ and $1.1^{50}=117.4$; even the best-case sigmoid factor $0.25$ gives $0.25^{10}=9.5\times10^{-7}$ after 10 steps. These are **vanishing / exploding gradients**.
* **Fix:** different non-linear recurrent modules. Each RNN block can be
  1. a **linear function** $\mathbf y_t=a(\mathbf W_{hy}\mathbf h_t)$, $\mathbf h_t=b(\mathbf W_{xh}[\mathbf x,\mathbf y_{t-1}]+\mathbf W_{hh}\mathbf h_{t-1})$,
  2. a **gated recurrent unit (GRU)**: updates a hidden representation with a **gating unit**,
  3. a **long short-term memory (LSTM)**: uses **two** hidden vectors, one for **long-term** and one for **short-term** memory; works well on longer time series.

### 17.7 Attention (slides 19–23)

**Motivation (slide 19).** The models so far build a hidden state $\mathbf h_t$ of **fixed dimension $n$**. (1) Regardless of the length of the context, *all history is stored in this one $n$-dimensional vector*. (2) That limits how much information the model's history can hold. **Attention** uses a **key–value store** instead.

**Idea (slides 20–21).** At every time step the model produces three vectors from the activation: a **value** $\mathbf v_i$, a **key** $\mathbf k_i$ and a **query** $\mathbf q_i$. The self-attention output at step $t$ is
$$\mathbf p_t=\text{softmax}\!\Big(\tfrac1{\sqrt d}\big[\mathbf q_t^\top\mathbf k_1,\ \dots,\ \mathbf q_t^\top\mathbf k_t\big]\Big),\qquad \mathbf z_t=\sum_{i=1}^{t}\mathbf p_t(i)\,\mathbf v_i.$$
(The slide divides the scores by a constant "C"; the standard choice is $\sqrt d$, the key dimension, Murphy eq. 15.43–15.44.) *In words:* compare the query with every key (a dot product measures similarity), turn the scores into **probabilities** with softmax, and return the **weighted average of the values**. It is a **soft dictionary lookup**.

**The attention matrix (slide 22).** Stack the rows $\mathbf p_t$ to get $\mathbf A$, a **stochastic matrix** of size $t\times t$: every row is a probability vector computed from the pairwise dot products between queries $\mathbf q_i$ and keys $\mathbf k_j$. **Masking** the matrix encodes causal structure (see the caution in Part G): the slide says a *backward-causal* model zeroes the entries a token must not look at, while a *fully causal* (unmasked) model zeroes nothing. In today's decoder language models the **mask hides the future**, so token $t$ attends only to tokens $\le t$.

**Attention module (slide 23).** A set of **linear layers** maps an activation $\mathbf x_t\mapsto(\mathbf q_t,\mathbf k_t,\mathbf v_t)$, and the output is computed as above; all of it is **tensor arithmetic** (matrix products, Lecture 02).

**Worked example (three tokens, $d=2$).** Take $\mathbf Q=\mathbf K=\begin{bmatrix}1&0\\0&1\\1&1\end{bmatrix}$ and $\mathbf V=\begin{bmatrix}1&0\\0&2\\3&3\end{bmatrix}$. Scores $\mathbf Q\mathbf K^\top/\sqrt2=\begin{bmatrix}0.707&0&0.707\\0&0.707&0.707\\0.707&0.707&1.414\end{bmatrix}$.

*Without a mask* (softmax of each row): $\mathbf A=\begin{bmatrix}0.401&0.198&0.401\\0.198&0.401&0.401\\0.248&0.248&0.503\end{bmatrix}$ (each row sums to 1) and $\mathbf Z=\mathbf A\mathbf V=\begin{bmatrix}1.604&1.599\\1.401&2.006\\1.759&2.007\end{bmatrix}$.

*With a causal mask* (token $t$ sees only tokens $\le t$; masked scores set to $-\infty$): $\mathbf A=\begin{bmatrix}1&0&0\\0.330&0.670&0\\0.248&0.248&0.503\end{bmatrix}$ and $\mathbf Z=\begin{bmatrix}1&0\\0.330&1.340\\1.759&2.007\end{bmatrix}$. Token 1 can only copy its own value $(1,0)$; token 3 is unchanged because it already sees everything.

### 17.8 Transformers and LLMs (slides 24–29)

* **Transformers** are sequence models made of many **attention layers** and **heads** (slide 24: a transformer layer has Attention Head 1, Attention Head 2, …, whose outputs are combined into $\mathbf z^{l+1}_t$). Each head can pick up a different kind of relationship.
* **What makes them different from RNNs:** the "hidden state" is the whole **attention matrix**, which **grows quadratically** with sequence length ($n^2$ entries: 1,000,000 for $n=1000$, 64,000,000 for $n=8000$). That is why long contexts are expensive.
* **LLMs** (large language models) are very large Transformers that **predict the next token** given a sequence of tokens. They learn relationships between concepts and topics in the **attention maps**, which depend on the provided **context** (slide 25). Slide 26 shows the architecture of Vaswani et al. 2017 (arXiv 1706.03762).
* **A departure from RNNs (slide 27):** instead of conditioning on a fixed-size state $\mathbf h_t$, **prediction is conditioned on a key–value store**. Transformers/LLMs are **seq2seq** models.

**Positional encodings (slide 28).** Attention alone **ignores order**: re-ordering the input gives the same output. Fix: build an **embedding matrix** (could be one-hot) and **add** the position vector: $\mathbf z_t\leftarrow\mathbf z_t+\mathbf p_t$, where $\mathbf p_t$ is the $t$-th row (the slide shows **RoPE** embeddings, a rotation-based variant).

**Model families (slide 29).**

| Model | What the slide says |
|---|---|
| **BERT** (Bidirectional Encoder Representations from Transformers) | Maps a modified (masked) sequence back to the original and computes the loss at the masked locations: "fill in the blanks", e.g. "Let's make [MASK] chicken! [SEP] It [MASK] great with orange sauce." |
| **GPT** (OpenAI) | Masked backward-causal transformer as the **decoder**; proprietary (an open variant exists from EleutherAI). |
| **Llama** (Meta; Llama 1, 2, 3, CodeLlama) | Open-source, **decoder-only**, different sizes and tokenizers (Llama 3: **128k tokens**, up to **405B parameters**); data from public sources but the dataset is not released. |
| Others | DeepSeek, Mistral, … |

### 17.9 Inference: autoregressive rollouts (slides 30–36)

A **decoder-only** model predicts the next token given the tokens so far. Start from `<bos>` and generate one token at a time: `<bos>` → *This*; `<bos> This` → *is*; `<bos> This is` → *a*; … Each step feeds the **entire history** through self-attention. The **rollout stops when `<eos>` is generated**.

**In math** (slide 36), the probability of a sampled sequence is the chain rule:
$$p(t_1,t_2,\dots,t_T|\boldsymbol\theta)=p(t_1|\langle\text{bos}\rangle,\boldsymbol\theta)\;p(t_2|\langle\text{bos}\rangle t_1,\boldsymbol\theta)\;p(t_3|\langle\text{bos}\rangle t_1t_2,\boldsymbol\theta)\cdots.$$

### 17.10 Training: teacher forcing, and "an LLM is a maximum likelihood estimate" (slides 37–38)

**Teacher forcing.** In training we already *know* the whole target sequence, so we feed the **true** previous tokens (not the model's own samples) and compute **all positions in one forward pass** (the causal mask keeps each position from seeing its own answer). The gradient is back-propagated through the **entire architecture for the entire sequence**. This is **very compute- and memory-intensive**, because each attention head computes an $n\times n$ matrix for a length-$n$ sequence.

**LLMs are trained on internet-scale data by maximum likelihood** (slide 38):
$$\boldsymbol\theta^*\in\arg\max_{\boldsymbol\theta}\,p(\mathcal D|\boldsymbol\theta).$$
*An LLM is just the maximum likelihood estimate that generates an internet-scale dataset.* After the log and the negative sign (Lecture 04) the loss is the **cross-entropy** of the next token.

*Running example (loss of one sentence).* Suppose the model gave these probabilities to the true tokens `This is a Puppy . <eos>`: $0.2,\,0.5,\,0.6,\,0.1,\,0.8,\,0.9$. Then
$$-\log p=1.609+0.693+0.511+2.303+0.223+0.105=\mathbf{5.444},\qquad\text{per token }5.444/6=0.907,\qquad\text{perplexity }e^{0.907}=\mathbf{2.48}.$$
A **perplexity** of 2.48 means the model is as unsure as if it were choosing between about 2.5 equally likely tokens at each step.

### 17.11 Guardrails and safety (slide 40)

1. Generative AI systems are trained by **maximum likelihood**: they model *what is likely in the data*, not what is safe.
2. Safety guidelines can be implemented by **screening prompts**, which is very difficult in practice.
3. **Reinforcement learning** is currently used to "align" LLMs with constraints: methods that **change the distribution being modelled using a reward signal**.

### 17.12 Summary (slide 41)

* **RNNs:** vec2seq (generation), seq2vec (classification), seq2seq (translation).
* The **attention** mechanism, **self-attention** and **multi-headed attention**.
* The **Transformer** architecture for NLP, and **language models with Transformers**.

```python
import numpy as np
sm = lambda z: np.exp(z - z.max(1, keepdims=True)) / np.exp(z - z.max(1, keepdims=True)).sum(1, keepdims=True)

# --- marginalization: p(pass) = sum_h p(pass|h) p(h)
prior = np.array([.6, .4]); p_pass = np.array([.2, .8]); print(prior @ p_pass)           # 0.44

# --- tiny scalar RNN
h, outs = 0., []
for x in [1, 0, -1.]:
    h = np.tanh(0.5 * x + 0.8 * h); outs.append(round(h, 4))
print(outs)                                                                               # [0.4621, 0.3537, -0.2137]

# --- parameter count of an RNN (independent of sequence length)
dx, H, dy = 10, 64, 10; print(dx * H + H * H + H + H * dy + dy)                           # 5450

# --- scaled dot-product attention, with and without a causal mask
Q = K = np.array([[1, 0], [0, 1], [1, 1.]]); V = np.array([[1, 0], [0, 2], [3, 3.]])
S = Q @ K.T / np.sqrt(Q.shape[1])
A = sm(S); print(A.round(3)); print((A @ V).round(3))                                     # rows sum to 1
mask = np.triu(np.ones((3, 3), bool), 1)                                                  # True above the diagonal = future
Ac = sm(np.where(mask, -1e9, S)); print(Ac.round(3)); print((Ac @ V).round(3))

# --- loss of one sentence under teacher forcing
p = np.array([.2, .5, .6, .1, .8, .9])
print(-np.log(p).sum(), -np.log(p).mean(), np.exp(-np.log(p).mean()))                     # 5.4445 0.9074 2.4779

# --- vanishing / exploding factors
print(0.9 ** 50, 1.1 ** 50, 0.25 ** 10)                                                   # 0.00515 117.39 9.5e-07
```

#### 📖 Textbook deep dive

##### Murphy Chapter 15 "Neural Networks for Sequences" (p.503–545)
* **§15.2 RNNs (p.503).** "A recurrent neural network or RNN is a neural network which maps from an input space of sequences to an output space of sequences in a stateful way": the prediction of $y_t$ depends on $x_t$ *and* the hidden state $h_t$, which is updated as the sequence is processed.
* **§15.2.1 Vec2Seq (p.503–505).** The output is generated one token at a time: sample $\tilde y_t$ from $h_t$ and feed it back to obtain $h_{t+1}$. Model (15.1): $p(y_{1:T}|\mathbf x)=\sum_{h_{1:T}}\prod_tp(y_t|h_t)p(h_t|h_{t-1},y_{t-1},\mathbf x)$, with $p(h_1|h_0,y_0,\mathbf x)=p(h_1|\mathbf x)$. Output (15.2) $p(y_t|h_t)=\text{Cat}(y_t|\text{softmax}(\mathbf W_{hy}h_t+\mathbf b_y))$, or (15.3) $\mathcal N(y_t|\mathbf W_{hy}h_t+\mathbf b_y,\sigma^2\mathbf I)$ for real values. Deterministic hidden state (15.4) $h_t=f(h_{t-1},y_{t-1},\mathbf x)$ with the standard update (15.5) $h_t=\varphi(\mathbf W_{xh}[\mathbf x;y_{t-1}]+\mathbf W_{hh}h_{t-1}+\mathbf b_h)$. "$y_t$ implicitly depends on all past observations … thus an RNN overcomes the limitations of standard Markov models, in that they can have unbounded memory … theoretically as powerful as a Turing machine."
* **§15.2.2 Seq2Vec (p.505–507)** classification from the final (or pooled) hidden state; the **bidirectional RNN** concatenates forward and backward states. **§15.2.3 Seq2Seq (p.507–508):** the **aligned** case (one output per input) and the **unaligned** case (**encoder–decoder**: the encoder compresses the input to a context vector, the decoder generates the output).
* **§15.2.4 Teacher forcing (p.509).** Training feeds the *true* previous output instead of the model's own sample; at test time the model must use its own samples (so errors can compound, "exposure bias"). **§15.2.5 Backpropagation through time (p.510):** unroll the RNN and apply backprop. **§15.2.6 Vanishing and exploding gradients (p.511).** **§15.2.7 Gating (p.512–514):** **GRU** and **LSTM** use gates ($\in(0,1)$) to decide what to keep or forget, giving a path along which gradients do not vanish. **§15.2.8 Beam search (p.515):** keep the top-$k$ partial sequences instead of sampling one token at a time.
* **§15.4 Attention (p.518–526).** Attention "computes a weighted average of a set of values, where the weights are derived by comparing the query vector to a set of keys" — a **dynamic** weighted combination of the inputs "rather than a fixed" one (p.519). **§15.4.1 soft dictionary lookup (p.519, eq. 15.34):** $\text{Attn}(\mathbf q,(\mathbf k_{1:m},\mathbf v_{1:m}))=\sum_{i=1}^m\alpha_i(\mathbf q,\mathbf k_{1:m})\mathbf v_i$ with $0\le\alpha_i\le1$ and $\sum_i\alpha_i=1$ (a *convex combination*), $\alpha_i=\text{softmax}_i\,a(\mathbf q,\mathbf k_i)$. **§15.4.2 kernel regression as non-parametric attention (p.520):** $f(x)=\sum_i\alpha_i(x,x_{1:n})y_i$ (15.36). **§15.4.3 parametric attention (p.521):** additive $a(\mathbf q,\mathbf k)=\mathbf w_v^\top\tanh(\mathbf W_q\mathbf q+\mathbf W_k\mathbf k)$ (15.42) or **scaled dot-product** $a(\mathbf q,\mathbf k)=\mathbf q^\top\mathbf k/\sqrt d$ (15.43), and for matrices $\text{Attn}(\mathbf Q,\mathbf K,\mathbf V)=\text{softmax}(\mathbf Q\mathbf K^\top/\sqrt d)\mathbf V$ (15.44) — our worked example. The $1/\sqrt d$ keeps the dot products from growing with $d$ (they have variance $d$ if the entries have variance 1) so the softmax does not saturate. **§15.4.4 seq2seq with attention (p.522–523)**, **soft vs hard attention (p.525)**: hard attention is non-differentiable and needs reinforcement learning.
* **§15.5 Transformers (p.526–533).** "The transformer model is a seq2seq model which uses attention in the encoder as well as the decoder, thus eliminating the need for RNNs." **§15.5.1 Self-attention (p.526–527, eq. 15.54):** $\mathbf y_i=\text{Attn}(\mathbf x_i,(\mathbf x_1,\mathbf x_1),\dots,(\mathbf x_n,\mathbf x_n))$ — query $=\mathbf x_i$, keys and values = all the inputs. In a **decoder** set $n=i-1$ "so all the previously generated outputs are available"; at training time all outputs are known so the function can be evaluated **in parallel**, "overcoming the sequential bottleneck of using RNNs." Example: in "the animal didn't cross the street because it was too tired/wide" self-attention resolves what "it" refers to (Fig. 15.23). **§15.5.2 Multi-headed attention (p.527–528, eq. 15.55):** head $i$ is $\text{Attn}(\mathbf W^{(q)}_i\mathbf q,\{\mathbf W^{(k)}_i\mathbf k_j,\mathbf W^{(v)}_i\mathbf v_j\})$; the heads are concatenated and projected — different heads capture different notions of similarity. **§15.5.3 Positional encoding (p.528):** attention is **permutation invariant** and "ignores the input word ordering", so concatenate/add a positional embedding (binary codes 000…111, or the sinusoidal basis of Vaswani et al.). **§15.5.5** compares Transformers, CNNs and RNNs; **§15.6** efficient (sparse, low-rank, memory) transformers address the **quadratic cost**.
* **§15.7 Language models (p.537–545).** **§15.7.1.2 BERT (p.539):** an **encoder-only**, non-generative model trained with the **masked language model** task (hide some tokens and predict them) and next-sentence prediction, then fine-tuned. **§15.7.2 Generative (causal) LLMs, §15.7.2.1 GPT (p.542):** a **decoder-only** transformer trained to predict the next token; applications (§15.7.2.2), T5 (§15.7.2.3).

##### Bishop (2006)
Bishop predates attention and Transformers (2017). The relevant background is **Chapter 13 "Sequential data"** (§13.1 Markov models, §13.2 hidden Markov models, §13.3 linear dynamical systems): recurrent networks are the neural counterpart of the hidden-state model, with a **learned** $h_t$ in place of a discrete latent state (see Lecture 16 notes).

##### Solved exercises and conceptual checks

**(a) Marginal from a joint table.** If $p(\text{low},\text{pass})=0.12$, $p(\text{high},\text{pass})=0.32$, $p(\text{low},\text{fail})=0.48$, $p(\text{high},\text{fail})=0.08$, then $p(\text{pass})=0.12+0.32=0.44$ and $p(\text{fail})=0.56$, and $p(\text{pass}|\text{high})=0.32/0.40=0.8$ ✔.

**(b) Why does the RNN need no sum over hidden states?** Because $p(h_t|h_{t-1},y_{t-1},\mathbf x)=\mathbb I(h_t=f(\cdot))$ puts probability 1 on **one** hidden sequence, so the sum has a single non-zero term $\prod_tp(y_t|h_t)$.

**(c) Why do the scores get divided by $\sqrt d$?** For $\mathbf q,\mathbf k$ with independent entries of mean 0 and variance 1, $\mathbf q^\top\mathbf k$ has variance $d$. Without scaling the softmax input has a large spread, so one entry gets probability $\approx1$ and the gradients vanish. Dividing by $\sqrt d$ restores variance 1. *Check:* with $d=64$, an unscaled score has standard deviation 8.

**(d) Memory of attention.** $n=8000$ tokens need $n^2=64{,}000{,}000$ scores *per head per layer*; a model with 32 heads and 32 layers would hold $\approx6.5\times10^{10}$ numbers for one forward pass if all were stored — which is why long-context models use tricks (Murphy §15.6).

**(e) Why can a Transformer train in parallel but an RNN cannot?** In an RNN, $h_t$ needs $h_{t-1}$ first (sequential). With **teacher forcing** a Transformer already knows every token, so all positions can be computed at once with a causal mask.

**(f) Causal mask versus bidirectional.** A **decoder** (GPT, Llama) masks the future: it models $p(t_i|t_{<i})$ and can *generate*. An **encoder** (BERT) attends in both directions and models *masked* tokens given *all* the others: better for understanding, not for left-to-right generation.

**(g) Perplexity.** $\text{PPL}=\exp\big(-\frac1T\sum_t\log p(t_t|t_{<t})\big)$, the exponentiated average negative log-likelihood. A model that always predicted uniformly over a vocabulary of 128,000 tokens would have perplexity exactly 128,000.

#### ➕ Textbook extra
Murphy §15.3 (1-d CNNs for sequences, WaveNet), §15.5.4 "putting it all together" (the encoder–decoder block diagram), §15.5.6–15.5.7 (vision transformers, other variants), §15.6 (efficient transformers), §15.7.1.1 ELMo, §15.7.2.3 T5 and Murphy §13.4 (training deep networks) go beyond this lecture. Reinforcement-learning-based alignment (slide 40) is not covered in either textbook chapter used here.

#### ✅ Key takeaways
* **Marginalize:** $p(y)=\sum_hp(y|h)p(h)$ — an expectation over the hidden variable (our example: 0.44).
* **RNN:** $\mathbf h_t=b(\mathbf W_{xh}[\mathbf x,\mathbf y_{t-1}]+\mathbf W_{hh}\mathbf h_{t-1})$, $\mathbf y_t=a(\mathbf W_{hy}\mathbf h_t)$; same weights every step; deterministic $h_t$ ⇒ no sum; **vec2seq** (captioning), **seq2vec** (sentiment, bidirectional), **seq2seq** (translation).
* **Backprop through time** unrolls the RNN; long sequences cause **vanishing/exploding gradients**; **GRU** and **LSTM** blocks fix it.
* **Attention:** $\mathbf z_t=\sum_i\mathbf p_t(i)\mathbf v_i$ with $\mathbf p_t=\text{softmax}(\mathbf q_t^\top\mathbf k_i/\sqrt d)$; the **attention matrix** is $t\times t$ and **row-stochastic**; **masking** sets who may see whom; **positional encodings** restore order.
* **Transformers/LLMs** condition on a **key–value store** instead of a fixed $\mathbf h_t$; cost grows **quadratically** in sequence length.
* **LLM = MLE:** $\boldsymbol\theta^*\in\arg\max p(\mathcal D|\boldsymbol\theta)$; trained with **teacher forcing**; sampled **autoregressively** until `<eos>`; **safety** is hard because the model only imitates data (alignment uses reinforcement learning).

---

## Lecture 18 — Gradient Descent

**About the slides.** 24 PDF pages; the slide numbers run to 25 because one slide (number 20) is not in the PDF. Order: recap of maximum likelihood (2) → **optimization in ML** (3) → two loss/gradient examples (4) → the **gradient** (5) → **gradient descent** (6) → **convex** and **non-convex** functions (7–11) → gradient descent for logistic regression and learning rates (12) → **stochastic gradient descent** (13–17) → **oscillations** and **momentum** (18–19) → **algorithm variants** (21) → **PyTorch** (22) → **backprop and automatic differentiation** (23) → summary (24).

**Big idea in one sentence.** Almost every ML model is fitted by minimizing a loss $J(\boldsymbol\theta)$; when no closed-form answer exists, **walk downhill**: repeatedly move $\boldsymbol\theta$ a small step in the direction of the negative gradient, using a random **mini-batch** to make each step cheap, and **momentum** to smooth the zig-zag.

### 18.1 Previously: the maximum likelihood method (slide 2)

$$p(\mathcal D|\boldsymbol\theta)=p(y_1,\dots,y_D\mid\mathbf x_1,\dots,\mathbf x_D;\boldsymbol\theta)\ \xrightarrow{\ \log\ }\ \log p(\mathcal D|\boldsymbol\theta)=\sum_{i=1}^{D}\log p(y_i|\mathbf x_i;\boldsymbol\theta)\ \xrightarrow{\ \text{scale and negate}\ }\ \mathcal L(\boldsymbol\theta|\mathcal D)=\frac1D\sum_{(\mathbf x_i,y_i)\in\mathcal D}l\big(y_i,f_{\boldsymbol\theta}(\mathbf x_i)\big).$$
The loss $l$ is the **negative log-likelihood** of one data point (squared error for the Gaussian, cross-entropy for Bernoulli/categorical). **This lecture is about how to minimize $\mathcal L$.**

### 18.2 Optimization in ML (slide 3)

Almost all ML algorithms solve an optimization problem. The axes along which problems differ:

* **constrained vs unconstrained** (constraints come in Lecture 20, SVMs),
* optimization of **continuous** functions,
* **convex vs non-convex**, **local vs global** optima,
* **analytic** gradient solving (set $\nabla=0$) vs iterative methods,
* **stochastic vs batch**,
* **smooth vs non-smooth** (e.g. the lasso's $|\theta|$ has a kink).

### 18.3 Two losses and their gradients (slide 4)

| | Linear regression ($y\in\mathbb R$) | Logistic regression ($y\in\{0,1\}$) |
|---|---|---|
| Loss | $\text{MSE}(\boldsymbol\theta|\mathcal D)=\frac1D\sum_i\big(y_i-\boldsymbol\theta^\top\mathbf x_i\big)^2$ | $J(\boldsymbol\theta|\mathcal D)=-\frac1D\sum_i\big[y_i\log\sigma(\boldsymbol\theta^\top\mathbf x_i)+(1-y_i)\log\big(1-\sigma(\boldsymbol\theta^\top\mathbf x_i)\big)\big]$ |
| Gradient | $\nabla_{\boldsymbol\theta}\text{MSE}=\frac2D\sum_i\big(\boldsymbol\theta^\top\mathbf x_i-y_i\big)\mathbf x_i$ | $\nabla_{\boldsymbol\theta}J=\frac1D\sum_i\big(\sigma(\boldsymbol\theta^\top\mathbf x_i)-y_i\big)\mathbf x_i$ |

Both have the same shape: **(prediction − truth) × input**, averaged over the data. (The factor 2 in the MSE gradient comes from differentiating the square; some texts divide the loss by 2 to remove it.)

*Running example, gradient at $\boldsymbol\theta=\mathbf 0$* (hours $h\to$ score $s$; design matrix $\mathbf X=[\mathbf 1,\ \mathbf h]$, so $\mathbf X^\top\mathbf y=(624,\,2804)$ from Lecture 06):
$$\nabla\text{MSE}(\mathbf 0)=\tfrac2{10}\big(\mathbf X\mathbf 0-\mathbf y\big)^\top\mathbf X=-\tfrac2{10}(624,\,2804)=\mathbf{(-124.8,\,-560.8)}.$$
For the pass/fail labels (passed: students 1, 3, 7, 9) the logistic gradient at $\boldsymbol\theta=\mathbf 0$ (where $\sigma=0.5$ for everyone) is $\frac1{10}\big(0.5\cdot10-4,\ \ 0.5\cdot41-25\big)=\mathbf{(0.1,\,-0.45)}$: it says "increase the hours weight", as it should.

### 18.4 The gradient (slide 5)

$J(\boldsymbol\theta)$ is a **multivariate function**. The **partial derivative** is the slope as only one entry of $\boldsymbol\theta$ is varied ($\mathbf e_i$ is the unit vector along axis $i$):
$$\frac{\partial}{\partial\theta_i}J(\boldsymbol\theta)=\lim_{\varepsilon\to0}\frac{J(\boldsymbol\theta+\varepsilon\mathbf e_i)-J(\boldsymbol\theta)}{\varepsilon}.$$
The **gradient** is the vector of all partial derivatives, $\nabla_{\boldsymbol\theta}J=\big(\frac{\partial J}{\partial\theta_1},\dots,\frac{\partial J}{\partial\theta_n}\big)^\top$. It points in the direction of **steepest increase**; the negative gradient points **downhill**. *Check by finite differences:* for $J(\theta)=(\theta-3)^2$ at $\theta=0$, $\frac{J(0.001)-J(0)}{0.001}=\frac{8.994001-9}{0.001}=-5.999\approx-6=J'(0)$ ✔.

### 18.5 Gradient descent (slide 6)

**Algorithm.** Pick a random starting point $\boldsymbol\theta_0$. For $n$ steps:
$$\boldsymbol\theta_{t+1}\leftarrow\boldsymbol\theta_t-\alpha_t\,\nabla_{\boldsymbol\theta}J(\boldsymbol\theta_t|\mathcal D),$$
where $\alpha_t$ is the **learning rate**. **Conditions on the learning rate:** $\sum_t\alpha_t=\infty$ (we can reach anywhere) and $\sum_t\alpha_t^2<\infty$ (the steps shrink fast enough to settle). *Example:* $\alpha_t=1/t$ satisfies both ($\sum1/t=\infty$, $\sum1/t^2=\pi^2/6=1.645$); a **constant** $\alpha$ fails the second condition (fine in practice for full-batch GD, but it never fully settles with noisy gradients).

**A one-dimensional example.** $J(\theta)=(\theta-3)^2$, $J'(\theta)=2(\theta-3)$, start $\theta_0=0$. The update gives $\theta_{t+1}-3=(1-2\alpha)(\theta_t-3)$: the distance to the minimum is multiplied by $(1-2\alpha)$ each step.

| learning rate $\alpha$ | $\theta_1,\theta_2,\theta_3,\theta_4,\theta_5$ | verdict |
|---|---|---|
| 0.1 | 0.6, 1.08, 1.464, 1.771, 2.017 | slow but steady (factor 0.8 per step) |
| 0.5 | 3, 3, 3, 3, 3 | factor 0: lands on the minimum in one step |
| 1.0 | 6, 0, 6, 0, 6 | factor $-1$: bounces forever |
| 1.1 | 6.6, −1.32, 8.18, −3.22, 10.5 | factor $-1.2$: **diverges** |

So it converges exactly when $|1-2\alpha|<1$, i.e. $0<\alpha<1$.

**The students' regression, in two dimensions.** The Hessian of the MSE is $\mathbf H=\frac2D\mathbf X^\top\mathbf X=\begin{bmatrix}2&8.2\\8.2&43.4\end{bmatrix}$, with eigenvalues $\lambda_{\min}=0.435$ and $\lambda_{\max}=44.97$ (**condition number** $103.4$, as in Lecture 06). Gradient descent converges iff $\alpha<2/\lambda_{\max}=0.0445$, and the speed is limited by the *flat* direction: each step shrinks that error component by $1-\alpha\lambda_{\min}$.

| $\alpha$ | steps to reach within 0.01 of $\boldsymbol\theta^*=(41.81,\,5.02)$ |
|---|---|
| 0.001 | 19,036 (stable but very slow) |
| 0.04 | **472** (close to the stability limit: fastest here) |
| 0.0445 or 0.05 | **diverges** (blows up) |

*The first three steps at $\alpha=0.04$:* $\boldsymbol\theta_1=(4.99,\,22.43)$, $\boldsymbol\theta_2=(2.23,\,4.28)$, $\boldsymbol\theta_3=(5.64,\,18.55)$ with the MSE going $2685\to1826\to1275$. Notice the **zig-zag** of the hours weight: the steep direction is overshot at every step. That is the **oscillation** problem of section 18.9.

### 18.6 Convex functions (slides 7–11)

A **convex** loss has the shape of a **bowl**: **every function value stays below the straight line joining any two of its points**:
$$J\big(\lambda\boldsymbol\theta+(1-\lambda)\boldsymbol\theta'\big)\le\lambda J(\boldsymbol\theta)+(1-\lambda)J(\boldsymbol\theta'),\qquad\lambda\in[0,1].$$
(Slide 7 says the function stays "underneath (or above)" a line; for convex functions it is the *chord* that lies above the curve.)

* A **non-convex** function (slide 8) can have several valleys; it may be **locally convex** near one of them.
* **Examples of convex functions (slide 9):** the **constant** $f(x)=c$; the **linear** $f(\mathbf x)=\mathbf w^\top\mathbf x$; the **MSE** $\frac1D\sum(y_i-f_{\boldsymbol\theta}(\mathbf x_i))^2$ (convex in $\boldsymbol\theta$ when $f$ is linear in $\boldsymbol\theta$). *Proof sketch:* its Hessian $\frac2D\mathbf X^\top\mathbf X$ is positive semi-definite (for any $\mathbf v$, $\mathbf v^\top\mathbf X^\top\mathbf X\mathbf v=\lVert\mathbf X\mathbf v\rVert^2\ge0$; eigenvalues here $0.435$ and $44.97$, both $>0$). *Numerical check:* the inequality above held on 10,000 random pairs $(\boldsymbol\theta,\boldsymbol\theta',\lambda)$ for our student MSE. The logistic loss is also convex.
* **Examples of non-convex functions (slides 10–11):** $f(x,y)=x^3y^5$ is not convex (its Hessian is indefinite; e.g. at $(1,1)$ the second derivatives are $f_{xx}=6$, $f_{yy}=20$, $f_{xy}=15$ and $\det=6\cdot20-15^2=-105<0$). Slide 11 shows polynomial fits of degree 3, 5 and 50 to data. Careful: it is the fitted *curve* that is not convex in $x$; the **loss** of a linear-in-parameters model is still convex in $\boldsymbol\theta$. **Neural network losses are non-convex** (Lecture 15).

**Why we care (summary slide):** for a **convex** function gradient descent converges to the **global minimum**; for a **non-convex** function it converges to *a* minimum, "usually the one closest to the starting point."

### 18.7 Gradient descent for logistic regression and learning rates (slide 12)

The slide shows (left) the loss contours with the path of $\boldsymbol\theta$ and (middle) the fitted sigmoid after 0, 100, 1000, 10,000 and 100,000 steps: the S-curve gets steeper as the separable data pull the weights larger. **Right: the effect of the learning rate on the loss curve.**

| curve | behaviour |
|---|---|
| **very high** learning rate | the loss **increases** (diverges) |
| **high** | the loss drops fast, then **oscillates/plateaus** above the best value |
| **low** | the loss decreases but **very slowly** |
| **good** | the fastest steady decrease |

### 18.8 Stochastic gradient descent, SGD (slides 13–17)

**Problem.** The full gradient averages over all $D$ data points, which is expensive. **Idea:** approximate it with a **random mini-batch** $\mathcal B\subset\mathcal D$ of size $B$:
$$\frac1B\sum_{(x_i,y_i)\in\mathcal B}\nabla_{\boldsymbol\theta}l_{\boldsymbol\theta}(x_i,y_i)\ \approx\ \frac1D\sum_{(x_i,y_i)\in\mathcal D}\nabla_{\boldsymbol\theta}l_{\boldsymbol\theta}(x_i,y_i).$$
It is a **Monte Carlo estimate** of the gradient (Lecture 09's idea of averages).

**Algorithm (slide 14).** Pick $\boldsymbol\theta_0$. For $n$ steps: **sample a batch** $\mathcal B\subset\mathcal D$ and update
$$\boldsymbol\theta_{t+1}\leftarrow\boldsymbol\theta_t-\alpha_t\,\frac1B\sum_{(x_i,y_i)\in\mathcal B}\nabla_{\boldsymbol\theta}l_{\boldsymbol\theta}(x_i,y_i).$$
*Less computation per step because we use a mini-batch.* **Robbins–Monro conditions** on $\alpha_t\in[0,1]$: $\sum_t\alpha_t=\infty$ and $\sum_t\alpha_t^2<\infty$. Because the mini-batch is sampled, **each update is noisy**; to bound the accumulated noise we need $\sum_t\alpha_t^2<\infty$ (slide 15). Slides 16–17: the sampling makes the optimization **noisy**, and the loss **fluctuates** around the minimum instead of settling.

*Numbers (our student MSE at $\boldsymbol\theta=\mathbf 0$; full gradient $(-124.8,\,-560.8)$; 20,000 random mini-batches each):*

| batch size $B$ | mean of the mini-batch gradient | standard deviation |
|---|---|---|
| 2 | $(-124.7,\,-559.6)$ | $(15.3,\ 258.3)$ |
| 5 | $(-124.9,\,-561.8)$ | $(7.6,\ 128.8)$ |

The mean equals the full gradient (**unbiased**) and the noise falls as $B$ grows: from $B=2$ to $B=5$ the standard deviation roughly halves (it would shrink exactly as $\sqrt{(D-B)/(B(D-1))}$ for sampling without replacement).

### 18.9 Oscillations and momentum (slides 18–19)

**Oscillations (slide 18).** SGD **zig-zags**: every gradient step is **perpendicular to the isocontours**, so in a long narrow valley (high condition number, like our students' loss) the iterate bounces across the valley while creeping along it; mini-batches add more noise.

**Momentum (slide 19)** smooths the updates by keeping a **running history** of past gradients. The standard form (Murphy 8.30–8.31) is
$$\mathbf m_t=\beta\,\mathbf m_{t-1}+\mathbf g_t,\qquad\boldsymbol\theta_{t+1}=\boldsymbol\theta_t-\alpha\,\mathbf m_t,$$
so $\mathbf m_t=\sum_\tau\beta^\tau\mathbf g_{t-\tau}$ is an exponentially weighted sum of past gradients: components that **alternate in sign cancel** (the oscillating direction), components that **point the same way add up** (the useful direction). With a constant gradient the effective step is $\alpha/(1-\beta)$ (e.g. $\beta=0.9$ gives $10\alpha$). *Slide 19's demo:* distill.pub/2017/momentum.

*Numbers (a long valley $J=\frac12(\theta_1^2+25\theta_2^2)$, start $(1,1)$, $\alpha=0.07$; distance to the minimum after 5, 10, 20, 40 steps):*

| $\beta$ | 5 | 10 | 20 | 40 |
|---|---|---|---|---|
| 0 (plain GD) | 0.735 | 0.487 | 0.234 | 0.055 |
| 0.5 | 0.573 | 0.210 | 0.029 | 0.0006 |
| 0.9 | 0.392 | 0.934 | 0.355 | 0.132 |

A **moderate** momentum ($\beta=0.5$) converges far faster than plain gradient descent. Too much ($\beta=0.9$ with this step size) **overshoots and swings back**: momentum and learning rate must be tuned together.

### 18.10 SGD variants (slide 21)

Momentum is primarily used when training **deep neural networks**. Different variants exist that work differently well:

1. **Adagrad**, 2. **RMSprop**, 3. **Adam** (the most popular), 4. **AdamW**.

These keep running statistics of the gradients to give **each parameter its own step size** (Murphy §8.4.6: AdaGrad (8.69), RMSProp (8.72), **Adam** combines momentum $\mathbf m_t=\beta_1\mathbf m_{t-1}+(1-\beta_1)\mathbf g_t$ (8.77) with an RMS-scaled step using $\mathbf s_t=\beta_2\mathbf s_{t-1}+(1-\beta_2)\mathbf g_t^2$ (8.78); **AdamW** decouples weight decay from the adaptive step).

### 18.11 PyTorch (slide 22)

PyTorch is a library for deep learning. (1) It **mostly mimics tensor arithmetic as in NumPy** (`numpy.array` ↔ `torch.Tensor`). (2) Tensors can be placed on **GPUs** (CUDA); other accelerators are supported but much less common. (3) It implements many deep-learning routines and distributed training. (4) It provides an **autodiff engine**: *you do not need to derive gradients by hand.*

### 18.12 Backprop and automatic differentiation (slide 23)

Assume functions are **chained**: $y=f_1(f_2(f_3(x)))$. The **chain rule** gives
$$\frac{dy}{dx}=\frac{df_1}{da}\Big|_{a=a_0}\cdot\frac{df_2}{db}\Big|_{b=b_0}\cdot\frac{df_3}{dx}\Big|_{x=x_0}.$$
To compute each factor we need: (1) the **inputs** $a_0,b_0,x_0$ of each component, computed in the **forward pass**; (2) the **analytic derivative** of each simple function, tracked in PyTorch's **computation graph**; (3) the actual derivatives, multiplied together in the **backward pass**. `x.detach()` removes a tensor from the computation graph.

*Worked example.* $y=\big(\sin(x^2)\big)^3$ at $x=1$: forward: $x_0=1$, $b_0=x_0^2=1$, $a_0=\sin 1=0.8415$, $y=0.8415^3=0.5958$. Backward: $\frac{df_1}{da}=3a^2=3(0.7081)=2.1242$, $\frac{df_2}{db}=\cos b=0.5403$, $\frac{df_3}{dx}=2x=2$; product $2.1242\cdot0.5403\cdot2=\mathbf{2.2954}$.

### 18.13 Summary (slide 24)

* Gradient descent is an optimization algorithm for **differentiable** functions.
* **Convex:** converges to the **global minimum**. **Non-convex:** converges to *a* minimum, usually the one closest to the start.
* **SGD** is a **Monte Carlo variant** that reduces the compute needed per gradient.
* **Momentum** increases the stability of SGD.

```python
import numpy as np
h = np.array([4, 2, 6, 1, 3, 5, 7, 3, 8, 2.]); s = np.array([66, 48, 71, 45, 57, 62, 77, 59, 83, 56.])
X = np.c_[np.ones(10), h]; D = len(h)

# --- gradient of the MSE and of the logistic loss at theta = 0
grad_mse = lambda th, X=X, y=s: 2 / len(y) * X.T @ (X @ th - y)
print(grad_mse(np.zeros(2)))                                      # [-124.8 -560.8]
t = np.array([1, 0, 1, 0, 0, 0, 1, 0, 1, 0.]); sg = lambda z: 1 / (1 + np.exp(-z))
print(X.T @ (sg(X @ np.zeros(2)) - t) / D)                        # [ 0.1  -0.45]

# --- gradient descent on the 1-D bowl (theta - 3)^2
for a in (0.1, 0.5, 1.0, 1.1):
    th = 0.; tr = []
    for _ in range(5): th -= a * 2 * (th - 3); tr.append(round(th, 3))
    print(a, tr)

# --- gradient descent on the students' MSE: stability limit and speed
H = 2 / D * X.T @ X; lam = np.linalg.eigvalsh(H); print(lam, 2 / lam.max())   # [ 0.435 44.965] 0.0445
for a in (0.001, 0.04, 0.05):
    th = np.zeros(2)
    for n in range(1, 200001):
        th = th - a * grad_mse(th)
        if np.abs(th).max() > 1e8 or np.abs(th - [41.8078, 5.0225]).max() < 0.01: break
    print(a, n, th.round(4))                                      # 19036 / 472 / diverged

# --- mini-batch gradients are unbiased and get less noisy with B
rng = np.random.default_rng(0); g0 = np.zeros(2)
for B in (2, 5):
    g = np.array([grad_mse(g0, X[i], s[i]) for i in (rng.choice(D, B, replace=False) for _ in range(20000))])
    print(B, g.mean(0).round(1), g.std(0).round(1))               # mean ~ (-124.8, -560.8)

# --- momentum on an ill-conditioned bowl J = 0.5*(t1^2 + 25 t2^2)
def run(beta, a=0.07, n=40):
    th = np.array([1., 1.]); m = np.zeros(2)
    for _ in range(n):
        m = beta * m + np.array([th[0], 25 * th[1]]); th = th - a * m
    return np.linalg.norm(th)
print([round(run(b), 4) for b in (0, 0.5, 0.9)])                  # [0.0549, 0.0006, 0.1315]

# --- autodiff by hand vs finite differences:  y = sin(x^2)^3 at x = 1
f = lambda x: np.sin(x ** 2) ** 3
print(3 * np.sin(1) ** 2 * np.cos(1) * 2, (f(1 + 1e-6) - f(1 - 1e-6)) / 2e-6)   # 2.2954 2.2954
```

(With PyTorch the last check is `x = torch.tensor(1., requires_grad=True); y = torch.sin(x**2)**3; y.backward(); x.grad` $\to 2.2954$.)

#### 📖 Textbook deep dive

##### Murphy Chapter 8 "Optimization" (p.275–302)
* **§8.1 Introduction (p.275–282).** **Local vs global** optima and their conditions (§8.1.1, p.275–276): at a local minimum the gradient is zero and the Hessian is positive semi-definite; "a zero gradient is not sufficient" because a stationary point may be a maximum or a **saddle** (eigenvalues of the Hessian of both signs); a strictly positive-definite Hessian means "the bottom of a bowl" (p.277). **Constrained vs unconstrained** (§8.1.2, p.277): constraints $g_j(\boldsymbol\theta)\le0$, $h_k(\boldsymbol\theta)=0$ define a **feasible set** $\mathcal C$ (8.4–8.5). **Convex vs nonconvex** (§8.1.3, p.277–281): convex sets, **convex functions** (a function is convex if the chord lies above it, p.278), the second-order characterization (Hessian positive semi-definite everywhere, p.279), **strongly convex** functions (p.280). Convexity is the property that makes every local minimum global. **Smooth vs nonsmooth** (§8.1.4, p.281) introduces **subgradients** (for $|\theta|$ and the lasso).
* **§8.2 First-order methods (p.282–289).** **Descent direction** (§8.2.1) and **step size / learning rate** (§8.2.2, p.284): a constant step works if it is small enough; **line search** (§8.2.2.2) picks it automatically. **Convergence rates** (§8.2.3, p.286): the speed of steepest descent depends on the **condition number** $\kappa=\lambda_{\max}/\lambda_{\min}$ of the Hessian — Figure 8.12 shows that a large $\kappa$ gives the zig-zag of slide 18; our students' loss has $\kappa=103$. **§8.2.4 Momentum (p.287–288):** "gradient descent can move very slowly along flat regions"; **heavy-ball momentum** $\mathbf m_t=\beta\mathbf m_{t-1}+\mathbf g_{t-1}$, $\boldsymbol\theta_t=\boldsymbol\theta_{t-1}-\eta_t\mathbf m_t$ (8.30–8.31) unrolls to $\mathbf m_t=\sum_\tau\beta^\tau\mathbf g_{t-\tau-1}$ (8.32), and with a constant gradient the sum is $\mathbf g/(1-\beta)$ (8.34). **Nesterov momentum** (8.35–8.38) evaluates the gradient at the "looked-ahead" point $\boldsymbol\theta_t+\beta\mathbf m_t$.
* **§8.3 Second-order methods (p.289–292).** **Newton's method** uses the Hessian (8.39–8.42); **BFGS/quasi-Newton** and **trust-region** methods approximate it. Too costly for models with millions of parameters, which is why first-order methods dominate deep learning.
* **§8.4 Stochastic gradient descent (p.292–302).** The setting (8.53): $\mathcal L(\boldsymbol\theta)=\mathbb E_{q(\mathbf z)}[\mathcal L(\boldsymbol\theta,\mathbf z)]$ and the update $\boldsymbol\theta_{t+1}=\boldsymbol\theta_t-\eta_t\nabla\mathcal L(\boldsymbol\theta_t,\mathbf z_t)=\boldsymbol\theta_t-\eta_t\mathbf g_t$ (8.54). **§8.4.1 finite sums (p.293):** $\mathcal L=\frac1N\sum_n\ell(y_n,f(\mathbf x_n;\boldsymbol\theta))$ (8.55); the gradient is approximated on a **minibatch** (8.57). **§8.4.2 SGD for linear regression (p.293–294):** $g_t=(\boldsymbol\theta_t^\top\mathbf x_n-y_n)\mathbf x_n$ (8.59), the **LMS (least mean squares)** algorithm $\boldsymbol\theta_{t+1}=\boldsymbol\theta_t-\eta_t(\boldsymbol\theta_t^\top\mathbf x_n-y_n)\mathbf x_n$ (8.60) — the per-example version of the slide's gradient. **§8.4.3 Choosing the step size (p.294–296):** the **Robbins–Monro conditions** $\sum\eta_t=\infty$, $\sum\eta_t^2<\infty$ (8.61); common schedules: **piecewise constant** (8.62), **exponential decay** (8.63), **polynomial decay** (8.64), linear **warm-up** then cosine cool-down, **cyclical** schedules (Fig. 8.18–8.19). **§8.4.4 Iterate averaging (p.297).** **§8.4.5 Variance reduction (p.297–298):** SVRG (8.66), SAGA (8.67). **§8.4.6 Preconditioned SGD (p.298–301):** **AdaGrad** $\theta_{t+1,d}=\theta_{t,d}-\eta_t\frac{g_{t,d}}{\sqrt{s_{t,d}+\epsilon}}$ (8.69), **RMSProp** (EWMA of squared gradients, 8.72–8.74) and **Adam** (8.77–8.81): "it is possible to combine RMSProp with momentum", with **bias correction** $\hat{\mathbf m}_t=\mathbf m_t/(1-\beta_1^t)$, $\hat{\mathbf s}_t=\mathbf s_t/(1-\beta_2^t)$ (8.80–8.81); §8.4.6.4 discusses issues with adaptive learning rates.

##### Bishop §5.2.4 "Gradient descent optimization" (p.240–241)
Bishop's update (5.41) is $\mathbf w^{(\tau+1)}=\mathbf w^{(\tau)}-\eta\nabla E(\mathbf w^{(\tau)})$ with $\eta>0$ the **learning rate**; "each step requires that the entire training set be processed", so these are **batch methods**. He remarks that plain **gradient (steepest) descent "turns out to be a poor algorithm"** for batch optimization, with **conjugate gradients** and **quasi-Newton** methods "much more robust and much faster"; to find a good minimum one may need to **restart from several random points** and compare them on a **validation set**. The **on-line (sequential) gradient descent**, $\mathbf w^{(\tau+1)}=\mathbf w^{(\tau)}-\eta\nabla E_n(\mathbf w^{(\tau)})$ (5.43), updates after *one* data point and is the ancestor of SGD. (Also §5.2.1–5.2.3 on **local minima** and the cost argument on p.240: using gradient information, finding a minimum takes $O(W^2)$ steps instead of $O(W^3)$ for a network with $W$ weights.) §5.3 (backpropagation) is the same as Lecture 15's.

##### Solved exercises and conceptual checks

**(a) Why does the 1-D recursion $(1-2\alpha)$ appear?** $J=(\theta-3)^2\Rightarrow\theta_{t+1}=\theta_t-2\alpha(\theta_t-3)\Rightarrow\theta_{t+1}-3=(1-2\alpha)(\theta_t-3)$. In general for a quadratic with curvature $\lambda$ the factor is $1-\alpha\lambda$, so stability needs $\alpha<2/\lambda_{\max}$ ($=0.0445$ for the students) and the slowest component decays like $(1-\alpha\lambda_{\min})^t$.

**(b) How many steps would an ideal $\alpha$ need?** Plain GD with the best constant step $\alpha=2/(\lambda_{\max}+\lambda_{\min})=0.0442$ has rate $\frac{\kappa-1}{\kappa+1}=\frac{102.4}{104.4}=0.981$ per step: reducing the error by $10^{-3}$ takes about $\ln(10^{-3})/\ln(0.981)\approx360$ steps. (Our run at $\alpha=0.04$ needed 472 to get within 0.01 absolute error.)

**(c) Is SGD's gradient estimate unbiased?** Yes: sampling uniformly, $\mathbb E[\frac1B\sum_{i\in\mathcal B}\nabla l_i]=\frac1D\sum_{i\in\mathcal D}\nabla l_i$. Unbiased, but with **variance** $\propto1/B$ (our table), which is why the learning rate must shrink ($\sum\alpha_t^2<\infty$) or the iterate keeps jittering.

**(d) When is a constant learning rate acceptable?** For full-batch GD on a smooth convex loss, a small constant $\alpha<2/\lambda_{\max}$ converges (see 18.5). With noisy mini-batches, a constant rate converges to a *neighbourhood* of the minimum whose size grows with $\alpha$; hence schedules that decrease $\alpha$.

**(e) Does GD always reach the global minimum for logistic regression?** The loss is **convex**, so there are no bad local minima, but for **linearly separable** data the minimum is at infinity (weights grow without bound: the steepening sigmoid of slide 12). Regularization (Lectures 13–14) gives a finite optimum.

**(f) Why not just set the gradient to zero?** For linear regression we can ($\boldsymbol\theta^*=(\mathbf X^\top\mathbf X)^{-1}\mathbf X^\top\mathbf y$, Lecture 07). For logistic regression and neural networks the equation $\nabla J=0$ has no closed-form solution, so we iterate.

**(g) Three ways to detect a bad learning rate.** Loss goes **up** or becomes `nan` (too high); loss **plateaus above** where it should (high, oscillating); loss **decreases glacially** (too low).

#### ➕ Textbook extra
Murphy §8.2.3 (convergence rates, with proofs for quadratics), §8.3 (second-order and quasi-Newton methods), §8.4.5 (variance reduction: SVRG, SAGA), §8.4.6.5 (non-diagonal preconditioners), §8.5–8.6 (non-gradient methods, derivative-free and proximal methods: needed for the lasso's non-smooth term), Murphy §13.4 (training neural networks: vanishing gradients, initialization, clipping) and Bishop §5.2.1–5.2.3 (local quadratic approximation, use of gradient information) go beyond this lecture. Bishop Appendix D (calculus of variations) and the line-search/conjugate-gradient algorithms of Bishop §5.4 (Hessian) are omitted.

#### ✅ Key takeaways
* **Optimize the loss $\mathcal L(\boldsymbol\theta|\mathcal D)=\frac1D\sum l(y_i,f_{\boldsymbol\theta}(\mathbf x_i))$**; gradients have the form (prediction − truth) × input: $\frac2D\sum(\boldsymbol\theta^\top\mathbf x_i-y_i)\mathbf x_i$ (MSE) and $\frac1D\sum(\sigma(\boldsymbol\theta^\top\mathbf x_i)-y_i)\mathbf x_i$ (logistic).
* **Gradient descent:** $\boldsymbol\theta_{t+1}\leftarrow\boldsymbol\theta_t-\alpha_t\nabla J$; conditions $\sum\alpha_t=\infty$, $\sum\alpha_t^2<\infty$. On a quadratic it converges iff $\alpha<2/\lambda_{\max}$ (students: $0.0445$); the condition number $103$ makes it slow and zig-zag.
* **Convex** (chord above the curve; MSE and logistic loss are convex) ⇒ global minimum; **non-convex** (neural networks) ⇒ a local minimum, usually near the start.
* **Learning rate:** too high diverges, high oscillates, low is slow, good is fast.
* **SGD:** use a random mini-batch of size $B$; unbiased gradient, noise $\propto1/B$, cheaper steps. **Momentum:** $\mathbf m_t=\beta\mathbf m_{t-1}+\mathbf g_t$ smooths oscillations (needs tuning with $\alpha$). Variants: Adagrad, RMSprop, **Adam**, AdamW.
* **PyTorch autodiff:** forward pass stores inputs, backward pass multiplies the local derivatives (chain rule); no hand-derived gradients.

---

## Lecture 19 — Midterm Example Exercise and Reference Sheet

**Midterm logistics (from the course page).** Monday **Nov 16**, in class (11:30–12:20), **on paper**, in **teams of two**, **simple calculator only**; multiple-choice and/or long-answer questions. The **reference sheet** (section 19.5) is handed out with the exam, so you do not need to memorize distribution formulas, but you must know how to *use* them.

**The skill being tested.** The *whole course recipe* applied to a new dataset: pick a distribution, parameterize it with the inputs, write the log-likelihood, negate to get the loss, and differentiate for gradient descent.

### 19.1 The problem (handwritten page)

Data $\mathcal D=\{(x_i,y_i)\mid i=1,\dots,D,\ x_i\in[0,1],\ y_i\in\mathbb R\}$; the sketch shows a curve whose **spread (noise) changes with $x$**: narrow on the left, wide near $x\approx0.75$. **Question: how can we build a probabilistic model that predicts $y$ given $x$?** (Note: *predict $y$ given $x$*, not the other way round; the page's wording is correct here.)

### 19.2 The solution, step by step

1. **Distribution.** $y\in\mathbb R$, so use the **Normal**: $p(y|x)\propto\exp\!\big(-\frac{(y-\mu)^2}{2\sigma^2}\big)$ with free parameters the mean $\mu\in\mathbb R$ and the **variance** $\sigma^2>0$.
2. **Maximum likelihood.** $\log p(y_1,\dots,y_D|x_1,\dots,x_D;\theta)=\log\prod_i p(y_i|x_i,\theta)=\sum_i\log p(y_i|x_i,\theta)$ (log turns a product into a sum: identities 1 and 5 of the sheet).
3. **Parameterize both parameters with the input** (the note in blue: *$\mu$ and $\sigma^2$ are predicted as functions of $x$*):
$$\mu=\boldsymbol\theta_1^\top\boldsymbol\phi(x),\qquad\sigma^2=a(\boldsymbol\theta_2^\top\boldsymbol\phi(x))=\exp(\boldsymbol\theta_2^\top\boldsymbol\phi(x)).$$
The **exp activation** guarantees $\sigma^2>0$ for any real $\boldsymbol\theta_2^\top\boldsymbol\phi$ (compare the sigmoid for a probability in $[0,1]$).
4. **Loss and gradients.** Write $r_i=y_i-\boldsymbol\theta_1^\top\boldsymbol\phi(x_i)$, $e_i=\exp(\boldsymbol\theta_2^\top\boldsymbol\phi(x_i))$.

### 19.3 ⚠️ A correction to the handwritten solution

The page writes $\log p(y|x,\theta)=-\dfrac{(y-\boldsymbol\theta_1^\top\boldsymbol\phi)^2}{2\exp(\boldsymbol\theta_2^\top\boldsymbol\phi)}$ and then minimizes $\sum_i\dfrac{r_i^2}{2e_i}$. This **drops the normalizing term**. The density is only *proportional* to $\exp(\cdot)$; when $\sigma^2$ depends on $x$ the constant of proportionality $1/\sqrt{2\pi\sigma^2}$ **depends on $\boldsymbol\theta_2$** and must stay:
$$\log p(y|x)=-\tfrac12\log(2\pi)-\tfrac12\log\sigma^2-\frac{(y-\mu)^2}{2\sigma^2}=-\tfrac12\log2\pi-\tfrac12\,\boldsymbol\theta_2^\top\boldsymbol\phi(x)-\frac{r^2}{2\exp(\boldsymbol\theta_2^\top\boldsymbol\phi(x))}$$
(identity 3: $\log\exp x=x$). So the correct loss (dropping only the constant $\frac12\log2\pi$) is
$$\boxed{\ \mathcal L(\boldsymbol\theta_1,\boldsymbol\theta_2)=\sum_{i=1}^D\Big[\tfrac12\,\boldsymbol\theta_2^\top\boldsymbol\phi(x_i)+\frac{\big(y_i-\boldsymbol\theta_1^\top\boldsymbol\phi(x_i)\big)^2}{2\exp(\boldsymbol\theta_2^\top\boldsymbol\phi(x_i))}\Big]\ }$$
**Why it matters:** without the $\frac12\boldsymbol\theta_2^\top\boldsymbol\phi$ term, the loss is minimized by sending $\boldsymbol\theta_2^\top\boldsymbol\phi\to+\infty$ (huge variance makes every residual cost $\to0$); the term is the *price of a wide distribution*. Also the page calls $\sigma^2$ the "std. dev."; $\sigma^2$ is the **variance** (std. dev. is $\sigma$). Expect marks for the correct term on the real exam. (If a question *tells* you to use $\sigma^2$ constant, the term is a constant and may be dropped; the exercise's whole point is that it is not.)

### 19.4 Gradients and a gradient-descent step

$$\nabla_{\boldsymbol\theta_1}\mathcal L=\sum_i-\frac{r_i}{e_i}\,\boldsymbol\phi(x_i),\qquad\nabla_{\boldsymbol\theta_2}\mathcal L=\sum_i\Big(\frac12-\frac{r_i^2}{2e_i}\Big)\boldsymbol\phi(x_i),$$
and gradient descent updates $\boldsymbol\theta_k\leftarrow\boldsymbol\theta_k-\alpha\nabla_{\boldsymbol\theta_k}\mathcal L$. Reading them: for $\boldsymbol\theta_1$, (error) × (input) weighted by *precision* $1/e_i$ (noisy points count less). For $\boldsymbol\theta_2$, the sign of $\tfrac12-\frac{r^2}{2e}$ says "increase the variance if the squared residual exceeds the predicted variance ($r^2>e$), decrease it otherwise": at the optimum $\mathbb E[r^2]=e$.

*Numerical check (one data point).* $\boldsymbol\phi=(1,0.5)$, $y=2$, $\boldsymbol\theta_1=(1,1)$, $\boldsymbol\theta_2=(0,0.5)$: $\mu=1.5$, $r=0.5$, $\boldsymbol\theta_2^\top\boldsymbol\phi=0.25$, $e=e^{0.25}=1.2840$. Loss $=0.125+\frac{0.25}{2.5681}=0.1250+0.0973=\mathbf{0.2224}$. $\nabla_{\boldsymbol\theta_1}=-\frac{0.5}{1.2840}(1,0.5)=\mathbf{(-0.3894,\,-0.1947)}$, $\nabla_{\boldsymbol\theta_2}=\big(0.5-\frac{0.25}{2.5681}\big)(1,0.5)=0.4027\,(1,0.5)=\mathbf{(0.4026,\,0.2013)}$. Both agree with finite differences. One step at $\alpha=0.1$: $\boldsymbol\theta_1\to(1.0389,\,1.0195)$, $\boldsymbol\theta_2\to(-0.0403,\,0.4799)$: the mean moves up toward $y=2$, and because $r^2=0.25<e=1.284$ the predicted variance shrinks.

### 19.5 The midterm reference sheet

**Distributions** (support; PMF/PDF; parameters):

| Distribution | Support | PMF or PDF | Parameters |
|---|---|---|---|
| Normal | $\mathbb R$ | $f(x)\propto\exp\!\big(-\frac{(x-\mu)^2}{2\sigma^2}\big)$ | $\mu\in\mathbb R,\ \sigma>0$ |
| Bernoulli | $\{0,1\}$ | $P(X=1)=p$ | $p\in[0,1]$ |
| Beta | $[0,1]$ | $f(x)\propto x^{\alpha-1}(1-x)^{\beta-1}$ | $\alpha,\beta>0$ |
| Categorical | $\{1,\dots,n\}$ | $P(X=i)=p_i$ | probability vector in $n$ dimensions |
| Binomial | $\mathbb N_0$ | $p(k)=\binom nk p^k(1-p)^{n-k}$ | trials $n\in\mathbb N$, success probability $p\in[0,1]$ |
| Poisson | $\mathbb N_0$ | $f(k)=\frac{\lambda^k}{k!}e^{-\lambda}$ | rate $\lambda>0$ |
| Weibull | $x\ge0$ | $f(x)=\frac k\lambda\big(\frac x\lambda\big)^{k-1}e^{-(x/\lambda)^k}$ | $\lambda,k>0$ |

(The sheet also plots each family for several parameter settings. Note the Normal is written with $\sigma$ in the table but the exercise's note uses the variance $\sigma^2$: read the question.) **Identities:** (1) $\log ab=\log a+\log b$; (2) $\log a^b=b\log a$; (3) $\log\exp x=x$; (4) $\sum_{i=1}^nx_i=x_1+\dots+x_n$; (5) $\prod_{i=1}^nx_i=x_1\cdots x_n$; (6) $p(a,b)=p(a|b)p(b)$; (7) $p(a|b)=\frac{p(b|a)p(a)}{p(b)}$ (Bayes' rule).

**How to choose the distribution (practice).** Real-valued label → **Normal**; two classes → **Bernoulli** (sigmoid); $n$ classes → **Categorical** (softmax); count of successes in $n$ trials → **Binomial** (e.g. $n\,\sigma(\cdot)$ trials); count of events → **Poisson** with $\lambda=\exp(\boldsymbol\theta^\top\boldsymbol\phi)$ (positive rate; **Poisson regression**: loss $\sum[\lambda_i-k_i\log\lambda_i]$ up to a constant, gradient $\sum(\lambda_i-k_i)\boldsymbol\phi_i$); positive durations → **Weibull**; a probability itself → **Beta**.

**Worked mini-example: Poisson regression.** $k_i\sim\text{Poisson}(\lambda_i)$, $\lambda_i=\exp(\theta\,x_i)$. Log-likelihood $=\sum[k_i\log\lambda_i-\lambda_i-\log k_i!]=\sum[k_i\theta x_i-e^{\theta x_i}]-\text{const}$, so $\mathcal L=\sum[e^{\theta x_i}-k_i\theta x_i]$ and $\mathcal L'=\sum(e^{\theta x_i}-k_i)x_i$. Same shape as before: **(prediction − truth) × input**. With data $(x,k)=(0,1),(1,3),(2,8)$: at $\theta=0$, $\mathcal L'=(1-1)\cdot0+(1-3)\cdot1+(1-8)\cdot2=-16$, so gradient descent increases $\theta$; the optimum solves $\sum(e^{\theta x_i}-k_i)x_i=0$, i.e. $(e^{\theta}-3)+2(e^{2\theta}-8)=0$, which numerically gives $\theta\approx1.04$.

### 19.6 Midterm checklist

* Be able to run the **six-step recipe** from memory on any label type (Lecture 12 table + the sheet).
* Keep **all** parameter-dependent terms in the log-density (see 19.3).
* Gradients have the form (prediction − truth) × input, for each parameter vector.
* Know GD stability ($\alpha<2/\lambda_{\max}$), convexity, SGD vs batch, momentum (Lecture 18).
* Know MAP = regularization, bias/variance, validation vs test (Lectures 13–14), NN as learned basis functions (Lecture 15), RNN/attention idea (Lectures 16–17).
* Calculator-only: practice **exp, log and square roots by hand** with 3 significant digits.

#### 📖 Textbook deep dive
* **Heteroscedastic regression** (input-dependent variance): Murphy p.59 (linear case, $\sigma^2$ depends on the input), §13.2.4.4 *"MLP for heteroskedastic regression"* (p.433: the network outputs both $\mu(x)$ and $\sigma^2(x)$, trained by the NLL with a positivity transform for $\sigma^2$) and p.376; the NLL $\frac12\log\sigma^2(x)+\frac{(y-\mu(x))^2}{2\sigma^2(x)}+\text{const}$ is exactly the corrected loss of 19.3. Bishop p.273 and p.311 mention heteroscedastic models (input-dependent noise variance), and Bishop's Gaussian log-likelihood (1.62) keeps the $-\frac12\ln\sigma^2$ term.
* **Distribution families:** Murphy §2.2 (Bernoulli/Binomial/Categorical, 2.2.1–2.2.2; Poisson §2.2.3.3; Normal §2.5; Beta §2.2.4) and Bishop §2.1–2.3 (binary, multinomial and Gaussian variables). **GLMs** (Murphy §12.1–12.2): Poisson regression with the log link is the standard count model.

#### ➕ Textbook extra
Murphy §2.7 (exponential family), §12.3 (generalized linear models, canonical link) and Bishop §2.4 (exponential family) generalize why "(prediction − truth) × input" is the gradient for so many models.

#### ✅ Key takeaways
* Midterm Nov 16, in class, paper, teams of two, calculator only.
* The exercise = Normal with **both** $\mu=\boldsymbol\theta_1^\top\boldsymbol\phi$ and $\sigma^2=\exp(\boldsymbol\theta_2^\top\boldsymbol\phi)$; correct loss $\sum[\frac12\boldsymbol\theta_2^\top\boldsymbol\phi+\frac{r^2}{2e}]$; gradients $-\frac r e\boldsymbol\phi$ and $(\frac12-\frac{r^2}{2e})\boldsymbol\phi$.
* The reference sheet gives seven distributions and seven identities; know how to apply them.

---

## Part D. Glossary

Entries are alphabetical; the lecture where the idea appears is in brackets.

* **Accuracy** — fraction of predictions that are correct, $(TP+TN)/\text{all}$. [08, 10]
* **Activation function** — non-linear function $h$ applied to each hidden unit's input: sigmoid, tanh, ReLU, leaky ReLU, softplus. With the identity the network collapses to one matrix. [15]
* **Adaptive basis functions** — basis functions whose own parameters ($\mathbf v_m,b_m$) are learned: $\phi_m(\mathbf x)=\sigma(\mathbf v_m^\top\mathbf x+b_m)$; this is a two-layer neural network. [15]
* **Argmax / argmin** — the *input* that gives the largest / smallest output. [04]
* **Axis (NumPy)** — a dimension of an array; in a reduce, `axis=k` makes dimension $k$ disappear. [02]
* **Backpropagation** — the chain rule applied backward through the network to get all gradients: $\delta_{\rm hidden}=h'(a)\odot(\mathbf W^\top\delta_{\rm out})$. [15]
* **Bayes' law** — $p(A|B)=p(B|A)p(A)/p(B)$; flips a conditional probability. [13]
* **Bernoulli distribution** — two outcomes (1 with probability $\theta$, 0 with $1-\theta$): $\theta^x(1-\theta)^{1-x}$. [03]
* **Beta distribution** — distribution on $[0,1]$ with density $\propto\theta^{\alpha-1}(1-\theta)^{\beta-1}$; mode $(\alpha-1)/(\alpha+\beta-2)$; the conjugate prior of the Bernoulli. [13]
* **Bias (estimator)** — $\mathbb E[\hat\theta]-\theta$. The variance MLE has bias factor $(n-1)/n$. [04, 10, 12]
* **Bias–variance trade-off** — simple models: high bias, low variance; complex models: low bias, high variance. [10]
* **Bessel's correction** — using $1/(n-1)$ instead of $1/n$ for the sample variance. [12]
* **Binary cross-entropy (BCE)** — logistic-regression loss $-\frac1D\sum[y\log p+(1-y)\log(1-p)]$. [08]
* **Bootstrap** — resample the data with replacement to approximate a sampling distribution. [11 (extra)]
* **Branch (Git)** — a separate line of commits; you work on `solution`, never on `main`. [05]
* **Broadcasting** — NumPy's rule for combining arrays of different shapes: align from the right; dimensions must be equal or 1. [02]
* **Categorical distribution** — $K$ outcomes with a probability vector in the simplex. [03, 09]
* **Central limit theorem (CLT)** — $\sqrt n(\bar X_n-\mu)\to\mathcal N(0,\sigma^2)$ for IID variables with finite variance. [11, 12]
* **Cholesky factor** — lower-triangular $\mathbf L$ with $\boldsymbol\Sigma=\mathbf L\mathbf L^\top$; used to sample correlated Gaussians. [06]
* **Classification** — predicting a discrete label. [08, 09]
* **Commit / commit hash** — a saved snapshot in Git / its unique fingerprint, which you submit. [05]
* **Condition number** — ratio of largest to smallest eigenvalue; large means a long narrow loss valley and slow gradient descent. [06, 07, 08]
* **Confidence interval** — a procedure whose interval captures the true parameter in a stated fraction of repeated experiments (not "probability this interval contains it"). [11]
* **Confusion matrix** — table of actual vs predicted classes. [08, 10]
* **Conjugate prior** — a prior whose posterior is in the same family (Beta prior + Bernoulli data ⇒ Beta posterior). [13]
* **Convex function** — a bowl-shaped function with one global minimum; logistic and least-squares losses are convex. [08]
* **Cross-entropy** — $H(p,q)=-\sum p\log q$; the classification loss. [09, 12]
* **Cross-validation (CV)** — averaging test error over several train/test splits; $K$-fold, leave-one-out. [10]
* **Data augmentation** — enlarging the training set with transformed copies (noise, flips, rotations) to regularize. [15]
* **Dataset $\mathcal D$** — the collection of data points. [04]
* **Decision boundary** — where predicted probability $=\tau$ (often 0.5). [08]
* **Deflating a baseline** — not tuning the comparison method as carefully as your own. [11]
* **Density (pdf)** — function whose *area* gives probability; can exceed 1. [03]
* **Design matrix** — matrix with one row per example (and a column of 1s for the intercept). [06, 07]
* **Dropout** — randomly drop each hidden unit with probability $p$ during training; at test time scale weights by the keep probability $1-p$ (or average several dropout passes: Monte Carlo dropout). [15]
* **dtype** — the type of numbers in a NumPy array. [02]
* **Early stopping** — stop training when the validation loss starts to rise; acts like $L^2$ regularization. [15]
* **Entropy** — average surprise $-\sum p\log p$. [12]
* **Error residual** — $e=f_{\boldsymbol\theta}(\mathbf x)-y$ (prediction minus truth). [07, 10]
* **Event space $\Omega$** — the set of all possible outcomes. [03]
* **Evidence** — $p(\mathcal D)$, the denominator of Bayes' law; drop it when maximizing over $\theta$. [13, 14]
* **F₁ / F$_\beta$ score** — harmonic-mean-style combination of precision and recall; $F_\beta=(1+\beta^2)PR/(\beta^2P+R)$. [10]
* **Fancy indexing** — indexing with a list of positions; returns a copy. [02]
* **Fallout (false-positive rate)** — $FP/(FP+TN)$. [10]
* **Generalization (gap)** — performance on new data (difference between test and training performance). [10]
* **Gradient** — vector of partial derivatives; points uphill. [08]
* **Gradient descent** — repeat $\boldsymbol\theta\leftarrow\boldsymbol\theta-\alpha\nabla J$. [08]
* **Hat matrix** — $\mathbf X(\mathbf X^\top\mathbf X)^{-1}\mathbf X^\top$, the projection of $\mathbf y$ onto the column space. [07]
* **Hessian** — matrix of second derivatives; curvature. [08]
* **Hidden unit** — an internal unit $z_m=h(\mathbf v_m^\top\mathbf x)$ whose value is not observed in the data. [15]
* **Hyperparameter** — a setting not learned by fitting (learning rate, degree, $\lambda$). [10, 11]
* **IID** — independent and identically distributed. [03]
* **Inductive bias** — the assumptions built into a learning algorithm. [10]
* **Indicator $\mathbb 1[\cdot]$** — 1 if the statement is true, else 0. [09]
* **Inductive bias** — built-in assumptions (a prior, an architecture) that make a method prefer some solutions. [14, 15]
* **Inverse / pseudo-inverse** — $\mathbf M^{-1}$ undoes $\mathbf M$; $(\mathbf X^\top\mathbf X)^{-1}\mathbf X^\top$ is the left pseudo-inverse. [06, 07]
* **KL divergence** — $\sum p\log(p/q)$; cross-entropy minus entropy. [12]
* **Laplace distribution** — $\frac1{2b}e^{-|w-\mu|/b}$; sharp peak at $\mu$, heavier tails than a Gaussian. [14]
* **Lasso regression** — least squares plus $\lambda\sum|\theta_i|$ ($L^1$; Laplace prior); gives exactly-zero weights (feature selection). [14]
* **Learning rate $\alpha$** — gradient-descent step size. [08]
* **Likelihood** — $p(\mathcal D|\boldsymbol\theta)$ as a function of $\boldsymbol\theta$; not a distribution over $\boldsymbol\theta$. [04]
* **Likelihood / prior / posterior** — $p(\mathcal D|\theta)$, $p(\theta)$, $p(\theta|\mathcal D)\propto p(\mathcal D|\theta)p(\theta)$. [13]
* **Linear in the parameters** — the model is a sum of parameters times fixed features. [07]
* **Log-likelihood** — the log of the likelihood: products become sums. [04]
* **Log-sum-exp** — $\log\sum e^{a_c}=m+\log\sum e^{a_c-m}$; avoids overflow. [09]
* **Logit** — the log-odds $\log\frac p{1-p}$; also the raw scores before softmax. [08, 09]
* **MAP estimate** — $\theta^*\in\arg\max_\theta p(\mathcal D|\theta)p(\theta)$; MLE plus a prior; equals minimizing loss + penalty. [13, 14]
* **Mask (boolean)** — array of True/False used to select/count elements. [02]
* **Matrix** — a 2-D tensor. [06]
* **Maximum likelihood estimate (MLE)** — $\arg\max_{\boldsymbol\theta}$ likelihood. [04]
* **Mean squared error (MSE)** — $\frac1D\sum(y-\hat y)^2$; Gaussian NLL up to constants. [07]
* **Multilayer perceptron (MLP)** — $\hat{\mathbf y}=\mathbf g(\mathbf W\mathbf h(\mathbf V\mathbf x))$: layers of perceptron-like units; a feed-forward network. [15]
* **Mutual information** — $I(X,Y)=H(X)+H(Y)-H(X,Y)$. [12]
* **Negative log-likelihood (NLL/D)** — $-\frac1D\log L$; the generic loss. [04]
* **Norm** — size of a vector: L1, L2, L∞. [06]
* **Normal (Gaussian) distribution** — bell curve $\mathcal N(\mu,\sigma^2)$. [03]
* **Normal equations** — $\mathbf X^\top\mathbf X\boldsymbol\theta=\mathbf X^\top\mathbf y$. [07]
* **Out-of-distribution (OOD)** — test data from a different distribution than the training data. [10]
* **Overfitting / underfitting** — good training but poor test performance / poor even on training data. [10]
* **Perceptron** — $f(\mathbf x)=\text{sign}(\mathbf w^\top\mathbf x+b)$; update $\mathbf w\leftarrow\mathbf w+y\mathbf x$ on mistakes; converges only for linearly separable data. [15]
* **Perplexity** — $2^{H}$; effective number of equally likely outcomes. [12 (extra)]
* **Precision / recall** — $TP/(TP+FP)$ / $TP/(TP+FN)$. [08, 10]
* **Probability simplex $\Delta_C$** — vectors with non-negative entries summing to 1. [09]
* **Pseudo-counts** — the Beta parameters $\alpha-1,\beta-1$ act like imaginary heads/tails added to the data. [13]
* **Random variable** — a rule turning outcomes into numbers. [03]
* **$R^2$** — $1-\text{RSS}/\text{TSS}$; fraction of variance explained. [07]
* **Reduce** — an operation that collapses an array along an axis (sum, mean, max, argmax). [02]
* **Regularization** — any method that constrains the parameters to reduce overfitting: $L^p$ penalties, weight decay, augmentation, early stopping, dropout. [14, 15]
* **ReLU** — $\max(0,x)$; non-saturating, but a unit with negative input has zero gradient ("dead"). [15]
* **Ridge regression** — least squares plus $\lambda\sum\theta_i^2$ ($L^2$; Gaussian prior); $(\mathbf X^\top\mathbf X+\lambda\mathbf I)^{-1}\mathbf X^\top\mathbf y$; shrinks, never exactly zero. [14]
* **ROC / PR curve** — TPR vs FPR / precision vs recall as the threshold varies. [08, 10]
* **Sample standard deviation** — $\sqrt{\frac1{n-1}\sum(x_i-\bar x)^2}$. [11, 12]
* **Scalar prediction error** — $\sigma(\boldsymbol\theta^\top\mathbf x)-y$. [08]
* **Seed** — fixed starting value of a random generator; aids debugging, not reproducibility. [11]
* **σ-algebra** — the collection of events we allow probabilities for. [03]
* **Sigmoid** — $\sigma(z)=1/(1+e^{-z})$. [08]
* **Singular matrix** — has no inverse (dependent columns). [06]
* **Softmax** — $e^{x_i}/\sum_je^{x_j}$; scores → probabilities. [09]
* **Soft-thresholding** — $\text{sign}(x)(|x|-\delta)_+$; the lasso update for one coordinate. [14]
* **Sparse model** — many parameters exactly 0. [14]
* **Standard error of the mean (SEM)** — $\hat\sigma/\sqrt n$. [11, 12]
* **Temperature** — divide logits by $T$ before softmax to flatten ($T$ large) or sharpen ($T$ small). [09 (extra)]
* **Tensor** — array with any number of dimensions. [02]
* **Threshold $\tau$** — probability cut-off for turning a probability into a class. [08]
* **Universal approximation theorem** — one hidden layer with enough units can approximate any continuous function (says nothing about generalization). [15]
* **Vanishing gradients** — gradients that shrink layer by layer ($\sigma'\le0.25$, so $\le0.25^L$ through $L$ sigmoid layers). [15]
* **Variance** — $\mathbb E[(X-\mu)^2]$. [03]
* **View vs copy** — slices/reshape share memory (view); lists and masks make copies. [02]
* **Weight decay** — $L^2$ penalty in a neural network; each step multiplies weights by $(1-\alpha\lambda)$. [14, 15]
* **Xavier / Glorot initialization** — weights $\sim\mathcal N(0,\sigma^2)$ with $\sigma^2=2/(n_{\rm in}+n_{\rm out})$; He uses $2/n_{\rm in}$ for ReLU. [14, 15]

---

**Added with Lectures 16–19:**

* **DFA**: deterministic finite automaton: a machine with states that jumps to a new state on each symbol (L16)
* **Markov chain**: random sequence where the next state depends only on the current one (L16)
* **hidden state**: unobserved summary $h_t$ of the past that is learned (L16–L17)
* **RNN**: recurrent neural network: one network reused each step with a hidden vector (L17)
* **LSTM / GRU**: RNN blocks with gates that let information and gradients flow over long spans (L17)
* **attention (query, key, value)**: soft dictionary lookup: softmax of query·key scores averages the values (L17)
* **Transformer / LLM**: stack of attention layers; an LLM is a Transformer fitted by maximum likelihood to text (L17)
* **teacher forcing**: training with the true previous tokens as input (L17)
* **perplexity**: $e^{\text{average token NLL}}$: effective number of equally likely choices (L17)
* **convex function**: bowl-shaped: chord lies above the curve; every local minimum is global (L18)
* **SGD / mini-batch**: gradient from a random subset of $B$ data points: unbiased, noisy, cheap (L18)
* **momentum**: $\mathbf m_t=\beta\mathbf m_{t-1}+\mathbf g_t$: running sum of past gradients that smooths zig-zag (L18)
* **Adam**: optimizer with momentum plus per-weight step sizes (L18)
* **Robbins–Monro**: $\sum\alpha_t=\infty,\ \sum\alpha_t^2<\infty$ for step sizes (L18)
* **autodiff**: gradients computed by chain rule over a forward/backward pass (L18)
* **heteroscedastic**: noise variance depends on the input (L19)


## Part E. Formula sheet

### E.1 Probability (L03)
* Bernoulli: $p(x;\theta)=\theta^x(1-\theta)^{1-x}$. Normal: $f(x)=\frac1{\sqrt{2\pi\sigma^2}}e^{-(x-\mu)^2/(2\sigma^2)}$.
* Independent: $p(A,B)=p(A)p(B)$. Bayes: $p(h|y)=p(h)p(y|h)/p(y)$.
* $\mathbb E[aX+b]=a\mathbb E[X]+b$; $\mathbb V[X]=\mathbb E[X^2]-\mu^2$; $\mathbb V[X+Y]=\mathbb V[X]+\mathbb V[Y]+2\text{Cov}[X,Y]$.
* Central 95% of $\mathcal N(\mu,\sigma^2)$: $\mu\pm1.96\sigma$; within $\pm1\sigma$: 68.27%.

### E.2 Maximum likelihood (L04)
* $L(\theta;\mathcal D)=\prod_ip(x_i|\theta)$; $\log L=\sum_i\log p(x_i|\theta)$; loss $=-\frac1D\log L$.
* Bernoulli: $\hat\theta=c_1/D$. Categorical: $\hat\theta_k=N_k/N$. Gaussian: $\hat\mu=\bar x$, $\hat\sigma^2=\frac1N\sum(x-\bar x)^2$ (biased: $\mathbb E=\frac{N-1}N\sigma^2$).
* MLE = minimizing $\text{KL}(p_{\rm emp}\|q_\theta)$.

### E.3 Linear algebra (L06)
* $\|\mathbf v\|_1=\sum|v_i|$, $\|\mathbf v\|_2=\sqrt{\sum v_i^2}$, $\|\mathbf v\|_\infty=\max|v_i|$.
* $(\mathbf A\mathbf B)_{ij}=\sum_\ell a_{i\ell}b_{\ell j}$; $(m\times n)(n\times k)=m\times k$.
* $(\mathbf A\mathbf B)^\top=\mathbf B^\top\mathbf A^\top$; $\mathbf X^+=(\mathbf X^\top\mathbf X)^{-1}\mathbf X^\top$.
* Cholesky: $\boldsymbol\Sigma=\mathbf L\mathbf L^\top$; sample $\mathbf y=\mathbf L\mathbf z+\boldsymbol\mu$, $\mathbf z\sim\mathcal N(\mathbf 0,\mathbf I)$.

### E.4 Regression (L07)
* Model $\hat y=\boldsymbol\varphi(\mathbf x)^\top\boldsymbol\theta$; loss $\text{MSE}=\frac1D\sum(y_i-\hat y_i)^2$.
* Gradient $\nabla\text{MSE}=\frac2D\mathbf X^\top(\mathbf X\boldsymbol\theta-\mathbf y)$; solution $\boldsymbol\theta=(\mathbf X^\top\mathbf X)^{-1}\mathbf X^\top\mathbf y$ = `pinv(X) @ y`.
* Simple regression: slope $=S_{xy}/S_{xx}$, intercept $=\bar y-\text{slope}\cdot\bar x$.
* $R^2=1-\text{RSS}/\text{TSS}$; ridge: $(\mathbf X^\top\mathbf X+\lambda\mathbf I)^{-1}\mathbf X^\top\mathbf y$.
* Gradient-descent stability: $\alpha<2/\lambda_{\max}(\mathbf H)$.

### E.5 Classification (L08, L09)
* Sigmoid: $\sigma(z)=1/(1+e^{-z})$; $\sigma'=\sigma(1-\sigma)$; $1-\sigma(z)=\sigma(-z)$; logit $=\log\frac p{1-p}$.
* $p(y{=}1|\mathbf x)=\sigma(\boldsymbol\theta^\top\mathbf x)$; BCE $J=-\frac1D\sum[y\log p+(1-y)\log(1-p)]$; $\nabla J=\frac1D\sum(p_i-y_i)\mathbf x_i=\frac1D\mathbf X^\top(\mathbf p-\mathbf y)$.
* GD: $\boldsymbol\theta\leftarrow\boldsymbol\theta-\alpha\nabla J$. Newton/IRLS: $\boldsymbol\theta\leftarrow\boldsymbol\theta-\mathbf H^{-1}\mathbf g$, $\mathbf H=\frac1D\mathbf X^\top\mathbf S\mathbf X$, $\mathbf S=\text{diag}(p_i(1-p_i))$.
* Softmax: $s(\mathbf x)_i=e^{x_i}/\sum_je^{x_j}$; $\text{softmax}(\mathbf a/T)$; $\log\text{softmax}_c=a_c-\text{lse}(\mathbf a)$; two classes: $\sigma(a-b)$.
* Cross-entropy $J=-\frac1D\sum_i\log p_{i,y_i}$; gradient $\frac1D\mathbf X^\top(\mathbf P-\mathbf Y)$.
* Metrics: precision $=\frac{TP}{TP+FP}$; recall $=\frac{TP}{TP+FN}$; FPR $=\frac{FP}{FP+TN}$; $F_1=\frac{2PR}{P+R}$; $F_\beta=\frac{(1+\beta^2)PR}{\beta^2P+R}$. Cost-sensitive threshold: predict 1 iff $p_1\ge\frac1{1+c}$.

### E.6 Generalization (L10)
* Expected squared error $=\text{bias}^2+\text{variance}+\text{noise}$; estimator MSE $=\text{var}+\text{bias}^2$.
* $K$-fold CV: $\bar e=\frac1K\sum_ke_k$; then retrain on all data.

### E.7 Statistics and information (L11, L12)
* CLT: $\sqrt n(\bar X_n-\mu)\to\mathcal N(0,\sigma^2)$; $\text{SEM}=\hat\sigma/\sqrt n$; $\hat\sigma=\sqrt{\frac1{n-1}\sum(x_i-\bar x)^2}$; $\Pr(|\bar X-\mu|\le\text{SEM})\approx0.682$; 95%: $\pm1.96\,\text{SEM}$ (use $t_{n-1}$ for small $n$).
* Entropy $H=-\sum p\log p$; cross-entropy $-\sum p\log q$; $\text{KL}=H_{ce}-H$; $H(Y|X)=H(X,Y)-H(X)$; $I=H(X)+H(Y)-H(X,Y)$; perplexity $2^H$.

### E.8 Bayes, MAP, regularization and neural networks (L13–L15)
* Bayes: $p(A|B)=\dfrac{p(B|A)p(A)}{p(B)}$; posterior $p(\theta|\mathcal D)=\dfrac{p(\mathcal D|\theta)p(\theta)}{p(\mathcal D)}\propto p(\mathcal D|\theta)p(\theta)$.
* MAP: $\theta^*\in\arg\max[\log p(\mathcal D|\theta)+\log p(\theta)]=\arg\min[\text{loss}-\log p(\theta)]$.
* Beta$(\alpha,\beta)$ prior, $H$ heads in $D$ flips: $\theta_{\rm MAP}=\dfrac{H+\alpha-1}{D+\alpha+\beta-2}$ (add-one smoothing: $\alpha=\beta=2$ gives $\frac{H+1}{D+2}$); posterior $=\text{Beta}(\alpha+H,\beta+D-H)$; mean $\frac{\alpha}{\alpha+\beta}$, mode $\frac{\alpha-1}{\alpha+\beta-2}$.
* $L^p$ prior $p(\theta)\propto e^{-\lambda\sum|\theta_i|^p}$ ⇒ $J(\theta)=\text{MSE}(\theta)+\lambda\sum_i|\theta_i|^p$ ($p=2$ ridge/Gaussian, $p=1$ lasso/Laplace).
* Ridge: $\hat\theta=(\mathbf X^\top\mathbf X+\lambda\mathbf I)^{-1}\mathbf X^\top\mathbf y$ for $\text{RSS}+\lambda\|\theta\|^2$, with $\lambda=\sigma^2/\tau^2$ (Gaussian noise variance over prior variance); orthonormal design: $\hat w=\hat w_{\rm ols}/(1+\lambda)$; SVD shrink factor $\sigma_j^2/(\sigma_j^2+\lambda)$.
* Lasso (for $\text{RSS}+\lambda\|\mathbf w\|_1$): coordinate $d$: $\hat w_d=\text{SoftThreshold}(c_d,\lambda/2)/a_d$, $a_d=\sum_nx_{nd}^2$, $c_d=\sum_nx_{nd}(y_n-\mathbf w_{-d}^\top\mathbf x_{n,-d})$; SoftThreshold$(x,\delta)=\text{sign}(x)(|x|-\delta)_+$; $\lambda_{\max}=2\|\mathbf X^\top\mathbf y\|_\infty$ (all weights 0 above it).
* Perceptron: $f(\mathbf x)=\text{sign}(\mathbf w^\top\mathbf x+b)$; misclassified iff $y\hat y<0$; update $\mathbf w\leftarrow\mathbf w+y\mathbf x$.
* MLP: $\hat{\mathbf y}=\mathbf g(\mathbf W\mathbf h(\mathbf V\mathbf x))$; backprop: $\partial L/\partial\mathbf W=\delta_{\rm out}\mathbf z^\top$, $\delta_{\rm hid}=h'(\mathbf a)\odot(\mathbf W^\top\delta_{\rm out})$, $\partial L/\partial\mathbf V=\delta_{\rm hid}\mathbf x^\top$. Output pairs: identity+MSE, sigmoid+BCE, softmax+CE (output gradient $\hat y-y$).
* Activations: $\sigma'=\sigma(1-\sigma)\le\frac14$; $\tanh x=2\sigma(2x)-1$, $\tanh'=1-\tanh^2$; ReLU $\max(0,x)$; leaky $\max(0,x)+\gamma\min(0,x)$; softplus $\log(1+e^x)$.
* Parameters of $D\to M\to C$: $M(D+1)+C(M+1)$. Xavier $\sigma^2=\frac2{n_{\rm in}+n_{\rm out}}$, He $\frac2{n_{\rm in}}$. Weight decay step: $\mathbf w\leftarrow(1-\alpha\lambda)\mathbf w-\alpha\nabla L$. Dropout (drop prob. $p$): test-time weights $\times(1-p)$.

### E.10 Sequences, optimization and the midterm sheet (L16–L19)
* Markov: $p(X_{t+1}|X_{0:t})=p(X_{t+1}|X_t)$; chain rule $p(x_{0:T})=p(x_0)\prod p(x_t|x_{<t})$; hidden-state sum $\sum_{h}\prod p(x_t|h_t)p(h_t|h_{t-1},x_{t-1})$.
* Attention: $\mathbf z_t=\sum_{i\le t}\text{softmax}_i(\mathbf q_t^\top\mathbf k_i/\sqrt d)\mathbf v_i$; LLM loss $=-\sum\log p(x_t|x_{<t})$; perplexity $=\exp(\text{mean NLL})$.
* Gradients: MSE $\frac2D\sum(\theta^\top x_i-y_i)x_i$; logistic $\frac1D\sum(\sigma(\theta^\top x_i)-y_i)x_i$; GD $\theta\leftarrow\theta-\alpha\nabla J$, stable iff $\alpha<2/\lambda_{\max}$.
* Momentum $\mathbf m_t=\beta\mathbf m_{t-1}+\mathbf g_t$, $\theta\leftarrow\theta-\alpha\mathbf m_t$. Robbins–Monro $\sum\alpha_t=\infty$, $\sum\alpha_t^2<\infty$.
* Midterm example: $\mu=\theta_1^\top\phi$, $\sigma^2=\exp(\theta_2^\top\phi)$; $\mathcal L=\sum[\frac12\theta_2^\top\phi+\frac{r^2}{2e}]$; $\nabla_{\theta_1}=\sum-\frac re\phi$, $\nabla_{\theta_2}=\sum(\frac12-\frac{r^2}{2e})\phi$.
* Poisson regression: $\lambda=\exp(\theta^\top\phi)$, $\mathcal L=\sum[\lambda_i-k_i\log\lambda_i]$, $\nabla=\sum(\lambda_i-k_i)\phi_i$.
* Sheet identities: $\log ab=\log a+\log b$, $\log a^b=b\log a$, $\log\exp x=x$, $p(a,b)=p(a|b)p(b)$, Bayes.

### E.9 Running-example numbers (to sanity-check your code)
| Quantity | Value |
|---|---|
| mean hours / score | 4.1 / 62.4 |
| OLS $\boldsymbol\theta$ | [41.8078, 5.0225] |
| MSE, RSS, $R^2$ | 8.2875, 82.875, 0.9370 |
| Logistic MLE $\mathbf w^*$ / NLL | [−6.6836, 1.4495] / 0.26438 |
| Decision boundary | 4.611 h |
| $H(0.4)$ | 0.971 bits |
| $I(\text{hours},\text{pass})$ | 0.2564 bits |
| Degree 1 CV MSE (5-fold / LOO) | 10.63 / 11.76 |
| SEM of scores | 3.824 |
| $\theta_{\rm MAP}$, Beta(2,2), $H=4,D=10$ | 5/12 = 0.4167 |
| Ridge slope, $\lambda=50$ (RSS form, free intercept) | 2.4833 (OLS 5.0225) |
| Lasso slope, $\lambda=100$ / $\lambda\ge491.2$ | 4.000 / 0 |
| MLP worked example: $\hat y$, loss, loss after one step ($\alpha=0.1$) | 1, 2, 0.28125 |
| Perceptron 4-point example | 4 updates, $\mathbf w=(0,-2,4)$ |

---

## Part F. Code cheat sheet (Python)

All blocks below were run; they assume the next block's imports and the running example.

### F.1 Setup and the running example
```python
import numpy as np
from scipy import stats
h = np.array([4, 2, 6, 1, 3, 5, 7, 3, 8, 2], dtype=float)           # hours
s = np.array([66, 48, 71, 45, 57, 62, 77, 59, 83, 56], dtype=float) # scores
t = (s >= 65).astype(int)                                           # pass/fail
y3 = np.digitize(s, [65, 75])                                       # 0/1/2
X = np.stack([np.ones_like(h), h], axis=1)                          # (10, 2) design matrix
```

### F.2 Tensors (L02)
```python
h.shape, h.ndim, h.dtype                 # (10,), 1, float64
mask = h >= 5;  mask.sum();  h[mask]     # count 4; values [6. 5. 7. 8.]
X.mean(axis=0)                           # [1.0, 4.1]  (axis 0 disappears)
h - h.mean()                             # broadcasting (10,) - ()
D = h[:, None] - h[None, :]              # (10,1) - (1,10) -> (10,10) all pairs
h.reshape(2, 5);  h.reshape(-1, 2)       # same data, new shape
v = h[1:-1]                              # VIEW: editing v edits h
c = h[[2, 4, 1]]                         # COPY (fancy indexing)
w = h.copy()                             # explicit copy
```

### F.3 Probability and MLE (L03, L04)
```python
def bernoulli(theta, x):                 # lecture function
    theta, x = np.atleast_1d(theta).astype(float), np.atleast_1d(x)
    if theta.shape[0] > 1 and x.shape[0] > 1:
        theta, x = theta.reshape(-1, 1), x.reshape(1, -1)
    return theta ** x * (1 - theta) ** (1 - x)

grid = np.linspace(0, 1, 101)
p = bernoulli(grid, t)                               # (101, 10)
L = p.prod(axis=-1)                                  # likelihood per theta
loss = -np.log(p + 1e-8).mean(axis=-1)               # NLL / D
grid[L.argmax()], grid[loss.argmin()]                # 0.4, 0.4
rng = np.random.default_rng(0)
samples = rng.normal(170, 10, size=1000)             # draw from N(170, 10^2)
```

### F.4 Linear algebra and regression (L06, L07)
```python
np.linalg.norm([3, 2], 1), np.linalg.norm([3, 2]), np.linalg.norm([3, 2], np.inf)  # 5, 3.606, 3
theta = np.linalg.pinv(X) @ s                         # [41.8078, 5.0225]
theta2 = np.linalg.solve(X.T @ X, X.T @ s)            # normal equations, same answer
resid = s - X @ theta
mse = (resid ** 2).mean()                             # 8.2875
r2 = 1 - (resid ** 2).sum() / ((s - s.mean()) ** 2).sum()   # 0.9370
Sigma = np.array([[4., 2.], [2., 3.]]); Lc = np.linalg.cholesky(Sigma)
y = Lc @ np.array([1., 1.]) + np.array([10., 20.])    # [12., 22.414]
def poly_fit(h, s, d):                                # polynomial basis fit
    return np.linalg.lstsq(np.vander(h, d + 1, increasing=True), s, rcond=None)[0]
```

### F.5 Logistic regression: GD and Newton (L08)
```python
sigmoid = lambda z: 1 / (1 + np.exp(-z))
def nll(w):  p = sigmoid(X @ w);  return -(t * np.log(p) + (1 - t) * np.log(1 - p)).mean()
def grad(w): return X.T @ (sigmoid(X @ w) - t) / len(t)

w = np.zeros(2)
for _ in range(5):                                    # gradient descent, alpha = 0.1
    w = w - 0.1 * grad(w)
w, nll(w)                                             # [-0.0747, 0.0909], 0.6607

w = np.zeros(2)
for _ in range(6):                                    # Newton / IRLS
    p = sigmoid(X @ w)
    H = (X * (p * (1 - p))[:, None]).T @ X / len(t)
    w = w - np.linalg.solve(H, grad(w))
w, nll(w)                                             # [-6.6836, 1.4495], 0.26438
p = sigmoid(X @ w)

def confusion(p, y, tau=0.5):
    pred = p >= tau
    TP = (pred & (y == 1)).sum(); FP = (pred & (y == 0)).sum()
    FN = (~pred & (y == 1)).sum(); TN = (~pred & (y == 0)).sum()
    return TP, FP, FN, TN                             # tau=0.5 -> 3, 1, 1, 5
```

### F.6 Softmax and cross-entropy (L09)
```python
def softmax(z):
    z = z - z.max(axis=-1, keepdims=True)             # stability: subtract the max
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)

softmax(np.array([2., 1., 0.]))                       # [0.6652, 0.2447, 0.0900]
softmax(np.array([1000., 1001., 1000.]))              # [0.2119, 0.5761, 0.2119]
def cross_entropy(P, y):                              # P: (N, C) probs, y: (N,) int labels
    return -np.log(P[np.arange(len(y)), y]).mean()
Y = np.eye(3)[y3]                                     # one-hot labels (10, 3)
W = np.zeros((2, 3))
P = softmax(X @ W)
G = X.T @ (P - Y) / len(y3)                           # gradient of the cross-entropy
cross_entropy(P, y3)                                  # log 3 = 1.0986
```

### F.7 Cross-validation (L10)
```python
def cv_mse(h, s, d, folds):
    errs = []
    for f in folds:
        m = np.ones(len(h), bool); m[f] = False       # mask = training indices
        th = poly_fit(h[m], s[m], d)
        pred = np.vander(h[f], d + 1, increasing=True) @ th
        errs.append(((s[f] - pred) ** 2).mean())
    return np.mean(errs)
folds = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]
[round(cv_mse(h, s, d, folds), 2) for d in (1, 2, 3)]   # [10.63, 15.27, 12.86]
```

### F.8 Statistics and information (L11, L12)
```python
A = np.array([.91, .93, .92, .90, .94]); B = np.array([.86, .88, .87, .85, .89])
A.mean(), A.std(ddof=1), stats.sem(A)                 # 0.92, 0.01581, 0.007071
stats.ttest_ind(A, B, equal_var=False)                # t = 5.0, p = 0.00105

def H(p):                                             # entropy in bits
    p = np.asarray(p, float).ravel(); p = p[p > 0]
    return -(p * np.log2(p)).sum()
J = np.array([[.5, .1], [.1, .3]])                    # joint (hours low/high x fail/pass)
H(J.sum(1)) + H(J.sum(0)) - H(J)                      # mutual information = 0.2564 bits
H(J) - H(J.sum(1))                                    # H(T|S) = 0.7145
```

### F.9 Git (L05)
```bash
git clone <url> && cd <repo>
git checkout -b solution          # create + switch to your branch
git add <files>; git commit -m "message"
git push -u origin solution
git log --oneline -3              # recent hashes
git rev-parse HEAD                # full hash to submit
```

### F.10 MAP with a Beta prior (L13)
```python
# continues from F.1; coin = "passed?" labels t, H heads out of D flips
H_, D_ = t.sum(), len(t)                                   # 4, 10
def theta_map(H, D, a, b): return (H + a - 1) / (D + a + b - 2)
print([round(theta_map(H_, D_, a, a), 4) for a in (1, 2, 3, 6)])   # [0.4, 0.4167, 0.4286, 0.45]
print([round(theta_map(10, 10, a, a), 4) for a in (1, 2, 3, 6)])   # [1.0, 0.9167, 0.8571, 0.75]
th = np.linspace(0, 1, 100001)                              # numerical check: argmax of the log-posterior
a = b = 2; lp = (H_ + a - 1) * np.log(th[1:-1]) + (D_ - H_ + b - 1) * np.log(1 - th[1:-1])
print(th[1:-1][lp.argmax()])                                # 0.41667
```

### F.11 Ridge and lasso (L14)
```python
# continues from F.1; RSS + lam*penalty, intercept NOT penalised
hc, sc = h - h.mean(), s - s.mean(); Sxx, Sxy = (hc ** 2).sum(), (hc * sc).sum()
soft = lambda z, d: np.sign(z) * max(abs(z) - d, 0)         # soft-thresholding
ridge = lambda lam: Sxy / (Sxx + lam)                       # one-feature ridge slope
lasso = lambda lam: soft(Sxy, lam / 2) / Sxx                # one-feature lasso slope
print(round(ridge(50), 4), round(lasso(100), 4), lasso(500))   # 2.4833 4.0 0.0
ridge_mat = lambda X, y, lam: np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ y)
def lasso_cd(X, y, lam, iters=500):                         # coordinate descent for ||y-Xw||^2 + lam*||w||_1
    w = np.zeros(X.shape[1]); a = (X ** 2).sum(0)
    for _ in range(iters):
        for d in range(X.shape[1]):
            c = X[:, d] @ (y - X @ w + X[:, d] * w[d])
            w[d] = soft(c, lam / 2) / a[d]
    return w
```

### F.12 Perceptron, MLP and dropout (L15)
```python
# continues from F.1
def perceptron(X, y, epochs=50):                            # w <- w + y*x on mistakes (bias = extra input 1)
    Xa = np.c_[np.ones(len(X)), X]; w = np.zeros(Xa.shape[1]); hist = []
    for _ in range(epochs):
        m = 0
        for xi, yi in zip(Xa, y):
            if yi * (xi @ w) <= 0: w = w + yi * xi; m += 1
        hist.append(m)
        if m == 0: break
    return w, hist
print(perceptron(np.array([[1, 3], [2, 2], [2, 0], [3, 1.]]), np.array([1, 1, -1, -1])))   # w=[0,-2,4], [3,1,0]

def mlp_step(x, y, V, W, lr=0.1):                           # ReLU hidden layer, identity output, L = 1/2 (yhat-y)^2
    a = V @ x; z = np.maximum(a, 0); yhat = W @ z
    dy = yhat - y; dW = np.outer(dy, z); da = (W.T @ dy) * (a > 0); dV = np.outer(da, x)
    return V - lr * dV, W - lr * dW, 0.5 * ((yhat - y) ** 2).sum()
V, W, L0 = mlp_step(np.array([1., 2.]), np.array([3.]), np.array([[.5, -1], [1, .5]]), np.array([[1, .5]]))
print(L0, V.round(2).tolist(), W.round(2).tolist())          # 2.0 [[0.5, -1.0], [1.1, 0.7]] [[1.0, 0.9]]

def dropout(hid, p, rng):                                    # training: keep each unit with probability 1-p
    return hid * (rng.random(hid.shape) >= p)
# test time: multiply the weights by the keep probability (1 - p), or average several dropout passes (MC dropout)
```

---

## Part G. Errata and discrepancies found in the sources

Every item below was checked against the file itself (slides re-viewed as images where the extraction was unclear; textbook text read from the PDF). Severity: **E** = formula/factual error, **M** = misleading or imprecise, **T** = typo/notation.

### G.1 Lecture slides

| # | Where | What the source says | What is right | Sev. |
|---|---|---|---|---|
| 1 | L10 slide 20, $F_\beta$ | $F_\beta=(1-\beta)\dfrac{P\times R}{\beta^2P+R}$ | $F_\beta=(1+\beta^2)\dfrac{P\times R}{\beta^2P+R}$ (Murphy §5.1.4; with $\beta=1$ it must reduce to $F_1$: $2PR/(P+R)$, which the slide's version does not — it gives 0) | E |
| 2 | L10 slide 20, Fallout | $\dfrac{\text{False Positives}}{\text{True Negatives}}$ | $\dfrac{FP}{FP+TN}$ (the false-positive rate); the slide's ratio is not even bounded by 1 | E |
| 3 | L10 slide 19 | "Accuracy is the same as precision in this case." | Accuracy $=(TP+TN)/\text{all}$; precision $=TP/(TP+FP)$. They coincide only in special cases | M |
| 4 | L11 slide 6, CLT | $\bar X_n=\sum_{i=1}^nX_i$ | $\bar X_n=\frac1n\sum X_i$ (the slide-9 definition); with a plain sum, $\sqrt n(\bar X_n-\mu)$ would not tend to $\mathcal N(0,\sigma^2)$ | E |
| 5 | L12 formula box | $H(X)=\mathbb E[\log X]$ | $H(X)=-\mathbb E[\log p(X)]$ (missing minus sign and $p$) | E |
| 6 | L11 slides 9–10; L12 | $\bar X\sim\mathcal N(\mu,\sigma)$ | In the $\mathcal N(\text{mean},\text{variance})$ notation used elsewhere it should read $\mathcal N(\mu,\sigma^2)$ with $\sigma=\hat\sigma/\sqrt n$ | T |
| 7 | L11 slide 11; L12 | A gap between ±1-SEM bars ⇒ "statistically significant" | Only a rough guide: bars that just touch correspond to $p\approx0.157$; 5% needs a difference of ≈2.77 SEM. Also, for $n=5$ runs ±1 SEM covers 62.6% (not 68.2%) under the $t$-distribution | M |
| 8 | L11 slide 10 | "$\mu$ lies in the interval with a probability of 68.2%" | A confidence interval's coverage is over repeated experiments; it is not the probability that this fixed interval contains $\mu$ (Murphy §4.7.5, p.158) | M |
| 9 | L12 right side | "$\hat\sigma$ with $n-1$ has lower bias ($\mathbb E[|\hat\sigma-\sigma|]$ smaller)" | $\mathbb E|\hat\sigma-\sigma|$ is a mean absolute error, not a bias. True in simulation ($0.2811$ vs $0.2844$ at $n=5$) but tiny; the $n-1$ *variance* is unbiased, the $n-1$ *std. dev.* is still slightly low ($0.939$ at $n=5$), and the $1/n$ variance has the smaller MSE ($0.362$ vs $0.503$) | M |
| 10 | L05 slides 3 vs 4 | Slide 3: a repository for each student; slide 4 title: "One Git Repository per Team" | Presumably individual assignments → per student, team assignments → per team; check the assignment handout | T |
| 11 | L13 handwritten notes, coin plots (example 1) | The posterior's dashed peak line is labelled **0.4** | With the plotted prior the posterior peaks at $6/14=0.43$ (the MLE alone is 0.4); a MAP equal to the MLE would mean the prior had no effect | T |
| 12 | L13 handwritten notes, coin plots vs derivation | The derivation uses a Beta$(2,2)$ prior (MAP $=\frac{H+1}{D+2}$), but the plotted heights match Beta$(3,3)$ (example 1) and Beta$(6,6)$ (example 2: MAP $0.75$) | With Beta$(2,2)$ and $H=D=10$ the MAP would be $11/12=0.917$, not $0.75$; state the prior's $\alpha,\beta$ together with each plot | M |
| 13 | L14 slides 11–12 | $-\log p(\mathcal D\mid\theta)=\text{MSE}(\theta)$ | Equal only up to a positive scale and an additive constant: $-\log p=\frac{\text{RSS}}{2\sigma^2}+\frac N2\log(2\pi\sigma^2)$. The minimizer is the same, but $\lambda$ then absorbs $\sigma^2$ and $N$ (see 14.4) | M |
| 14 | L14 slides 16–17 | Loss objective $\lVert\mathbf y-\mathbf X\theta\rVert_2+\alpha\lVert\theta\rVert_2^2$ (and $\ldots+\alpha\lVert\theta\rVert_1$) | The data term must be the **squared** norm (RSS, Murphy Eq. 11.52/11.74); with the un-squared norm the problem is different and no longer the Gaussian-noise MAP | M |
| 15 | L15 slide 23 | $\tanh$: $h(x)=2\sigma(x)-1$ | $\tanh(x)=2\sigma(2x)-1$; $2\sigma(x)-1=\tanh(x/2)$ (at $x=1$: $0.462$ vs $0.762$) | E |
| 16 | L15 slide 38 | Dropout test time: "scale weights by $p$" ($p$ = drop probability) | Multiply by the **keep probability $1-p$** (Murphy p.456). Both agree only at $p=0.5$; at $p=0.2$ the training-time mean output is $0.8\times$ the full one | E |
| 17 | L15 slides 36–37 | Both the early-stopping figure and the dropout (sub-network ensemble) figure are cited as "Fig 7.4 of Goodfellow" | One citation is wrong; the ensemble-of-subnetworks figure is a different figure of that chapter (Fig. 7.6 in the online book) | T |
| 18 | L15 slide 34 | Photo captioned "hamster" | It is the sample animal photo from the imgaug README (a quokka); harmless but confusing | T |

### G.2 Textbooks

| # | Where | What the source says | What is right | Sev. |
|---|---|---|---|---|
| 19 | Murphy §11.2.2.1, Eq. 11.11 (p.373) and §7.7.3, Eq. 7.238 (p.268) | "for any $\mathbf v>0$, $\mathbf v^\top\mathbf X^\top\mathbf X\mathbf v=\|\mathbf X\mathbf v\|^2>0$" | "for any $\mathbf v\ne\mathbf 0$" ($\mathbf v>0$ is meaningless for a vector) | T |
| 20 | Bishop §4.3.2, p.206 | "the error function is concave … and hence has a unique minimum"; "…$\mathbf H$ is positive definite. It follows that the error function is a concave function of $\mathbf w$" | The cross-entropy error is **convex** (positive-definite Hessian ⇒ convex). "Concave … unique minimum" is a slip | E |
| 21 | Murphy Ex 10.1 (p.367–368) | (10.120) is called "the gradient of the NLL" but equals $\sum_i(y_{ic}-\mu_{ic})\mathbf x_i$; (10.121) has a leading minus sign yet the exercise says to show the *NLL* Hessian is positive definite | Those two formulas are the gradient and Hessian of the **log-likelihood**; for the NLL flip both signs: gradient $\sum(\mu-y)\mathbf x$, Hessian $+\sum\mu_c(\delta_{cc'}-\mu_{c'})\mathbf x\mathbf x^\top$ | M |
| 22 | Murphy §10.3.2.4 and Ex 10.1c | Multinomial-logistic Hessian "positive definite" | $\text{diag}(\boldsymbol\mu)-\boldsymbol\mu\boldsymbol\mu^\top$ has a zero eigenvalue (vector of ones) so the full Hessian is only **positive semi-definite** until one class's weights are fixed to 0 (Lecture 09 proof). Convexity still holds | M |
| 23 | Murphy §10.2.3.4, Eq. 10.25 (p.344) | Writes $\|\cdot\|^2>0$ and concludes "strictly convex" | Needs $\mathbf S^{1/2}\mathbf X\mathbf v\ne\mathbf 0$ for $\mathbf v\ne\mathbf0$, i.e. $\mathbf X$ must have full column rank; otherwise only $\ge0$ (convex, not strictly) | T |

### G.2b Lecture slides added with Lectures 16–19

| # | Where | What the source says | What is right | Sev. |
|---|---|---|---|---|
| 24 | L16 slide (language models) | "predict $x$ given $y$" | It means predict $y$ given $x$ (the order is swapped) | T |
| 25 | L16 hidden-state formula | "=" between the chain-rule product and the hidden-state sum | It is a modelling assumption (Markovian $h_t$), not an identity | M |
| 26 | L17 slides 14–15 | Titled "Seq2Vec" for seq2seq; slide 15 inputs $x_1,x_2,x_2$ | Should be seq2seq, inputs $x_1,x_2,x_3$ | T |
| 27 | L17 slide 20 | Scores divided by a constant "C" | Standard choice $C=\sqrt d$ (Murphy 15.43) | E |
| 28 | L17 slide 22 | Mask wording/figure | Decoder LMs use a causal mask (hide the future); check which direction the figure zeroes | M |
| 29 | L18 slide 7 | Function stays "underneath (or above)" a line | For convex functions the chord lies above the curve | T |
| 30 | L19 midterm example | Loss $\sum\frac{r^2}{2\exp(\theta_2^\top\phi)}$, and $\sigma^2$ called "std. dev." | Keep $\frac12\theta_2^\top\phi$ (from $\log\sigma$); $\sigma^2$ is the variance | E |

### G.3 Things that are *not* errors but easy to misread
* Murphy writes the RSS with a factor $\frac12$ and no $1/D$; the slides use $\frac1D$. Same minimizer.
* Murphy's printed page numbers differ from the PDF page by 30; Bishop's printed numbers differ from the PDF index by 19.
* Slides state $p(y|x)$ with $\sigma=1$; Murphy keeps $\sigma^2$ free. The weights do not depend on it, only $\hat\sigma^2=\text{MSE}$ does.
* There is **no Lecture 01** in the uploaded materials (numbering starts at 02).
* Lecture 15 slide numbers (printed in the corner) differ from PDF page numbers by up to 5 (the PDF has 34 pages, the slides run to 39); the notes use the printed slide numbers.
* Lecture 14's $p=0.5$ and $p=4$ contour plots are not used for learning: for $p<1$ the penalty is not convex (not a norm), and $p=1$ is the smallest $p$ that keeps the problem convex.
* Lecture 14's "$\lambda$" depends on how the loss is scaled (MSE vs RSS vs $\frac12$RSS): the shapes of the paths are the same but the numeric $\lambda$ values differ by factors of $N$ and 2. The notes state which form each number uses.

---

