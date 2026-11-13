# CMPT 423 Machine Learning — Easy Edition

**Course:** CMPT 423 Machine Learning · **Instructor:** Lucas Lehnert · **Term:** Fall 2026 · **University of Saskatchewan** · **Language:** Python
**Textbooks:** Murphy, *Probabilistic Machine Learning: An Introduction* (2022) and Bishop, *Pattern Recognition and Machine Learning* (2006).
**Covers:** Lectures 02 to 19 (there is no Lecture 01). New lectures are added as they are posted.

---

## How to use this edition

This is the **same course material as the Complete Study Notes, written for a first-year student.** Every topic, formula and number is kept. What changes is how it is explained.

* **One-line idea first.** Each lecture starts with the point of the lecture in one sentence.
* **Plain words, then the formula.** Every symbol is read out in words. Formulas are built up in small steps.
* **Real numbers.** Every idea is worked through on the same ten students (below), so the numbers match the Complete Study Notes exactly.
* **🧪 Check yourself.** Four quick questions with answers.
* **📖 Go deeper (optional), then the full Textbook deep dive and Textbook extra.** The plain-words intuition comes first. After it comes the **complete** textbook material, exactly as in the Complete Study Notes: Murphy and Bishop sections with page numbers, extra examples and the solved exercises. You can hide all of these with the switch at the top of the website.
* **✅ What you must be able to do.** A checklist to review before a quiz or exam.

Use this edition to **understand**, then use the Complete Study Notes for the full detail, the formula sheet, the code cheat sheet and exam preparation.

### The one running example

Ten students each reported **hours studied** $h$ and then wrote a test. We record the **score** $s$ (out of 100) and whether they **passed** ($t=1$ if the score is 65 or more, otherwise 0).

| Student | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| hours $h$ | 4 | 2 | 6 | 1 | 3 | 5 | 7 | 3 | 8 | 2 |
| score $s$ | 66 | 48 | 71 | 45 | 57 | 62 | 77 | 59 | 83 | 56 |
| passed $t$ | 1 | 0 | 1 | 0 | 0 | 0 | 1 | 0 | 1 | 0 |
| class $y_3$ | 1 | 0 | 1 | 0 | 0 | 0 | 2 | 0 | 2 | 0 |

The 3-class label $y_3$ is 0 for a score under 65 (fail), 1 for 65–74 (pass) and 2 for 75 or more (distinction). Four of the ten passed.

---

## Part A. Maths you need (in plain words)

You do not need to be a maths expert. These are the only ideas the later lectures lean on.

* **A vector** is a list of numbers, like $\mathbf x=[1,\ 4]$. **A matrix** is a table of numbers. **Transpose** ($\mathbf X^\top$) flips a table so rows become columns.
* **A dot product** multiplies two lists entry by entry and adds the results: $[1,4]\cdot[3,2]=1\cdot3+4\cdot2=11$. We write it $\mathbf w^\top\mathbf x$. Almost every model starts with "multiply each input by a weight and add".
* **The sum sign** $\sum_{i=1}^{D}$ means "add up the term for $i=1,2,\dots,D$". The **average** of $D$ numbers is $\frac1D\sum$.
* **A hat** ($\hat y$) means "our prediction". No hat means the real value.
* **Exponent and log.** $e^{x}$ is a number that grows very fast and is always positive. $\log$ is its undo button: $\log(e^x)=x$. Logs turn products into sums: $\log(ab)=\log a+\log b$. That is why we use them on likelihoods, which are long products.
* **A slope (derivative)** tells you how fast something changes. At the bottom of a bowl the slope is zero. The **gradient** collects the slopes for several settings at once and points **uphill**.
* **Probability** is a number from 0 to 1. $p(A\mid B)$ is read "the probability of $A$ given $B$".
* **Parameters** (the Greek letter $\boldsymbol\theta$, "theta") are the knobs of a model. **Training** means turning the knobs to make a **loss** (a score of how wrong the model is) as small as possible.

### The recipe that runs through the whole course

1. **Assume** how the data were produced (a bell curve, a coin flip, …).
2. **Likelihood:** how probable is the real data under this assumption?
3. **Log:** turns the product into a sum.
4. **Optimize:** find the knob settings that make the data most probable.

The loss functions of the course all come from this recipe: **squared error** (bell-curve assumption), **cross-entropy** (coin or dice assumption). In week 6 we add a **prior** (a belief about the knobs before seeing data), which gives **regularization**.

---


## Lecture 02 — Tensor Arithmetic

**The one-line idea.** Instead of looping over numbers one at a time, tell NumPy to do the same thing to **a whole box of numbers at once**. It is shorter, less error-prone, and much faster.

**Why it matters.** Machine learning does millions of small calculations. A Python `for` loop does them one by one and is slow. Assignment 1 literally asks for code with **no `for` loops and no `if/else`**.

### 2.1 What is a tensor?

A **tensor** is a box of numbers arranged along some number of directions:

* one number = a **scalar** (0 directions),
* a list = a **vector** (1 direction),
* a table = a **matrix** (2 directions),
* a stack of tables = a **3-D tensor** (3 directions).

The number of directions is the **rank** (NumPy calls it `ndim`). The **shape** says how long each direction is.

| rank | name | example shape | in our running example |
|---|---|---|---|
| 0 | scalar | `()` | the mean hours, 4.1 |
| 1 | vector | `(10,)` | all ten `hours` |
| 2 | matrix | `(10, 2)` | one row per student: [1, hours] |
| 3 | 3-D tensor | `(6700, 28, 28)` | a stack of MNIST images (Assignment 1) |

**Habit #1: print `.shape` all the time.** Most bugs in this course are shape bugs.

`h = np.array([4,2,6,1,3,5,7,3,8,2.])` has shape `(10,)`. The matrix `X = np.stack([np.ones(10), h], axis=1)` has shape `(10, 2)`: ten rows (students) and two columns (a constant 1 and the hours).

### 2.2 `dtype`: what kind of number?

Every array holds one kind of number: `float64` (decimals), `int64` (whole numbers) or `bool` (True/False). Pick it on purpose. A wrong type causes **silent** bugs such as whole-number division or a "mask" made of decimals.

```python
np.array([1, 2, 3]).dtype            # int64
np.array([1, 2, 3.0]).dtype          # float64  (one decimal makes everything decimal)
(h >= 5).dtype                       # bool
(h >= 5).astype(np.int32).dtype      # int32    (explicit conversion)
```

### 2.3 Element-wise operations

`+ - * / **` between arrays act on **every element separately** and give an array of the same shape. `@` is different (section 2.9).

* `h * 2` gives `[8, 4, 12, 2, 6, 10, 14, 6, 16, 4]`.
* `h ** 2` gives `[16, 4, 36, 1, 9, 25, 49, 9, 64, 4]`.

### 2.4 Masks: comparisons that replace `if`

A comparison like `h >= 5` also works element by element and gives True/False for each student. That list of True/False is a **mask**. It replaces a whole `for … if …` loop.

```python
m = h >= 5
# [False False  True False False  True  True False  True False]
m.sum()          # 4            (True counts as 1, so this counts students)
h[m]             # [6. 5. 7. 8.] the hours of those students
h[m].mean()      # 6.5
```

The **sum of a mask is a count.** Assignment 1 uses the same idea to pick all images assigned to cluster $j$: `x[assign == j]`.

### 2.5 Reduce operations and `axis`

A **reduce** squashes an array along one direction into fewer numbers: `sum`, `mean`, `max`, `min`, `argmax`, `argmin`. The `axis` argument says **which direction disappears**.

> **Memory trick:** `axis=k` means "dimension $k$ disappears from the shape."

For `X` of shape `(10, 2)`:

| call | result shape | meaning |
|---|---|---|
| `X.mean(axis=0)` | `(2,)` | average of each column: `[1.0, 4.1]` |
| `X.sum(axis=1)` | `(10,)` | total of each row: `[5,3,7,2,4,6,8,4,9,3]` |
| `X.mean()` | `()` | average of all 20 numbers: 2.55 |

`argmax` gives the **position** of the biggest value, not the value itself. Counting starts at 0: `s.argmax()` is `8` (student 9, score 83) and `s.argmin()` is `3` (student 4, score 45).

### 2.6 `reshape`

Re-arrange the same numbers into a different shape. The total count must stay the same. A `-1` means "work this one out for me".

```python
h.reshape(2, 5)
# [[4. 2. 6. 1. 3.]
#  [5. 7. 3. 8. 2.]]
h.reshape(-1, 2).shape    # (5, 2)
```

For Assignment 1, `images.reshape(-1, 784)` flattens each 28×28 image into one row of 784 pixels.

### 2.7 Broadcasting: stretching to fit

When two arrays have different shapes, NumPy **pretends to stretch** any direction of size 1 (or a missing direction) to match the other, without copying anything. This lets you subtract a mean from every row, or build a table of all pairs, in one line.

**The rule.** Line the shapes up from the **right**. Two sizes are compatible if they are **equal** or **one of them is 1**. If any pair fails you get an error.

| shape A | shape B | result | meaning |
|---|---|---|---|
| (5, 1) | (1, 7) | (5, 7) ✓ | every combination |
| (10,) | () | (10,) ✓ | `h - h.mean()` |
| (6700, 784) | (784,) | (6700, 784) ✓ | subtract the mean image from every image |
| (6700, 784) | (6700,) | ✗ error | 784 ≠ 6700; reshape to `(6700, 1)` first |
| (10, 1) | (1, 10) | (10, 10) ✓ | all pairs of students |

**Example 1: centering.** `h - h.mean()` with `h.mean() = 4.1` gives `[-0.1, -2.1, 1.9, -3.1, -1.1, 0.9, 2.9, -1.1, 3.9, -2.1]`. These always add to 0.

**Example 2: all pairs.** `D = h[:, None] - h[None, :]` has shape `(10, 10)` and `D[i, j] = h[i] - h[j]`. Row 0 (student 1, 4 hours) is `[0, 2, -2, 3, 1, -1, -3, 1, -4, 2]`. The trick `h[:, None]` inserts a direction of size 1 (the same as `reshape(-1, 1)`).

### 2.8 Slicing

`a[start:stop:step]` picks a regular run of elements. `start` is included, `stop` is **excluded**, counting starts at 0, and negative numbers count from the end.

```python
ar = np.arange(5)     # [0 1 2 3 4]
ar[1:-1]              # [1 2 3]      drop first and last
ar[:-1]               # [0 1 2 3]
ar[::-1]              # [4 3 2 1 0]  reversed
ar[::-2]              # [4 2 0]
```

### 2.9 Matrix multiplication with `@`

`A @ B` multiplies matrices: each entry is a sum of products, $c_{ij}=\sum_\ell a_{i\ell}b_{\ell j}$. That is why one `@` can replace a double loop. The **inner sizes must match**: `(n, m) @ (m, k)` gives `(n, k)`.

With `X` of shape `(10, 2)` and parameters $\theta=[41.8078,\ 5.0225]$ (from Lecture 07), `X @ theta` has shape `(10,)` and gives all ten predicted scores at once.

### 2.10 Views versus copies (the sneaky bug)

A **slice** (and `reshape`) does **not** copy the data. It is a **window onto the same memory**, so changing the window changes the original. Use `.copy()` if you want a separate array.

```python
a = h.copy()
v = a[1:-1]          # a window onto a[1..8]
v += 100
print(a)             # [4. 102. 106. 101. 103. 105. 107. 103. 108. 2.]   <- 'a' changed!
```

### 2.11 Fancy indexing (lists of positions) makes copies

```python
h[[2, 4, 1]]         # [6. 3. 2.]   students 3, 5, 2
```

Indexing with a **list of positions** (or a mask) builds a **copy**, so editing the result leaves the original alone. It is also not very fast, so use it sparingly. Summary: **slice gives a view; list or mask gives a copy.**

### 2.12 Checklist for writing NumPy

1. Know the shape of every array (print it).
2. Replace loops with element-wise operations, masks, reduces, broadcasting and `@`.
3. Set dtypes on purpose.
4. Never change a slice unless you mean to change the original.

#### 🧪 Check yourself

1. `X` has shape `(10, 2)`. What shape does `X.mean(axis=0)` have? *(`(2,)`: direction 0 disappears.)*
2. `m = h >= 5`. What does `m.sum()` give? *(4, the number of students who studied at least 5 hours.)*
3. Can you add shapes `(6700, 784)` and `(6700,)`? *(No. The right-most sizes 784 and 6700 differ. Reshape the second to `(6700, 1)`.)*
4. You change a slice `v = a[1:-1]`. Does `a` change? *(Yes, a slice is a view. Use `.copy()` to avoid it.)*

#### 📖 Go deeper: plain-words intuition (optional)

* **Why is it fast?** A Python loop does a lot of bookkeeping per number. NumPy hands the whole array to compiled code that does the arithmetic in a tight loop.
* **Broadcasting costs nothing extra.** The stretching is virtual: no giant copy is made in memory.
* **Where you will see this again.** The "(prediction − truth) × input" step in Lecture 08 is one matrix product, with no loop over students.

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

#### ✅ What you must be able to do

* Say what a tensor's rank and shape are, and print shapes habitually.
* Replace a loop with an element-wise operation, a mask, a reduce, broadcasting or `@`.
* Predict the shape produced by `axis=k` reduces and by broadcasting.
* Explain views versus copies and avoid the sneaky bug.

---

## Lecture 03 — Probability

**The one-line idea.** Probability is the language for saying **how uncertain** something is. Every model in this course is a **probability distribution with adjustable knobs (parameters)**.

**Why it matters.** Data is noisy and models are never perfect. We cannot predict one outcome, but we *can* say how often each outcome happens over many repeats.

### 3.1 Random events

A **random event** is something you cannot predict in advance: a coin flip, a die roll, whether it rains. The lecture's story is Schrödinger's cat: a Geiger counter triggers a poison hammer at a *random time*. We cannot say *when*, but we can describe how likely each time is.

**Running example.** "Will a randomly chosen student from our ten pass?" We cannot predict one student, but 4 of 10 passed, so the pass frequency is $0.4$.

### 3.2 The event space $\Omega$ (and σ-algebras, briefly)

First list everything that *could* happen. That list is the **event space** $\Omega$ (read "omega"). One outcome is $\omega\in\Omega$ ("omega is an element of Omega").

| Experiment | $\Omega$ |
|---|---|
| Coin flip | {heads, tails} |
| Die roll | {1,2,3,4,5,6} |
| A student's score | any number from 0 to 100 |
| A student's result (running example) | {fail, pass} |

**The formal part, in plain words.** We only ask about a chosen collection $\Sigma$ of subsets of $\Omega$, called a **σ-algebra** ("sigma-algebra"). It must follow three sensible rules:

1. the whole set $\Omega$ is allowed,
2. if you can ask about "A", you can ask about "not A",
3. if you can ask about A, B, C, …, you can ask about "A or B or C or …".

For a die, $\Sigma$ is simply *all* subsets of {1,…,6}. You will **not** be asked to construct one in the assignment.

A **probability measure** $p$ gives each allowed event a number from 0 to 1, with $p(\emptyset)=0$ (impossible), $p(\Omega)=1$ (something happens), and probabilities **add** for events that do not overlap.

### 3.3 Probability rules

* $0\le p\le 1$: 0 means impossible, 1 means certain.
* Something in $\Omega$ happens: $p(\Omega)=1$.
* Events that **cannot happen together** add: $p(A\text{ or }B)=p(A)+p(B)$.

*Worked example.* Fair die, each face $1/6$:
$$p(\text{even})=p(2)+p(4)+p(6)=\tfrac16+\tfrac16+\tfrac16=\tfrac12.$$

*Running example.* With the 3-class labels, $p(\text{class }0)=6/10$, $p(\text{class }1)=2/10$, $p(\text{class }2)=2/10$, and $p(\text{class }1\text{ or }2)=0.2+0.2=0.4$, which equals the pass rate. ✔

### 3.4 Random variables

Computers need numbers, not words. A **random variable** (rv) is a rule that turns each outcome into a number. **Capital** $X$ is the rule; **lowercase** $x$ is one value it takes. "$X=1$" means "the coin came up heads".

*Running example.* $T=1$ if the student passed, 0 if not. $T=1$ for students 1, 3, 7, 9.

### 3.5 Distributions

A **distribution** says how likely each value is. For a discrete rv the list of probabilities is the **pmf** (probability mass function), $p(x)=\Pr(X=x)$. Each $p(x)$ is between 0 and 1 and they **sum to 1**. For a continuous rv the function is a **density** (pdf).

#### 3.5.1 Bernoulli: one yes/no event

Two outcomes (1 or 0) and one knob $\theta$ ("theta") = the probability of a 1.

| $x$ | probability |
|---|---|
| 1 | $\theta$ |
| 0 | $1-\theta$ |

**One-line formula (the lecture's trick).**
$$\text{bernoulli}(x;\theta)=\theta^{x}(1-\theta)^{1-x}.$$
Read it as: "theta to the power $x$, times (1 minus theta) to the power (1 minus $x$)". Check: $x=1$ gives $\theta^1(1-\theta)^0=\theta$; $x=0$ gives $\theta^0(1-\theta)^1=1-\theta$. (Anything to the power 0 is 1.)

*Worked example.* Pass/fail with $\theta=0.4$: $p(T=1)=0.4$, $p(T=0)=0.6$.

#### 3.5.2 Categorical: one of $K$ outcomes

Like Bernoulli but with $K$ outcomes. The **list of $K$ probabilities that sums to 1 is the model**. A fair die is [1/6, …, 1/6]. Our 3-class result has $\boldsymbol\theta=[0.6,\,0.2,\,0.2]$.

#### 3.5.3 Normal (Gaussian): the bell curve

For measurements like height or score. Most values sit near the centre; far-away values are rare. Two knobs: $\mu$ (mean, the centre) and $\sigma$ (standard deviation, the width). $\sigma^2$ is the **variance**. Written $\mathcal N(\mu,\sigma^2)$. The **standard normal** is $\mathcal N(0,1)$.

$$f(x;\mu,\sigma)=\frac{1}{\sqrt{2\pi\sigma^2}}\exp\!\Big(-\frac{(x-\mu)^2}{2\sigma^2}\Big).$$

How to read it: $(x-\mu)^2$ is the squared distance from the centre. $\exp(-\text{something})$ shrinks as "something" grows. So the density falls as $x$ moves away from $\mu$. The front factor only makes the total area equal 1.

*Worked example.* Heights $\mathcal N(170,10^2)$: $f(170)=1/(10\sqrt{2\pi})=0.03989$ and $f(190)=0.03989\cdot e^{-2}=0.005399$ (two standard deviations away, 7.4 times lower). The probability of landing within $\pm1\sigma$ (160 to 180) is $0.6827$. A height above 210 (4σ) has probability $3.2\times10^{-5}$.

**Probability versus density.** For a continuous variable the probability of *exactly* 170.0000 is zero. Probability comes from **area**: $\Pr(X\in[a,b])=\int_a^b f(x)\,dx$. Higher density means a more likely neighbourhood.

### 3.6 Independence and IID

Two events are **independent** if one does not affect the other. For independent events probabilities **multiply**: $p(A\text{ and }B)=p(A)\,p(B)$.

**IID = Independent and Identically Distributed**: every data point (i) is independent of the others and (ii) comes from the same distribution. We assume this for essentially every model in the course because it lets us **multiply** per-sample probabilities (Lecture 04).

*Worked example.* Biased coin $\theta=0.7$: $p(\text{H,H,T})=0.7\cdot0.7\cdot0.3=0.147$.

*Running example.* If each student passes independently with $\theta=0.4$, the probability of exactly the observed sequence $1,0,1,0,0,0,1,0,1,0$ is $0.4^4\cdot0.6^6=0.0011944$. (The probability of *some* sequence with exactly 4 passes is $\binom{10}{4}\cdot0.0011944=0.2508$.)

### 3.7 Models versus reality

The real world is the true random process. We only see **measurements** (cat alive or dead). We *choose* a **probabilistic model** (a distribution or density) to describe them. Two consequences:

* every model rests on **assumptions** and can never be certain to match the true process,
* a model is designed by an expert and then **tested on data**. A **mismatch** between model and reality is always possible. This is why Lecture 10 (generalization) exists.

#### 🧪 Check yourself

1. A coin has $\theta=0.7$ for heads. What is $p(\text{H,H,T})$ if flips are independent? *(0.7 · 0.7 · 0.3 = 0.147.)*
2. What does the Bernoulli formula give for $x=0$? *(θ⁰(1−θ)¹ = 1 − θ.)*
3. Heights are $\mathcal N(170,10^2)$. Is the probability of exactly 170.0 zero or 0.03989? *(Zero. 0.03989 is the **density**; probability is area.)*
4. What does IID let us do? *(Multiply the per-sample probabilities.)*

#### 📖 Go deeper: plain-words intuition (optional)

* **Why a σ-algebra at all?** For simple cases (dice) every subset is fine. For continuous things (any number between 0 and 100) some strange subsets cannot be given a consistent probability, so mathematicians restrict to a well-behaved collection. In practice you can ignore it.
* **Density is not probability.** A density can be bigger than 1 (a very narrow bell curve). What must equal 1 is the **total area**.
* **Why multiply?** "Both happen" with independence is "chance of the first, then chance of the second." That is the same reasoning as 0.7 · 0.7 · 0.3 above.

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

#### ✅ What you must be able to do

* Say what $\Omega$, an event and a random variable are, and the three probability rules.
* Write the Bernoulli, categorical and normal distributions and use them on numbers.
* Explain probability versus density.
* State what IID means and why it lets us multiply.

---

## Lecture 04 — Maximum Likelihood Estimation (MLE)

**The one-line idea.** Given data and a model with knobs, **turn the knobs to the setting that makes the data you actually saw as probable as possible.**

**Why it matters.** This is the single most important idea in the course. Linear regression, logistic regression, softmax regression and the Gaussian/GMM of Assignment 1 are all fitted this way.

### 4.1 The question

A **dataset** $\mathcal D=\{x_1,\dots,x_D\}$ has $D$ points. Our pass/fail results:

$$\mathcal D=\{1,0,1,0,0,0,1,0,1,0\}\qquad(1=\text{passed},\;0=\text{failed}),\ D=10.$$

There are $c_1=4$ passes and $c_0=6$ fails. The model is $\text{Bernoulli}(\theta)$. **Which $\theta$ explains this best?** Your gut says $4/10=0.4$. MLE proves it and gives a *method* that works when your gut cannot help.

### 4.2 The likelihood

Pick a guess for $\theta$ and ask: "if the world really worked like this, how probable would *exactly my dataset* be?" That probability, seen as a function of $\theta$, is the **likelihood**.

Data points are **IID** (Lecture 03), so their probabilities multiply:
$$L(\theta;\mathcal D)=\prod_{i=1}^{D}\theta^{x_i}(1-\theta)^{1-x_i}=\theta^{c_1}(1-\theta)^{c_0}.$$

$\prod$ (capital pi) means "multiply all of these", just as $\sum$ means "add all of these".

For our data $L(\theta)=\theta^4(1-\theta)^6$. Try some values:

| guess $\theta$ | $\theta^4(1-\theta)^6$ |
|---|---|
| 0.1 | 0.0000531 |
| 0.2 | 0.000419 |
| 0.3 | 0.000953 |
| **0.4** | **0.001194** ← biggest |
| 0.5 | 0.000977 |
| 0.6 | 0.000531 |
| 0.7 | 0.000175 |

The winner is the **maximum likelihood estimate**:
$$\theta^*=\arg\max_\theta L(\theta;\mathcal D).$$
Read $\arg\max$ as "the input that gives the biggest output" (not the output itself). The star means "best".

*Important:* the likelihood is a function of **$\theta$** with the data fixed. It is **not** a probability distribution over $\theta$. The column above does not sum to 1.

### 4.3 Why the numbers get tiny

Multiplying many numbers below 1 collapses toward 0. With 100 flips at 40% heads the likelihood at $\theta=0.4$ is $5.9\times10^{-30}$; with 1000 flips it is $5.2\times10^{-293}$. A computer's `float64` bottoms out near $10^{-308}$. Below that it becomes 0 (**underflow**) and the maximization breaks.

### 4.4 The log trick

Take the logarithm. Two facts make this legal and useful:

1. $\log(ab)=\log a+\log b$ and $\log(a^n)=n\log a$. **Products become sums**, which are numerically safe.
2. $\log$ is **strictly increasing** (bigger input gives bigger output), so the $\theta$ that maximizes $L$ also maximizes $\log L$. The peak does not move.

For the pass/fail data:
$$\log L(\theta)=4\log\theta+6\log(1-\theta).$$
At $\theta=0.4$: $4\log0.4+6\log0.6=4(-0.9163)+6(-0.5108)=-6.730$. (Natural log throughout.)

### 4.5 Find the peak: slope equals zero

At the top of a hill the slope is 0. Rules: the slope of $\log\theta$ is $1/\theta$, and the slope of $\log(1-\theta)$ is $-1/(1-\theta)$.
$$\frac{d}{d\theta}\log L=\frac4\theta-\frac6{1-\theta}=0\;\Rightarrow\;4(1-\theta)=6\theta\;\Rightarrow\;\theta^*=\frac{4}{10}=0.4.$$
In general $\theta^*=c_1/(c_1+c_0)=c_1/D$, the fraction of ones.

**Is it a peak, not a valley?** The second derivative $-\frac4{\theta^2}-\frac6{(1-\theta)^2}$ at $\theta=0.4$ is $-25-16.67=-41.67<0$, a downward-curving hill ✔. At $\theta=0.5$ the slope is $4/0.5-6/0.5=-4<0$: already going down, so the peak is to the left of 0.5 ✔.

### 4.6 From likelihood to a loss

Machine learning prefers to *minimize* things. Since $\arg\max f=\arg\min(-f)$:
$$\theta^*=\arg\min_\theta\ \Big(-\frac1D\log L(\theta;\mathcal D)\Big).$$
This is the **negative average log-likelihood (NLL/D)**, a **loss**: lower is better. Dividing by $D$ does not move the minimum; it just makes numbers comparable across dataset sizes (in code: `.mean()` instead of `.sum()`).

| $\theta$ | NLL/D $=-\frac1{10}(4\log\theta+6\log(1-\theta))$ |
|---|---|
| 0.4 | **0.6730** |
| 0.5 | 0.6931 ($=\log 2$) |

The fair-coin guess loses slightly to the best guess, as it should. Note $0.6931$ is also the loss of a model that always says "50/50". Keep it as a reference point for logistic regression (Lecture 08).

### 4.7 More data gives a sharper peak

With 10 samples the likelihood hill is wide (many $\theta$ look plausible). With 1000 it is a narrow spike. The raw values also shrink ($\sim10^{-3}\to10^{-29}\to10^{-289}$), but the **log-likelihood averaged over $D$** shows a clear peak even for 10,000 samples. More data means a more *certain* estimate.

### 4.8 What did we assume?

* The data follow a **Bernoulli** model (two outcomes).
* The points are **IID**.

If either is wrong the answer can be meaningless even if the algebra is perfect. Also, **MLE gives a point estimate**: one value of $\theta$, not a range of plausible values (Lectures 11 and 12 deal with how much to trust it).

### 4.9 Notation details

* $\theta\in[0,1]$ includes the endpoints. $\log$ needs inputs $>0$, so the lecture code adds a tiny `1e-8` inside the log to avoid $\log 0$.
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

*Trace 1.* `theta = np.array([0.4])` and `x = np.array([0,0,1])`: the `if` is skipped; shapes `(1,)` and `(3,)` broadcast to `(3,)`: $[0.6,\,0.6,\,0.4]$.

*Trace 2 (our dataset on a grid of $\theta$ values).*

```python
x = np.array([1,0,1,0,0,0,1,0,1,0])
theta = np.linspace(0, 1, 101)               # shape (101,)
p = bernoulli(theta, x)                      # (101,1) and (1,10) -> (101, 10)
L = p.prod(axis=-1)                          # likelihood per theta, shape (101,)
logL_mean = np.log(p + 1e-8).mean(axis=-1)   # (1/D) log L per theta
theta[L.argmax()]                            # 0.4
theta[(-logL_mean).argmin()]                 # 0.4  (same answer)
```

#### 🧪 Check yourself

1. Data: 7 heads in 10 flips. What is the MLE of $\theta$? *(7/10 = 0.7.)*
2. Why do we use $\log L$ instead of $L$? *(Products become sums, no underflow, and the peak does not move.)*
3. Is the likelihood a probability distribution over $\theta$? *(No. It does not sum to 1; the data is fixed and $\theta$ varies.)*
4. NLL/D at $\theta=0.5$ for our data? *(0.6931, worse than 0.6730 at $\theta=0.4$.)*

#### 📖 Go deeper: plain-words intuition (optional)

* **Think of a dial.** For each setting of the dial $\theta$ we ask "how surprised would this model be by my data?" MLE picks the dial setting that is *least surprised*.
* **Why does "slope = 0" find the best?** A hill's top is flat for an instant. The second-derivative check tells you it is a hill, not a valley.
* **Why does the answer equal the fraction of ones?** Because $\theta^4(1-\theta)^6$ rewards a $\theta$ that matches how often 1 and 0 actually occurred.

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

#### ✅ What you must be able to do

* State the likelihood, log-likelihood and NLL/D for a Bernoulli model.
* Derive and compute $\theta^*=c_1/D$ by setting the slope to zero.
* Explain why we take logs (underflow, sums, same peak).
* State what MLE assumes (model form, IID) and that it gives one point estimate.

---

## Lecture 05 — Git and GitLab

**The one-line idea.** Git is a "save history" for your code. You hand in an assignment by giving the grader **one exact snapshot** (a *commit hash*).

**Why it matters.** Doing this wrong can mean the grader sees the wrong code, or none. This is the only lecture with **no textbook chapter**; the books cover maths, not tooling.

### 5.1 Why use version control?

1. Software development is hard, especially in machine learning.
2. To keep it simple, big projects are built in **many small steps**.
3. Those steps are **tracked** with a version control system such as Git, the standard tool.

**Plain words.** Each save is a **commit** (a snapshot plus a message). You can look back at any snapshot, compare versions, undo mistakes, and try a risky idea on a separate **branch** without breaking what works.

### 5.2 How Assignment 1 is set up

* Every student has a **private GitLab group** on the department server. Your NSID (university username) is in the address:
  `https://git.cs.usask.ca/cmpt-423-f26-private-abc123` (`abc123` stands for your NSID). Only you and the instructor should have access.
* Inside is a **private repository** for the assignment, e.g. `…/cmpt-423-820-f26-a1`.

*A wording inconsistency in the slides.* Slide 3 says each student has a repository; slide 4's title says "One Git Repository per Team". Likely meaning: individual assignments use per-student repositories, team assignments use one per team. Check the assignment handout if unsure.

### 5.3 The two branches

| Branch | Who can write | What it holds |
|---|---|---|
| **`main`** | Instructor: read and write. **Student: read only.** | The starter code. **You cannot merge into `main`.** |
| **`solution`** | Instructor and student | **Your work.** You solve the project here through commits 1, 2, 3, … |

Commands from the slide:

```bash
git branch solution          # create the branch
git checkout solution        # switch to it
# or in one command:
git checkout -b solution
```

You may open a **merge request** on GitLab to track your changes against `main`, but **do not merge it**. Leave your solution in the `solution` branch.

### 5.4 A typical work session

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

What each word means:

* **clone** = copy the online repository to your computer.
* **add** = choose what goes in the next snapshot.
* **commit** = save the snapshot.
* **push** = upload your commits to GitLab.
* `origin` = the nickname for the GitLab copy.

### 5.5 Submitting: the commit hash

You submit the **commit hash** that should be graded. A hash is a long fingerprint of one snapshot, e.g. `81de7dd8034e500d402762a23e347357415ed79d` (the slide's example; real hashes are 40 characters).

1. **Check** on the GitLab website that you are on the **`solution`** branch and that **all files to be submitted appear there**.
2. **Copy the commit hash** of the intended (usually latest) commit and submit it, together with the PDF report on Canvas.

```bash
git log --oneline -3      # shows short hashes of recent commits
git rev-parse HEAD        # prints the full hash of the current commit
```

**Why a hash?** It identifies *exactly* one version. If you keep pushing after submitting, the grader still grades the snapshot you named.

#### 🧪 Check yourself

1. Can you merge your work into `main`? *(No. `main` is read-only for students; work in `solution`.)*
2. What does `git push -u origin solution` do? *(Uploads your `solution` commits to GitLab.)*
3. What do you submit for grading? *(The commit hash, plus the PDF report on Canvas.)*
4. Why check the website before submitting? *(To be sure the `solution` branch really contains every file.)*

#### 📖 Go deeper: plain-words intuition (optional)

* **Think of video-game save slots.** Each commit is a save slot you can return to, with a note on what changed.
* **Branch = a parallel copy of the story.** You experiment on `solution` and `main` stays untouched.
* **Commit before you push.** A push only uploads commits; unsaved edits are not included.

#### 📖 Textbook deep dive

Neither Murphy nor Bishop covers Git. The nearest idea in the course is **reproducibility of experiments**, discussed in Lecture 11 (experiment reporting): to re-create a result you need to know exactly which code produced it. A commit hash is that "which code" label.

#### ➕ Textbook extra — none
(No textbook coverage.)

#### ✅ What you must be able to do

* Clone, create the `solution` branch, stage, commit and push.
* Explain why `main` is never modified and why you do not merge.
* Find and submit the correct commit hash after checking the website.

---

## Lecture 06 — Linear Algebra

**The one-line idea.** Use **lists of numbers (vectors)** and **tables of numbers (matrices)** and a few short formulas instead of loops.

**Why it matters.** Real data has many numbers per example (an MNIST image has 784 pixels). Everything from Lecture 07 on is written with vectors and matrices.

### 6.1 Vectors

A **vector** is an ordered list of numbers. A 2-number vector is an arrow (or a point): "3 to the right, 2 up".

**Notation.** Vectors are **lowercase bold**, matrices **uppercase bold**:
$$\mathbf v=\begin{bmatrix}x_1\\ \vdots\\ x_n\end{bmatrix}\in\mathbb R^n.$$
"$\in\mathbb R^n$" reads "lives in $n$-dimensional real space", meaning a list of $n$ real numbers.

**Worked example.** $\mathbf v=[3,2]^\top\in\mathbb R^2$. Our ten students' hours form $\mathbf h\in\mathbb R^{10}$. A $28\times28$ image flattened to 784 numbers is a vector in $\mathbb R^{784}$: **one image is one point in 784-dimensional space.** You cannot draw 784 dimensions, but the rules are the same as in 2-D.

**Column versus row; transpose.** A column vector is $n\times1$. Its **transpose** $\mathbf v^\top$ is the $1\times n$ row $[3,2]$. Maths texts use columns, NumPy code usually stores each example as a **row**, a classic source of shape bugs.

### 6.2 Matrices are 2-D tensors

A **matrix** has $m$ rows and $n$ columns, written $\mathbf M\in\mathbb R^{m\times n}$. A dataset of $D$ examples with $n$ features is a $D\times n$ matrix with one example per row. Our student design matrix (a column of 1s and the hours) is $\mathbf X\in\mathbb R^{10\times 2}$.

### 6.3 Norms: size and distance

A **norm** $\|\mathbf v\|$ measures the size of a vector. Applied to a *difference* of two vectors it gives the **distance** between two points.

Slide example: $\mathbf p_1=(1,2)$, $\mathbf p_2=(4,4)$, so $\mathbf v=\mathbf p_2-\mathbf p_1=(3,2)$.

| Norm | Formula | Value for $(3,2)$ | Picture |
|---|---|---|---|
| **L1** | $\sum_i\lvert v_i\rvert$ | $3+2=\mathbf 5$ | walking along city blocks |
| **L2** (Euclidean) | $\sqrt{\sum_i v_i^2}$ | $\sqrt{9+4}=\sqrt{13}\approx\mathbf{3.61}$ | straight line |
| **L∞** | $\max_i\lvert v_i\rvert$ | $\max(3,2)=\mathbf 3$ | largest single difference |

**L2 is used most.** Its **square** $\|\mathbf x-\boldsymbol\mu\|^2=\sum_i(x_i-\mu_i)^2$ is the sum of squared differences. MSE (Lecture 07) and k-means (Assignment 1) are built from it.

*Running example.* $\mathbf h=(4,2,6,1,3,5,7,3,8,2)$: $\|\mathbf h\|_1=41$, $\|\mathbf h\|_2=\sqrt{217}=14.731$, $\|\mathbf h\|_\infty=8$. Distance between student 1 (4 h, 66 pts) and student 2 (2 h, 48 pts): difference $(2,18)$, so L1 $=20$ and L2 $=\sqrt{328}=18.11$.

### 6.4 Dot product and matrix multiplication

**Dot product.** Multiply two vectors entry by entry and add: $\mathbf a^\top\mathbf b=\sum_i a_ib_i$. For example $[1,2,3]\cdot[4,5,6]=4+10+18=32$. A vector dotted with itself is its squared length: $\mathbf v^\top\mathbf v=\|\mathbf v\|_2^2$.

**Matrix multiplication.** For $\mathbf A\in\mathbb R^{m\times n}$ and $\mathbf B\in\mathbb R^{n\times k}$, the product $\mathbf C=\mathbf A\mathbf B\in\mathbb R^{m\times k}$ has entries
$$c_{ij}=\sum_{\ell=1}^{n}a_{i\ell}\,b_{\ell j}\qquad(\text{row }i\text{ of }\mathbf A\ \cdot\ \text{column }j\text{ of }\mathbf B).$$
**Shape rule:** $(m\times\mathbf n)\cdot(\mathbf n\times k)\to(m\times k)$. The inner sizes must match; the outer sizes give the result.

*Worked example.*
$$\begin{bmatrix}1&2\\3&4\end{bmatrix}\begin{bmatrix}5\\6\end{bmatrix}=\begin{bmatrix}1\cdot5+2\cdot6\\3\cdot5+4\cdot6\end{bmatrix}=\begin{bmatrix}17\\39\end{bmatrix}.$$
In NumPy: `np.array([[1,2],[3,4]]) @ np.array([5,6])` gives `[17 39]`.

*Running example (the key product of Lecture 07).* With $\mathbf X$ the $10\times2$ design matrix:
$$\mathbf X^\top\mathbf X=\begin{bmatrix}10&41\\41&217\end{bmatrix},\qquad \mathbf X^\top\mathbf y=\begin{bmatrix}624\\2804\end{bmatrix}.$$
Check: top-left $=\sum 1\cdot1=10$; off-diagonal $=\sum h_i=41$; bottom-right $=\sum h_i^2=217$; and $624=\sum s_i$, $2804=\sum h_is_i$.

### 6.5 A matrix moves vectors (change of basis)

Multiplying a vector by a matrix **moves it to a new place**. The slides call $\mathbf M\mathbf v$ a **linear projection into a new basis**.

**Slide example.** $\mathbf M=\begin{bmatrix}1&-1\\1&0\end{bmatrix}$, whose **columns** are $\mathbf x_1'=(1,1)$ and $\mathbf x_2'=(-1,0)$.
$$\mathbf M\begin{bmatrix}1\\0\end{bmatrix}=\begin{bmatrix}1\\1\end{bmatrix},\qquad \mathbf M\begin{bmatrix}0\\1\end{bmatrix}=\begin{bmatrix}-1\\0\end{bmatrix}.$$
**The columns of $\mathbf M$ are where the basic arrows $(1,0)$ and $(0,1)$ land**; every other vector follows. Check with $\mathbf v=(3,2)$: $\mathbf M\mathbf v=3\cdot(1,1)+2\cdot(-1,0)=(1,3)$ ✔.

*Why it matters:* Assignment 1 turns random noise into image-like vectors by multiplying by a matrix (the Cholesky factor, section 6.7). In Lecture 07, the design matrix times $\boldsymbol\theta$ combines the *columns* of $\mathbf X$ with weights from $\boldsymbol\theta$.

### 6.6 Tools for regression: identity, inverse, pseudo-inverse

(Used in Lecture 07; these facts come from the Lecture 07 slides and the textbooks.)

| Name | Definition | Role |
|---|---|---|
| **Identity** $\mathbf I$ | 1s on the diagonal, 0s elsewhere; $\mathbf I\mathbf v=\mathbf v$ | "multiply by 1" |
| **Inverse** $\mathbf M^{-1}$ | $\mathbf M^{-1}\mathbf M=\mathbf I$ | "divide by $\mathbf M$"; only for square, non-degenerate matrices |
| **Symmetric** | $\mathbf M^\top=\mathbf M$ | covariance matrices; $\mathbf X^\top\mathbf X$ |
| **Lower triangular** | zeros above the diagonal | cheap to solve; the Cholesky factor |
| **Outer product** $\mathbf v\mathbf v^\top$ | $(n\times1)(1\times n)=n\times n$ | appears in the covariance formula |

**Singular matrices (no inverse).** A matrix with a zero column, or a column that is a copy or combination of others, "squashes space flat" and loses information, so it cannot be undone. Linear regression's pseudo-inverse $(\mathbf X^\top\mathbf X)^{-1}\mathbf X^\top$ needs the columns of $\mathbf X$ to be **linearly independent**. In our example $\det(\mathbf X^\top\mathbf X)=10\cdot217-41^2=489\ne0$, with eigenvalues $2.175$ and $224.82$ (condition number $103.4$), so it is invertible:
$$(\mathbf X^\top\mathbf X)^{-1}=\frac1{489}\begin{bmatrix}217&-41\\-41&10\end{bmatrix}.$$

### 6.7 Cholesky factor: making correlated noise (Assignment 1)

Any symmetric positive-definite matrix $\boldsymbol\Sigma$ can be written $\boldsymbol\Sigma=\mathbf L\mathbf L^\top$ with $\mathbf L$ lower triangular. To sample $\mathbf y\sim\mathcal N(\boldsymbol\mu,\boldsymbol\Sigma)$: draw $\mathbf z\sim\mathcal N(\mathbf 0,\mathbf I)$ (independent 1-D normals) and set $\mathbf y=\mathbf L\mathbf z+\boldsymbol\mu$.

*Worked example.* $\boldsymbol\Sigma=\begin{bmatrix}4&2\\2&3\end{bmatrix}$ has $\mathbf L=\begin{bmatrix}2&0\\1&1.4142\end{bmatrix}$, and indeed $\mathbf L\mathbf L^\top=\begin{bmatrix}4&2\\2&3\end{bmatrix}$ ✔. With $\mathbf z=(1,1)$ and $\boldsymbol\mu=(10,20)$: $\mathbf y=(2+0+10,\ 1+1.4142+20)=(12,\ 22.414)$.

#### 🧪 Check yourself

1. What is the L2 length of $(3,2)$? *($\sqrt{13}\approx3.61$.)*
2. $\mathbf A$ is $5\times3$ and $\mathbf B$ is $3\times4$. Shape of $\mathbf A\mathbf B$? *($5\times4$; inner 3 matches.)*
3. $[1,2,3]\cdot[4,5,6]$? *(4 + 10 + 18 = 32.)*
4. Where do the columns of $\mathbf M$ come from? *(They are where $(1,0)$ and $(0,1)$ land.)*

#### 📖 Go deeper: plain-words intuition (optional)

* **Dot product as agreement.** It is big when two vectors point the same way and 0 when they are perpendicular.
* **Why $\mathbf X^\top\mathbf X$ shows up.** It collects all "feature times feature" sums, which is what least squares needs.
* **Why Cholesky works.** $\mathbf L$ stretches and tilts a round cloud of independent noise into a tilted oval with the covariance you want.

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

#### ✅ What you must be able to do

* Compute L1, L2 and L∞ norms and distances.
* Multiply matrices by hand and predict result shapes.
* Explain a matrix as a map that moves the basic arrows, and describe identity, inverse and symmetric.
* Use the Cholesky factor to turn independent noise into correlated samples.

---

## Lecture 07 — Linear Regression

**The one-line idea.** Linear regression is *fitting the best straight line through dots* so you can **predict a number** — here, a test score from hours studied.

**Where it fits.** In Lecture 04 you learned the recipe: *assume a distribution → write the likelihood → take the log → find the best settings.* This lecture runs that recipe on the first real model. The result is the loss called **mean squared error**, and a formula that gives the answer in one step.

### 7.1 The problem in plain words

You have pairs of numbers: an **input** $x$ (hours studied) and an **output** $y$ (test score). You want a rule that turns a new input into a good guess for the output.

Our ten students again: hours $h$ = 4, 2, 6, 1, 3, 5, 7, 3, 8, 2 and scores $s$ = 66, 48, 71, 45, 57, 62, 77, 59, 83, 56.

### 7.2 Two ways to look at the same line

**Way 1: draw the line.** A straight line is $\hat y=\theta_0+\theta_1x$.

* $\theta_0$ (theta-zero) is the **intercept**: the guess when $x=0$.
* $\theta_1$ (theta-one) is the **slope**: how much the guess goes up for each extra unit of $x$.
* $\hat y$ ("y-hat") is the **prediction**. The hat always means "our guess", not the real value.

**Way 2: a bell curve sitting on the line.** Two students who study the same hours do not get exactly the same score. So we say: *for a given $x$, the score is random, and its average is on the line.* The scores spread out like a bell curve (a **normal distribution**) around the line.

Both ways give the **same line**. Way 2 matters because it lets us use the likelihood recipe, and it explains *why* we measure error the way we do.

### 7.3 The recipe, step by step

1. **Assume** each score $y_i$ comes from a bell curve whose centre is $\theta_0+\theta_1x_i$ (and whose spread is fixed at 1, a simplification).
2. **Likelihood:** multiply the bell-curve heights for all students. A good line makes the actual scores sit near the centres, so the product is big.
3. **Log:** the log turns the product into a sum, and it cancels the "exp" inside the bell curve. Each student then contributes a term like $-\tfrac12(\text{error})^2$.
4. **Drop constants:** numbers that do not depend on $\theta$ (like the $\tfrac12$) cannot change *which* line is best, so we throw them away. Adding a constant to a curve moves it up or down but does not move its peak.

What is left: **make the squared errors as small as possible.** Averaged, this is the loss

$$\text{MSE}(\boldsymbol\theta)=\frac1D\sum_{i=1}^{D}\big(y_i-\hat y_i\big)^2,\qquad \hat y_i=\theta_0+\theta_1x_i .$$

**Reading the formula.** $D$ is the number of students. $y_i-\hat y_i$ is the **error** (also called the residual): real score minus guess. We **square** it so that "5 too high" and "5 too low" do not cancel each other out, and so that big mistakes count a lot more. Then we average. **MSE** stands for *mean squared error*.

> **The big insight.** Using squared error is the same as *assuming the scores spread like a bell curve around the prediction*. If your data do not behave like that (for example, a number that can only be 1–6 like a die), squared error is the wrong tool.

### 7.4 Solving it in one step

For logistic regression (Lecture 08) we will have to walk downhill step by step. Here there is a direct formula. The idea: the best line is where the slope of the loss is zero, like the bottom of a bowl.

**Step 1: put the data in a table.** Give each student a row $[1,\ x_i]$. The extra 1 is there so the intercept $\theta_0$ gets multiplied by something. Stack the rows into a matrix $\mathbf X$ (10 rows, 2 columns). Put the scores in a column $\mathbf y$. Then all ten predictions at once are $\mathbf X\boldsymbol\theta$.

**Step 2: the formula.** Setting the slope to zero gives the **normal equations** $\mathbf X^\top\mathbf X\boldsymbol\theta=\mathbf X^\top\mathbf y$, and so

$$\boldsymbol\theta=(\mathbf X^\top\mathbf X)^{-1}\mathbf X^\top\mathbf y .$$

**Reading it:** $\mathbf X^\top$ is $\mathbf X$ flipped on its side (rows become columns). $(\cdot)^{-1}$ is the matrix "undo" button, the way $\tfrac15$ undoes multiplying by 5. The combination $(\mathbf X^\top\mathbf X)^{-1}\mathbf X^\top$ is called the **pseudo-inverse** of $\mathbf X$, and it only works when the columns of $\mathbf X$ are not copies of each other. In Python it is one line:

```python
theta = np.linalg.pinv(X) @ y
```

**Our numbers.**

$$\mathbf X^\top\mathbf X=\begin{bmatrix}10&41\\41&217\end{bmatrix},\qquad \mathbf X^\top\mathbf y=\begin{bmatrix}624\\2804\end{bmatrix}.$$

Inverting the 2×2 matrix (its determinant is $10\cdot217-41^2=489$) and multiplying gives

$$\boldsymbol\theta=\begin{bmatrix}41.8078\\5.0225\end{bmatrix}.$$

**What it means:** a student who studies 0 hours is predicted to score about 41.8, and **every extra hour adds about 5.02 points.**

*A second way to get the same slope (for one input):* slope $=S_{xy}/S_{xx}=245.6/48.9=5.0225$, where $S_{xx}$ adds up the squared distances of the hours from their mean and $S_{xy}$ adds up (hours distance) × (score distance). Intercept $=\bar s-\text{slope}\cdot\bar h=62.4-5.0225\cdot4.1=41.8078$. Same answer, which is a good check.

| student | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| hours | 4 | 2 | 6 | 1 | 3 | 5 | 7 | 3 | 8 | 2 |
| score | 66 | 48 | 71 | 45 | 57 | 62 | 77 | 59 | 83 | 56 |
| prediction | 61.90 | 51.85 | 71.94 | 46.83 | 56.88 | 66.92 | 76.97 | 56.88 | 81.99 | 51.85 |
| error | 4.10 | −3.85 | −0.94 | −1.83 | 0.12 | −4.92 | 0.03 | 2.12 | 1.01 | 4.15 |

* Squared errors add up to **82.875**, so **MSE = 8.2875**. Taking the square root gives **2.879 points**, the "typical miss".
* The errors add up to exactly **0**. That is what "the best line" means.
* A new student who studies 5.5 hours: $41.8078+5.0225\cdot5.5=69.43$.

**How good is the line? $R^2$.** First measure how much the scores vary in total, ignoring hours: $\text{TSS}=\sum(s-\bar s)^2=1316.4$. Then

$$R^2=1-\frac{\text{leftover error}}{\text{total variation}}=1-\frac{82.875}{1316.4}=0.937 .$$

So hours studied explains about **93.7%** of the ups and downs in the scores. $R^2=0$ would mean the line is no better than always guessing the average score.

### 7.5 Curves with a "linear" model

What if the dots curve? We can still use the same machinery by **making new input columns** from $x$. For example $[1,\ x,\ x^2,\ x^3]$. These new columns are called **basis functions**. The prediction is

$$\hat y=\theta_0+\theta_1x+\theta_2x^2+\theta_3x^3 .$$

It is still called *linear* regression because the **unknown numbers $\theta$** only appear in a straight-line way (multiplied and added). The curve is in $x$, not in $\theta$. The same pseudo-inverse formula works on the bigger table.

| curve | training MSE |
|---|---|
| straight line (degree 1) | 8.288 |
| degree 2 | 8.272 |
| degree 3 | 6.711 |

A bendier curve always fits the *training* students better. That does **not** mean it will predict *new* students better. Lecture 10 shows it often does worse.

**Another basis: the staircase.** Split hours into three groups (low ≤ 3, middle 4–6, high ≥ 7). Give each student a "one-hot" row such as $[0,1,0]$ meaning "middle group". The best $\boldsymbol\theta$ is simply the **average score of each group**: $[53,\ 66.33,\ 80]$. Its MSE is 20.87, worse than the line because the real pattern climbs smoothly instead of in steps.

```python
h = np.array([4,2,6,1,3,5,7,3,8,2.]);  s = np.array([66,48,71,45,57,62,77,59,83,56.])
X = np.stack([np.ones_like(h), h], axis=1)
theta = np.linalg.pinv(X) @ s                         # [41.8078, 5.0225]
resid = s - X @ theta
mse = (resid**2).mean()                               # 8.2875
r2 = 1 - (resid**2).sum() / ((s - s.mean())**2).sum() # 0.9370
```

#### 🧪 Check yourself

1. In the line $\hat y=41.8+5.02x$, what does 5.02 mean in words? *(Each extra hour adds about 5 points.)*
2. Why do we square the errors instead of just adding them? *(Positive and negative errors would cancel; squaring also punishes big misses.)*
3. A student studies 3 hours. What does the line predict? *($41.8078+5.0225\cdot3=56.88$.)*
4. If MSE is 8.29, roughly how far off is a typical prediction? *(About 2.9 points, the square root.)*

#### 📖 Go deeper: plain-words intuition (optional)

* **Why the formula works (geometry).** Picture the ten scores as one arrow in 10-dimensional space. The predictions are the closest point to that arrow among all points a line could produce. "Closest" means the leftover arrow (the errors) is at a right angle to the inputs. That right-angle condition *is* the normal equations. This is also why the errors add to 0 and are uncorrelated with hours.
* **Why not just "invert"?** Computers prefer methods called QR or SVD (that is what `pinv` uses) because inverting $\mathbf X^\top\mathbf X$ directly can lose accuracy when inputs are nearly copies of each other.
* **Textbook:** Murphy §11.2 (p.371–381) covers all of this; Bishop §3.1 (p.138–146) covers basis functions. The full notes have the detailed version.
* **Preview of week 6:** adding a penalty $\lambda\|\boldsymbol\theta\|^2$ ("ridge") shrinks the slope. With $\lambda=10$ our slope drops from 5.02 to 4.17. You will meet this properly in Lecture 14.

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

#### ✅ What you must be able to do

* Say why squared error goes with the bell-curve assumption.
* Build the table $\mathbf X$ (with a column of 1s) and use $\boldsymbol\theta=\text{pinv}(\mathbf X)\,\mathbf y$.
* Read slope and intercept in words, and compute a prediction.
* Compute MSE and $R^2$ and say what they mean.
* Explain why a "linear" model can still draw a curve (new columns from $x$).

---

## Lecture 08 — Logistic Regression

**The one-line idea.** When the answer is **yes or no** (pass or fail, spam or not spam), predict a **probability between 0 and 1** instead of a number, and then decide.

**Where it fits.** Lecture 07 predicted a number. A straight line can output −3 or 47, which makes no sense as a probability. We need a squashing step. The recipe is the same as before, but there is **no one-step formula**, so we meet our first **walk-downhill method** (gradient descent).

### 8.1 The problem

Each example has an input $\mathbf x$ and a label $y$ that is either 0 or 1. In our running example the input is $[1,\ h]$ (a 1 and the hours studied) and $y=1$ means the student passed.

### 8.2 The sigmoid: a squashing machine

$$\sigma(z)=\frac{1}{1+e^{-z}}$$

It takes **any** number $z$ and squeezes it into the range 0 to 1, in an S shape:

| $z$ | −10 | −1 | 0 | 1 | 10 |
|---|---|---|---|---|---|
| $\sigma(z)$ | ≈0 | 0.269 | 0.5 | 0.731 | ≈1 |

Big negative scores mean "almost surely no", zero means "50–50", and big positive scores mean "almost surely yes". (This $\sigma$ is a function; it has nothing to do with the standard deviation, which also uses the letter σ.)

### 8.3 The model

$$p(y=1\mid\mathbf x)=\sigma(\boldsymbol\theta^\top\mathbf x).$$

Three steps: (1) compute a **score** $\boldsymbol\theta^\top\mathbf x=\theta_0+\theta_1h$ (multiply and add, like in Lecture 07); (2) squash the score into a probability; (3) imagine flipping a coin that lands "pass" with that probability. It is the same coin as in Lecture 04, **except the coin's bias now depends on the input.**

### 8.4 The recipe

1. **Assume** each label is a coin flip with probability $\sigma(\boldsymbol\theta^\top\mathbf x_i)$ of being 1.
2. **Likelihood:** multiply the probabilities the model gave to what *actually* happened.
3. **Log:** products become sums.
4. **Optimize:** walk downhill (next sections).

**One formula for both cases.** If $y_i=1$ the probability the model gave to the truth is $p_i=\sigma(\boldsymbol\theta^\top\mathbf x_i)$; if $y_i=0$ it is $1-p_i$. The trick to avoid an "if":

$$\text{prob. of truth}=y_i\,p_i+(1-y_i)(1-p_i).$$

When $y_i=1$ the second part becomes 0; when $y_i=0$ the first part becomes 0. Taking logs gives $y_i\log p_i+(1-y_i)\log(1-p_i)$.

### 8.5 The loss: binary cross-entropy

Flip the sign (so "good" becomes "small") and average:

$$J(\boldsymbol\theta)=-\frac1D\sum_{i=1}^{D}\Big[y_i\log p_i+(1-y_i)\log(1-p_i)\Big].$$

**Plain meaning:** for each student, the loss is **how surprised the model was by the truth.** If the truth is "pass", the loss is $-\log(\text{probability given to pass})$.

| truth is 1; model said $p=$ | 0.9 | 0.7 | 0.5 | 0.1 | 0.01 |
|---|---|---|---|---|---|
| loss | 0.105 | 0.357 | 0.693 | 2.303 | 4.605 |

Being **confidently wrong** is punished hardest. Being right and confident costs almost nothing.

**A baseline to beat.** With $\boldsymbol\theta=\mathbf 0$ every student gets $p=0.5$ and the loss is $\log2=0.6931$. This is the "I have no idea" score. Any useful model must do better.

### 8.6 Gradient descent: walking downhill in fog

Imagine the loss as a hilly landscape and your settings $\boldsymbol\theta$ as where you stand. You cannot see the valley, but you can feel which way is **uphill** at your feet. That direction is the **gradient** $\nabla J$ (one slope for each setting). So:

$$\boldsymbol\theta\ \leftarrow\ \boldsymbol\theta-\alpha\,\nabla J(\boldsymbol\theta)\qquad(\text{repeat many times}).$$

* The minus sign means "step **downhill**".
* $\alpha$ is the **learning rate**, the size of each step. Too big and you overshoot the valley; too small and it takes forever.

**The gradient has a nice form:**

$$\nabla J=\frac1D\sum_{i=1}^D\big(p_i-y_i\big)\,\mathbf x_i .$$

Read it as **(prediction − truth) × input, averaged.** If the model is already right ($p_i\approx y_i$) that student pushes the settings almost nowhere. If it is wrong, it pushes the settings in the direction that fixes the mistake.

In Python there is no loop needed:

```python
p    = 1 / (1 + np.exp(-X @ theta))   # predicted probabilities
grad = X.T @ (p - y) / D              # the gradient
theta = theta - alpha * grad          # one step downhill
```

#### A traced run on the ten students

Start at $\boldsymbol\theta=[0,0]$ with $\alpha=0.1$. Every student gets $p=0.5$.

* Sum of (prediction − truth) over students: $5-4=1$. Sum of (prediction − truth) × hours: $0.5\cdot41-(4+6+7+8)=-4.5$.
* Gradient $=[1,\,-4.5]/10=[0.1,\,-0.45]$.
* New settings: $[0,0]-0.1\cdot[0.1,-0.45]=[-0.01,\ 0.045]$. The loss drops from 0.6931 to 0.6769.

| step | $\boldsymbol\theta$ | loss |
|---|---|---|
| 0 | [0, 0] | 0.6931 |
| 1 | [−0.0100, 0.0450] | 0.6769 |
| 2 | [−0.0243, 0.0668] | 0.6711 |
| 3 | [−0.0405, 0.0785] | 0.6673 |
| 4 | [−0.0574, 0.0857] | 0.6640 |
| 5 | [−0.0747, 0.0909] | 0.6607 |
| … after 100,000 steps | [−6.6836, 1.4495] | **0.26438** |

Progress is slow at the start, but the weight on hours becomes positive, which means "more hours, more likely to pass".

**The final model** is $\sigma(-6.6836+1.4495\,h)$. The **decision boundary** is where the probability is exactly 0.5, i.e. where the score is 0: $h=6.6836/1.4495=4.611$ hours.

| student | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| hours | 4 | 2 | 6 | 1 | 3 | 5 | 7 | 3 | 8 | 2 |
| passed | 1 | 0 | 1 | 0 | 0 | 0 | 1 | 0 | 1 | 0 |
| model's $p$ | 0.292 | 0.022 | 0.882 | 0.005 | 0.088 | 0.637 | 0.970 | 0.088 | 0.993 | 0.022 |

The two surprises are students 1 (studied 4 h, passed, model said only 29%) and 6 (5 h, failed, model said 64%). They account for most of the total loss.

### 8.7 From probability to a decision

To say yes or no, pick a **threshold** $\tau$ ("tau"): say "pass" if the probability is at least $\tau$. The usual choice is 0.5, but it is a choice.

**Count the four kinds of outcome (the confusion matrix):**

| | said yes | said no |
|---|---|---|
| **really yes** | TP (true positive) | FN (false negative) |
| **really no** | FP (false positive) | TN (true negative) |

* **Accuracy** $=\dfrac{TP+TN}{\text{all}}$: how often you are right. Misleading when one class is rare.
* **Precision** $=\dfrac{TP}{TP+FP}$: *when I say yes, how often am I right?*
* **Recall** $=\dfrac{TP}{TP+FN}$: *of all the real yeses, how many did I catch?*
* **False-positive rate** $=\dfrac{FP}{FP+TN}$: *of all real noes, how many did I wrongly flag?*

For our model (4 real passes, 6 real fails):

| $\tau$ | TP | FP | FN | TN | accuracy | precision | recall |
|---|---|---|---|---|---|---|---|
| 0.2 | 4 | 1 | 0 | 5 | 0.9 | 0.80 | 1.00 |
| 0.5 | 3 | 1 | 1 | 5 | 0.8 | 0.75 | 0.75 |
| 0.7 | 3 | 0 | 1 | 6 | 0.9 | 1.00 | 0.75 |
| 0.9 | 2 | 0 | 2 | 6 | 0.8 | 1.00 | 0.50 |

**Lower the threshold:** you say yes more, so you catch more real yeses (recall up) but make more false alarms (precision down). Raising it does the opposite. Which is right depends on what costs more: missing a disease, or a false alarm?

**Curves.** Sweep $\tau$ from 0 to 1 and plot recall against the false-positive rate (the **ROC curve**) or precision against recall (the **PR curve**). A perfect model hugs the top-left corner of the ROC plot; a coin flip lies on the diagonal. The **AUC** (area under the ROC curve) is the chance that a random real pass is ranked above a random real fail. For us it is $23/24=0.958$.

### 8.8 Judging the model honestly

Scores on the same data you trained on are **optimistic**. Honest scoring uses a **test set** the model never saw: predict its labels, then compute the loss, accuracy, precision and recall. Doing this over several train/test splits is **cross-validation** (Lecture 10). Our ten-student numbers above are all training numbers.

#### 🧪 Check yourself

1. What does $\sigma(0)$ equal, and what does it mean? *(0.5, a 50–50 guess.)*
2. The model says $p=0.01$ but the student passed. Is the loss small or large? *(Large, 4.605.)*
3. In "(prediction − truth) × input", what happens to a student the model already gets right? *(Their term is near zero, so they barely move the settings.)*
4. You want to catch as many real cases as possible and false alarms are cheap. Raise or lower $\tau$? *(Lower it; recall goes up.)*

#### 📖 Go deeper: plain-words intuition (optional)

* **Why the gradient is so tidy.** The sigmoid's slope is $\sigma(1-\sigma)$. When you apply the chain rule to the log of the likelihood, those factors cancel and only (truth − prediction) × input is left.
* **Why 100,000 steps?** The loss landscape here is a long narrow valley. Steps big enough to move along it quickly overshoot across it. Rescaling the hours column helps a lot.
* **Textbook:** Murphy §10.2 (p.339 onward) and §5.1 (p.169–176) for thresholds, ROC and PR curves; Bishop §4.3.2 (p.205).
* **A warning for later:** if the classes can be split perfectly by a line, the best weights run off to infinity. Regularization (Lecture 14) fixes this.

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

#### ✅ What you must be able to do

* Say what the sigmoid does and compute it for a few values.
* Explain the model as "score → squash → coin flip".
* Explain binary cross-entropy as "surprise at the truth".
* Do one gradient-descent step by hand using (prediction − truth) × input.
* Compute accuracy, precision and recall from a confusion matrix, and say how $\tau$ changes them.

---

## Lecture 09 — Softmax Regression

**The one-line idea.** When there are **more than two answers** (which digit is it, 0 to 9?), give every answer a score, then turn the scores into probabilities that add up to 1.

**Where it fits.** Logistic regression (Lecture 08) handles yes/no. Softmax regression handles $C$ choices. It also brings in **cross-entropy**, the loss used by almost every classifier, including large neural networks.

### 9.1 The problem

Each example has an input $\mathbf x$ (for example a picture of a handwritten digit) and a label $y\in\{0,1,\dots,C-1\}$. For digits $C=10$.

**Running example.** We split the scores into three classes: **0 = fail** (score under 65), **1 = pass** (65–74), **2 = distinction** (75 or more). The ten labels are $[1,0,1,0,0,0,2,0,2,0]$, so the class counts are $[6,2,2]$. The input is $[1,\ h]$ as before.

### 9.2 One score per class

A model reads the input and produces **one score per class**: $\mathbf z=[z_0,z_1,\dots,z_{C-1}]$. Scores can be any numbers (−3, 12, …). But the likelihood needs **probabilities**: no negatives, and all adding to 1. For a simple model the scores are $\mathbf z=\mathbf W\mathbf x$ with one row of weights per class. (A neural network is just a more powerful way to compute $\mathbf z$.)

### 9.3 What counts as a valid set of probabilities?

A list of numbers that are each $\ge0$ and sum to exactly 1. This set has a name, the **probability simplex**. $(0.5,\ 0.3,\ 0.2)$ is valid. $(0.5,\ 0.9,\ -0.4)$ is not (it has a negative number).

### 9.4 Softmax: scores to probabilities

$$\text{softmax}(\mathbf z)_i=\frac{e^{z_i}}{e^{z_0}+e^{z_1}+\dots+e^{z_{C-1}}}.$$

Two steps: (1) $e^{z}$ makes every score **positive** and exaggerates the leader; (2) dividing by the total makes everything **sum to 1**.

```python
p = np.exp(z)
p = p / p.sum()
```

**Example with scores $(2,\ 1,\ 0)$:**

| class | $e^{\text{score}}$ | ÷ 11.107 |
|---|---|---|
| 0 | 7.389 | **0.6652** |
| 1 | 2.718 | **0.2447** |
| 2 | 1.000 | **0.0900** |

They add to 1 and the highest score gets the highest probability. **Only the differences between scores matter:** adding 100 to every score changes nothing, and equal scores give equal probabilities: $(5,5,5)\to(\tfrac13,\tfrac13,\tfrac13)$.

> **Link to Lecture 08.** With two classes, softmax gives $\dfrac{e^a}{e^a+e^b}=\sigma(a-b)$. For $a=1.3$, $b=-0.4$ both give 0.8455. So **logistic regression is softmax with two classes.**

### 9.5 The likelihood of one label

We want the probability the model gave to the **correct** class. Write $\mathbb 1[c=y_i]$ for "1 if class $c$ is the correct one, else 0". Then

$$P(y_i)=\prod_{c}\big(p_{ic}\big)^{\mathbb 1[c=y_i]} .$$

Every wrong class is raised to the power 0, which turns it into 1 and makes it disappear. Only the correct class's probability is left. After taking logs:

$$\log P=\sum_i\sum_c\mathbb 1[c=y_i]\ \log p_{ic}.$$

### 9.6 Cross-entropy

For a truth vector $\mathbf p$ and a predicted vector $\mathbf q$, the **cross-entropy** is $H(\mathbf p,\mathbf q)=-\sum_i p_i\log q_i$. When the truth is "all weight on the correct class", it is simply $-\log(\text{probability given to the correct class})$. The **loss** is its average over the data:

$$J=-\frac1D\sum_{i=1}^{D}\log p_{i,\,y_i}.$$

**In words:** how surprised was the model by the correct answer? It is the same idea as Lecture 08, and binary cross-entropy is the two-class case.

**Example.** Scores $[0.5,\ 1.5,\ -1.0]$ give probabilities $[0.2537,\ 0.6897,\ 0.0566]$. If the true class is 1, the loss is $-\log0.6897=0.3715$. If the true class is 0, the loss is $-\log0.2537=1.3715$.

**Baselines to beat** (our 3 classes, counts $[6,2,2]$):

* Always say $(\tfrac13,\tfrac13,\tfrac13)$: loss $=\log3=1.0986$.
* Always say the class frequencies $(0.6,0.2,0.2)$: loss $=0.9503$.

### 9.7 Finding the weights

Minimize $J$ with gradient descent, exactly as in Lecture 08. The gradient for the weights of class $c$ is

$$\nabla_{\mathbf w_c}J=\frac1D\sum_i\big(p_{ic}-\mathbb 1[c=y_i]\big)\mathbf x_i ,$$

the same **(prediction − truth) × input**, once for every class. To predict, pick the class with the biggest probability.

**First step on the students** (all weights 0 so every probability is $\tfrac13$; $\alpha=0.1$). The gradient is

$$\mathbf G=\begin{bmatrix}-0.2667&0.1333&0.1333\\-0.2333&0.3667&-0.1333\end{bmatrix}$$

(rows: intercept, hours; columns: class 0, 1, 2). The update $-0.1\,\mathbf G$ gives new weights $\begin{bmatrix}0.0267&-0.0133&-0.0133\\0.0233&-0.0367&0.0133\end{bmatrix}$ and the loss falls from 1.0986 to 1.0759. Reading it: more hours push class 2 (distinction) up and class 0 (fail) down, which matches the data.

**A warning.** Both distinction students (7 and 8 hours) are separated from everyone else by hours alone. When classes can be split perfectly, the unregularized best weights keep growing forever while the loss keeps shrinking. Adding a tiny penalty on large weights (regularization, Lecture 14) fixes this. With one we get:

| student | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| hours | 4 | 2 | 6 | 1 | 3 | 5 | 7 | 3 | 8 | 2 |
| true class | 1 | 0 | 1 | 0 | 0 | 0 | 2 | 0 | 2 | 0 |
| predicted | 0 ✗ | 0 | 1 | 0 | 0 | 1 ✗ | 2 | 0 | 2 | 0 |

8 of 10 are right; the errors are the same students (1 and 6) that logistic regression got wrong.

### 9.8 A numerical safety trick

$e^{1000}$ is too big for a computer, so `np.exp(1000)` becomes `inf` and the answer turns into `nan`. Since subtracting the same number from every score changes nothing, **subtract the biggest score first**:

```python
def softmax(x):
    z = x - x.max()
    e = np.exp(z)
    return e / e.sum()
softmax(np.array([1000., 1001., 1000.]))   # [0.2119, 0.5761, 0.2119]
```

#### 🧪 Check yourself

1. Why do we use $e^{z}$ and not just divide each score by the total? *(Scores can be negative; $e^{z}$ is always positive.)*
2. Scores $(5,5,5)$. What are the probabilities? *(One third each.)*
3. The model gives the correct class probability 0.5. What is the loss? *($-\log0.5=0.693$.)*
4. Which two lectures' losses become the same thing when $C=2$? *(Cross-entropy and binary cross-entropy.)*

#### 📖 Go deeper: plain-words intuition (optional)

* **Why the gradient has the same shape as logistic regression.** The softmax and the log in the loss cancel in the same tidy way the sigmoid and log did in Lecture 08, leaving (prediction − truth).
* **Textbook:** Murphy §2.5 (p.53–56) and §10.3 (p.350); Bishop §4.3.4 (p.209).

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

#### ✅ What you must be able to do

* Compute softmax for a small score vector by hand.
* Explain why only score differences matter.
* Compute cross-entropy loss for one example and explain it as surprise.
* Write the gradient as (prediction − truth) × input.
* Explain the subtract-the-max trick.

---

## Lecture 10 — Generalization

**The one-line idea.** Doing well on data you have **already seen** proves little. What matters is doing well on **new** data. That ability is **generalization**.

**Why it matters.** Think of studying from past exams: scoring 100% on questions you memorized proves nothing about the real exam.

### 10.1 The bias–variance trade-off

Fit curves of different flexibility to the same points:

| | Too simple | Too complex |
|---|---|---|
| Flexibility | low | high |
| Behaviour | **high bias, low variance** (underfits) | **low bias, high variance** (overfits) |

* **Bias:** error from wrong or too-simple assumptions (a straight line forced through curved data).
* **Variance:** how much the fitted model would *change* if a different training set were drawn.
* The **error residual** is $e=f_{\boldsymbol\theta}(\mathbf x)-y$ (prediction minus truth). A *pattern* in the residuals, instead of random scatter, means the model is missing structure.

The gap between training performance and test performance is the **generalization gap**.

**Worked example on the students** (polynomial of degree $d$ in hours). Train on the first 7 students, test on the last 3 (hours 3, 8, 2; scores 59, 83, 56):

| degree $d$ | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| train MSE (7 students) | 7.30 | 7.10 | 6.71 | 4.58 | 2.49 |
| **test MSE** (3 students) | **14.89** | 16.07 | 10.62 | 177.7 | 392.1 |

Training error falls steadily as the model grows, but test error explodes for degrees 4 and 5. That is **overfitting**. (With 7 training points a degree-5 polynomial has 6 parameters, so it nearly memorizes the training set.) Only 3 test points is a noisy measurement, which is why we use cross-validation (section 10.5).

#### 10.1.1 Making bias and variance concrete (a simulation)

Suppose the truth is $f(h)=40+5h$ and each score is $f(h)$ plus Gaussian noise with $\sigma=3$. Make 20,000 fresh sets of ten scores at the same ten hours, fit a degree-$d$ polynomial each time, and look at the prediction at $h=4.5$ (true value $62.5$):

| degree | bias² | variance | bias² + variance |
|---|---|---|---|
| 0 (constant) | 4.03 | 0.88 | 4.91 |
| 1 (line) | ≈ 0 | 0.91 | **0.91** |
| 2 | ≈ 0 | 2.26 | 2.26 |
| 5 | ≈ 0 | 4.08 | 4.08 |

Check against theory. For a constant fit the variance is $\sigma^2/n=9/10=0.90$ (simulated 0.88) and the bias² is $(60.5-62.5)^2=4.0$ (simulated 4.03). For a line the variance is $9(0.1+0.0033)=0.93$ (simulated 0.91). **The line matches the truth's shape, so it has no bias and the lowest variance. Extra flexibility adds only variance.** Expected squared error at a point $=$ bias² $+$ variance $+$ noise ($\sigma^2=9$ here).

### 10.2 Model selection

1. How do we choose the best model?
2. Many algorithms have settings (the learning rate, the choice of basis functions, the polynomial degree, …). These are **hyperparameters**.
3. Usually many combinations are tried in parallel. This is **grid search**.

### 10.3 The MLE is a point estimate: does it carry over?

1. A dataset $\mathcal D$ is drawn from an overall data distribution $p_{\text{data}}$.
2. We compute the MLE $\boldsymbol\theta_{\mathcal D}$ from it.
3. **Inference:** we use $\boldsymbol\theta_{\mathcal D}$ to predict on a test dataset $\mathcal D_{\text{test}}$.

Is $\boldsymbol\theta_{\mathcal D}$ also the best parameter for the test data? Not necessarily. The MLE is best for the data it *saw*. So we evaluate on an **independently drawn test dataset**.

**Same-process requirement.** Training and test data must come from the same process (all IID from $p_{\text{data}}$). If the test data come from a *different* distribution, what we learned may not help. That is **out-of-distribution (OOD) generalization**. *Example:* a pass/fail model fitted on one course's students and applied to another course with a different exam.

### 10.4 The quantity we would like to know

After training we would like the average test loss over *all test datasets that could exist*:
$$\mathbb E_{\mathcal D_{\text{test}}\sim p_{\text{data}}}\Big[\frac1{|\mathcal D_{\text{test}}|}\sum_{(\mathbf x_i,y_i)\in\mathcal D_{\text{test}}}l\big(f_{\boldsymbol\theta}(\mathbf x_i),y_i\big)\Big].$$
There are infinitely many such datasets, so we cannot compute it. We **approximate by testing on several datasets and averaging.**

### 10.5 Cross-validation (CV)

**Idea.** Split one dataset into $K$ pieces (the lecture uses 5). Run $K$ experiments: in experiment $k$, train on the other pieces and test on piece $k$, giving error $e_k$. **Report the average** $\bar e=\frac1K\sum_ke_k$. Compare models or hyperparameters by $\bar e$ and pick the best. **Finally retrain the best model on the entire dataset** and use that for inference.

**Worked example on the students.** Five folds of two students each; per-fold test MSE for the line (degree 1): $17.24,\ 3.88,\ 15.76,\ 2.98,\ 13.28$, so the mean is $\bar e=10.63$.

| degree | 5-fold CV MSE | leave-one-out (LOO) CV MSE | train MSE |
|---|---|---|---|
| 1 | **10.63** | **11.76** | 8.29 |
| 2 | 15.27 | 16.14 | 8.27 |
| 3 | 12.86 | 14.59 | 6.71 |
| 4 | — | 48.56 | 6.70 |
| 5 | — | 81.67 | 5.81 |

**Decision:** the straight line is the best model (lowest CV error under both schemes), even though degree 5 has the best *training* error. Having chosen degree 1, **retrain it on all ten students**: $\boldsymbol\theta=[41.81,\,5.02]$. The honest expected error on a new student is about $11$ to $12$ (MSE), an RMSE of $\approx3.3$ to $3.4$ points, noticeably more than the optimistic training $8.29$.

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
2. With maximum likelihood, **the loss on the test set is a good start**.
3. A training error far below the test error means **overfitting**.
4. In cross-validation the values are simply **averaged**.

#### 10.6.1 Regression (MSE)
Inspect the **error residuals** and how they are spread. They should look like the Gaussian noise the model assumed (Lecture 07). *Students:* the residuals $[4.10,-3.85,-0.94,-1.83,0.12,-4.92,0.03,2.12,1.01,4.15]$ are small and balanced around 0, consistent with the assumption.

#### 10.6.2 Yes/no classification (logistic regression)
Outputs are 0/1. The slide says "Accuracy is the same as precision in this case", which is imprecise (see Part B): accuracy is $(TP+TN)/\text{all}$ and precision is $TP/(TP+FP)$. Other measures:

* $F_1=2\,\dfrac{\text{Precision}\times\text{Recall}}{\text{Precision}+\text{Recall}}$.
* $F_\beta=(1+\beta^2)\dfrac{\text{Precision}\times\text{Recall}}{\beta^2\,\text{Precision}+\text{Recall}}$ (the slide prints "$(1-\beta)$", a typo). $\beta>1$ weights recall more; $\beta<1$ weights precision more.
* **Fallout** (false-positive rate) $=\dfrac{FP}{FP+TN}$ (the slide leaves out the $FP$ in the denominator).
* **Sensitivity** $=\dfrac{TP}{TP+FN}$ (the same as recall).

**ROC and precision–recall curves.** The model outputs $p(y=1|\mathbf x,\boldsymbol\theta)$. Sliding the threshold $\tau$ makes you predict $y=1$ more or less often, which controls the false-positive and true-positive rates. Plot TPR against FPR (ROC) or precision against recall (PR) over all $\tau$. (Student numbers: Lecture 08, section 8.7.)

#### 10.6.3 Many classes (softmax regression)
* **Accuracy** $=\dfrac{\#\text{ correct predictions}}{\text{test set size}}$.
* **Confusion matrix** (rows = actual class, columns = predicted class). The **normalized** version divides each row by the number of examples in that class, so each row sums to 1 (slide example: row 0 = [0.9, 0.08, 0.02], row 1 = [0.1, 0.8, 0.1], row 2 = [0.05, 0.1, 0.85]). The **count** version shows raw counts (the lecture shows one for CIFAR-10 images). It shows *which* classes get confused.

*Students (3 classes, regularized fit from Lecture 09):* true class 0 (6 students): 5 predicted 0 and 1 predicted 1, giving row $[0.833,0.167,0]$. True class 1 (students 1, 3): 1 predicted 0 and 1 predicted 1, giving $[0.5,0.5,0]$. True class 2 (students 7, 9): both predicted 2, giving $[0,0,1]$. Overall accuracy $8/10$.

### 10.7 Bias and fairness

A model learns the quirks of the data it was trained on. The slides give four real examples:

1. Amazon's hiring algorithm decided not to invite women to interview.
2. Google's ad algorithm showed high-income jobs to men much more often than to women.
3. A machine-learning algorithm denied credit based on race or gender.
4. A health-care algorithm offered less care to Black patients.

A practical first step: report performance **per group**, not only on average. A model can have high average accuracy while failing a small group badly (e.g. overall accuracy $0.9$ when a group making up 10% of the data is 100% wrong).

### 10.8 Inductive bias

Every algorithm makes built-in assumptions: its **inductive bias**.

* **Occam's razor:** use the simplest explanation of the data.
* Inductive biases can be good: our world is not random (the sun will rise tomorrow).
* Ask: *what are the inductive biases of the algorithm I use? Are they clearly understood?*

*Examples:* linear regression assumes the mean is a *line* in the features and the noise is Gaussian. Logistic regression assumes a *linear* decision boundary. IID sampling assumes train and test come from the same process.

### 10.9 No Free Lunch (Wolpert and Macready, 1997)

Binary classification with inputs $x_1,x_2,x_3\in\{0,1\}$ and a label $y$:

| $x_1$ | $x_2$ | $x_3$ | $y$ |
|---|---|---|---|
| 0 | 1 | 0 | 0 |
| 0 | 1 | 1 | 1 |
| 1 | 0 | 0 | 0 |
| 1 | 0 | 1 | 0 |

Many logic functions fit this data. There are $2^3=8$ possible inputs, so $2^8=256$ Boolean functions. The four rows pin down four outputs, leaving four inputs free, so **$2^4=16$ functions fit the data perfectly**. For example $f_1=x_2\wedge x_3$ and $f_2=\neg x_1\wedge x_3$ both fit, yet they disagree on the unseen input $(0,0,1)$: $f_1=0$, $f_2=1$. Which one a learner picks depends on its inductive bias. Every algorithm may do well on one problem and poorly on another:

> **There is no single ML/AI algorithm that performs well on all datasets.**

A method wins only when its assumptions match the problem.

### 10.10 Lecture summary

* Complex models can have very different training and test performance: the **generalization gap**.
* The **bias–variance trade-off**.
* **Overfitting** (good on training, bad on test) and **underfitting** (not even good on training).
* Performance is estimated with a **validation set** or **cross-validation**.

#### 🧪 Check yourself

1. Training MSE keeps falling as degree rises, but test MSE explodes. What is this called? *(Overfitting.)*
2. In 5-fold CV, what do you report? *(The average of the 5 test errors.)*
3. After choosing the best model by CV, what is the last step? *(Retrain it on the entire dataset.)*
4. A model has 0.9 overall accuracy. Is it fair? *(Not necessarily. Check per-group performance.)*

#### 📖 Go deeper: plain-words intuition (optional)

* **Bias = stubborn, variance = jumpy.** A stiff model is always a bit wrong in the same way. A very flexible model changes wildly with each new sample.
* **Why the training error lies.** The model was tuned on those exact points, so it is graded on a test it has already seen.
* **Why 16 functions fit.** The data fix only 4 of 8 inputs; the other 4 are free, so there are $2^4$ ways to fill them in. Only your assumptions choose among them.

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

#### ✅ What you must be able to do

* Explain the bias–variance trade-off and recognise overfitting from train vs test error.
* Run K-fold cross-validation, pick a model, and retrain on all data.
* Compute accuracy, precision, recall, F1 and read a confusion matrix.
* Explain inductive bias and the No Free Lunch theorem.

---

## Lecture 11 — Experiment Reporting

**The one-line idea.** Every ML run has randomness, so one number is not evidence. **Run it several times and report the mean with its standard error.**

**Why it matters.** Model A scores higher than model B. Is A *really* better, or did you get lucky this run? Honest reporting answers that.

### 11.1 The question

> *Given methods $A$ and $B$, how can we tell whether one outperforms the other in a statistically significant way?*

Each run of an experiment is like **drawing a random number $X_i$** from a distribution **we do not know**. We only see the samples $x_1,\dots,x_n$.

### 11.2 The Central Limit Theorem (CLT)

Let $X_1,X_2,\dots$ be **IID** random variables with mean $\mu$ and variance $\sigma^2$ (any shape). Let the sample average be $\bar X_n=\frac1n\sum_{i=1}^nX_i$. Then for large $n$
$$\sqrt n\,(\bar X_n-\mu)\ \longrightarrow\ \mathcal N(0,\sigma^2).$$

**In words: an average of IID random numbers is approximately bell-shaped, whatever the shape of the individual numbers** (as long as the mean and variance are finite). The lecture shows a lumpy discrete distribution on {0,…,5} whose averages still form a smooth bell curve.

*Running example.* Each $X_i$ is one pass/fail result with pass probability $0.4$, about as un-bell-shaped as it gets (two spikes). Averages of $n$ of them have standard deviation $\sqrt{0.4\cdot0.6/n}$:

| $n$ | sd of $\bar X_n$ | 68% of averages fall in |
|---|---|---|
| 10 | 0.155 | $0.4\pm0.155$ |
| 100 | 0.049 | $0.4\pm0.049$ |
| 1000 | 0.0155 | $0.4\pm0.0155$ |

*(Slide typo: the first CLT slide defines $\bar X_n=\sum_iX_i$ without the factor $1/n$. The later slides are correct. See Part B.)*

### 11.3 The CLT in ML experiments

Repeat the experiment and record $X_1,\dots,X_n$. The average $\bar X$ is approximately normal, with spread
$$\sigma=\frac{\hat\sigma_X}{\sqrt n},\qquad \hat\sigma_X=\sqrt{\frac1{n-1}\sum_{i=1}^n(X_i-\bar X)^2}.$$
Here $\sigma$ is the spread **of the average**, not of single runs.

A normal variable lands within $\pm1\sigma$ of its true mean $68.27\%$ of the time (the bell picture: 34.1% on each side).

### 11.4 The standard error of the mean (SEM)

So the true mean $\mu$ lies in
$$\Big[\ \bar X-\frac{\hat\sigma_X}{\sqrt n},\ \ \bar X+\frac{\hat\sigma_X}{\sqrt n}\ \Big]\quad\text{with probability }68.2\%.$$
The quantity $\dfrac{\hat\sigma_X}{\sqrt n}$ is the **standard error of the mean (SEM)**. In Python: `scipy.stats.sem`.

**The $\sqrt n$ rule.** To halve the uncertainty you need **four times** as many runs: with $\hat\sigma=0.05$, 25 runs give $0.05/5=0.010$ and 100 runs give $0.05/10=0.005$.

*Running example, SEM of the students' scores.* $n=10$, $\bar x=62.4$, $\hat\sigma=12.094$ (using $n-1$), so $\text{SEM}=12.094/\sqrt{10}=3.824$ and the 68% interval is $[58.58,\ 66.22]$. (`scipy.stats.sem(s)` returns 3.8245; with `ddof=0` it would be 3.628.)

### 11.5 Plotting and stating results

Plot $\bar X_A$ and $\bar X_B$ with error bars $\bar X\pm\text{sem}$. **If there is a gap between the error bars, the difference is statistically significant** (a rule of thumb; see the caution below). You can also write "model A has performance $\bar X_A\pm\text{err}$". Always say in the caption whether the bars are SEM or standard deviation.

**Worked example (5 runs each).**

| Model | accuracies | mean | $\hat\sigma$ | SEM | $\bar X\pm$ SEM |
|---|---|---|---|---|---|
| A | .91 .93 .92 .90 .94 | 0.92 | 0.01581 | 0.00707 | [0.9129, 0.9271] |
| B | .86 .88 .87 .85 .89 | 0.87 | 0.01581 | 0.00707 | [0.8629, 0.8771] |

The intervals do not touch (B's top is $0.8771$, A's bottom is $0.9129$, a gap of $0.0358$), so A is credibly better. A formal test agrees: Welch's t-test gives $t=5.00$, $p=0.0011$.

```python
import numpy as np
from scipy import stats
A = np.array([.91, .93, .92, .90, .94]); B = np.array([.86, .88, .87, .85, .89])
A.mean(), stats.sem(A)            # 0.92, 0.007071
B.mean(), stats.sem(B)            # 0.87, 0.007071
stats.ttest_ind(A, B, equal_var=False)   # t = 5.0, p = 0.00105
```

**A caution about the "gap" rule.** With equal SEMs, two ±1-SEM bars that *just touch* mean a difference of 2 SEM. The standard error of the *difference* is $\sqrt2\,\text{SEM}$, so $z=2/\sqrt2=1.41$ and $p\approx0.157$, not 0.05. To reach 5% you need a difference of about $1.96\sqrt2=2.77$ SEM. So a gap between ±1-SEM bars is a *rule of thumb*; for borderline cases use a proper test or ±2 SEM. With only $n=5$ runs the 68.2% is not exact either: $\Pr(|T_4|<1)=62.6\%$ for the $t$-distribution (67.3% at $n=25$).

### 11.6 Seeds and reproducibility

* You can often set a **seed** to make random numbers repeat.
* But exact repeats are only guaranteed with the **same code, libraries, OS**, and often the same **hardware** (e.g. GPUs).
* Advanced features (distributed training) and small library upgrades often break determinism.
* So many ML experiments are **not exactly reproducible**, but general statements like *"Model A tends to significantly outperform model B"* **should be**.

Seeds are still great for **debugging**. For science, **repeat, report errors, and base conclusions on hypothesis testing** (the SEM argument above). When repeating is too expensive (e.g. training LLMs), document the compute limits and argue from patterns.

```python
rng = np.random.default_rng(423)      # seed it for debugging...
x = rng.normal(size=3)                # ...same numbers on re-run (on the same setup)
```

### 11.7 Hyperparameters and "deflating a baseline"

* To claim "A beats B", **try different hyperparameter settings for both** models.
* Then compare the **best A** with the **best B**.
* **Repeat every setting** several times.
* Not tuning the baseline properly is called **"deflating a baseline"**: it makes your method look better than it is.

Choose hyperparameters on validation data or by cross-validation (Lecture 10), **never on the final test data**.

### 11.8 Reporting checklist (for assignments)

* Repeat experiments; report mean ± SEM and **say which error you plot**.
* Tune both methods equally; repeat every combination.
* Do not treat a seed as proof of reproducibility.
* Document compute limits when you cannot repeat. Use the provided LaTeX template unmodified for reports (syllabus).

#### 🧪 Check yourself

1. Scores have $\hat\sigma=12.094$ over $n=10$. SEM? *($12.094/\sqrt{10}=3.824$.)*
2. To make the SEM half as big, how many times more runs? *(Four times.)*
3. Two ±1-SEM bars just touch. Is that already $p<0.05$? *(No. It is about $p\approx0.16$.)*
4. What is "deflating a baseline"? *(Tuning your own method but not the competitor.)*

#### 📖 Go deeper: plain-words intuition (optional)

* **Why does averaging calm things down?** Lucky and unlucky runs cancel out, so averages wiggle less than single runs. That shrinking is the $1/\sqrt n$.
* **SEM is not the spread of runs.** Standard deviation says how much single runs vary. SEM says how well pinned-down the *average* is.
* **Why the CLT matters.** We never need to know the shape of the run-to-run distribution to draw a bell-shaped interval for the mean.

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

#### ✅ What you must be able to do

* State the CLT in words and use it to justify the SEM interval.
* Compute mean, standard deviation (n−1) and SEM, and plot or state results with error bars.
* Apply the gap rule of thumb and know its limits.
* Describe good reporting: repeats, tuning both methods, seeds only for debugging.

---

## Lecture 12 — Statistics

**The one-line idea.** This lecture ties together the first third of the course: **how surprised are we on average (entropy)**, **how much does one thing tell us about another (mutual information)**, and **where the standard error formula comes from**.

**About the slides.** The lecture is one page of handwritten notes: formulas at the top, then Schrödinger's cat, the CLT, the derivation of the SEM, the $n$ versus $n-1$ discussion, and reporting with error bars. Items below follow the slide order.

### 12.1 The formula box

1. **Entropy:** the slide writes $H(X)=\mathbb E[\log X]$. The correct definition is $H(X)=-\mathbb E[\log p(X)]=-\sum_xp(x)\log p(x)$ (missing minus sign and $p$; see Part B).
2. **Mutual information:** $I(X,Y)=H(X)+H(Y)-H(X,Y)$.
3. **Conditional entropy:** $H(Y|X)=-\sum_{x,y}p(x,y)\log\dfrac{p(x,y)}{p(x)}$.
4. **Sample standard deviation:** $\sigma=\sqrt{\dfrac1{N-1}\sum_{i=1}^N(x_i-\bar x)^2}$.

#### 12.1.1 Entropy: "how surprised are we, on average?"

Entropy measures the **uncertainty** of a random variable. A coin that always lands heads has 0 uncertainty. A fair coin has the most a two-outcome variable can have. With $\log_2$ the unit is the **bit**: the average number of yes/no questions needed to learn the outcome.

$$H(X)=-\sum_{k}p(k)\log_2p(k)\qquad\text{(with }0\log0:=0).$$
$-\log_2p(k)$ is the **surprise** of outcome $k$ (rare outcome, big surprise). $H$ is the *average* surprise.

*Running example.* The pass rate is $\theta=0.4$:
$$H(0.4)=-(0.4\log_20.4+0.6\log_20.6)=0.4(1.3219)+0.6(0.7370)=0.5288+0.4422=\mathbf{0.971}\text{ bits}.$$
$H(0.5)=1$ bit (the maximum), $H(0.9)=0.469$ bits (a 90/10 event is quite predictable), and the 3-class label with probabilities $(0.6,0.2,0.2)$ has $H=1.371$ bits (the maximum for 3 classes is $\log_23=1.585$ bits).

#### 12.1.2 Joint and conditional entropy

* **Joint entropy** $H(X,Y)=-\sum_{x,y}p(x,y)\log p(x,y)$ is the uncertainty of the pair.
* **Conditional entropy** $H(Y|X)$ is *the uncertainty left in $Y$ after you learn $X$*. Equivalently $H(Y|X)=H(X,Y)-H(X)$.

*Running example.* Let $S$ = studied low ($\le4$ h) or high ($\ge5$ h) and $T$ = failed or passed, with joint distribution

| | fail | pass | row total |
|---|---|---|---|
| low | 0.5 | 0.1 | 0.6 |
| high | 0.1 | 0.3 | 0.4 |
| total | 0.6 | 0.4 | 1 |

$H(S)=H(0.4)=0.971$, $H(T)=0.971$, and $H(S,T)=-\big(0.5\log_20.5+2\cdot0.1\log_20.1+0.3\log_20.3\big)=0.5+0.6644+0.5211=1.6855$ bits. Then
$$H(T|S)=H(S,T)-H(S)=1.6855-0.971=\mathbf{0.7145}\text{ bits}.$$
Direct check: given *low* hours $P(\text{pass})=\tfrac16$, uncertainty $H(\tfrac16)=0.650$; given *high* hours $P(\text{pass})=\tfrac34$, $H(\tfrac34)=0.811$. Weighted: $0.6(0.650)+0.4(0.811)=0.390+0.325=0.7145$ ✔.

#### 12.1.3 Mutual information: "how much does knowing $X$ tell me about $Y$?"
$$I(X,Y)=H(X)+H(Y)-H(X,Y)=H(Y)-H(Y|X)=H(X)-H(X|Y)\ \ge0,$$
and $I=0$ exactly when $X$ and $Y$ are independent.

*Running example:* $I(S,T)=0.971+0.971-1.6855=\mathbf{0.2564}$ bits. Learning whether a student studied a lot removes 0.256 bits of the 0.971 bits of uncertainty about passing (26%). Same answer as $H(T)-H(T|S)=0.971-0.7145=0.2564$ ✔.

### 12.2 Schrödinger's cat again: model versus reality

* **Random variables** ("cat alive", "cat poisoned") are the **measurements we collect**.
* Underneath is the **random process** on $\Omega$, which we cannot see.
* There can be a **mismatch between the designed model and the true process**.
* The slide's sentence: *"Any model is designed by an expert and validated (tested) by collecting data."*

*Takeaway:* every number we compute (a mean, a SEM, an interval) describes *the model's world*. Test data, cross-validation and repeated experiments tell us whether that world resembles the real one.

### 12.3 The CLT (from the notes)

For IID $X_1,\dots,X_n$ with mean $\mu$ and variance $\sigma^2$ (any shape), let $\bar X_n=\frac1n\sum_{i=1}^nX_i$. Then
$$\sqrt n\,(\bar X_n-\mu)\ \overset{\text{app.}}{\sim}\ \mathcal N(0,\sigma^2)\qquad(\text{A}).$$
**The question.** For two models $A$ and $B$ we sample statistics $e_A$ and $e_B$. How can we tell, from $n$ samples, that the true mean lies within a certain range?

### 12.4 Deriving the interval

**Step 1: a standard normal fact.** For $S\sim\mathcal N(0,\sigma^2)$:
$$\Pr(-\sigma\le S\le\sigma)=0.682\qquad(\text{B}).$$

**Step 2: put (A) into (B).** With $S=\sqrt n(\bar X_n-\mu)$:
$$\Pr\big(-\sigma\le\sqrt n(\bar X_n-\mu)\le\sigma\big)=0.682.$$
Divide through by $\sqrt n$:
$$\Pr\Big(-\frac{\sigma}{\sqrt n}\le\bar X_n-\mu\le\frac{\sigma}{\sqrt n}\Big)=0.682\qquad(\text{C}).$$
$\bar X_n$ comes from the data and $\mu$ is the true mean we want to measure.

**Step 3: we do not know $\sigma$, so estimate it.**
$$\hat\sigma_n=\sqrt{\frac1{n-1}\sum_{i=1}^n(x_i-\bar x)^2},\qquad\bar x=\frac1n\sum_{i=1}^nx_i\qquad(\text{D}).$$

**Step 4: put (D) into (C):**
$$\Pr\Big(-\frac{\hat\sigma_n}{\sqrt n}\le\bar X_n-\mu\le\frac{\hat\sigma_n}{\sqrt n}\Big)=0.682,$$
and $\dfrac{\hat\sigma_n}{\sqrt n}$ **is defined as the standard error of the mean (SEM)**.

*Worked example (students' scores).* $n=10$, $\bar x=62.4$, $\hat\sigma=12.094$: $\text{SEM}=12.094/\sqrt{10}=3.824$, so $62.4-3.824\le\mu\le62.4+3.824$, i.e. $\mu\in[58.58,\,66.22]$ with 68.2% confidence.

### 12.5 Why $1/(n-1)$ and not $1/n$?

* The **variance** is $\sigma^2=\mathbb E[(X-\mu)^2]$.
* The natural estimate swaps the expectation for an average and $\mu$ for the sample average:
$$\hat\sigma^2=\frac1n\sum_{i=1}^n(x_i-\bar x)^2.$$
* The slide says **$\hat\sigma=\sqrt{\frac1{n-1}\sum(x_i-\bar x)^2}$ has lower bias**: $\mathbb E[|\hat\sigma-\sigma|]$ is smaller with the $n-1$ term.

**Numerical check (simulation, true $\sigma=1$, 200,000 repetitions each):**

| $n$ | $\mathbb E[\hat\sigma^2]$, $1/n$ | $\mathbb E[\hat\sigma^2]$, $1/(n-1)$ | $\mathbb E|\hat\sigma-1|$, $1/n$ | $\mathbb E|\hat\sigma-1|$, $1/(n-1)$ |
|---|---|---|---|---|
| 2 | 0.503 | **1.006** | 0.5353 | 0.5350 |
| 5 | 0.799 | **0.998** | 0.2844 | 0.2811 |
| 10 | 0.899 | **0.999** | 0.1899 | 0.1885 |

* The $1/n$ **variance** is biased: $\mathbb E[\hat\sigma^2_{1/n}]=\frac{n-1}{n}\sigma^2$ (0.5, 0.8, 0.9 for $n=2,5,10$). The $1/(n-1)$ version is exactly **unbiased** for the variance.
* For the **standard deviation**, the $n-1$ version is *closer on average*, as the slide says, but the gain is small and it is still slightly low (0.80, 0.94, 0.97 for $n=2,5,10$), because the square root of an unbiased variance is not unbiased.
* **Why it happens:** $\bar x$ is computed from the same data, so the points sit closer to $\bar x$ than to the true $\mu$. One "degree of freedom" is used up. (The $n-1$ is called **Bessel's correction**.)
* **A subtlety the notes skip:** "unbiased" does not mean "smaller error". For $n=5$ the mean squared error of the variance estimate is $0.362$ with $1/n$ but $0.503$ with $1/(n-1)$. The biased estimator wins on MSE (the bias–variance trade-off of Lecture 10). For **reporting a SEM** the lecture uses $n-1$, which is the standard convention.

In NumPy: `x.std(ddof=1)` gives the $1/(n-1)$ version; `x.std()` gives $1/n$. `scipy.stats.sem(x)` uses `ddof=1`.

### 12.6 Reporting two models with the SEM

For models with averages $\bar X_n$ and $\bar Y_n$, draw each as a dot with a ±1 SEM bar. **If the intervals are apart, we can state that $Y$ performs statistically significantly better than $X$.** The numbers are in Lecture 11: $A=0.92\pm0.0071$ and $B=0.87\pm0.0071$ with a gap of $0.0358$. Remember the caution there: merely non-overlapping ±1-SEM bars correspond to $p\approx0.16$, not 0.05.

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

#### 🧪 Check yourself

1. What is the entropy of a fair coin? *(1 bit, the maximum for two outcomes.)*
2. $H(S,T)=1.6855$ and $H(S)=0.971$. What is $H(T|S)$? *(1.6855 − 0.971 = 0.7145 bits.)*
3. When is mutual information exactly 0? *(When the two variables are independent.)*
4. Why divide by $n-1$ for the sample variance? *(The mean was computed from the same data, which uses up one degree of freedom; dividing by $n$ would be too small on average.)*

#### 📖 Go deeper: plain-words intuition (optional)

* **Entropy as 20 questions.** A fair coin needs one yes/no question. A coin that is 99% heads needs almost none, because you can nearly guess.
* **Mutual information as "shared information".** It is the uncertainty about one thing that disappears once you know the other.
* **Where $n-1$ comes from.** Data points are closer to their *own* average than to the true mean, so measuring spread around $\bar x$ underestimates. Dividing by $n-1$ nudges it back up.

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

#### ✅ What you must be able to do

* Compute entropy, conditional entropy and mutual information from a small table.
* Derive the SEM interval from the CLT step by step.
* Explain why we divide by n−1 and which NumPy call does it.
* Explain model versus reality and why validation is needed.

---

## Lecture 13 — Bayes' Law and MAP Estimation

**The one-line idea.** Maximum likelihood asks, "which settings make my data most likely?" **MAP** asks, "which settings are most believable once I add what I believed *before* seeing the data?" That earlier belief is called the **prior**, and it stops the answer from going wild when you have little data.

### 13.1 Bayes' law, from a picture

Draw a big square for *everything that could happen*. Draw a green circle for event $A$ ("the student passed") and a blue circle for event $B$ ("studied at least 4 hours"). The circles overlap. A **probability is a share of the square's area.**

* $p(A)$ = green area ÷ square area. $p(B)$ = blue area ÷ square area.
* $p(A,B)$ = overlap area ÷ square area ("both happen").
* $p(A\mid B)$ = overlap ÷ **blue circle**. Read the bar as "given that": *pretend you are already inside the blue circle; what fraction of it is also green?*
* $p(B\mid A)$ = overlap ÷ green circle.

The overlap can be written two ways: $p(A,B)=p(A\mid B)\,p(B)$ and $p(A,B)=p(B\mid A)\,p(A)$. Both equal the same overlap, so they equal each other. Divide by $p(B)$ and you get **Bayes' law**:

$$p(A\mid B)=\frac{p(B\mid A)\,p(A)}{p(B)}.$$

**What it is for:** it **flips** a conditional probability. If you know how likely $B$ is when $A$ is true, you can work out how likely $A$ is when $B$ is true.

**Check on our ten students** (hours 4, 2, 6, 1, 3, 5, 7, 3, 8, 2; students 1, 3, 7, 9 passed):

* 4 of 10 passed, so $p(A)=0.4$.
* 5 of 10 studied at least 4 hours, so $p(B)=0.5$.
* Students with both (hours 4, 6, 7, 8): 4 of 10, so $p(A,B)=0.4$.
* Directly: $p(A\mid B)=0.4/0.5=0.8$ and $p(B\mid A)=0.4/0.4=1.0$.
* Bayes: $\dfrac{1.0\times0.4}{0.5}=0.8$ ✔.

### 13.2 The problem with maximum likelihood

Maximum likelihood (Lecture 04) picks $\theta$ to make the data as probable as possible. With flexible models and little data it can **memorize noise**. The slide shows a curve fitted with a polynomial of degree 3, 5 and 50. The degree-50 curve touches every training dot but swings wildly in between and shoots off to infinity at the edge.

| | training error | test error |
|---|---|---|
| **Overfitting** | low | high |
| **Underfitting** | high | high |

*Why it happens:* the problem is **under-constrained**. Many different settings fit the training data almost equally well, and maximum likelihood has no way to prefer a sensible one.

*Our numbers:* a degree-7 polynomial on the ten students fits the training data almost perfectly (training MSE 3.40, which is just the unavoidable spread between students with the same hours). But leave-one-out testing (Lecture 10) gives an error of about 10,000, which is absurd. That is overfitting.

### 13.3 The Bayesian idea: give the settings a prior

Instead of treating the setting $\theta$ as one unknown number, treat it as **uncertain**, with its own probability distribution. Bayes' law then reads:

$$\underbrace{p(\theta\mid\mathcal D)}_{\text{posterior}}=\frac{\overbrace{p(\mathcal D\mid\theta)}^{\text{likelihood}}\ \overbrace{p(\theta)}^{\text{prior}}}{\underbrace{p(\mathcal D)}_{\text{evidence}}}.$$

| Name | In plain words |
|---|---|
| **Prior** $p(\theta)$ | what I believed about $\theta$ **before** the data |
| **Likelihood** $p(\mathcal D\mid\theta)$ | how well $\theta$ explains the data (the same thing as Lecture 04) |
| **Evidence** $p(\mathcal D)$ | the probability of the data overall. Hard to compute, but it is just a constant, so we can ignore it. |
| **Posterior** $p(\theta\mid\mathcal D)$ | what I believe **after** the data: a compromise between prior and data |

### 13.4 The MAP estimate

**MAP** (maximum a posteriori) picks the most believable setting after seeing the data: the peak of the posterior. Because the evidence is a constant, it cannot change *where* the peak is, so

$$\theta^*=\arg\max_\theta\ p(\mathcal D\mid\theta)\ p(\theta).$$

*($\propto$ means "proportional to".)* In words: **likelihood × prior**. With logs it becomes log-likelihood **plus** log-prior. If the prior is flat (every $\theta$ equally believable), MAP is the same as maximum likelihood.

### 13.5 Worked example: a coin with a Beta prior

**Data.** Ten flips: $1,0,1,0,0,0,1,0,1,0$ (1 = heads). So $D=10$ flips and $H=4$ heads.

**Likelihood.** For a coin with heads-probability $\theta$: $\theta^{H}(1-\theta)^{D-H}$. Its peak (the maximum likelihood answer) is $H/D=0.4$.

**The problem to fix.** With few flips the answer can be extreme. If all ten flips were heads, maximum likelihood says $\theta=1$: "this coin *always* lands heads". Nobody believes that. We want a prior that prefers **coins close to fair (0.5).**

**The Beta distribution** is a bump over $\theta\in[0,1]$:

$$p(\theta)\propto\theta^{\alpha-1}(1-\theta)^{\beta-1}.$$

$\alpha$ and $\beta$ are settings of the prior. With $\alpha=\beta=2$ it is a gentle arch centred at 0.5. Larger equal values (like 6 and 6) make a taller, narrower arch: a **stronger** belief that the coin is fair.

**Posterior.** Multiply likelihood and prior (with $\alpha=\beta=2$):

$$\theta^{H}(1-\theta)^{D-H}\cdot\theta^{1}(1-\theta)^{1}=\theta^{H+1}(1-\theta)^{D-H+1}.$$

**The prior acted like one extra imaginary head and one extra imaginary tail.** Its peak is

$$\theta_{\text{MAP}}=\frac{H+1}{D+2}=\frac{4+1}{10+2}=\frac5{12}=0.4167.$$

This is slightly pulled from 0.4 toward 0.5. (Adding one imaginary head and tail is called **add-one smoothing**.) For a general Beta prior:

$$\theta_{\text{MAP}}=\frac{H+\alpha-1}{D+\alpha+\beta-2}.$$

| prior | $\alpha,\beta$ | answer for $H=4$ of 10 | answer for $H=10$ of 10 |
|---|---|---|---|
| flat (no opinion) | 1, 1 | 0.400 (same as max. likelihood) | 1.000 |
| weak | 2, 2 | 0.417 | 0.917 |
| medium | 3, 3 | 0.429 | 0.857 |
| strong | 6, 6 | 0.450 | **0.750** |

**Reading the table.** With ten heads out of ten, maximum likelihood says "always heads" (1.0). A fair-coin prior pulls the answer back, and the **stronger the prior, the harder the pull**. With lots of data the likelihood wins and the prior matters less.

#### 🧪 Check yourself

1. In the area picture, what does $p(A\mid B)$ divide by? *(The area of circle B, not the whole square.)*
2. Why can we ignore the evidence $p(\mathcal D)$ when finding the best $\theta$? *(It does not depend on $\theta$, so it cannot move the peak.)*
3. When does MAP equal maximum likelihood? *(When the prior is flat.)*
4. With 3 heads in 3 flips and a Beta(2,2) prior, what is $\theta_{\text{MAP}}$? *($\frac{3+1}{3+2}=0.8$, not 1.)*

#### 📖 Go deeper: plain-words intuition (optional)

* **Conjugate prior.** The Beta prior is "conjugate" to coin-flip data: the posterior is again a Beta, namely Beta$(\alpha+H,\ \beta+D-H)$. You just add the counts. $\alpha-1$ and $\beta-1$ act like imaginary heads and tails ("pseudo-counts").
* **Mode vs mean.** MAP is the *peak* (mode) of the posterior. The *average* of the posterior is $\frac{\alpha+H}{\alpha+\beta+D}$, a blend of the prior mean and the data.
* **Textbook:** Murphy §4.5 (p.120–122), §4.6.2 (p.130–133); Bishop §1.1 (p.4–11) and §2.1.1 (p.71).

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

#### ✅ What you must be able to do

* Explain Bayes' law with the area picture, and use it on a small example.
* Name prior, likelihood, evidence and posterior, and say what each means.
* Explain why MAP = "likelihood × prior" and why the evidence can be dropped.
* Compute $\theta_{\text{MAP}}$ for a coin with a Beta prior and describe how a stronger prior changes it.

---

## Lecture 14 — Lasso and Ridge Regression

**The one-line idea.** To stop a model from overfitting, **punish big weights**. Doing that is exactly the same as having a **prior belief that weights are small**. The *shape* of the punishment decides what happens: **ridge** makes weights small, **lasso** makes many of them exactly **zero**.

### 14.1 Recap from Lecture 13

MAP picks the setting that maximizes **likelihood × prior**. Lecture 13 showed this with coins. Now we use it for regression, where the settings are the weights $\theta_1,\dots,\theta_n$.

### 14.2 Why we need it: too many features

In Lecture 07 we made new columns from $x$ (basis functions) such as $x,\ x^2,\ x^3,\dots$. A line in these new columns can draw a curve. Common choices:

| family | formula | shape |
|---|---|---|
| polynomial | $x^k$ | affects every $x$ |
| Gaussian bump | $\exp\!\big(-\frac{(x-\mu_k)^2}{s^2}\big)$ | a hill around the centre $\mu_k$ |
| sigmoid | $\sigma\!\big(\frac{x-\mu_k}{s}\big)$ | a smooth step at $\mu_k$ |

*Numbers:* for a student with 4 hours, bumps centred at 2, 5, 8 (width 2) give $(0.368,\ 0.779,\ 0.018)$. The student is closest to the centre at 5.

**The trouble:** if you use too many features, the model starts fitting random noise. Each weight $\theta_i$ tells you how much feature $i$ matters. If a weight is **exactly zero**, that feature is **removed from the model**. That is called **feature selection**. We need a way to push useless weights toward zero.

### 14.3 A prior that prefers small weights

Choose a prior that says "big weights are unlikely":

$$p(\boldsymbol\theta)\propto e^{-\lambda\sum_i|\theta_i|^p}.$$

* The sum $\sum|\theta_i|^p$ is a **penalty** that grows when weights are large. Bigger penalty means a less believable setting.
* $\lambda$ is the **strength** of the belief. Bigger $\lambda$ means "weights are surely small".
* $p$ is the **shape** of the penalty. The slide shows 2-D pictures of equal-belief curves:

| $p$ | the curve looks like | name |
|---|---|---|
| 0.5 | a spiky star | very sparse (rarely used) |
| **1** | a **diamond** with corners on the axes | **Laplace prior** (lasso) |
| **2** | a **circle** | **Gaussian prior** (ridge) |
| 4 | a rounded square | rarely used |

*Numbers* (with $\lambda=1$): the prior value $e^{-|\theta|^p}$ at $\theta=0.5,1,2$ is 0.607, 0.368, 0.135 for $p=1$, and 0.779, 0.368, 0.018 for $p=2$. The $p=1$ curve has a **sharp peak at 0**; the $p=2$ curve has a rounded top.

### 14.4 How the prior and the data combine

MAP maximizes the log of (likelihood × prior). Maximizing is the same as minimizing the negatives:

* the negative log-likelihood is the **squared-error loss** (Lecture 07), and
* the negative log-prior is the **penalty** $\lambda\sum_i|\theta_i|^p$.

So MAP in regression means **minimizing**

$$J(\boldsymbol\theta)=\underbrace{\text{MSE}(\boldsymbol\theta)}_{\text{fit the data}}+\underbrace{\lambda\sum_i|\theta_i|^p}_{\text{keep weights small}}.$$

Larger $\lambda$ gives a simpler model; $\lambda=0$ gives ordinary regression. The two terms pull in opposite directions and the best $\boldsymbol\theta$ is a compromise.

*(Small print: the exact $\lambda$ depends on how the loss is scaled. The slide treats MSE as the negative log-likelihood, which is true up to a constant factor. That changes the meaning of $\lambda$ but not the shape of the answer.)*

**The picture (slide 12).** Take just two weights $w_1,w_2$.

* The data-fit loss is a set of **ellipses** around the best-fit point (the ordinary regression answer). Every ellipse is a set of weights with the same error.
* The penalty is a **region around the origin**: a **circle** for $p=2$ and a **diamond** for $p=1$.
* The answer is where the smallest ellipse **just touches** the region.

On a circle, the touching point can be anywhere, so both weights are usually non-zero. A diamond has **corners on the axes**, and ellipses tend to touch a corner first. At a corner one weight is **exactly zero**. This is why lasso makes sparse models.

### 14.5 Ridge and lasso

$$\textbf{Ridge:}\ \ J=\text{MSE}+\lambda\sum_i\theta_i^2\qquad\qquad \textbf{Lasso:}\ \ J=\text{MSE}+\lambda\sum_i|\theta_i|.$$

* **Ridge** (circle, $L^2$): shrinks all weights smoothly toward zero but **never exactly to zero**.
* **Lasso** (diamond, $L^1$): shrinks and also sets some weights to **exactly zero**. It is "sparse".

**Ridge has a formula.** In the squared-error form $\|\mathbf y-\mathbf X\boldsymbol\theta\|^2+\lambda\|\boldsymbol\theta\|^2$:

$$\hat{\boldsymbol\theta}=(\mathbf X^\top\mathbf X+\lambda\mathbf I)^{-1}\mathbf X^\top\mathbf y .$$

It is Lecture 07's formula with $\lambda$ added on the diagonal. (This also repairs the case where $\mathbf X^\top\mathbf X$ cannot be inverted, the "add a tiny $\epsilon$" trick of Lecture 06.) **Lasso has no neat formula** because of the sharp corner at 0, so it is solved by repeated steps. Rule of thumb: **do not penalize the intercept** $\theta_0$, and put all features on the same scale first.

**Numbers: one feature (hours).** Ridge slope $=S_{xy}/(S_{xx}+\lambda)$ with $S_{xy}=245.6$, $S_{xx}=48.9$. Lasso slope $=(S_{xy}-\lambda/2)/S_{xx}$, or 0 if that goes below zero.

| $\lambda$ | ridge slope | lasso slope |
|---|---|---|
| 0 | 5.0225 | 5.0225 |
| 50 | 2.4833 | 4.5112 |
| 100 | 1.6494 | 4.0000 |
| 400 | – | 0.9325 |
| 500 | – | **0** |
| 1000 | 0.2342 | 0 |

Ridge divides by a bigger number, so the slope shrinks smoothly but never reaches zero (even at $\lambda=1000$ it is 0.234). Lasso subtracts a fixed amount, so it hits **exactly zero** once $\lambda$ is large enough (here about 491). Beyond that the model just predicts the average score.

**Numbers: lasso removes junk.** Add two columns of pure random numbers next to hours, standardize, and fit:

| $\lambda$ | lasso: [hours, junk1, junk2] | ridge: [hours, junk1, junk2] |
|---|---|---|
| 0 | [10.78, −0.51, 0.07] | [10.78, −0.51, 0.07] |
| 10 | – | [4.77, −2.16, −0.34] |
| 20 | **[10.11, 0, 0]** | – |
| 100 | [6.11, 0, 0] | [0.96, −0.63, −0.25] |

At $\lambda=20$ **lasso keeps the real feature and sets both junk weights to exactly 0.** Ridge keeps all three at every setting, so junk keeps leaking noise in.

```python
soft = lambda z, d: np.sign(z) * max(abs(z) - d, 0)   # "soft-thresholding"
def lasso_cd(X, y, lam, iters=500):                   # one weight at a time, repeat
    w = np.zeros(X.shape[1]); a = (X ** 2).sum(0)
    for _ in range(iters):
        for d in range(X.shape[1]):
            c = X[:, d] @ (y - X @ w + X[:, d] * w[d])
            w[d] = soft(c, lam / 2) / a[d]
    return w
```

Soft-thresholding means: move a number toward zero by a fixed amount, and if that crosses zero, stop at zero.

### 14.6 Regularization in general

**Regularization** means *any* method that limits how complicated a model can be, to reduce overfitting. The $L^p$ penalty is one form. (Neural networks add others: dropout, early stopping, and more. See Lecture 15.)

### 14.7 MAP as an "inductive bias"

An **inductive bias** is a built-in preference that makes an algorithm favour some answers over others. A prior is one. Two examples from the slide:

1. **Weight decay:** $L^2$ or $L^1$ penalties for linear regression and also for neural networks.
2. **Starting values:** neural networks start their weights from random numbers with a carefully chosen spread, such as **Xavier initialization** (spread $\sigma^2=\frac2{n_{\text{in}}+n_{\text{out}}}$). That is also like putting a prior on the weights before training.

### 14.8 Real data example: prostate cancer

The slides predict a medical measurement (log PSA level) from 8 patient features.

* **Ridge** plot: as $\lambda$ grows, **every** coefficient shrinks smoothly toward 0 and none is exactly zero.
* **Lasso** plot: as $\lambda$ grows, coefficients fall to **exactly zero one by one**. With a large $\lambda$ only a few features (like lcavol) stay. This is **feature selection**, as the slide's red note says.

#### 🧪 Check yourself

1. Which of ridge and lasso can set a weight to exactly zero? *(Lasso.)*
2. Make $\lambda$ very large. What happens to the weights? *(They are all pushed to 0, so the model underfits.)*
3. Why does a diamond make zeros more likely than a circle? *(Its corners sit on the axes, and the error ellipse usually touches a corner first.)*
4. Ridge gave slope 2.48 at $\lambda=50$ and OLS gave 5.02. By what factor did it shrink? *(About 0.49, i.e. $48.9/98.9$.)*

#### 📖 Go deeper: plain-words intuition (optional)

* **Ridge as a shrink factor.** In the SVD view, ridge multiplies each direction by $\frac{\sigma_j^2}{\sigma_j^2+\lambda}$. Directions the data pins down well ($\sigma_j^2$ large) are barely touched; noisy directions are shrunk a lot.
* **Lasso is biased.** Soft-thresholding shrinks even the large weights. A common fix is to use lasso to pick features and then refit those features with ordinary regression.
* **Choosing $\lambda$:** try many values and pick the best by cross-validation (Lecture 10).
* **Elastic net** mixes both penalties. **Group lasso** removes whole groups of features at once.
* **Textbook:** Murphy §11.3 ridge (p.381), §11.4 lasso (p.385–397), §4.5.3 weight decay (p.123); Bishop §3.1.4 (p.144).

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

#### ✅ What you must be able to do

* Say why a penalty on weights equals a prior belief that weights are small.
* Write $J=\text{MSE}+\lambda\sum|\theta|^p$ and say what $\lambda$ and $p$ control.
* Contrast ridge and lasso (smooth shrinkage vs exact zeros) and explain it with the circle-and-diamond picture.
* Use the ridge formula and the one-feature slope formulas on a small example.

---

## Lecture 15 — Multilayer Perceptron (Neural Networks)

**The one-line idea.** In linear regression *you* choose the features. A neural network **learns its own features** from the data: each "hidden unit" is a small logistic-regression-like model, and all of them are trained together by gradient descent.

### 15.1 Where we are

Lecture 14 showed that too many hand-made features cause overfitting, and lasso can switch the useless ones off. The new question: **can the model discover good features by itself?** Yes. That is a neural network.

### 15.2 The perceptron: one artificial neuron

A biological neuron adds up the signals arriving on its inputs and fires if the total is big enough. The **perceptron** copies that:

$$f(\mathbf x)=\text{sign}(\mathbf w^\top\mathbf x+b).$$

Multiply each input by a weight, add everything plus a bias $b$, and output **+1** if the total is positive and **−1** if it is negative.

| model | formula | output |
|---|---|---|
| linear regression | $\mathbf w^\top\mathbf x+b$ | any number |
| logistic regression | $\sigma(\mathbf w^\top\mathbf x+b)$ | a probability |
| **perceptron** | $\text{sign}(\mathbf w^\top\mathbf x+b)$ | +1 or −1 |

*Example:* with $w=0.8$ per hour and $b=-2.4$, a 4-hour student scores $0.8\cdot4-2.4=0.8>0$, so "pass". A 2-hour student scores $-0.8$, so "fail". The boundary is $h=3$.

*History:* the 1969 book *Perceptrons* by Minsky and Papert showed that a single perceptron cannot do some simple tasks. That contributed to the first "AI winter" (years of reduced funding). The 2024 Nobel Prize in Physics was awarded for foundational work on neural networks.

### 15.3 How a perceptron learns

Labels are $y\in\{-1,+1\}$. A point is **misclassified** when the true label and the prediction have opposite signs. The learning rule: **look at each point; if it is wrong, nudge the weights toward it.**

$$\mathbf w\leftarrow\mathbf w+y\,\mathbf x\qquad(\text{only for misclassified points; the bias is an extra input fixed at 1}).$$

**Worked example.** Points $(1,3)$ and $(2,2)$ have label $+1$. Points $(2,0)$ and $(3,1)$ have label $-1$. Start with all weights 0.

| epoch | point | wrong? | new weights (bias, $w_1$, $w_2$) |
|---|---|---|---|
| 1 | (1,3), +1 | yes (score 0) | (1, 1, 3) |
| 1 | (2,2), +1 | no | (1, 1, 3) |
| 1 | (2,0), −1 | yes | (0, −1, 3) |
| 1 | (3,1), −1 | yes (score 0) | (−1, −4, 2) |
| 2 | (1,3) | no | (−1, −4, 2) |
| 2 | (2,2), +1 | yes | **(0, −2, 4)** |
| 3 | all four | no mistakes | done |

The final rule is $\text{sign}(-2x_1+4x_2)$ and all four points are right. It took 4 updates.

**It only works when a straight line can separate the classes.** On the Iris flowers, Setosa vs the rest *can* be separated using petal length and width, and the perceptron settles down. Virginica vs the rest *cannot*, so the algorithm **keeps oscillating forever**.

### 15.4 The XOR problem

XOR ("exclusive or") is 1 only when the two inputs differ:

| $x_1$ | $x_2$ | $y$ |
|---|---|---|
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 0 |

No straight line separates $\{(0,1),(1,0)\}$ from $\{(0,0),(1,1)\}$. A single perceptron makes 4 mistakes in every one of its epochs, forever. Two fixes: (1) use fixed features (adding the product $x_1x_2$ makes it solvable), or (2) **learn the features**, which is the neural network.

### 15.5 Learned features: sigmoid bases

Make each feature a small logistic model with its own settings:

$$\phi_m(\mathbf x)=\sigma(\mathbf v_m^\top\mathbf x+b_m),\qquad \hat y=\sum_m w_m\,\phi_m(\mathbf x).$$

Train $\mathbf w$ **and** every $\mathbf v_m,b_m$ together by gradient descent. This is a **two-layer network**.

*Numbers:* two hidden units, $\phi_1(h)=\sigma(2(h-3))$ and $\phi_2(h)=\sigma(2(h-6))$, output weights $(40,20)$ and bias 30. For 4 hours: $\phi_1=0.881$, $\phi_2=0.018$, so $\hat s=30+40(0.881)+20(0.018)=65.6$. For 7 hours: $\phi_1\approx1$, $\phi_2=0.881$, so $\hat s=87.6$. Training would also slide the steps and heights to fit the data.

### 15.6 The multilayer perceptron (MLP)

$$\hat{\mathbf y}=\mathbf g\big(\mathbf W\,\mathbf h(\mathbf V\mathbf x)\big).$$

Read it from the inside out:

1. $\mathbf V\mathbf x$: the first layer of weights mixes the inputs into $M$ scores.
2. $\mathbf h(\cdot)$: apply an **activation function** to each score. The results are the **hidden units**.
3. $\mathbf W(\cdot)$: the second layer of weights mixes the hidden units into $C$ outputs.
4. $\mathbf g(\cdot)$: a final **output activation** that depends on the task (below).

**One full pass with numbers.** Two inputs, two hidden units, one output; $h$ = ReLU (see 15.8), $g$ = identity.

$$\mathbf V=\begin{pmatrix}0.5&-1\\1&0.5\end{pmatrix},\ \ \mathbf W=\begin{pmatrix}1&0.5\end{pmatrix},\ \ \mathbf x=\begin{pmatrix}1\\2\end{pmatrix},\ \ \text{target }y=3.$$

*Forward (inputs to output):*

1. $\mathbf V\mathbf x=(0.5\cdot1-1\cdot2,\ 1\cdot1+0.5\cdot2)=(-1.5,\ 2)$.
2. ReLU keeps positives and turns negatives into 0: $(0,\ 2)$.
3. Output $=1\cdot0+0.5\cdot2=1$.
4. Loss $=\tfrac12(1-3)^2=2$.

*Backward (this is **backpropagation**: work out how the loss changes with each weight, starting at the output and going back using the chain rule):*

* Output error $=\hat y-y=-2$.
* Gradient for $\mathbf W$: $(-2)\cdot(0,2)=(0,\ -4)$.
* Error sent back to the hidden units: $\mathbf W^\top(-2)=(-2,\ -1)$. The first unit was switched off by ReLU, so it passes back 0: $(0,\ -1)$.
* Gradient for $\mathbf V$: $\begin{pmatrix}0&0\\-1&-2\end{pmatrix}$.

*One step with learning rate 0.1:* $\mathbf V\to\begin{pmatrix}0.5&-1\\1.1&0.7\end{pmatrix}$ and $\mathbf W\to(1,\ 0.9)$. New forward pass: hidden $(0,\ 2.5)$, output $2.25$, loss $0.28$. The loss fell from **2 to 0.28** and the output moved toward the target 3.

### 15.7 Matching the output to the task

The output activation and the loss go together. The loss is always "minus the log-likelihood" (Lectures 07–09):

| task | output activation | loss |
|---|---|---|
| predict a number | none (identity) | mean squared error |
| yes / no | sigmoid | binary cross-entropy |
| one of $C$ classes | softmax | cross-entropy |

*Numbers:* yes/no with score 1.2: $\sigma(1.2)=0.769$; loss if the truth is "yes" $=0.263$, if "no" $=1.46$. Three classes with scores $(2,1,0)$: probabilities $(0.665,\ 0.245,\ 0.090)$; loss if the first class is true $=0.408$.

### 15.8 Activation functions

The hidden activation $h$ must be **non-linear**. Here is why:

* **Identity** ($h(x)=x$): $\mathbf W(\mathbf V\mathbf x)=(\mathbf W\mathbf V)\mathbf x$, a single matrix. Two layers collapse into one layer and **nothing is gained**. (A hidden layer smaller than the input does compress the data, giving a matrix of "low rank".)
* **Sigmoid** $\sigma(x)$: squashes to (0,1). Far from zero the curve is flat, so its slope is tiny: **vanishing gradients**.
* **Tanh** $=2\sigma(2x)-1$: squashes to (−1,1) and is centred at 0. Also flat far from zero. (The slide writes $2\sigma(x)-1$; that is a typo. See "Mistakes found in the slides" at the end.)
* **ReLU** $=\max(0,x)$: zero for negative inputs, the input itself for positive ones. It usually trains deep networks better than sigmoid. Downside: a unit whose input is always negative has **zero gradient** (a "dead" unit).
* **Leaky ReLU** $=\max(0,x)+\gamma\min(0,x)$: lets a small slope $\gamma$ through on the negative side, fixing dead units.
* **Softplus** $=\log(1+e^x)$: a smooth ReLU, rarely used.

| $x$ | −2 | −1 | 0 | 1 | 2 |
|---|---|---|---|---|---|
| sigmoid | 0.119 | 0.269 | 0.5 | 0.731 | 0.881 |
| slope of sigmoid | 0.105 | 0.197 | 0.25 | 0.197 | 0.105 |
| tanh | −0.964 | −0.762 | 0 | 0.762 | 0.964 |
| ReLU | 0 | 0 | 0 | 1 | 2 |
| leaky ReLU ($\gamma=0.1$) | −0.2 | −0.1 | 0 | 1 | 2 |
| softplus | 0.127 | 0.313 | 0.693 | 1.313 | 2.127 |

**Why gradients vanish.** Backpropagation multiplies one slope per layer. A sigmoid's slope is **at most 0.25**, so after $L$ layers the signal is at most $0.25^L$: 0.25 after one layer, 0.0625 after two, about 0.001 after five, and about $10^{-6}$ after ten. The early layers barely learn. ReLU's slope is 1 on its active side, so the signal does not shrink.

### 15.9 Network architecture

The **architecture** is the shape of the network.

* **Depth** is the number of layers; **width** is the number of units in a layer.
* **Fully connected** layers link every unit to every unit of the next layer. **Sparse** layers keep only some links.
* **Skip connections** let a layer feed one further ahead.
* A feed-forward network is a **directed acyclic graph** (arrows go one way, no loops). Loops would make backpropagation go round forever.

**Counting parameters.** $D$ inputs, $M$ hidden units, $C$ outputs, with biases: $M(D+1)+C(M+1)$. For images of 784 pixels with 100 hidden units and 10 outputs: $100\cdot785+10\cdot101=79{,}510$ settings.

### 15.10 Universal approximation

**Theorem (informal):** a network with **one hidden layer** and enough units can approximate **any** continuous function as closely as you like.

*Experiment:* fit $y=\sin(2\pi x)$ using $M$ randomly placed sigmoid units (training only the output weights):

| hidden units | 2 | 5 | 10 | 20 | 50 |
|---|---|---|---|---|---|
| biggest error | 1.13 | 0.037 | 0.0004 | $10^{-8}$ | $10^{-10}$ |

Two units are useless, five are close, and twenty are essentially exact.

**But there are limits:**

1. For $D$ inputs, the number of units needed can grow **exponentially** with $D$ (the "curse of dimensionality": a grid with 10 points per axis needs 10, 100, 1000 points for $D=1,2,3$ and $10^{10}$ for $D=10$).
2. The network may need to be very wide.
3. The theorem is about **fitting** the data. It does **not** promise good predictions on new data.

### 15.11 Depth versus width

Deep ReLU networks with limited width are also universal. In practice **adding layers often works better than adding width**. Each layer builds on the features from the one before it, so the model learns simple things first and combines them into more abstract ones. That is a useful built-in preference (inductive bias). The slide's picture shows each layer **folding** the input space like paper, so a few layers create very many pieces.

### 15.12 Regularization for neural networks

Neural networks overfit too. Main remedies:

1. **Weight decay** ($L^2$ or $L^1$ penalty): exactly Lecture 14. Each step shrinks the weights by a small factor, so they "decay". With step size $\alpha$ and strength $\lambda$ the factor is $(1-\alpha\lambda)$.
2. **Data augmentation:** make new training examples by changing old ones (adding noise, flipping or rotating images). More varied data means better generalization.
3. **Noise robustness / flat minima:** small-batch SGD adds noise to the steps, which makes training skip over narrow, sharp valleys and settle in wide, flat ones. Flat valleys tend to work better on new data.
4. **Early stopping:** training error keeps falling but test error eventually rises (a U shape). **Stop at the lowest validation error.** It limits how far the weights can travel, so it acts like an $L^2$ penalty.
5. **Dropout:** during training, randomly switch off each hidden unit with probability $p$ (for example 0.5), so each step trains a thinned-out sub-network. No unit can rely on another being present. At test time use the full network with the weights **multiplied by the keep probability $1-p$**, or average several dropout runs ("Monte Carlo dropout").

*Dropout numbers:* hidden values $(1,2,3,4)$, all output weights 0.5, so the full output is 5. With drop probability 0.5 the training-time average output is 2.5 ($=0.5\times5$). With drop probability 0.2 it is 4.0 ($=0.8\times5$). So at test time we must scale by $1-p=0.8$. (The slide says "scale by $p$", which is only right when $p=0.5$.)

### 15.13 Summary

* Deep networks **learn their features**; higher layers hold more abstract ones.
* Depth usually helps more than width.
* They can fit almost anything, which is exactly why they **need regularization**.

```python
# the worked example in code
x = np.array([1., 2.]); V = np.array([[.5, -1], [1, .5]]); W = np.array([[1, .5]]); y = np.array([3.])
a = V @ x; z = np.maximum(a, 0); yhat = W @ z                 # forward: [-1.5 2] [0 2] [1]
dy = yhat - y; dW = np.outer(dy, z)                           # backward
da = (W.T @ dy) * (a > 0); dV = np.outer(da, x)
V2, W2 = V - 0.1 * dV, W - 0.1 * dW                           # one gradient step -> loss 0.28125
```

#### 🧪 Check yourself

1. Why can a single perceptron not solve XOR? *(No straight line separates the classes.)*
2. What goes wrong if every hidden activation is the identity? *(The layers collapse into one linear map.)*
3. Why do sigmoid networks train badly when very deep? *(Each layer multiplies the signal by at most 0.25, so it vanishes.)*
4. With drop probability 0.3, by what factor do you scale the weights at test time? *(0.7, the keep probability.)*

#### 📖 Go deeper: plain-words intuition (optional)

* **Xavier initialization.** Choose starting weights with spread $\sigma^2=\frac2{n_{\text{in}}+n_{\text{out}}}$ so signals neither explode nor vanish. For 784 inputs and 100 outputs, $\sigma^2=2/884=0.00226$. For ReLU, "He initialization" ($2/n_{\text{in}}$) is recommended. All-equal starting weights would never work: every hidden unit would get the same gradient and stay identical.
* **Exploding gradients** are tamed by **gradient clipping**.
* **Textbook:** Murphy Chapter 13 (p.425–458: XOR p.426, activations p.428, depth p.434, backprop p.438, initialization p.452, dropout p.455); Bishop §4.1.7 perceptron (p.192), §5.1–5.3 (p.227–245), §5.5 regularization (p.256).

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

#### ✅ What you must be able to do

* Write the perceptron rule and run a few updates by hand.
* Explain why XOR needs a hidden layer.
* Compute a forward pass of a tiny MLP and say what backpropagation does.
* Match output activation to loss (identity+MSE, sigmoid+BCE, softmax+cross-entropy).
* Explain why we need non-linear activations, and compare sigmoid and ReLU.
* Name five regularization methods and say how dropout is scaled at test time.

---

## Lecture 16 — Markov Assumption and Hidden State Models

**The one-line idea.** To predict the next word (or the next anything) in a sequence, a model needs a **summary of the past**. If that summary is enough on its own, we have a **Markov** model; if the summary is not visible and must be learned, we have a **hidden-state model**.

### 16.1 Languages and machines that read symbols

A **language** is just a set of allowed strings. A **DFA** (deterministic finite automaton) is a tiny machine with a few **states**; every time it reads a symbol it jumps to a new state by a fixed rule. *Example:* a machine with two states, "even number of 1s so far" and "odd number of 1s so far", flips state each time it reads a 1. The **state remembers just enough about the past** to answer the question.

### 16.2 Markov chains: the future depends only on the present

Make the jumps **random** and you get a **Markov chain**. The **Markov assumption** in words: *given the present state, the past does not matter*:
$$p(X_{t+1}\mid X_0,\dots,X_t)=p(X_{t+1}\mid X_t).$$

**Our students.** Each student passes with probability 0.4. Track "even or odd number of passes so far". The jump table (rows = now, columns = next) is
$$\mathbf A=\begin{bmatrix}0.6&0.4\\0.4&0.6\end{bmatrix}.$$
Start in "even" for sure. After $t$ steps, the chance of "even" is

| $t$ | 1 | 2 | 3 | 4 |
|---|---|---|---|---|
| P(even) | 0.6 | 0.52 | 0.504 | 0.5008 |

There is a neat formula, $P(\text{even after }t)=\frac12(1+0.2^t)$. The chain **forgets where it started** and settles at 50/50.

### 16.3 Language models

A **language model** gives a probability to a whole sequence, of any length. The **chain rule** always lets us split it into "next symbol given everything before":
$$p(x_0,\dots,x_T)=p(x_0)\prod_{t=1}^Tp(x_t\mid x_0,\dots,x_{t-1}).$$
Problem: the "everything before" **keeps growing**. So we predict the next symbol from a fixed-size summary instead. This turns sequence modelling into the regression/classification you already know. (The slide says "predict $x$ given $y$"; it means $y$ given $x$. See the errata.)

### 16.4 Hidden-state models

Add a **hidden state** $h_t$: a summary that is **not observed**; we only see the symbols $x_t$. Each step has two parts:

* a **hidden representation model** $p(h_t\mid h_{t-1},x_{t-1})$: how the summary is updated when a new symbol arrives;
* a **prediction head** $p(x_t\mid h_t)$: what symbol to expect given the summary (like the last layer of a neural network).

Because we never see $h_t$, the probability of a sequence must **add up over every possible hidden sequence** (this is **marginalization**, "add up all the ways it could have happened"). Two warnings from the slide: the hidden state **must be Markovian** (enough on its own) or the model cannot generate sequences correctly, and the hidden state must be **learned**, not hand-designed. The step from the chain-rule product to the hidden-state sum is an **assumption**, not a mathematical identity.

### 16.5 A tiny example, by hand

Two hidden states: "low-study day" (0) and "high-study day" (1). Observations: fail (0) or pass (1). Start probabilities $(0.6,0.4)$; transitions $\begin{bmatrix}0.7&0.3\\0.4&0.6\end{bmatrix}$; chance of a pass is 0.2 in state 0 and 0.8 in state 1. We observe **pass, fail, pass**.

**Brute force:** list all $2^3=8$ hidden sequences, multiply the probabilities along each, and add. For hidden sequence 101: $0.4\cdot0.8=0.32$, then $0.4\cdot0.8=0.32$, then $0.3\cdot0.8=0.24$, product $0.02458$. The eight terms add to **0.08998**.

**Smarter (the forward algorithm):** for each state, keep "probability of what I saw so far AND being in this state". The vectors are $(0.12,0.32)$, then $(0.1696,0.0456)$, then $(0.0274,0.0626)$, which add to **0.0900**: the same answer with far less work. With 100 steps and 10 states that is about $10^4$ multiplications instead of $10^{100}$.

### 16.6 Summary

* A **Markov chain** has a visible state and the future depends only on it.
* A **language model** is a probability over sequences; the chain rule splits it into next-symbol predictions.
* A **hidden-state model** adds a learned summary $h_t$ and sums over all hidden sequences.

#### 🧪 Check yourself

1. In words, what does the Markov assumption say? *(Given the present state, the past gives no extra information about the future.)*
2. In the pass-count chain, what is $P(\text{even})$ after 2 steps? *(0.52.)*
3. Why do we sum over hidden sequences? *(We never observe them, and many of them could have produced the same symbols.)*
4. What did the forward algorithm save us? *(Work: $TK^2$ multiplications instead of $K^T$ terms.)*

#### 📖 Go deeper: plain-words intuition (optional)

* **Hidden Markov model (HMM).** Our tiny example *is* an HMM: a hidden Markov chain plus emissions. The forward algorithm computes the likelihood; training by maximum likelihood is done by the EM algorithm (beyond this course).
* **Why this leads to RNNs.** If we let a neural network *compute* $h_t$ from $h_{t-1}$ and $x_t$ instead of using fixed tables, we get a recurrent neural network (next lecture).
* **Textbook:** Murphy §15.2 (p.503–505) and §3.6.1.2 (p.101); Bishop §13.1–13.2.

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

#### ✅ What you must be able to do

* Say what the Markov assumption means and check it on a tiny chain.
* Compute the distribution of a two-state chain after a few steps.
* Write the chain-rule factorization of a sequence and explain why the history grows.
* Explain why a hidden-state model needs a sum over hidden sequences, and compute it by brute force for three steps.

---

## Lecture 17 — Sequence Modelling (RNNs, Attention, Transformers and LLMs)

**The one-line idea.** Chatbots and translators are sequence models: they read tokens one at a time, keep a **learned summary** of what they have seen, and predict the next token. The newest ones (Transformers) skip the single summary and let every token **look back at all earlier tokens** ("attention"). An LLM is simply the maximum likelihood estimate fitted to a gigantic pile of text.

### 17.1 Marginalizing: adding up the possibilities

To get the chance of one thing from a joint table, **add up over everything else**. Example: if the joint chance of (hours, pass) is known, the chance of "pass" alone is the sum over all hour values. Last lecture's sum over hidden sequences is exactly this.

### 17.2 A recurrent neural network (RNN)

An **RNN** reuses the **same small network at every step**. It holds a hidden vector $\mathbf h_t$ (the summary), updates it from the new input, and predicts an output:
$$\mathbf h_t=\text{(neural net)}(\mathbf h_{t-1},\mathbf x_t),\qquad\text{output distribution}=\text{softmax}(\mathbf W\mathbf h_t).$$
Three jobs: **vec2seq** (one input vector, a sequence out, e.g. an image caption), **seq2vec** (a sequence in, one answer out, e.g. sentiment), **seq2seq** (a sequence in, a sequence out, e.g. translation). A **bidirectional** RNN reads the sequence both ways. Text is first cut into **tokens** (word pieces) and each token becomes a number.

### 17.3 Training: backpropagation through time and its problems

Unroll the RNN over time and apply backpropagation (Lecture 15). Because the same weights are multiplied again and again, gradients **shrink toward 0 (vanishing)** or **blow up (exploding)** over long sequences. **GRU** and **LSTM** blocks add "gates" (learned on/off switches) that let information and gradients flow through many steps.

### 17.4 Attention: a soft dictionary lookup

An RNN squeezes the whole past into **one** vector of fixed size, which limits memory. **Attention** instead keeps everything, as a dictionary of **keys** and **values**. Each token makes a **query** $\mathbf q$, a **key** $\mathbf k$ and a **value** $\mathbf v$. To read:

1. compare the query with every key by a dot product (a similarity score);
2. turn the scores into probabilities with **softmax** (divide by $\sqrt d$ first to keep them tame);
3. return the **weighted average of the values**.

$$\mathbf z_t=\sum_{i\le t}\text{softmax}\Big(\tfrac{1}{\sqrt d}\mathbf q_t^\top\mathbf k_i\Big)\,\mathbf v_i.$$

**Tiny example (three tokens, $d=2$).** With no mask the attention weights for token 1 are $(0.401,0.198,0.401)$ and its output is $(1.604,1.599)$. With a **causal mask** (a token may only look at itself and earlier tokens) token 1 sees only itself: weights $(1,0,0)$, output $(1,0)$; token 2 gets weights $(0.330,0.670)$ and output $(0.330,1.340)$; token 3 is unchanged at $(1.759,2.007)$.

### 17.5 Transformers and LLMs

A **Transformer** stacks attention layers with ordinary neural-network layers. Because attention ignores order, we add **positional encodings** (e.g. rotary, "RoPE") so it knows who comes first. Families: **BERT** (reads both directions, good for understanding), **GPT** and **Llama** (read left to right, good for generating). All of this is tensor arithmetic from Lecture 02.

### 17.6 Using the model: autoregressive rollout

To **generate** text: feed the prompt, get a probability over the next token, **pick one** (sample, or take the likeliest), append it, and repeat until an end token appears.

### 17.7 Training: teacher forcing and "an LLM is maximum likelihood"

In training we know the true text, so we feed the **true previous tokens** (teacher forcing) and score all positions at once. The loss is the **negative log probability of the true next token** (cross-entropy), exactly the recipe of Lecture 04. **An LLM is the maximum likelihood estimate fitted to an internet-sized dataset.**

*Number example.* The model gives the true tokens of "This is a Puppy . ⟨end⟩" the probabilities $0.2,0.5,0.6,0.1,0.8,0.9$. The loss is $1.609+0.693+0.511+2.303+0.223+0.105=5.444$; per token $0.907$; **perplexity** $e^{0.907}=2.48$: the model is as unsure as choosing between about 2.5 equally likely tokens each time.

### 17.8 Guardrails

A model trained only to imitate text will imitate harmful text too. **Guardrails** (extra training with human feedback and reinforcement learning, plus filters) steer the model toward safe, useful answers.

### 17.9 Summary

* RNN = one network reused each step, with a learned summary; attention = look back at all tokens.
* Transformers use attention; LLMs are Transformers trained by maximum likelihood on text.
* Generate by sampling one token at a time; train with teacher forcing.

#### 🧪 Check yourself

1. Why do RNN gradients vanish? *(The same weights are multiplied many times; factors below 1 shrink the signal to nearly 0.)*
2. In attention, what do query, key and value do? *(The query is compared with the keys to get weights; the output is the weighted average of the values.)*
3. What does a causal mask do? *(It stops a token from looking at later tokens.)*
4. Probabilities 0.5 and 0.5 for two true tokens: what is the perplexity? *(2.)*

#### 📖 Go deeper: plain-words intuition (optional)

* **Why divide by $\sqrt d$?** Dot products of long vectors are large; large scores make softmax almost 0/1 (hard choice) and its gradient tiny. Dividing by $\sqrt d$ keeps the scores moderate.
* **Why teacher forcing is fast.** All positions are computed in one pass, but the cost per layer grows with the **square** of the sequence length (an $n\times n$ attention table).
* **Textbook:** Murphy Chapter 15 (p.503–545); Bishop Chapter 13 (sequential data).

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

#### ✅ What you must be able to do

* Name the three RNN jobs (vec2seq, seq2vec, seq2seq) and give an example of each.
* Explain vanishing and exploding gradients and what GRU/LSTM gates do.
* Compute attention weights from queries and keys, with and without a causal mask.
* Explain autoregressive generation and teacher forcing.
* Compute the loss and perplexity of a sentence from the token probabilities.

---

## Lecture 18 — Gradient Descent

**The one-line idea.** Many models have no formula for the best weights. So we **walk downhill**: from where we stand, find the direction in which the loss goes down fastest, take a small step, and repeat. Taking steps on a random handful of data (a **mini-batch**) makes each step cheap, and **momentum** smooths the zig-zag.

### 18.1 Where the loss comes from

Maximum likelihood (Lectures 04–09): take the probability of the data, take the log, put a minus sign in front, average. The result is a **loss** $\mathcal L(\boldsymbol\theta)$ that is small when the model fits well. This lecture is about *how to make it small*.

### 18.2 The gradient: the uphill arrow

The **gradient** is a list of slopes, one per weight: how much the loss changes if you nudge that weight alone. It points **uphill** (steepest increase), so we move the **opposite** way. Check on $J(\theta)=(\theta-3)^2$ at $\theta=0$: nudging by 0.001 gives $\frac{8.994001-9}{0.001}=-5.999\approx-6$, the true slope.

For both regression and logistic regression the gradient has the same shape, **(prediction − truth) × input, averaged over the data**:
$$\nabla\text{MSE}=\tfrac2D\sum(\boldsymbol\theta^\top\mathbf x_i-y_i)\mathbf x_i,\qquad\nabla J_{\text{logistic}}=\tfrac1D\sum(\sigma(\boldsymbol\theta^\top\mathbf x_i)-y_i)\mathbf x_i.$$
On our ten students at $\boldsymbol\theta=\mathbf 0$ the regression gradient is $(-124.8,-560.8)$ (both negative: "raise both weights") and the pass/fail gradient is $(0.1,-0.45)$.

### 18.3 Gradient descent

$$\boldsymbol\theta_{\text{new}}=\boldsymbol\theta_{\text{old}}-\alpha\,\nabla\mathcal L.$$
$\alpha$ is the **learning rate** (step size). For it to behave over time we want $\sum\alpha_t=\infty$ (we can get anywhere) and $\sum\alpha_t^2<\infty$ (the steps shrink enough to settle), for example $\alpha_t=1/t$.

**One-dimensional example**, $(\theta-3)^2$ starting at 0. Each step multiplies the distance to 3 by $(1-2\alpha)$:

| $\alpha$ | what happens |
|---|---|
| 0.1 | slow and steady: 0.6, 1.08, 1.464, 1.771, 2.017 |
| 0.5 | lands on 3 in one step |
| 1.0 | bounces 6, 0, 6, 0 forever |
| 1.1 | **diverges**: 6.6, −1.32, 8.18, −3.22, 10.5 |

**The students' regression.** The loss is steep in one direction and nearly flat in the other (curvatures 44.97 and 0.435, ratio 103). Gradient descent is stable only if $\alpha<2/44.97=0.0445$:

| $\alpha$ | steps needed |
|---|---|
| 0.001 | 19,036 (safe but very slow) |
| 0.04 | **472** (fastest here) |
| 0.05 | **blows up** |

It zig-zags across the steep direction while crawling along the flat one.

### 18.4 Convex or not?

A function is **convex** if it is a **bowl**: the straight line joining any two points on the curve lies **above** the curve. A bowl has one lowest point, so gradient descent finds the **global minimum**. Squared-error and logistic losses are convex. $f(x,y)=x^3y^5$ is not. **Neural-network losses are non-convex**: gradient descent finds *a* valley, usually the one near where it started.

### 18.5 Learning-rate curves

Plot loss against steps: **very high** rate, loss goes up; **high**, it drops then wobbles; **low**, it creeps; **good**, it drops steadily.

### 18.6 Stochastic gradient descent (SGD)

Averaging over all $D$ data points is slow. Instead pick a random **mini-batch** of $B$ points and use their average gradient. It is **unbiased** (right on average) but **noisy**. With the students at $\boldsymbol\theta=\mathbf 0$ (true gradient $(-124.8,-560.8)$): $B=2$ gives noise (standard deviation) about $(15.3,258)$, $B=5$ about $(7.6,129)$, half as much. The **Robbins–Monro** conditions on the step sizes (the two sums above) keep the noise under control.

### 18.7 Momentum

Keep a running total of past gradients: $\mathbf m_t=\beta\mathbf m_{t-1}+\mathbf g_t$ and step by $-\alpha\mathbf m_t$. Back-and-forth parts cancel; steady parts add up. On a long narrow valley, after 40 steps the distance to the minimum is 0.055 with no momentum, **0.0006 with $\beta=0.5$**, but 0.13 with $\beta=0.9$ (too much: it overshoots). Momentum and learning rate must be tuned together.

### 18.8 Fancier optimizers, PyTorch and autodiff

**Adagrad, RMSprop, Adam** (the most popular) and **AdamW** give **each weight its own step size**. **PyTorch** mimics NumPy but runs on GPUs and **computes gradients for you** (automatic differentiation): the **forward pass** computes values and stores them; the **backward pass** multiplies the local derivatives by the chain rule. Example: $y=(\sin(x^2))^3$ at $x=1$ has derivative $3\sin^2(1)\cdot\cos(1)\cdot2=2.2954$.

### 18.9 Summary

* Gradient descent minimizes a differentiable loss by stepping against the gradient.
* Convex: reaches the global minimum. Non-convex: reaches a valley near the start.
* SGD: cheaper noisy steps. Momentum: steadier steps.

#### 🧪 Check yourself

1. Which way do we step relative to the gradient? *(Opposite: the gradient points uphill.)*
2. For $(\theta-3)^2$ with $\alpha=1.1$, does it converge? *(No: each step multiplies the distance by $-1.2$.)*
3. What does a bigger mini-batch buy you? *(Less noise per step, at higher cost.)*
4. What can go wrong with too much momentum? *(It overshoots and swings back.)*

#### 📖 Go deeper: plain-words intuition (optional)

* **Why the stability limit is $2/\lambda_{\max}$.** On a bowl with curvature $\lambda$ each step multiplies the error by $1-\alpha\lambda$; that has size below 1 only if $\alpha\lambda<2$. The steepest direction sets the limit; the flattest sets the speed.
* **Why separable logistic data never finishes.** If a line separates the classes perfectly, making the weights larger always lowers the loss, so they grow forever; regularization (Lectures 13–14) fixes that.
* **Textbook:** Murphy Chapter 8 (p.275–302); Bishop §5.2.4 (p.240).

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

#### ✅ What you must be able to do

* Write the gradient of the MSE and the logistic loss as (prediction − truth) × input.
* Run gradient descent by hand and say when a learning rate is too big (below 2 over the largest curvature).
* Say whether a function is convex and what that means for gradient descent.
* Explain mini-batch SGD, the Robbins–Monro conditions and why SGD is noisy.
* Explain momentum and name Adam, RMSprop and Adagrad.
* Differentiate a chain of functions by forward and backward pass.

---

## Lecture 19 — Midterm Example Exercise and Reference Sheet

**The one-line idea.** The midterm example asks you to run the **whole course recipe** on a new problem: pick a distribution, parameterize it with the input, write the loss, and differentiate it. This one has a twist: **both the mean and the spread depend on $x$**.

**Midterm facts.** Monday **Nov 16**, in class, **on paper**, **teams of two**, **simple calculator only**, multiple choice and/or long answer. A **reference sheet** (below) is provided.

### 19.1 The problem

Data pairs $(x_i,y_i)$, $x_i\in[0,1]$, $y_i$ any real number. The picture shows noise that is small on the left and big near $x\approx0.75$. Build a model of $y$ given $x$.

### 19.2 The recipe, step by step

1. **Distribution:** $y$ is a real number, so use the **Normal**, with a mean $\mu$ and a **variance** $\sigma^2$ (not the "std. dev.", as the page says).
2. **Likelihood:** multiply over the data points, then take the log so the product becomes a sum.
3. **Parameterize both:** $\mu=\boldsymbol\theta_1^\top\boldsymbol\phi(x)$ and $\sigma^2=\exp(\boldsymbol\theta_2^\top\boldsymbol\phi(x))$. The $\exp$ keeps the variance positive, like the sigmoid keeps a probability between 0 and 1.
4. **Loss:** negate the log-likelihood.

### 19.3 ⚠️ The handwritten page misses a term

The page's loss is $\sum\frac{r_i^2}{2\exp(\boldsymbol\theta_2^\top\boldsymbol\phi)}$ with $r_i=y_i-\boldsymbol\theta_1^\top\boldsymbol\phi(x_i)$. The density is only *proportional* to the exponential; the missing constant $1/\sqrt{2\pi\sigma^2}$ **depends on $\boldsymbol\theta_2$**, so its log, $-\frac12\boldsymbol\theta_2^\top\boldsymbol\phi$, must stay. Correct loss:
$$\mathcal L=\sum_i\Big[\tfrac12\boldsymbol\theta_2^\top\boldsymbol\phi(x_i)+\frac{r_i^2}{2\exp(\boldsymbol\theta_2^\top\boldsymbol\phi(x_i))}\Big].$$
Without that term the loss is "minimized" by making the variance huge. The extra term is the **price of a wide distribution**.

### 19.4 Gradients and one step

With $e_i=\exp(\boldsymbol\theta_2^\top\boldsymbol\phi(x_i))$:
$$\nabla_{\boldsymbol\theta_1}\mathcal L=\sum-\frac{r_i}{e_i}\boldsymbol\phi(x_i),\qquad\nabla_{\boldsymbol\theta_2}\mathcal L=\sum\Big(\tfrac12-\frac{r_i^2}{2e_i}\Big)\boldsymbol\phi(x_i).$$
In words: the mean's gradient is (error) × (input), counted less for noisy points; the variance's gradient says "raise the variance if the squared error is bigger than the predicted variance, lower it otherwise".

**Numbers (one point).** $\boldsymbol\phi=(1,0.5)$, $y=2$, $\boldsymbol\theta_1=(1,1)$, $\boldsymbol\theta_2=(0,0.5)$: mean 1.5, $r=0.5$, $e=e^{0.25}=1.284$, loss $0.2224$, gradients $(-0.3894,-0.1947)$ and $(0.4026,0.2013)$. One step with $\alpha=0.1$ gives $\boldsymbol\theta_1=(1.0389,1.0195)$ and $\boldsymbol\theta_2=(-0.0403,0.4799)$.

### 19.5 The reference sheet

| Distribution | Support | Formula | Parameters |
|---|---|---|---|
| Normal | all reals | $\propto\exp(-\frac{(x-\mu)^2}{2\sigma^2})$ | $\mu$, $\sigma>0$ |
| Bernoulli | {0,1} | $P(X=1)=p$ | $p\in[0,1]$ |
| Beta | [0,1] | $\propto x^{\alpha-1}(1-x)^{\beta-1}$ | $\alpha,\beta>0$ |
| Categorical | {1,…,n} | $P(X=i)=p_i$ | probability vector |
| Binomial | 0,1,2,… | $\binom nkp^k(1-p)^{n-k}$ | $n$, $p$ |
| Poisson | 0,1,2,… | $\frac{\lambda^k}{k!}e^{-\lambda}$ | $\lambda>0$ |
| Weibull | $x\ge0$ | $\frac k\lambda(\frac x\lambda)^{k-1}e^{-(x/\lambda)^k}$ | $\lambda,k>0$ |

**Identities:** $\log ab=\log a+\log b$; $\log a^b=b\log a$; $\log\exp x=x$; sums and products written out; $p(a,b)=p(a|b)p(b)$; Bayes: $p(a|b)=\frac{p(b|a)p(a)}{p(b)}$.

**Which one to pick:** real number → Normal; two classes → Bernoulli; many classes → Categorical; counts of successes in $n$ tries → Binomial; counts of events → Poisson with $\lambda=\exp(\boldsymbol\theta^\top\boldsymbol\phi)$; positive durations → Weibull; a probability → Beta.

**Mini-example (Poisson).** Loss $\sum[e^{\theta x_i}-k_i\theta x_i]$, gradient $\sum(e^{\theta x_i}-k_i)x_i$: again (prediction − truth) × input. For $(x,k)=(0,1),(1,3),(2,8)$ the gradient at $\theta=0$ is $-16$ (so increase $\theta$) and the best $\theta\approx1.04$.

### 19.6 Checklist for Monday

* Run the six-step recipe on any label type; keep **every** parameter-dependent term.
* Gradients are (prediction − truth) × input.
* Know learning-rate stability, convexity, SGD, momentum, MAP/regularization, validation, neural nets as learned features, RNN/attention ideas.
* Practice $\exp$, $\log$ and square roots by hand.

#### 🧪 Check yourself

1. Why is $\sigma^2=\exp(\cdot)$ and not just $\boldsymbol\theta_2^\top\boldsymbol\phi$? *(The variance must be positive.)*
2. What term does the handwritten loss drop? *(The $\frac12\boldsymbol\theta_2^\top\boldsymbol\phi$ term from the normalizing constant.)*
3. Which distribution for the number of emails per hour? *(Poisson.)*
4. What is $\log(a^3b)$? *(3 log a + log b.)*

#### 📖 Go deeper: plain-words intuition (optional)

* **Why variance-dependent models are called heteroscedastic.** "Hetero-" means different: the noise level differs from input to input.
* **Textbook:** Murphy §13.2.4.4 (p.433), Bishop p.273.

#### 📖 Textbook deep dive
* **Heteroscedastic regression** (input-dependent variance): Murphy p.59 (linear case, $\sigma^2$ depends on the input), §13.2.4.4 *"MLP for heteroskedastic regression"* (p.433: the network outputs both $\mu(x)$ and $\sigma^2(x)$, trained by the NLL with a positivity transform for $\sigma^2$) and p.376; the NLL $\frac12\log\sigma^2(x)+\frac{(y-\mu(x))^2}{2\sigma^2(x)}+\text{const}$ is exactly the corrected loss of 19.3. Bishop p.273 and p.311 mention heteroscedastic models (input-dependent noise variance), and Bishop's Gaussian log-likelihood (1.62) keeps the $-\frac12\ln\sigma^2$ term.
* **Distribution families:** Murphy §2.2 (Bernoulli/Binomial/Categorical, 2.2.1–2.2.2; Poisson §2.2.3.3; Normal §2.5; Beta §2.2.4) and Bishop §2.1–2.3 (binary, multinomial and Gaussian variables). **GLMs** (Murphy §12.1–12.2): Poisson regression with the log link is the standard count model.

#### ➕ Textbook extra
Murphy §2.7 (exponential family), §12.3 (generalized linear models, canonical link) and Bishop §2.4 (exponential family) generalize why "(prediction − truth) × input" is the gradient for so many models.

#### ✅ What you must be able to do

* Run the six-step recipe for a Normal model with input-dependent mean and variance.
* Write the correct loss (with the 1/2 theta2 phi term) and both gradients.
* Pick the right distribution from the reference sheet for a given kind of label.
* Use the log identities and Bayes' rule from the sheet.

---

## Part B. Mistakes found in the slides (Lectures 02–15)

While checking every formula against the slides and textbooks we found these. They are small, but they can mislead you in an exam.

| Lecture and slide | What the slide says | What is right |
|---|---|---|
| L05 slides 3–4 | Slide 3: a repository for each student. Slide 4's title: "One Git Repository per Team" | Probably individual assignments use per-student repositories and team assignments use one per team. Check the assignment handout. |
| L10 slide 20 | $F_\beta=(1-\beta)\frac{P\times R}{\beta^2P+R}$ | $F_\beta=(1+\beta^2)\frac{P\times R}{\beta^2P+R}$. With $\beta=1$ it must become $F_1=2PR/(P+R)$; the slide's version gives 0. |
| L10 slide 20 | Fallout $=\frac{FP}{TN}$ | Fallout (false-positive rate) $=\frac{FP}{FP+TN}$. |
| L10 slide 19 | "Accuracy is the same as precision in this case." | Accuracy $=(TP+TN)/\text{all}$; precision $=TP/(TP+FP)$. They match only in special cases. |
| L11 slide 6 | $\bar X_n=\sum_iX_i$ in the CLT | $\bar X_n=\frac1n\sum X_i$. Without the $1/n$ the CLT statement fails. |
| L11–12 | $\bar X\sim\mathcal N(\mu,\sigma)$ | In the course's $\mathcal N(\text{mean},\text{variance})$ notation it is $\mathcal N(\mu,\sigma^2)$ with $\sigma=\hat\sigma/\sqrt n$. |
| L11 slide 11 | A gap between ±1-SEM bars means "statistically significant" | Only a rough guide. Bars that just touch give $p\approx0.157$; 5% needs about 2.77 SEM. With $n=5$ runs, ±1 SEM covers 62.6%, not 68.2%. |
| L11 slide 10 | "$\mu$ lies in the interval with probability 68.2%" | The 68.2% is about repeating the experiment many times, not the chance that this one fixed interval contains $\mu$. |
| L12 formula box | $H(X)=\mathbb E[\log X]$ | $H(X)=-\mathbb E[\log p(X)]$ (missing minus sign and $p$). |
| L12 right side | "$\hat\sigma$ with $n-1$ has lower bias" | It has a slightly smaller *average error*, which is not the same as bias. The $n-1$ variance is unbiased; the $n-1$ standard deviation is still slightly low. |
| L13 coin plots | The posterior's dashed peak is labelled 0.4 | It is 0.43. With the plotted prior the peak moves a little away from the data-only 0.4. |
| L13 coin plots | The derivation uses a Beta(2,2) prior, but the plotted heights match Beta(3,3) and Beta(6,6) | State the prior's $\alpha,\beta$ with each plot. With Beta(2,2) and 10 heads of 10, the answer would be 0.917, not 0.75. |
| L14 slides 11–12 | $-\log p(\mathcal D\mid\theta)=\text{MSE}(\theta)$ | Equal only up to a constant factor and a constant. The best setting is the same, but $\lambda$ then absorbs the noise variance and the number of examples. |
| L14 slides 16–17 | Loss written $\|\mathbf y-\mathbf X\theta\|_2+\alpha\|\theta\|_2^2$ | The data term must be **squared**: $\|\mathbf y-\mathbf X\theta\|_2^2$. |
| L15 slide 23 | tanh is $2\sigma(x)-1$ | It is $2\sigma(2x)-1$. (At $x=1$: 0.762, not 0.462.) |
| L15 slide 38 | Dropout at test time: "scale weights by $p$" | Multiply by the **keep probability** $1-p$. They agree only when $p=0.5$. |
| L16 slide (language models) | "predict $x$ given $y$" | It means predict $y$ given $x$. |
| L16 hidden-state formula | "=" between the chain-rule product and the hidden-state sum | A modelling assumption (the hidden state is Markovian), not an identity. |
| L17 slides 14–15 | Titled "Seq2Vec"; inputs $x_1,x_2,x_2$ | It is seq2seq; the inputs are $x_1,x_2,x_3$. |
| L17 slide 20 | Scores divided by a constant "C" | The standard choice is $\sqrt d$. |
| L18 slide 7 | Function stays "underneath (or above)" a line | For a convex function the chord lies **above** the curve. |
| L19 midterm example | Loss drops the $\frac12\theta_2^\top\phi$ term; $\sigma^2$ called "std. dev." | Keep the term (it comes from $\log\sigma$); $\sigma^2$ is the variance. |

(The Complete Study Notes list a few more, including typos in figure captions.)

---

