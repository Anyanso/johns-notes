# -*- coding: utf-8 -*-
# Content for the final exam prep site. $...$ = math (pre-rendered by KaTeX).

DAYS = [
 ("Now → Dec 3", "3–4 h a week", "Close the new material and the midterm", [
  "Get your marked midterm back. Redo every question you lost marks on, from a blank page, until you can do it without looking.",
  "Learn the two topics the midterm skipped: card 10 (sequence models, L16–17) and card 13 (SVMs, L20). Then practice groups K and L.",
  "Add each new lecture (e.g. PCA in week 12) to your notes as it is posted.",
  "Finish Assignment 3 (due Thu Dec 3, 4 pm). Assignment questions are the closest replica of exam questions.",
 ]),
 ("After A3 is in", "1–2 h a day", "Rebuild the foundations", [
  "Topics 1–4 (tensors, probability, MLE, linear algebra). Read each card, then practice groups C, D, B.",
  "Topics 5–7 (regression, logistic/softmax, generalization). Practice groups E, F, G.",
  "Topics 8 (MAP, ridge, lasso) and 9 (neural networks). Practice groups H, I.",
  "Mark every question Got it or Review. The readiness panel at the bottom counts them for you.",
 ]),
 ("Final week", "2–3 h a day", "The recipe, optimization, and a full mock", [
  "Practice group A, all of it, on paper: the recipe is the most likely long-answer question.",
  "Topics 11–13 (gradient descent, the L19 example, SVMs). Practice groups J and L again.",
  "Sit the mock exam alone, on paper, with a simple calculator, timed (150 minutes). Mark it with the marking notes.",
  "Every lost mark goes on a list. Redo those questions two days later.",
 ]),
 ("Last two days", "2 h a day", "Close every gap", [
  "Redo every question still marked Review until the readiness panel shows each group at 80% or more.",
  "From a blank page, derive the losses for Poisson, Bernoulli, Normal (fixed and learned variance), Binomial and Weibull, and write the soft-margin SVM.",
  "Read the common mistakes list in every topic card and the ten costly mistakes under the recipe.",
 ]),
 ("Day before", "1 h", "Light review, then rest", [
  "Skim the ‘What to memorize’ list once. Do not start anything new.",
  "Pack: pencil, eraser, simple calculator with fresh batteries, student ID. Sleep.",
 ]),
]

TEMPLATE_STEPS = [
 ("Pick the distribution", "Look at the label $y$. Real number: Normal. Two classes: Bernoulli. Many classes: Categorical. Count with no upper limit: Poisson. Successes out of $n$ tries: Binomial. Positive duration: Weibull. Write at most one sentence of reason (whole numbers? never negative? bounded?)."),
 ("Parameterize with the input", "Name the distribution's parameter(s) and make each a function of the features. The activation $a$ must keep the parameter in its allowed range: sigmoid for $p\\in[0,1]$, exp for a rate or variance $>0$, identity for a mean. Example: $\\lambda=a(\\boldsymbol\\theta^\\top\\boldsymbol\\phi(x))=\\exp(\\boldsymbol\\theta^\\top\\boldsymbol\\phi(x))$."),
 ("Write the likelihood and take the log", "Independent points: $p(\\mathcal D|\\boldsymbol\\theta)=\\prod_i p(y_i|x_i;\\boldsymbol\\theta)$. Then $\\log$ turns the product into a sum (sheet identity 1). Use sheet identities 2 and 3 to simplify, and keep every term that contains $\\boldsymbol\\theta$."),
 ("Negate to get the loss", "$J(\\boldsymbol\\theta)=-\\sum_i\\log p(y_i|x_i;\\boldsymbol\\theta)$. You may drop terms that do not contain $\\boldsymbol\\theta$ (like $\\log y!$ or $\\tfrac12\\log2\\pi$). Say that you dropped them."),
 ("Differentiate", "The midterm excluded gradient derivations; that limit was for the midterm only, so be ready to derive them on the final until your professor says otherwise. Chain rule: first the derivative with respect to the parameter ($\\lambda$ or $z=\\boldsymbol\\theta^\\top\\boldsymbol\\phi$), then multiply by $\\boldsymbol\\phi$. Check the result has the shape (prediction − truth) × input."),
 ("Gradient descent", "If a gradient is given, apply the update $\\boldsymbol\\theta\\leftarrow\\boldsymbol\\theta-\\alpha\\nabla_{\\boldsymbol\\theta}J$. In code the gradient is one matrix product: <code>x_feat.T @ (pred - y)</code>."),
]

WORKED = r"""
<p><strong>Problem.</strong> A shop records the number of customers $y_i\in\{0,1,2,\dots\}$ per day, with features $\boldsymbol\phi(x_i)$. Build and fit a probabilistic model.</p>
<ol>
<li><strong>Distribution.</strong> The label is a whole number, never negative, with no fixed maximum: Poisson, $P(Y=k)=\frac{\lambda^k}{k!}e^{-\lambda}$, $\lambda>0$.</li>
<li><strong>Parameterize.</strong> $\lambda_i=\exp(z_i)$ with $z_i=\boldsymbol\theta^\top\boldsymbol\phi(x_i)$. The exp keeps $\lambda>0$.</li>
<li><strong>Log-likelihood.</strong> $\log\prod_i p(y_i|x_i)=\sum_i\big[y_i\log\lambda_i-\lambda_i-\log y_i!\big]=\sum_i\big[y_iz_i-e^{z_i}-\log y_i!\big]$ (identities 1, 2, 3).</li>
<li><strong>Loss.</strong> $J(\boldsymbol\theta)=\sum_i\big[e^{z_i}-y_iz_i\big]$. The term $\log y_i!$ has no $\boldsymbol\theta$, so it is dropped.</li>
<li><strong>Gradient.</strong> $\dfrac{\partial J}{\partial z_i}=e^{z_i}-y_i$ and $\dfrac{\partial z_i}{\partial\boldsymbol\theta}=\boldsymbol\phi(x_i)$, so $\nabla_{\boldsymbol\theta}J=\sum_i(\lambda_i-y_i)\,\boldsymbol\phi(x_i)$.</li>
<li><strong>Update.</strong> $\boldsymbol\theta\leftarrow\boldsymbol\theta-\alpha\sum_i(\lambda_i-y_i)\boldsymbol\phi(x_i)$. Vectorized: <code>theta -= lr * x_feat.T @ (np.exp(x_feat @ theta) - y)</code>.</li>
</ol>
<p><strong>Check.</strong> One point, $\boldsymbol\phi=(1,2)$, $\boldsymbol\theta=(0.5,0.3)$, $y=3$: $z=1.1$, $\lambda=e^{1.1}=3.004$, gradient $=(3.004-3)(1,2)=(0.004,0.008)$. The model already predicts about 3, so the step is tiny.</p>
"""

# ---------------- Topics ----------------
# (id, title, lectures, likelihood, why, must[], formulas[], traps[], read, drill)
TOPICS = [
 ("t1","Tensors and NumPy","L02","Medium",
  "The assignments are coded in NumPy, so expect shape questions and short code reading.",
  ["Shape of a tensor, and what an axis is. $\\texttt{X.mean(axis=0)}$ collapses the rows and leaves one value per column.",
   "Broadcasting: compare shapes from the right; each pair must be equal or one of them 1; a missing leading axis counts as 1.",
   "<code>reshape</code> with <code>-1</code>, <code>np.stack</code>, boolean masks <code>h[h&gt;=5]</code>, <code>@</code> as matrix product.",
   "Why code is vectorized (no Python loops over data)."],
  ["$(D,n)\\cdot(n,)\\to(D,)$","$(D,n)^\\top(D,)\\to(n,)$"],
  ["Subtracting a $(D,)$ vector from a $(D,1)$ array gives $(D,D)$, not $(D,)$.","<code>sum(axis=1)</code> removes axis 1 (the columns), it does not sum the columns of each row separately."],
  "Lecture 02; Murphy §7.2 p.242 for the matrix-product ideas.", "C1–C9"),
 ("t2","Probability and distributions","L03","High",
  "Every question starts by choosing or reading a distribution. The reference sheet lists seven.",
  ["Random variable, PMF vs PDF, independence, expectation and variance.",
   "Bayes' rule and the chain rule $p(a,b)=p(a|b)p(b)$ (sheet identities 6 and 7).",
   "Bernoulli, Categorical, Binomial, Poisson, Normal, Beta, Weibull: support and parameters, and which label type each fits."],
  ["$p(a|b)=\\dfrac{p(b|a)p(a)}{p(b)}$","$\\mathbb E[X]=\\sum xp(x)$, $\\text{Var}(X)=\\mathbb E[X^2]-\\mathbb E[X]^2$"],
  ["A density can exceed 1; a probability cannot.","$P(D|+)$ is not $P(+|D)$."],
  "Lecture 03; Murphy §2.2 p.35, §2.3 p.44, §2.4 p.49, §2.5 p.53, §2.6 p.57.", "A1, D1–D10"),
 ("t3","Maximum likelihood","L04","High",
  "This is the heart of the course. Assignment 2, the midterm example and most long answers are this topic.",
  ["The recipe: likelihood → log → negate → loss. Loss = negative log-likelihood.",
   "Gaussian MLE: sample mean and the $1/n$ variance. Bernoulli MLE: fraction of ones.",
   "Multivariate Gaussian MLE: mean vector and covariance $\\frac1n\\sum(\\mathbf x-\\boldsymbol\\mu)(\\mathbf x-\\boldsymbol\\mu)^\\top$ (Assignment 1).",
   "Keep every term that contains a parameter. Drop only true constants."],
  ["$\\hat\\mu=\\frac1n\\sum x_i$, $\\hat\\sigma^2=\\frac1n\\sum(x_i-\\hat\\mu)^2$","$J=-\\sum_i\\log p(y_i|x_i;\\boldsymbol\\theta)$"],
  ["Dropping $\\frac12\\log\\sigma^2$ when $\\sigma^2$ depends on the input (the handwritten midterm example does this).","Calling $\\sigma^2$ the standard deviation."],
  "Lecture 04; Murphy §4.2 p.107, §4.2.3 p.110, §4.2.5 p.111, §4.2.6 p.112.", "A1–A12, B1, D2–D6, D10"),
 ("t4","Linear algebra","L06","Medium",
  "Needed for normal equations, covariance, Cholesky sampling and gradient shapes.",
  ["Matrix–vector and matrix–matrix products, transpose, inverse, norms.",
   "Positive definite matrices; Cholesky $\\Sigma=\\mathbf L\\mathbf L^\\top$ and sampling $\\boldsymbol\\mu+\\mathbf L\\mathbf z$.",
   "Eigenvalues and condition number $\\lambda_{\\max}/\\lambda_{\\min}$; least squares $\\hat{\\boldsymbol\\theta}=(\\mathbf X^\\top\\mathbf X)^{-1}\\mathbf X^\\top\\mathbf y$."],
  ["$(\\mathbf{AB})^\\top=\\mathbf B^\\top\\mathbf A^\\top$","$\\Sigma=\\mathbf L\\mathbf L^\\top$"],
  ["$\\mathbf X^\\top\\mathbf X$ is singular when features are dependent or $D<n$.","Cholesky $\\mathbf L$ is lower triangular, so multiply $\\mathbf L\\mathbf z$, not $\\mathbf z\\mathbf L$."],
  "Lecture 06; Murphy §7.6.3 p.266 (Cholesky), §7.7.3 p.268 (least squares), §7.8 p.269.", "B1–B9, E3–E10"),
 ("t5","Linear regression","L07","High",
  "Normal label, identity activation. Appears directly and as the model behind the others.",
  ["$y=\\boldsymbol\\theta^\\top\\boldsymbol\\phi(x)+\\varepsilon$, $\\varepsilon\\sim\\mathcal N(0,\\sigma^2)$ gives MSE as the loss.",
   "Closed-form solution and gradient $\\frac2D\\sum(\\boldsymbol\\theta^\\top\\mathbf x_i-y_i)\\mathbf x_i$.",
   "Basis functions $\\boldsymbol\\phi$: polynomial features, and how too many cause overfitting."],
  ["$\\nabla\\text{MSE}=\\frac2D\\mathbf X^\\top(\\mathbf X\\boldsymbol\\theta-\\mathbf y)$"],
  ["Mixing $\\frac1D$ and $\\frac12$ conventions: the minimizer is the same, the gradient differs by a factor."],
  "Lecture 07; Murphy §4.2.7 p.114, §11.2 p.371.", "E1–E6, A8"),
 ("t6","Logistic and softmax regression","L08, L09","High",
  "Bernoulli and Categorical labels. The gradient is again (prediction − truth) × input.",
  ["Sigmoid $\\sigma(z)=\\frac1{1+e^{-z}}$, decision boundary $\\boldsymbol\\theta^\\top\\mathbf x=0$.",
   "Binary cross-entropy and its gradient; softmax and categorical cross-entropy.",
   "Precision, recall, F1, accuracy, false-positive rate from a confusion matrix."],
  ["$\\nabla J=\\frac1D\\sum(\\sigma(\\boldsymbol\\theta^\\top\\mathbf x_i)-y_i)\\mathbf x_i$","$\\text{softmax}(\\mathbf z)_k=\\frac{e^{z_k}}{\\sum_je^{z_j}}$","$F_1=\\frac{2PR}{P+R}$"],
  ["The slide’s $F_\\beta$ formula has a sign error: it is $(1+\\beta^2)\\frac{PR}{\\beta^2P+R}$.","Fallout is $\\frac{FP}{FP+TN}$."],
  "Lectures 08–09; Murphy §2.4.2 p.50, §2.4.3 p.52, §5.1.2 p.169, §2.5.2 p.54, §10.3 p.350.", "F1–F10, A7, A9"),
 ("t7","Generalization and experiments","L10, L11, L12","Medium",
  "Train/validation/test, bias–variance, cross-validation, error bars, entropy.",
  ["Overfitting vs underfitting from train and validation error. The test set is used once.",
   "k-fold cross-validation: $k$ fits, each on $(k-1)/k$ of the data.",
   "Standard error of the mean $\\hat\\sigma/\\sqrt n$; about 95% interval $\\pm1.96$ SEM.",
   "Entropy $H=-\\sum p\\log p$, cross-entropy."],
  ["$\\text{SEM}=\\hat\\sigma/\\sqrt n$","$H(p)=-\\sum_xp(x)\\log p(x)$"],
  ["Touching ±1-SEM bars are not a 5% test.","Do not tune on the test set."],
  "Lectures 10–12; Murphy §1.2.3 p.12, §4.5.4 p.124, §4.5.5 p.125, §2.8.6 p.72, §6.1 p.207.", "G1–G10"),
 ("t8","Bayes, MAP and regularization","L13, L14","High",
  "A prior adds a penalty to the loss. Likely to be asked as ‘which prior gives which penalty’.",
  ["MAP = maximize likelihood × prior = minimize loss $-\\log p(\\boldsymbol\\theta)$ added.",
   "Gaussian prior → ridge ($L_2$): shrinks. Laplace prior → lasso ($L_1$): sets weights exactly to 0.",
   "Beta prior on a coin: $\\theta_{\\text{MAP}}=\\frac{H+\\alpha-1}{D+\\alpha+\\beta-2}$.",
   "Ridge: $(\\mathbf X^\\top\\mathbf X+\\lambda\\mathbf I)^{-1}\\mathbf X^\\top\\mathbf y$. Lasso: soft thresholding."],
  ["$J=\\text{loss}+\\lambda\\|\\boldsymbol\\theta\\|_2^2$ (ridge)","$\\text{soft}(c,t)=\\text{sign}(c)\\max(|c|-t,0)$"],
  ["The data term in the ridge loss must be squared: $\\|\\mathbf y-\\mathbf X\\boldsymbol\\theta\\|_2^2$.","Do not regularize the bias unless told to."],
  "Lectures 13–14; Murphy §2.3 p.44, §2.7.4 p.63, §4.5.1 p.121, §11.3 p.381, §11.4 p.385.", "H1–H9, D2"),
 ("t9","Neural networks","L15","Medium",
  "Learned basis functions. Expect parameter counts, a forward pass and the reasons for each design choice.",
  ["Perceptron rule and why XOR needs a hidden layer.",
   "MLP $\\hat{\\mathbf y}=\\mathbf g(\\mathbf W\\mathbf h(\\mathbf V\\mathbf x))$; parameters of $D\\to M\\to C$: $M(D+1)+C(M+1)$.",
   "Activations (sigmoid, tanh, ReLU) and why they must be non-linear; output activation matches the label.",
   "Backpropagation as the chain rule; vanishing gradients; dropout, weight decay, early stopping."],
  ["$\\sigma'=\\sigma(1-\\sigma)\\le\\frac14$","$\\tanh x=2\\sigma(2x)-1$"],
  ["At test time dropout multiplies by the keep probability $1-p$.","Without non-linear activations, many layers collapse to one linear map."],
  "Lecture 15; Murphy §13.2 p.426, §13.3 p.438, §13.4 p.446.", "I1–I10"),
 (r"t10",r"Sequence models",r"L16, L17",r"Medium",
  r"Skipped on the midterm, so expect at least one question: Markov chains, marginalizing a hidden state, an RNN step, attention weights, and how LLMs are trained.",
  [r"Markov chain $p(X_{t+1}|X_{0:t})=p(X_{t+1}|X_t)$; propagate a distribution with the transition matrix; a DFA state as memory.",
   r"Hidden-state model $p(x_{0:T})=\sum_{h}\prod_tp(x_t|h_t)p(h_t|h_{t-1},x_{t-1})$; marginalize $p(y)=\sum_hp(y|h)p(h)$; the forward algorithm.",
   r"RNN $\mathbf h_t=b(\mathbf W_{xh}\mathbf x_t+\mathbf W_{hh}\mathbf h_{t-1})$, $\mathbf y_t=a(\mathbf W_{hy}\mathbf h_t)$; shared weights; vec2seq, seq2vec, seq2seq; vanishing/exploding gradients; GRU and LSTM.",
   r"Attention $\mathbf z_t=\sum_i\text{softmax}(\mathbf q_t^\top\mathbf k_i/\sqrt d)\,\mathbf v_i$; causal mask; positional encodings; quadratic cost; LLM = MLE with teacher forcing, autoregressive rollout, perplexity."],
  [r"$p(y)=\sum_hp(y|h)p(h)$",r"$\mathbf z_t=\sum_i\mathbf p_t(i)\mathbf v_i$, $\mathbf p_t=\text{softmax}(\mathbf q_t^\top\mathbf k_i/\sqrt d)$",r"$\text{perplexity}=\exp(\text{mean NLL})$"],
  [r"Forgetting to sum over hidden sequences (the largest single path is the Viterbi path, not the probability).",r"Each attention row sums to 1; the causal mask hides the future, not the past.",r"Shared RNN weights: the parameter count does not grow with sequence length."],
  r"Lectures 16–17; Murphy §3.6.1.2 p.101, §15.2 p.503 (RNNs), §15.4–15.5 (attention, transformers); Bishop §13.1–13.2.", r"K1–K13"),
 ("t11","Gradient descent","L18","High",
  "Concepts, hand-steps and derivations. Be ready to derive a gradient, read one and run an update.",
  ["Update $\\boldsymbol\\theta\\leftarrow\\boldsymbol\\theta-\\alpha\\nabla J$. Learning-rate behaviour: too high diverges, low is slow.",
   "Stable iff $\\alpha<2/\\lambda_{\\max}$ on a quadratic.",
   "Convex (bowl) vs non-convex; SGD with mini-batches, noise, Robbins–Monro; momentum; Adam.",
   "Autodiff: forward pass stores values, backward pass multiplies local derivatives."],
  ["$\\boldsymbol\\theta_{t+1}=\\boldsymbol\\theta_t-\\alpha\\nabla J$","$\\mathbf m_t=\\beta\\mathbf m_{t-1}+\\mathbf g_t$"],
  ["For convex functions the chord lies above the curve.","SGD’s gradient is unbiased but noisy; its noise falls as the batch grows."],
  "Lecture 18; Murphy §8.1.3 p.277, §8.2 p.282, §8.2.4 p.287, §8.4 p.292.", "J1–J10"),
 ("t12","The midterm example and sheet","L19","High",
  "Your professor’s own sample question. Know the loss and both gradients cold.",
  ["Normal with $\\mu=\\boldsymbol\\theta_1^\\top\\boldsymbol\\phi$ and $\\sigma^2=\\exp(\\boldsymbol\\theta_2^\\top\\boldsymbol\\phi)$.",
   "Correct loss $\\sum[\\frac12\\boldsymbol\\theta_2^\\top\\boldsymbol\\phi+\\frac{r^2}{2e}]$ with $r=y-\\boldsymbol\\theta_1^\\top\\boldsymbol\\phi$, $e=\\exp(\\boldsymbol\\theta_2^\\top\\boldsymbol\\phi)$.",
   "The sheet lists the seven distributions and seven identities. Learn where each is."],
  ["$\\nabla_{\\boldsymbol\\theta_1}=\\sum-\\frac re\\boldsymbol\\phi$","$\\nabla_{\\boldsymbol\\theta_2}=\\sum(\\frac12-\\frac{r^2}{2e})\\boldsymbol\\phi$"],
  ["The handwritten page drops the $\\frac12\\boldsymbol\\theta_2^\\top\\boldsymbol\\phi$ term. Keep it."],
  "Lecture 19 (midterm example and supplement); Murphy §13.2.4.4 p.433.", "A3, A6, A10"),
 (r"t13",r"Support vector machines and constraints",r"L20",r"High",
  r"New since the midterm. Expect margin and distance calculations, the constrained problem, a small Lagrangian, and slack/hinge reasoning.",
  [r"Projection $\mathbf a^\top\mathbf b=\|\mathbf a\|\|\mathbf b\|\cos\theta$; boundary $\mathbf x^\top\mathbf w=b$; signed distance $(\mathbf x^\top\mathbf w-b)/\|\mathbf w\|$; margin $2/\|\mathbf w\|$.",
   r"One constraint for both classes: $y_i(\mathbf x_i^\top\mathbf w-b)\ge1$ with $y_i\in\{-1,+1\}$; support vectors sit on the curbs.",
   r"Hard margin: $\min\tfrac12\|\mathbf w\|^2$ subject to the constraints. Lagrangian $\mathcal L=f+\lambda c$: set every partial, including $\partial\mathcal L/\partial\lambda$, to 0; KKT for inequalities.",
   r"Soft margin: $\min\tfrac12\|\mathbf w\|^2+C\sum\xi_i$; $\xi_i=\max(0,1-m_i)$ is the hinge loss; the role of $C$; SVM versus logistic regression."],
  [r"$d=\frac2{\|\mathbf w\|}$",r"$y_i(\mathbf x_i^\top\mathbf w-b)\ge1-\xi_i,\ \xi_i\ge0$",r"$\xi_i=\max(0,1-y_i(\mathbf x_i^\top\mathbf w-b))$"],
  [r"The lecture writes $\mathbf x^\top\mathbf w-b$ (minus $b$); the textbooks write $\mathbf w^\top\mathbf x+b$.",r"$\xi_i>1$ means misclassified; $0<\xi_i\le1$ means inside the street but correct.",r"Large $C$ = narrow street (variance); small $C$ = wide street (bias)."],
  r"Lecture 20; Murphy §17.3 (SVMs), §8.5 (constrained optimization); Bishop §7.1 p.326, §7.1.1 p.331, Appendix E p.707.", r"L1–L11"),
]

# ---------------- Questions ----------------
from questions import GROUPS, Q
BONUS=set()

# ---------------- Mock exam ----------------
MOCK_INTRO = r"""Sit this alone, on paper, with a simple calculator and only a reference sheet like the midterm’s (seven distributions and seven identities). 150 minutes, 100 marks. Cover the answers. Then mark yourself with the marking notes; 70 or more is a strong result."""
MOCK_MINUTES = 150

MOCK_MC = [
 (r"Which distribution suits ‘minutes until the next bus arrives’ (positive, not bounded)?", ["Poisson","Normal","Weibull","Binomial"], 2, "A positive duration: Weibull (the Poisson is for counts)."),
 (r"$\sigma^2=\exp(\boldsymbol\theta_2^\top\boldsymbol\phi)$ is used because", ["it is faster to compute","it keeps the variance positive","it makes the loss convex","it removes the bias"], 1, "exp maps any real number to a positive one."),
 (r"The gradient of the logistic loss for one point is", [r"$(y-\sigma(z))\mathbf x$",r"$(\sigma(z)-y)\mathbf x$",r"$\sigma(z)(1-\sigma(z))\mathbf x$",r"$2(z-y)\mathbf x$"], 1, "(prediction − truth) × input."),
 (r"Gradient descent on a quadratic with largest curvature 20 is stable iff $\alpha<$", ["0.05","0.1","0.2","2"], 1, r"$2/\lambda_{\max}=2/20=0.1$."),
 (r"A Laplace prior on the weights corresponds to", ["ridge","lasso","dropout","early stopping"], 1, r"$-\log$ of a Laplace density is $|\theta|/b$: an $L_1$ penalty."),
 (r"A network $3\to4\to2$ (with biases) has how many parameters?", ["20","26","28","32"], 1, r"$4(3+1)+2(4+1)=16+10=26$."),
 (r"Training error 1%, validation error 18%. The best first step is", ["a bigger model","more regularization or more data","lower the learning rate","remove the validation set"], 1, "A large gap is overfitting."),
 (r"7 heads in 10 flips with a Beta$(3,3)$ prior. The MAP estimate is", ["0.7","0.643","0.667","0.5"], 1, r"$(7+2)/(10+4)=9/14=0.643$."),
 (r"An SVM has $\mathbf w=(6,8)$. The margin (street width) is", ["0.1","0.2","2","10"], 1, r"$2/\|\mathbf w\|=2/10$."),
 (r"Which training points determine a hard-margin SVM’s boundary?", ["all of them equally","the points farthest from the boundary","the support vectors on the margin","the misclassified points"], 2, "Only points with $y(\\mathbf x^\\top\\mathbf w-b)=1$ (nonzero multipliers) matter."),
 (r"Increasing $C$ in a soft-margin SVM", ["widens the street","narrows the street and allows fewer violations","removes the constraints","turns it into logistic regression"], 1, "Slack becomes expensive, so the model fits the training data more tightly (more variance)."),
 (r"Each row of a self-attention matrix", ["sums to 1","sums to the sequence length","is one-hot","has entries that can be negative"], 0, "Each row is a softmax over the visible keys."),
 (r"The Markov assumption says", ["all states are equally likely","the future is independent of the past given the present","the sequence is i.i.d.","the transition matrix is symmetric"], 1, r"$p(X_{t+1}|X_{0:t})=p(X_{t+1}|X_t)$."),
 (r"Transformers need positional encodings because", ["softmax is unstable","attention by itself ignores token order","they have no hidden layers","they are trained by MAP"], 1, "A weighted sum of values does not change when the inputs are shuffled."),
 (r"An LLM is trained by", ["maximizing the likelihood of the next token (cross-entropy)","minimizing the hinge loss","k-means clustering","maximizing the margin"], 0, r"$\boldsymbol\theta^*=\arg\max p(\mathcal D|\boldsymbol\theta)$ with teacher forcing."),
]

MOCK_LONG = [
 ("Long answer 1 (20 marks): waiting times",
  r"""The time $y_i>0$ (in minutes) a customer waits depends on features $\boldsymbol\phi(x_i)$. Use the Weibull with known shape $k=1$, $f(y)=\frac1\lambda e^{-y/\lambda}$, and $\lambda=\exp(\boldsymbol\theta^\top\boldsymbol\phi(x))$. (a) Why this distribution and this activation? (3) (b) Derive the loss $J(\boldsymbol\theta)$, showing each transformation and naming the sheet identities (7). (c) Derive $\nabla_{\boldsymbol\theta}J$ (5). (d) For one point $\boldsymbol\phi=(1,1)$, $y=2$, $\boldsymbol\theta=(0,0)$, compute the gradient and one step with $\alpha=0.1$ (5).""",
  r"""<ol class="steps"><li>(a) Waiting times are positive and unbounded; the Weibull is for positive durations. exp keeps the scale $\lambda>0$ for any $\boldsymbol\theta$.</li>
<li>(b) $\log\prod_if(y_i)=\sum_i\log f(y_i)$ (identity 1) $=\sum_i[-\log\lambda_i-y_i/\lambda_i]$ (identities 1, 2).</li>
<li>With $\lambda_i=e^{z_i}$: $\log\lambda_i=z_i$ (identity 3) and $1/\lambda_i=e^{-z_i}$. Negate: $J=\sum_i[z_i+y_ie^{-z_i}]$. Nothing to drop.</li>
<li>(c) $\partial J/\partial z_i=1-y_ie^{-z_i}$ and $\partial z_i/\partial\boldsymbol\theta=\boldsymbol\phi_i$, so $\nabla J=\sum_i(1-y_i/\lambda_i)\boldsymbol\phi(x_i)$.</li>
<li>(d) $z=0$, $\lambda=1$, $1-2/1=-1$: gradient $(-1,-1)$. Step: $(0,0)-0.1(-1,-1)=(0.1,0.1)$.</li></ol>
<p><em>Marking:</em> −2 for a missing minus sign; −1 for not naming identities; −2 for a wrong sign in the update; full marks for (d) only with the correct gradient.</p>"""),
 ("Long answer 2 (17 marks): regularization and honest evaluation",
  r"""Fit $y=\theta x$ (no intercept) to $(1,1)$ and $(2,3)$. (a) Show that a Gaussian prior $\theta\sim\mathcal N(0,\tau^2)$ with Gaussian noise of variance $\sigma^2$ gives the ridge objective, and give $\lambda$ (5). (b) Compute $\hat\theta$ for $\lambda=0$ and $\lambda=1$ (4). (c) You try five values of $\lambda$ with 5-fold cross-validation. How many fits is that, and which data choose $\lambda$? (3) (d) The chosen model has validation errors with mean 2.0 and sd 0.4 over the 5 folds. Give the SEM and an approximate 95% interval (5).""",
  r"""<ol class="steps"><li>(a) MAP minimizes $-\log p(\mathcal D|\theta)-\log p(\theta)=\frac1{2\sigma^2}\sum(y_i-\theta x_i)^2+\frac{\theta^2}{2\tau^2}+$ const.</li>
<li>Multiply by $2\sigma^2$: $\sum(y_i-\theta x_i)^2+\lambda\theta^2$ with $\lambda=\sigma^2/\tau^2$.</li>
<li>(b) $\theta=\frac{\sum xy}{\sum x^2+\lambda}$ with $\sum xy=1+6=7$, $\sum x^2=1+4=5$: $\lambda=0$ gives $1.4$; $\lambda=1$ gives $7/6=1.167$.</li>
<li>(c) $5\times5=25$ fits. $\lambda$ is chosen by the average validation score over the folds; the test set is used once at the end.</li>
<li>(d) SEM $=0.4/\sqrt5=0.179$; interval $2.0\pm1.96(0.179)=2.0\pm0.35$.</li></ol>
<p><em>Marking:</em> (a) needs both the likelihood and the prior term; −2 for using the test set to choose $\lambda$.</p>"""),
 ("Long answer 3 (16 marks): a small neural network",
  r"""Input $\mathbf x=(2,1)$. Hidden layer $\mathbf W=\begin{bmatrix}1&-1\\0.5&1\end{bmatrix}$, $\mathbf b=(0,-1)$, ReLU. Output $\hat y=\mathbf w^\top\mathbf h+c$ with $\mathbf w=(1,3)$, $c=-1$. Loss $(\hat y-y)^2$ with $y=2$. (a) Count the parameters (2). (b) Forward pass: $\mathbf z$, $\mathbf h$, $\hat y$, loss (5). (c) Backpropagate to $\mathbf w$, $c$ and $\mathbf W$ (6). (d) One gradient step on $\mathbf w$ and $c$ with $\alpha=0.1$ (3).""",
  r"""<ol class="steps"><li>(a) $2(2+1)+1(2+1)=9$.</li>
<li>(b) $\mathbf z=\mathbf W\mathbf x+\mathbf b=(2-1+0,\ 1+1-1)=(1,1)$; $\mathbf h=(1,1)$; $\hat y=1+3-1=3$; loss $(3-2)^2=1$.</li>
<li>(c) $\partial L/\partial\hat y=2(\hat y-y)=2$. Output: $\partial L/\partial\mathbf w=2\mathbf h=(2,2)$, $\partial L/\partial c=2$.</li>
<li>Hidden: $\partial L/\partial\mathbf h=2\mathbf w=(2,6)$; both units active (ReLU slope 1), so $\partial L/\partial\mathbf z=(2,6)$.</li>
<li>$\partial L/\partial\mathbf W=\begin{bmatrix}2\\6\end{bmatrix}\mathbf x^\top=\begin{bmatrix}4&2\\12&6\end{bmatrix}$; $\partial L/\partial\mathbf b=(2,6)$.</li>
<li>(d) $\mathbf w=(1,3)-0.1(2,2)=(0.8,2.8)$; $c=-1-0.2=-1.2$.</li></ol>
<p><em>Marking:</em> −2 if the ReLU derivative is ignored; −1 per arithmetic slip.</p>"""),
 ("Long answer 4 (17 marks): support vector machine",
  r"""Negative points $(0,0),(1,0)$; positive points $(2,1),(3,2)$; rule $\mathbf x^\top\mathbf w-b$ with $\mathbf w=(1,1)$, $b=2$. (a) Check every hard-margin constraint and name the support vectors (5). (b) Give the margin and the signed distance of $(3,2)$ (3). (c) Explain why no other line has a wider margin for these points (3). (d) Two new positive points arrive: $(1.5,1)$ and $(0.5,0.5)$. Compute their slacks and the soft-margin objective with $C=1$; say which is misclassified (6).""",
  r"""<ol class="steps"><li>(a) $m=y(\mathbf x^\top\mathbf w-b)$: $(0,0)$: $-(0-2)=2$; $(1,0)$: $-(1-2)=1$; $(2,1)$: $3-2=1$; $(3,2)$: $5-2=3$. All $\ge1$ ✓. Support vectors (on the curbs): $(1,0)$ and $(2,1)$.</li>
<li>(b) $\|\mathbf w\|=\sqrt2$; margin $2/\sqrt2=1.414$. Signed distance of $(3,2)$: $3/\sqrt2=2.12$.</li>
<li>(c) The closest opposite points $(1,0)$ and $(2,1)$ are $\sqrt2$ apart along the direction $(1,1)$, and the street already has that width; no street can be wider than the gap between the classes.</li>
<li>(d) $(1.5,1)$: $m=2.5-2=0.5$, $\xi=0.5$ (inside the street, correct). $(0.5,0.5)$: $m=1-2=-1$, $\xi=2$ (wrong side: misclassified).</li>
<li>Objective $\tfrac12\|\mathbf w\|^2+C\sum\xi=\tfrac12\cdot2+(0.5+2)=3.5$ (the original points have $\xi=0$).</li></ol>
<p><em>Marking:</em> −2 if labels are used as 0/1 instead of ±1; −1 for a missing $\tfrac12$.</p>"""),
 ("Long answer 5 (15 marks): sequence models",
  r"""(a) A hidden state $h\in\{0,1\}$ has $p(h)=(0.5,0.5)$, $p(x{=}1|h{=}0)=0.1$, $p(x{=}1|h{=}1)=0.7$. Find $p(x{=}1)$ and $p(h{=}1|x{=}1)$ (4). (b) Attention with $d=2$: query $(1,1)$, keys $(1,0)$ and $(0,2)$, values $(4,0)$ and $(0,4)$. Compute the weights and the output (5). (c) A model gives the true tokens probabilities $0.5,0.4,0.8,0.25$. Compute the per-token loss and perplexity (3). (d) Explain teacher forcing in two sentences (3).""",
  r"""<ol class="steps"><li>(a) $p(x{=}1)=0.5(0.1)+0.5(0.7)=0.4$; $p(h{=}1|x{=}1)=0.35/0.4=0.875$.</li>
<li>(b) Scores $(1,2)/\sqrt2=(0.707,1.414)$; $e^{0.707}=2.028$, $e^{1.414}=4.113$; weights $(0.330,0.670)$; output $0.330(4,0)+0.670(0,4)=(1.321,2.679)$.</li>
<li>(c) NLLs $0.693,0.916,0.223,1.386$; total $3.219$; per token $0.805$; perplexity $e^{0.805}=2.24$.</li>
<li>(d) In training we know the whole target, so we feed the true previous tokens instead of the model’s own samples and compute all positions in one pass. A causal mask stops each position from seeing the token it must predict.</li></ol>
<p><em>Marking:</em> (b) −2 if the scores are not scaled by $\sqrt d$ (accept a stated alternative constant).</p>"""),
]

MISTAKES = [
 "Dropping a term that contains a parameter (the $\\frac12\\boldsymbol\\theta_2^\\top\\boldsymbol\\phi$ in the midterm example).",
 "Writing the loss without the minus sign, so you maximize instead of minimize.",
 "Forgetting $\\boldsymbol\\phi(x)$ in the gradient (the chain rule through $z=\\boldsymbol\\theta^\\top\\boldsymbol\\phi$).",
 "Choosing the Normal for counts or a Bernoulli for counts. Say why the others fail.",
 "Using the standard deviation where the variance belongs, or the reverse.",
 "Getting broadcasting wrong: $(D,)$ against $(D,1)$ gives $(D,D)$.",
 "Wrong sign in the update: it is $\\boldsymbol\\theta-\\alpha\\nabla J$.",
 "Dropout scaling: use the keep probability.",
 "SVM labels are $\\pm1$, not $0/1$; check every constraint with $y_i(\\mathbf x_i^\\top\\mathbf w-b)\\ge1$.",
 "Attention: forgetting the $\\sqrt d$ scaling or the softmax, so the weights do not sum to 1.",
]

# ---------------- Self-contained lessons (shown at the top of each topic card) ----------------
LESSONS = {
"t1": r"""<p>A <strong>tensor</strong> is a block of numbers with a <strong>shape</strong>. A list of $n$ numbers has shape $(n,)$; a table with $D$ rows and $n$ columns has shape $(D,n)$. In <code>X.mean(axis=0)</code> the axis you name <em>disappears</em>: rows are averaged away and you get one number per column, shape $(n,)$. <code>X.sum(axis=1)</code> gives one number per row, shape $(D,)$.</p>
<p><strong>Broadcasting</strong> lets NumPy combine different shapes. Line the shapes up from the <em>right</em>. Each pair of sizes must be equal or one of them 1; a missing leading size counts as 1. So $(6700,784)-(784,)$ works (the vector is applied to every row), $(3,1)+(1,4)$ gives $(3,4)$, and $(4,)+(3,)$ fails. Danger: $(D,)-(D,1)$ silently gives $(D,D)$.</p>
<p><strong>Matrix product:</strong> $(a,b)\,@\,(b,c)\to(a,c)$; the inner sizes must match. A whole loss over all data is one vectorized line, for example <code>((X @ theta - y)**2).mean()</code>. <strong>Boolean mask:</strong> <code>m = h &gt;= 5</code> is a True/False array; <code>h[m]</code> keeps the True entries; <code>m.sum()</code> counts them.</p>""",
"t2": r"""<p>A <strong>random variable</strong> takes values with probabilities. Discrete: a <strong>PMF</strong> $p(x)$ with $\sum p=1$. Continuous: a <strong>PDF</strong> $f(x)$ with area 1 (it can be above 1 at a point). Two variables are <strong>independent</strong> when $p(a,b)=p(a)p(b)$. Expectation: $\mathbb E[X]=\sum xp(x)$; variance: $\mathbb E[X^2]-\mathbb E[X]^2$.</p>
<p><strong>Bayes’ rule</strong> turns $p(b|a)$ into $p(a|b)$: $p(a|b)=\frac{p(b|a)p(a)}{p(b)}$. Example: prevalence 1%, test catches 90% of the sick, 5% false alarms. $P(+)=0.9(0.01)+0.05(0.99)=0.0585$, so $P(\text{sick}|+)=0.009/0.0585=0.154$.</p>
<p><strong>Which distribution?</strong> Two classes: Bernoulli. $n$ classes: Categorical. Successes out of $n$ tries: Binomial. Count with no upper limit: Poisson. Any real number: Normal. Positive duration: Weibull. A probability itself: Beta. The sheet gives each formula; your job is to <em>choose</em> and to read parameters.</p>""",
"t3": r"""<p><strong>Maximum likelihood</strong> picks the parameters that make the observed data most probable. Independent data: likelihood $=\prod_ip(y_i|x_i;\boldsymbol\theta)$. Take the log (a product becomes a sum, same maximizer), then negate: the <strong>loss is the negative log-likelihood</strong>.</p>
<p><em>Coin, 7 heads in 10:</em> $L(p)=p^7(1-p)^3$, $\log L=7\log p+3\log(1-p)$, derivative $\frac7p-\frac3{1-p}=0\Rightarrow\hat p=0.7$. <em>Normal:</em> the MLE of the mean is the sample mean and of the variance is $\frac1n\sum(x_i-\hat\mu)^2$ (data $2,4,6$: $\hat\mu=4$, $\hat\sigma^2=8/3$). <em>Multivariate Normal:</em> mean vector and $\Sigma=\frac1n\sum(\mathbf x-\hat{\boldsymbol\mu})(\mathbf x-\hat{\boldsymbol\mu})^\top$.</p>
<p>To make the model depend on the input, let the parameter be a function of the features: $\lambda=\exp(\boldsymbol\theta^\top\boldsymbol\phi)$, $p=\sigma(\boldsymbol\theta^\top\boldsymbol\phi)$, $\mu=\boldsymbol\theta^\top\boldsymbol\phi$. The activation keeps the parameter legal. Then follow the six-step template above. This is the skill the exam tests most.</p>""",
"t4": r"""<p>Matrix–vector product $\mathbf A\mathbf x$ combines columns of $\mathbf A$; $(\mathbf{AB})^\top=\mathbf B^\top\mathbf A^\top$. A symmetric matrix is <strong>positive definite</strong> if $\mathbf v^\top\mathbf A\mathbf v>0$ for every nonzero $\mathbf v$; covariance matrices (with $\varepsilon\mathbf I$ added) are. The <strong>Cholesky factor</strong> is the lower-triangular $\mathbf L$ with $\Sigma=\mathbf L\mathbf L^\top$. To sample: draw $\mathbf z\sim\mathcal N(\mathbf 0,\mathbf I)$ and return $\boldsymbol\mu+\mathbf L\mathbf z$ (for rows, <code>mean + Z @ L.T</code>).</p>
<p><em>Cholesky of $\begin{bmatrix}4&2\\2&3\end{bmatrix}$:</em> $L_{11}=2$, $L_{21}=1$, $L_{22}=\sqrt{3-1}=1.414$.</p>
<p><strong>Least squares:</strong> $\hat{\boldsymbol\theta}=(\mathbf X^\top\mathbf X)^{-1}\mathbf X^\top\mathbf y$. It fails when $\mathbf X^\top\mathbf X$ is singular (dependent features, or fewer points than features). The <strong>condition number</strong> $\lambda_{\max}/\lambda_{\min}$ of $\mathbf X^\top\mathbf X$ measures how stretched the loss is; large means slow, zig-zag gradient descent.</p>""",
"t5": r"""<p>Model: $y=\boldsymbol\theta^\top\boldsymbol\phi(x)+\text{noise}$, noise $\sim\mathcal N(0,\sigma^2)$. So $y|x\sim\mathcal N(\boldsymbol\theta^\top\boldsymbol\phi(x),\sigma^2)$ and the negative log-likelihood is the <strong>mean squared error</strong> $\frac1D\sum(y_i-\boldsymbol\theta^\top\boldsymbol\phi(x_i))^2$ (up to constants). The best $\boldsymbol\theta$ solves the normal equations or is found by gradient descent.</p>
<p><em>Fit by hand:</em> points $(1,2),(2,4),(3,5)$: slope $=S_{xy}/S_{xx}=3/2=1.5$, intercept $=\bar y-1.5\bar x=0.667$.</p>
<p><strong>Basis functions:</strong> $\boldsymbol\phi(x)=(1,x,x^2,\dots)$ lets a straight-line machine fit curves, because the model is still linear in $\boldsymbol\theta$. High degree fits the training points exactly but wiggles between them: overfitting.</p>""",
"t6": r"""<p><strong>Logistic regression</strong> (two classes): $P(y=1|x)=\sigma(\boldsymbol\theta^\top\mathbf x)$ with $\sigma(z)=1/(1+e^{-z})$. Loss: $-[y\log p+(1-y)\log(1-p)]$ (Bernoulli NLL). Decision boundary: $\boldsymbol\theta^\top\mathbf x=0$ (where $p=0.5$). With $\boldsymbol\theta=(-3,1)$ on $(1,h)$ the boundary is $h=3$ and $p(h=5)=\sigma(2)=0.881$.</p>
<p><strong>Softmax regression</strong> ($C$ classes): probabilities $\frac{e^{z_k}}{\sum_je^{z_j}}$, loss $-\log p_{\text{true}}$. Logits $(2,1,0)$ give $(0.665,0.245,0.090)$; true class 1 costs $0.408$.</p>
<p><strong>Metrics</strong> from TP, FP, FN, TN: accuracy $=\frac{TP+TN}{\text{all}}$, precision $=\frac{TP}{TP+FP}$, recall $=\frac{TP}{TP+FN}$, $F_1=\frac{2PR}{P+R}$, false-positive rate $=\frac{FP}{FP+TN}$. The ROC curve plots recall against FPR as the threshold varies.</p>""",
"t7": r"""<p><strong>Generalization</strong> means doing well on new data. Split data into train, validation and test. Fit on train, choose settings on validation, touch test once at the end. Small train error with big validation error: <strong>overfitting</strong> (high variance; fix with more data or regularization). Both errors large: <strong>underfitting</strong> (high bias; fix with a richer model).</p>
<p><strong>$k$-fold cross-validation:</strong> split into $k$ parts; $k$ times train on $k-1$ parts and score on the remaining one; average the scores.</p>
<p><strong>Error bars:</strong> the mean of $n$ runs has <strong>standard error</strong> $\hat\sigma/\sqrt n$ (CLT: the mean is approximately Normal with variance $\sigma^2/n$). About 95% interval: $\pm1.96$ SEM. Five runs, sd 5: SEM $=2.24$, interval $\pm4.4$.</p>
<p><strong>Entropy</strong> $H=-\sum p\log p$ measures uncertainty: $(0.5,0.25,0.25)$ has $1.5$ bits. Cross-entropy $-\sum p\log q$ is the classification loss.</p>""",
"t8": r"""<p>Bayes for parameters: posterior $\propto$ likelihood $\times$ prior. <strong>MAP</strong> maximizes it, which means minimizing $\text{loss}-\log p(\boldsymbol\theta)$: a prior becomes a <strong>penalty</strong>. A Gaussian prior gives <strong>ridge</strong>, $\text{loss}+\lambda\|\boldsymbol\theta\|_2^2$ (shrinks weights). A Laplace prior gives <strong>lasso</strong>, $\text{loss}+\lambda\|\boldsymbol\theta\|_1$ (sets some weights exactly to 0). Choose $\lambda$ on validation data.</p>
<p><em>Coin with a Beta$(\alpha,\beta)$ prior:</em> $\theta_{\text{MAP}}=\frac{H+\alpha-1}{D+\alpha+\beta-2}$. Seven heads in ten with Beta$(2,2)$: $8/12=0.667$, pulled from $0.7$ toward $0.5$.</p>
<p><em>Ridge in one dimension</em> (no intercept): $\theta=\frac{\sum xy}{\sum x^2+\lambda}$; with $\sum xy=25$, $\sum x^2=14$: $\lambda=0$ gives $1.786$, $\lambda=2$ gives $1.5625$. <em>Lasso:</em> soft threshold $\text{sign}(c)\max(|c|-t,0)$.</p>""",
"t9": r"""<p>A <strong>neural network</strong> learns its own basis functions: hidden unit $=$ activation of a linear combination. Forward pass for $\mathbf x=(1,2)$, $\mathbf W=\begin{bmatrix}1&-1\\0.5&0.5\end{bmatrix}$, $\mathbf b=(0,-1)$, ReLU, output $\mathbf w=(2,4)$, bias $0.5$: hidden $\mathbf z=(-1,0.5)\to(0,0.5)$; output $=4(0.5)+0.5=2.5$.</p>
<p><strong>Parameters</strong> of $D\to M\to C$: $M(D+1)+C(M+1)$ (e.g. $4\to5\to3$: $25+18=43$). <strong>Why non-linear activations?</strong> With identity activations the layers collapse into one linear map. <strong>Activations:</strong> sigmoid (max slope $\frac14$, so deep stacks have vanishing gradients), tanh, ReLU $\max(0,x)$. Pair the output with the label: identity+squared error, sigmoid+binary cross-entropy, softmax+cross-entropy.</p>
<p><strong>Perceptron:</strong> $\text{sign}(\mathbf w^\top\mathbf x+b)$; update $\mathbf w\leftarrow\mathbf w+y\mathbf x$ on mistakes; it cannot learn XOR (no line separates it), a hidden layer can. <strong>Regularization:</strong> weight decay (ridge), dropout (drop units in training, multiply by keep probability $1-p$ at test), early stopping (stop when validation error rises), data augmentation.</p>""",
"t10": r"""<p>A <strong>Markov chain</strong> forgets the past given the present: $p(X_{t+1}|X_{0:t})=p(X_{t+1}|X_t)$. With a transition matrix $\mathbf A$ (rows sum to 1) the distribution after $t$ steps is (start)$\,\mathbf A^t$. A DFA such as ‘even number of 1s’ shows the idea: one bit of state is a sufficient summary of the whole history.</p>
<p>A <strong>language model</strong> is $p(x_0,\dots,x_T)=p(x_0)\prod_tp(x_t|x_{0:t-1})$; the history grows, so we introduce a <strong>hidden state</strong> $h_t$ and <strong>marginalize</strong>: $p(y)=\sum_hp(y|h)p(h)$. <em>Example:</em> prior $(0.6,0.4)$, $p(\text{pass}|h)=(0.2,0.8)$ gives $p(\text{pass})=0.44$. The forward algorithm does the sum over all hidden paths in $TK^2$ steps.</p>
<p>An <strong>RNN</strong> makes $h_t$ deterministic and learned: $\mathbf h_t=b(\mathbf W_{xh}\mathbf x_t+\mathbf W_{hh}\mathbf h_{t-1})$, same weights every step. Backprop through time multiplies many factors: vanishing ($0.9^{50}=0.005$) or exploding ($1.1^{50}=117$); GRU and LSTM fix it. <strong>Attention</strong> replaces the single summary with a key–value store: weights $\text{softmax}(\mathbf q_t^\top\mathbf k_i/\sqrt d)$, output $\sum_i$ weight$\times\mathbf v_i$. A <strong>Transformer</strong> stacks attention heads (cost grows like $n^2$) and adds positional encodings. An <strong>LLM</strong> is maximum likelihood on next tokens, trained with teacher forcing and used by autoregressive rollout until $\langle\text{eos}\rangle$.</p>""",
"t11": r"""<p>To minimize a differentiable loss: $\boldsymbol\theta\leftarrow\boldsymbol\theta-\alpha\nabla J(\boldsymbol\theta)$. The gradient points uphill, so we step the other way; $\alpha$ is the learning rate. <em>Example</em> $J=(\theta-2)^2$ from $0$, $\alpha=0.25$: $1,\ 1.5,\ 1.75,\dots$ each step halves the distance.</p>
<p><strong>Learning rate:</strong> too high diverges, high oscillates, low is slow. On a quadratic it is stable iff $\alpha<2/\lambda_{\max}$ (the largest curvature). <strong>Convex</strong> (bowl; chord above the curve): one global minimum. MSE and logistic loss are convex; neural-network losses are not.</p>
<p><strong>SGD:</strong> use a random mini-batch of size $B$ for each step: cheaper, unbiased, noisy (noise falls as $B$ grows); step sizes need $\sum\alpha_t=\infty$, $\sum\alpha_t^2<\infty$. <strong>Momentum</strong> $\mathbf m_t=\beta\mathbf m_{t-1}+\mathbf g_t$ smooths the zig-zag. Adam, RMSprop, Adagrad give each weight its own step. Autodiff (PyTorch): forward pass stores values, backward pass multiplies local derivatives.</p>""",
"t12": r"""<p>The sample question applies the recipe to a plot whose noise grows with $x$. <strong>Distribution:</strong> Normal (real label). <strong>Parameterize both parameters:</strong> $\mu=\boldsymbol\theta_1^\top\boldsymbol\phi(x)$, $\sigma^2=\exp(\boldsymbol\theta_2^\top\boldsymbol\phi(x))$ (exp keeps the variance positive). <strong>Log-density:</strong> $-\frac12\log2\pi-\frac12\log\sigma^2-\frac{(y-\mu)^2}{2\sigma^2}$, and $\log\sigma^2=\boldsymbol\theta_2^\top\boldsymbol\phi$ by identity 3. <strong>Loss:</strong> $\sum_i\big[\frac12\boldsymbol\theta_2^\top\boldsymbol\phi(x_i)+\frac{(y_i-\boldsymbol\theta_1^\top\boldsymbol\phi(x_i))^2}{2\exp(\boldsymbol\theta_2^\top\boldsymbol\phi(x_i))}\big]$.</p>
<p>The handwritten solution leaves out the first term. It must stay because it depends on $\boldsymbol\theta_2$; without it the variance grows without limit. The same method works for any distribution on the sheet.</p>""",
"t13": r"""<p>Labels are $y\in\{-1,+1\}$ and the score is $\mathbf x^\top\mathbf w-b$. The <strong>decision boundary</strong> is $\mathbf x^\top\mathbf w=b$, the two <strong>curbs</strong> are $\mathbf x^\top\mathbf w=b\pm1$, and the <strong>signed distance</strong> of a point is $(\mathbf x^\top\mathbf w-b)/\|\mathbf w\|$ (the projection idea: $\mathbf a^\top\mathbf b/\|\mathbf b\|$ is the shadow of $\mathbf a$ on $\mathbf b$). The street is $d=2/\|\mathbf w\|$ wide, so <em>a wider street means a smaller $\|\mathbf w\|$</em>.</p>
<p><strong>Hard-margin SVM:</strong> $\min\tfrac12\|\mathbf w\|^2$ such that $y_i(\mathbf x_i^\top\mathbf w-b)\ge1$ for all $i$. Points with equality are the <strong>support vectors</strong>; only they fix the line. <em>Example:</em> 1-D points $3$ (fail) and $4$ (pass) give $w=2$, $b=7$, boundary $3.5$, margin $1$.</p>
<p><strong>Constraints</strong> are handled with a <strong>Lagrangian</strong> $\mathcal L=f+\lambda c$: set $\nabla_{\boldsymbol\theta}\mathcal L=\mathbf 0$ and $\partial\mathcal L/\partial\lambda=c=0$ (equality case; inequalities use $\lambda\ge0$ and KKT). If the classes overlap the hard problem has no solution, so add <strong>slack</strong>: $\min\tfrac12\|\mathbf w\|^2+C\sum\xi_i$ with $y_i(\mathbf x_i^\top\mathbf w-b)\ge1-\xi_i$, $\xi_i\ge0$. At the optimum $\xi_i=\max(0,1-m_i)$, the <strong>hinge loss</strong>: regularizer + loss, like ridge with $\lambda=1/(2C)$. Large $C$: narrow street, few violations; small $C$: wide street.</p>""",
}

MEMORIZE = [
 ("Not on the sheet: know these", [
  r"Sigmoid $\sigma(z)=\frac1{1+e^{-z}}$; softmax $\frac{e^{z_k}}{\sum_je^{z_j}}$.",
  r"MSE $=\frac1D\sum(y_i-\hat y_i)^2$; least squares $\hat{\boldsymbol\theta}=(\mathbf X^\top\mathbf X)^{-1}\mathbf X^\top\mathbf y$.",
  r"Loss $=-\sum\log p(y_i|x_i;\boldsymbol\theta)$; Bernoulli NLL $-[y\log p+(1-y)\log(1-p)]$.",
  r"Full Normal density (the sheet only gives $\propto$): $\frac1{\sqrt{2\pi\sigma^2}}\exp\!\big(-\frac{(y-\mu)^2}{2\sigma^2}\big)$. The constant $\frac1{\sqrt{2\pi\sigma^2}}$ must be kept whenever $\sigma^2$ depends on the input; its log is $-\frac12\log2\pi-\frac12\log\sigma^2$.",
  r"Gaussian MLE: mean, and variance with $1/n$; covariance $\frac1n\sum(\mathbf x-\hat{\boldsymbol\mu})(\mathbf x-\hat{\boldsymbol\mu})^\top$.",
  r"Precision $\frac{TP}{TP+FP}$, recall $\frac{TP}{TP+FN}$, $F_1=\frac{2PR}{P+R}$, FPR $\frac{FP}{FP+TN}$, accuracy.",
  r"SEM $=\hat\sigma/\sqrt n$; 95% $\approx\pm1.96$ SEM; entropy $-\sum p\log p$.",
  r"MAP: $\text{loss}-\log p(\boldsymbol\theta)$; Gaussian prior $\to$ ridge, Laplace $\to$ lasso; Beta MAP $\frac{H+\alpha-1}{D+\alpha+\beta-2}$.",
  r"Ridge $(\mathbf X^\top\mathbf X+\lambda\mathbf I)^{-1}\mathbf X^\top\mathbf y$; soft threshold $\text{sign}(c)\max(|c|-t,0)$.",
  r"Network parameters $M(D+1)+C(M+1)$; dropout test scale $=1-p$; $\sigma'\le\frac14$.",
  r"Gradient descent $\boldsymbol\theta\leftarrow\boldsymbol\theta-\alpha\nabla J$; stable iff $\alpha<2/\lambda_{\max}$; momentum $\mathbf m_t=\beta\mathbf m_{t-1}+\mathbf g_t$.",
  r"Cholesky $\Sigma=\mathbf L\mathbf L^\top$, sample $\boldsymbol\mu+\mathbf L\mathbf z$.",
  r"Markov chain: distribution after $t$ steps $=$ start $\cdot\mathbf A^t$; marginal $p(y)=\sum_hp(y|h)p(h)$; RNN $\mathbf h_t=b(\mathbf W_{xh}\mathbf x_t+\mathbf W_{hh}\mathbf h_{t-1})$.",
  r"Attention $\text{softmax}(\mathbf q^\top\mathbf k_i/\sqrt d)$, output $\sum_ip_i\mathbf v_i$; perplexity $=\exp(\text{mean NLL})$; LLM $=$ maximum likelihood.",
  r"SVM: margin $2/\|\mathbf w\|$, signed distance $(\mathbf x^\top\mathbf w-b)/\|\mathbf w\|$, constraint $y_i(\mathbf x_i^\top\mathbf w-b)\ge1$, soft margin $\tfrac12\|\mathbf w\|^2+C\sum\xi_i$, hinge $\max(0,1-m)$.",
  r"Lagrangian $\mathcal L=f+\lambda c$; set every partial (including $\partial\mathcal L/\partial\lambda$) to 0.",
 ]),
 ("Probably provided (if the final gives a sheet like the midterm’s): do not memorize", [
  "The seven distributions (Normal, Bernoulli, Beta, Categorical, Binomial, Poisson, Weibull) with their supports, PMF/PDF and parameters.",
  r"$\log ab=\log a+\log b$, $\log a^b=b\log a$, $\log\exp x=x$, sums and products, $p(a,b)=p(a|b)p(b)$, Bayes’ rule.",
 ]),
]
