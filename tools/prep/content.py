# -*- coding: utf-8 -*-
# Content for the midterm prep site. $...$ = math (pre-rendered by KaTeX).

DAYS = [
 ("Thu Nov 12 · tonight", "2 hours", "Learn the recipe, then do it blind", [
  "Read “The answer template” below and the worked Poisson example (10 min).",
  "Close the page. On paper, redo steps 1–4 of the template for a count label with exp activation. This is Steps 1–4 of Assignment 2 (30 min).",
  "Do the midterm example (Lecture 19): Normal with mean and variance both depending on x. Write the loss. Gradients are bonus. Check against Practice A3 (30 min).",
  "Practice group A, all of it (40 min). Mark each question Got it or Review.",
 ]),
 ("Fri Nov 13", "4–5 hours", "Fill the foundations", [
  "Morning: topics 1–4 (tensors, probability, MLE, linear algebra). Read the card, then do practice groups B, C, D.",
  "Afternoon: topics 5–8 (regression, logistic, softmax, generalization). Practice groups E, F, G.",
  "Evening: redo every question you marked Review. Then meet your partner for 30 minutes and quiz each other on the formula list.",
 ]),
 ("Sat Nov 14", "4–5 hours", "Close the gaps and rehearse", [
  "Morning: topics 8–11 (MAP, ridge/lasso, neural networks, gradient descent). Practice groups H, I, J.",
  "Midday: sit the Mock exam below with your partner as a team of two. One calculator, paper only, 50 minutes. Use the reference sheet only.",
  "Afternoon: mark it together using the marking notes. Every lost mark goes on a list. Fix that list.",
 ]),
 ("Sun Nov 15", "1–2 hours", "Light review, then rest", [
  "Re-derive the Poisson and the Normal-with-variance losses one more time from a blank page.",
  "Read the common mistakes list at the end of each topic card. Skim the reference sheet so you know where things are.",
  "Pack: pencil, eraser, simple calculator with fresh batteries. Sleep.",
 ]),
]

TEMPLATE_STEPS = [
 ("Pick the distribution", "Look at the label $y$. Real number: Normal. Two classes: Bernoulli. Many classes: Categorical. Count with no upper limit: Poisson. Successes out of $n$ tries: Binomial. Positive duration: Weibull. Write at most one sentence of reason (whole numbers? never negative? bounded?)."),
 ("Parameterize with the input", "Name the distribution's parameter(s) and make each a function of the features. The activation $a$ must keep the parameter in its allowed range: sigmoid for $p\\in[0,1]$, exp for a rate or variance $>0$, identity for a mean. Example: $\\lambda=a(\\boldsymbol\\theta^\\top\\boldsymbol\\phi(x))=\\exp(\\boldsymbol\\theta^\\top\\boldsymbol\\phi(x))$."),
 ("Write the likelihood and take the log", "Independent points: $p(\\mathcal D|\\boldsymbol\\theta)=\\prod_i p(y_i|x_i;\\boldsymbol\\theta)$. Then $\\log$ turns the product into a sum (sheet identity 1). Use sheet identities 2 and 3 to simplify, and keep every term that contains $\\boldsymbol\\theta$."),
 ("Negate to get the loss", "$J(\\boldsymbol\\theta)=-\\sum_i\\log p(y_i|x_i;\\boldsymbol\\theta)$. You may drop terms that do not contain $\\boldsymbol\\theta$ (like $\\log y!$ or $\\tfrac12\\log2\\pi$). Say that you dropped them."),
 ("Differentiate (bonus: not required)", "Your professor said you will not be asked to derive gradients. Learn it to understand step 6, but spend your time on steps 1–4. Chain rule: first the derivative with respect to the parameter ($\\lambda$ or $z=\\boldsymbol\\theta^\\top\\boldsymbol\\phi$), then multiply by $\\boldsymbol\\phi$. Check the result has the shape (prediction − truth) × input."),
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
  "Lecture 02; Murphy §7.2 p.242 for the matrix-product ideas.", "C1–C5"),
 ("t2","Probability and distributions","L03","High",
  "Every question starts by choosing or reading a distribution. The reference sheet lists seven.",
  ["Random variable, PMF vs PDF, independence, expectation and variance.",
   "Bayes' rule and the chain rule $p(a,b)=p(a|b)p(b)$ (sheet identities 6 and 7).",
   "Bernoulli, Categorical, Binomial, Poisson, Normal, Beta, Weibull: support and parameters, and which label type each fits."],
  ["$p(a|b)=\\dfrac{p(b|a)p(a)}{p(b)}$","$\\mathbb E[X]=\\sum xp(x)$, $\\text{Var}(X)=\\mathbb E[X^2]-\\mathbb E[X]^2$"],
  ["A density can exceed 1; a probability cannot.","$P(D|+)$ is not $P(+|D)$."],
  "Lecture 03; Murphy §2.2 p.35, §2.3 p.44, §2.4 p.49, §2.5 p.53, §2.6 p.57.", "A1, D1, D3"),
 ("t3","Maximum likelihood","L04","High",
  "This is the heart of the exam. Assignment 2 Steps 1–4 and the midterm example are this topic.",
  ["The recipe: likelihood → log → negate → loss. Loss = negative log-likelihood.",
   "Gaussian MLE: sample mean and the $1/n$ variance. Bernoulli MLE: fraction of ones.",
   "Multivariate Gaussian MLE: mean vector and covariance $\\frac1n\\sum(\\mathbf x-\\boldsymbol\\mu)(\\mathbf x-\\boldsymbol\\mu)^\\top$ (Assignment 1).",
   "Keep every term that contains a parameter. Drop only true constants."],
  ["$\\hat\\mu=\\frac1n\\sum x_i$, $\\hat\\sigma^2=\\frac1n\\sum(x_i-\\hat\\mu)^2$","$J=-\\sum_i\\log p(y_i|x_i;\\boldsymbol\\theta)$"],
  ["Dropping $\\frac12\\log\\sigma^2$ when $\\sigma^2$ depends on the input (the handwritten midterm example does this).","Calling $\\sigma^2$ the standard deviation."],
  "Lecture 04; Murphy §4.2 p.107, §4.2.3 p.110, §4.2.5 p.111, §4.2.6 p.112.", "A1–A6, B1–B2, D2, D4"),
 ("t4","Linear algebra","L06","Medium",
  "Needed for normal equations, covariance, Cholesky sampling and gradient shapes.",
  ["Matrix–vector and matrix–matrix products, transpose, inverse, norms.",
   "Positive definite matrices; Cholesky $\\Sigma=\\mathbf L\\mathbf L^\\top$ and sampling $\\boldsymbol\\mu+\\mathbf L\\mathbf z$.",
   "Eigenvalues and condition number $\\lambda_{\\max}/\\lambda_{\\min}$; least squares $\\hat{\\boldsymbol\\theta}=(\\mathbf X^\\top\\mathbf X)^{-1}\\mathbf X^\\top\\mathbf y$."],
  ["$(\\mathbf{AB})^\\top=\\mathbf B^\\top\\mathbf A^\\top$","$\\Sigma=\\mathbf L\\mathbf L^\\top$"],
  ["$\\mathbf X^\\top\\mathbf X$ is singular when features are dependent or $D<n$.","Cholesky $\\mathbf L$ is lower triangular, so multiply $\\mathbf L\\mathbf z$, not $\\mathbf z\\mathbf L$."],
  "Lecture 06; Murphy §7.6.3 p.266 (Cholesky), §7.7.3 p.268 (least squares), §7.8 p.269.", "B3, E1–E3"),
 ("t5","Linear regression","L07","High",
  "Normal label, identity activation. Appears directly and as the model behind the others.",
  ["$y=\\boldsymbol\\theta^\\top\\boldsymbol\\phi(x)+\\varepsilon$, $\\varepsilon\\sim\\mathcal N(0,\\sigma^2)$ gives MSE as the loss.",
   "Closed-form solution and gradient $\\frac2D\\sum(\\boldsymbol\\theta^\\top\\mathbf x_i-y_i)\\mathbf x_i$.",
   "Basis functions $\\boldsymbol\\phi$: polynomial features, and how too many cause overfitting."],
  ["$\\nabla\\text{MSE}=\\frac2D\\mathbf X^\\top(\\mathbf X\\boldsymbol\\theta-\\mathbf y)$"],
  ["Mixing $\\frac1D$ and $\\frac12$ conventions: the minimizer is the same, the gradient differs by a factor."],
  "Lecture 07; Murphy §4.2.7 p.114, §11.2 p.371.", "E1–E3"),
 ("t6","Logistic and softmax regression","L08, L09","High",
  "Bernoulli and Categorical labels. The gradient is again (prediction − truth) × input.",
  ["Sigmoid $\\sigma(z)=\\frac1{1+e^{-z}}$, decision boundary $\\boldsymbol\\theta^\\top\\mathbf x=0$.",
   "Binary cross-entropy and its gradient; softmax and categorical cross-entropy.",
   "Precision, recall, F1, accuracy, false-positive rate from a confusion matrix."],
  ["$\\nabla J=\\frac1D\\sum(\\sigma(\\boldsymbol\\theta^\\top\\mathbf x_i)-y_i)\\mathbf x_i$","$\\text{softmax}(\\mathbf z)_k=\\frac{e^{z_k}}{\\sum_je^{z_j}}$","$F_1=\\frac{2PR}{P+R}$"],
  ["The slide’s $F_\\beta$ formula has a sign error: it is $(1+\\beta^2)\\frac{PR}{\\beta^2P+R}$.","Fallout is $\\frac{FP}{FP+TN}$."],
  "Lectures 08–09; Murphy §2.4.2 p.50, §2.4.3 p.52, §5.1.2 p.169, §2.5.2 p.54, §10.3 p.350.", "F1–F4"),
 ("t7","Generalization and experiments","L10, L11, L12","Medium",
  "Train/validation/test, bias–variance, cross-validation, error bars, entropy.",
  ["Overfitting vs underfitting from train and validation error. The test set is used once.",
   "k-fold cross-validation: $k$ fits, each on $(k-1)/k$ of the data.",
   "Standard error of the mean $\\hat\\sigma/\\sqrt n$; about 95% interval $\\pm1.96$ SEM.",
   "Entropy $H=-\\sum p\\log p$, cross-entropy."],
  ["$\\text{SEM}=\\hat\\sigma/\\sqrt n$","$H(p)=-\\sum_xp(x)\\log p(x)$"],
  ["Touching ±1-SEM bars are not a 5% test.","Do not tune on the test set."],
  "Lectures 10–12; Murphy §1.2.3 p.12, §4.5.4 p.124, §4.5.5 p.125, §2.8.6 p.72, §6.1 p.207.", "G1–G4"),
 ("t8","Bayes, MAP and regularization","L13, L14","High",
  "A prior adds a penalty to the loss. Likely to be asked as ‘which prior gives which penalty’.",
  ["MAP = maximize likelihood × prior = minimize loss $-\\log p(\\boldsymbol\\theta)$ added.",
   "Gaussian prior → ridge ($L_2$): shrinks. Laplace prior → lasso ($L_1$): sets weights exactly to 0.",
   "Beta prior on a coin: $\\theta_{\\text{MAP}}=\\frac{H+\\alpha-1}{D+\\alpha+\\beta-2}$.",
   "Ridge: $(\\mathbf X^\\top\\mathbf X+\\lambda\\mathbf I)^{-1}\\mathbf X^\\top\\mathbf y$. Lasso: soft thresholding."],
  ["$J=\\text{loss}+\\lambda\\|\\boldsymbol\\theta\\|_2^2$ (ridge)","$\\text{soft}(c,t)=\\text{sign}(c)\\max(|c|-t,0)$"],
  ["The data term in the ridge loss must be squared: $\\|\\mathbf y-\\mathbf X\\boldsymbol\\theta\\|_2^2$.","Do not regularize the bias unless told to."],
  "Lectures 13–14; Murphy §2.3 p.44, §2.7.4 p.63, §4.5.1 p.121, §11.3 p.381, §11.4 p.385.", "D2, H1–H3"),
 ("t9","Neural networks","L15","Medium",
  "Learned basis functions. Expect parameter counts, a forward pass and the reasons for each design choice.",
  ["Perceptron rule and why XOR needs a hidden layer.",
   "MLP $\\hat{\\mathbf y}=\\mathbf g(\\mathbf W\\mathbf h(\\mathbf V\\mathbf x))$; parameters of $D\\to M\\to C$: $M(D+1)+C(M+1)$.",
   "Activations (sigmoid, tanh, ReLU) and why they must be non-linear; output activation matches the label.",
   "Backpropagation as the chain rule; vanishing gradients; dropout, weight decay, early stopping."],
  ["$\\sigma'=\\sigma(1-\\sigma)\\le\\frac14$","$\\tanh x=2\\sigma(2x)-1$"],
  ["At test time dropout multiplies by the keep probability $1-p$.","Without non-linear activations, many layers collapse to one linear map."],
  "Lecture 15; Murphy §13.2 p.426, §13.3 p.438, §13.4 p.446.", "I1–I4"),
 ("t11","Gradient descent","L18","High",
  "Concepts and short hand-steps. You will not be asked to derive gradients, but you should read a given gradient and run an update.",
  ["Update $\\boldsymbol\\theta\\leftarrow\\boldsymbol\\theta-\\alpha\\nabla J$. Learning-rate behaviour: too high diverges, low is slow.",
   "Stable iff $\\alpha<2/\\lambda_{\\max}$ on a quadratic.",
   "Convex (bowl) vs non-convex; SGD with mini-batches, noise, Robbins–Monro; momentum; Adam.",
   "Autodiff: forward pass stores values, backward pass multiplies local derivatives."],
  ["$\\boldsymbol\\theta_{t+1}=\\boldsymbol\\theta_t-\\alpha\\nabla J$","$\\mathbf m_t=\\beta\\mathbf m_{t-1}+\\mathbf g_t$"],
  ["For convex functions the chord lies above the curve.","SGD’s gradient is unbiased but noisy; its noise falls as the batch grows."],
  "Lecture 18; Murphy §8.1.3 p.277, §8.2 p.282, §8.2.4 p.287, §8.4 p.292.", "J1–J5"),
 ("t12","The midterm example and sheet","L19","High",
  "Your professor’s own sample. Know the loss cold; the gradients are bonus.",
  ["Normal with $\\mu=\\boldsymbol\\theta_1^\\top\\boldsymbol\\phi$ and $\\sigma^2=\\exp(\\boldsymbol\\theta_2^\\top\\boldsymbol\\phi)$.",
   "Correct loss $\\sum[\\frac12\\boldsymbol\\theta_2^\\top\\boldsymbol\\phi+\\frac{r^2}{2e}]$ with $r=y-\\boldsymbol\\theta_1^\\top\\boldsymbol\\phi$, $e=\\exp(\\boldsymbol\\theta_2^\\top\\boldsymbol\\phi)$.",
   "The sheet lists the seven distributions and seven identities. Learn where each is."],
  ["$\\nabla_{\\boldsymbol\\theta_1}=\\sum-\\frac re\\boldsymbol\\phi$","$\\nabla_{\\boldsymbol\\theta_2}=\\sum(\\frac12-\\frac{r^2}{2e})\\boldsymbol\\phi$"],
  ["The handwritten page drops the $\\frac12\\boldsymbol\\theta_2^\\top\\boldsymbol\\phi$ term. Keep it."],
  "Lecture 19 (midterm example and supplement); Murphy §13.2.4.4 p.433.", "A3, A6, M1"),
]

# ---------------- Questions ----------------
# group: (code, title, intro)
GROUPS = [
 ("A","The recipe: derive a loss and gradient","Closest to the assignment and the midterm example. Do these on paper first."),
 ("B","Assignment 1: Gaussians and clustering","The GMM assignment: fit Gaussians, assign points, sample, score."),
 ("C","Tensors and NumPy","Shapes, axes, broadcasting."),
 ("D","Probability and maximum likelihood","Bayes, Bernoulli MLE, MAP, log of a product."),
 ("E","Linear algebra and regression","Normal equations, gradient shapes."),
 ("F","Logistic and softmax","Sigmoid, cross-entropy, metrics, boundaries."),
 ("G","Generalization and statistics","Diagnosing train/validation error, cross-validation, error bars, entropy."),
 ("H","MAP, ridge and lasso","Priors become penalties."),
 ("I","Neural networks","Counting, forward pass, design reasons."),
 ("J","Gradient descent","Steps, stability, SGD, momentum, autodiff."),
]

# (id, group, difficulty 1-3, stem, answer)
Q = []
def q(i,g,d,stem,ans): Q.append((i,g,d,stem,ans))

q("A1","A",1,r"""Name the best distribution from the reference sheet for each label, with one reason.
<ol type="a"><li>Whether a customer clicks an ad (yes/no).</li><li>The number of emails an office receives per hour.</li><li>The height of a plant in cm, any real number.</li><li>Which of 5 species a flower belongs to.</li><li>Out of 20 patients, how many recover.</li><li>Days until a machine part fails (positive, not bounded).</li></ol>""",
r"""<ol type="a"><li><strong>Bernoulli</strong>: two outcomes, parameter $p\in[0,1]$ (sigmoid).</li><li><strong>Poisson</strong>: whole non-negative count with no fixed maximum, rate $\lambda>0$ (exp).</li><li><strong>Normal</strong>: any real value, mean from $\boldsymbol\theta^\top\boldsymbol\phi$.</li><li><strong>Categorical</strong>: one of $n$ classes, probability vector (softmax).</li><li><strong>Binomial</strong> with $n=20$ known and $p=\sigma(\cdot)$: a count that cannot exceed the number of trials.</li><li><strong>Weibull</strong>: positive durations with a flexible failure rate.</li></ol>""")

q("A2","A",2,r"""<strong>Assignment-2 style.</strong> Daily customer counts $y_i\in\{0,1,2,\dots\}$, features $\boldsymbol\phi(x_i)$. (1) Justify a distribution in at most 3 sentences. (2) Write $p(y|x;\boldsymbol\theta)$ with $\lambda=\exp(\boldsymbol\theta^\top\boldsymbol\phi(x))$. (3) Derive the loss $J$ showing each transformation. (4) Give $\nabla_{\boldsymbol\theta}J$. (5) One point has $\boldsymbol\phi=(1,2)$, $\boldsymbol\theta=(0.5,0.3)$, $y=3$. Compute $\lambda$ and the gradient.""",
r"""<ol><li>Counts are whole, non-negative and have no fixed maximum, so Bernoulli, Categorical and Binomial do not fit and the Normal would give probability to negative and fractional values. Use the Poisson.</li>
<li>$p(y|x;\boldsymbol\theta)=\dfrac{\lambda^y}{y!}e^{-\lambda}$, $\lambda=\exp(\boldsymbol\theta^\top\boldsymbol\phi(x))$.</li>
<li>$\log\prod_ip(y_i|x_i)=\sum_i[y_i\log\lambda_i-\lambda_i-\log y_i!]=\sum_i[y_iz_i-e^{z_i}-\log y_i!]$, with $z_i=\boldsymbol\theta^\top\boldsymbol\phi(x_i)$. Negate and drop $\log y_i!$ (no $\boldsymbol\theta$): $J=\sum_i[e^{z_i}-y_iz_i]$.</li>
<li>$\nabla_{\boldsymbol\theta}J=\sum_i(e^{z_i}-y_i)\boldsymbol\phi(x_i)=\sum_i(\lambda_i-y_i)\boldsymbol\phi(x_i)$.</li>
<li>$z=0.5+0.6=1.1$, $\lambda=e^{1.1}=3.004$; gradient $=(3.004-3)(1,2)=(0.004,\,0.008)$.</li></ol>""")

q("A3","A",3,r"""<strong>The midterm example.</strong> $y|x\sim\mathcal N(\mu,\sigma^2)$ with $\mu=\boldsymbol\theta_1^\top\boldsymbol\phi(x)$ and $\sigma^2=\exp(\boldsymbol\theta_2^\top\boldsymbol\phi(x))$. Write the loss for the data set, then $\nabla_{\boldsymbol\theta_1}$ and $\nabla_{\boldsymbol\theta_2}$. Evaluate the loss and both gradients for one point with $\boldsymbol\phi=(1,0.5)$, $y=2$, $\boldsymbol\theta_1=(1,1)$, $\boldsymbol\theta_2=(0,0.5)$.""",
r"""Let $r=y-\boldsymbol\theta_1^\top\boldsymbol\phi$ and $e=\exp(\boldsymbol\theta_2^\top\boldsymbol\phi)$.
<p>$\log p=-\tfrac12\log2\pi-\tfrac12\boldsymbol\theta_2^\top\boldsymbol\phi-\dfrac{r^2}{2e}$, so $J=\sum_i\Big[\tfrac12\boldsymbol\theta_2^\top\boldsymbol\phi(x_i)+\dfrac{r_i^2}{2e_i}\Big]$ (the constant $\frac12\log2\pi$ dropped).</p>
<p>$\nabla_{\boldsymbol\theta_1}J=\sum_i-\dfrac{r_i}{e_i}\boldsymbol\phi(x_i)$, $\quad\nabla_{\boldsymbol\theta_2}J=\sum_i\Big(\tfrac12-\dfrac{r_i^2}{2e_i}\Big)\boldsymbol\phi(x_i)$.</p>
<p>Numbers: $\mu=1.5$, $r=0.5$, $e=e^{0.25}=1.284$. $J=0.125+\frac{0.25}{2.568}=0.2224$. $\nabla_{\boldsymbol\theta_1}=-\frac{0.5}{1.284}(1,0.5)=(-0.3894,-0.1947)$. $\nabla_{\boldsymbol\theta_2}=(0.5-0.0973)(1,0.5)=(0.4026,0.2013)$.</p>""")

q("A4","A",3,r"""<strong>A distribution you did not see in class.</strong> Failure times $y>0$ follow a Weibull with <em>known</em> shape $k$ and scale $\lambda=\exp(\boldsymbol\theta^\top\boldsymbol\phi(x))$: $f(y)=\frac k\lambda\big(\frac y\lambda\big)^{k-1}e^{-(y/\lambda)^k}$. Derive the loss and its gradient. Evaluate the gradient for $k=2$, $y=3$, $z=\boldsymbol\theta^\top\boldsymbol\phi=1$, $\boldsymbol\phi=(1,1)$.""",
r"""<p>$-\log f=-\log k+\log\lambda-(k-1)(\log y-\log\lambda)+(y/\lambda)^k=-\log k-(k-1)\log y+k\log\lambda+y^k\lambda^{-k}$.</p>
<p>With $\lambda=e^z$ ($\log\lambda=z$, $\lambda^{-k}=e^{-kz}$): $J=\sum_i\big[kz_i+y_i^ke^{-kz_i}\big]$ (dropping $-\log k-(k-1)\log y$).</p>
<p>$\dfrac{\partial J}{\partial z}=k-ky^ke^{-kz}=k\big(1-(y/\lambda)^k\big)$, so $\nabla_{\boldsymbol\theta}J=\sum_ik\big(1-(y_i/\lambda_i)^k\big)\boldsymbol\phi(x_i)$.</p>
<p>Numbers: $\lambda=e=2.718$, $(3/2.718)^2=1.218$, so $\partial J/\partial z=2(1-1.218)=-0.436$ and the gradient is $(-0.436,-0.436)$. Negative: increase $z$ (the observed time is longer than the model expects).</p>
<p>The method is the same six steps: only the algebra is new. Do not expect to memorize this one.</p>""")

q("A5","A",2,r"""<strong>Binomial.</strong> $k_i$ successes out of a known $n_i$ trials, $p_i=\sigma(\boldsymbol\theta^\top\boldsymbol\phi(x_i))$. Write the loss and gradient. Evaluate for $n=10$, $k=7$, $z=0.5$, $\boldsymbol\phi=(1,2)$.""",
r"""<p>$\log\binom nk$ has no $\boldsymbol\theta$. $J=-\sum_i\big[k_i\log p_i+(n_i-k_i)\log(1-p_i)\big]$.</p>
<p>Using $\sigma'=\sigma(1-\sigma)$: $\nabla_{\boldsymbol\theta}J=\sum_i(n_ip_i-k_i)\boldsymbol\phi(x_i)$. Again (prediction − truth) × input, with the prediction being the expected count $np$.</p>
<p>Numbers: $p=\sigma(0.5)=0.6225$, $np=6.225$, $np-k=-0.775$, gradient $=(-0.775,-1.551)$. Loss term $=-[7\log0.6225+3\log0.3775]=6.241$.</p>""")

q("A6","A",2,r"""In the midterm example, a student minimizes only $\sum_i r_i^2/(2e_i)$. What goes wrong? Give the argument in two sentences, and say what the missing term represents.""",
r"""Each term $r_i^2/(2e_i)$ shrinks toward 0 as $e_i=\exp(\boldsymbol\theta_2^\top\boldsymbol\phi)\to\infty$, so the minimizer sends the predicted variance to infinity and the model says nothing. The missing term $\frac12\boldsymbol\theta_2^\top\boldsymbol\phi=\frac12\log\sigma^2$ comes from the normalizing constant $1/\sqrt{2\pi\sigma^2}$, which depends on $\boldsymbol\theta_2$; it is the price of a wide distribution and balances the first term.""")

q("B1","B",1,r"""Data $\{2,4,6\}$ come from a Normal. Find the MLE of the mean and variance. Why does the variance use $1/n$, not $1/(n-1)$?""",
r"""$\hat\mu=4$; $\hat\sigma^2=\frac13[(−2)^2+0+2^2]=\frac83=2.667$. Maximum likelihood gives the $1/n$ version; $1/(n-1)$ is the unbiased estimator, a different criterion. (In the course the Gaussian class divides by $n$ and adds a small $\varepsilon\mathbf I$ for numerical stability.)""")

q("B2","B",2,r"""Points $(1,2)$, $(3,4)$, $(5,0)$ are fit with a 2-D Gaussian by maximum likelihood. Find the mean and covariance.""",
r"""Mean $=(3,2)$. Centered points: $(-2,0)$, $(0,2)$, $(2,-2)$. $\Sigma=\frac13\sum\mathbf d\mathbf d^\top=\frac13\begin{bmatrix}8&-4\\-4&8\end{bmatrix}=\begin{bmatrix}2.667&-1.333\\-1.333&2.667\end{bmatrix}$. In code: <code>Xc = X - X.mean(0); Xc.T @ Xc / n</code>.""")

q("B3","B",2,r"""$\Sigma=\begin{bmatrix}4&2\\2&3\end{bmatrix}$. Find the Cholesky factor $\mathbf L$ and use it to turn $\mathbf z=(1,-1)$ (standard normal) into a sample with mean $\boldsymbol\mu=(0,0)$.""",
r"""$L_{11}=\sqrt4=2$; $L_{21}=2/2=1$; $L_{22}=\sqrt{3-1^2}=\sqrt2=1.414$. So $\mathbf L=\begin{bmatrix}2&0\\1&1.414\end{bmatrix}$; check $\mathbf L\mathbf L^\top=\begin{bmatrix}4&2\\2&3\end{bmatrix}$ ✓. Sample $=\boldsymbol\mu+\mathbf L\mathbf z=(2,\,1-1.414)=(2,-0.414)$. For a batch of rows $\mathbf Z$ of shape $(n,d)$: <code>mean + Z @ L.T</code>.""")

q("B4","B",1,r"""Cluster means $\boldsymbol\mu_1=(0,0)$ and $\boldsymbol\mu_2=(4,4)$. Assign the point $(3,2)$. Show the squared-distance trick used in the assignment code.""",
r"""$\|\mathbf x-\boldsymbol\mu_1\|^2=9+4=13$; $\|\mathbf x-\boldsymbol\mu_2\|^2=1+4=5$. Assign to cluster 2. The code expands $\|\mathbf x\|^2-2\mathbf x\cdot\boldsymbol\mu+\|\boldsymbol\mu\|^2$ so all distances come from one matrix product; then <code>argmin</code> over clusters.""")

q("B5","B",2,r"""In Assignment 1 the score fell from 528 ($k=1$) to 251 ($k=10$) to 134 ($k=50$). Does the lower score mean the model is better? What would you check?""",
r"""Not necessarily. With $k=50$ each group has only about 135 images, so each Gaussian can nearly memorize its group: the lower score largely reflects overfitting (a Gaussian fits a group almost perfectly when the group is small relative to the number of directions of variation). Check the score on held-out images and compare, and look whether samples are just copies of training digits.""")

q("C1","C",1,r"""Give the result shape, or say it fails.
<ol type="a"><li>$(3,1)+(1,4)$</li><li>$(5,)+(3,1)$</li><li>$(4,)+(3,)$</li><li>$(6700,784)-(784,)$</li><li>$(10,)-(10,1)$</li></ol>""",
r"""<ol type="a"><li>$(3,4)$</li><li>$(3,5)$</li><li>Fails: 4 and 3 are unequal and neither is 1.</li><li>$(6700,784)$: the $(784,)$ is treated as $(1,784)$ and applied to every row.</li><li>$(10,10)$: a silent bug, not an error.</li></ol>""")

q("C2","C",1,r"""$\mathbf X$ has shape $(10,3)$. Give the shapes of <code>X.mean(axis=0)</code>, <code>X.sum(axis=1)</code>, <code>X.T @ X</code> and <code>X @ np.ones(3)</code>.""",
r"""$(3,)$, $(10,)$, $(3,3)$, $(10,)$. The axis you name is the one that disappears.""")

q("C3","C",1,r"""<code>images</code> has shape $(6700,28,28)$. How do you flatten each image into a vector without hard-coding 6700? What shape does <code>np.arange(784).reshape(-1,28)</code> have?""",
r"""<code>images.reshape(-1, 784)</code> gives $(6700,784)$ (the <code>-1</code> is computed from the total size). <code>np.arange(784).reshape(-1,28)</code> has shape $(28,28)$.""")

q("C4","C",1,r"""<code>h = np.array([4,2,6,1,3,5,7,3,8,2])</code>, <code>m = h &gt;= 5</code>. Give <code>m</code>, <code>h[m]</code>, <code>m.sum()</code>, and <code>h[m].mean()</code>.""",
r"""<code>m</code> = F,F,T,F,F,T,T,F,T,F. <code>h[m]</code> = [6,5,7,8]. <code>m.sum()</code> = 4 (True counts as 1). Mean of $[6,5,7,8]=6.5$.""")

q("C5","C",2,r"""<code>y</code> has shape $(D,)$ and your model returns <code>pred</code> of shape $(D,1)$. What does <code>(pred - y).mean()</code> compute and why is it wrong? Give two fixes.""",
r"""Broadcasting makes <code>pred - y</code> a $(D,D)$ matrix of all pairwise differences, so the mean is meaningless but no error is raised. Fix: <code>pred.ravel() - y</code> (or <code>pred[:,0]</code>), or <code>pred - y[:, None]</code>. Always check shapes with <code>.shape</code> when a loss looks off.""")

q("D1","D",1,r"""A disease has prevalence 1%. A test has sensitivity 90% (positive if sick) and false-positive rate 5%. A person tests positive. Find $P(\text{sick}|+)$.""",
r"""$P(+)=0.9\cdot0.01+0.05\cdot0.99=0.009+0.0495=0.0585$. $P(\text{sick}|+)=\dfrac{0.009}{0.0585}=0.154$. Only about 15%, because healthy people vastly outnumber the sick.""")

q("D2","D",1,r"""A coin gives 7 heads in 10 flips. (a) MLE of $\theta=P(\text{heads})$. (b) The log-likelihood at the MLE. (c) MAP with a Beta$(2,2)$ prior.""",
r"""(a) $\hat\theta=7/10=0.7$. (b) $7\ln0.7+3\ln0.3=-2.497-3.612=-6.109$. (c) $\theta_{\text{MAP}}=\dfrac{H+\alpha-1}{D+\alpha+\beta-2}=\dfrac{8}{12}=0.667$: pulled toward 0.5 by the prior.""")

q("D3","D",1,r"""Independent flips $(1,0,1)$ with $P(1)=p$. Write the likelihood and the log-likelihood. Which sheet identities do you use?""",
r"""Likelihood $p\cdot(1-p)\cdot p=p^2(1-p)$. Log: $2\log p+\log(1-p)$, using identity 1 ($\log ab=\log a+\log b$) and identity 2 ($\log a^b=b\log a$). Setting the derivative $2/p-1/(1-p)=0$ gives $p=2/3$.""")

q("D4","D",2,r"""Explain in two sentences why we maximize the <em>log</em> of the likelihood, and why the loss has a minus sign.""",
r"""The likelihood of independent data is a product of many small numbers; the log turns it into a sum (easy to differentiate, no underflow) without moving the maximizer because log is increasing. Optimizers minimize, so we negate: the loss is the negative log-likelihood.""")

q("E1","E",2,r"""Data $(x,y)=(1,2),(2,4),(3,5)$. Fit $y=\theta_0+\theta_1x$ by least squares.""",
r"""$\bar x=2$, $\bar y=3.667$. $S_{xy}=(-1)(-1.667)+0+(1)(1.333)=3$; $S_{xx}=2$. Slope $=1.5$; intercept $=3.667-1.5\cdot2=0.667$. Prediction at $x=4$: $6.667$. (Normal equations give the same.)""")

q("E2","E",1,r"""$\mathbf X$ is $(D,n)$, $\boldsymbol\theta$ is $(n,)$, $\mathbf y$ is $(D,)$. Give the shape of every step of $\nabla\text{MSE}=\frac2D\mathbf X^\top(\mathbf X\boldsymbol\theta-\mathbf y)$.""",
r"""$\mathbf X\boldsymbol\theta$: $(D,)$. Minus $\mathbf y$: $(D,)$. $\mathbf X^\top$: $(n,D)$; times a $(D,)$ vector gives $(n,)$, the same shape as $\boldsymbol\theta$, as a gradient must.""")

q("E3","E",2,r"""When is $\mathbf X^\top\mathbf X$ not invertible, and what do you do?""",
r"""When columns of $\mathbf X$ are linearly dependent (repeated or redundant features) or when there are fewer data points than features. Remedies: drop features, use the pseudo-inverse, add ridge $\lambda\mathbf I$ (always invertible for $\lambda>0$), or use gradient descent.""")

q("F1","F",1,r"""$z=\boldsymbol\theta^\top\mathbf x=0.8$. Find $\sigma(z)$, the loss if $y=1$, and the loss if $y=0$.""",
r"""$\sigma(0.8)=0.690$. $y=1$: $-\ln0.690=0.371$. $y=0$: $-\ln(1-0.690)=1.171$. The loss is bigger when the prediction disagrees with the label.""")

q("F2","F",1,r"""Logits $(2,1,0)$ and the true class is the first. Compute the softmax and the cross-entropy loss.""",
r"""$e^2=7.389$, $e^1=2.718$, $e^0=1$; sum $11.107$. Softmax $=(0.665,0.245,0.090)$. Loss $=-\ln0.665=0.408$.""")

q("F3","F",1,r"""Confusion counts: TP 40, FP 10, FN 20, TN 30. Compute accuracy, precision, recall, $F_1$ and false-positive rate.""",
r"""Accuracy $=70/100=0.70$. Precision $=40/50=0.80$. Recall $=40/60=0.667$. $F_1=\frac{2(0.8)(0.667)}{0.8+0.667}=0.727$. FPR $=FP/(FP+TN)=10/40=0.25$.""")

q("F4","F",1,r"""A logistic model with $\boldsymbol\theta=(-3,1)$ on features $(1,h)$. What probability does it give at $h=3$, $h=5$? Where is the decision boundary?""",
r"""$z=-3+h$. At $h=3$: $z=0$, $p=0.5$. At $h=5$: $z=2$, $p=\sigma(2)=0.881$. The boundary is $z=0$, i.e. $h=3$.""")

q("G1","G",1,r"""A model has training error 2% and validation error 15%. Diagnose it and give two fixes. What if both were 20%?""",
r"""Large gap: overfitting (high variance). Fixes: more data, regularization (ridge/lasso/dropout), fewer features, early stopping. Both 20%: underfitting (high bias): use a richer model or better features, reduce regularization.""")

q("G2","G",1,r"""5-fold cross-validation on 100 points. How many models are fit, and on how many points is each trained and tested?""",
r"""5 models; each trains on 80 and tests on 20. Report the mean (and error bar) of the 5 scores. The final test set stays untouched.""")

q("G3","G",2,r"""Five runs give a mean accuracy of 80 with standard deviation 5. Compute the standard error and an approximate 95% interval.""",
r"""$\text{SEM}=5/\sqrt5=2.236$. About 95% interval: $80\pm1.96(2.236)=80\pm4.4$. (With only 5 runs the true coverage of ±1 SEM is below 68%.)""")

q("G4","G",1,r"""Compute the entropy in bits of $p=(0.5,0.25,0.25)$.""",
r"""$H=-[0.5\log_2 0.5+2(0.25\log_2 0.25)]=0.5+2(0.5)=1.5$ bits.""")

q("H1","H",2,r"""One-feature regression without intercept, data $(x,y)=(1,2),(2,4),(3,5)$. Fit $\theta$ with ridge penalty $\lambda\theta^2$ for $\lambda=0$ and $\lambda=2$.""",
r"""$\theta=\dfrac{\sum xy}{\sum x^2+\lambda}$. $\sum xy=25$, $\sum x^2=14$. $\lambda=0$: $1.786$. $\lambda=2$: $25/16=1.5625$. The penalty shrinks the weight toward 0.""")

q("H2","H",2,r"""In coordinate-descent lasso, the unpenalized value is $c=3$ and the threshold is $t=1$. Another coordinate has $c=0.7$. Apply soft thresholding to both.""",
r"""$\text{soft}(3,1)=2$; $\text{soft}(0.7,1)=0$. Lasso shrinks big weights by $t$ and sets small ones exactly to zero: that is why it selects features.""")

q("H3","H",1,r"""Complete: (a) Gaussian prior on weights gives ___ regularization. (b) Laplace prior gives ___. (c) Which one produces exact zeros? (d) What does a bigger $\lambda$ do to bias and variance?""",
r"""(a) Ridge, $L_2$ penalty $\lambda\|\boldsymbol\theta\|_2^2$. (b) Lasso, $L_1$ penalty $\lambda\|\boldsymbol\theta\|_1$. (c) Lasso. (d) More bias, less variance (a stronger prior).""")

q("I1","I",1,r"""A network has 4 inputs, 5 hidden units and 3 outputs (with biases). How many parameters?""",
r"""$M(D+1)+C(M+1)=5\cdot5+3\cdot6=25+18=43$.""")

q("I2","I",2,r"""Input $\mathbf x=(1,2)$. Hidden layer: $\mathbf W=\begin{bmatrix}1&-1\\0.5&0.5\end{bmatrix}$, $\mathbf b=(0,-1)$, ReLU. Output: $\mathbf w=(2,4)$, bias $0.5$, identity. Compute the output.""",
r"""$\mathbf z=\mathbf W\mathbf x+\mathbf b=(1-2+0,\ 0.5+1-1)=(-1,0.5)$. ReLU: $(0,0.5)$. Output $=2\cdot0+4\cdot0.5+0.5=2.5$.""")

q("I3","I",1,r"""(a) Why can one perceptron not learn XOR? (b) What happens if every hidden activation is the identity? (c) Why do deep sigmoid networks train badly? (d) Drop probability 0.3: what is the test-time scaling?""",
r"""(a) No single straight line separates XOR's classes. (b) The layers collapse into one linear map. (c) $\sigma'\le\frac14$, so each layer multiplies the backward signal by at most 0.25: the gradient vanishes. (d) Multiply by the keep probability $1-0.3=0.7$.""")

q("I4","I",1,r"""Match the output activation and loss: (a) real-valued label, (b) binary label, (c) $C$-class label.""",
r"""(a) Identity + squared error (Normal NLL). (b) Sigmoid + binary cross-entropy (Bernoulli NLL). (c) Softmax + categorical cross-entropy (Categorical NLL). Same recipe every time.""")

q("J1","J",2,r"""Minimize $J(\theta)=(\theta-2)^2$ from $\theta_0=0$ with $\alpha=0.25$. Give $\theta_1,\theta_2,\theta_3$. For which $\alpha$ does it converge?""",
r"""$J'=2(\theta-2)$. $\theta_1=0-0.25(-4)=1$; $\theta_2=1-0.25(-2)=1.5$; $\theta_3=1.5-0.25(-1)=1.75$. Each step halves the distance. The recursion is $\theta_{t+1}-2=(1-2\alpha)(\theta_t-2)$: converges iff $|1-2\alpha|<1$, i.e. $0<\alpha<1$. In general stable iff $\alpha<2/\lambda_{\max}$ (curvature here is 2).""")

q("J2","J",1,r"""Why use mini-batches? State the cost and the benefit, and the conditions on the step sizes.""",
r"""Benefit: each step is far cheaper than a pass over all $D$ points; the mini-batch gradient is an unbiased estimate of the full gradient. Cost: noise, shrinking like $1/\sqrt B$. Robbins–Monro: $\sum\alpha_t=\infty$ and $\sum\alpha_t^2<\infty$ (for example $\alpha_t=1/t$).""")

q("J3","J",2,r"""With momentum $\mathbf m_t=\beta\mathbf m_{t-1}+\mathbf g_t$ and a constant gradient $g=1$, what does $m_t$ tend to for $\beta=0.9$? What does momentum do to an oscillating component?""",
r"""$m\to g/(1-\beta)=10$: the effective step is ten times bigger along a consistent direction. Components that alternate in sign cancel. Too much momentum overshoots, so $\alpha$ and $\beta$ are tuned together.""")

q("J4","J",2,r"""Use the chain rule for $y=(3x+1)^2$ at $x=1$, written as $y=a^2$, $a=3x+1$. Which passes of autodiff compute which values?""",
r"""Forward: $a=4$, $y=16$. Backward: $dy/da=2a=8$, $da/dx=3$, so $dy/dx=24$. The forward pass stores the intermediate values; the backward pass multiplies local derivatives.""")

q("J5","J",1,r"""True or false, with a reason: (a) MSE for linear regression is convex. (b) A neural-network loss is convex. (c) For a convex loss, gradient descent with a small enough step reaches the global minimum. (d) A larger learning rate always converges faster.""",
r"""(a) True: its Hessian $\frac2D\mathbf X^\top\mathbf X$ is positive semi-definite. (b) False: non-convex. (c) True. (d) False: too large diverges (very high), or oscillates (high); the best is just below the stability limit.""")

# ---------------- Mock exam ----------------
MOCK_INTRO = r"""Sit this as a team of two, on paper, with a simple calculator and only the reference sheet (the distribution table and seven identities). 50 minutes. Cover the answers. Then mark yourselves with the notes."""

MOCK_MC = [
 (r"Which distribution suits ‘number of bugs found per release’?", ["Bernoulli","Binomial","Poisson","Normal"], 2, "Whole non-negative count with no fixed maximum."),
 (r"$\sigma^2=\exp(\boldsymbol\theta_2^\top\boldsymbol\phi)$ is used because", ["it is faster to compute","it keeps the variance positive","it makes the loss convex","it removes the bias"], 1, "exp maps any real number to a positive one."),
 (r"The gradient of the logistic loss for one point is", [r"$(y-\sigma(z))\mathbf x$",r"$(\sigma(z)-y)\mathbf x$",r"$\sigma(z)(1-\sigma(z))\mathbf x$",r"$2(z-y)\mathbf x$"], 1, "(prediction − truth) × input."),
 (r"Gradient descent on a quadratic with largest curvature 20 is stable iff $\alpha<$", ["0.05","0.1","0.2","2"], 1, "$2/\\lambda_{\\max}=2/20=0.1$."),
 (r"A Laplace prior on the weights corresponds to", ["ridge","lasso","dropout","early stopping"], 1, "$L_1$ penalty."),
 (r"A network $3\to4\to2$ (with biases) has how many parameters?", ["20","26","28","32"], 1, "$4(3+1)+2(4+1)=16+10=26$.") ,
 (r"Training error 1%, validation error 18%. The best first step is", ["a bigger model","more regularization or more data","lower the learning rate","remove the validation set"], 1, "Overfitting."),
 (r"Shapes $(5,1)$ and $(3,)$ are added. The result is", [r"$(5,3)$",r"$(5,1)$",r"$(3,)$","an error"], 0, "$(3,)$ becomes $(1,3)$; broadcasting gives $(5,3)$."),
 (r"A coin shows 7 heads in 10 flips. With a Beta$(2,2)$ prior the MAP estimate of $P(\text{heads})$ is", ["0.7","0.667","0.75","0.5"], 1, r"$(H+\alpha-1)/(D+\alpha+\beta-2)=8/12$."),
 (r"For a convex function, the chord between two points lies", ["below the curve","above (or on) the curve","only on the curve","has no relation"], 1, "That is the definition of convexity."),
]

MOCK_LONG = [
 ("Long answer 1 (15 marks): counts with a log link",
  r"""A factory records the number of defects $y_i\in\{0,1,2,\dots\}$ on each unit, with feature vector $\boldsymbol\phi(x_i)$. (a) Choose a distribution and justify it (2 marks). (b) Parameterize it with an activation that keeps it valid (2). (c) Derive the loss $J(\boldsymbol\theta)$, showing each transformation (6). (d) You are told $\nabla_{\boldsymbol\theta}J=\sum_i(\lambda_i-y_i)\boldsymbol\phi(x_i)$. For one point $\boldsymbol\phi=(1,1)$, $y=4$, $\boldsymbol\theta=(0,0)$, compute the gradient and one gradient-descent step with $\alpha=0.1$ (5).""",
  r"""<p>(a) Poisson: whole, non-negative, unbounded count. (b) $\lambda=\exp(\boldsymbol\theta^\top\boldsymbol\phi)$ so $\lambda>0$.</p>
<p>(c) $\log\prod_i\frac{\lambda_i^{y_i}e^{-\lambda_i}}{y_i!}=\sum_i[y_iz_i-e^{z_i}-\log y_i!]$; negate, drop $\log y_i!$: $J=\sum_i[e^{z_i}-y_iz_i]$.</p>
<p>(d) $z=0$, $\lambda=1$; gradient $=(1-4)(1,1)=(-3,-3)$. $\boldsymbol\theta_{\text{new}}=(0,0)-0.1(-3,-3)=(0.3,0.3)$.</p>
<p><em>Marking:</em> lose marks for a missing minus sign, for not saying you dropped $\log y!$, or for a wrong sign in the update.</p>"""),
 ("Long answer 2 (15 marks): mean and variance both learned",
  r"""Heights $y_i$ of plants depend on the age $x_i$; the spread grows with age. Use a Normal with $\mu=\boldsymbol\theta_1^\top\boldsymbol\phi(x)$ and $\sigma^2=\exp(\boldsymbol\theta_2^\top\boldsymbol\phi(x))$. (a) Write $\log p(y|x)$ (4). (b) Write the loss and say which term must not be dropped and why (5). (c) You are told $\nabla_{\boldsymbol\theta_2}J=\sum_i\big(\frac12-\frac{r_i^2}{2e_i}\big)\boldsymbol\phi(x_i)$. Explain in words what happens to the predicted variance when $r_i^2>e_i$ (4). (d) Name the sheet identities you used (2).""",
  r"""<p>(a) $\log p=-\frac12\log2\pi-\frac12\boldsymbol\theta_2^\top\boldsymbol\phi-\frac{(y-\boldsymbol\theta_1^\top\boldsymbol\phi)^2}{2\exp(\boldsymbol\theta_2^\top\boldsymbol\phi)}$.</p>
<p>(b) $J=\sum_i[\frac12\boldsymbol\theta_2^\top\boldsymbol\phi(x_i)+\frac{r_i^2}{2e_i}]$. The $\frac12\boldsymbol\theta_2^\top\boldsymbol\phi$ term depends on $\boldsymbol\theta_2$ (it is $\frac12\log\sigma^2$), so it must stay; without it the variance runs to infinity.</p>
<p>(c) If $r^2>e$ the bracket is negative, so gradient descent increases $\boldsymbol\theta_2^\top\boldsymbol\phi$ and raises the variance; if $r^2<e$ it lowers it. At the optimum $r^2\approx e$ on average.</p>
<p>(d) Identity 3 ($\log\exp x=x$), identities 1 and 2 for the log of a product and of a quotient.</p>"""),
 ("Long answer 3 (10 marks): Gaussians and a clustering step",
  r"""Points $(1,2),(3,4),(5,0)$. (a) Fit a 2-D Gaussian by maximum likelihood: mean and covariance (4). (b) Two cluster means $(0,0)$ and $(4,4)$: assign the point $(3,2)$ and show the distances (2). (c) Explain how you sample from a Gaussian with covariance $\Sigma$ using the Cholesky factor (2). (d) Why can a very small cluster give a misleadingly good score (2)?""",
  r"""<p>(a) Mean $(3,2)$; $\Sigma=\frac13\begin{bmatrix}8&-4\\-4&8\end{bmatrix}$.</p>
<p>(b) Distances$^2$ $13$ and $5$: cluster 2.</p>
<p>(c) $\Sigma=\mathbf L\mathbf L^\top$; draw $\mathbf z\sim\mathcal N(\mathbf 0,\mathbf I)$ and return $\boldsymbol\mu+\mathbf L\mathbf z$.</p>
<p>(d) A Gaussian fit to few points can match them almost exactly (overfitting), so the training score looks great but will not generalize.</p>"""),
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
<p>To make the model depend on the input, let the parameter be a function of the features: $\lambda=\exp(\boldsymbol\theta^\top\boldsymbol\phi)$, $p=\sigma(\boldsymbol\theta^\top\boldsymbol\phi)$, $\mu=\boldsymbol\theta^\top\boldsymbol\phi$. The activation keeps the parameter legal. Then follow the six-step template above. This is the skill the midterm tests.</p>""",
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
"t11": r"""<p>To minimize a differentiable loss: $\boldsymbol\theta\leftarrow\boldsymbol\theta-\alpha\nabla J(\boldsymbol\theta)$. The gradient points uphill, so we step the other way; $\alpha$ is the learning rate. <em>Example</em> $J=(\theta-2)^2$ from $0$, $\alpha=0.25$: $1,\ 1.5,\ 1.75,\dots$ each step halves the distance.</p>
<p><strong>Learning rate:</strong> too high diverges, high oscillates, low is slow. On a quadratic it is stable iff $\alpha<2/\lambda_{\max}$ (the largest curvature). <strong>Convex</strong> (bowl; chord above the curve): one global minimum. MSE and logistic loss are convex; neural-network losses are not.</p>
<p><strong>SGD:</strong> use a random mini-batch of size $B$ for each step: cheaper, unbiased, noisy (noise falls as $B$ grows); step sizes need $\sum\alpha_t=\infty$, $\sum\alpha_t^2<\infty$. <strong>Momentum</strong> $\mathbf m_t=\beta\mathbf m_{t-1}+\mathbf g_t$ smooths the zig-zag. Adam, RMSprop, Adagrad give each weight its own step. Autodiff (PyTorch): forward pass stores values, backward pass multiplies local derivatives.</p>""",
"t12": r"""<p>The sample question applies the recipe to a plot whose noise grows with $x$. <strong>Distribution:</strong> Normal (real label). <strong>Parameterize both parameters:</strong> $\mu=\boldsymbol\theta_1^\top\boldsymbol\phi(x)$, $\sigma^2=\exp(\boldsymbol\theta_2^\top\boldsymbol\phi(x))$ (exp keeps the variance positive). <strong>Log-density:</strong> $-\frac12\log2\pi-\frac12\log\sigma^2-\frac{(y-\mu)^2}{2\sigma^2}$, and $\log\sigma^2=\boldsymbol\theta_2^\top\boldsymbol\phi$ by identity 3. <strong>Loss:</strong> $\sum_i\big[\frac12\boldsymbol\theta_2^\top\boldsymbol\phi(x_i)+\frac{(y_i-\boldsymbol\theta_1^\top\boldsymbol\phi(x_i))^2}{2\exp(\boldsymbol\theta_2^\top\boldsymbol\phi(x_i))}\big]$.</p>
<p>The handwritten solution leaves out the first term. It must stay because it depends on $\boldsymbol\theta_2$; without it the variance grows without limit. The same method works for any distribution on the sheet.</p>""",
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
 ]),
 ("Provided on the exam sheet: do not memorize", [
  "The seven distributions (Normal, Bernoulli, Beta, Categorical, Binomial, Poisson, Weibull) with their supports, PMF/PDF and parameters.",
  r"$\log ab=\log a+\log b$, $\log a^b=b\log a$, $\log\exp x=x$, sums and products, $p(a,b)=p(a|b)p(b)$, Bayes’ rule.",
 ]),
]
