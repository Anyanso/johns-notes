# -*- coding: utf-8 -*-
# Final exam question bank. Every answer is a numbered, step-by-step solution.
# $...$ = math (pre-rendered by KaTeX at build time).

def S(*steps, tip=None):
    out = '<ol class="steps">' + ''.join('<li>%s</li>' % s for s in steps) + '</ol>'
    if tip:
        out += '<p class="tip"><strong>Exam tip.</strong> %s</p>' % tip
    return out

# (code, title, intro, topic card id)
GROUPS = [
 ("A", "The recipe: derive a loss and its gradient", "The heart of the course. Pick a distribution, parameterize it, write the log-likelihood, negate, differentiate. Do every one on paper first.", "t3"),
 ("B", "Gaussians, clustering and sampling (Assignment 1)", "Fit Gaussians by maximum likelihood, evaluate densities, assign points, sample with Cholesky, read the score.", "t4"),
 ("C", "Tensors and NumPy", "Shapes, axes, broadcasting, masks, views and copies.", "t1"),
 ("D", "Probability and maximum likelihood", "Bayes, expectation and variance, independence, and MLE by hand.", "t2"),
 ("E", "Linear algebra and linear regression", "Normal equations, MSE and its gradient, basis functions, norms, eigenvalues.", "t5"),
 ("F", "Logistic, softmax and metrics", "Sigmoid, cross-entropy, decision boundaries, thresholds, precision and recall.", "t6"),
 ("G", "Generalization and statistics", "Overfitting, cross-validation, error bars, comparing models, entropy.", "t7"),
 ("H", "MAP, ridge and lasso", "Priors become penalties. Closed forms, soft thresholding, choosing $\\lambda$.", "t8"),
 ("I", "Neural networks", "Counting parameters, forward passes, backpropagation, perceptron, dropout.", "t9"),
 ("J", "Gradient descent", "Steps by hand, stability, SGD, momentum, autodiff.", "t11"),
 ("K", "Sequence models (Lectures 16–17)", "Markov chains, hidden states, marginalization, RNNs, attention, LLMs.", "t10"),
 ("L", "SVMs and constrained optimization (Lecture 20)", "Margin, constraints, support vectors, Lagrangians, slack and the hinge loss.", "t13"),
]

Q = []
def q(i, g, d, stem, ans): Q.append((i, g, d, stem, ans))

# =====================================================================
# A. The recipe
# =====================================================================
q("A1","A",1,r"""Name the best distribution from the reference sheet for each label, with one reason.
<ol type="a"><li>Whether a customer clicks an ad (yes/no).</li><li>The number of emails an office receives per hour.</li><li>The height of a plant in cm, any real number.</li><li>Which of 5 species a flower belongs to.</li><li>Out of 20 patients, how many recover.</li><li>Days until a machine part fails (positive, not bounded).</li></ol>""",
S(r"Ask three questions of each label: is it whole? can it be negative? is there a maximum?",
  r"(a) Two outcomes → <strong>Bernoulli</strong>, parameter $p\in[0,1]$, activation sigmoid.",
  r"(b) Whole, never negative, no fixed maximum → <strong>Poisson</strong>, rate $\lambda>0$, activation exp.",
  r"(c) Any real number → <strong>Normal</strong>, mean $\mu=\boldsymbol\theta^\top\boldsymbol\phi$ (identity).",
  r"(d) One of $5$ unordered classes → <strong>Categorical</strong>, probability vector from softmax.",
  r"(e) Whole, at most 20 → <strong>Binomial</strong> with $n=20$ known and $p=\sigma(\cdot)$.",
  r"(f) Positive duration → <strong>Weibull</strong> (scale through exp).",
  tip=r"Write the reason in one short sentence; ‘count with no maximum’ is the whole justification for Poisson."))

q("A2","A",2,r"""<strong>Assignment-2 style.</strong> Daily customer counts $y_i\in\{0,1,2,\dots\}$, features $\boldsymbol\phi(x_i)$. (1) Justify a distribution in at most 3 sentences. (2) Write $p(y|x;\boldsymbol\theta)$ with $\lambda=\exp(\boldsymbol\theta^\top\boldsymbol\phi(x))$. (3) Derive the loss $J$ showing each transformation. (4) Give $\nabla_{\boldsymbol\theta}J$. (5) One point has $\boldsymbol\phi=(1,2)$, $\boldsymbol\theta=(0.5,0.3)$, $y=3$. Compute $\lambda$ and the gradient.""",
S(r"<strong>Distribution.</strong> Counts are whole, non-negative and have no fixed maximum. Bernoulli, Categorical and Binomial have the wrong support; the Normal gives probability to $-2$ or $2.5$ customers. Use the Poisson.",
  r"<strong>Model.</strong> $p(y|x;\boldsymbol\theta)=\dfrac{\lambda^y}{y!}e^{-\lambda}$ with $\lambda=\exp(z)$, $z=\boldsymbol\theta^\top\boldsymbol\phi(x)$.",
  r"<strong>Likelihood → log.</strong> $\log\prod_ip(y_i|x_i)=\sum_i\log p(y_i|x_i)$ (identity 1) $=\sum_i[y_i\log\lambda_i-\lambda_i-\log y_i!]$ (identities 1, 2).",
  r"<strong>Simplify.</strong> $\log\lambda_i=\log e^{z_i}=z_i$ (identity 3), so the sum is $\sum_i[y_iz_i-e^{z_i}-\log y_i!]$.",
  r"<strong>Negate, drop constants.</strong> $J(\boldsymbol\theta)=\sum_i[e^{z_i}-y_iz_i]$; $\log y_i!$ has no $\boldsymbol\theta$ (say so).",
  r"<strong>Gradient.</strong> $\partial J/\partial z_i=e^{z_i}-y_i$ and $\partial z_i/\partial\boldsymbol\theta=\boldsymbol\phi(x_i)$, so $\nabla_{\boldsymbol\theta}J=\sum_i(\lambda_i-y_i)\boldsymbol\phi(x_i)$.",
  r"<strong>Numbers.</strong> $z=0.5\cdot1+0.3\cdot2=1.1$, $\lambda=e^{1.1}=3.004$; gradient $=(3.004-3)(1,2)=(0.004,\,0.008)$.",
  tip="The gradient always ends as (prediction − truth) × features. If yours does not, look for a sign error."))

q("A3","A",3,r"""<strong>The midterm example.</strong> $y|x\sim\mathcal N(\mu,\sigma^2)$ with $\mu=\boldsymbol\theta_1^\top\boldsymbol\phi(x)$ and $\sigma^2=\exp(\boldsymbol\theta_2^\top\boldsymbol\phi(x))$. Write the loss for the data set, then $\nabla_{\boldsymbol\theta_1}$ and $\nabla_{\boldsymbol\theta_2}$. Evaluate the loss and both gradients for one point with $\boldsymbol\phi=(1,0.5)$, $y=2$, $\boldsymbol\theta_1=(1,1)$, $\boldsymbol\theta_2=(0,0.5)$.""",
S(r"Name the pieces: $r=y-\boldsymbol\theta_1^\top\boldsymbol\phi$ (residual) and $e=\exp(\boldsymbol\theta_2^\top\boldsymbol\phi)$ (variance).",
  r"Full log-density (keep the constant’s $\sigma^2$!): $\log p=-\tfrac12\log2\pi-\tfrac12\log\sigma^2-\dfrac{r^2}{2\sigma^2}=-\tfrac12\log2\pi-\tfrac12\boldsymbol\theta_2^\top\boldsymbol\phi-\dfrac{r^2}{2e}$.",
  r"Negate, sum, drop only $\tfrac12\log2\pi$: $J=\sum_i\Big[\tfrac12\boldsymbol\theta_2^\top\boldsymbol\phi(x_i)+\dfrac{r_i^2}{2e_i}\Big]$.",
  r"$\nabla_{\boldsymbol\theta_1}$: $\partial(r^2)/\partial\boldsymbol\theta_1=-2r\boldsymbol\phi$, so $\nabla_{\boldsymbol\theta_1}J=\sum_i-\dfrac{r_i}{e_i}\boldsymbol\phi(x_i)$.",
  r"$\nabla_{\boldsymbol\theta_2}$: $\partial(1/e)/\partial\boldsymbol\theta_2=-\boldsymbol\phi/e$, so $\nabla_{\boldsymbol\theta_2}J=\sum_i\Big(\tfrac12-\dfrac{r_i^2}{2e_i}\Big)\boldsymbol\phi(x_i)$.",
  r"Numbers: $\mu=1+0.5=1.5$, $r=0.5$, $e=e^{0.25}=1.284$. $J=0.125+0.25/2.568=0.2224$.",
  r"$\nabla_{\boldsymbol\theta_1}=-\frac{0.5}{1.284}(1,0.5)=(-0.3894,-0.1947)$; $\nabla_{\boldsymbol\theta_2}=(0.5-0.0973)(1,0.5)=(0.4026,0.2013)$.",
  tip=r"The handwritten solution forgot $\tfrac12\boldsymbol\theta_2^\top\boldsymbol\phi$. Never drop a term that contains a parameter."))

q("A4","A",3,r"""<strong>A distribution you did not see in class.</strong> Failure times $y>0$ follow a Weibull with <em>known</em> shape $k$ and scale $\lambda=\exp(\boldsymbol\theta^\top\boldsymbol\phi(x))$: $f(y)=\frac k\lambda\big(\frac y\lambda\big)^{k-1}e^{-(y/\lambda)^k}$. Derive the loss and its gradient. Evaluate the gradient for $k=2$, $y=3$, $z=\boldsymbol\theta^\top\boldsymbol\phi=1$, $\boldsymbol\phi=(1,1)$.""",
S(r"Take the log of each factor: $\log f=\log k-\log\lambda+(k-1)(\log y-\log\lambda)-(y/\lambda)^k$.",
  r"Negate and collect the $\log\lambda$ terms: $-\log f=-\log k-(k-1)\log y+k\log\lambda+y^k\lambda^{-k}$.",
  r"Substitute $\lambda=e^z$: $\log\lambda=z$ and $\lambda^{-k}=e^{-kz}$.",
  r"Drop $-\log k-(k-1)\log y$ (no $\boldsymbol\theta$): $J=\sum_i[kz_i+y_i^ke^{-kz_i}]$.",
  r"Differentiate: $\partial J/\partial z=k-ky^ke^{-kz}=k\big(1-(y/\lambda)^k\big)$, so $\nabla_{\boldsymbol\theta}J=\sum_ik\big(1-(y_i/\lambda_i)^k\big)\boldsymbol\phi(x_i)$.",
  r"Numbers: $\lambda=e^1=2.718$, $(3/2.718)^2=1.218$, $\partial J/\partial z=2(1-1.218)=-0.436$, gradient $(-0.436,-0.436)$.",
  r"Interpret: negative means increase $z$ (the observed time is longer than the model expects).",
  tip="New distributions on the sheet are fair game. The six steps never change; only the algebra does."))

q("A5","A",2,r"""<strong>Binomial.</strong> $k_i$ successes out of a known $n_i$ trials, $p_i=\sigma(\boldsymbol\theta^\top\boldsymbol\phi(x_i))$. Write the loss and gradient. Evaluate for $n=10$, $k=7$, $z=0.5$, $\boldsymbol\phi=(1,2)$.""",
S(r"Log of the sheet’s PMF: $\log\binom nk+k\log p+(n-k)\log(1-p)$.",
  r"Negate, drop $\log\binom nk$ (no $\boldsymbol\theta$): $J=-\sum_i[k_i\log p_i+(n_i-k_i)\log(1-p_i)]$.",
  r"Use $\frac{d}{dz}\log\sigma(z)=1-\sigma$ and $\frac{d}{dz}\log(1-\sigma(z))=-\sigma$.",
  r"So $\partial J/\partial z_i=-[k_i(1-p_i)-(n_i-k_i)p_i]=n_ip_i-k_i$ and $\nabla_{\boldsymbol\theta}J=\sum_i(n_ip_i-k_i)\boldsymbol\phi(x_i)$.",
  r"Numbers: $p=\sigma(0.5)=0.6225$, $np=6.225$, $np-k=-0.775$; gradient $(-0.775,-1.551)$.",
  r"Loss term: $-[7\log0.6225+3\log0.3775]=6.241$.",
  tip="Prediction here is the expected count $np$, truth is $k$: again (prediction − truth) × input."))

q("A6","A",2,r"""In the midterm example, a student minimizes only $\sum_i r_i^2/(2e_i)$. What goes wrong? Give the argument in two sentences, and say what the missing term represents.""",
S(r"Look at one term: $r_i^2/(2e_i)$ shrinks to 0 as $e_i=\exp(\boldsymbol\theta_2^\top\boldsymbol\phi)\to\infty$.",
  r"So the minimizer sends the predicted variance to infinity: a useless model that says ‘anything can happen’.",
  r"The missing term $\tfrac12\boldsymbol\theta_2^\top\boldsymbol\phi=\tfrac12\log\sigma^2$ comes from the normalizing constant $1/\sqrt{2\pi\sigma^2}$, which depends on $\boldsymbol\theta_2$.",
  r"It is the price of a wide distribution and balances the first term; at the optimum $r^2\approx e$."))

q("A7","A",2,r"""<strong>Logistic regression from scratch.</strong> $y\in\{0,1\}$, $p=\sigma(\boldsymbol\theta^\top\boldsymbol\phi)$. Derive the loss and the gradient. Evaluate both for $\boldsymbol\phi=(1,2)$, $\boldsymbol\theta=(-1,0.5)$, $y=0$.""",
S(r"Bernoulli PMF: $p^y(1-p)^{1-y}$. Log: $y\log p+(1-y)\log(1-p)$ (identity 2).",
  r"Negate and sum: $J=-\sum_i[y_i\log p_i+(1-y_i)\log(1-p_i)]$ (binary cross-entropy).",
  r"Chain rule with $\sigma'=\sigma(1-\sigma)$: $\partial J/\partial z_i=p_i-y_i$, so $\nabla J=\sum_i(p_i-y_i)\boldsymbol\phi_i$.",
  r"Numbers: $z=-1+0.5\cdot2=0$, $p=\sigma(0)=0.5$. Loss $=-\log(1-0.5)=0.693$.",
  r"Gradient $=(0.5-0)(1,2)=(0.5,\,1.0)$: step against it to lower $p$ for this negative example."))

q("A8","A",2,r"""<strong>Least squares is a Normal likelihood.</strong> Assume $y|x\sim\mathcal N(\boldsymbol\theta^\top\boldsymbol\phi,1)$. Derive $J$ and $\nabla J$. Evaluate both for two points $\boldsymbol\phi_1=(1,1),y_1=3$ and $\boldsymbol\phi_2=(1,2),y_2=4$ at $\boldsymbol\theta=(1,1)$.""",
S(r"Log-density with $\sigma^2=1$: $-\tfrac12\log2\pi-\tfrac12(y-\boldsymbol\theta^\top\boldsymbol\phi)^2$.",
  r"Negate, sum, drop $\tfrac12\log2\pi$ (no $\boldsymbol\theta$, because $\sigma^2$ is fixed): $J=\tfrac12\sum_i(y_i-\boldsymbol\theta^\top\boldsymbol\phi_i)^2$.",
  r"Gradient: $\nabla J=-\sum_ir_i\boldsymbol\phi_i$ with $r_i=y_i-\boldsymbol\theta^\top\boldsymbol\phi_i$.",
  r"Predictions $2$ and $3$, residuals $r=(1,1)$, so $J=\tfrac12(1+1)=1$.",
  r"$\nabla J=-[(1,1)+(1,2)]=(-2,-3)$: gradient descent increases both weights.",
  tip=r"Fixed variance → the constant really is constant. Learned variance (A3) → it is not."))

q("A9","A",2,r"""<strong>Softmax regression.</strong> $C$ classes, $\mathbf p=\text{softmax}(\mathbf z)$ with $z_k=\boldsymbol\theta_k^\top\boldsymbol\phi$. Write the loss for one point and $\partial J/\partial z_k$. Evaluate for logits $\mathbf z=(1,0,-1)$ when the true class is the second.""",
S(r"Categorical PMF: $\prod_kp_k^{\mathbb 1[y=k]}$. Log, negate: $J=-\log p_y$ (cross-entropy).",
  r"Write $-\log p_y=-z_y+\log\sum_je^{z_j}$.",
  r"Differentiate: $\partial J/\partial z_k=p_k-\mathbb 1[y=k]$; then $\nabla_{\boldsymbol\theta_k}J=(p_k-\mathbb 1[y=k])\boldsymbol\phi$.",
  r"$e^1=2.718$, $e^0=1$, $e^{-1}=0.368$, sum $4.086$: $\mathbf p=(0.665,0.245,0.090)$.",
  r"Loss $=-\log0.245=1.408$; $\partial J/\partial\mathbf z=(0.665,-0.755,0.090)$: push the true logit up, the others down."))

q("A10","A",3,r"""<strong>Only the variance is learned.</strong> $y|x\sim\mathcal N(0,\sigma^2)$ with $\sigma^2=\exp(z)$, $z=\boldsymbol\theta^\top\boldsymbol\phi$. Derive $J$ and $\nabla J$. For $y=2$, $z=0$, $\boldsymbol\phi=(1,1)$, evaluate the loss term and the gradient. Which $z$ is best for this point alone?""",
S(r"Log-density: $-\tfrac12\log2\pi-\tfrac12z-\dfrac{y^2}{2e^z}$.",
  r"Negate, drop $\tfrac12\log2\pi$: $J=\sum_i\big[\tfrac12z_i+\tfrac{y_i^2}{2}e^{-z_i}\big]$.",
  r"$\partial J/\partial z=\tfrac12-\tfrac{y^2}{2}e^{-z}$, so $\nabla J=\sum_i\big(\tfrac12-\tfrac{y_i^2}{2e^{z_i}}\big)\boldsymbol\phi_i$.",
  r"Numbers: loss term $=0+4/2=2$; $\partial J/\partial z=0.5-2=-1.5$; gradient $(-1.5,-1.5)$.",
  r"Set $\partial J/\partial z=0$: $e^z=y^2=4$, so $z=\log4=1.386$ (variance equals the squared distance from the mean)."))

q("A11","A",2,r"""<strong>Waiting times.</strong> The Weibull with $k=1$ is $f(y)=\frac1\lambda e^{-y/\lambda}$. With $\lambda=\exp(z)$, write $J$ and $\partial J/\partial z$. Evaluate for $y=0.5$, $z=0$.""",
S(r"Log: $-\log\lambda-y/\lambda=-z-ye^{-z}$.",
  r"Negate and sum: $J=\sum_i[z_i+y_ie^{-z_i}]$. Nothing to drop.",
  r"$\partial J/\partial z=1-ye^{-z}=1-y/\lambda$.",
  r"Numbers: $\lambda=1$, $\partial J/\partial z=1-0.5=0.5>0$: decrease $z$ (shorter predicted wait), because the observed wait $0.5$ is shorter than $\lambda=1$."))

q("A12","A",1,r"""<strong>Spot the error.</strong> A student writes for the Poisson model: “$J(\boldsymbol\theta)=\sum_i[y_iz_i-e^{z_i}]$, then I run gradient descent.” What is wrong, what happens, and how do you fix it?""",
S(r"That expression is the log-likelihood (with $\log y!$ dropped), not its negative.",
  r"Gradient descent on it <em>lowers</em> the likelihood: the weights move to make the data less probable and $\lambda$ drifts away from the counts.",
  r"Fix: negate. $J=\sum_i[e^{z_i}-y_iz_i]$, or run gradient <em>ascent</em> on the original.",
  r"Check: at the optimum $\sum_i(\lambda_i-y_i)\boldsymbol\phi_i=\mathbf 0$, so predictions match counts on average.",
  tip="The minus sign is one of the most common lost marks. Write ‘loss = negative log-likelihood’ as your first line."))

# =====================================================================
# B. Gaussians, clustering, sampling
# =====================================================================
q("B1","B",1,r"""Data $\{2,4,6\}$ come from a Normal. Find the MLE of the mean and variance. Why does the variance use $1/n$, not $1/(n-1)$?""",
S(r"Mean: $\hat\mu=\frac{2+4+6}3=4$.",
  r"Squared deviations: $(-2)^2,0^2,2^2=4,0,4$.",
  r"MLE variance divides by $n=3$: $\hat\sigma^2=8/3=2.667$.",
  r"Maximum likelihood gives $1/n$ (set the derivative of the log-likelihood to 0). $1/(n-1)$ gives the unbiased estimator ($=4$ here), a different criterion."))

q("B2","B",2,r"""Points $(1,2)$, $(3,4)$, $(5,0)$ are fit with a 2-D Gaussian by maximum likelihood. Find the mean and covariance.""",
S(r"Mean of each coordinate: $\hat{\boldsymbol\mu}=(3,2)$.",
  r"Centre the points: $\mathbf d=(-2,0),(0,2),(2,-2)$.",
  r"Outer products $\mathbf d\mathbf d^\top$: $\begin{bmatrix}4&0\\0&0\end{bmatrix},\begin{bmatrix}0&0\\0&4\end{bmatrix},\begin{bmatrix}4&-4\\-4&4\end{bmatrix}$; sum $\begin{bmatrix}8&-4\\-4&8\end{bmatrix}$.",
  r"Divide by $n=3$: $\Sigma=\begin{bmatrix}2.667&-1.333\\-1.333&2.667\end{bmatrix}$.",
  r"In code: <code>Xc = X - X.mean(0); Xc.T @ Xc / n</code>."))

q("B3","B",2,r"""$\Sigma=\begin{bmatrix}4&2\\2&3\end{bmatrix}$. Find the Cholesky factor $\mathbf L$ and use it to turn $\mathbf z=(1,-1)$ (standard normal) into a sample with mean $\mathbf 0$.""",
S(r"$L_{11}=\sqrt{\Sigma_{11}}=\sqrt4=2$.",
  r"$L_{21}=\Sigma_{21}/L_{11}=2/2=1$.",
  r"$L_{22}=\sqrt{\Sigma_{22}-L_{21}^2}=\sqrt{3-1}=1.414$. So $\mathbf L=\begin{bmatrix}2&0\\1&1.414\end{bmatrix}$.",
  r"Check $\mathbf L\mathbf L^\top=\begin{bmatrix}4&2\\2&3\end{bmatrix}$ ✓.",
  r"Sample $=\mathbf L\mathbf z=(2\cdot1,\ 1\cdot1+1.414\cdot(-1))=(2,-0.414)$. For rows $\mathbf Z$: <code>mean + Z @ L.T</code>."))

q("B4","B",1,r"""Cluster means $\boldsymbol\mu_1=(0,0)$ and $\boldsymbol\mu_2=(4,4)$. Assign the point $(3,2)$. Show the squared-distance trick used in the assignment code.""",
S(r"$\|\mathbf x-\boldsymbol\mu_1\|^2=3^2+2^2=13$.",
  r"$\|\mathbf x-\boldsymbol\mu_2\|^2=1^2+2^2=5$. Assign to cluster 2.",
  r"Trick: $\|\mathbf x-\boldsymbol\mu\|^2=\|\mathbf x\|^2-2\mathbf x^\top\boldsymbol\mu+\|\boldsymbol\mu\|^2$, so all distances come from one matrix product <code>X @ M.T</code>.",
  r"Then <code>argmin</code> over the cluster axis."))

q("B5","B",2,r"""In Assignment 1 the score fell from 528 ($k=1$) to 251 ($k=10$) to 134 ($k=50$). Does the lower score mean the model is better? What would you check?""",
S(r"More clusters means fewer points per Gaussian (about 135 images each at $k=50$).",
  r"A Gaussian fitted to few points relative to the dimension can nearly memorize them: the training score improves because of overfitting.",
  r"Check: score on held-out images; compare across $k$ on that.",
  r"Also look at samples: copies of training digits are a warning sign."))

q("B6","B",1,r"""Evaluate $\log\mathcal N(x=3;\ \mu=1,\ \sigma^2=4)$ by hand.""",
S(r"Full log-density: $-\tfrac12\log(2\pi\sigma^2)-\dfrac{(x-\mu)^2}{2\sigma^2}$.",
  r"Constant: $2\pi\cdot4=25.13$, $\log25.13=3.224$, half is $1.612$.",
  r"Quadratic: $(3-1)^2/(2\cdot4)=4/8=0.5$.",
  r"Total: $-1.612-0.5=-2.112$."))

q("B7","B",2,r"""In the Gaussian assignment the code adds $\varepsilon\mathbf I$ to the covariance. Why? When does the plain MLE covariance fail?""",
S(r"The MLE covariance $\frac1n\sum\mathbf d\mathbf d^\top$ has rank at most $n-1$.",
  r"If $n\le d$ (fewer points than dimensions, e.g. a small cluster of 784-pixel images), or some pixels are constant, $\Sigma$ is singular.",
  r"Singular $\Sigma$ has no inverse and $\log|\Sigma|=-\infty$, so the density blows up.",
  r"Adding $\varepsilon\mathbf I$ makes every eigenvalue at least $\varepsilon$: positive definite, invertible, and Cholesky works."))

q("B8","B",1,r"""You draw <code>Z</code> of shape $(5,2)$ from $\mathcal N(\mathbf 0,\mathbf I)$ and have $\mathbf L$ ($2\times2$) and <code>mean</code> of shape $(2,)$. Write the sampling line and give the shape of every step. Why <code>L.T</code>?""",
S(r"One sample as a column: $\boldsymbol\mu+\mathbf L\mathbf z$.",
  r"Stacking samples as rows transposes it: $\mathbf z^\top\mathbf L^\top$, so the batch is <code>Z @ L.T</code>: $(5,2)@(2,2)\to(5,2)$.",
  r"<code>mean</code> $(2,)$ broadcasts over the 5 rows: result $(5,2)$.",
  r"Using <code>Z @ L</code> would give the covariance $\mathbf L^\top\mathbf L$, which is wrong."))

q("B9","B",2,r"""Points $\mathbf x_1=(1,0)$, $\mathbf x_2=(0,2)$; means $\boldsymbol\mu_1=(0,0)$, $\boldsymbol\mu_2=(1,1)$. Build the $2\times2$ squared-distance matrix with the trick and assign each point.""",
S(r"$\|\mathbf x\|^2=(1,4)$; $\|\boldsymbol\mu\|^2=(0,2)$.",
  r"Cross terms $\mathbf X\mathbf M^\top=\begin{bmatrix}0&1\\0&2\end{bmatrix}$.",
  r"$D_{ij}=\|\mathbf x_i\|^2-2(\mathbf X\mathbf M^\top)_{ij}+\|\boldsymbol\mu_j\|^2$: $\begin{bmatrix}1-0+0&1-2+2\\4-0+0&4-4+2\end{bmatrix}=\begin{bmatrix}1&1\\4&2\end{bmatrix}$.",
  r"Row 1 is a tie; <code>argmin</code> returns the first: cluster 1. Row 2: cluster 2."))

# =====================================================================
# C. Tensors and NumPy
# =====================================================================
q("C1","C",1,r"""Give the result shape, or say it fails: (a) $(3,1)+(1,4)$ (b) $(5,)+(3,1)$ (c) $(4,)+(3,)$ (d) $(6700,784)-(784,)$ (e) $(10,)-(10,1)$""",
S(r"Rule: align shapes from the right; each pair must be equal or one of them 1; a missing size counts as 1.",
  r"(a) $3$ vs $1$, $1$ vs $4$ → $(3,4)$.",
  r"(b) $(5,)\to(1,5)$; with $(3,1)$ → $(3,5)$.",
  r"(c) $4$ vs $3$, neither is 1 → fails.",
  r"(d) $(784,)\to(1,784)$, applied to every row → $(6700,784)$.",
  r"(e) $(10,)\to(1,10)$ against $(10,1)$ → $(10,10)$: a silent bug, not an error."))

q("C2","C",1,r"""$\mathbf X$ has shape $(10,3)$. Give the shapes of <code>X.mean(axis=0)</code>, <code>X.sum(axis=1)</code>, <code>X.T @ X</code> and <code>X @ np.ones(3)</code>.""",
S(r"The axis you name disappears.",
  r"<code>X.mean(axis=0)</code>: rows removed → $(3,)$.",
  r"<code>X.sum(axis=1)</code>: columns removed → $(10,)$.",
  r"<code>X.T @ X</code>: $(3,10)@(10,3)\to(3,3)$.",
  r"<code>X @ np.ones(3)</code>: $(10,3)@(3,)\to(10,)$ (row sums)."))

q("C3","C",1,r"""<code>images</code> has shape $(6700,28,28)$. How do you flatten each image into a vector without hard-coding 6700? What shape does <code>np.arange(784).reshape(-1,28)</code> have?""",
S(r"<code>images.reshape(-1, 784)</code>: the <code>-1</code> is computed from the total size, giving $(6700,784)$.",
  r"<code>np.arange(784)</code> has 784 entries; $784/28=28$, so <code>reshape(-1,28)</code> gives $(28,28)$."))

q("C4","C",1,r"""<code>h = np.array([4,2,6,1,3,5,7,3,8,2])</code>, <code>m = h &gt;= 5</code>. Give <code>m</code>, <code>h[m]</code>, <code>m.sum()</code>, and <code>h[m].mean()</code>.""",
S(r"Compare elementwise: <code>m</code> = F,F,T,F,F,T,T,F,T,F.",
  r"Keep the True entries: <code>h[m]</code> = [6,5,7,8].",
  r"True counts as 1: <code>m.sum()</code> = 4.",
  r"Mean of [6,5,7,8] $=26/4=6.5$."))

q("C5","C",2,r"""<code>y</code> has shape $(D,)$ and your model returns <code>pred</code> of shape $(D,1)$. What does <code>(pred - y).mean()</code> compute and why is it wrong? Give two fixes.""",
S(r"Broadcast: $(D,1)$ against $(1,D)$ → $(D,D)$, all pairwise differences.",
  r"The mean of that matrix is meaningless and no error is raised.",
  r"Fix 1: <code>pred.ravel() - y</code> (or <code>pred[:,0]</code>).",
  r"Fix 2: <code>pred - y[:, None]</code>. Always print <code>.shape</code> when a loss looks off."))

q("C6","C",2,r"""$\mathbf X$ has shape $(4,3)$. (a) What does <code>X - X.mean(axis=0)</code> do and what shape is it? (b) Does <code>X / X.sum(axis=1)</code> work? If not, fix it.""",
S(r"(a) <code>X.mean(axis=0)</code> is $(3,)$, one mean per column; it broadcasts over rows: $(4,3)$. Every column now has mean 0 (centering).",
  r"(b) <code>X.sum(axis=1)</code> is $(4,)$. Align from the right: $3$ vs $4$ → error.",
  r"Fix: keep the axis: <code>X / X.sum(axis=1, keepdims=True)</code>, i.e. $(4,3)/(4,1)$ → $(4,3)$; each row now sums to 1."))

q("C7","C",2,r"""<code>X</code> is $(100,3)$, <code>theta</code> is $(3,)$, <code>y</code> is $(100,)$. Give the shape at each step of <code>((X @ theta - y)**2).mean()</code>. What changes if <code>theta</code> is $(3,1)$?""",
S(r"<code>X @ theta</code>: $(100,)$. Minus <code>y</code>: $(100,)$. Squared: $(100,)$. Mean: scalar.",
  r"With <code>theta</code> $(3,1)$: <code>X @ theta</code> is $(100,1)$.",
  r"$(100,1)-(100,)$ broadcasts to $(100,100)$: the loss is silently wrong.",
  r"Keep parameter vectors 1-D, or reshape <code>y</code> to $(100,1)$ consistently."))

q("C8","C",1,r"""<code>s = np.array([66,48,71,45])</code>. Compute <code>t = (s &gt;= 65).astype(int)</code>, <code>s[t == 1].mean()</code>, and say what <code>np.where(s &gt;= 65, 'P', 'F')</code> returns.""",
S(r"Comparison: [True, False, True, False]; as integers $t=[1,0,1,0]$.",
  r"<code>t == 1</code> selects 66 and 71; their mean is $68.5$.",
  r"<code>np.where</code> picks elementwise: ['P','F','P','F'] (an if/else without a loop)."))

q("C9","C",2,r"""<code>a = np.arange(5); b = a[1:3]; b[0] = 99</code>. What is <code>a</code>? Then <code>c = a[[1,2]]; c[0] = 7</code>. What is <code>a</code> now? Explain.""",
S(r"A slice is a <strong>view</strong>: <code>b</code> shares memory with <code>a</code>.",
  r"So <code>b[0]=99</code> changes <code>a[1]</code>: <code>a</code> = [0, 99, 2, 3, 4].",
  r"Fancy indexing with a list makes a <strong>copy</strong>: <code>c</code> is new memory.",
  r"So <code>c[0]=7</code> leaves <code>a</code> = [0, 99, 2, 3, 4]. Use <code>.copy()</code> when you need independence."))

# =====================================================================
# D. Probability and MLE
# =====================================================================
q("D1","D",1,r"""A disease has prevalence 1%. A test has sensitivity 90% and false-positive rate 5%. A person tests positive. Find $P(\text{sick}|+)$.""",
S(r"Write what you know: $P(\text{sick})=0.01$, $P(+|\text{sick})=0.9$, $P(+|\text{healthy})=0.05$.",
  r"Total probability: $P(+)=0.9\cdot0.01+0.05\cdot0.99=0.009+0.0495=0.0585$.",
  r"Bayes: $P(\text{sick}|+)=0.009/0.0585=0.154$.",
  r"Interpret: only about 15%, because healthy people vastly outnumber the sick."))

q("D2","D",1,r"""A coin gives 7 heads in 10 flips. (a) MLE of $\theta=P(\text{heads})$. (b) The log-likelihood at the MLE. (c) MAP with a Beta$(2,2)$ prior.""",
S(r"(a) Log-likelihood $7\log\theta+3\log(1-\theta)$; derivative $7/\theta-3/(1-\theta)=0$ gives $\hat\theta=0.7$.",
  r"(b) $7\ln0.7+3\ln0.3=-2.497-3.612=-6.109$.",
  r"(c) Add the prior’s log $(\alpha-1)\log\theta+(\beta-1)\log(1-\theta)$: $\theta_{\text{MAP}}=\frac{H+\alpha-1}{D+\alpha+\beta-2}=\frac8{12}=0.667$, pulled toward 0.5."))

q("D3","D",1,r"""Independent flips $(1,0,1)$ with $P(1)=p$. Write the likelihood and the log-likelihood, name the sheet identities, and find the MLE.""",
S(r"Independence multiplies: $p\cdot(1-p)\cdot p=p^2(1-p)$.",
  r"Log (identities 1 and 2): $2\log p+\log(1-p)$.",
  r"Derivative: $2/p-1/(1-p)=0\Rightarrow2(1-p)=p\Rightarrow p=2/3$."))

q("D4","D",2,r"""Explain in two sentences why we maximize the <em>log</em> of the likelihood, and why the loss has a minus sign.""",
S(r"The likelihood of independent data is a product of many small numbers; the log turns it into a sum (easy to differentiate, no underflow).",
  r"Log is increasing, so the maximizer does not move.",
  r"Optimizers minimize, so we negate: loss = negative log-likelihood."))

q("D5","D",1,r"""$X$ takes values $0,1,2$ with probabilities $0.2,0.5,0.3$. Find $\mathbb E[X]$, $\mathbb E[X^2]$ and $\text{Var}(X)$.""",
S(r"$\mathbb E[X]=0(0.2)+1(0.5)+2(0.3)=1.1$.",
  r"$\mathbb E[X^2]=0+1(0.5)+4(0.3)=1.7$.",
  r"$\text{Var}(X)=\mathbb E[X^2]-\mathbb E[X]^2=1.7-1.21=0.49$."))

q("D6","D",2,r"""Counts $2,0,3,1$ come from a Poisson$(\lambda)$ with no features. Derive the MLE of $\lambda$ and evaluate it.""",
S(r"Log-likelihood: $\sum_i[y_i\log\lambda-\lambda-\log y_i!]$.",
  r"Derivative: $\sum_iy_i/\lambda-n=0$.",
  r"So $\hat\lambda=\frac1n\sum y_i$, the sample mean.",
  r"$\hat\lambda=(2+0+3+1)/4=1.5$."))

q("D7","D",1,r"""A joint table: $p(a_1,b_1)=0.12$, $p(a_1,b_2)=0.28$, $p(a_2,b_1)=0.18$, $p(a_2,b_2)=0.42$. Find the marginals. Are $A$ and $B$ independent?""",
S(r"Marginal of $A$ (sum rows): $p(a_1)=0.4$, $p(a_2)=0.6$.",
  r"Marginal of $B$ (sum columns): $p(b_1)=0.3$, $p(b_2)=0.7$.",
  r"Check every cell: $0.4\cdot0.3=0.12$, $0.4\cdot0.7=0.28$, $0.6\cdot0.3=0.18$, $0.6\cdot0.7=0.42$ ✓.",
  r"All match, so $p(a,b)=p(a)p(b)$: independent. One mismatch would be enough to say ‘not independent’."))

q("D8","D",1,r"""$X$ is uniform on $[0,0.5]$. What is its density? Is a density above 1 a problem? Find $P(0.1\le X\le0.2)$.""",
S(r"Area must be 1: height $\times0.5=1$, so $f(x)=2$ on $[0,0.5]$.",
  r"A density is not a probability; it can exceed 1. Only areas are probabilities.",
  r"$P(0.1\le X\le0.2)=2\times0.1=0.2$."))

q("D9","D",1,r"""20% of email is spam. The word ‘winner’ appears in 60% of spam and 5% of normal mail. An email contains ‘winner’. Find $P(\text{spam}|\text{winner})$.""",
S(r"$P(\text{winner})=0.6\cdot0.2+0.05\cdot0.8=0.12+0.04=0.16$.",
  r"Bayes: $P(\text{spam}|\text{winner})=0.12/0.16=0.75$."))

q("D10","D",2,r"""Ten students: 6 fail, 2 pass, 2 distinction. Find the MLE of the Categorical probabilities. Why is it the fraction in each class?""",
S(r"Log-likelihood: $6\log\theta_0+2\log\theta_1+2\log\theta_2$ with $\sum\theta_k=1$.",
  r"Maximize under the constraint (Lagrange multiplier): $n_k/\theta_k=\lambda$ for every $k$, so $\theta_k\propto n_k$.",
  r"Normalize: $\hat{\boldsymbol\theta}=(0.6,0.2,0.2)$.",
  tip="This is a Lagrangian, the same tool as Lecture 20."))

# =====================================================================
# E. Linear algebra and linear regression
# =====================================================================
q("E1","E",2,r"""Data $(x,y)=(1,2),(2,4),(3,5)$. Fit $y=\theta_0+\theta_1x$ by least squares and predict at $x=4$.""",
S(r"Means: $\bar x=2$, $\bar y=11/3=3.667$.",
  r"$S_{xy}=\sum(x-\bar x)(y-\bar y)=(-1)(-1.667)+0+(1)(1.333)=3$; $S_{xx}=1+0+1=2$.",
  r"Slope $\theta_1=S_{xy}/S_{xx}=1.5$; intercept $\theta_0=\bar y-1.5\bar x=0.667$.",
  r"Prediction at $x=4$: $0.667+6=6.667$."))

q("E2","E",1,r"""$\mathbf X$ is $(D,n)$, $\boldsymbol\theta$ is $(n,)$, $\mathbf y$ is $(D,)$. Give the shape of every step of $\nabla\text{MSE}=\frac2D\mathbf X^\top(\mathbf X\boldsymbol\theta-\mathbf y)$.""",
S(r"$\mathbf X\boldsymbol\theta$: $(D,)$; minus $\mathbf y$: $(D,)$.",
  r"$\mathbf X^\top$ is $(n,D)$; times a $(D,)$ vector gives $(n,)$.",
  r"Same shape as $\boldsymbol\theta$, as every gradient must be."))

q("E3","E",2,r"""When is $\mathbf X^\top\mathbf X$ not invertible, and what do you do?""",
S(r"When columns of $\mathbf X$ are linearly dependent (repeated or redundant features).",
  r"When there are fewer data points than features.",
  r"Remedies: drop features, use the pseudo-inverse, add ridge $\lambda\mathbf I$ (always invertible for $\lambda>0$), or use gradient descent."))

q("E4","E",2,r"""Solve the normal equations for $(1,2),(2,4),(3,5)$ with $\mathbf X=[\mathbf 1,\mathbf x]$: compute $\mathbf X^\top\mathbf X$, $\mathbf X^\top\mathbf y$ and $\hat{\boldsymbol\theta}$.""",
S(r"$\mathbf X^\top\mathbf X=\begin{bmatrix}3&6\\6&14\end{bmatrix}$ (count, $\sum x$; $\sum x$, $\sum x^2$).",
  r"$\mathbf X^\top\mathbf y=(\sum y,\sum xy)=(11,25)$.",
  r"Determinant $3\cdot14-6\cdot6=6$; inverse $\frac16\begin{bmatrix}14&-6\\-6&3\end{bmatrix}$.",
  r"$\hat{\boldsymbol\theta}=\frac16(14\cdot11-6\cdot25,\ -6\cdot11+3\cdot25)=\frac16(4,9)=(0.667,1.5)$ ✓ matches E1."))

q("E5","E",2,r"""Same data, $\boldsymbol\theta=(0,1)$. Compute the MSE and $\nabla\text{MSE}=\frac2D\mathbf X^\top(\mathbf X\boldsymbol\theta-\mathbf y)$.""",
S(r"Predictions $\mathbf X\boldsymbol\theta=(1,2,3)$; errors $\mathbf X\boldsymbol\theta-\mathbf y=(-1,-2,-2)$.",
  r"MSE $=(1+4+4)/3=3$.",
  r"$\mathbf X^\top(\text{errors})=(-1-2-2,\ -1-4-6)=(-5,-11)$.",
  r"Multiply by $2/3$: $(-3.333,-7.333)$. Both negative: increase intercept and slope."))

q("E6","E",1,r"""With basis $\boldsymbol\phi(x)=(1,x,x^2)$ and $\boldsymbol\theta=(1,-1,0.5)$, predict at $x=2$. Why is this still ‘linear’ regression?""",
S(r"$\boldsymbol\phi(2)=(1,2,4)$.",
  r"$\boldsymbol\theta^\top\boldsymbol\phi=1-2+2=1$.",
  r"The model is linear in $\boldsymbol\theta$ (a weighted sum), so the same normal equations and MSE apply, even though it is a curve in $x$."))

q("E7","E",2,r"""$J(\boldsymbol\theta)=\tfrac12\boldsymbol\theta^\top\mathbf A\boldsymbol\theta$ with $\mathbf A=\text{diag}(100,1)$. Give the condition number, the largest stable step size, and explain why gradient descent is slow here.""",
S(r"Eigenvalues are the diagonal: $100$ and $1$. Condition number $\kappa=100/1=100$.",
  r"Stability needs $\alpha<2/\lambda_{\max}=2/100=0.02$.",
  r"Along the flat direction each step multiplies the error by $1-\alpha\cdot1\ge0.98$: about 100s of steps.",
  r"Large $\kappa$ = a long narrow valley: the steep direction caps $\alpha$, the flat one is then slow."))

q("E8","E",1,r"""$\mathbf v=(3,-4)$. Give $\|\mathbf v\|_1$, $\|\mathbf v\|_2$, $\|\mathbf v\|_\infty$. Find the distance between $(1,2)$ and $(4,6)$.""",
S(r"$\|\mathbf v\|_1=|3|+|-4|=7$.",
  r"$\|\mathbf v\|_2=\sqrt{9+16}=5$.",
  r"$\|\mathbf v\|_\infty=\max(3,4)=4$.",
  r"Difference $(3,4)$, length $5$."))

q("E9","E",1,r"""$\mathbf A$ is $2\times3$, $\mathbf B$ is $3\times4$. Give the shapes of $\mathbf{AB}$ and $(\mathbf{AB})^\top$. Is $\mathbf A^\top\mathbf B$ defined?""",
S(r"$\mathbf{AB}$: inner sizes $3=3$ → $2\times4$.",
  r"$(\mathbf{AB})^\top=\mathbf B^\top\mathbf A^\top$: $(4\times3)(3\times2)=4\times2$.",
  r"$\mathbf A^\top$ is $3\times2$; with $\mathbf B$ $3\times4$ the inner sizes $2\ne3$: not defined."))

q("E10","E",2,r"""$\mathbf A=\begin{bmatrix}2&1\\1&2\end{bmatrix}$. Find its eigenvalues. Is it positive definite? What is its condition number?""",
S(r"$\det(\mathbf A-\lambda\mathbf I)=(2-\lambda)^2-1=0\Rightarrow\lambda=1$ or $3$.",
  r"Both positive → positive definite ($\mathbf v^\top\mathbf A\mathbf v>0$ for all $\mathbf v\ne\mathbf 0$).",
  r"Condition number $3/1=3$."))

# =====================================================================
# F. Logistic, softmax, metrics
# =====================================================================
q("F1","F",1,r"""$z=\boldsymbol\theta^\top\mathbf x=0.8$. Find $\sigma(z)$, the loss if $y=1$, and the loss if $y=0$.""",
S(r"$\sigma(0.8)=1/(1+e^{-0.8})=1/1.449=0.690$.",
  r"$y=1$: $-\ln0.690=0.371$.",
  r"$y=0$: $-\ln(1-0.690)=-\ln0.310=1.171$. The loss is larger when the prediction disagrees."))

q("F2","F",1,r"""Logits $(2,1,0)$, true class is the first. Compute the softmax and the cross-entropy loss.""",
S(r"$e^2=7.389$, $e^1=2.718$, $e^0=1$; sum $11.107$.",
  r"Softmax $=(0.665,0.245,0.090)$.",
  r"Loss $=-\ln0.665=0.408$."))

q("F3","F",1,r"""TP 40, FP 10, FN 20, TN 30. Compute accuracy, precision, recall, $F_1$ and false-positive rate.""",
S(r"Accuracy $=(40+30)/100=0.70$.",
  r"Precision $=40/(40+10)=0.80$.",
  r"Recall $=40/(40+20)=0.667$.",
  r"$F_1=\frac{2(0.8)(0.667)}{0.8+0.667}=0.727$.",
  r"FPR $=10/(10+30)=0.25$."))

q("F4","F",1,r"""Logistic model $\boldsymbol\theta=(-3,1)$ on features $(1,h)$. Probabilities at $h=3$ and $h=5$? Where is the decision boundary?""",
S(r"$z=-3+h$.",
  r"$h=3$: $z=0$, $p=0.5$.",
  r"$h=5$: $z=2$, $p=\sigma(2)=0.881$.",
  r"Boundary where $z=0$: $h=3$."))

q("F5","F",2,r"""Two points: $\mathbf x_1=(1,1)$, $y_1=1$; $\mathbf x_2=(1,-1)$, $y_2=0$. Start at $\boldsymbol\theta=(0,0)$. Compute the mean BCE gradient and one step with $\alpha=1$.""",
S(r"$z=0$ for both, so $p_1=p_2=0.5$.",
  r"Per point $(p-y)\mathbf x$: $(-0.5)(1,1)=(-0.5,-0.5)$ and $(0.5)(1,-1)=(0.5,-0.5)$.",
  r"Mean: $(0,-0.5)$.",
  r"Step: $\boldsymbol\theta=(0,0)-1\cdot(0,-0.5)=(0,0.5)$: now $z_1=0.5>0$ and $z_2=-0.5<0$, both correct."))

q("F6","F",2,r"""Compute softmax$(1001,1000,999)$ by hand. Why does code subtract the maximum first?""",
S(r"Softmax ignores adding a constant to every logit: subtract 1001 → $(0,-1,-2)$, same as $(2,1,0)$.",
  r"Result $(0.665,0.245,0.090)$ (F2).",
  r"$e^{1001}$ overflows to <code>inf</code> and gives <code>nan</code>; after subtracting the max the largest term is $e^0=1$."))

q("F7","F",2,r"""Scores $0.9,0.8,0.6,0.4,0.3$ with true labels $1,0,1,1,0$. Find precision, recall and FPR at threshold 0.5 and at 0.35.""",
S(r"Threshold 0.5: predict positive for 0.9, 0.8, 0.6. TP $=2$ (0.9, 0.6), FP $=1$ (0.8), FN $=1$ (0.4), TN $=1$.",
  r"Precision $2/3=0.667$, recall $2/3=0.667$, FPR $1/2=0.5$.",
  r"Threshold 0.35: add 0.4 (a positive). TP $=3$, FP $=1$, FN $=0$, TN $=1$.",
  r"Precision $3/4=0.75$, recall $1$, FPR $0.5$. Lowering the threshold raised recall here without new false alarms.",
  tip="Each threshold is one point on the ROC curve (recall against FPR)."))

q("F8","F",2,r"""Precision $0.8$, recall $0.5$. Compute $F_1$ and $F_2$ with $F_\beta=(1+\beta^2)\frac{PR}{\beta^2P+R}$.""",
S(r"$PR=0.4$.",
  r"$F_1=2(0.4)/(0.8+0.5)=0.615$.",
  r"$F_2=5(0.4)/(4\cdot0.8+0.5)=2/3.7=0.541$: lower, because $F_2$ weights recall more and recall is the weak one.",
  tip=r"The slide’s $F_\beta$ has a sign error. With $\beta=1$ the correct formula must give $F_1$."))

q("F9","F",1,r"""$\boldsymbol\theta=(-4,1,2)$ on features $(1,x_1,x_2)$. Write the decision boundary and classify $(2,2)$ with its probability.""",
S(r"Boundary $z=0$: $-4+x_1+2x_2=0$, i.e. $x_1+2x_2=4$ (a line).",
  r"At $(2,2)$: $z=-4+2+4=2$.",
  r"$p=\sigma(2)=0.881>0.5$: class 1."))

q("F10","F",2,r"""(a) Why not use squared error with a sigmoid for classification? (b) What happens to logistic-regression weights when the training data are perfectly separable?""",
S(r"(a) Cross-entropy is the Bernoulli negative log-likelihood: convex in $\boldsymbol\theta$ with gradient $(p-y)\mathbf x$.",
  r"Squared error through a sigmoid is non-convex and its gradient has an extra $\sigma'$ factor that vanishes for confident mistakes.",
  r"(b) The loss keeps falling as $\|\boldsymbol\theta\|\to\infty$ (probabilities pushed to 0 and 1), so the MLE does not exist.",
  r"Fix: regularization (a prior) or early stopping."))

# =====================================================================
# G. Generalization and statistics
# =====================================================================
q("G1","G",1,r"""Training error 2%, validation error 15%. Diagnose it and give two fixes. What if both were 20%?""",
S(r"Large gap between train and validation: overfitting (high variance).",
  r"Fixes: more data, regularization (ridge/lasso/dropout), fewer features, early stopping.",
  r"Both 20%: underfitting (high bias). Use a richer model or better features, or reduce regularization."))

q("G2","G",1,r"""5-fold cross-validation on 100 points. How many models are fit, and on how many points is each trained and tested?""",
S(r"Split into 5 folds of 20.",
  r"5 fits; each trains on 80 and tests on the remaining 20.",
  r"Report the mean (and an error bar) of the 5 scores. The final test set stays untouched."))

q("G3","G",2,r"""Five runs give a mean accuracy of 80 with standard deviation 5. Compute the standard error and an approximate 95% interval.""",
S(r"SEM $=\hat\sigma/\sqrt n=5/\sqrt5=2.236$.",
  r"95% interval $\approx80\pm1.96(2.236)=80\pm4.4$.",
  r"Caveat: with only 5 runs the true coverage of $\pm1$ SEM is below 68%."))

q("G4","G",1,r"""Compute the entropy in bits of $p=(0.5,0.25,0.25)$.""",
S(r"$H=-\sum p\log_2p$.",
  r"$-0.5\log_20.5=0.5$; $-0.25\log_20.25=0.5$ each.",
  r"$H=0.5+0.5+0.5=1.5$ bits."))

q("G5","G",1,r"""Polynomial degree vs error: degree 1 (train 30, val 32), degree 3 (train 10, val 12), degree 9 (train 1, val 25). Which degree do you choose, and what is wrong with the others?""",
S(r"Choose by validation error: degree 3 (12).",
  r"Degree 1: both errors high → underfitting (bias).",
  r"Degree 9: tiny train, big validation → overfitting (variance).",
  r"Then report the chosen model once on the test set."))

q("G6","G",1,r"""Leave-one-out cross-validation on 50 points: how many fits, and what is each trained on? Why is it expensive?""",
S(r"$k=n=50$ folds of size 1.",
  r"50 fits, each on 49 points, tested on 1.",
  r"Cost grows with $n$ fits; it is nearly unbiased but the 50 scores are highly correlated."))

q("G7","G",3,r"""Model A: mean 82, sd 4 over 5 runs. Model B: mean 79, sd 3 over 5 runs. Is the difference convincing at about the 5% level?""",
S(r"SEM$_A=4/\sqrt5=1.789$; SEM$_B=3/\sqrt5=1.342$.",
  r"Standard error of the difference $=\sqrt{1.789^2+1.342^2}=\sqrt{3.2+1.8}=2.236$.",
  r"Difference $82-79=3$, which is $3/2.236=1.34$ standard errors.",
  r"Below 1.96, so not convincing; run more seeds before claiming A is better."))

q("G8","G",2,r"""Compute the entropy of a fair coin and of $(0.9,0.1)$ in bits. Then the cross-entropy loss when the truth is class 1 and the model gives $(0.8,0.2)$.""",
S(r"Fair coin: $-2(0.5\log_20.5)=1$ bit.",
  r"$(0.9,0.1)$: $-(0.9\log_20.9+0.1\log_20.1)=0.137+0.332=0.469$ bits (less uncertain).",
  r"Cross-entropy with a one-hot truth is $-\log q_{\text{true}}=-\log_20.8=0.322$ bits ($0.223$ nats)."))

q("G9","G",2,r"""A teammate standardizes features using the mean and sd of the <em>whole</em> data set, then splits into train and test, and tries 20 settings on the test set. Name both mistakes and the fix.""",
S(r"Mistake 1: leakage. Test data influenced the scaling; fit the scaler on training data only, then apply it to validation and test.",
  r"Mistake 2: tuning on the test set makes the test score optimistic (it was used for selection).",
  r"Fix: train / validation (or cross-validation) for choosing settings; touch the test set once at the end."))

q("G10","G",1,r"""For data $2,4,6$ compute the variance with $1/n$ and with $1/(n-1)$. Which is the MLE, which is unbiased?""",
S(r"Squared deviations sum to $8$.",
  r"$1/n$: $8/3=2.667$ — the MLE (slightly too small on average).",
  r"$1/(n-1)$: $8/2=4$ — unbiased for the variance."))

# =====================================================================
# H. MAP, ridge, lasso
# =====================================================================
q("H1","H",2,r"""One-feature regression without intercept, data $(1,2),(2,4),(3,5)$. Fit $\theta$ with ridge penalty $\lambda\theta^2$ for $\lambda=0$ and $\lambda=2$.""",
S(r"Loss $\sum(y-\theta x)^2+\lambda\theta^2$; derivative $-2\sum x(y-\theta x)+2\lambda\theta=0$.",
  r"Solve: $\theta=\dfrac{\sum xy}{\sum x^2+\lambda}$.",
  r"$\sum xy=2+8+15=25$; $\sum x^2=1+4+9=14$.",
  r"$\lambda=0$: $25/14=1.786$. $\lambda=2$: $25/16=1.5625$ — shrunk toward 0."))

q("H2","H",2,r"""Coordinate-descent lasso: the unpenalized value is $c=3$ with threshold $t=1$; another coordinate has $c=0.7$. Apply soft thresholding to both.""",
S(r"$\text{soft}(c,t)=\text{sign}(c)\max(|c|-t,0)$.",
  r"$\text{soft}(3,1)=\max(2,0)=2$.",
  r"$\text{soft}(0.7,1)=\max(-0.3,0)=0$.",
  r"Lasso shrinks big weights by $t$ and sets small ones exactly to 0: feature selection."))

q("H3","H",1,r"""Complete: (a) A Gaussian prior on weights gives ___ regularization. (b) A Laplace prior gives ___. (c) Which one produces exact zeros? (d) What does a bigger $\lambda$ do to bias and variance?""",
S(r"(a) Ridge, $L_2$ penalty $\lambda\|\boldsymbol\theta\|_2^2$.",
  r"(b) Lasso, $L_1$ penalty $\lambda\|\boldsymbol\theta\|_1$.",
  r"(c) Lasso (the corner of the $L_1$ ball).",
  r"(d) More bias, less variance (a stronger prior)."))

q("H4","H",3,r"""Derive ridge regression from MAP: $y|\mathbf x\sim\mathcal N(\boldsymbol\theta^\top\mathbf x,\sigma^2)$ and prior $\theta_j\sim\mathcal N(0,\tau^2)$. What is $\lambda$?""",
S(r"MAP maximizes $p(\mathcal D|\boldsymbol\theta)p(\boldsymbol\theta)$, i.e. minimizes $-\log p(\mathcal D|\boldsymbol\theta)-\log p(\boldsymbol\theta)$.",
  r"Likelihood part: $\frac1{2\sigma^2}\sum_i(y_i-\boldsymbol\theta^\top\mathbf x_i)^2+$ const.",
  r"Prior part: $-\log p(\boldsymbol\theta)=\frac1{2\tau^2}\sum_j\theta_j^2+$ const.",
  r"Multiply by $2\sigma^2$: $\sum_i(y_i-\boldsymbol\theta^\top\mathbf x_i)^2+\frac{\sigma^2}{\tau^2}\|\boldsymbol\theta\|_2^2$.",
  r"So $\lambda=\sigma^2/\tau^2$: a narrow prior (small $\tau$) means strong shrinkage."))

q("H5","H",2,r"""7 heads in 10 flips. MAP with (a) Beta$(1,1)$, (b) Beta$(5,5)$. Explain the difference.""",
S(r"Formula: $\theta_{\text{MAP}}=\frac{H+\alpha-1}{D+\alpha+\beta-2}$.",
  r"(a) $\frac{7+0}{10+0}=0.7$: the uniform prior adds nothing, MAP = MLE.",
  r"(b) $\frac{7+4}{10+8}=\frac{11}{18}=0.611$: a stronger belief in a fair coin pulls the estimate toward 0.5.",
  r"More data would shrink the prior’s influence."))

q("H6","H",2,r"""$\mathbf X^\top\mathbf X=\text{diag}(2,8)$ and $\mathbf X^\top\mathbf y=(4,8)$. Compute least squares and ridge with $\lambda=2$.""",
S(r"Least squares: $\hat{\boldsymbol\theta}=(4/2,\ 8/8)=(2,1)$.",
  r"Ridge: $(\mathbf X^\top\mathbf X+2\mathbf I)^{-1}\mathbf X^\top\mathbf y=(4/4,\ 8/10)=(1,0.8)$.",
  r"The weakly determined direction (eigenvalue 2) shrinks by half; the well-determined one (8) only by 20%."))

q("H7","H",1,r"""Soft-threshold with $t=1$: $c=-2.5$, $c=0.4$, $c=1.0$. Which features survive?""",
S(r"$\text{soft}(-2.5,1)=-1\cdot\max(1.5,0)=-1.5$ (sign kept).",
  r"$\text{soft}(0.4,1)=0$.",
  r"$\text{soft}(1.0,1)=0$ (exactly at the threshold).",
  r"Only the first feature survives."))

q("H8","H",1,r"""Ridge results: $\lambda=0.01$: train 1.0, val 4.0. $\lambda=1$: train 2.0, val 2.5. $\lambda=100$: train 6.0, val 6.5. Which $\lambda$? Describe each.""",
S(r"Pick by validation: $\lambda=1$ (2.5).",
  r"$\lambda=0.01$: big gap → overfitting.",
  r"$\lambda=100$: both high → underfitting (too much shrinkage).",
  r"Never pick by training error, which always prefers the smallest $\lambda$."))

q("H9","H",2,r"""Show that a Laplace prior $p(\theta_j)\propto\exp(-|\theta_j|/b)$ gives the lasso penalty.""",
S(r"$-\log p(\boldsymbol\theta)=\sum_j|\theta_j|/b+$ const.",
  r"Added to the negative log-likelihood: loss $+\frac1b\|\boldsymbol\theta\|_1$.",
  r"So $\lambda\propto1/b$: a sharper prior (small $b$) gives a stronger $L_1$ penalty."))

# =====================================================================
# I. Neural networks
# =====================================================================
q("I1","I",1,r"""A network has 4 inputs, 5 hidden units and 3 outputs (with biases). How many parameters?""",
S(r"Input → hidden: each of 5 units has 4 weights + 1 bias: $5\cdot5=25$.",
  r"Hidden → output: each of 3 outputs has 5 weights + 1 bias: $3\cdot6=18$.",
  r"Total $M(D+1)+C(M+1)=43$."))

q("I2","I",2,r"""Input $\mathbf x=(1,2)$. Hidden layer $\mathbf W=\begin{bmatrix}1&-1\\0.5&0.5\end{bmatrix}$, $\mathbf b=(0,-1)$, ReLU. Output $\mathbf w=(2,4)$, bias $0.5$, identity. Compute the output.""",
S(r"$\mathbf z=\mathbf W\mathbf x+\mathbf b=(1-2+0,\ 0.5+1-1)=(-1,0.5)$.",
  r"ReLU: $(0,0.5)$.",
  r"Output $=2\cdot0+4\cdot0.5+0.5=2.5$."))

q("I3","I",1,r"""(a) Why can one perceptron not learn XOR? (b) What if every hidden activation is the identity? (c) Why do deep sigmoid networks train badly? (d) Drop probability 0.3: test-time scaling?""",
S(r"(a) No single straight line separates XOR’s classes.",
  r"(b) A product of linear maps is linear: the layers collapse into one.",
  r"(c) $\sigma'\le\frac14$, so each layer shrinks the backward signal at least 4×: vanishing gradients.",
  r"(d) Multiply by the keep probability $1-0.3=0.7$."))

q("I4","I",1,r"""Match the output activation and loss: (a) real-valued label, (b) binary label, (c) $C$-class label, (d) count label.""",
S(r"(a) Identity + squared error (Normal NLL).",
  r"(b) Sigmoid + binary cross-entropy (Bernoulli NLL).",
  r"(c) Softmax + categorical cross-entropy (Categorical NLL).",
  r"(d) Exp + Poisson NLL. Same recipe every time: the head is the distribution’s parameter."))

q("I5","I",3,r"""One hidden ReLU unit: $a=w_1x+b_1$, $h=\text{ReLU}(a)$, $\hat y=w_2h+b_2$, loss $\tfrac12(\hat y-y)^2$. With $x=1$, $w_1=2$, $b_1=-1$, $w_2=3$, $b_2=0.5$, $y=3$, run the forward pass and backpropagate to every parameter.""",
S(r"Forward: $a=2-1=1$, $h=1$, $\hat y=3+0.5=3.5$, loss $\tfrac12(0.5)^2=0.125$.",
  r"Start backward: $\partial L/\partial\hat y=\hat y-y=0.5$.",
  r"Output layer: $\partial L/\partial w_2=0.5\cdot h=0.5$; $\partial L/\partial b_2=0.5$.",
  r"Into the hidden unit: $\partial L/\partial h=0.5\cdot w_2=1.5$; ReLU passes it because $a>0$: $\partial L/\partial a=1.5$.",
  r"First layer: $\partial L/\partial w_1=1.5\cdot x=1.5$; $\partial L/\partial b_1=1.5$.",
  tip="Backprop = multiply local derivatives from the loss backward. If $a<0$, ReLU would block every gradient below it."))

q("I6","I",1,r"""Count the parameters of an MLP $784\to128\to64\to10$ with biases.""",
S(r"$128(784+1)=100{,}480$.",
  r"$64(128+1)=8{,}256$.",
  r"$10(64+1)=650$.",
  r"Total $109{,}386$."))

q("I7","I",2,r"""Perceptron with $\mathbf w=(0,0)$, $b=0$; update $\mathbf w\leftarrow\mathbf w+y\mathbf x$, $b\leftarrow b+y$ when $y(\mathbf w^\top\mathbf x+b)\le0$. Process $\mathbf x=(1,2)$, $y=+1$, then $\mathbf x=(2,-1)$, $y=-1$.""",
S(r"Point 1: $y(\mathbf w^\top\mathbf x+b)=0\le0$ → mistake. $\mathbf w=(1,2)$, $b=1$.",
  r"Point 2: $\mathbf w^\top\mathbf x+b=2-2+1=1$; times $y=-1$ gives $-1\le0$ → mistake.",
  r"Update: $\mathbf w=(1,2)-(2,-1)=(-1,3)$, $b=0$.",
  r"Check point 2 now: $-2-3+0=-5<0$ ✓ correct side."))

q("I8","I",1,r"""Compute $\sigma'(0)$ and $\sigma'(2)$. What is the largest factor 10 sigmoid layers can pass back?""",
S(r"$\sigma'=\sigma(1-\sigma)$. At 0: $0.5\cdot0.5=0.25$ (the maximum).",
  r"At 2: $\sigma=0.881$, $\sigma'=0.881\cdot0.119=0.105$.",
  r"Ten layers: at most $0.25^{10}=9.5\times10^{-7}$: vanishing gradients. ReLU has slope 1 when active."))

q("I9","I",2,r"""Name four ways to regularize a neural network and say what each does.""",
S(r"Weight decay: add $\lambda\|\mathbf W\|^2$ (a Gaussian prior, as ridge).",
  r"Dropout: randomly zero units in training so no unit can rely on another.",
  r"Early stopping: stop when validation error starts to rise.",
  r"Data augmentation: add transformed copies (shifts, flips) so the model learns invariances."))

q("I10","I",2,r"""Dropout with drop probability $p=0.5$ on hidden activations $(2,4,6,8)$. In training the mask is $(1,0,1,0)$. What is used in training and at test time?""",
S(r"Training: multiply by the mask → $(2,0,6,0)$.",
  r"Test: no mask, scale by keep probability $0.5$ → $(1,2,3,4)$, so the expected input to the next layer matches training.",
  r"(Inverted dropout does the opposite: divide by $0.5$ in training, nothing at test.)"))

# =====================================================================
# J. Gradient descent
# =====================================================================
q("J1","J",2,r"""Minimize $J(\theta)=(\theta-2)^2$ from $\theta_0=0$ with $\alpha=0.25$. Give $\theta_1,\theta_2,\theta_3$. For which $\alpha$ does it converge?""",
S(r"$J'(\theta)=2(\theta-2)$.",
  r"$\theta_1=0-0.25(-4)=1$; $\theta_2=1-0.25(-2)=1.5$; $\theta_3=1.5-0.25(-1)=1.75$.",
  r"Recursion $\theta_{t+1}-2=(1-2\alpha)(\theta_t-2)$.",
  r"Converges iff $|1-2\alpha|<1$, i.e. $0<\alpha<1$ ($=2/\lambda_{\max}$ with curvature 2)."))

q("J2","J",1,r"""Why use mini-batches? State the cost, the benefit, and the step-size conditions.""",
S(r"Benefit: each step costs $B$ points, not all $D$; the mini-batch gradient is an unbiased estimate.",
  r"Cost: noise, shrinking like $1/\sqrt B$.",
  r"Robbins–Monro: $\sum\alpha_t=\infty$ and $\sum\alpha_t^2<\infty$ (e.g. $\alpha_t=1/t$)."))

q("J3","J",2,r"""Momentum $\mathbf m_t=\beta\mathbf m_{t-1}+\mathbf g_t$ with constant $g=1$, $\beta=0.9$: what does $m_t$ tend to? What happens to an oscillating component?""",
S(r"Fixed point $m=\beta m+g\Rightarrow m=g/(1-\beta)=10$.",
  r"So steps are 10× bigger along a consistent direction.",
  r"Components that flip sign cancel out in the running sum.",
  r"Too much momentum overshoots; tune $\alpha$ and $\beta$ together."))

q("J4","J",2,r"""Chain rule for $y=(3x+1)^2$ at $x=1$ with $a=3x+1$. Which autodiff pass computes which value?""",
S(r"Forward pass: $a=4$, $y=16$ (stored).",
  r"Backward pass: $dy/da=2a=8$, $da/dx=3$.",
  r"Multiply: $dy/dx=24$."))

q("J5","J",1,r"""True or false, with a reason: (a) MSE for linear regression is convex. (b) A neural-network loss is convex. (c) For a convex loss, small-step GD reaches the global minimum. (d) A larger learning rate always converges faster.""",
S(r"(a) True: Hessian $\frac2D\mathbf X^\top\mathbf X$ is positive semi-definite.",
  r"(b) False: hidden layers make it non-convex (many minima, symmetries).",
  r"(c) True.",
  r"(d) False: too large oscillates or diverges."))

q("J6","J",2,r"""$J=\theta_1^2+5\theta_2^2$, start $(1,1)$, $\alpha=0.1$. Do two steps. What is the stability limit?""",
S(r"Gradient $(2\theta_1,10\theta_2)$; at $(1,1)$: $(2,10)$.",
  r"Step 1: $(1,1)-0.1(2,10)=(0.8,0)$.",
  r"Step 2: gradient $(1.6,0)$ → $(0.64,0)$.",
  r"Curvatures 2 and 10: stable iff $\alpha<2/10=0.2$. At $\alpha=0.1$ the steep coordinate is solved in one step."))

q("J7","J",1,r"""Per-example gradients are $1,3,-1,5$. Full gradient? Batch $\{2,4\}$ and batch $\{1,3\}$ (size 2)? Why is SGD unbiased?""",
S(r"Full mean $=(1+3-1+5)/4=2$.",
  r"Batch $\{2,4\}$: $(3+5)/2=4$; batch $\{1,3\}$: $(1-1)/2=0$.",
  r"Averaged over all equally likely batches the estimate equals 2: unbiased, but each one is noisy."))

q("J8","J",1,r"""Momentum with $\beta=0.9$, $m_0=0$ and gradients $1,1,1$. Compute $m_1,m_2,m_3$.""",
S(r"$m_1=0+1=1$.",
  r"$m_2=0.9+1=1.9$.",
  r"$m_3=1.71+1=2.71$ (heading toward 10)."))

q("J9","J",2,r"""$f=\sigma(wx+b)$ with $w=2$, $x=1$, $b=-1$. Use the forward and backward pass to get $\partial f/\partial w$ and $\partial f/\partial b$.""",
S(r"Forward: $z=2-1=1$, $f=\sigma(1)=0.731$.",
  r"Local derivative $df/dz=f(1-f)=0.197$.",
  r"$\partial z/\partial w=x=1$, $\partial z/\partial b=1$.",
  r"So $\partial f/\partial w=0.197$, $\partial f/\partial b=0.197$."))

q("J10","J",1,r"""Four training-loss curves: (a) rises and explodes, (b) falls very slowly, (c) falls fast but bounces up and down, (d) falls fast then flattens low. What learning rate does each suggest?""",
S(r"(a) Far too high: diverging.",
  r"(b) Too low.",
  r"(c) High: oscillating around the minimum; lower it or use decay/momentum.",
  r"(d) Good: just below the stability limit."))

# =====================================================================
# K. Sequence models
# =====================================================================
q("K1","K",1,r"""A two-state Markov chain has $\mathbf A=\begin{bmatrix}0.9&0.1\\0.5&0.5\end{bmatrix}$ (row = current state) and starts in state 1. Find the distribution after 1 and 2 steps, and the stationary distribution.""",
S(r"Step 1: start $(1,0)$ times $\mathbf A$ = first row $(0.9,0.1)$.",
  r"Step 2: $(0.9,0.1)\mathbf A=(0.9\cdot0.9+0.1\cdot0.5,\ 0.9\cdot0.1+0.1\cdot0.5)=(0.86,0.14)$.",
  r"Stationary $\boldsymbol\pi=\boldsymbol\pi\mathbf A$: $\pi_2=0.1\pi_1+0.5\pi_2\Rightarrow\pi_1=5\pi_2$.",
  r"With $\pi_1+\pi_2=1$: $\boldsymbol\pi=(0.833,0.167)$."))

q("K2","K",2,r"""Hidden $h\in\{A,B\}$ with prior $(0.3,0.7)$; $p(y{=}1|A)=0.9$, $p(y{=}1|B)=0.2$. Find $p(y{=}1)$ and $p(A|y{=}1)$.""",
S(r"Marginalize: $p(y{=}1)=\sum_hp(y{=}1|h)p(h)=0.9\cdot0.3+0.2\cdot0.7=0.27+0.14=0.41$.",
  r"Bayes: $p(A|y{=}1)=0.27/0.41=0.659$.",
  tip=r"‘Marginal’ = sum the hidden variable out. It is an expectation $\mathbb E_h[p(y|h)]$."))

q("K3","K",3,r"""HMM: start $\boldsymbol\pi=(0.6,0.4)$, transitions $\mathbf T=\begin{bmatrix}0.7&0.3\\0.4&0.6\end{bmatrix}$, $p(x{=}1|h{=}0)=0.2$, $p(x{=}1|h{=}1)=0.8$. Compute $p(x=(1,1))$ by the forward algorithm, and check by brute force.""",
S(r"$\boldsymbol\alpha_1=\boldsymbol\pi\odot p(x{=}1|h)=(0.6\cdot0.2,\ 0.4\cdot0.8)=(0.12,0.32)$.",
  r"Predict: $\boldsymbol\alpha_1\mathbf T=(0.12\cdot0.7+0.32\cdot0.4,\ 0.12\cdot0.3+0.32\cdot0.6)=(0.212,0.228)$.",
  r"Emit: $\boldsymbol\alpha_2=(0.212\cdot0.2,\ 0.228\cdot0.8)=(0.0424,0.1824)$.",
  r"$p(x)=0.0424+0.1824=0.2248$.",
  r"Brute force over the 4 hidden paths: $0.0168+0.0288+0.0256+0.1536=0.2248$ ✓."))

q("K4","K",2,r"""Scalar RNN $h_t=\tanh(0.5x_t+0.8h_{t-1})$, $h_0=0$, inputs $(1,1)$. Compute $h_1,h_2$. Count the parameters for $d_x=5$, $H=8$, $d_y=3$ (with biases on hidden and output).""",
S(r"$h_1=\tanh(0.5)=0.462$.",
  r"$h_2=\tanh(0.5+0.8\cdot0.462)=\tanh(0.870)=0.701$.",
  r"Parameters: $\mathbf W_{xh}$: $5\cdot8=40$; $\mathbf W_{hh}$: $64$; hidden bias $8$; $\mathbf W_{hy}$: $24$; output bias $3$.",
  r"Total $139$, the same for any sequence length (weights are shared over time)."))

q("K5","K",3,r"""Attention with $d=2$: query $\mathbf q=(1,0)$; keys $(1,0),(0,1),(1,1)$; values $(1,0),(0,1),(2,2)$. Compute the scores, the weights and the output $\mathbf z$.""",
S(r"Scores $\mathbf q^\top\mathbf k_i/\sqrt2$: $(1,0,1)/1.414=(0.707,0,0.707)$.",
  r"Softmax: $e^{0.707}=2.028$, $e^0=1$; sum $5.056$; weights $(0.401,0.198,0.401)$ (sum to 1).",
  r"Output $\mathbf z=0.401(1,0)+0.198(0,1)+0.401(2,2)=(1.203,1.000)$.",
  r"With a causal mask at position 1 only key 1 is visible: weight 1, so $\mathbf z=(1,0)$."))

q("K6","K",2,r"""A language model gives the true tokens probabilities $0.5$, $0.25$, $0.8$. Compute the total negative log-likelihood, the per-token loss and the perplexity.""",
S(r"$-\ln0.5=0.693$, $-\ln0.25=1.386$, $-\ln0.8=0.223$.",
  r"Total $2.303$; per token $2.303/3=0.768$.",
  r"Perplexity $e^{0.768}=2.15$: like choosing among about 2 equally likely tokens each step."))

q("K7","K",1,r"""The DFA for ‘even number of 1s’ has states even/odd, starts in even, accepts in even. Trace $1011$ and $0110$.""",
S(r"Rule: a 1 flips the state, a 0 keeps it.",
  r"$1011$: even →1 odd →0 odd →1 even →1 odd. Ends odd: reject (three 1s).",
  r"$0110$: even →0 even →1 odd →1 even →0 even. Accept (two 1s).",
  r"The single state bit is a sufficient summary of the past: a Markov state."))

q("K8","K",2,r"""Parameters: (a) an order-3 Markov model over $K=4$ symbols; (b) an HMM with 3 hidden states emitting 4 symbols. Why does the hidden state help?""",
S(r"(a) Each of $K^3=64$ histories needs a distribution over 4 symbols with 3 free numbers: $64\cdot3=192$.",
  r"(b) Transitions $3\cdot2=6$, start $2$, emissions $3\cdot3=9$: total $17$.",
  r"A hidden state compresses the history; observations under an HMM depend on the whole past with few parameters."))

q("K9","K",2,r"""Three tokens, decoder self-attention. Which entries of the $3\times3$ attention matrix does a causal mask remove? Why is the mask needed during teacher forcing?""",
S(r"Token $t$ may attend only to tokens $\le t$: remove (set scores to $-\infty$) entries $(1,2)$, $(1,3)$, $(2,3)$.",
  r"Each row is still a softmax, so it sums to 1 over the allowed entries.",
  r"Teacher forcing feeds the whole true sequence in one pass; without the mask position $t$ could read the token it must predict."))

q("K10","K",2,r"""A gradient passes through 50 recurrent steps that each multiply it by $0.9$, or by $1.1$. Compute both. What fixes this?""",
S(r"$0.9^{50}=0.0052$: vanishing.",
  r"$1.1^{50}=117$: exploding.",
  r"Fixes: gated blocks (GRU, LSTM with a long-term memory path), gradient clipping for explosions, or attention, which reaches any past position directly."))

q("K11","K",1,r"""(a) How many entries does one attention matrix have for 2000 tokens? (b) What happens when the context doubles? (c) Why do Transformers need positional encodings?""",
S(r"(a) $2000^2=4{,}000{,}000$.",
  r"(b) $n^2$ → four times as many.",
  r"(c) Attention is a weighted sum that ignores order; adding a position vector $\mathbf z_t+\mathbf p_t$ lets the model know where each token is."))

q("K12","K",1,r"""Classify each task as vec2seq, seq2vec or seq2seq: image captioning, review sentiment, English→French translation, next-word chat reply.""",
S(r"Image captioning: vector (image features) → sequence: vec2seq.",
  r"Sentiment: sequence → one label: seq2vec (bidirectional helps).",
  r"Translation: sequence → sequence: seq2seq.",
  r"Chat reply: prompt sequence → response sequence, decoded autoregressively: seq2seq."))

q("K13","K",2,r"""Explain in four lines how an LLM is trained and used: name the objective, teacher forcing, autoregressive rollout, and why safety is hard.""",
S(r"Objective: maximum likelihood, $\boldsymbol\theta^*=\arg\max p(\mathcal D|\boldsymbol\theta)$ = minimize next-token cross-entropy.",
  r"Teacher forcing: feed the true previous tokens and train all positions in one masked pass.",
  r"Use: start from $\langle\text{bos}\rangle$, sample one token at a time and feed it back, stop at $\langle\text{eos}\rangle$.",
  r"Safety: the model imitates whatever is likely in the data; screening prompts is hard, so alignment uses reinforcement learning with a reward."))

# =====================================================================
# L. SVMs and constrained optimization
# =====================================================================
q("L1","L",1,r"""$\mathbf w=(3,4)$, $b=5$, rule $\mathbf x^\top\mathbf w-b$. (a) Margin width? (b) Signed distance of $(1,1)$? (c) Point $(1,0)$ has label $+1$: correct? its slack?""",
S(r"(a) $\|\mathbf w\|=5$, so $d=2/5=0.4$.",
  r"(b) Score $3+4-5=2$; signed distance $2/5=0.4$ on the $+$ side.",
  r"(c) Score $3-5=-2$; $y\cdot\text{score}=-2<0$: misclassified.",
  r"Slack $\xi=\max(0,1-(-2))=3$ (more than 1 means wrong side)."))

q("L2","L",2,r"""1-D data: $x=1,2$ with $y=-1$; $x=5,6$ with $y=+1$. Find the hard-margin SVM ($w$, $b$), the margin and the support vectors. Check the constraints.""",
S(r"The widest street lies between the closest opposite points 2 and 5: boundary at $3.5$, width $3$.",
  r"$d=2/|w|=3\Rightarrow w=2/3$; boundary $wx-b=0$ at $3.5\Rightarrow b=7/3=2.333$.",
  r"Check: $x=5$: $\tfrac{10}3-\tfrac73=1$ ✓ on the curb; $x=2$: $-(\tfrac43-\tfrac73)=1$ ✓ on the curb.",
  r"$x=1$ and $x=6$ give $1.667>1$. Support vectors: $x=2$ and $x=5$."))

q("L3","L",2,r"""Minimize $f=x_1^2+2x_2^2$ subject to $x_1+x_2=3$ with a Lagrangian.""",
S(r"$\mathcal L=x_1^2+2x_2^2+\lambda(x_1+x_2-3)$.",
  r"$\partial_{x_1}$: $2x_1+\lambda=0$; $\partial_{x_2}$: $4x_2+\lambda=0$; $\partial_\lambda$: $x_1+x_2=3$.",
  r"From the first two: $x_1=2x_2$. Then $3x_2=3$: $x_2=1$, $x_1=2$, $\lambda=-4$.",
  r"$f^*=4+2=6$.",
  tip=r"The $\partial\mathcal L/\partial\lambda=0$ row is the constraint itself."))

q("L4","L",2,r"""Soft margin in 1-D with $w=1$, $b=3$. Points $(x,y)$: $(1,-1)$, $(2.5,-1)$, $(3.5,+1)$, $(2,+1)$. Compute each slack and the objective with $C=1$.""",
S(r"Margin value $m=y(xw-b)$; slack $\xi=\max(0,1-m)$.",
  r"$(1,-1)$: $m=2$, $\xi=0$ (outside the street).",
  r"$(2.5,-1)$: $m=0.5$, $\xi=0.5$ (inside, correct). $(3.5,+1)$: $m=0.5$, $\xi=0.5$.",
  r"$(2,+1)$: $m=-1$, $\xi=2$ (wrong side).",
  r"Objective $\tfrac12w^2+C\sum\xi=0.5+3=3.5$."))

q("L5","L",1,r"""Short answers: (a) What are support vectors? (b) Why minimize $\tfrac12\|\mathbf w\|^2$? (c) Why labels $\pm1$? (d) Is an SVM a probability model?""",
S(r"(a) Points exactly on the curbs ($y(\mathbf x^\top\mathbf w-b)=1$); only they fix the line.",
  r"(b) Margin $=2/\|\mathbf w\|$, so a smaller $\|\mathbf w\|$ is a wider street; the half and square only make the derivative clean.",
  r"(c) So one inequality $y_i(\mathbf x_i^\top\mathbf w-b)\ge1$ covers both classes.",
  r"(d) No: no $p(y|\mathbf x)$, no likelihood; the score is a distance."))

q("L6","L",1,r"""$\mathbf a=(2,1)$, $\mathbf b=(3,4)$. Compute $\mathbf a^\top\mathbf b$, the shadow (projection) of $\mathbf a$ on $\mathbf b$, and $\cos\theta$.""",
S(r"$\mathbf a^\top\mathbf b=6+4=10$.",
  r"$\|\mathbf b\|=5$, so the shadow is $10/5=2$.",
  r"$\|\mathbf a\|=\sqrt5$; $\cos\theta=10/(\sqrt5\cdot5)=0.894$."))

q("L7","L",1,r"""$\mathbf w=(6,8)$, $b=20$. Find the distance from the origin to the boundary and the margin.""",
S(r"$\|\mathbf w\|=10$.",
  r"Origin to boundary: $b/\|\mathbf w\|=2$.",
  r"Margin $2/\|\mathbf w\|=0.2$."))

q("L8","L",3,r"""Two 1-D points: $x=1$ ($y=-1$) and $x=3$ ($y=+1$). Solve the hard-margin SVM with a Lagrangian ($\lambda_i\ge0$) and find the multipliers.""",
S(r"Constraints $y_i(x_iw-b)-1\ge0$: $3w-b-1\ge0$ and $b-w-1\ge0$.",
  r"$\mathcal L=\tfrac12w^2-\lambda_1(3w-b-1)-\lambda_2(b-w-1)$.",
  r"$\partial_w$: $w-3\lambda_1+\lambda_2=0$; $\partial_b$: $\lambda_1-\lambda_2=0$. So $\lambda_1=\lambda_2=\lambda$, $w=2\lambda$.",
  r"Both points are support vectors (tight): $3w-b=1$, $b-w=1$ → $w=1$, $b=2$, $\lambda=0.5\ge0$ ✓.",
  r"Margin $2/w=2$ = the gap between 1 and 3 ✓; boundary at $b/w=2$."))

q("L9","L",2,r"""Compare the hinge loss $\max(0,1-m)$ and the logistic loss $\log(1+e^{-m})$ at $m=0.5$ and $m=2$. What does the difference mean for which points influence the line?""",
S(r"$m=0.5$: hinge $0.5$; logistic $\log(1+e^{-0.5})=0.474$.",
  r"$m=2$: hinge $0$; logistic $\log(1+e^{-2})=0.127$.",
  r"Hinge is exactly 0 beyond the margin, so those points have no effect (only support vectors matter).",
  r"Logistic loss is never 0, so every point pushes a little."))

q("L10","L",2,r"""Write the soft-margin SVM as ‘regularizer + loss’. What is the ridge-style $\lambda$ for $C=5$? What do small and large $C$ do, and how do you choose it?""",
S(r"At the optimum $\xi_i=\max(0,1-y_i(\mathbf x_i^\top\mathbf w-b))$, so the problem is $\tfrac12\|\mathbf w\|^2+C\sum_i\text{hinge}_i$.",
  r"Divide by $C$: $\sum\text{hinge}+\frac1{2C}\|\mathbf w\|^2$, so $\lambda=1/(2C)=0.1$.",
  r"Small $C$: slack is cheap, wide street, more violations (bias). Large $C$: narrow street, few violations (variance).",
  r"Choose $C$ by validation / cross-validation."))

q("L11","L",2,r"""Why can the hard-margin problem have no solution, and how do slack variables fix it? What does $\xi_i>1$ mean?""",
S(r"If the classes overlap, no $(\mathbf w,b)$ satisfies all $y_i(\mathbf x_i^\top\mathbf w-b)\ge1$: the feasible set is empty.",
  r"Slack relaxes each constraint to $\ge1-\xi_i$ with $\xi_i\ge0$; large enough $\xi_i$ always works, and $C\sum\xi_i$ keeps them small.",
  r"$\xi_i=0$: outside the street; $0<\xi_i\le1$: inside but correct; $\xi_i>1$: misclassified."))
