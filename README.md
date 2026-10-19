# OptiAlpha: Convex Regularized Alpha Combination and Dynamic Multi-Factor Allocation Engine

[![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13%20%7C%203.14-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-61%20passed-brightgreen.svg)]()
[![Coverage](https://img.shields.io/badge/coverage-99.6%25-brightgreen.svg)]()
[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

> **OptiAlpha** is an institutional-grade Python framework for regularized multi-factor alpha combination, dynamic covariance shrinkage, and online signal allocation. It solves the multi-factor aggregation dilemma by trading off expected predictive Information Coefficients (IC), factor portfolio risk, $`L_1`$ turnover transaction costs, and factor concentration constraints via accelerated proximal gradient methods (FISTA).

---

## The Quantitative Pipeline: OrthoQuant $\to$ OptiAlpha

In modern quantitative equity and multi-asset factor investing, the signal construction pipeline follows a rigorous two-stage architecture:

```
[ Raw Alpha Factors ]
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│ 1. OrthoQuant (Signal Pre-Processing & Clean-up)            │
│    • Cross-Sectional Risk Neutralization (FWL via Thin QR)  │
│    • Löwdin Symmetric Orthogonalization (F^T F = I_K)       │
└─────────────────────────────────────────────────────────────┘
         │
         ▼  (Orthonormal Residual Factor Streams)
┌─────────────────────────────────────────────────────────────┐
│ 2. OptiAlpha (Alpha Combination & Factor Allocation)        │
│    • Ledoit-Wolf Analytical Covariance Shrinkage            │
│    • Convex L1-Regularized Turnover Optimization (FISTA)    │
│    • Online Exponentiated Gradient Adaptive Weighting       │
│    • Meucci Effective Number of Alphas (ENF) Attribution    │
└─────────────────────────────────────────────────────────────┘
         │
         ▼
[ Optimized Composite Alpha Score w_t^T F_t ]
```

---

## Mathematical Notation Reference

| Symbol | Type | Description |
| :--- | :--- | :--- |
| $`N \in \mathbb{N}`$ | Dimension | Number of assets in the universe ($`N \ge K`$) |
| $`K \in \mathbb{N}`$ | Dimension | Number of candidate alpha factors |
| $`T \in \mathbb{N}`$ | Dimension | Number of historical observation periods (trading days) |
| $`\mathbf{F}_t \in \mathbb{R}^{N \times K}`$ | Matrix | Cross-sectional factor exposure matrix at time $`t`$ |
| $`\mathbf{ic} \in \mathbb{R}^K`$ | Vector | Information Coefficient vector ($`\mathrm{Rank\ IC}`$) |
| $`\mathbf{\Omega}_{\alpha} \in \mathbb{R}^{K \times K}`$ | Matrix | Factor covariance matrix |
| $`\mathbf{w} \in \mathbb{R}^K`$ | Vector | Factor allocation weight vector |
| $`\mathbf{w}_{\text{prev}} \in \mathbb{R}^K`$ | Vector | Previous rebalance weights for turnover control |
| $`\gamma \ge 0`$ | Scalar | Risk aversion penalty on factor portfolio variance |
| $`\lambda_{\text{turnover}} \ge 0`$ | Scalar | $`L_1`$ transaction cost penalty |
| $`\lambda_2 \ge 0`$ | Scalar | $`L_2`$ ridge regularization parameter |
| $`\lVert \cdot \rVert_1, \lVert \cdot \rVert_2`$ | Norms | Vector $`L_1`$ and $`L_2`$ norms |
| $`\lVert \cdot \rVert_F`$ | Norm | Matrix Frobenius norm |

---

## Mathematical Architecture & Theoretical Formulations

### 1. Fundamental Law of Active Management & Optimal Alpha Allocation

Let $`\mathbf{F}_t \in \mathbb{R}^{N \times K}`$ be the cross-sectional factor exposure matrix for $`N`$ assets across $`K`$ candidate alpha factors at time $`t`$.  
Let $`\mathbf{r}_{t+1} \in \mathbb{R}^N`$ be the forward asset returns.  
Let $`\mathbf{ic} = (\mathrm{IC}_1, \dots, \mathrm{IC}_K)^T \in \mathbb{R}^K`$ denote the vector of forecast Information Coefficients ($`\mathrm{Rank\ IC}`$).  
Let $`\mathbf{\Omega}_{\alpha} \in \mathbb{R}^{K \times K}`$ be the factor covariance matrix.

According to the Fundamental Law of Active Management (Grinold & Kahn, 2000; Qian & Hua, 2004), the unconstrained weight vector $`\mathbf{w}^*`$ that maximizes the composite Information Ratio (IR) satisfies:

$$
\mathbf{w}^* = \mathbf{\Omega}_{\alpha}^{-1} \mathbf{ic} \qquad \text{(1)}
$$

The maximal achievable composite Information Ratio is:

$$
\mathrm{IR}^* = \sqrt{\mathbf{ic}^T \mathbf{\Omega}_{\alpha}^{-1} \mathbf{ic}} \qquad \text{(2)}
$$

#### Theorem 1 (The OrthoQuant Closed-Form Connection)

If factors $`\mathbf{\widetilde{F}}`$ are orthonormalized via **OrthoQuant**'s Löwdin Symmetric Orthogonalization ($`\mathbf{\widetilde{F}}^T \mathbf{\widetilde{F}} = \mathbf{I}_K`$), then the factor correlation matrix collapses to the identity matrix $`\mathbf{C} = \mathbf{I}_K`$. Consequently:

$$
\mathbf{\Omega}_{\alpha} = \mathrm{diag}(\sigma_1^2, \dots, \sigma_K^2) \implies w_k^* = \frac{\mathrm{IC}_k}{\sigma_k^2} \qquad \text{(3)}
$$

*Proof:* Substituting $`\mathbf{\Omega}_{\alpha} = \mathbf{D}_{\sigma^2}`$ into Equation (1), $`\mathbf{\Omega}_{\alpha}^{-1} = \mathbf{D}_{1/\sigma^2}`$, yielding $`w_k^* = \mathrm{IC}_k / \sigma_k^2`$. Multicollinearity and ill-conditioned matrix inversion are completely eliminated. $`\blacksquare`$

---

### 2. Convex Regularized Objective with Turnover & Concentration Bounds

In practical portfolio management, factor allocations cannot change drastically from period to period due to market friction, slippage, and turnover costs. Furthermore, single-factor concentration must be bounded.

OptiAlpha formulates factor combination as a strictly convex optimization problem:

$$
\min_{\mathbf{w} \in \mathcal{W}} \quad \mathcal{L}(\mathbf{w}) = \underbrace{-\mathbf{ic}^T \mathbf{w}}_{\text{Predictive Return}} + \underbrace{\frac{\gamma}{2} \mathbf{w}^T \mathbf{\Omega}_{\alpha} \mathbf{w}}_{\text{Factor Risk Penalty}} + \underbrace{\lambda_{\text{turnover}} \lVert \mathbf{w} - \mathbf{w}_{\text{prev}} \rVert_1}_{\text{L1 Turnover Friction}} + \underbrace{\frac{\lambda_2}{2} \lVert \mathbf{w} \rVert_2^2}_{\text{L2 Ridge Shrinkage}} \qquad \text{(4)}
$$

subject to convex budget and box constraints:

$$
\mathcal{W} = \lbrace \mathbf{w} \in \mathbb{R}^K : l_k \le w_k \le u_k \; \forall k, \quad \sum_{k=1}^K w_k = s \rbrace \qquad \text{(5)}
$$

where:
- $`\gamma \ge 0`$: Risk aversion parameter penalizing factor portfolio volatility.
- $`\lambda_{\text{turnover}} \ge 0`$: $`L_1`$ turnover penalty against prior allocation $`\mathbf{w}_{\text{prev}}`$.
- $`l_k, u_k`$: Lower/upper bounds ($`l_k = 0, u_k = 1`$ enforces a long-only simplex $`\Delta^K`$).

---

### 3. Accelerated Proximal Gradient Solver (FISTA)

Because the $`L_1`$ turnover term $`\lVert \mathbf{w} - \mathbf{w}_{\text{prev}} \rVert_1`$ is non-smooth, classical gradient descent fails. OptiAlpha implements the **Fast Iterative Shrinkage-Thresholding Algorithm (FISTA)** (Beck & Teboulle, 2009) with $`\mathcal{O}(1/k^2)`$ optimal convergence rate:

#### Step 1: Smooth Gradient Step

$$
\nabla f(\mathbf{y}^{(k)}) = -\mathbf{ic} + \gamma \mathbf{\Omega}_{\alpha} \mathbf{y}^{(k)} + \lambda_2 \mathbf{y}^{(k)}, \quad \mathbf{z}^{(k)} = \mathbf{y}^{(k)} - \frac{1}{L} \nabla f(\mathbf{y}^{(k)})
$$

where $`L = \lVert \gamma \mathbf{\Omega}_{\alpha} + \lambda_2 \mathbf{I} \rVert_2`$ is the exact Lipschitz constant computed via spectral norm.

#### Step 2: Shifted Soft-Thresholding ($L_1$ Proximal Operator)

$$
\mathbf{v}^{(k)} = \mathbf{w}_{\text{prev}} + \mathrm{sign}(\mathbf{z}^{(k)} - \mathbf{w}_{\text{prev}}) \odot \max \left( \lvert \mathbf{z}^{(k)} - \mathbf{w}_{\text{prev}} \rvert - \frac{\lambda_{\text{turnover}}}{L}, \, 0 \right)
$$

#### Step 3: Box-Constrained Simplex Projection

$$
\mathbf{w}^{(k+1)} = \Pi_{\mathcal{W}}(\mathbf{v}^{(k)}) = \mathrm{clip}(\mathbf{v}^{(k)} - \mu^*, \mathbf{l}, \mathbf{u})
$$

where dual parameter $`\mu^*`$ is solved via bisection root finding such that $`\sum_{i=1}^K w_i = s`$.

#### Step 4: Nesterov Momentum Acceleration

$$
t_{k+1} = \frac{1 + \sqrt{1 + 4 t_k^2}}{2}, \quad \mathbf{y}^{(k+1)} = \mathbf{w}^{(k+1)} + \left( \frac{t_k - 1}{t_{k+1}} \right) (\mathbf{w}^{(k+1)} - \mathbf{w}^{(k)})
$$

---

### 4. Analytical Ledoit-Wolf Covariance Shrinkage

Sample covariance matrices $`\mathbf{S} = \frac{1}{T-1} \mathbf{X}^T \mathbf{X}`$ are notoriously ill-conditioned when $`T`$ is small relative to factor count $`K`$. OptiAlpha computes the asymptotically optimal shrinkage intensity $`\delta^* \in [0, 1]`$ (Ledoit & Wolf, 2004):

$$
\mathbf{\Sigma}^* = \delta^* \mathbf{T} + (1 - \delta^*) \mathbf{S} \qquad \text{(6)}
$$

- **Constant Correlation Target ($`\mathbf{T}`$):** Preserves individual sample variances $`S_{ii}`$ while shrinking cross-sectional correlations to the mean correlation $`\bar{r} = \frac{2}{K(K-1)} \sum_{i < j} R_{ij}`$.
- **Optimal Shrinkage Parameter:**

$$
\delta^* = \min \left( 1, \, \max \left( 0, \, \frac{\hat{\pi} - \hat{\rho}}{T \cdot \lVert \mathbf{S} - \mathbf{T} \rVert_F^2} \right) \right)
$$

---

### 5. Online Adaptive Allocation (Exponentiated Gradient)

For real-time streaming environments, OptiAlpha provides an online multiplicative update rule (Freund & Schapire, 1997; Kivinen & Warmuth, 1997) guaranteeing sublinear regret $`\mathcal{O}(\sqrt{T \ln K})`$:

$$
w_{k, t+1} = (1 - \varepsilon) \frac{w_{k, t} \exp(\eta \cdot \hat{r}_{k, t})}{\sum_{j=1}^K w_{j, t} \exp(\eta \cdot \hat{r}_{j, t})} + \frac{\varepsilon}{K} \qquad \text{(7)}
$$

where $`\eta > 0`$ is the learning rate and $`\varepsilon \in [0, 1]`$ prevents premature factor starvation.

---

### 6. Risk Attribution & Diversification Metrics

#### Euler's Risk Decomposition

Total factor portfolio volatility $`\sigma_p = \sqrt{\mathbf{w}^T \mathbf{\Omega}_{\alpha} \mathbf{w}}`$ decomposes additively into Percentage Risk Contributions ($`\mathrm{PRC}_k`$):

$$
\mathrm{PRC}_k = \frac{w_k (\mathbf{\Omega}_{\alpha} \mathbf{w})_k}{\mathbf{w}^T \mathbf{\Omega}_{\alpha} \mathbf{w}}, \quad \sum_{k=1}^K \mathrm{PRC}_k = 1.0 \qquad \text{(8)}
$$

#### Effective Number of Alphas (ENF)

Following Meucci (2009), the effective diversification entropy across alpha streams is:

$$
\mathrm{ENF} \triangleq \exp \left( -\sum_{k=1}^K p_k \ln p_k \right) \in [1, K] \qquad \text{(9)}
$$

where $`p_k = \max(\mathrm{PRC}_k, 0)`$.

---

## Strategy Comparison Matrix

| Strategy | Objective Function | Turnover Control | Handles Multicollinearity | Optimal Solver |
| :--- | :--- | :--- | :--- | :--- |
| **Equal Weighting (1/K)** | Heuristic benchmark | None | Poor | Closed-form $`\mathcal{O}(1)`$ |
| **IC-Weighted** | $`w_k \propto \max(0, \mathrm{IC}_k)`$ | None | Moderate | Closed-form $`\mathcal{O}(K)`$ |
| **Grinold-Kahn Analytical** | $`\mathbf{\Omega}^{-1} \mathbf{ic}`$ | None | Requires Invertible $`\mathbf{\Omega}`$ | Linear Solve $`\mathcal{O}(K^3)`$ |
| **OptiAlpha Convex (FISTA)** | $`\min -\mathbf{ic}^T \mathbf{w} + \frac{\gamma}{2} \mathbf{w}^T \mathbf{\Omega} \mathbf{w} + \lambda \lVert \Delta \mathbf{w} \rVert_1`$ | **Rigorous $`L_1`$ Penalty** | **Excellent (L2 Ridge + Ledoit-Wolf)** | **FISTA $`\mathcal{O}(K \log K)`$ per iter** |
| **OptiAlpha Online (EG)** | Multiplicative Weights Online Regret | Adaptive Decay | Good | Online $`\mathcal{O}(K)`$ per step |

---

## Installation

```bash
git clone https://github.com/huuan26/OptiAlpha.git
cd OptiAlpha
pip install -e .
```

---

## Quickstart

```python
import numpy as np
import pandas as pd
from optialpha import (
    generate_synthetic_alpha_panel,
    ledoit_wolf_shrinkage,
    convex_alpha_combination,
    equal_weight_combination,
    information_coefficient,
    effective_number_of_alphas,
    turnover_ratio,
)

# 1. Generate multi-factor panel (100 assets, 5 factors, 120 periods)
panel = generate_synthetic_alpha_panel(n_assets=100, n_factors=5, n_periods=120)
factors_t0 = panel["factors_panel"][0]
returns_t0 = panel["returns_panel"][0]
ic_history = panel["ic_history"]

# 2. Estimate cleaned covariance matrix via Ledoit-Wolf Shrinkage
shrunk_cov, shrinkage_delta = ledoit_wolf_shrinkage(ic_history, target="constant_correlation")
print(f"Optimal Ledoit-Wolf Shrinkage Intensity (delta): {shrinkage_delta:.4f}")

# 3. Forecast Information Coefficients (Rolling Mean IC)
forecast_ic = ic_history.mean(axis=0).to_numpy()

# 4. Solve Regularized Convex Alpha Combination with Turnover Penalty
w_prior = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
res = convex_alpha_combination(
    factors=factors_t0,
    ic_vector=forecast_ic,
    covariance=shrunk_cov,
    risk_aversion=1.5,
    turnover_penalty=0.03,
    w_prev=w_prior,
    lower_bound=0.0,
    upper_bound=0.40,
    solver="fista",
)

print("\n--- OptiAlpha Allocation Results ---")
print("Optimal Factor Weights:", np.round(res.weights, 4))
print(f"Convergence: {res.converged} in {res.iterations} iterations")
print(f"Predicted Information Coefficient: {res.predicted_ic:.4f}")
print(f"Realized Turnover from Prior Allocation: {res.turnover:.4f}")

# 5. Evaluate Composite Alpha vs Equal-Weight Benchmark
composite_ic = information_coefficient(res.composite_alpha, returns_t0, method="spearman")
eq_alpha, _ = equal_weight_combination(factors_t0)
eq_ic = information_coefficient(eq_alpha, returns_t0, method="spearman")

print(f"\nEqual-Weight Rank IC: {eq_ic:.4f}")
print(f"OptiAlpha Composite Rank IC: {composite_ic:.4f}")

# 6. Measure Effective Diversification (Meucci ENF)
enf = effective_number_of_alphas(res.weights, shrunk_cov)
print(f"Effective Number of Alphas (ENF): {enf:.2f} / 5.00")
```

---

## Testing & Verification

Run the full test suite with coverage report:

```bash
pytest
```

---

## Academic References & Bibliography

1. **Grinold, R. C., & Kahn, R. N. (2000).** *Active Portfolio Management: A Quantitative Approach for Producing Superior Returns and Controlling Risk* (2nd ed.). McGraw-Hill.
2. **Qian, E. E., & Hua, R. H. (2004).** Active risk and information ratio in multi-factor models. *The Journal of Portfolio Management*, 30(3), 20–34.
3. **Ledoit, O., & Wolf, M. (2004).** A well-conditioned estimator for large-dimensional covariance matrices. *Journal of Multivariate Analysis*, 88(2), 365–411.
4. **Beck, A., & Teboulle, M. (2009).** A fast iterative shrinkage-thresholding algorithm for linear inverse problems. *SIAM Journal on Imaging Sciences*, 2(1), 183–202.
5. **Duchi, J., Shalev-Shwartz, S., Singer, Y., & Chandra, T. (2008).** Efficient projections onto the $`l_1`$-ball for predictors in high dimensions. *ICML*, 272–279.
6. **Meucci, A. (2009).** Managing diversification. *Risk*, 22(5), 74–79.
7. **Choueifaty, Y., & Coignard, Y. (2008).** Toward maximum diversification. *The Journal of Portfolio Management*, 35(1), 40–51.
8. **Freund, Y., & Schapire, R. E. (1997).** A decision-theoretic generalization of on-line learning and an application to boosting. *Journal of Computer and System Sciences*, 55(1), 119–139.

---

## Author & License

- **Author**: [Phan Bùi Hữu Ân (huuan26)](https://github.com/huuan26)
- **License**: MIT License - see the [LICENSE](LICENSE) file for details.
