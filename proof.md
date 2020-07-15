# 証明

<!-- ## イェンセンの不等式

限界効用逓減であるような効用関数 $u$, 任意の確率くじ $X$ について以下が成り立つ。

$$E[u(X)] \leq u(E[X])$$

### 証明

$X$ は $i = 1, \ldots, n$ について、確率 $p_i$ で $x_i$ 円となる確率くじとする。
$x_1 < \cdots < x_n$ を仮定して一般性を失わない。

$$E[u(X)] = \sum_{i=1}^{n} p_i u(x_i)$$

$$u(E[X]) = u(\sum_{i=1}^{n} p_i x_i)$$ -->

## 資産配分問題の最適解

資産 $i$ のリターンを $R_i$ とし、$E[R_i] = \mu_i$, $Var[R_i] = \sigma^2_{i}$ とする。$R_i$, $R_j$ の相関係数を $\rho_{i, j}$ とする。
資産 $1, \ldots, n$ に関するポートフォリオ $P = (w_1, \ldots, w_n)$ のリターンを $R_P$ とする。
投資家の目的関数 $f(R_P)$ を

$$f(R_P) = E[R_P] - \frac{\gamma}{2} Var[R_P]$$

とする。ただし、 $\gamma > 0$ とする。
このとき、 最適な $P = (w_1, \ldots, w_n)$ は以下を満たす。

$$
  \left (
    \begin{array}{ccccc}
      1 & 1 & \cdots & 1 & 0 \\
      \gamma \sigma^2_1 & \gamma \rho_{1, 2} \sigma_1 \sigma_2 & \cdots & \gamma \rho_{1, n} \sigma_1 \sigma_n & 1 \\
      \vdots & \vdots & \ddots & \vdots & \vdots \\
      \gamma \rho_{1, n} \sigma_1 \sigma_{n} & \gamma \rho_{2, n} \sigma_2 \sigma_{n} & \cdots & \gamma  \sigma^2_n & 1
    \end{array}
  \right )
  \left (
    \begin{array}{c}
      w_1 \\
      \vdots \\
      w_n \\
      \lambda
    \end{array}
  \right )
  =
  \left (
    \begin{array}{c}
      1 \\
      \mu_1 \\
      \vdots \\
      \mu_n
    \end{array}
  \right )
$$

### 導出の方針
- $f$を偏微分した値が0となるような $P$ を求める
- $\sum_{i = 1}^{n} w_i = 1$ の制約があるので、ラグランジュの未定乗数法を用いる
- $Var[\sum_{i=1}^{n} w_i R_i] = \sum_{i=1}^{n} \sum_{j=1}^{n} \rho_{i, j} w_i w_j \sigma_{i} \sigma_{j}$ が成り立つことを利用する
