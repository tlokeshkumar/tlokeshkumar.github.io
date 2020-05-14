---
layout: post
title: "Math Geeks Exclusive: Interpretability Analysis in Linear Regression"
author: Lokesh Kumar
date: '2020-05-09 17:00:00'
category: 'Machine-Learning'
summary: A detailed analysis of 
thumbnail: Regression meets Game Theory.png
commments: true
---
## Contents
* Standardized Multiple Regression Model
* Revisiting R-squared for Regression Analysis ($$R^2$$)
* Problems caused by Multicollinearity
* Incremental Influence of Predictors in Multiple Determination
* Incremental Net Effects
* Co-operative Game Theory - Shapley Values
* Grand Finale

## Standardized Multiple Regression Model

Consider we have $n$ predictor variables $$\left(X_1,...,X_n\right)$$ and target variable $$Y$$. For each data point $$i$$, these variables assume the form $$\left(x_{1i},...,x_{ni}, y_i\right)$$ respectively. Now, we can formulate the linear model

$$
\begin{equation}
y_i = \beta_0 + \beta_1x_{i1}+\beta_2x_{i2}+...\beta_nx_{in}+\epsilon_i
\label{genModel}
\end{equation}
$$

which is linear combination of the features, and $$\epsilon \sim \mathcal{N}(0, \sigma^2)$$. $$\eqref{genModel}$$ can also be rewritten as, 

$$
\begin{equation}
y_i = \sum_{j=0}^n\beta_jx_{ij}+\epsilon_i
\end{equation}
$$

where $$x_{i0}=1$$. Now since $$\mathbb{E}[\epsilon_i]=0$$, we also know

$$
\begin{equation}
\mathbb{E}[y]=\beta_0 + \beta_1x_1+\beta_2x_2 + ... + \beta_nx_n
\end{equation}
$$

Our main aim throughout the post will be to engage in answering **how each predictor ($$\beta_iX_i$$) influences the target variable estimation**. We also will try answering the question when our model suffers from multicollinearity (i.e) when predictors are highly correlated. 

Lets take we have $$N$$ data samples and we write the above formulation in matrix form,

$$
\begin{equation}
\begin{bmatrix}
x_{10}&x_{11}&x_{12}&\dots&x_{1n} \\
x_{20}&x_{11}&x_{22}&\dots&x_{2n} \\
\vdots&\vdots&\vdots&\dots&\vdots \\
\vdots&\vdots&\vdots&\dots&\vdots \\
x_{N0}&x_{N1}&x_{N2}&\dots&x_{Nn} \\
\end{bmatrix}

\begin{bmatrix}
\beta_0 \\
\beta_1 \\
\vdots \\
\beta_n
\end{bmatrix}
+
\begin{bmatrix}
\epsilon_0 \\
\epsilon_1 \\
\vdots \\
\vdots \\
\epsilon_m
\end{bmatrix}

=
\begin{bmatrix}
y_1 \\
y_2 \\
\vdots \\
\vdots \\
y_N
\end{bmatrix}
\end{equation}

\Longleftrightarrow

X\vec{\beta}+\vec{\epsilon}=\vec{Y}
$$

Let $$\vec{b}=[b_0, b_1, ..., b_n]^T$$ be the estimate of $$\beta$$ obtained by **least squares estimation**, 

$$
\begin{equation}
X^TX\vec{b}=X^T\vec{Y} \Longrightarrow \vec{b} = \left(X^TX\right)^{-1}X^T\vec{Y}
\end{equation}
$$

Can you think of any problem here? What if there are some variables in $$X$$ which are very small relative to others (orders of magnitude separation)? What if some features in $$X$$ are correlated with other variables? This will cause ill-conditioning of $$X^TX$$ which inturn results in numerical issues during its inversion. More precisely, we might potentially suffer from **round off errors** in computation. So, we need to reparameterize the regression model into **standardized regression model** using **correlation transform** to circumvent this issue.

### Correlation Transformation

Correlation transformation is simple modification of usual feature normalization procedure in machine learning. In feature normalization we make sure that each feature is centered (0 mean) and unit standard deviation. Standard feature normalization is given by these equations below,

$$
\begin{equation}
    \begin{aligned}
    (y_i)_{norm} &= \frac{y_i - \overline{y}}{s_y} \\
    (x_{ik})_{norm} &= \frac{x_{ik} - \overline{x_k}}{s_k}
    \end{aligned}
\end{equation}
$$

where $$\overline{y}, \overline{x_k}$$ are means of $$y, x_k$$ columns respectively, and $$s_y, s_k$$ are their respective standard deviations. The correlation transformation requires **only** one alteration, that is,

$$
\begin{equation}
    \begin{aligned}
    y_i^{*} &= \frac{1}{\sqrt{N-1}}\left(\frac{y_i - \overline{y}}{s_y}\right) = \frac{1}{\sqrt{N-1}}(y_i)_{norm} \\
    x_{ik}^{*} &= \frac{1}{\sqrt{N-1}}\left(\frac{x_{ik} - \overline{x_k}}{s_k}\right) = \frac{1}{\sqrt{N-1}}(x_{ik})_{norm}
    \end{aligned}
    \label{corTrans}
\end{equation}
$$

The regression model with the correlation transformed variables $$x_k^{*}, y^{*}$$ is called the **standardized regression model** and is defined formally as,

$$
\begin{equation}
y_i^{*} = \beta_1^{*}x_{i1}^{*} + ... + \beta_{n}^{*}x_{in}^{*} + \epsilon_i^{*}
\label{standModel}
\end{equation}
$$

Note the absence of the intercept term ($$\beta_0$$). Its straight forward to observe that the least squares calculation will always result in intercept term being $$0$$. How to recover original $$\beta_0, \beta_1, ..., \beta_n$$ from the estimated $$\beta_1^{*}, ..., \beta_n^{*}$$? Substituting $$\eqref{corTrans}$$ in $$\eqref{standModel}$$, we get


$$
\begin{equation}
\frac{1}{\sqrt{N-1}}\left(\frac{y_i - \overline{y}}{s_y}\right) = \beta_1^{*}\frac{1}{\sqrt{N-1}}\left(\frac{x_{i1} - \overline{x_1}}{s_1}\right) + \beta_2^{*}\frac{1}{\sqrt{N-1}}\left(\frac{x_{i2} - \overline{x_2}}{s_2}\right) + ... 
\end{equation}
$$

and now algebraic manipulations and comparing coefficients from the generalized regression model, we get,

$$
\begin{equation}
    \begin{aligned}
    \beta_k &= \frac{s_y}{s_k}\beta_k^{*} \ \ \ \ \forall k \in \{1,2,..,n\} \\
    \beta_0 &= \overline{y} - \beta_1\overline{x}_1 - ... - \beta_{n}\overline{x}_n
    \end{aligned}
\end{equation}
$$

> **Important Note:** 
>
> * For simplicity we refer $$\beta_i^{*}$$ as $$a_i$$ (the $$i^{th}$$ coefficient in standardized regression model). 
> 
> * Due to advantages in dealing with standardized model, we will assume that the input has underwent a covaraince transformation and the model we assume there after is always the standardized model.
> 
> * We refer $$x_{i1}^{*}$$ as $$x_{i1}, ..., x_{ik}^{*}$$ as $$x_{ik}, ...$$. (i.e) we assume standardized inputs from now on.

Just to familiarize you with this change, let me write standardized multiple regression with added noise again below, (again the intercept term is omitted)

$$
\begin{equation}
y=a_1x_1+a_2x_2+...+a_nx_n+\epsilon
\label{newStan}
\end{equation}
$$

The least squares objective squared deviation is,

$$
\begin{equation}
S^2=\sum_{i=1}^N(y_i - a_1x_{i1} - a_2x_{i2} - ... - a_nx_{in})^2
\label{lsError}
\end{equation}
$$

Minimizing this equation with respect to the parameters of the model (or) projecting the parameter vector on to the column space of $$X$$ (here $$X$$ is matrix of standardized feature variables), we obtain a normal system of equations of the form,

$$
\begin{equation}
C\vec{a}=\vec{r}
\label{standLS}
\end{equation}
$$

where $$C=X^TX$$ is the matrix of correlations $$r_{ij}$$ (between features $$x_i\ \&\ x_j$$), $$\vec{r}=X^T\vec{y}$$ is a vector of $$n$$ correlations $$r_{yi}$$ of $$y$$ with each $$x_j$$ feature variable. $$\vec{a}$$ is the vector of regression coefficients. Solving this linear system, we get the solution,

$$
\begin{equation}
\vec{a}=C^{-1}\vec{r}
\label{stanLSSol}
\end{equation}
$$

where $$C^{-1}$$ is the inverse correlation matrix. The square deviation can be represented in matrix form as,

$$
\begin{equation}
    \begin{aligned}
    S^2 &= \left(\vec{y}-X\vec{a}\right)^T\left(\vec{y}-X\vec{a}\right)\\
    &= \vec{y}^T\vec{y} - 2\vec{a}^T\left(X^T\vec{y}\right) + \vec{a}^T(X^TX)\vec{a} \\
    &= 1 - 2\vec{a}^T\vec{r} + \vec{a}^TC\vec{a}
    \end{aligned}
    \label{sqError}
\end{equation}
$$


## Revisiting R-squared for Regression Analysis ($$R^2$$)

Regression fit quality can be quantified by **coefficient of multiple determination** which is defined as,

$$
\begin{equation}
R^2 = 1 - \frac{RSS}{TSS}
\label{r2}
\end{equation}
$$

where $$R^2$$ represents the coefficient of multiple determination, $$RSS$$ is the residual sum of squares a.k.a $$S^2$$ (see $$\eqref{sqError}$$). $$TSS$$ is the total sum of squares which measures the deviation of $$y$$ from its mean $$\overline{y}; = \sum_i (y_i - \overline{y})^2$$. Its now clear that $$TSS = 1$$ as the data is standardized. 

Substituting $$\eqref{sqError}$$ in $$\eqref{r2}$$ we get,

$$
\begin{equation}
R^2 = 1 - S^2 = 2\vec{a}^T\vec{r} - \vec{a}^TC\vec{a} = \vec{a}^T\vec{r}
\label{r2_decomp}
\end{equation}
$$

$$R^2$$ is a bounded entity which always lies between 0 to 1. $$\eqref{r2_decomp}$$ gives a very important decomposition of $$R^2$$, (i,e)

$$
\begin{equation}
R^2 = \sum_{j=1}^nr_{yj}a_j = \sum_{j=1}^n NEF_j
\end{equation}
$$

Take a moment and let this result sink in. The entities $$r_{yj}a_j$$ is called the **net effect of the $$j^{th}$$ predictor** ($$NEF_j$$). This quantity plays an important role in gauging the importance of individual predictors in multiple linear regression setting. **Net effect of a predictor $$j$$ is the total influence that predictor $$j$$ has on the target, considering both its direct effect and indirect effect (via correlation with other predictors).**

To get a more clear picture, lets separate out the direct effect and indirect effect from $$NEF_j$$. Substitute $$\eqref{standLS}$$, $$\eqref{stanLSSol}$$ in $$\eqref{r2_decomp}$$ to get

$$
\begin{equation}
R^2 = \vec{r}^Ta = \vec{r}^TC^{-1}\vec{r} = \vec{a}^TC\vec{a}
\end{equation}
$$

Expanding the matrix product, we get an alternative way of expressing $$R^2$$,

$$
\begin{equation}
R^2 = \vec{a}^TC\vec{a} = \sum_{j}\overbrace{\Big(\underbrace{a_j^2}_{\text{direct}} + \underbrace{a_j\sum_{k\ne j}r_{jk}a_k}_{\text{indirect}}\Big)}^{NEF_j}
\label{dir_indir_nef}
\end{equation}
$$

Ah! now the the direct and indirect influences represented by $$NEF$$ is clear. From the above equation, $$NEF_j = a_j^2 + a_j\sum_{k\ne j}r_{jk}a_k$$, which can be divided into direct and indirect counterparts,

$$
\begin{aligned}
(NEF\ direct)_j &= a_j^2 \\
(NEF\ indirect)_j &= a_j\sum_{k\ne j}r_{jk}a_k
\end{aligned}
$$

Now, we will briefly look into the details of multicollinearity problem in linear regression, and see why people make a huge issue out of it.

## Problems caused by Multicollinearity

Multicollinearity arrises when the feature varaiables (predictor variables) are highly correlated among themselves. Some problems are listed below,

* Makes the parameter values sensitive. Values vary over a wide range when small change is made to the model. (like removing a feature, reducing the sample size, etc)

* Estimated standard deviation of predictor variables become large (quantified by **Variance Inflation Factor**)

* Signs of regression coefficients can be opposite to pair-wise correlations between target variable and the corresponding feature. $$(a_jr_{yj} < 0)$$

* Statistical significance of regression coefficients becomes questionable, as regression coefficients need not now, reflect the statistical relation that exists among the predictor variable and the target variable.

* Multicollinearity has bad effects on analysis of the influence of individual predictors (/variables) on the target variable.

To understand the problems of multicollinearity better, lets investigate $$NEF$$ has to say. Multicollinearity can change the sign of $$a_j$$ to opposite to that of the pairwise correlation $$r_{yj}$$, which means $$a_jr_{yj} = (NEF)_j < 0$$. What does this mean in terms of direct and indirect influence interpretation of net effect? As direct influence term in $$NEF_j \ge 0$$, it essentially means, that indirect influence is $$< 0$$ and overpowers direct influence. Analysing this with help of $$\eqref{dir_indir_nef}$$, 

$$
\begin{equation}
NEF_j < 0 \implies a_j^2 + a_j\sum_{k\ne j}r_{jk}a_k < 0 \implies a_j\sum_{k\ne j}r_{jk}a_k < -a_j^2
\end{equation}
$$

**Does $$NEF_j < 0$$ mean predictor $$j$$ must be removed from multiple regression formulation?**

NO. We will show later in the post that **any additional variable increases the coefficient of multiple determination $$R^2$$**. $$NEF_j < 0$$ means that the definition of net effects as $$r_{yj}a_j$$ is inadequate and not really completely representative of variable's influence. This calls for modification of net effects formulation (motivation for incremental net effects)!

## Incremental Influence of Predictors in Multiple Determination

**Objective**: In order to better understand the predictor's influence on the regression model, we now use the **incremental approach** (i,e) try to whats the marginal gain/loss in terms of performance does the inclusion of additional variable has on the multiple regression model.


Lets now aim to calculate the influence of $$x_n$$ (the last standardized variable) in the standardized regression model. The matrix $$C$$ (correlation matrix of all $$x$$'s) can be written in a block form,

$$
\begin{equation}
C = \begin{pmatrix}
A & \vec{\rho} \\
\vec{\rho}^T & 1
\end{pmatrix}
\end{equation}
$$

where $$A \in \mathbb{R}^{n-1 \times n-1}$$ matrix of correlation of $$(x_1, ... ,x_{n-1})$$  among themselves and $$\vec{\rho} \in \mathbb{R}^{n-1 \times 1}$$ is the correlation of $$(x_1, ... ,x_{n-1})$$ with $$x_n$$. Using the result from block matrix inversion, we get

$$
\begin{equation}
C^{-1} = \begin{pmatrix}
A^{-1} + q^{-1}\vec{b}\ \vec{b}^T & -q^{-1}\vec{b} \\
-q^{-1}\vec{b} & q^{-1}
\end{pmatrix}
\label{cinv_block}
\end{equation}
$$

where,

$$
\begin{equation}
    \begin{aligned}
    \vec{b} &= A^{-1}\vec{\rho}  \\
    q &= 1-\vec{\rho}^T\vec{b}
    \end{aligned}
    \label{blockInv}
\end{equation}
$$

Anything similar you can see with $$\vec{b}$$ in $$\eqref{blockInv}$$ and $$\vec{a}$$ in $$\eqref{stanLSSol}$$? Indeed! So $$\vec{b}$$ is the vector of regression coefficients when $$x_n$$ is regressed for, from other $$n-1$$ variables $$x_1, ..., x_{n-1}$$. Following the notations from $$\eqref{newStan}$$, $$\vec{b}$$ is the least squares solution for

$$
\begin{equation}
x_n = b_{1}x_1+b_{2}x_2+...+b_{n-1}x_{n-1} + \epsilon
\label{xn_model}
\end{equation}
$$

Similarly, observing $$\vec{\rho}^T\vec{b}$$ in $$\eqref{blockInv}$$ and noticing that it has similar similar to $$R^2$$ in $$\eqref{r2_decomp}$$ ($$\vec{a} \equiv \vec{b}, \vec{\rho} \equiv \vec{r}$$). So we denote $$\vec{\rho}^T\vec{b}$$ as $$R^2_{n.(-n)} = R^2_{n.123...n-1}$$ (i.e) the multiple determination coefficient for model $$\eqref{xn_model}$$. $$q$$ in $$\eqref{blockInv}$$ can be written as,

$$
\begin{equation}
q = 1 - R^2_{n.-n}
\label{q_exp}
\end{equation}
$$

Using $$\eqref{cinv_block}$$ in $$\eqref{stanLSSol}$$ for obtaining the solution for $$\eqref{newStan}$$,

$$
\begin{equation}
    \begin{aligned}
    \vec{a} = C^{-1}\vec{r} &=  \begin{pmatrix}
    A^{-1} + q^{-1}\vec{b}\ \vec{b}^T & -q^{-1}\vec{b} \\
    -q^{-1}\vec{b} & q^{-1}
    \end{pmatrix}
    \begin{pmatrix}
    \vec{r}_{y, -n}\\
    r_{yn}
    \end{pmatrix}\\
    &=
    \begin{pmatrix}
    A^{-1}\vec{r}_{y,-n} - q^{-1}(r_{yn}-\vec{b}^T\vec{r}_{y,-n})\vec{b})\\
    q^{-1}(r_{yn}-\vec{b}^T\vec{r}_{y,-n})
    \end{pmatrix}
    \end{aligned}
    \label{inc_soln}
\end{equation}
$$

where $$\vec{r} \in \mathbb{R}^n$$ a vector of all correlations between $$\{x_1,...,x_n\}$$ and $$y$$. $$r_{y,-n} \in \mathbb{R}^{n-1}$$ is the first $$n-1$$ correlations in $$r_y$$ and $$r_{yn}$$ is the correlation between $$x_n$$ and $$y$$ (last element in $$r_y$$).

In $$\eqref{inc_soln}$$, $$A^{-1}\vec{r}_{y,-n}$$ is the solution for the model, (multiple regression model with predictor variables $$x_1,.x_2,...,x_n$$),

$$
\begin{equation}
y = \beta_1x_1 + \beta_2x_2 + ... + \beta_{n-1}x_{n-1}
\label{n_1_model}
\end{equation}
$$

so we continue with the same notation in $$\eqref{n_1_model}$$, taking inspiration from $$\eqref{stanLSSol}$$, we get

$$
\begin{equation}
\vec{\beta} = A^{-1}\vec{r}_{y,-n} \in \mathbb{R}^{n-1}
\end{equation}
$$

The interpretation of this model, is that without $$x_n$$ how other features $$x_1, x_2, ..., x_{n-1}$$ influence the predictor variables. Now to understand the incremental gains obtained by including $$x_n$$ in regression analysis, we try to write $$R^2$$ of model $$\eqref{newStan}$$ in terms of $$R^2$$ of model as defined in $$\eqref{n_1_model}$$ + some entity depending on $$x_n$$.

Calculating $$R^2$$ of the model $$\eqref{newStan}$$ using $$\eqref{r2_decomp}$$ ($$a$$ defined in $$\eqref{inc_soln}$$),

$$
\begin{equation}
    \begin{aligned}
    R^2 &= \begin{pmatrix}
        \vec{r}_{y,-n} \\
        r_{yn}
        \end{pmatrix}
        \begin{pmatrix}
        A^{-1}\vec{r}_{y,-n} - q^{-1}(r_{yn}-\vec{b}^T\vec{r}_{y,-n})\vec{b})\\
        q^{-1}(r_{yn}-\vec{b}^T\vec{r}_{y,-n})
        \end{pmatrix} \\
        &= \vec{r}^T_{y,-n}A^{-1}\vec{r}_{y,-n} + q^{-1}\left(r_{yn}-\vec{b}^T\vec{r}_{y,-n}\right)^2
    \end{aligned}
    \label{r2_marginal}
\end{equation}
$$

Now, what is $$R^2_{n,-n}$$ (i.e) the $$R^2$$ of model $$\eqref{n_1_model}$$? We can use the expressions $$\eqref{r2_decomp}$$ ($$\vec{a} \equiv \vec{\beta}, \vec{r} \equiv \vec{r}_{y,n-1}$$), and write it as,

$$
\begin{equation}
R^2_{y,-n} = \vec{r}^T_{y,-n}\vec{\beta} = \vec{r}^T_{y,-n}A^{-1}\vec{r}_{y,-n}
\label{r2_n_1}
\end{equation}
$$

Applying $$\eqref{r2_n_1}$$ in $$\eqref{r2_marginal}$$, (and substituting $$q$$ from $$\eqref{q_exp}$$)

$$
\begin{equation}
R^2 = R^2_{y,-n} + \left(r_{yn}-\vec{b}^T\vec{r}_{y,-n}\right)^2/\left(1 - R^2_{n,-n}\right)
\label{inc_r2}
\end{equation}
$$

We see that addition of $$x_n$$ is clearly benificial as it increases the $$R^2$$ by a non-negative amount.

## Incremental Net Effect


From out analysis its clear that **addition of extra variables contribute in a non-negative way to $$R^2$$**. Now in just few steps, a new interesting result emerges, that is **incremental net effect** of a predictor.

Continuing our analysis, from the last row of $$\eqref{inc_soln}$$, we get 

$$
\begin{equation}
a_n = q^{-1}\left(r_{yn}-\vec{b}^T\vec{r}_{y,-n}\right)
\label{an_value}
\end{equation}
$$

Substituting $$\eqref{an_value}$$, $$\eqref{q_exp}$$ in $$\eqref{inc_r2}$$,

$$
\begin{equation}
R^2 = R^2_{y, -n} + \overbrace{a_n^2\left(1-R^2_{n,-n}\right)}^{U_n}
\end{equation}
$$

**Interpretation of the equation**: $$R^2$$ of model $$\eqref{newStan}$$ can be decomposed to the sum of $$R^2$$ of $$n-1$$ variables and the incremental value $$U_n = a_n^2\left(1-R^2_{n,-n}\right)$$ which depends on

* $$a_n^2$$ the coefficient of the nth predictor
* $$1 - R^2_{n,-n}$$ where $$R^2_{n,-n}$$ is the multiple determination in model $$\eqref{n_1_model}$$. As $$R^2 = 1-S^2$$ from $$\eqref{r2_decomp}$$, this term actually is the least squares error $$\eqref{lsError}$$ for model $$\eqref{n_1_model}$$.

Without loss of generality, assuming any predictor $$j$$ instead of $$x_n$$, we can write

$$
\begin{equation}
R^2 = R^2_{y,-j} + U_j
\label{inc_r2_j}
\end{equation}
$$

**The incremental gain in $$R^2$$ due to addition of predictor $$j$$ is given  by $$U_j$$ which is a measure of usefulness of the predictor $$j$$**.

Averaging $$\eqref{inc_r2_j}$$ for all n expressions, to combine all the marginal contributions from each predictor, we arrive at,

$$
\begin{equation}
R^2 = \frac{1}{n}\sum_{j=1}^nR^2_{y,-j} + \sum_{j=1}^n\left(\frac{1}{n}U_j\right)
\end{equation}
$$

**Note that incremental net effects is always positve, irrespective of whether influences of collinieariy is present or not**, unlike in the case of net effects defined in $$\eqref{dir_indir_nef}$$ which can be negative under correlated settings.