---
layout: post
title: "The In-Depth Guide to Interpretability Analysis in Linear Regression"
author: Lokesh Kumar
date: '2020-05-18 12:00:00'
category: 'Machine-Learning'
summary: Linear Regression is no doubt one of the most used machine learning algorithm. But how many of you know about details of linear regression such as how addition of a new extra variable affect the model?, when to remove a feature? Interested to answer these questions? Then this post is for you!
thumbnail: Interpretable Regression - 2.png
commments: true
---
## Contents
{: #contents}
*  <a href="#intro"><font size="5">Introduction</font></a>
*  <a href="#prereq"><font size="5"> Setting Pre-requisites right</font></a>
*  <a href="#predInfluence"><font size="5"> Incremental Influence of Predictors in Multiple Determination</font></a>
*  <a href="#inef"><font size="5"> Incremental Net Effect</font></a>
*  <a href="#eg"><font size="5"> A Simple Example</font></a>

Code used in the blog can be accessed on [GitHub](https://github.com/tlokeshkumar/interpretable-linear-regression)


## Introduction
{: #intro}
<a href="#contents"><button>Back to Contents</button></a>


Ever wondered what's the effect of the addition of an extra predictor variable $$x_i$$ to a linear regression model $$\eqref{newStan}$$? Can we separate the individual influences of each predictor on the target variable $$y$$? If we can do that, we can determine what features are important the most and invest more in collecting those features more accurately. Such an analysis also helps to explain the outcomes of the model and helps us to make costly decisions based on the model. This kind of analysis is particularly useful in the finance industry, medical industry where we can't afford to make decisions based on a model which we couldn't interpret. In the previous post [Everything You Need To Know About Interpretability in Linear Regression: Net Effects]({{ site.baseurl }}{% link _posts/2020-05-14-interpretable-linear-regression-1.md %}) we saw an interpretability technique called **Net Effects**, and in this post, we will see how we can define a more robust **Incremental Net Effects** which has a very nice intuitive explanation from cooperative game theory (reserved for another post) and `more interpretable than the traditional net effects`. 

If you have read the previous blog post [Everything You Need To Know About Interpretability in Linear Regression: Net Effects]({{ site.baseurl }}{% link _posts/2020-05-14-interpretable-linear-regression-1.md %}), then feel free to skip [Setting Pre-requisites right](#prereq) section. That post covers the prerequisites in detail.

## Setting Pre-requisites right
{: #prereq}
<a href="#contents"><button>Back to Contents</button></a>


Consider a standard regression model defined as follows,

$$
\begin{equation}
y=a_1x_1+a_2x_2+...+a_nx_n+\epsilon
\label{newStan}
\end{equation}
$$


We are given a dataset $\mathcal{D} = \\{X, y\\}$, where we are given $N$ data points $(X \in \mathbb{R}^{N \times n}, y \in \mathbb{R}^N)$. We are required to fit a straight line on this dataset such that we minimize the mean squared error $$\eqref{lsError}$$.

$$
\begin{equation}
S^2=\sum_{i=1}^N(y_i - a_1x_{i1} - a_2x_{i2} - ... - a_nx_{in})^2
\label{lsError}
\end{equation}
$$

This optimization problem has a closed-form solution and it can be represented as a solution to the set of linear equations expressed in matrix form $$\eqref{standLS}$$. We are skipping the derivation of this formula as its straightforward.

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


### Revisiting R-squared for Regression Analysis ($$R^2$$)

Regression fit quality can be quantified by **coefficient of multiple determination** which is a bounded metric from $[0,1]$. It represents the relative decrease in squared error obtained when we consider fit model $$\eqref{newStan}$$ as opposed to a simple constant model (i.e) a model which predicts a constant value irrespective of the input $x \in \mathbb{R}^n$. It also conveys how the inclusion of the feature variables $x$ helps in reduction of $S^2$.

A very important decomposition of $R^2$ which formed the foundation of net effects,

$$
\begin{equation}
R^2 = \sum_{j=1}^nr_{yj}a_j = \sum_{j=1}^n NEF_j
\label{r2_decomp}
\end{equation}
$$

$$r_{yj}a_j$$ is called the net effect of the $$j^{th}$$ predictor ($$NEF_j$$). **Net effect of a predictor $$j$$ is the total influence that predictor $$j$$ has on the target, considering both its direct effect and indirect effect (via correlation with other predictors).**

Consider an example of 10 predictors on a synthetically created linear dataset with Gaussian noise. We calculate the net effect of each predictor in the below plot.

{% include _plots/neteffects_2/netEffectsCoeff.html %}


### Problems caused by Multicollinearity

Multicollinearity arrises when the feature variables (predictor variables) are highly correlated among themselves. In that case, then he estimated standard deviation of predictor variables become large (quantified by **Variance Inflation Factor**). This also affects the interpretability measures, as now with correlated variables, it will be difficult to unmask the separate influences of each variable. Since the interpretability is under question now, the statistical significance of the predictor variables is also inaccurate. 

Multicollinearity can change the sign of $$a_j$$ to opposite to that of the pairwise correlation $$r_{yj}$$, which means $$a_jr_{yj} = (NEF)_j < 0$$. This is counter-intuitive because say for example $r_{yj} > 0$, which means positively correlated and $a_j < 0$, then from the model we can say that $y$ decreases by $a_j$ units for every unit increase in $x_j$. But the correlation paints a completely different picture as it says $x_j$ and $y$ are increasing together. This is what we mean by saying inaccuracies in statistical significance as now $NEF_j$ are no longer accurate descriptions of the predictor's influence on $y$.

The same dataset but the input features are correlated. Notice how the NEF values are negative for some predictors which are supposed to mean that they contribute negatively to $R^2$. We need a more sophisticated method to explain these observations as we will see in the next section.


{% include _plots/neteffects_2/netEffectsCoeffCorr.html %}


## Incremental Influence of Predictors in Multiple Determination
{: #predInfluence}
<a href="#contents"><button>Back to Contents</button></a>


**Objective**: In order to better understand the predictor's influence on the regression model, we now use the **incremental approach** (i,e) try to whats the marginal gain/loss in terms of performance does the inclusion of additional variable has on the multiple regression model.


Lets now aim to calculate the influence of $$x_n$$ (the last standardized variable) in the standardized regression model. The matrix $$C$$ (correlation matrix of all $$x$$'s) can be written in a block form,

<script type="math/tex; mode=display">
$$
\begin{equation}
C = \begin{pmatrix}
A & \vec{\rho} \\
\vec{\rho}^T & 1
\end{pmatrix}
\end{equation}
$$
</script>

where $$A \in \mathbb{R}^{n-1 \times n-1}$$ matrix of correlation of $$(x_1, ... ,x_{n-1})$$  among themselves and $$\vec{\rho} \in \mathbb{R}^{n-1 \times 1}$$ is the correlation of $$(x_1, ... ,x_{n-1})$$ with $$x_n$$. Using the result from block matrix inversion, we get

<script type="math/tex; mode=display">
$$
\begin{equation}
C^{-1} = \begin{pmatrix}
A^{-1} + q^{-1}\vec{b}\ \vec{b}^T & -q^{-1}\vec{b} \\
-q^{-1}\vec{b} & q^{-1}
\end{pmatrix}
\label{cinv_block}
\end{equation}
$$
</script>

where,

<script type="math/tex; mode=display">
$$
\begin{equation}
    \begin{aligned}
    \vec{b} &= A^{-1}\vec{\rho}  \\
    q &= 1-\vec{\rho}^T\vec{b}
    \end{aligned}
    \label{blockInv}
\end{equation}
$$
</script>

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

<script type="math/tex; mode=display">
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
</script>

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

<script type="math/tex; mode=display">
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
</script>

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

**We see that addition of $$x_n$$ is clearly beneficial as it increases the $$R^2$$ by a non-negative amount.**

## Incremental Net Effect
{: #inef}
<a href="#contents"><button>Back to Contents</button></a>


From our analysis, it's clear that **addition of extra variables contributes in a non-negative way to $$R^2$$**. Now in just a few steps, a new interesting result emerges, that is **incremental net effect** of a predictor.

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

**The incremental gain in $$R^2$$ due to addition of predictor $$j$$ is given by $$U_j$$ which is a measure of the usefulness of the predictor $$j$$**.

Averaging $$\eqref{inc_r2_j}$$ for all n expressions, to combine all the marginal contributions from each predictor, we arrive at,

$$
\begin{equation}
R^2 = \frac{1}{n}\sum_{j=1}^nR^2_{y,-j} + \sum_{j=1}^n\left(\frac{1}{n}U_j\right)
\label{inef_r2}
\end{equation}
$$

In the above equation, collecting all the terms related to predictor $j$, we arrive at $INEF_j$. aka incremental net effect for predictor $j$. **Note that incremental net effects are always positive, irrespective of whether influences of collinearity is present or not**, unlike in the case of net effects defined in $$\eqref{r2_decomp}$$ which can be negative under correlated settings.

Ok, lets now jump into an example and understand the calculation of incremental net effects.

## A Simple Example
{: #eg}
<a href="#contents"><button>Back to Contents</button></a>

Lets take a simplifying 2 predictor example, $y = a_1x_1 + a_2x_2$, then the $$\eqref{standLS}$$ becomes

<script type="math/tex; mode=display">
\begin{equation}
    \begin{bmatrix}
        1 & r_{12} \\
        r_{12} & 1
    \end{bmatrix}
    \begin{bmatrix}
        a_1 \\
        a_2
    \end{bmatrix}
    =
    \begin{bmatrix}
        r_{y1} \\
        r_{y2}
    \end{bmatrix}
\end{equation}
</script>

when solved leads us to this solution,

<script type="math/tex; mode=display">
\begin{equation}
    \begin{aligned}
        a_1 &= \frac{r_{y1}-r_{y2}r_{12}}{1-r^2_{12}} \\
        a_2 &= \frac{r_{y2}-r_{y1}r_{12}}{1-r^2_{12}}
    \end{aligned}
\end{equation}
</script>

Now coefficient of multiple determination can be computed using $\eqref{r2_decomp}$,

<script type="math/tex; mode=display">
\begin{equation}
    R^2 = a_1r_{y1} + a_2r_{y2} = \frac{r^2_{y1}+r^2_{y2}-2r_{y1}r_{y2}r_{21}}{1-r^2_{12}}
    \label{2pred_r2}
\end{equation}
</script>

Before we finish closing the example by calculating the incremental net effect for each predictor, lets better appreciate why the traditional net effect is inadequate with this example. Assume $r_{y2}=0$, what will happen to the net effects computed in $\eqref{2pred_r2}$?

<script type="math/tex; mode=display">
    \begin{equation}
        R^2 = a_1r_{y1} + a_2r_{y2} = \frac{r^2_{y1}}{1-r^2_{12}} + 0 = NEF_1 + NEF_2
    \end{equation}
</script>

So is the above equation trying to convey that addition of predictor 2, is not contributing anything to $R^2$? But we have seen in $\eqref{inc_r2}$ that addition of a new predictor always increases $R^2$ determined by $U_j$. So this is another reason to learn incremental net effects as they are always positive and sum up to $R^2$ giving a nice interpretable notion to them.

Getting back to calculating $INEF_1, INEF_2$, using $\eqref{inef_r2}$ in our 2 predictor case,

<script type="math/tex; mode=display">
\begin{equation}
    \begin{aligned}
        R^2 &= \left(\frac{1}{2}R^2_{y,-2} + \frac{1}{2}a_1^2(1-r^2_{21})\right) + \left(\frac{1}{2}R^2_{y,-1} + \frac{1}{2}a_2^2(1-r^2_{21})\right) \\
        &= \frac{1}{2}\left(r^2_{y1} + \frac{(r_{y1}-r_{y2}r_{12})^2}{1-r^2_{12}}\right) + \frac{1}{2}\left(r^2_{y2} + \frac{(r_{y2}-r_{y1}r_{12})^2}{1-r^2_{12}}\right) \\
        &= INEF_1 + INEF_2
    \end{aligned}
\end{equation}
</script>

Now lets consider the same assumption of $r_{y2}=0$ and analyse what happens,

<script type="math/tex; mode=display">
    R^2 = INEF_1 + INEF_2 = r^2_{y1}\frac{1 - r^2_{y1}/2}{1 - r^2_{12}} + r^2_{y2}\frac{r^2_{y2}/2}{1 - r^2_{12}}
</script>

and its clear to see that both $INEF_1, INEF_2$ are positive despite $r_{y2} = 0$. This captures the contribution of each predictor in a more accurate sense than the traditional net effects.

Notice in the below plot, how effectively incremental net effects distribute the influences of the predictors in a more interpretable way, even when the correlation is present and NEF values are negative.

{% include _plots/neteffects_2/netEffectsCoeffSV.html %}

Using $$\eqref{inef_r2}$$, and then collecting every element related to each predictor to compute the INEFs for each predictor might be cumbersome. We can do this type of analysis for 2 predictors or maybe till 3 predictors. So to get an intuitive explanation for incremental net effects, we take help of cooperative game theory. Consider the setting of the linear regression as a game and the players are the predictors. The total reward obtained by the players is $R^2$. Now the problem is `How can the total reward obtained by a coalition of all the players (predictors) be distributed among the individual players?` That's reserved for another post, the final post in the series **Interpretability in Linear Regression**.

The code used in the creation of the plots and values is [here](https://github.com/tlokeshkumar/interpretable-linear-regression). In the implementation of incremental net effects, I have used the cooperative game theory interpretation of incremental net effects to compute their values.