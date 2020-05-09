---
layout: post
title: Regression Meets Game Theory
author: Lokesh Kumar
date: '2020-05-09 17:00:00'
category: 'Machine-Learning'
summary: We address different methods to compute comparative importance of predictors in multiple linear regression. We also appreciate how ideas from co-operative game theory lends hands when models suffer from "multicollinearity"
thumbnail: Regression meets Game Theory.png
commments: true
---
## Contents
* Standardized Multiple Regression Model
* Revisiting R-squared for Regression Analysis ($$R^2$$)
* Problems caused by Multicollinearity
* Single Predictor's influence on $$R^2$$
* Incremental Net Effects
* Co-operative Game Theory - Shapley Values
* Grand Finale

## Standardized Multiple Regression Model

Consider we have $n$ predictor variables $$\left(X_1,...,X_n\right)$$ and target variable $$Y$$. For each data point $$i$$, these variables assume the form $$\left(X_{1i},...,X_{ni}, Y_i\right)$$ respectively. Now, we can formulate the linear model

$$
\begin{equation}
y_i = \beta_0 + \beta_1x_{i1}+\beta_2x_{i2}+...\beta_nx_{in}+\epsilon_i
\end{equation}
$$

which is linear combination of the features, and $$\epsilon \sim \mathcal{N}(0, \sigma^2)$$. This can also be rewritten as, 

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

Our main aim throughout the post will be to engage in answering how each predictor ($$\beta_iX_i$$) influences the target variable estimation. We also will try answering the question when our model suffers from multicollinearity. 

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

Can you think of any problem here? What if there are some variables in $$X$$ which are very small relative to others (orders of magnitude separation)? What if some features in $$X$$ are correlated with each other? This causes ill-conditioning of $$X^TX$$ and we run into numerical issues during its inversion. More precisely, we might potentially suffer from **round off errors** in computation. So, we need to reparameterize the regression model into **standardized regression model** using **correlation transform** to circumvent this issue.

### Correlation Transformation

Correlation transformation is simple modification of usual feature normalization procedure in machine learning. In feature normalization we make sure that each feature is centered (0 mean) and unit standard deviation. Standard feature normalization is given by these equations below,

$$
\begin{equation}
(y_i)_{norm} = \frac{y_i - \overline{y}}{s_y}
\end{equation}
$$

$$
\begin{equation}
(x_{ik})_{norm} = \frac{x_{ik} - \overline{x_k}}{s_k}
\end{equation}
$$

where $$\overline{Y}, \overline{X_k}$$ are means of $$Y, X_k$$ columns respectively, and $$s_y, s_k$$ are their respective standard deviations. The correlation transformation requires **only** one alteration, that is,

$$
\begin{equation}
y_i^{*} = \frac{1}{\sqrt{N-1}}\left(\frac{y_i - \overline{y}}{s_y}\right) = \frac{1}{\sqrt{N-1}}(y_i)_{norm}
\end{equation}
$$

$$
\begin{equation}
x_{ik}^{*} = \frac{1}{\sqrt{N-1}}\left(\frac{x_{ik} - \overline{x_k}}{s_k}\right) = \frac{1}{\sqrt{N-1}}(x_{ik})_{norm}
\end{equation}
$$

The regression model with the correlation transformed variables $$X_k^{*}, Y^{*}$$ is called the **standardized regression model** and is defined as follows,

$$
\begin{equation}
y_i^{*} = \beta_1^{*}x_{i1}^{*} + ... + \beta_{n}^{*}x_{in}^{*} + \epsilon_i^{*}
\end{equation}
$$

Note the absence of the intercept term ($$\beta_0$$). Its straight forward to observe that the least squares calculation will always result in intercept term being $$0$$. How to recover original $$\beta_0, \beta_1, ..., \beta_n$$ from the estimated $$\beta_1^{*}, ..., \beta_n^{*}$$? Simple substitution from correlation transform equations reveals us the answer.


$$
\begin{equation}
\frac{1}{\sqrt{N-1}}\left(\frac{y_i - \overline{y}}{s_y}\right) = \beta_1^{*}\frac{1}{\sqrt{N-1}}\left(\frac{x_{i1} - \overline{x_1}}{s_1}\right) + \beta_2^{*}\frac{1}{\sqrt{N-1}}\left(\frac{x_{i2} - \overline{x_2}}{s_2}\right) + ... 
\end{equation}
$$

and now algebraic manipulations and comparing coefficients from the generalized regression model, we get,

$$
\begin{equation}
\beta_k = \frac{s_y}{s_k}\beta_k^{*} \ \ \ \ \forall k \in \{1,2,..,n\}
\end{equation}
$$

$$
\begin{equation}
\beta_0 = \overline{y} - \beta_1\overline{x}_1 - ... - \beta_{n}\overline{x}_n
\end{equation}
$$

> **Important Note:** 
>
> * For simplicity we refer $$\beta_i^{*}$$ as $$a_i$$ (the $$i^{th}$$ coefficient in standardized regression model). 
> 
> * From now on, due to advantages in dealing with standardized model, we will assume that the input has underwent a covaraince transformation. 
> 
> * We refer $$x_{i1}^{*}$$ as $$x_{i1}, ..., x_{ik}^{*}$$ as $$x_{ik}, ...$$. (i.e) we assume standardized inputs from now on.

Just to familiarize with this change, standardized multiple regression with added noise is,

$$
\begin{equation}
y=a_1x_1+a_2x_2+...+a_nx_n+\epsilon
\end{equation}
$$

The least squares objective squared deviation is,

$$
\begin{equation}
S^2=\sum_{i=1}^N(y_i - a_1x_{i1} - a_2x_{i2} - ... - a_nx_{in})^2
\end{equation}
$$

Minimizing this equation with respect to the parameters of the model, we obtain a normal system of equations of the form,

$$
\begin{equation}
C\vec{a}=\vec{r}
\end{equation}
$$

where $$C=X^TX$$ is the matrix of correlations $$r_{ij}$$ (between features $$x_i\ \&\ x_j$$), $$\vec{r}=X^T\vec{y}$$ is a vector of $$n$$ correlations $$r_{yi}$$ of $$y$$ with each $$x_j$$ feature variable. $$\vec{a}$$ is the vector of regression coefficients. Solving this linear system, we get the solution,

$$
\begin{equation}
\vec{a}=C^{-1}\vec{r}
\end{equation}
$$

where $$c^{-1}$$ is the inverse correlation matrix. The square deviation can be represented in matrix form as,

$$
\begin{aligned}
S^2 &= \left(\vec{y}-X\vec{a}\right)^T\left(\vec{y}-X\vec{a}\right)\\
&= \vec{y}^T\vec{y} - 2\vec{a}^T\left(X^T\vec{y}\right) + \vec{a}^T(X^TX)\vec{a} \\
&= 1 - 2\vec{a}^T\vec{r} + \vec{a}^TC\vec{a}
\end{aligned}
$$


## Revisiting R-squared for Regression Analysis ($$R^2$$)

Regression fit quality can be quantified by **coefficient of multiple determination** which is defined as,

$$
\begin{equation}
R^2 = 1 - \frac{RSS}{TSS}
\end{equation}
$$

where $$R^2$$ represents the coefficient of multiple determination, $$RSS$$ is the residual sum of squares a.k.a $$S^2$$. $$TSS$$ is the total sum of squares which measures the deviation of $$y$$ from its mean $$\overline{y}; = \sum_i (y_i - \overline{y})^2$$. Its now clear that $$TSS = 1$$ as the data is standardized.

$$
\begin{equation}
R^2 = 1 - S^2 = 2\vec{a}^T\vec{r} - \vec{a}^TC\vec{a} = \vec{a}^T\vec{r}
\end{equation}
$$

$$R^2$$ is a bounded entity which always lies between 0 to 1. The above equation gives a very important decomposition of $$R^2$$, (i,e)

$$
\begin{equation}
R^2 = \sum_{j=1}^nr_{yj}a_j = \sum_{j=1}^n NEF_j
\end{equation}
$$

Take a moment and let this result sink in. The entities $$r_{yj}a_j$$ is called the **net effect of the $$j^{th}$$ predictor** ($$NEF_j$$). This quantity plays an important role in gauging the importance of individual predictors in multiple linear regression setting. Net effect of a predictor $$j$$ is the total influence that predictor $$j$$ has on the target, considering both its direct effect and indirect effect (via correlation with other predictors). 

To get a more clear picture, take your time and understand these equations below. The following equation is obtained by substituting the least squares solution in $$R^2$$ expression,

$$
\begin{equation}
R^2 = \vec{r}^Ta^T = \vec{r}^TC^{-1}\vec{r} = \vec{a}^TC\vec{a}
\end{equation}
$$

So, another way to represent $$R^2$$ is,

$$
\begin{equation}
R^2 = \vec{a}^TC\vec{a} = \sum_{i}\left(a_j^2 + a_j\sum_{k\ne j}r_{jk}a_k\right)
\end{equation}
$$

Ah! now the the direct and indirect influences represented by $$NEF$$ is clear. From the above equation, $$NEF_j = a_j^2 + a_j\sum_{k\ne j}r_{jk}a_k$$, which can be divided into direct and indirect counterparts,

$$
\begin{aligned}
(NEF\ direct)_i &= a_j^2 \\
(NEF\ indirect)_i &= a_j\sum_{k\ne j}r_{jk}a_k
\end{aligned}
$$

Now, we will briefly look into the details of multicollinearity problem in linear regression, and see why people make a huge issue out of it.

## Problems caused by Multicollinearity

Multicollinearity arrises when the feature varaiables (predictor variables) are highly correlated among themselves. Some problems are listed below,

* Makes the parameter values sensitive. Values vary over a wide range when small change is made to the model. (like removing a feature, reducing the sample size, etc)

* Sum of squares associated with the predictor variable varies and depends on the model design (depends on variables included in the model)

* Estimated standard deviation of predictor variables become large

* Signs of regression coefficients can be opposite to pair-wise correlations between target variable and the corresponding feature.

* Statistical significance of regression coefficients becomes questionable, as regression coefficients need not now, reflect the statistical relation that exists among the predictor variable and the target variable.

* Multicollinearity has bad effects on analysis of the influence of individual predictors (/variables) on the target variable.

To understand the problems of multicollinearity better, lets investigate $$NEF$$ expressions. Multicollinearity can change the sign of $$a_j$$ to opposite to that of the pairwise correlation $$r_{yj}$$, which means $$a_jr_{yj} = (NEF)_j < 0$$. What does this mean in terms of direct and indirect influence interpretation of net effect? As direct influence term in $$NEF_j \ge 0$$, it essentially means, that indirect influence is $$< 0$$ and overpowers direct influence.

$$
\begin{equation}
NEF_j < 0 \implies a_j^2 + a_j\sum_{k\ne j}r_{jk}a_k < 0 \implies a_j\sum_{k\ne j}r_{jk}a_k < -a_j^2
\end{equation}
$$

**Does $$NEF_j < 0$$ mean predictor $$j$$ must be removed from multiple regression formulation?**

NO. We will show later in the post that **any additional variable increases the coefficient of multiple determination $$R^2$$**. $$NEF_j < 0$$ means that the definition of net effects as $$r_{yj}a_j$$ needs more validation. This calls for modification net effects formulation (motivation for incremental net effects)!