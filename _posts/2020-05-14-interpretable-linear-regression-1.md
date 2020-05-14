---
layout: post
title: "Interpretable Linear Regression: Net Effects"
author: Lokesh Kumar
date: '2020-05-14 15:02:00'
category: 'Machine-Learning'
summary: A regular problem in multiple regression is asserting the relative influence of the predictors in the model. Net Effects is a well-known technique that is used to measure the shares that each predictor have on the target variable in the coefficient of multiple determination.
thumbnail: Interpretable Regression - 1.png
comments: true
---
# Contents

{: #contents}
* <a href="#standardized_models"><font size="5">Standardized Multiple Regression Model</font>
* <a href="#r2_explanation"><font size="5">Revisiting R-squared for Regression Analysis</font>
* <a href="#NetEffects"><font size="5">Net Effects</font>
* <a href="#Multicollinearity"><font size="5">Problems caused by Multicollinearity</font>

Code used in the blog can be accessed on [GitHub](https://github.com/tlokeshkumar/interpretable-linear-regression)

## Standardized Multiple Regression Model
{: #standardized_models}
<a href="#contents"><button>Back to Contents</button></a>

Consider we have $n$ predictor variables $$\left(X_1,...,X_n\right)$$ and target variable $$Y$$. For each data point $$i$$, these variables assume the form $$\left(x_{1i},...,x_{ni}, y_i\right)$$ respectively. Now, we can formulate the linear model

\begin{equation}
y_i = \beta_0 + \beta_1x_{i1}+\beta_2x_{i2}+...\beta_nx_{in}+\epsilon_i
\label{genModel}
\end{equation}

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
$
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

\Longleftrightarrow

X\vec{\beta}+\vec{\epsilon}=\vec{Y}
\end{equation}
$

Let $$\vec{b}=[b_0, b_1, ..., b_n]^T$$ be the estimate of $$\beta$$ obtained by **least squares estimation**, 

$$
\begin{equation}
X^TX\vec{b}=X^T\vec{Y} \Longrightarrow \vec{b} = \left(X^TX\right)^{-1}X^T\vec{Y}
\end{equation}
$$

Can you think of any problem here? What if there are some variables in $$X$$ which are very small relative to others (orders of magnitude separation)? What if some features in $$X$$ are correlated with other variables? This will cause ill-conditioning of $$X^TX$$ which in turn results in numerical issues during its inversion. More precisely, we might potentially suffer from **round off errors** in computation. So, we need to reparameterize the regression model into **standardized regression model** using **correlation transform** to circumvent this issue.

### Correlation Transformation

Correlation transformation is simple modification of usual feature normalization procedure in machine learning. In feature normalization we make sure that each feature is centered (0 mean) and unit standard deviation. Standard feature normalization is given by these equations below,

\begin{equation}
    \begin{aligned}
    (y_i)_{norm} &= \frac{y_i - \overline{y}}{s_y} \\
    (x_{ik})_{norm} &= \frac{x_{ik} - \overline{x_k}}{s_k}
    \end{aligned}
\end{equation}

where $$\overline{y}, \overline{x_k}$$ are means of $$y, x_k$$ columns respectively, and $$s_y, s_k$$ are their respective standard deviations. The correlation transformation requires **only** one alteration, that is,

\begin{equation}
    \begin{aligned}
    y_i^{*} &= \frac{1}{\sqrt{N-1}}\left(\frac{y_i - \overline{y}}{s_y}\right) = \frac{1}{\sqrt{N-1}}(y_i)_{norm} \\
    x_{ik}^{*} &= \frac{1}{\sqrt{N-1}}\left(\frac{x_{ik} - \overline{x_k}}{s_k}\right) = \frac{1}{\sqrt{N-1}}(x_{ik})_{norm}
    \end{aligned}
    \label{corTrans}
\end{equation}

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

\begin{equation}
    \begin{aligned}
    \beta_k &= \frac{s_y}{s_k}\beta_k^{*} \ \ \ \ \forall k \in \{1,2,..,n\} \\
    \beta_0 &= \overline{y} - \beta_1\overline{x}_1 - ... - \beta_{n}\overline{x}_n
    \end{aligned}
\end{equation}

Look below for a well-commented code implementing the correlation transform and standardizing the dataset


```python
def standardize_dataset(X,y):
    '''
    Parameters
    ---
    X: Unnormalized Predictor data instances
    y: Unnormalized target data instances

    Returns
    ---
    Standardized X and y which can be used 
    for other tasks
    '''
    # X = [x_1, x_2, ... x_n]
    # no intercept term
    # X = N x n matrix
    # y = N dim vector
    
    # concatenating (X,y) to perform columnwise standardization
    # dataset =  N x (n+1) matrix
    dataset=np.c_[X,y]

    # Taking mean along axis=0 => mean = N dim vector
    mean = dataset.mean(axis=0)
    sq_diff = (dataset-mean)**2
    sq_diff = sq_diff.sum(axis=0)/(N-1)
    
    # Standard deviation taken along axis=0 (columns)
    std = np.sqrt(sq_diff)
    
    # Aplying the Correlation Transform on the dataset
    dataset = (dataset-mean)/(std*np.sqrt(N-1))
    
    # If beta is the true coefficients for the original model
    # The transformed beta which is the solution for standardized model is 
    # beta_reg=beta[1:]*std[:-1]/std[-1]
    
    X_norm=dataset[:,:-1]
    Y_norm=dataset[:, -1]
    return X_norm, Y_norm

```

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

where $$C^{-1}$$ is the inverse correlation matrix. The square deviation (Residual Sum of Squares, $$RSS$$) can be represented in matrix form as,


\begin{equation}
    \begin{aligned}
    S^2 &= \left(\vec{y}-X\vec{a}\right)^T\left(\vec{y}-X\vec{a}\right)\\
    &= \vec{y}^T\vec{y} - 2\vec{a}^T\left(X^T\vec{y}\right) + \vec{a}^T(X^TX)\vec{a} \\
    &= 1 - 2\vec{a}^T\vec{r} + \vec{a}^TC\vec{a}
    \end{aligned}
    \label{sqError}
\end{equation}


Let's code this and understand what we are exactly doing,

```python
# Note that these functions are a part of a class StandardizedLinearRegression
# For complete code: https://github.com/tlokeshkumar/interpretable-linear-regression

# Lets define a function set_XY(X, Y) which takes as input the data matrix (X) and 
# the target vector (y)
def set_XY(self, X, y):
    '''
    Parameters
    ---
    X: N x n data matrix
    y: N dim vector

    Computes the least squares solution and stores the estimated parameters
    in self.beta_estimate
    '''
    self.X=X
    self.y=y
    
    # Correlation Matrix of predictor variables
    # C_{ij} = r_{ij} = correlation between x_i and x_j
    # C_{ii} = 1, if X is standardized
    self.C = self.X.T.dot(self.X)

    # Correlation vector between target variables (y) and 
    # predictor variables (x): r_{xy}
    self.r = self.X.T.dot(self.y)

    # Check for 1x1 dimension
    if(len(self.C.shape)):
        # C^{-1}r
        self.beta_estimate = np.linalg.inv(self.C).dot(self.r)
    else:
        # (1/C)*r (if C is scalar)
        self.beta_estimate = (1/self.C)*self.r
```


## Revisiting R-squared for Regression Analysis ($$R^2$$)
{: #r2_explanation }
<a href="#contents"><button>Back to Contents</button></a>

Measuring the quality of the fit can be done in many ways. One such metric is to use the residual sum of squares $$S^2$$ $$\eqref{lsError}$$. We can also define a relative metric of performance, called the coefficient of determination or coefficient of multiple determination ($$R^2$$). To better appreciate what that is, 

Let's consider a simple model, a constant where the effect of predictor variables are not considered.

$$
\begin{equation}
y = a_0 + \epsilon
\label{baseline}
\end{equation}
$$

This will be our baseline model. So we will try to answer how well does our predictions $$y$$ improve when we consider model $$\eqref{genModel}$$ which captures linear relationships between predictor variables as opposed to a model $$\eqref{baseline}$$ which completely disregards such a relationship.

You will immediately realize that $$\eqref{baseline}$$ solution is just the mean of the target variables ($$\bar{y}$$). The residual sum of squares of model $$\eqref{baseline}$$ is called $$TSS$$ (total sum of squares).

Relative regression fit quality can be quantified by **coefficient of multiple determination** which is defined as,

$$
\begin{equation}
R^2 = 1 - \frac{RSS}{TSS}
\label{r2}
\end{equation}
$$

where $$R^2$$ represents the coefficient of multiple determination, $$RSS$$ is the residual sum of squares a.k.a $$S^2$$ (see $$\eqref{sqError}$$). $$TSS$$ is the total sum of squares which measures the deviation of $$y$$ from its mean $$\overline{y}; = \sum_i (y_i - \overline{y})^2$$. **$$R^2$$ is proportion reduction in squared error in using model $$\eqref{genModel}$$ instead of choosing a simple intercept model $$\eqref{baseline}$$**.

Its now clear that $$TSS = 1$$ as the data is standardized. 

```python
# Function to find RSS: Residual Sum of Squares (S^2)
def RSS(self):
    '''
    Residual sum of squares
    '''
    # Estimated y
    y_pred = self.X.dot(self.beta_estimate)
    
    # difference in prediction to true value
    # y_true-y_pred
    error=self.y - y_pred
    
    # Squared sum of differences = Residual Sum of Squares
    return (error).T.dot(error)

# Function to find TSS: Total Sum of Squares
def TSS(self):
    '''
    TSS for an intercept model
    (Y = \beta_0)
    '''
    # Squared sum of differences between y and its mean
    return (y-y.mean()).T.dot(y-y.mean())

# Function to calculate R^2
def r2(self):
    '''
    R^2 = 1 - (RSS/TSS)

    For standardized models,
    R^2 = 1 - RSS
    (TSS=1)
    '''
    return 1 - self.RSS()/self.TSS()
```

## Net Effects
{: #NetEffects }
<a href="#contents"><button>Back to Contents</button></a>

Now coming back to the analysis,


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
\label{r2_nef}
\end{equation}
$$

Take a moment and let this result sink in. The entities $$r_{yj}a_j$$ is called the **net effect of the $$j^{th}$$ predictor** ($$NEF_j$$). This quantity plays an important role in gauging the importance of individual predictors in a multiple linear regression setting. **Net effect of a predictor $$j$$ is the total influence that predictor $$j$$ has on the target, considering both its direct effect and indirect effect (via correlation with other predictors).**

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

Let's see the code to get an even more clear picture of whats going on.

```python
def net_effect(self):
    '''
    Returns a n dim vector of net effect of each predictor variable
    
    self.beta_estimate : The least squares solution for the standardized linear regression
    self.r : Correlation vector between target variables (y) and predictor variables (x): r_{xy}
            = X^Ty (in standardized multiple regression model)
    '''
    return self.beta_estimate*self.r
```

Visualizing net effects of each predictor variables and its corresponding regression coefficients, 

**Specifications**

* Given 10-dimensional feature vector
* Noise follows Gaussian distribution
* For the dataset generation code: [Visit my GitHub repo](https://github.com/tlokeshkumar/interpretable-linear-regression)

{% include _plots/neteffects/netEffectsCoeff.html %}


Now, we will briefly look into the details of the multicollinearity problem in linear regression, and see why people make a huge issue out of it.

## Problems caused by Multicollinearity
{: #Multicollinearity }
<a href="#contents"><button>Back to Contents</button></a>

Multicollinearity arrises when the feature variables (predictor variables) are highly correlated among themselves. Some problems are listed below,

* Makes the parameter values sensitive. Values vary over a wide range when a small change is made to the model. (like removing a feature, reducing the sample size, etc)

* Estimated standard deviation of predictor variables becomes large (quantified by **Variance Inflation Factor**)

* Signs of regression coefficients can be opposite to pair-wise correlations between target variable and the corresponding feature. $$(a_jr_{yj} < 0)$$

* Statistical significance of regression coefficients becomes questionable, as regression coefficients need not now, reflect the statistical relation that exists among the predictor variable and the target variable.

* Multicollinearity has bad effects on the analysis of the influence of individual predictors (/variables) on the target variable.

To understand the problems of multicollinearity better, lets investigate $$NEF$$ has to say. Multicollinearity can change the sign of $$a_j$$ to opposite to that of the pairwise correlation $$r_{yj}$$, which means $$a_jr_{yj} = (NEF)_j < 0$$. What does this mean in terms of direct and indirect influence interpretation of net effect? As direct influence term in $$NEF_j \ge 0$$, it essentially means, that indirect influence is $$< 0$$ and overpowers direct influence. Analysing this with help of $$\eqref{dir_indir_nef}$$, 

$$
\begin{equation}
NEF_j < 0 \implies a_j^2 + a_j\sum_{k\ne j}r_{jk}a_k < 0 \implies a_j\sum_{k\ne j}r_{jk}a_k < -a_j^2
\end{equation}
$$

**Does $$NEF_j < 0$$ mean predictor $$j$$ must be removed from multiple regression formulation?**

NO. We will show later in the post that **any additional variable increases the coefficient of multiple determination $$R^2$$**. $$NEF_j < 0$$ means that the definition of net effects as $$r_{yj}a_j$$ is inadequate and not completely representative of variable's influence. This calls for the modification of net effects formulation (motivation for incremental net effects)!

As an example, let's take an example of a severely correlated dataset (synthetically generated, with the same parameters as above) and observe its $$NEF$$ and parameter coefficients,

{% include _plots/neteffects/netEffectsSV.html %}

In the above graph, you can see that NEF values for some predictors are **negative**, and since $$\eqref{r2_nef}$$ is valid irrespective of the presence of multicollinearity or not, we also notice some predictors having NEF values **$$> 1.0$$**. So **the notion of NEF values capturing influences of individual predictors breaks down.** 

This calls for a more sophisticated and interesting measure called **Incremental Net Effects** which takes support from **co-operative game theory**, which I have also plotted in the above plot. You can notice how incremental net effects can spread the influence effectively among the predictors. Also, it has interesting properties like its always positive (even in the correlated case when net effects become negative) and like net effects, they sum up to $$R^2$$.