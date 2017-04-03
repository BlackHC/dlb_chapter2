This are my notes and observations from reading the [Linear Algebra chapter](http://www.deeplearningbook.org/contents/linear_algebra.html) of the [Deep Learning](http://www.deeplearningbook.org/) book.

The following notes are presented in order of value to the reader. I start with a discussion of the Moore-Penrose pseudoinverse, followed by a short reflection on "broadcasting".

## Moore-Penrose pseudoinverse

The Moore-Penrose pseudoinverse is defined as:

> [$$ A^{+}:=\lim _{\alpha \searrow  0}\left( A^{T}A+\alpha I\right) ^{-1}A^{T} $$](http://www.deeplearningbook.org/contents/linear_algebra.html#pff)

You can click on the formula to jump into the book. (It's really awesome that all of it is online and you can deep link to pages.)

This formula is mystifying. What is its origin? And why does it do what the book says it does?

We can get a feeling for its origin by looking at the equation $Ax=b$ that it solves:
$$
\begin{align*} Ax &=b &|& A^{T}\cdot \\ A^{T}Ax &=A^{T}b &|& \left( A^{T}A\right) ^{-1} \cdot \\ \left( A^{T}A\right) ^{-1}A^{T}Ax &=\left( A^{T}A\right) ^{-1}A^{T}b & \\ x &=\left( A^{T}A\right) ^{-1}A^{T}b & \end{align*}
$$

So we start with the equation that we want to solve and find a more complex form for solving for $x$ than the usual left-multiplication with the inverse $A^{-1}$. This is already useful if $A^T A$ is invertible and $A$ is not square (because matrix inverses only exist for square matrices). 

This already looks similar to the pseudo-inverse, except for the $+\alpha I$ term. 

### Aside: there is also a right pseudo-inverse
We can guess its form by starting with:

$$
\begin{align*} yA&=b&|&\cdot A^{T}\\ yAA^{T}&=bA^{T}&|&\cdot \left( AA^{T}\right) ^{-1}\\ yAA^{T}\left( AA^{T}\right) ^{-1}&=bA^{T}\left( AA^{T}\right) ^{-1}\\ y&=bA^{T}\left( AA^{T}\right) ^{-1}\end{align*}
$$

From this, we can make an educated guess:

$$
^{+}A = A^{T}\lim _{\alpha \searrow 0}\left( AA^{T}+\alpha I\right) ^{-1}
$$

Since we cannot always invert $A^TA$ (respectively $AA^T$), an obvious question is:

### Why can we invert $AA^T+ \alpha I$ for positive $\alpha$?
Let's examine this. [For a matrix to be invertible](https://en.wikipedia.org/wiki/Invertible_matrix), its kernel has to only contain the zero vector:
$$\ker \left( A^{T}A+\alpha I\right) =\left\{ 0\right\}$$
To prove that this is the case for $\left( AA^{T}+\alpha I\right)$, we need to show that:
$$  \left( A^{T}A+\alpha I\right) v = 0 \implies v = 0$$
So starting with the left side, we can rephrase it as follows:
$$ \begin{align*} 
& & \left( A^{T}A+\alpha I\right) v&= 0\\ 
& \Leftrightarrow &A^{T}Av+\alpha Iv&= 0\\ 
& \Leftrightarrow &A^{T}Av &= -\alpha v
\end{align*} $$
If $v \ne 0$, this would mean that $-\alpha$ is a negative [eigenvalue](https://en.wikipedia.org/wiki/Eigenvalues_and_eigenvectors) of $A^TA$ as $\alpha$ is $>0$.

#### Can $A^TA$ have negative eigenvalues? 
Let's assume $v \ne 0$ and $-\alpha$ is a negative eigenvalue, that is $A^{T}Av = -\alpha v$ holds (and $\alpha > 0$). We can left-multiply with $v^T$ and obtain:
$$ 
\begin{align*} & \Leftrightarrow v^{T}A^{T}Av=v^{T}\left( -\alpha v\right) \\ & \Leftrightarrow \left\| Av\right\| _{2}^{2}=-\alpha v^{T}v=-\alpha \left\| v\right\| _{2}^{2}\\ & \Leftrightarrow -\alpha =\dfrac {\left\| Av\right\| _{2}^{2}} {\left\| v\right\| _{2}^{2}}\geq 0
\end{align*} 
$$
(We can divide by $\left\| v\right\| _{2}^{2}$ because we assume $v\ne0$.)
Now this means, that $-\alpha$ has to be $>0$, so it is not a negative eigenvalue and a contradiction to our initial assumption $\alpha > 0$. In fact, we have just shown that $A^TA$ in general can only have non-negative eigenvalues. (We can show the same for $AA^T$ the same way.) 

Because $A^TA$ cannot have negative eigenvalues, this means that $v$ cannot be $\ne 0$, so $v = 0$. This is what we wanted to show, and this means that the kernel only contains the zero vector, and $AA^T + \alpha I$ is invertible for $\alpha > 0$.

### What if $A^T A$ was invertible?

For a moment, let's consider the case when $A^T A$ is invertible: Is the Moore-Penrose pseudoinverse then equal to $\left (A^T A \right)^{-1} A^T$?
 
Matrix inversion is continuous in the space of invertible matrices. You might remember the definition of continuity from school. [A better definition of continuity](https://en.wikipedia.org/wiki/Continuous_function#Definition_in_terms_of_limits_of_sequences) is that it means being able to swap function and limit application:

> A function $f$ is continuous on its domain iff for any convergent series $x_n \to x$ for which $f \left( x \right )$ is defined
> $$ \lim_{n \to \infty } f \left ( x_n \right ) = f \left ( \lim_{n \to \infty} x_n \right ) = f \left( x \right).$$ 

Assuming $A^T A$ is invertible and using the fact that matrix inversion is continuous for invertible matrices, we now see:
$$ \begin{align*} 
\lim _{\alpha \searrow 0}\left( A^{T} A +\alpha I\right) ^{-1} A^T &= \left( \lim _{\alpha \searrow 0} A^{T} A +\alpha I\right) ^{-1} A^T \\
&= \left( A^{T}A+0 \, I\right) ^{-1} A^T \\
&= \left( A^{T}A\right) ^{-1} A^T
\end{align*}  $$

So in this special case, the pseudoinverse is exactly the solution we have come up with ourselves.

### Two properties of the pseudoinverse
There are two properties mentioned in text that are interesting but not obvious:

> [When $A$ has more columns than rows, then solving a linear equation using the pseudoinverse provides one of the many possible solutions. Specifically, it provides the solution $x=A^+y$ with minimal Euclidean norm $\left\| x \right\|_2$ among all possible solutions.
    When $A$ has more rows than columns, it is possible for there to be no solution. In this case, using the pseudoinverse gives us the $x$ for which $Ax$ is as close as possible to $y$ in terms of Euclidean norm $\left\| Ax - y \right\|_2$.](http://www.deeplearningbook.org/contents/linear_algebra.html#pf10)

These properties are not obvious and their deduction is enlightening towards the chosen definition of the pseudoinverse, specifically the use of the limit and the constraint of $\alpha$ to be $>0$.

#### A related optimization problem
To prove the properties, let's look at the following regularized minimum-least-squares problem. This is something that is formulated in more detail in [Chapter 4 of the book](http://www.deeplearningbook.org/contents/numerical.html#pf11), but it is quite useful here:

$$\min _{x}\left| \left| Ax-b\right| \right| _{2}^{2}+\alpha \left\| x\right\| _{2}^{2}, \, \alpha > 0$$

Let's solve this optimization problem. First, we set up a cost function:

$$ \begin{align*} 
c_\alpha\left( x\right) &=\left\| Ax-b\right\| _{2}^{2}+\alpha \left\| x\right\| _{2}^{2}\\
& =\left( Ax-b\right) ^{T}\left( Ax-b\right) +\alpha x^{T}x\\
& =x^{T}AAx-2b^{T}Ax+b^{T}b+\alpha x^{T} x \end{align*} \\
$$
The first derivative of $c_\alpha$ is:
$$ \nabla c_\alpha\left( x\right) =2A^{T}Ax-2A^{T}b+2\alpha x$$
And the second derivative is:
$$ H_\alpha \left ( x \right ) = 2A^T A + 2\alpha I$$
$H_\alpha$ is [positive definite](https://en.wikipedia.org/wiki/Positive-definite_matrix), that is $v^T H_\alpha \left ( x \right ) v \ge 0$ for all $v$. Why? We have already seen that $A^T A$ only has non-negative eigenvalues, so it is positive semidefinite by definition and $\alpha I$ is trivially positive definite for $\alpha > 0$. The sum of the two is positive definite again.  Thus $c$ is a [strictly convex function](https://en.wikipedia.org/wiki/Convex_function). This is along to lines of the one-dimensional case: when the second derivative is $> 0$ everywhere, the function is strictly convex. Convex functions have a global minimum and, for strictly convex functions, this global minimum is unique. So we know there is only exactly one point that minimizes the cost function $c_\alpha$.

We can determine this global minimum $x^*_\alpha$ by solving $\nabla c_\alpha \left ( x \right ) = 0$:

$$ \begin{align*} 
& & \nabla c_\alpha\left( x\right) &= 0 \\
& \Leftrightarrow & A^{T}Ax-A^{T}b+ax&=0\\
& \Leftrightarrow &\left( A^{T}A+\alpha I\right) x&=A^{T}b\\
& \Leftrightarrow & x&=\left( A^{T}A+\alpha I\right) ^{-1}A^{T}b\end{align*}
$$

This is exactly the definition of the pseudoinverse without the limit. So we have found:

$$
\DeclareMathOperator*{\argmin}{arg\,min} 
x^*_{\alpha} = \argmin _{x}\left| \left| Ax-b\right| \right| _{2}^{2}+\alpha \left\| x\right\| _{2}^{2} = \left( A^{T}A+\alpha I\right) ^{-1}A^{T}b
$$

with $c_\alpha \left (x^*_\alpha \right ) \leq c_\alpha \left ( x \right )$ for all $x$, and $x^*_\alpha$ denotes the minimum point.

This expression is continuous in $\alpha$, so we can take the limit $\alpha \searrow 0$. Remember, we need the constraint of $\alpha > 0$ for $A^TA + \alpha I$ to be invertible, so we can only take the limit from above. Taking the limit, we obtain:

$$
\DeclareMathOperator*{\argmin}{arg\,min} 
x^*= \lim_{x \searrow 0} \left( A^{T}A+\alpha I\right) ^{-1}A^{T}b
$$
with $c \left (x^* \right ) \leq c \left ( x \right )$ for all $x$ with
$$c \left ( x \right ) = \lim_{\alpha \searrow 0} \left\| Ax-b\right\| _{2}^{2}+\alpha \left\| x\right\| _{2}^{2} = \left\| Ax-b\right\| _{2}^{2}.$$

What do we gain from this? Well for one, we now know that this expression minimizes $\left| \left| Ax-b\right| \right| _{2}^{2}$. Furthermore, if there is a solution for $A x =b$, we can easily use a similar approach to see that the solution $x^*$ is smaller under Euclidean norm than any (other) solution $\hat x$ for $Ax=b$.

We do this in two steps. First, we observe that

$$A\hat x = b \Leftrightarrow A \hat x - b = 0 \Leftrightarrow \left\| A\widehat {x}-b\right\| _{2}^{2} = 0 $$

and, because  $x^*_\alpha$ minimizes $c_\alpha$,

$$
\begin{align*}
& & c_{\alpha}\left( x_{\alpha }^{*}\right) &\leq c_{\alpha }\left( \hat {x}\right) \\
& \Leftrightarrow & c_{\alpha}\left( x_{\alpha }^{*}\right) &\leq \left\| A\hat {x}-b\right\| _{2}^{2}+\left\| \hat {x}\right\| _{2}^{2}=0+\alpha \left\| \hat x\right\| _{2}^{2} \\
& \Leftrightarrow & c_{\alpha}\left( x_{\alpha }^{*}\right) & \leq \alpha \left\| \hat x\right\| _{2}^{2} \end{align*}
$$

This expression is again continuous, so we can take the limit $\alpha \searrow 0$, and we see:

$$\begin{align*}
& c \left( x^{*} \right) \leq 0 \\
\Leftrightarrow &  \left\| Ax^{*}-b\right\| _{2}^{2} \le 0
\end{align*}
$$
Because norms are always non-negative, we have $0 \le \left\| Ax^{*}-b\right\| _{2}^{2} \le 0$, so $\left\| Ax^{*}-b\right\| _{2}^{2} = 0$. And we have observed above that this is equivalent to $Ax^{*} = b$. So if there is at least one exact solution to the problem, we are sure to obtain an exact one, too. To be fair, we could have deduced this in the previous section. However, we can take another limit on the inequality $c_{\alpha}\left( x_{\alpha }^{*}\right) \leq \alpha \left\| \hat x\right\| _{2}^{2}$ and obtain a more interesting result. This time, we only take the limit $\alpha \searrow 0$ of $x^{*}_\alpha$, but keep $c_\alpha$ fixed:
$$
\begin{align*}
&&c_{\alpha}\left( x_{\alpha }^{*}\right) &\leq \alpha \left\| \hat x\right\| _{2}^{2} \\
\Rightarrow  && c_{\alpha}\left( \lim_{\alpha \searrow 0} x_{\alpha }^{*}\right) &\leq \alpha \left\| \hat x\right\| _{2}^{2}  \\
\Leftrightarrow && c_{\alpha}\left( x^{*} \right) &\leq \alpha \left\| \hat x\right\| _{2}^{2} \\
\Leftrightarrow && \left\| Ax^{*}-b\right\| _{2}^{2}+\alpha \left\| x^{*}\right\| _{2}^{2} &\leq \alpha \left\| \hat x\right\| _{2}^{2} \\
\Rightarrow && \alpha \left\| x^{*}\right\| _{2}^{2} &\leq \alpha \left\| \hat x\right\| _{2}^{2} \\
\Leftrightarrow && \left\| x^{*}\right\| _{2}^{2} &\leq \left\| \hat x\right\| _{2}^{2} 
\end{align*}
$$

Here, we use $\left\| Ax^{*}-b\right\| _{2}^{2} \ge 0$ to be able to drop the term and $\alpha > 0$ to preserve the direction of the inequality, and we obtain the second property about the length of $\left \| x^{*} \right \|$.

Now we have proved both properties that were mentioned in the book. On the one hand, if there are solutions, we will obtain one with minimal norm. On the other hand, if there are no solutions, using the pseudoinverse will provide us with an $x$ that at least minimizes $\left| \left| Ax-b\right| \right| _{2}^{2}$.

To convince yourself that the expressions above are indeed continuous, we can use the technical argument that we can rewrite $c_\alpha \left (x \right )$ into $c \left (\alpha, x \right )$ and see that it is a continuous function in two variables instead of going the route of using functional analysis and treating $c_\alpha$ as a convergent series of functions.

## Broadcasting notation: oh my...

The broadcasting notation in the Deep Learning book is weird. For one moment, let's ignore its origins from [numpy](https://docs.scipy.org/doc/numpy-1.10.1/user/basics.broadcasting.html), and let's look at what's happening.

> [In the context of deep learning, we also use some less conventional notation. We allow the addition of matrix and a vector, yielding another matrix: $C=A+b$, where $C_{i,j}=A_{i,j}+b_j$. In other words, the vector $b$ is added to each row of the matrix. This shorthand eliminates the need to deﬁne a matrix with $b$ copied into each row before doing the addition. This implicit copying of $b$ to many locations is called broadcasting.](http://www.deeplearningbook.org/contents/linear_algebra.html#pf4)

Let's say $A \in \mathbb{R}^{3 \times 3}$ and $b \in \mathbb{R}^3$, for example:
$$ A = \left[\begin{matrix}0 & 0 & 0\\1 & 1 & 1\\2 & 2 & 2\end{matrix}\right] , b = \left[\begin{matrix}1\\2\\3\end{matrix}\right] $$
Then, with broadcasting:
$$ A + b = \left[\begin{matrix}1 & 2 & 3\\2 & 3 & 4\\3 & 4 & 5\end{matrix}\right] $$

How do we get there? Essentially, we take the vector $b \in \mathbb{R}^3$, interpret it as a row vector, and add it to every row of the matrix, so:

$$ A + b= A + \left[\begin{matrix}1\\1\\1\end{matrix}\right] b^T $$

Wouldn't it make more sense to write this as $A + b^T$? The reason for the unintuitive notation lies in the details of broadcasting: whereas in maths, a vector $\mathbb{R}^3$ is identified as a column matrix $\mathbb{R}^{3 \times 1}$ , in the context of broadcasting it is treated as a row matrix $\mathbb{R}^{1 \times 3}$. This row matrix is then repeated across the $1$ dimension to make it into a $3 \times 3$ matrix that can be applied to $A$.  However, for matrix multiplications, $b$ continues to be treated as column matrix. This is really confusing. So what's the reason for this ambiguity?

My best guess is that in ML contexts, data entries (examples) are often treated as row vectors. For example, a design matrix stores a number of examples, each in a separate row. So if you want to change the bias of your examples, you want to add a bias vector as row-vector to each row. Indeed, in chapter 8, broadcasting is used to express batch normalization in eq (8.35):

[$$H'=\dfrac{H-\mu}{\sigma}$$](http://www.deeplearningbook.org/contents/optimization.html#pf2d)

I don't think this excuses the lack of clarity introduced with this notation, but then again it is too late to change and make everybody use a transposed design matrix... :) 

## Looking back and ahead

There is much more I could write about, but I think these were the most interesting bits from my notes. We have revisited convex functions, continuity and limits to motivate the curious definition of the Moore-Penrose pesudoinverse. Last but not least, we have looked at the origin of the broadcasting notation and why it might be confusing to someone who is new to ML. Next up: reading Chapter 3 and thinking about probability theory.

Stay tuned, \
 Andreas

PS: This a gist repost of a post on my blog http://blog.blackhc.net/2017/03/dlb-chapter2/index.html. I wish Medium was supporting LaTeX formulas properly...