---
layout: post
category: concepts
title: flow matching (vector fields)
---

Picking up from the previous discussion on [probability paths](flowmatching-probpaths.html), let's continue the exploration of flow matching.

### vector fields
A vector field, $$u_t$$, is said to define an ODE whose solution is a flow, $$\psi_t$$.

The vector field is like the GPS that tells you "in 50 feet, turn right", etc. The flow, is then the route you end up taking if you listen to the GPS.

For every datapoint $$z \in \mathbb{R}^d$$, $$u_t( \cdot \vert z)$$ denotes the **conditional vector field**.

The analogy is let $$x$$ be some starting point in San Diego, and $$z$$ be the place you want to end up at. At each moment in time along our journey, the GPS, $$u_t(x \vert z)$$, will tell us how to move so that we end up at our destination.

So what is the actual equation for the conditional vector field? Let's find it in the case of Gaussian probability paths.

### conditional gaussian vector field
We start by constructing a specific flow, one that exactly follows the Gaussian probability path. Our target flow is:

$$
\psi_t(x \vert z) = \alpha_t z + \beta_t x
$$

In our case, we set our starting point to be:

$$
X_0 \sim p_\text{init} = \mathcal{N}(0,I_d)
$$

That means the ODE trajectory (our position given a timepoint), $$X_t$$, is:

$$
X_t = \psi_t(X_0 \vert z) = \alpha_t z + \beta_t X_0
$$

In other words, our trajectory matches our conditional probability path:

$$
X_t \sim \mathcal{N}(\alpha_tz, \beta_t^2I_d) = p_t(\cdot \vert z)
$$

Now, we need to extract the vector field from the flow. We do this by taking the time-derivative of the flow.

Intuitively, this makes sense. In the 1-D physics case, we know the derivative of position is velocity. So in general, we have:

$$
\frac{\text{d}}{\text{d}t}\psi_t(x \vert z) = u_t(\psi_t(x\vert z) \vert z)
$$

$$
\Leftrightarrow \dot{\alpha}_tz + \dot{\beta}_tx = u_t(\alpha_t z + \beta_tx \vert z)
$$

Now, we can use our actual starting point $$x = X_0$$.

$$
\Leftrightarrow \dot{\alpha}_tz + \dot{\beta}_tX_0 = u_t(\alpha_t z + \beta_tX_0 \vert z)
$$

We make two substitutions and rearrange: 

$$
X_t = \alpha_t z + \beta_tX_0, \quad X_0 = \frac{X_t-\alpha_tz}{\beta_t}
$$

$$
\Leftrightarrow \dot{\alpha}_tz + \dot{\beta}_t \frac{X_t - \alpha_tz}{\beta_t} = u_t(X_t \vert z)
$$

$$
\Leftrightarrow \left(\dot{\alpha}_t - \frac{\dot{\beta}_t}{\beta_t}\alpha_t \right)z + \frac{\dot{\beta}_t}{\beta_t}X_t = u_t(X_t \vert z)
$$

So, this is how we can construct a target vector field given a probability path.


If $$\alpha=t$$ and $$\beta=1-t$$, we have a specific probability path referred to as the **Gaussian CondOT probability path**:

$$
p_t(x|z) = \mathcal{N}(tz, (1-t)^2I_d)
$$

In this case, it is easy to compute:

$$
\alpha_t = t, \dot{\alpha}_t = 1, \quad \beta_t=1-t, \dot{\beta}_t = -1
$$

Then the corresponding vector field is:

$$
u_t(x \vert z) = \left( 1 - \frac{-t}{1-t} \right)z + \frac{-1}{1-t}x
$$

$$
\Leftrightarrow \left( \frac{1-t}{1-t} - \frac{-t}{1-t} \right)z + \frac{-1}{1-t}x
$$

$$
\Leftrightarrow \frac{1}{1-t} z - \frac{1}{1-t}x
$$

$$
\Leftrightarrow \frac{z-x}{1-t}
$$

If $$x=X_t$$ and $$X_0 = \epsilon$$, we can substitute:

$$
\Leftrightarrow \frac{z-X_t}{1-t}
$$

$$
\Leftrightarrow \frac{z-(tz+(1-t)\epsilon)}{1-t}
$$

$$
\Leftrightarrow \frac{(1-t)z - (1-t)\epsilon}{1-t}
$$

$$
\Leftrightarrow z - \epsilon
$$

For the Gaussian CondOT path, the conditional vector field is just:

$$
u_t(X_t \vert z) = z-\epsilon
$$

### marginal vector field
...