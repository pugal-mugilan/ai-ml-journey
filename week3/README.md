Linear Algebra + Calculus + Gradient Descent

## What I Learned

### Linear Algebra 
- **Vectors & basis**: Every vector is a combination of basis vectors (î, ĵ)
- **Matrix transformations**: Matrix columns = where basis vectors land after transformation
- **Determinant**: Area scaling factor. det < 0 = orientation flip. det = 0 = dimension collapse (singular)
- **Inverse matrices**: Only exist when det ≠ 0. det = 0 means info is destroyed (many-to-one), so you can't reverse it
- **Column space**: All possible outputs of a transformation
- **Null space**: All inputs that get squished to zero

### Calculus 
- **Derivatives**: How much output changes when input changes (the "nudge" method)
- **Power rule**: x^n → n·x^(n-1)
- **Chain rule**: Nested functions → outer'(inner) × inner' (like km → m → cm conversion)
- **Two ways to find derivatives**: Numerical (plug in values, approximate) vs Analytical (derive a formula, exact)

### Gradient Descent 
- **The problem**: We have data but don't know what function produced it. We propose a function with unknown parameters.
- **The algorithm**: Start with random parameters → measure error (cost) → use derivative to find direction → nudge parameters → repeat
- **Cost function (MSE)**: Same logic as variance — square errors to prevent cancellation and punish big mistakes
- **Learning rate**: Step size. Too big = overshoot. Too small = takes forever.
- **Why it matters**: This is how every ML model — from linear regression to GPT — learns its parameters from data.

## Key Connections to ML
- det = 0 → redundant features → multicollinearity → regularization fixes this
- Chain rule → backpropagation (how neural networks learn)
- Gradient descent → the optimization engine behind all ML training

## Files
- `gradient_descent_from_scratch.py` — Gradient descent implemented in NumPy with learning rate comparison

## Resources Used
- 3Blue1Brown: Essence of Linear Algebra (Chapters 1-7)
- 3Blue1Brown: Essence of Calculus (Chapters 1-4)