---
name: matrix-operations
description: Use specialized matrix types with symbolic tags for efficient linear algebra. Use when working with diagonal, block, or tagged matrices to avoid unnecessary dense computations.
---

# Matrix Operations

linsdex provides specialized matrix types that track structural properties for automatic optimization. Instead of always using dense matrices, the library uses diagonal, block, and tagged matrices to avoid unnecessary computation.

## When to Use

- Working with diagonal covariance matrices
- Building block-structured systems (e.g., position + velocity states)
- Optimizing linear algebra with zero or identity matrices
- Understanding the matrix type system in linsdex

## Matrix Types

### DiagonalMatrix

Stores only diagonal elements for O(n) operations instead of O(n²) or O(n³).

```python
import jax.numpy as jnp
from linsdex import DiagonalMatrix

# Create from diagonal elements
diag_elements = jnp.array([1.0, 2.0, 3.0])
D = DiagonalMatrix(diag_elements)

# Create identity matrix
I = DiagonalMatrix.eye(3)

# Efficient operations
D_inv = D.get_inverse()    # O(n) instead of O(n³)
log_det = D.get_log_det()  # O(n) instead of O(n³)
elements = D.get_elements()  # Get diagonal as array
```

### DenseMatrix

General dense matrices when structure cannot be exploited.

```python
from linsdex import DenseMatrix, TAGS

# Create a dense matrix
elements = jnp.array([[1.0, 0.5], [0.5, 2.0]])
M = DenseMatrix(elements, tags=TAGS.no_tags)

# Operations
M_inv = M.get_inverse()
chol = M.get_cholesky()
log_det = M.get_log_det()
```

### Block Matrices

For higher-order systems with natural block structure (e.g., position + velocity in tracking).

```python
from linsdex.matrix.block import Block2x2Matrix
from linsdex import DiagonalMatrix, DenseMatrix, TAGS

# Create a 2x2 block matrix
# [[A, B],
#  [C, D]]
A = DiagonalMatrix.eye(2)
B = DenseMatrix(jnp.zeros((2, 2)), tags=TAGS.zero_tags)
C = DenseMatrix(jnp.zeros((2, 2)), tags=TAGS.zero_tags)
D = DiagonalMatrix.eye(2)

block_matrix = Block2x2Matrix(A, B, C, D)

# Operations work on the block structure
inv = block_matrix.get_inverse()
```

## Matrix Tags

Tags track properties like zero and infinite values, enabling symbolic simplification before numerical computation.

```python
from linsdex import TAGS

# Available tag configurations
TAGS.no_tags   # Regular matrix (non-zero, non-infinite)
TAGS.zero_tags # Matrix is zero
TAGS.inf_tags  # Matrix has infinite elements (represents total uncertainty)
```

### How Tags Work

Tags propagate through operations automatically:

```python
from linsdex import DenseMatrix, TAGS

# Create a zero matrix
zero = DenseMatrix(jnp.zeros((3, 3)), tags=TAGS.zero_tags)
nonzero = DenseMatrix(jnp.eye(3), tags=TAGS.no_tags)

# Operations are detected symbolically
result = zero @ nonzero  # Detected as zero without computation
result = nonzero + zero  # Detected as nonzero without addition
```

### Infinite Tags for Uncertainty

Infinite matrices represent total uncertainty (precision = 0):

```python
# Used in potentials to represent uninformative priors
inf_precision = DenseMatrix(jnp.zeros((3, 3)), tags=TAGS.inf_tags)

# This indicates "no information" about a variable
```

## Code Examples

### Efficient Diagonal Operations

```python
from linsdex import DiagonalMatrix, StandardGaussian

dim = 100

# Independent dimensions with diagonal covariance
variances = jnp.ones(dim)
Sigma = DiagonalMatrix(variances)

# All operations are O(n) instead of O(n³)
precision = Sigma.get_inverse()
log_det = Sigma.get_log_det()
chol = Sigma.get_cholesky()

# Use in Gaussian distributions
mu = jnp.zeros(dim)
dist = StandardGaussian(mu, Sigma)
```

### Block Matrix for State Space Models

```python
from linsdex.matrix.block import Block2x2Matrix
from linsdex import DiagonalMatrix, DenseMatrix, TAGS

# 2D state: [position, velocity]
# Continuous-time dynamics: d/dt [x, v] = [[0, 1], [0, 0]] [x, v]
# Discrete transition matrix (Euler approximation):

dt = 0.1
dim = 1

# Position block
A11 = DiagonalMatrix.eye(dim)  # x_new = x + ...
A12 = DiagonalMatrix(jnp.ones(dim) * dt)  # ... + dt * v
A21 = DenseMatrix(jnp.zeros((dim, dim)), tags=TAGS.zero_tags)  # v_new = ...
A22 = DiagonalMatrix.eye(dim)  # ... + v

transition_matrix = Block2x2Matrix(A11, A12, A21, A22)
```

### Creating Matrices with Correct Tags

```python
from linsdex import DenseMatrix, DiagonalMatrix, TAGS

# Regular (non-zero) matrix
M = DenseMatrix(jnp.eye(3), tags=TAGS.no_tags)

# Zero matrix (will be detected in operations)
Z = DenseMatrix(jnp.zeros((3, 3)), tags=TAGS.zero_tags)

# Diagonal matrix (automatically handles tags)
D = DiagonalMatrix(jnp.array([1.0, 2.0, 3.0]))
```

### Matrix Operations

```python
from linsdex import DiagonalMatrix, DenseMatrix, TAGS

D = DiagonalMatrix(jnp.array([2.0, 3.0]))
M = DenseMatrix(jnp.array([[1.0, 0.5], [0.5, 1.0]]), tags=TAGS.no_tags)

# Matrix-vector multiplication
v = jnp.array([1.0, 2.0])
result = D @ v  # Efficient diagonal multiplication

# Matrix-matrix operations
result = D @ M  # Diagonal times dense

# Inverse
D_inv = D.get_inverse()

# Cholesky decomposition
chol = M.get_cholesky()

# Log determinant
log_det = D.get_log_det()
```

### Using with Gaussian Distributions

```python
from linsdex import StandardGaussian, NaturalGaussian, DiagonalMatrix

dim = 5

# Independent Gaussian with diagonal covariance
mu = jnp.zeros(dim)
Sigma = DiagonalMatrix.eye(dim) * 0.5  # Scalar multiplication

std_dist = StandardGaussian(mu, Sigma)

# Convert to natural form
nat_dist = std_dist.to_nat()  # Precision is also DiagonalMatrix
```

## Key Classes

- `DiagonalMatrix(elements)` - Diagonal matrix from 1D array
- `DenseMatrix(elements, tags)` - Dense matrix with symbolic tags
- `Block2x2Matrix(A, B, C, D)` - 2x2 block matrix
- `Block3x3Matrix(...)` - 3x3 block matrix
- `TAGS` - Symbolic tags for optimization

## Common Methods

All matrix types support:

- `get_inverse()` - Matrix inverse
- `get_cholesky()` - Cholesky decomposition
- `get_log_det()` - Log determinant
- `get_elements()` - Raw array elements
- `@` operator - Matrix multiplication (matmul)
- Scalar multiplication and addition

## Tips

- Use `DiagonalMatrix` whenever dimensions are independent to save computation
- Set correct tags when creating `DenseMatrix` to enable symbolic optimization
- Block matrices are useful for higher-order state space models
- Tags propagate automatically through operations
- The library chooses the most efficient representation for operation results
- Use `DiagonalMatrix.eye(n)` for identity matrices
