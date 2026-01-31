import jax
import jax.numpy as jnp
import equinox as eqx
import pytest
from linsdex.util.parallel_scan import parallel_scan, parallel_segmented_scan, segmented_scan, _tree_concatenate
from linsdex.linear_functional.linear_functional import LinearFunctional
from linsdex.linear_functional.quadratic_form import QuadraticForm
from linsdex.matrix.diagonal import DiagonalMatrix
from linsdex.matrix.tags import TAGS
import jax.tree_util as jtu
from typing import Union, Tuple, Optional, Any

class MockElem(eqx.Module):
    u: Any

    @property
    def batch_size(self):
        if hasattr(self.u, "batch_size"):
            return self.u.batch_size
        return self.u.shape[0]

    def __getitem__(self, idx):
        return MockElem(self.u[idx])

def test_tree_concatenate_structural_mismatch():
    dim = 2
    batch1 = 3
    batch2 = 2

    # Tree 1: simple array for logZ
    class MockTree(eqx.Module):
        u: LinearFunctional
        logZ: jnp.ndarray

    # Broadcast tags to match batch dimension
    tags1 = jtu.tree_map(lambda x: jnp.broadcast_to(x, (batch1,)), TAGS.no_tags)
    tree1 = MockTree(
        u=LinearFunctional(DiagonalMatrix(jnp.ones((batch1, dim)), tags=tags1), jnp.zeros((batch1, dim))),
        logZ=jnp.zeros(batch1)
    )

    # Tree 2: QuadraticForm for logZ
    tags2 = jtu.tree_map(lambda x: jnp.broadcast_to(x, (batch2,)), TAGS.no_tags)
    tree2 = MockTree(
        u=LinearFunctional(DiagonalMatrix(jnp.ones((batch2, dim)), tags=tags2), jnp.zeros((batch2, dim))),
        logZ=QuadraticForm(DiagonalMatrix(jnp.ones((batch2, dim)), tags=tags2), jnp.zeros((batch2, dim)), jnp.zeros(batch2))
    )

    # This should work now
    concatenated = _tree_concatenate(tree1, tree2)

    assert concatenated.u.b.shape[0] == batch1 + batch2
    assert isinstance(concatenated.logZ, QuadraticForm)
    assert concatenated.logZ.c.shape[0] == batch1 + batch2

def test_parallel_scan_structural_mismatch():
    dim = 2
    n = 4
    batch = n

    elems = MockElem(u=jnp.ones((batch, dim)))

    # Operator: promotes array to LinearFunctional
    def operator(a, b):
        ua, ub = a.u, b.u
        if isinstance(ua, LinearFunctional) or isinstance(ub, LinearFunctional):
            if not isinstance(ua, LinearFunctional):
                ua = LinearFunctional.identity(dim).zeros_like(LinearFunctional.identity(dim)) + ua
            if not isinstance(ub, LinearFunctional):
                ub = LinearFunctional.identity(dim).zeros_like(LinearFunctional.identity(dim)) + ub
            return MockElem(u=ua + ub)
        else:
            return MockElem(u=LinearFunctional.identity(dim).zeros_like(LinearFunctional.identity(dim)) + (ua + ub))

    result = parallel_scan(operator, elems)

    assert result.u.b.shape[0] == n
    assert isinstance(result.u, LinearFunctional)

def test_parallel_segmented_scan_basic():
    class MyObject(eqx.Module):
        x: jnp.ndarray
        y: jnp.ndarray

        @property
        def batch_size(self):
            return self.x.shape[0]

        def __getitem__(self, idx):
            return MyObject(self.x[idx], self.y[idx])

    def make_object(i):
        return MyObject(jnp.ones((2, 2))*i, i)

    n_elements = 6
    elems = jax.vmap(make_object)(jnp.arange(n_elements)+1)

    def operator(a, b):
        return MyObject(x=a.x + b.x, y=a.y + b.y)

    keep_indices = jnp.array([0, 2, 3, 5])
    reset_mask = (jnp.arange(elems.batch_size)[:,None] == keep_indices[None,:]).sum(axis=-1).astype(bool)

    out1 = parallel_segmented_scan(operator, elems, reset_mask)
    out2 = segmented_scan(operator, elems, reset_mask)

    assert jnp.allclose(out1.x, out2.x)
    assert jnp.allclose(out1.y, out2.y)

def test_parallel_segmented_scan_mismatch():
    dim = 2
    n = 4
    elems = MockElem(u=jnp.ones((n, dim)))
    reset_mask = jnp.array([True, False, True, False])

    def operator(a, b):
        ua, ub = a.u, b.u
        if isinstance(ua, LinearFunctional) or isinstance(ub, LinearFunctional):
            if not isinstance(ua, LinearFunctional):
                ua = LinearFunctional.identity(dim).zeros_like(LinearFunctional.identity(dim)) + ua
            if not isinstance(ub, LinearFunctional):
                ub = LinearFunctional.identity(dim).zeros_like(LinearFunctional.identity(dim)) + ub
            return MockElem(u=ua + ub)
        else:
            return MockElem(u=LinearFunctional.identity(dim).zeros_like(LinearFunctional.identity(dim)) + (ua + ub))

    out1 = parallel_segmented_scan(operator, elems, reset_mask)
    out2 = segmented_scan(operator, elems, reset_mask)

    assert isinstance(out1.u, LinearFunctional)
    assert isinstance(out2.u, LinearFunctional)
    assert out1.u.b.shape[0] == n
    assert out2.u.b.shape[0] == n
