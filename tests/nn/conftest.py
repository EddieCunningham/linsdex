import pytest
import jax.random as random

@pytest.fixture
def key():
  return random.PRNGKey(0)
