# Copyright 2023 The Flax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import jax
import numpy as np
import pytest

from flax.experimental import nnx


class TestContainers:

  def test_node_idenpotence(self):
    x = nnx.Node(1)
    x = nnx.Node(x)

    assert isinstance(x, nnx.Node)

  def test_variable_idenpotence(self):
    x = nnx.Variable(1)
    x = nnx.Variable(x)

    assert isinstance(x, nnx.Variable)
    assert x.value == 1

  def test_variable_cannot_change_collection(self):
    x = nnx.Param(1)

    with pytest.raises(ValueError, match="is not compatible with return type"):
      x = nnx.BatchStat(x)

  def test_container_cannot_change_type(self):
    x = nnx.Variable(1)

    with pytest.raises(ValueError, match="is not compatible with return type"):
      x = nnx.Node(x)

    x = nnx.Node(2)

    with pytest.raises(ValueError, match="is not compatible with return type"):
      x = nnx.Variable(x)

  def test_static_is_empty(self):
    leaves = jax.tree_util.tree_leaves(nnx.Static(1))

    assert len(leaves) == 0

  def test_static_empty_pytree(self):
    static = nnx.Static(2)

    static = jax.tree_map(lambda x: x + 1, static)

    assert static.value == 2

  def test_static_array_not_jitable(self):
    @jax.jit
    def f(x):
      return x

    # first time you don't get an error due to a bug in jax
    f(nnx.Static(np.random.uniform(size=(10, 10))))

    with pytest.raises(ValueError):
      f(nnx.Static(np.random.uniform(size=(10, 10))))
