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

from typing import Callable

import jax

from flax.experimental import nnx


# lets pretend this function loads a pretrained model from a checkpoint
def load_backbone():
  return nnx.Linear(784, 128, ctx=nnx.context(0))


# create a simple linear classifier using a pretrained backbone
class Classifier(nnx.Module):

  def __init__(
      self, backbone: Callable[[jax.Array], jax.Array], *, ctx: nnx.Context
  ):
    self.backbone = backbone
    self.head = nnx.Linear(128, 10, ctx=ctx)

  def __call__(self, x):
    x = self.backbone(x)
    x = nnx.relu(x)
    x = self.head(x)
    return x


backbone = load_backbone()

# create the classifier using the pretrained backbone, here we are technically
# doing "parameter surgery", however, compared to Haiku/Flax where you must manually
# construct the parameter structure, in NNX this is done automatically
model = Classifier(backbone, ctx=nnx.context(42))

# create a filter to select all the parameters that are not part of the
# backbone, i.e. the classifier parameters
is_trainable = nnx.All(
    nnx.Param, lambda path, node: path.startswith("backbone")
)

# partition the parameters into trainable and non-trainable parameters
(trainable_params, non_trainable), moduledef = model.partition(
    is_trainable, ...
)

print("trainable_params =", jax.tree_map(jax.numpy.shape, trainable_params))
print("non_trainable = ", jax.tree_map(jax.numpy.shape, non_trainable))
