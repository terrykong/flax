---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.13.8
---

+++ {"id": "2a9f78765c0c"}

# Scale up Flax Modules on multiple devices

This guide shows how to scale up [Flax Modules](https://flax.readthedocs.io/en/latest/developer_notes/module_lifecycle.html) on multiple devices and hosts using [`jax.jit`](https://jax.readthedocs.io/en/latest/jax-101/02-jitting.html) (formerly [`experimental.pjit`](https://jax.readthedocs.io/en/latest/jax.experimental.pjit.html#module-jax.experimental.pjit)) and [`flax.linen`](https://flax.readthedocs.io/en/latest/api_reference/flax.linen/index.html).

+++ {"id": "b1e0e5fc8bc1"}

## Flax and `jax.jit` scaled up

[`jax.jit`](https://jax.readthedocs.io/en/latest/_autosummary/jax.jit.html) follows the [Single Program Multi Data (SPMD)](https://jax.readthedocs.io/en/latest/glossary.html#term-SPMD) paradigm and automatically compiles your code to run it on multiple devices. You need to only specify how you want the input and output of your code to be partitioned, and the compiler will figure out how to: 1) partition everything inside; and 2) compile inter-device communications.

Flax provides several functionalities that can help you use auto-SPMD on [Flax Modules](https://flax.readthedocs.io/en/latest/developer_notes/module_lifecycle.html), including:

1. An interface to specify partitions of your data when defining [`flax.linen.Module`](https://flax.readthedocs.io/en/latest/api_reference/flax.linen/module.html).
2. Utility functions to generate the sharding information that `jax.jit` requires to run.
3. An interface to customize your axis names called "logical axis annotations" to decouple both your Module code and partition plan to experiment with different partition layouts more easily.

You can learn more about `jax.jit` APIs for scaling up in [JAX in multi-process environments](https://jax.readthedocs.io/en/latest/multi_process.html) and [Distributed arrays and automatic parallelization](https://jax.readthedocs.io/en/latest/notebooks/Distributed_arrays_and_automatic_parallelization.html) on JAX's documentation site.

+++ {"id": "a9601432b448"}

## Setup

Import some necessary dependencies.

**Note:** This guide uses the `--xla_force_host_platform_device_count=8` flag to emulate multiple devices in a CPU environment in a Google Colab/Jupyter Notebook. You don't need this if you are already using a multi-device TPU environment.

```{code-cell}
:id: 867203db3bef
:tags: [skip-execution]

# Once Flax v0.6.10 is released, there is no need to do this.
# ! pip3 install -qq "git+https://github.com/google/flax.git@main#egg=flax"
```

```{code-cell}
:id: f8f42d1174e5

import os
os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=8'
```

```{code-cell}
:id: b8da40732f0b

import functools
from typing import Optional, Callable

import numpy as np
import jax
from jax import lax, random, numpy as jnp

import flax
from flax import struct, traverse_util, linen as nn
from flax.core import freeze, unfreeze
from flax.training import train_state, checkpoints

import optax # Optax for common losses and optimizers. 
```

```{code-cell}
:id: bcc30de1d6eb

print(f'We have 8 fake JAX devices now: {jax.devices()}')
```

+++ {"id": "c0d280def897"}

The code below shows how to import and set up the JAX-level device API, following JAX's [Distributed arrays and automatic parallelization](https://jax.readthedocs.io/en/latest/notebooks/Distributed_arrays_and_automatic_parallelization.html) guide:

1. Start a 2x4 device `mesh` (8 devices) using JAX's `mesh_utils.create_device_mesh`. This layout is the same as the one of a [TPU v3-8](https://cloud.google.com/tpu/docs/system-architecture-tpu-vm#single_tpu_board).

2. Annotate each axis with a name using the `axis_names` parameter in `jax.sharding.Mesh`. A typical way to annotate axis names is `axis_name=('data', 'model')`, where:
  * `'data'`: the mesh dimension used for data-parallel sharding of the batch dimension of inputs and activations.
  * `'model'`: the mesh dimension used for sharding parameters of the model across devices.
  
3. Make a simple utility function `mesh_sharding` for generating a sharding object from the mesh and any layout.

```{code-cell}
:id: 684fe9fe13a0

from jax.sharding import Mesh, PartitionSpec, NamedSharding
from jax.lax import with_sharding_constraint
from jax.experimental import mesh_utils
```

```{code-cell}
:id: 4589d7a6d4bb

# Create a mesh and annotate each axis with a name.
device_mesh = mesh_utils.create_device_mesh((2, 4))
print(device_mesh)

mesh = Mesh(devices=device_mesh, axis_names=('data', 'model'))
print(mesh)

def mesh_sharding(pspec: PartitionSpec) -> NamedSharding:
  return NamedSharding(mesh, pspec)
```

+++ {"id": "307d39db6d94"}

## Define a layer

Before defining a simple model, create an example layer called `DotReluDot` (by subclassing `flax.linen.Module`). The layer creates two parameters `W1` and `W2` for dot product multiplication, and uses the `jax.nn.relu` (ReLU) activation function in-between.

To shard the parameters efficiently, apply the following APIs to annotate the parameters and intermediate variables:

1. Use [`flax.linen.with_partitioning`](https://flax.readthedocs.io/en/latest/api_reference/_autosummary/flax.linen.with_partitioning.html#flax.linen.with_partitioning) to decorate the initializer function when creating sub-layers or raw parameters.

2. Apply [`jax.lax.with_sharding_constraint`](https://github.com/google/jax/blob/main/jax/_src/pjit.py#L1516) (formerly, `pjit.with_sharding_constraint`) to annotate intermediate variables like `y` and `z` to force a particular sharding pattern when the ideal constraint is known.

  * This step is optional, but can sometimes help auto-SPMD to partition efficiently. In the example below, the call is not required, because XLA will figure out the same sharding layout for `y` and `z` regardless.

```{code-cell}
:id: b74c049968dc

class DotReluDot(nn.Module):
  depth: int
  dense_init: Callable = nn.initializers.xavier_normal()
  @nn.compact
  def __call__(self, x):
    
    y = nn.Dense(self.depth, 
                 kernel_init=nn.with_partitioning(self.dense_init, (None, 'model')),
                 use_bias=False,  # or overwrite with `bias_init`
                 )(x)

    y = jax.nn.relu(y)
    # Force a local sharding annotation.
    y = with_sharding_constraint(y, mesh_sharding(PartitionSpec('data', 'model')))

    W2 = self.param(
        'W2', 
        nn.with_partitioning(self.dense_init, ('model', None)),
        (self.depth, x.shape[-1]))
    
    z = jnp.dot(y, W2)
    # Force a local sharding annotation.
    z = with_sharding_constraint(z, mesh_sharding(PartitionSpec('data', None)))

    # Return a tuple to conform with the API `flax.linen.scan` as shown in the cell below.
    return z, None
```

+++ {"id": "cbac5321c08e"}

Note that device axis names like `'data'`, `'model'` or `None` are passed into both [`flax.linen.with_partitioning`](https://flax.readthedocs.io/en/latest/api_reference/_autosummary/flax.linen.with_partitioning.html) and [`jax.lax.with_sharding_constraint`](https://github.com/google/jax/blob/main/jax/_src/pjit.py#L1516) API calls. This refers to how each dimension of this data should be sharded — either across one of the device mesh dimensions, or not sharded at all.

For example:

* When you define `W1` with shape `(x.shape[-1], self.depth)` and annotate as `(None, 'model')`:

  * The first dimension (of length `x.shape[-1]`) will be replicated across all devices.
  * The second dimension (of length `self.depth`) will be sharded over the `'model'` axis of the device mesh. This means `W1` will be sharded 4-way on devices `(0, 4)`, `(1, 5)`, `(2, 6)` and `(3, 7)`, on this dimension.

* When you annotate the output `z` as `('data', None)`:

  * The first dimension — the batch dimension — will be sharded over the `'data'` axis. This means half of the batch will be processed on devices `0-3` (first four devices), and another half on devices `4-7` (the remaining four devices).
  * The second dimension — the data depth dimension — will be replicated across all devices.

+++ {"id": "b8389c11af79"}

## Define a model with `flax.linen.scan` lifted transformation

Having created `DotReluDot`, you can now define the `MLP` model (by subclassing [`flax.linen.Module`](https://flax.readthedocs.io/en/latest/api_reference/flax.linen/module.html#flax.linen.Module)) as multiple layers of `DotReluDot`.

To replicate identical layers, you can either use [`flax.linen.scan`](https://flax.readthedocs.io/en/latest/api_reference/_autosummary/flax.linen.scan.html), or a for-loop:

* `flax.linen.scan` can provide faster compilation times.
* The for-loop can be faster on runtime.

The code below shows how to apply both methods, and default with the for-loop, so that all the parameters are two-dimensional and you can visualize their sharding. 

The `flax.linen.scan` code is just to show that this API works with [Flax lifted transforms](https://flax.readthedocs.io/en/latest/developer_notes/lift.html#supported-transformations).

```{code-cell}
:id: a0ea0dcccbc3

class MLP(nn.Module):
  num_layers: int
  depth: int
  use_scan: bool
  @nn.compact
  def __call__(self, x):
    if self.use_scan:
      x, _ = nn.scan(DotReluDot, length=self.num_layers, 
                     variable_axes={"params": 0},
                     split_rngs={"params": True},
                     metadata_params={nn.PARTITION_NAME: None}
                     )(self.depth)(x)
    else:
      for i in range(self.num_layers):
        x, _ = DotReluDot(self.depth)(x)
    return x
```

+++ {"id": "44395b62561d"}

Now, create a `model` instance, and a sample input `x`.

```{code-cell}
:id: 5686299b4839

# MLP hyperparameters.
BATCH, LAYERS, DEPTH, USE_SCAN = 8, 4, 1024, False
# Create fake inputs.
x = jnp.ones((BATCH, DEPTH))
# Initialize a PRNG key.
k = random.PRNGKey(0)

# Create an Optax optimizer.
optimizer = optax.adam(learning_rate=0.001)
# Instantiate the model.
model = MLP(LAYERS, DEPTH, USE_SCAN)
```

+++ {"id": "5b3abfef359d"}

## Specify sharding

Next, you need to tell `jax.jit` how to shard our data across devices.

### The input's sharding

For data parallelism, you can shard the batched _input_ `x` across the `data` axis by denoting the batch axis as `'data'`. Then, use [`jax.device_put`](https://jax.readthedocs.io/en/latest/_autosummary/jax.device_put.html) to place it onto the correct `device`s.

```{code-cell}
:id: 8b913a2e57d3

x_sharding = mesh_sharding(PartitionSpec('data', None)) # dimensions: (batch, length)
x = jax.device_put(x, x_sharding)
jax.debug.visualize_array_sharding(x)
```

+++ {"id": "06d134795ae1"}

### The output's sharding

You need to compile `model.init()` (that is, [`flax.linen.Module.init()`](https://flax.readthedocs.io/en/latest/api_reference/flax.linen/module.html#flax.linen.Module.init)), and its output as a pytree of parameters. Additionally, you may sometimes need wrap it with a [`flax.training.train_state`](https://flax.readthedocs.io/en/latest/api_reference/flax.training.html#flax.training.train_state.TrainState) to track other variables, such as optimizer states, and that would make the output an even more complex pytree.

To achieve this, luckily, you don't have to hardcode the output's sharding by hand. Instead, you can:

1. Evaluate `model.init` (in this case, a wrapper of it) abstractly using [`jax.eval_shape`](https://jax.readthedocs.io/en/latest/_autosummary/jax.eval_shape.html).

1. Use [`flax.linen.get_sharding`](https://flax.readthedocs.io/en/latest/api_reference/_autosummary/flax.linen.get_sharding.html) to automatically generate the `jax.sharding.NamedSharding`.
   * This step utilizes the [`flax.linen.with_partitioning`](https://flax.readthedocs.io/en/latest/api_reference/_autosummary/flax.linen.with_partitioning.html) annotations in the earlier definition to generate the correct sharding for the parameters.

```{code-cell}
:id: 19094ec63385

def init_fn(k, x, model, optimizer):
  variables = model.init(k, x) # Initialize the model.
  state = train_state.TrainState.create( # Create a `TrainState`.
    apply_fn=model.apply,
    params=variables['params'],
    tx=optimizer)
  return state
```

```{code-cell}
:id: e49264a3c78e

# Create an abstract closure to wrap the function before feeding it in
# because `jax.eval_shape` only takes pytrees as arguments.
abstract_variables = jax.eval_shape(
    functools.partial(init_fn, model=model, optimizer=optimizer), k, x)

# This `state_sharding` has the same pytree structure as `state`, the output
# of the `init_fn`.
state_sharding = nn.get_sharding(abstract_variables, mesh)
state_sharding
```

+++ {"id": "2ec24614050b"}

## Compile the code

Now you can apply [`jax.jit`](https://jax.readthedocs.io/en/latest/jax-101/02-jitting.html) to your `init_fn`, but with two extra arguments: `in_shardings` and `out_shardings`.

Run it to get the `initialized_state`, in which parameters are sharded exactly as instructed:

```{code-cell}
:id: 5b6e699df733

jit_init_fn = jax.jit(init_fn, static_argnums=(2, 3),
                      in_shardings=(mesh_sharding(None), x_sharding),  # PRNG key and x
                      out_shardings=state_sharding)

initialized_state = jit_init_fn(k, x, model, optimizer)

# for weight, partitioned in initialized_state.params['DotReluDot_0'].items():
#     print(f'Sharding of {weight}: {partitioned.names}')
jax.debug.visualize_array_sharding(initialized_state.params['DotReluDot_0']['Dense_0']['kernel'].value)
jax.debug.visualize_array_sharding(initialized_state.params['DotReluDot_0']['W2'].value)
```

+++ {"id": "8f74b009f11f"}

## Inspect the Module output

Note that in the output of `initialized_state`, the `params` `W1` and `W2` are of type [`flax.linen.Partitioned`](https://flax.readthedocs.io/en/latest/api_reference/_autosummary/flax.linen.Partitioned.html). This is a wrapper around the actual `jax.Array` that allows Flax to record the axis names associated with it. 

You can access the raw `jax.Array` by adding `.value` when outside `jit`, or by `.unbox()` when inside.

```{code-cell}
:id: 19243982c892

print(type(initialized_state.params['DotReluDot_0']['Dense_0']['kernel']))
print(type(initialized_state.params['DotReluDot_0']['Dense_0']['kernel'].value))
print(initialized_state.params['DotReluDot_0']['Dense_0']['kernel'].names)
print(initialized_state.params['DotReluDot_0']['Dense_0']['kernel'].value.shape)
```

+++ {"id": "2beee7d27bdb"}

You can also check the underlying [`jax.sharding`](https://jax.readthedocs.io/en/latest/jax.sharding.html) of each parameter, which is now more internal than `NamedSharding`. Note that numbers like `initialized_state.step` are replicated across all devices.

```{code-cell}
:id: 2067c419a826

initialized_state.params['DotReluDot_0']['Dense_0']['kernel'].value.sharding
```

```{code-cell}
:id: d7cf0baa334b

print(initialized_state.step)
initialized_state.step.sharding
```

+++ {"id": "273547d3ab89"}

You can use [`jax.tree_map`](https://jax.readthedocs.io/en/latest/_autosummary/jax.tree_util.tree_map.html) to perform mass computation on a dict of boxed params, in the same way as on a dict of JAX arrays.

```{code-cell}
:id: 29b3dae156a2

diff = jax.tree_map(
    lambda a, b: a - b, 
    initialized_state.params['DotReluDot_0'], initialized_state.params['DotReluDot_0'])
print(jax.tree_map(jnp.shape, diff))
diff_array = diff['Dense_0']['kernel'].value
print(type(diff_array))
print(diff_array.shape)
```

+++ {"id": "f7e1ccb14c6b"}

## Compile the train step and inference 

Create a `jit`ted training step as follows:

```{code-cell}
:id: 4e3cc300cfee

@functools.partial(jax.jit, in_shardings=(state_sharding, x_sharding), 
                   out_shardings=state_sharding)
def train_step(state, x):
  # A fake loss function.
  def loss_unrolled(params):
    y = model.apply({'params': params}, x)
    return y.sum()
  grad_fn = jax.grad(loss_unrolled)
  grads = grad_fn(state.params)
  state = state.apply_gradients(grads=grads)
  return state

with mesh:
  new_state = train_step(initialized_state, x)
```

```{code-cell}
:id: 91c6c2662c12

print(f'Sharding of Weight 1:')
jax.debug.visualize_array_sharding(initialized_state.params['DotReluDot_0']['Dense_0']['kernel'].value)
print(f'Sharding of Weight 2:')
jax.debug.visualize_array_sharding(initialized_state.params['DotReluDot_0']['W2'].value)
```

+++ {"id": "2bae79e2e71b"}

Then, create a compiled inference step. Note that the output is also sharded along `(data, None)`.

```{code-cell}
:id: c9264a48b9ee

@functools.partial(jax.jit, in_shardings=(state_sharding, x_sharding), 
                   out_shardings=x_sharding)
def apply_fn(state, x):
  return state.apply_fn({'params': state.params}, x)

with mesh:
  y = apply_fn(new_state, x)
print(type(y))
print(y.dtype)
print(y.shape)
jax.debug.visualize_array_sharding(y)
```

+++ {"id": "7daa9e6e6eb4"}

## Profiling

If you are running on a TPU pod or a pod slice, you can use a custom `block_all` utility function, as defined below, to measure the performance:

```{code-cell}
:id: a68d7cb2eb89

%%timeit

def block_all(xs):
  jax.tree_map(lambda x: x.block_until_ready(), xs)
  return xs

with mesh:
  new_state = block_all(train_step(initialized_state, x))
```

+++ {"id": "51420b514d53"}

## Logical axis annotation

JAX's automatic SPMD encourages users to explore different sharding layouts to find the optimal one. To this end, in Flax you actually can annotate the dimensions of any data with more descriptive axis names (not just device mesh axis names like `'data'` and `'model'`). 

The `LogicalDotReluDot` and `LogicalMLP` Module definition below are similar to the Modules you created earlier, except for the following:

1. All axes are annotated with more concrete, meaningful names, such as `'embed'`, `'hidden'`, `'batch'` and `'layer'`. These names are referred to as _logical axis names_ in Flax. They make the dimensional changes inside model definitions more readable.

2. [`flax.linen.with_logical_partitioning`](https://flax.readthedocs.io/en/latest/api_reference/_autosummary/flax.linen.with_logical_partitioning.html) replaces `flax.linen.with_partitioning`; and [`flax.linen.with_logical_constraint`](https://flax.readthedocs.io/en/latest/api_reference/_autosummary/flax.linen.with_logical_constraint.html#flax-linen-with-logical-constraint) replaces `jax.lax.with_sharding_constraint`, to recognize the logical axis names.

```{code-cell}
:id: a26f85a9e772

class LogicalDotReluDot(nn.Module):
  depth: int
  dense_init: Callable = nn.initializers.xavier_normal()
  @nn.compact
  def __call__(self, x):    
    y = nn.Dense(self.depth, 
                 kernel_init=nn.with_logical_partitioning(self.dense_init, ('embed', 'hidden')),
                 use_bias=False,  # or overwrite with `bias_init`
                 )(x)

    y = jax.nn.relu(y)
    # Force a local sharding annotation.
    y = with_sharding_constraint(y, mesh_sharding(PartitionSpec('data', 'model')))

    W2 = self.param(
        'W2', 
        nn.with_logical_partitioning(self.dense_init, ('hidden', 'embed')),
        (self.depth, x.shape[-1]))

    z = jnp.dot(y, W2)
    # Force a local sharding annotation.
    z = nn.with_logical_constraint(z, ('batch', 'embed'))
    return z, None

class LogicalMLP(nn.Module):
  num_layers: int
  depth: int
  use_scan: bool
  @nn.compact
  def __call__(self, x):
    if self.use_scan:
      x, _ = nn.scan(LogicalDotReluDot, length=self.num_layers, 
                    variable_axes={"params": 0},
                    split_rngs={"params": True},
                    metadata_params={nn.PARTITION_NAME: 'layer'}
                    )(self.depth)(x)
    else:
      for i in range(self.num_layers):
        x, _ = LogicalDotReluDot(self.depth)(x)
    return x
```

+++ {"id": "0de93ec6cbd6"}

Now, initiate a model and try to figure out what sharding its `state` should have.

To allow the device mesh to take your model correctly, you need to decide which of these logical axis names are mapped to the device axis `'data'` or `'model'`. This rule is a list of (`logical_axis_name`, `device_axis_name`) tuples, and [`flax.linen.logical_to_mesh_sharding`](https://flax.readthedocs.io/en/latest/api_reference/_autosummary/flax.linen.logical_to_mesh_sharding.html#flax-linen-logical-to-mesh-sharding) will convert them to the kind of sharding that the device mesh can understand.

This allows you to change the rules and try out new partition layouts without modifying the model definition.

```{code-cell}
:id: 14db7a1e30fd

# Unspecified rule means unsharded by default, so no need to specify `('embed', None)` and `('layer', None)`.
rules = (('batch', 'data'),
         ('hidden', 'model'))

logical_model = LogicalMLP(LAYERS, DEPTH, USE_SCAN)

logical_abstract_variables = jax.eval_shape(
    functools.partial(init_fn, model=logical_model, optimizer=optimizer), k, x)
logical_state_spec = nn.get_partition_spec(logical_abstract_variables)
print('annotations are logical, not mesh-specific: ', 
      logical_state_spec.params['LogicalDotReluDot_0']['Dense_0']['kernel'])

logical_state_sharding = nn.logical_to_mesh_sharding(logical_state_spec, mesh, rules)
print('sharding annotations are mesh-specific: ', 
      logical_state_sharding.params['LogicalDotReluDot_0']['Dense_0']['kernel'].spec)
```

+++ {"id": "58475fffb2de"}

You can verify that the `logical_state_spec` here has the same content as `state_spec` in the previous ("non-logical") example. This allows you to `jax.jit` your Module's [`flax.linen.Module.init`](https://flax.readthedocs.io/en/latest/api_reference/flax.linen/module.html#flax.linen.Module.init) and [`flax.linen.Module.apply`](https://flax.readthedocs.io/en/latest/api_reference/flax.linen/module.html#flax.linen.Module.apply) the same way in the above above.

```{code-cell}
:id: 589ff774bb4c

state_sharding.params['DotReluDot_0'] == logical_state_sharding.params['LogicalDotReluDot_0']
```

```{code-cell}
:id: 77e07a0ab309

logical_jit_init_fn = jax.jit(init_fn, static_argnums=(2, 3),
                      in_shardings=(mesh_sharding(None), x_sharding),  # PRNG key and x
                      out_shardings=logical_state_sharding)

logical_initialized_state = logical_jit_init_fn(k, x, logical_model, optimizer)
```

```{code-cell}
:id: fb53bc20e0f9

print(f'Sharding of Weight 1:')
jax.debug.visualize_array_sharding(logical_initialized_state.params['LogicalDotReluDot_0']['Dense_0']['kernel'].value)
print(f'Sharding of Weight 2:')
jax.debug.visualize_array_sharding(logical_initialized_state.params['LogicalDotReluDot_0']['W2'].value)
```

+++ {"id": "ae1754a3031d"}

## When to use device axis / logical axis

Choosing when to use a device or logical axis depends on how much you want to control the partitioning of your model:

* **Device mesh axis**: If you want a very simple model, or you are very confident of your way of partitioning, defining it with __device mesh axis__ can potentially save you a few extra lines of code of converting the logical naming back to the device naming.

* **logical naming**: On the other hand, the __logical naming__ helpers can be useful for exploring different sharding layouts. Use this if you want to experiment around and find the most optimal partition layout for your model.

* **Device axis names**: In really advanced use cases, you may have more complicated sharding patterns that require annotating *activation* dimension names differently from *parameter* dimension names. If you wish to have more fine-grained control on manual mesh assignments, directly using __device axis names__ could be more helpful.

+++ {"id": "576bdd5cd782"}

## Save the data

To save the cross-device array, you can use [`flax.training.checkpoints`](https://flax.readthedocs.io/en/latest/_modules/flax/training/checkpoints.html), as shown in the [Save and load checkpoints guide - Multi-host/multi-process checkpointing](https://flax.readthedocs.io/en/latest/guides/use_checkpointing.html#multi-host-multi-process-checkpointing). This is especially required if you are running on a multi-host environment (for example, a TPU pod).

Keep in mind that to restore the arrays to the desired partition, you need to provide a sample `target` pytree that has the same structure and has the desired [`jax.sharding.Sharding`](https://jax.readthedocs.io/en/latest/jax.sharding.html#jax.sharding.Sharding) in place for each JAX array. The sharding you use to restore the array doesn't necessarily need to be the same as the ones you used to store the array.
