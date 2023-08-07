Why NNX?
========

Flax Linen is currently the most flexible and powerful way to write neural networks in JAX. The main features that have made it so popular are `State collections <https://flax.readthedocs.io/en/latest/glossary.html#term-Variable-collections>`__, `RNG handling <https://flax.readthedocs.io/en/latest/glossary.html#term-RNG-sequences>`__, `Collection-aware lifted transformations <https://flax.readthedocs.io/en/latest/developer_notes/lift.html>`__, and `Leaf metadata <https://flax.readthedocs.io/en/latest/api_reference/flax.linen/_autosummary/flax.linen.with_partitioning.html#flax.linen.with_partitioning>`__.


However, Linen's power has come at a cost:

* The ``init`` and ``apply`` APIs require a learning curve (on top of JAX's learning curve).
* The Module's dataclass and ``compact`` semantics drift away from regular Python semantics and have a very complex internal implementation.
* It is not very easily to integrate pre-trained models into bigger models as the Module structure is separate from the ``params`` structure.
* The implementation of the lifted transformations is very complex.

Flax NNX is an attempt to keep the features that made Linen great while simplifying the API and making it more Pythonic.


NNX is Pythonic
---------------

* Example of building a Module


.. codediff::
  :title_left: NNX
  :title_right: Linen
  :sync:

  from flax.experimental import nnx
  import jax
  import jax.numpy as jnp


  class Count(nnx.Variable): pass

  class Linear(nnx.Module):

    def __init__(self, din: int, dout: int, *, ctx: nnx.Context):
      self.din, self.dout = din, dout
      key = ctx.make_rng("params")
      self.w = nnx.Param(jax.random.uniform(key, (din, dout)))
      self.b = nnx.Param(jnp.zeros((dout,)))
      self.count = Count(0)  # track the number of calls

    def __call__(self, x) -> jax.Array:
      self.count += 1
      return x @ self.w + self.b


  model = Linear(din=5, dout=2, ctx=nnx.context(0))
  x = jnp.ones((1, 5))
  y = model(x)

  ---

  import flax.linen as nn
  import jax
  import jax.numpy as jnp


  class Linear(nn.Module):
    din: int
    dout: int

    def setup(self):
      din, dout = self.din, self.dout
      key = self.make_rng("params") if self.is_initializing() else None
      self.w = self.variable("params", "w", jax.random.uniform, key, (din, dout))
      self.b = self.variable("params", "b", jnp.zeros, (dout,))
      self.count = self.variable("counts", "count", lambda: 0)

    def __call__(self, x) -> jax.Array:
      self.count.value += 1
      return x @ self.w.value + self.b.value

  model = Linear(din=5, dout=2)
  x = jnp.ones((1, 5))
  variables = model.init(jax.random.PRNGKey(0), x)
  params, counts = variables["params"], variables["counts"]
  y, updates = model.apply(
      {"params": params, "counts": counts}, x, mutable=["counts"]
  )
  counts = updates["counts"]

.. codediff::
  :title_left: NNX
  :title_right: Linen
  :sync:

  print(f"{model.count = }")
  print(f"{model.w = }")
  print(f"{model.b = }")
  print(f"{model = }")

  ---

  bounded_model = model.bind({"params": params, "counts": counts})

  print(f"{bounded_model.count.value = }")
  print(f"{bounded_model.w.value = }")
  print(f"{bounded_model.b.value = }")
  print(f"{bounded_model = }")

**Output:**

.. tab-set::

  .. tab-item:: NNX
    :sync: NNX

    .. code-block:: python

      model.count = 1
      model.w = Array([[0.0779959 , 0.8061936 ],
            [0.05617034, 0.55959475],
            [0.3948189 , 0.5856023 ],
            [0.82162833, 0.27394366],
            [0.07696676, 0.8982161 ]], dtype=float32)
      model.b = Array([0., 0.], dtype=float32)
      model = Linear(
        din=5,
        dout=2
      )


  .. tab-item:: Linen
    :sync: Linen

    .. code-block:: python

      bounded_model.count.value = 1
      bounded_model.w.value = Array([[0.76684463, 0.51083136],
            [0.3042251 , 0.77967715],
            [0.20216525, 0.03781104],
            [0.68387973, 0.9263613 ],
            [0.47634053, 0.7418159 ]], dtype=float32)
      bounded_model.b.value = Array([0., 0.], dtype=float32)
      bounded_model = Linear(
          # attributes
          din = 5
          dout = 2
      )


NNX is friendly for beginners
-----------------------------

* Example of training in eager mode

```python
import numpy as np

X = np.random.uniform(size=(1000, 1))
Y = 0.8 * X + 0.4 + np.random.normal(scale=0.1, size=(1000, 1))

model = Linear(1, 1, ctx=nnx.context(0))

for step in range(500):
  idx = np.random.randint(0, 1000, size=(32,))
  x, y = X[idx], Y[idx]

  def loss_fn(model: Linear):
    y_pred = model(x)
    return jnp.mean((y_pred - y) ** 2)

  loss, grads = nnx.value_and_grad(loss_fn, wrt=nnx.Param)(model)

  params = model.filter(nnx.Param)
  params = jax.tree_map(lambda p, g: p - 0.1 * g, params, grads)
  model.update_state(params)

  if step % 100 == 0:
    y_pred = model(X)
    loss = np.mean((y_pred - Y) ** 2)
    print(f"Step {step}: loss={loss:.4f}")

print(f"\n{model.w = }")
print(f"{model.b = }")
```


## NNX is friendly for advanced users
* Example of manual scan over layer 


```python
class Block(nnx.Module):

  def __init__(self, dim: int, *, ctx: nnx.Context):
    self.linear = nnx.Linear(dim, dim, ctx=ctx)
    self.bn = nnx.BatchNorm(dim, ctx=ctx)
    self.dropout = nnx.Dropout(0.5)

  def __call__(self, x: jax.Array, *, ctx: nnx.Context) -> jax.Array:
    x = self.linear(x)
    x = self.bn(x, ctx=ctx)
    x = self.dropout(x, ctx=ctx)
    x = jax.nn.gelu(x)
    return x
```

```python
from functools import partial


class ScanMLP(nnx.Module):

  def __init__(self, dim: int, *, n_layers: int, ctx: nnx.Context):
    self.n_layers = n_layers
    keys, ctxdef = ctx.partition()
    params_key = jax.random.split(keys["params"], n_layers)

    @partial(jax.vmap, out_axes=(0, None, None))
    def create_block(params_key):
      ctx = ctxdef.merge({"params": params_key})
      (params, batch_stats), moduledef = Block(dim, ctx=ctx).partition(
          nnx.Param, nnx.BatchStat
      )
      return params, batch_stats, moduledef

    params, batch_stats, moduledef = create_block(params_key)
    self.layers = moduledef.merge(params, batch_stats)

  def __call__(self, x: jax.Array, *, ctx: nnx.Context):
    keys, ctxdef = ctx.partition()
    dropout_key = jax.random.split(keys["dropout"], self.n_layers)
    (params, batch_stats), moduledef = self.layers.partition(
        nnx.Param, nnx.BatchStat
    )

    def scan_fn(
        carry: tuple[jax.Array, nnx.State], inputs: tuple[nnx.State, jax.Array]
    ):
      (x, batch_stats), (params, dropout_key) = carry, inputs
      module = moduledef.merge(params, batch_stats)
      x = module(x, ctx=ctxdef.merge({"dropout": dropout_key}))
      params, _ = module.partition(nnx.Param)
      return (x, batch_stats), params

    (x, batch_stats), params = jax.lax.scan(
        scan_fn, (x, batch_stats), (params, dropout_key)
    )
    self.layers.update_state((params, batch_stats))
    return x
```


## Parameter surgery is intuitive
* Simple parameter surgery example

## 



```python
def load_pretrained_model():
  ctx = nnx.context(0)
  model = nnx.Sequence([
      lambda x: x.reshape((x.shape[0], -1)),
      nnx.Linear(784, 1024, ctx=ctx),
  ])
  return model
```

```python
class Classifier(nnx.Module):

  def __init__(self, backbone: nnx.Sequence, *, ctx: nnx.Context):
    self.backbone = backbone
    self.head = nnx.Linear(1024, 10, ctx=ctx)

  def __call__(self, x: jax.Array):
    x = self.backbone(x)
    x = self.head(x)
    return x


pretrained_model = load_pretrained_model()
model = Classifier(pretrained_model, ctx=nnx.context(0))
y = model(jnp.ones((1, 28, 28)))

print("y.shape =", y.shape)
print("state =", jax.tree_map(jnp.shape, model.get_state()))
```

## Hacking Modules is possible
* You can change the layers of an existing Module just by replacing the fields

```python

```

## What about Pytree-based libraries?
* Equinox, Treex, [PytreeClass](https://github.com/ASEM000/PyTreeClass)
* Shared mutable reference not allow


## Road Ahead
