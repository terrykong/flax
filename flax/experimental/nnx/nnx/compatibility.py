import dataclasses
from typing import Any
import jax
import typing as tp
from flax.experimental.nnx.nnx.contextlib import Context
from flax.experimental.nnx.nnx import helpers

from flax.experimental.nnx.nnx.module import Module, ModuleDef

from flax.experimental.nnx.nnx.state import State
from flax.experimental.nnx.nnx import containers
from flax import linen

M = tp.TypeVar('M', bound=Module)


@dataclasses.dataclass
class Functional(tp.Generic[M]):
  module_type: tp.Type[M]
  moduledef: tp.Optional[ModuleDef[M]]
  args: tuple[tp.Any, ...]
  kwargs: dict[str, tp.Any]

  def init(self, *, ctx: tp.Optional[Context] = None) -> State:
    kwargs = {}
    if ctx is not None:
      kwargs['ctx'] = ctx
    module = self.module_type(*self.args, **self.kwargs, **kwargs)
    state, moduledef = module.partition()
    self.moduledef = moduledef
    return state

  def apply(self, *states: tp.Any):
    assert self.moduledef is not None
    return self.moduledef.apply(*states)


def functional(cls: tp.Type[M]) -> tp.Callable[..., Functional[M]]:
  def _functional_constructor(*args: tp.Any, **kwargs: tp.Any) -> Functional[M]:
    return Functional(cls, None, args, kwargs)

  return _functional_constructor


class LinenWrapper(Module):

  def __init__(
      self,
      module: linen.Module,
      *args: tp.Any,
      ctx: tp.Optional[Context] = None,
      **kwargs: tp.Any
  ):
    self.module = module

    rngs = (
        {name: stream.key for name, stream in ctx._rngs.items()} if ctx else {}
    )
    variables = module.init(rngs, *args, **kwargs)

    self.states = helpers.Dict(
        (collection, containers.variable_type(collection)(value))
        for collection, value in variables.items()
    )

  def __call__(
      self, *args: Any, ctx: tp.Optional[Context] = None, **kwargs: Any
  ) -> Any:
    rngs = (
        {name: stream.key for name, stream in ctx._rngs.items()} if ctx else {}
    )
    variables = {collection: value for collection, value in self.states.items()}
    return self.module.apply(variables, *args, rngs=rngs, **kwargs)
