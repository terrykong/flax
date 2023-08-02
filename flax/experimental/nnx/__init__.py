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

from .nnx.containers import (
    BatchStat as BatchStat,
    Cache as Cache,
    Container as Container,
    ContainerMetadata as ContainerMetadata,
    Intermediate as Intermediate,
    Node as Node,
    Param as Param,
    Static as Static,
    Variable as Variable,
    with_metadata as with_metadata,
)
from .nnx.contextlib import Context as Context, context as context
from .nnx.dataclasses import (
    dataclass as dataclass,
    field as field,
    node_field as node_field,
    param_field as param_field,
    static_field as static_field,
    var_field as var_field,
)
from .nnx.errors import TraceContextError as TraceContextError
from .nnx.helpers import (
    Dict as Dict,
    Sequence as Sequence,
    TrainState as TrainState,
)
from .nnx.module import (
    Module as Module,
    ModuleDef as ModuleDef,
    Pure as Pure,
    PureModule as PureModule,
)
from .nnx.nn import initializers as initializers
from .nnx.nn.activations import (
    celu as celu,
    elu as elu,
    gelu as gelu,
    glu as glu,
    hard_sigmoid as hard_sigmoid,
    hard_silu as hard_silu,
    hard_swish as hard_swish,
    hard_tanh as hard_tanh,
    leaky_relu as leaky_relu,
    log_sigmoid as log_sigmoid,
    log_softmax as log_softmax,
    logsumexp as logsumexp,
    normalize as normalize,
    one_hot as one_hot,
    relu as relu,
    relu6 as relu6,
    selu as selu,
    sigmoid as sigmoid,
    silu as silu,
    soft_sign as soft_sign,
    softmax as softmax,
    softplus as softplus,
    standardize as standardize,
    swish as swish,
    tanh as tanh,
)
from .nnx.nn.linear import (
    Conv as Conv,
    Embed as Embed,
    Linear as Linear,
)
from .nnx.nn.normalization import (
    BatchNorm as BatchNorm,
    LayerNorm as LayerNorm,
)
from .nnx.nn.stochastic import (
    Dropout as Dropout,
)
from .nnx.nodes import (
    is_node as is_node,
    register_node_type as register_node_type,
)
from .nnx.partitioning import (
    All as All,
    Not as Not,
    buffers as buffers,
)
from .nnx.pytreelib import (
    Pytree as Pytree,
    TreeNode as TreeNode,
)
from .nnx.spmd import (
    PARTITION_NAME as PARTITION_NAME,
    get_partition_spec as get_partition_spec,
    logical_axis_rules as logical_axis_rules,
    logical_to_mesh as logical_to_mesh,
    with_logical_constraint as with_logical_constraint,
    with_logical_partitioning as with_logical_partitioning,
)
from .nnx.state import (
    State as State,
)
from .nnx.transforms import (
    Remat as Remat,
    Scan as Scan,
    grad as grad,
    jit as jit,
    remat as remat,
    scan as scan,
)
from .nnx import (
    compatibility as compatibility,
)
