from typing import Callable, Optional, List, Any

import torch
import torch.distributed as dist
import torch.distributed._functional_collectives as fc
import torch.distributed.distributed_c10d as c10d
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.parameter import Parameter

from fairscale.nn.model_parallel.utils import divide_and_check_no_remainder, split_tensor_along_last_dim

import os

USE_CUDA = os.environ.get('USE_CUDA', False)

# Some how xla init will slow down the CUDA speed.
if not USE_CUDA:
    import torch_xla.core.xla_model as xm

TAG = None
RANKSET = None
GROUP_SIZE = None


def set_g_group():
    global TAG
    global RANKSET
    global GROUP_SIZE

    assert USE_CUDA, "This hack is only for PyTorch non-XLA CUDA paths, i.e., eager and inductor."
    TAG, RANKSET, GROUP_SIZE = fc._expand_group(c10d._get_default_group())


def get_model_parallel_rank():
    if USE_CUDA:
        return dist.get_rank()
    return xm.get_ordinal()


def get_model_parallel_world_size():
    if USE_CUDA:
        return dist.get_world_size()
    return xm.xrt_world_size()


def get_model_parallel_group():
    return None


class _CopyToModelParallelRegion(torch.autograd.Function):
    """Pass the input to the model parallel region."""

    @staticmethod
    def forward(ctx, input_, groups, world_size, rank):  # type: ignore
        ctx.groups, ctx.world_size, ctx.rank = groups, world_size, rank
        return input_

    @staticmethod
    def backward(ctx, grad_output):  # type: ignore
        groups, world_size, rank = ctx.groups, ctx.world_size, ctx.rank
        return my_reduce(grad_output, groups, world_size, rank)


class _ReduceFromModelParallelRegion(torch.autograd.Function):
    """All-redcue the input from the model parallel region."""

    @staticmethod
    def forward(ctx, input_, groups, world_size, rank):  # type: ignore
        return my_reduce(input_, groups, world_size, rank)

    @staticmethod
    def backward(ctx, grad_output):  # type: ignore
        return grad_output


class _ScatterToModelParallelRegion(torch.autograd.Function):
    """Split the input and keep only the corresponding chuck to the rank."""

    @staticmethod
    def forward(ctx, input_, groups, world_size, rank):  # type: ignore
        ctx.groups, ctx.world_size, ctx.rank = groups, world_size, rank
        return my_split(input_, groups, world_size, rank)

    @staticmethod
    def backward(ctx, grad_output):  # type: ignore
        groups, world_size, rank = ctx.groups, ctx.world_size, ctx.rank
        return my_gather(grad_output, groups, world_size, rank)


class _GatherFromModelParallelRegion(torch.autograd.Function):
    """Gather the input from model parallel region and concatinate."""

    @staticmethod
    def forward(ctx, input_, groups, world_size, rank):  # type: ignore
        ctx.groups, ctx.world_size, ctx.rank = groups, world_size, rank
        return my_gather(input_, groups, world_size, rank)

    @staticmethod
    def backward(ctx, grad_output):  # type: ignore
        groups, world_size, rank = ctx.groups, ctx.world_size, ctx.rank
        return my_split(grad_output, groups, world_size, rank)


# -----------------
# Helper functions.
# -----------------


def copy_to_model_parallel_region(input_: torch.Tensor, groups, world_size,
                                  rank) -> torch.Tensor:
    return _CopyToModelParallelRegion.apply(input_, groups, world_size, rank)


def reduce_from_model_parallel_region(input_: torch.Tensor, groups, world_size,
                                      rank) -> torch.Tensor:
    return _ReduceFromModelParallelRegion.apply(input_, groups, world_size,
                                                rank)


def scatter_to_model_parallel_region(input_: torch.Tensor, groups, world_size,
                                     rank) -> torch.Tensor:
    return _ScatterToModelParallelRegion.apply(input_, groups, world_size,
                                               rank)


def gather_from_model_parallel_region(input_: torch.Tensor, groups, world_size,
                                      rank) -> torch.Tensor:
    return _GatherFromModelParallelRegion.apply(input_, groups, world_size,
                                                rank)


# Below copied from fairscale/nn/model_parallel/layers.py


def my_reduce(input_: torch.Tensor, groups, world_size, rank) -> torch.Tensor:
    """All-reduce the the input tensor across model parallel group."""
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_

    # All-reduce.
    if USE_CUDA:
        input_ = torch.ops.c10d_functional.all_reduce(input_, "sum", TAG,
                                                      RANKSET, GROUP_SIZE)
    else:
        input_ = xm.all_reduce(xm.REDUCE_SUM, input_, groups=groups)

    return input_


def my_split(input_: torch.Tensor, groups, world_size, rank) -> torch.Tensor:
    """Split the tensor along its last dimension and keep the
    corresponding slice."""
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_

    # Split along last dimension.
    input_list = split_tensor_along_last_dim(input_, world_size)

    # Note: torch.split does not create contiguous tensors by default.
    output = input_list[rank].contiguous()

    return output


def my_gather(input_: torch.Tensor, groups, world_size, rank) -> torch.Tensor:
    """Gather tensors and concatinate along the last dimension."""
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_

    if USE_CUDA:
        last_dim = input_.dim() - 1

        # Using all_reduce to achieve all_gather as torch.ops.c10d_functional.all_gather_into_tensor
        # is buggy in 16 bits.
        size = input_.size(last_dim)
        padding = [0] * (2 * input_.dim())
        ordinal = rank
        left, right = ordinal, world_size - 1 - ordinal
        idx = input_.dim() - 1 - last_dim
        padding[2 * idx] = left * size
        padding[2 * idx + 1] = right * size
        output = torch.ops.c10d_functional.all_reduce(F.pad(input_,
                                                            padding), "sum",
                                                      TAG, RANKSET, GROUP_SIZE)
    else:
        output = xm.all_gather(input_, dim=-1, groups=groups)

    return output


def _initialize_affine_weight(
    weight: torch.Tensor,
    out_features: int,
    in_features: int,
    per_partition_size: int,
    partition_dim: int,
    init_method: Callable[[torch.Tensor], torch.Tensor],
    world_size: int,
    rank: int,
    stride: int = 1,
    return_master_weight: bool = False,
) -> Optional[torch.Tensor]:
    """Initialize affine weight for model parallel.

    Build the master weight on all processes and scatter
    the relevant chunk."""

    # If we only use 1 process for model parallelism, bypass scatter.
    if world_size == 1:
        init_method(weight)
        if return_master_weight:
            return weight
        return None

    # Initialize master weight
    master_weight = torch.empty(out_features,
                                in_features,
                                dtype=weight.dtype,
                                requires_grad=False)
    init_method(master_weight)

    # Split and copy
    per_partition_per_stride_size = divide_and_check_no_remainder(
        per_partition_size, stride)
    weight_list = torch.split(master_weight,
                              per_partition_per_stride_size,
                              dim=partition_dim)
    my_weight_list = weight_list[rank::world_size]

    with torch.no_grad():
        torch.cat(my_weight_list, dim=partition_dim, out=weight)
    if return_master_weight:
        return master_weight
    return None


class ParallelEmbedding(torch.nn.Module):
    """Embedding parallelized in the embedding dimension.

    This is mainly adapted from torch.nn.Embedding and all the default
    values are kept.
    Arguments:
        num_embeddings: vocabulary size.
        embedding_dim: size of hidden state.
        init_method: method to initialize weights.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
        max_norm: Optional[float] = None,
        norm_type: float = 2.0,
        scale_grad_by_freq: bool = False,
        sparse: bool = False,
        init_method: Callable[[torch.Tensor],
                              torch.Tensor] = init.xavier_normal_,
        keep_master_weight_for_test: bool = False,
        world_size: Optional[int] = None,
        rank: Optional[int] = None,
        groups: Optional[List] = None,
    ) -> None:
        super(ParallelEmbedding, self).__init__()

        if world_size is None:
            self.groups = get_model_parallel_group()
            self.world_size = get_model_parallel_world_size()
            self.rank = get_model_parallel_rank()
        else:
            self.groups = groups
            self.world_size = world_size
            self.rank = rank

        # Keep the input dimensions.
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.max_norm = max_norm
        self.norm_type = scale_grad_by_freq
        self.scale_grad_by_freq = scale_grad_by_freq
        self.sparse = sparse
        self._weight = None
        # Divide the weight matrix along the embedding dimension.
        self.embedding_dim_per_partition = divide_and_check_no_remainder(
            self.embedding_dim, self.world_size)

        # Allocate weights.
        self.weight = Parameter(
            torch.Tensor(self.num_embeddings,
                         self.embedding_dim_per_partition))
        # And initialize.
        _initialize_affine_weight(
            self.weight,
            self.num_embeddings,
            self.embedding_dim,
            self.embedding_dim_per_partition,
            1,
            init_method,
            self.world_size,
            self.rank,
            stride=1,
            return_master_weight=False,
        )

    def forward(self, input_: torch.Tensor) -> torch.Tensor:  # type: ignore
        input_parallel = copy_to_model_parallel_region(input_, self.groups,
                                                       self.world_size,
                                                       self.rank)
        output_parallel = F.embedding(
            input_parallel,
            self.weight,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )
        output = gather_from_model_parallel_region(output_parallel,
                                                   self.groups,
                                                   self.world_size, self.rank)
        return output


class ColumnParallelLinear(torch.nn.Module):
    """Linear layer with column parallelism.

    The linear layer is defined as Y = XA + b. A is parallelized along
    its second dimension as A = [A_1, ..., A_p].

    Arguments:
        in_features: first dimension of matrix A.
        out_features: second dimension of matrix A.
        bias: If true, add bias
        gather_output: If true, call all-gether on output and make Y avaiable
                       to all GPUs, otherwise, every GPU will have its output
                       which is Y_i = XA_i
        init_method: method to initialize weights. Note that bias is always set
                     to zero.
        stride: For the strided linear layers.
        keep_master_weight_for_test: This was added for testing and should be
                                     set to False. It returns the master weights
                                     used for initialization.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        gather_output: bool = True,
        init_method: Callable[[torch.Tensor],
                              torch.Tensor] = init.xavier_normal_,
        stride: int = 1,
        keep_master_weight_for_test: bool = False,
        world_size: Optional[int] = None,
        rank: Optional[int] = None,
        groups: Optional[List] = None,
        quant: bool = False,
    ) -> None:
        super(ColumnParallelLinear, self).__init__()

        if world_size is None:
            self.groups = get_model_parallel_group()
            self.world_size = get_model_parallel_world_size()
            self.rank = get_model_parallel_rank()
        else:
            self.groups = groups
            self.world_size = world_size
            self.rank = rank

        # Keep input parameters
        self.in_features = in_features
        self.out_features = out_features
        self.gather_output = gather_output
        self.quant = quant
        # Divide the weight matrix along the last dimension.
        self.output_size_per_partition = divide_and_check_no_remainder(
            out_features, self.world_size)

        # Parameters.
        # Note: torch.nn.functional.linear performs XA^T + b and as a result
        # we allocate the transpose.
        if quant:
            self.weight = Parameter(torch.empty(
                (self.output_size_per_partition, self.in_features),
                dtype=torch.int8),
                                    requires_grad=False)
            self.weight_scaler = Parameter(torch.zeros(1), requires_grad=False)
        else:
            self.weight = Parameter(
                torch.Tensor(self.output_size_per_partition, self.in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(self.output_size_per_partition))
            # Always initialize bias to zero.
            with torch.no_grad():
                self.bias.zero_()
        else:
            self.register_parameter("bias", None)

        # Initialize weight.
        self.master_weight = _initialize_affine_weight(
            self.weight,
            self.out_features,
            self.in_features,
            self.output_size_per_partition,
            0,
            init_method,
            self.world_size,
            self.rank,
            stride=stride,
            return_master_weight=keep_master_weight_for_test,
        )

    def get_master_weight(self) -> torch.Tensor:
        return gather_from_model_parallel_region(
            self.weight.data.transpose(0, 1), self.groups, self.world_size,
            self.rank).transpose_(0, 1)

    def forward(self, input_: torch.Tensor) -> torch.Tensor:  # type: ignore
        # Set up backprop all-reduce.
        input_parallel = copy_to_model_parallel_region(input_, self.groups,
                                                       self.world_size,
                                                       self.rank)
        # Matrix multiply.
        if self.quant:
            output_parallel = F.linear(input_parallel, self.weight, self.bias)
            output_parallel = output_parallel * self.weight_scaler
        else:
            output_parallel = F.linear(input_parallel, self.weight, self.bias)
        if self.gather_output:
            # All-gather across the partitions.
            output = gather_from_model_parallel_region(output_parallel,
                                                       self.groups,
                                                       self.world_size,
                                                       self.rank)
        else:
            output = output_parallel
        return output


class RowParallelLinear(torch.nn.Module):
    """Linear layer with row parallelism.

    The linear layer is defined as Y = XA + b. A is parallelized along
    its first dimension and X along its second dimension as:
               -   -
              | A_1 |
              | .   |
          A = | .   |        X = [X_1, ..., X_p]
              | .   |
              | A_p |
               -   -
    Arguments:
        in_features: first dimension of matrix A.
        out_features: second dimension of matrix A.
        bias: If true, add bias. Note that bias is not parallelized.
        input_is_parallel: If true, we assume that the input is already
                           split across the GPUs and we do not split
                           again.
        init_method: method to initialize weights. Note that bias is always set
                     to zero.
        stride: For the strided linear layers.
        keep_master_weight_for_test: This was added for testing and should be
                                     set to False. It returns the master weights
                                     used for initialization.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        input_is_parallel: bool = False,
        init_method: Callable[[torch.Tensor],
                              torch.Tensor] = init.xavier_normal_,
        stride: int = 1,
        keep_master_weight_for_test: bool = False,
        world_size: Optional[int] = None,
        rank: Optional[int] = None,
        groups: Optional[List] = None,
        quant: bool = False,
    ):
        super(RowParallelLinear, self).__init__()

        if world_size is None:
            self.groups = get_model_parallel_group()
            self.world_size = get_model_parallel_world_size()
            self.rank = get_model_parallel_rank()
        else:
            self.groups = groups
            self.world_size = world_size
            self.rank = rank

        # Keep input parameters
        self.in_features = in_features
        self.out_features = out_features
        self.input_is_parallel = input_is_parallel
        self.quant = quant
        # Divide the weight matrix along the last dimension.
        self.input_size_per_partition = divide_and_check_no_remainder(
            in_features, self.world_size)

        # Parameters.
        # Note: torch.nn.functional.linear performs XA^T + b and as a result
        # we allocate the transpose.
        if quant:
            self.weight = Parameter(torch.empty(
                (self.out_features, self.input_size_per_partition),
                dtype=torch.int8),
                                    requires_grad=False)
            self.weight_scaler = Parameter(torch.zeros(1), requires_grad=False)
        else:
            self.weight = Parameter(
                torch.Tensor(self.out_features, self.input_size_per_partition))
        if bias:
            self.bias = Parameter(torch.Tensor(self.out_features))
            # Always initialize bias to zero.
            with torch.no_grad():
                self.bias.zero_()
        else:
            self.register_parameter("bias", None)

        # Initialize weight.
        self.master_weight = _initialize_affine_weight(
            self.weight,
            self.out_features,
            self.in_features,
            self.input_size_per_partition,
            1,
            init_method,
            self.world_size,
            self.rank,
            stride=stride,
            return_master_weight=keep_master_weight_for_test,
        )

    def get_master_weight(self) -> torch.Tensor:
        return gather_from_model_parallel_region(self.weight.data, self.groups,
                                                 self.world_size, self.rank)

    def forward(self, input_: torch.Tensor) -> torch.Tensor:  # type:ignore
        # Set up backprop all-reduce.
        if self.input_is_parallel:
            input_parallel = input_
        else:
            input_parallel = scatter_to_model_parallel_region(
                input_, self.groups, self.world_size, self.rank)
        # Matrix multiply.
        if self.quant:
            output_parallel = F.linear(input_parallel, self.weight, self.bias)
            output_parallel = output_parallel * self.weight_scaler
        else:
            output_parallel = F.linear(input_parallel, self.weight)
        # All-reduce across all the partitions.
        output_ = reduce_from_model_parallel_region(output_parallel,
                                                    self.groups,
                                                    self.world_size, self.rank)
        if self.bias is not None:
            output = output_ + self.bias
        else:
            output = output_
        return output
