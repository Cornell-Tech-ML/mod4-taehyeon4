from typing import Tuple

from . import operators
from .autodiff import Context
from .fast_ops import FastOps
from .tensor import Tensor
from .tensor_functions import Function, rand, tensor

max_reduce = FastOps.reduce(operators.max, -1e9)

# List of functions in this file:
# - avgpool2d: Tiled average pooling 2D
# - argmax: Compute the argmax as a 1-hot tensor
# - Max: New Function for max operator
# - max: Apply max reduction
# - softmax: Compute the softmax as a tensor
# - logsoftmax: Compute the log of the softmax as a tensor - See https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations
# - maxpool2d: Tiled max pooling 2D
# - dropout: Dropout positions based on random noise, include an argument to turn off


def tile(input: Tensor, kernel: Tuple[int, int]) -> Tuple[Tensor, int, int]:
    """Reshape an image tensor for 2D pooling

    Args:
    ----
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
    -------
        Tensor of size batch x channel x new_height x new_width x (kernel_height * kernel_width) as well as the new_height and new_width value.

    """
    batch, channel, height, width = input.shape
    kh, kw = kernel
    assert height % kh == 0
    assert width % kw == 0

    new_height = height // kh
    new_width = width // kw

    # Create a view that splits the height and width dimensions into tiles
    tiled = input.contiguous().view(batch, channel, new_height, kh, new_width, kw)

    # Make the tensor contiguous after permute
    result = (
        tiled.permute(0, 1, 2, 4, 3, 5)
        .contiguous()
        .view(batch, channel, new_height, new_width, kh * kw)
    )

    return result, new_height, new_width


def avgpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Tiled average pooling 2D.

    Args:
    ----
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
    -------
        Pooled tensor batch x channel x new_height x new_width

    """
    tiled, new_height, new_width = tile(input, kernel)
    batch, channel = input.shape[:2]
    return tiled.mean(dim=4).view(batch, channel, new_height, new_width)


def argmax(input: Tensor, dim: int) -> Tensor:
    """Compute the argmax as a 1-hot tensor.

    Args:
        input: input tensor
        dim: dimension to reduce over

    Returns:
    -------
        1-hot tensor with same shape as input where argmax is 1, else 0

    """
    return input == max_reduce(input, dim)


class Max(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, dim: Tensor) -> Tensor:
        """Forward pass for Max reduction.

        Args:
        ----
            ctx: Context for backprop
            input: Input tensor
            dim: Dimension to reduce over

        Returns:
        -------
            Tensor with max values along dim

        """
        dim_val = int(dim.item())
        ctx.save_for_backward(input, dim_val)
        return max_reduce(input, dim_val)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        """Backward pass for Max.

        Args:
            ctx: Context from forward pass
            grad_output: Gradient of the loss wrt output

        Returns:
        -------
            Tuple of:
            - Gradient wrt input
            - Gradient wrt dim (always 0.0)

        """
        input, dim = ctx.saved_values
        return grad_output * argmax(input, dim), 0.0


def max(input: Tensor, dim: int) -> Tensor:
    """Apply max reduction along dimension."""
    return Max.apply(input, input._ensure_tensor(dim))


def softmax(input: Tensor, dim: int) -> Tensor:
    """Compute softmax along dimension.

    Args:
        input: Input tensor
        dim: Dimension to apply softmax over

    Returns:
    -------
        Tensor with softmax applied along dim

    """
    input_exp = input.exp()
    return input_exp / input_exp.sum(dim)


def logsoftmax(input: Tensor, dim: int) -> Tensor:
    """Compute log softmax along dimension.

    Uses the LogSumExp trick for numerical stability.

    Args:
        input: Input tensor
        dim: Dimension to apply log softmax over

    Returns:
    -------
        Tensor with log softmax applied along dim

    """
    x_max = max(input, dim)
    shifted_exp = (input - x_max).exp()
    return input - (shifted_exp.sum(dim).log() + x_max)


def maxpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Tiled max pooling 2D.

    Args:
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
    -------
        Pooled tensor batch x channel x new_height x new_width

    """
    batch, channel = input.shape[:2]
    tiled, new_height, new_width = tile(input, kernel)
    return max(tiled, dim=4).view(batch, channel, new_height, new_width)


def dropout(input: Tensor, p: float, ignore: bool = False) -> Tensor:
    """Apply dropout with probability p.

    Args:
        input: Input tensor
        p: Dropout probability
        ignore: If True, return input unchanged

    Returns:
    -------
        Tensor with dropout applied

    """
    if ignore:
        return input
    return input * (rand(input.shape) > p)
