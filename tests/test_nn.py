import pytest
from hypothesis import given

import minitorch
from minitorch import Tensor

from .strategies import assert_close
from .tensor_strategies import tensors


@pytest.mark.task4_3
@given(tensors(shape=(1, 1, 4, 4)))
def test_avg(t: Tensor) -> None:
    out = minitorch.avgpool2d(t, (2, 2))
    assert_close(
        out[0, 0, 0, 0], sum([t[0, 0, i, j] for i in range(2) for j in range(2)]) / 4.0
    )

    out = minitorch.avgpool2d(t, (2, 1))
    assert_close(
        out[0, 0, 0, 0], sum([t[0, 0, i, j] for i in range(2) for j in range(1)]) / 2.0
    )

    out = minitorch.avgpool2d(t, (1, 2))
    assert_close(
        out[0, 0, 0, 0], sum([t[0, 0, i, j] for i in range(1) for j in range(2)]) / 2.0
    )
    minitorch.grad_check(lambda t: minitorch.avgpool2d(t, (2, 2)), t)


@pytest.mark.task4_4
@given(tensors(shape=(2, 3, 4)))
def test_max(t: Tensor) -> None:
    def test_max_reduction_along_dim(
        tensor: Tensor, dim: int, expected_shape: tuple
    ) -> None:
        out = minitorch.nn.max(tensor, dim)
        # Test shape
        assert out.shape == expected_shape
        # Test value
        assert_close(
            out[0, 0, 0],
            max(
                [
                    tensor[*([0] * dim + [i] + [0] * (2 - dim))]
                    for i in range(tensor.shape[dim])
                ]
            ),
        )

    # Test max reduction along each dimension
    test_cases = [
        (0, (1, 3, 4)),  # First dimension
        (1, (2, 1, 4)),  # Second dimension
        (2, (2, 3, 1)),  # Third dimension
    ]

    for dim, expected_shape in test_cases:
        test_max_reduction_along_dim(t, dim, expected_shape)

    # Gradient checks
    def test_gradients() -> None:
        # Automatic gradient check (with noise to avoid non-differentiable points)
        noisy_tensor = t + (minitorch.rand(t.shape) * 1e-4)
        minitorch.grad_check(lambda x: minitorch.nn.max(x, 0), noisy_tensor)

        # Manual gradient check
        t.requires_grad_(True)
        out = minitorch.nn.max(t, 2)  # max along last dimension
        out.sum().backward()

        assert t.grad is not None

        # Check gradient values
        for i in range(2):
            for j in range(3):
                max_val = out[i, j, 0]
                for k in range(4):
                    expected_grad = 1.0 if t[i, j, k] == max_val else 0.0
                    assert_close(t.grad[i, j, k], expected_grad)

    test_gradients()


@pytest.mark.task4_4
@given(tensors(shape=(1, 1, 4, 4)))
def test_max_pool(t: Tensor) -> None:
    out = minitorch.maxpool2d(t, (2, 2))
    print(out)
    print(t)
    assert_close(
        out[0, 0, 0, 0], max([t[0, 0, i, j] for i in range(2) for j in range(2)])
    )

    out = minitorch.maxpool2d(t, (2, 1))
    assert_close(
        out[0, 0, 0, 0], max([t[0, 0, i, j] for i in range(2) for j in range(1)])
    )

    out = minitorch.maxpool2d(t, (1, 2))
    assert_close(
        out[0, 0, 0, 0], max([t[0, 0, i, j] for i in range(1) for j in range(2)])
    )


@pytest.mark.task4_4
@given(tensors())
def test_drop(t: Tensor) -> None:
    q = minitorch.dropout(t, 0.0)
    idx = q._tensor.sample()
    assert q[idx] == t[idx]
    q = minitorch.dropout(t, 1.0)
    assert q[q._tensor.sample()] == 0.0
    q = minitorch.dropout(t, 1.0, ignore=True)
    idx = q._tensor.sample()
    assert q[idx] == t[idx]


@pytest.mark.task4_4
@given(tensors(shape=(1, 1, 4, 4)))
def test_softmax(t: Tensor) -> None:
    q = minitorch.softmax(t, 3)
    x = q.sum(dim=3)
    assert_close(x[0, 0, 0, 0], 1.0)

    q = minitorch.softmax(t, 1)
    x = q.sum(dim=1)
    assert_close(x[0, 0, 0, 0], 1.0)

    minitorch.grad_check(lambda a: minitorch.softmax(a, dim=2), t)


@pytest.mark.task4_4
@given(tensors(shape=(1, 1, 4, 4)))
def test_log_softmax(t: Tensor) -> None:
    q = minitorch.softmax(t, 3)
    q2 = minitorch.logsoftmax(t, 3).exp()
    for i in q._tensor.indices():
        assert_close(q[i], q2[i])

    minitorch.grad_check(lambda a: minitorch.logsoftmax(a, dim=2), t)
