from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, Iterable, Tuple, Protocol


# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
    ----
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
    -------
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$

    """
    vals1 = list(vals)
    vals2 = list(vals)
    vals1[arg] += epsilon
    vals2[arg] -= epsilon
    return (f(*vals1) - f(*vals2)) / (2 * epsilon)


variable_count = 1


class Variable(Protocol):
    """A protocol for the Scalar class. For more details, refer to the
    Scalar class in the scalar.py file.

    Note: The method docstrings are omitted here as they are already
    well-documented in the Scalar class, and repeating them could lead to
    redundancy or potential inconsistencies.
    """

    def accumulate_derivative(self, x: Any) -> None: ...  # noqa: D102

    @property
    def unique_id(self) -> int: ...  # noqa: D102

    def is_leaf(self) -> bool: ...  # noqa: D102

    def is_constant(self) -> bool: ...  # noqa: D102

    @property
    def parents(self) -> Iterable["Variable"]: ...  # noqa: D102

    def chain_rule(self, d_output: Any) -> Iterable[Tuple[Variable, Any]]: ...  # noqa: D102


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """Computes the topological order of the computation graph.

    Args:
    ----
        variable: The right-most variable

    Returns:
    -------
        Non-constant Variables in topological order starting from the right.

    """
    order: deque[Variable] = deque()
    seen = set()

    def visit(var: Variable) -> None:
        if var in seen or var.is_constant():
            return
        if not var.is_leaf():
            for parent in var.parents:
                if not parent.is_constant():
                    visit(parent)
        seen.add(var)
        order.appendleft(var)

    visit(variable)
    return order


def backpropagate(variable: Variable, deriv: Any) -> None:
    """Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    There is no return value. The results should be written to the leaf nodes
    through `accumulate_derivative`.

    Args:
    ----
        variable: The right-most variable
        deriv: Its derivative that we want to propagate backward to the leaves.

    """
    derivatives = defaultdict(int)
    derivatives[variable] = deriv

    for node in topological_sort(variable):
        if not node.is_leaf():
            for parent, d in node.chain_rule(derivatives[node]):
                if parent.is_constant():
                    continue
                derivatives[parent] += d
        else:
            node.accumulate_derivative(derivatives[node])


@dataclass
class Context:
    """Context class is used by `Function` to store information during the forward pass."""

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        """Store the given `values` if they need to be used during backpropagation."""
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        """Returns the saved values."""
        return self.saved_values
