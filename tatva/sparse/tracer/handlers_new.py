from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

import numpy as np
import scipy.sparse as sps
from jax.extend.core import JaxprEqn
from numpy.typing import NDArray

from tatva.sparse.tracer.common import _get_shape
from tatva.sparse.tracer.state import CouplingAccumulator, TraceState


class PrimitiveHandler(ABC):
    @abstractmethod
    def propagate_deps(
        self,
        eqn: JaxprEqn,
        state: TraceState,
        acc: CouplingAccumulator,
        trial_test_split: int | None,
    ) -> None:
        """Forward dependency set propagation & coupling accumulation."""
        raise NotImplementedError

    def eval_concrete(
        self, in_vals: list[NDArray | None], params: dict[str, Any]
    ) -> NDArray | None:
        """Evaluate concrete numpy values for routing indices (returns None if unknown)."""
        return None

    def introduces_nonlinearity(self, eqn: JaxprEqn, invar_active: list[bool]) -> bool:
        """Determine if this primitive introduces non-affine behavior on active inputs."""
        return False

    def propagate_tags(self, eqn: JaxprEqn, invar_tags: list[int]) -> int:
        """Forward tag propagation (default: bitwise OR of all input tags)."""
        mask = 0
        for t in invar_tags:
            mask |= t
        return mask

    def get_index_invar_indices(self, eqn: JaxprEqn) -> list[int]:
        """Return invar indices that are used as index arrays (e.g. gather/scatter index)."""
        return []


class ElementwiseUnary(PrimitiveHandler):
    """Handler for elementwise unary operations (1 input -> 1 output).

    Covers both:
    - linear/affine unary ops (is_nonlinear=False)
    - nonlinear unary ops (is_nonlinear=True)
    """

    def __init__(
        self, is_nonlinear: bool, eval_fn: Callable[..., NDArray] | None = None
    ):
        self.is_nonlinear = is_nonlinear
        self.eval_fn = eval_fn

    def propagate_deps(
        self,
        eqn: JaxprEqn,
        state: TraceState,
        acc: CouplingAccumulator,
        trial_test_split: int | None,
    ) -> None:
        in_d = state.get(eqn.invars[0])
        oshp = _get_shape(eqn.outvars[0]) if eqn.outvars else ()

        # adjust shape if operation changes tensor layout (e.g. rehspae)
        if in_d.shape != oshp and int(np.prod(in_d.shape)) == int(np.prod(oshp)):
            dep_out = in_d.reshape(oshp)
        else:
            dep_out = in_d

        state.set(eqn.outvars[0], dep_out)

        # record second-order couplings ONLY if the operation is nonlinear
        if self.is_nonlinear:
            dep_out.record_couplings(acc, trial_test_split)

    def introduces_nonlinearity(self, eqn: JaxprEqn, invar_active: list[bool]) -> bool:
        # a unary op introduces nonlinearity if it is inherently nonlinear AND its input
        # is active
        return self.is_nonlinear and bool(invar_active) and invar_active[0]

    def eval_concrete(
        self, in_vals: list[NDArray | None], params: dict[str, Any]
    ) -> NDArray | None:
        if in_vals[0] is None:
            return None
        if self.eval_fn is not None:
            try:
                return self.eval_fn(in_vals[0], params)
            except TypeError:
                return self.eval_fn(in_vals[0])
        return None


class IntegerPowHandler(PrimitiveHandler):
    """Handler for integer power operations (x ** n)."""

    def propagate_deps(
        self,
        eqn: JaxprEqn,
        state: TraceState,
        acc: CouplingAccumulator,
        trial_test_split: int | None,
    ) -> None:
        in_d = state.get(eqn.invars[0])
        state.set(eqn.outvars[0], in_d.copy())
        if self.introduces_nonlinearity(eqn, [in_d.dep.nnz > 0]):
            in_d.record_couplings(acc, trial_test_split)

    def introduces_nonlinearity(self, eqn: JaxprEqn, invar_active: list[bool]) -> bool:
        y = eqn.params.get("y", 0)
        return (y >= 2 or y <= -1) and bool(invar_active) and invar_active[0]

    def eval_concrete(
        self, in_vals: list[NDArray | None], params: dict[str, Any]
    ) -> NDArray | None:
        if in_vals[0] is not None:
            return in_vals[0] ** params["y"]
        return None


class Broadcast(PrimitiveHandler):
    """Handler for broadcast operations."""

    def propagate_deps(
        self,
        eqn: JaxprEqn,
        state: TraceState,
        acc: CouplingAccumulator,
        trial_test_split: int | None,
    ) -> None:
        in_d = state.get(eqn.invars[0])
        oshp = _get_shape(eqn.outvars[0]) if eqn.outvars else ()

        state.set(eqn.outvars[0], in_d.broadcast_to(oshp))

    def introduces_nonlinearity(self, eqn: JaxprEqn, invar_active: list[bool]) -> bool:
        return False

    def eval_concrete(
        self, in_vals: list[NDArray | None], params: dict[str, Any]
    ) -> NDArray | None:
        if in_vals[0] is None:
            return None

        x = np.asarray(in_vals[0])
        shape = params["shape"]
        bdims = params["broadcast_dimensions"]

        newshape = [1] * len(shape)
        for i, b in enumerate(bdims):
            newshape[b] = x.shape[i] if x.ndim > 0 else 1

        return np.broadcast_to(x.reshape(newshape), shape).copy()


class SliceHandler(PrimitiveHandler):
    """Handler for static `slice` primitive."""

    def propagate_deps(
        self,
        eqn: JaxprEqn,
        state: TraceState,
        acc: CouplingAccumulator,
        trial_test_split: int | None,
    ) -> None:
        from tatva.sparse.tracer.state import SparseDepSet

        in_d = state.get(eqn.invars[0])
        par = eqn.params
        ss, ls = par["start_indices"], par["limit_indices"]
        st = par["strides"] or [1] * len(ss)
        oshp = _get_shape(eqn.outvars[0]) if eqn.outvars else ()

        if in_d.shape != oshp:
            # Map input row-major indices through Python slice
            idx = np.arange(int(np.prod(in_d.shape))).reshape(in_d.shape)
            sl = tuple(slice(s, l, t) for s, l, t in zip(ss, ls, st))
            sub_idx = idx[sl].ravel()
            dep_out = SparseDepSet(in_d.dep[sub_idx], oshp)
        else:
            dep_out = in_d

        state.set(eqn.outvars[0], dep_out)

    def introduces_nonlinearity(self, eqn: JaxprEqn, invar_active: list[bool]) -> bool:
        return False

    def eval_concrete(
        self, in_vals: list[np.ndarray | None], params: dict[str, Any]
    ) -> np.ndarray | None:
        if in_vals[0] is None:
            return None
        ss, ls = params["start_indices"], params["limit_indices"]
        st = params["strides"] or [1] * len(ss)
        sl = tuple(slice(s, l, t) for s, l, t in zip(ss, ls, st))
        return np.asarray(in_vals[0])[sl]


class ReverseHandler(PrimitiveHandler):
    """Handler for `rev` (reverse/flip axes) primitive."""

    def propagate_deps(
        self,
        eqn: JaxprEqn,
        state: TraceState,
        acc: CouplingAccumulator,
        trial_test_split: int | None,
    ) -> None:
        from tatva.sparse.tracer.state import SparseDepSet

        in_d = state.get(eqn.invars[0])
        dimensions = eqn.params["dimensions"]
        oshp = _get_shape(eqn.outvars[0]) if eqn.outvars else ()

        # Construct index array and flip specified dimensions
        idx = np.arange(int(np.prod(in_d.shape))).reshape(in_d.shape)
        sl = tuple(
            slice(None, None, -1) if i in dimensions else slice(None)
            for i in range(len(in_d.shape))
        )
        sub_idx = idx[sl].ravel()
        dep_out = SparseDepSet(in_d.dep[sub_idx], oshp)
        state.set(eqn.outvars[0], dep_out)

    def introduces_nonlinearity(self, eqn: JaxprEqn, invar_active: list[bool]) -> bool:
        return False

    def eval_concrete(
        self, in_vals: list[np.ndarray | None], params: dict[str, Any]
    ) -> np.ndarray | None:
        if in_vals[0] is None:
            return None
        dims = params["dimensions"]
        return np.flip(np.asarray(in_vals[0]), axis=dims)


class PadHandler(PrimitiveHandler):
    """Handler for `pad` primitive."""

    def propagate_deps(
        self,
        eqn: JaxprEqn,
        state: TraceState,
        acc: CouplingAccumulator,
        trial_test_split: int | None,
    ) -> None:
        from tatva.sparse.tracer.state import SparseDepSet

        in_d = state.get(eqn.invars[0])
        low, high, interior = zip(*eqn.params["padding_config"])
        oshp = _get_shape(eqn.outvars[0]) if eqn.outvars else ()

        # Build index mapping matrix with -1 for padded positions
        idx = np.arange(int(np.prod(in_d.shape))).reshape(in_d.shape)
        padded_idx = np.pad(
            idx,
            [(l, h) for l, h in zip(low, high)],
            mode="constant",
            constant_values=-1,
        )

        # Handle interior padding strides if present
        if any(in_stride > 0 for in_stride in interior):
            slices = []
            for dim, in_stride in enumerate(interior):
                if in_stride > 0:
                    slices.append(slice(None, None, in_stride + 1))
                else:
                    slices.append(slice(None))
            # Reshape index structure for interior insertion
            final_idx = np.full(oshp, -1, dtype=int)
            final_idx[tuple(slices)] = padded_idx
            flat_map = final_idx.ravel()
        else:
            flat_map = padded_idx.ravel()

        # Build output CSR matrix: valid indices copy from in_d.dep, padded (-1) get 0
        valid_mask = flat_map >= 0
        valid_src = flat_map[valid_mask]

        # Sub-slice valid rows into destination CSR
        out_dep = sps.csr_matrix((int(np.prod(oshp)), state.n_dofs), dtype=bool)
        if valid_src.size > 0:
            sub_mat = in_d.dep[valid_src]
            out_dep[valid_mask] = sub_mat

        state.set(eqn.outvars[0], SparseDepSet(out_dep.tocsr(), oshp))

    def introduces_nonlinearity(self, eqn: JaxprEqn, invar_active: list[bool]) -> bool:
        return False

    def eval_concrete(
        self, in_vals: list[np.ndarray | None], params: dict[str, Any]
    ) -> np.ndarray | None:
        if in_vals[0] is None:
            return None
        x = np.asarray(in_vals[0])
        padding_config = params["padding_config"]
        pad_width = [(l, h) for l, h, _ in padding_config]
        # Evaluate constant padding (operand 1 in invars if constant fill value passed)
        fill_val = in_vals[1] if len(in_vals) > 1 and in_vals[1] is not None else 0
        return np.pad(x, pad_width, mode="constant", constant_values=fill_val)
