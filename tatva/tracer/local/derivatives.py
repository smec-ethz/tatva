"""Rank-local structural derivative analysis for generic scalar forms.

A local executable consumes compact compute inputs, while derivative coordinates
may live in a different storage layout (for example [owned | ghosts]).  We
therefore trace the executable directly and seed an explicit sparse relation
from its local input rows into one unified symbolic coordinate layout.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import numpy as np
import scipy.sparse as sps
from numpy.typing import NDArray

from tatva.tracer.helpers import _shape_of
from tatva.tracer.local.dof_plan import LocalDofPlan
from tatva.tracer.local.layout import TensorLayout
from tatva.tracer.lowering.executor import LocalExecutable
from tatva.tracer.program.analysis import analyze
from tatva.tracer.program.dependencies import DependencySet
from tatva.tracer.program.derivatives import (
    DerivativeTrace,
    JaxprDerivativeTrace,
    trace_seeded_derivatives,
)
from tatva.tracer.program.forms import FormSpec, SymbolicLayout
from tatva.tracer.program.materialize import materialize_plan
from tatva.tracer.support import require_registered_operations


def _freeze_i64(values) -> NDArray[np.int64]:
    result = np.asarray(values, dtype=np.int64).ravel().copy()
    result.flags.writeable = False
    return result


@dataclass(frozen=True, slots=True)
class LocalDerivativeTrace:
    """Derivative sparsity for one rank-local scalar form."""

    trace: DerivativeTrace
    block_global_ids: dict[str, NDArray[np.int64]]
    storage_global_dofs: NDArray[np.int64]
    global_size: int

    def __post_init__(self) -> None:
        storage = _freeze_i64(self.storage_global_dofs)
        if self.global_size < 0:
            raise ValueError("global_size must be nonnegative")
        if np.any((storage < 0) | (storage >= self.global_size)):
            raise ValueError("storage_global_dofs contains out-of-range DOFs")

        frozen: dict[str, NDArray[np.int64]] = {}
        for block in self.trace.symbolic_layout.blocks:
            try:
                ids = _freeze_i64(self.block_global_ids[block.name])
            except KeyError as exc:
                raise ValueError(
                    f"missing global IDs for symbolic block {block.name!r}"
                ) from exc
            if ids.size != block.size:
                raise ValueError(
                    f"block {block.name!r} has {block.size} symbols but "
                    f"{ids.size} global IDs"
                )
            frozen[block.name] = ids

        object.__setattr__(self, "storage_global_dofs", storage)
        object.__setattr__(self, "block_global_ids", frozen)

    @property
    def root(self) -> JaxprDerivativeTrace:
        return self.trace.root

    @property
    def tangent(self) -> sps.csr_matrix:
        return self.trace.tangent

    @property
    def hessian(self) -> sps.csr_matrix:
        return self.trace.hessian

    @property
    def row_block_names(self) -> tuple[str, ...]:
        return self.trace.symbolic_layout.row_block_names

    @property
    def column_block_names(self) -> tuple[str, ...]:
        return self.trace.symbolic_layout.column_block_names

    def global_hessian_coo(self) -> sps.coo_matrix:
        hessian = self.hessian
        layout = self.trace.symbolic_layout
        if len(layout.blocks) != 1:
            raise AttributeError(
                "global_hessian_coo is only defined for a one-block energy form"
            )
        ids = self.block_global_ids[layout.blocks[0].name]
        if not np.array_equal(ids, self.storage_global_dofs):
            raise AttributeError(
                "energy coordinates do not match canonical storage ordering"
            )
        local = hessian.tocoo()
        return sps.coo_matrix(
            (
                local.data.copy(),
                (ids[local.row], ids[local.col]),
            ),
            shape=(self.global_size, self.global_size),
        )


def _global_rows_for_local_layout(layout: TensorLayout) -> NDArray[np.int64]:
    return layout.local_rows_to_global_rows(
        np.arange(layout.local_size, dtype=np.int64)
    ).ravel()


def _selected_local_global_rows(
    *,
    global_rows: NDArray[np.int64],
    selected_global: NDArray[np.int64],
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    if selected_global.size == global_rows.size and np.array_equal(
        selected_global, global_rows
    ):
        local_rows = np.arange(global_rows.size, dtype=np.int64)
        return local_rows, global_rows

    selected = set(int(row) for row in selected_global)
    mask = np.fromiter(
        (int(row) in selected for row in global_rows),
        dtype=bool,
        count=global_rows.size,
    )
    local_rows = np.flatnonzero(mask).astype(np.int64, copy=False)
    return local_rows, global_rows[local_rows]


def _build_local_symbolic_seeds(
    *,
    form: FormSpec,
    executable: LocalExecutable,
    dof_plan: LocalDofPlan,
    global_inputs: tuple[Any, ...],
    jaxpr_input_shapes: tuple[tuple[int, ...], ...],
) -> tuple[
    SymbolicLayout,
    tuple[DependencySet, ...],
    dict[str, NDArray[np.int64]],
]:
    if not executable.input_indices or executable.input_indices[0] != 0:
        raise RuntimeError("local executable does not have a live first state input")
    if len(jaxpr_input_shapes) != len(executable.input_indices):
        raise RuntimeError("local executable ABI and captured JAXPR inputs disagree")

    slot_by_original = {
        original: slot for slot, original in enumerate(executable.input_indices)
    }

    # First determine each local symbolic block and its global coordinate IDs.
    block_rows_by_slot: dict[str, tuple[int, NDArray[np.int64]]] = {}
    block_global_ids: dict[str, NDArray[np.int64]] = {}
    block_specs: list[tuple] = []

    for block in form.coordinates:
        original_index = block.input_index
        if original_index >= len(global_inputs):
            raise ValueError(
                f"coordinate block {block.name!r} references missing input "
                f"{original_index}"
            )
        slot = slot_by_original.get(original_index)
        if slot is None:
            raise RuntimeError(
                f"coordinate block {block.name!r} is compiler-dead in the local form"
            )

        original_size = int(np.prod(np.shape(global_inputs[original_index])))
        selected_global = block.rows(original_size)

        if original_index == 0:
            # Symbolic state coordinates use storage order, but the executable
            # receives compute order.  This distinction is represented by the
            # explicit root seed below rather than by inserting a JAX gather.
            symbolic_global_rows = np.asarray(
                dof_plan.storage.global_dofs, dtype=np.int64
            )
            _, global_ids = _selected_local_global_rows(
                global_rows=symbolic_global_rows,
                selected_global=selected_global,
            )

            compute_global = np.asarray(dof_plan.compute_global, dtype=np.int64)
            selected_set = set(int(row) for row in global_ids)
            executable_rows = np.flatnonzero(
                np.fromiter(
                    (int(row) in selected_set for row in compute_global),
                    dtype=bool,
                    count=compute_global.size,
                )
            ).astype(np.int64, copy=False)
        else:
            local_layout = executable.plan.input_layouts[original_index]
            if local_layout is None:
                raise RuntimeError(
                    f"coordinate block {block.name!r} has no local input layout"
                )
            local_global_rows = _global_rows_for_local_layout(local_layout)
            executable_rows, global_ids = _selected_local_global_rows(
                global_rows=local_global_rows,
                selected_global=selected_global,
            )

        block_global_ids[block.name] = np.asarray(global_ids, dtype=np.int64)
        block_rows_by_slot[block.name] = (slot, executable_rows)
        block_specs.append(
            (block.name, global_ids.size, block.role, block.value_source)
        )

    symbolic_layout = SymbolicLayout.from_sizes(tuple(block_specs))

    matrices = [
        sps.lil_matrix(
            (int(np.prod(shape, dtype=np.int64)), symbolic_layout.size),
            dtype=bool,
        )
        for shape in jaxpr_input_shapes
    ]

    # Bind executable rows to symbolic columns.  For the first state input the
    # symbolic ordering is storage-local, so map through compute_global ->
    # storage-local positions explicitly.
    for form_block, symbolic_block in zip(
        form.coordinates, symbolic_layout.blocks, strict=True
    ):
        slot, executable_rows = block_rows_by_slot[form_block.name]
        global_ids = block_global_ids[form_block.name]

        if form_block.input_index == 0:
            lookup = {
                int(global_id): symbolic_block.offset + local_index
                for local_index, global_id in enumerate(global_ids)
            }
            compute_global = np.asarray(dof_plan.compute_global, dtype=np.int64)
            rows: list[int] = []
            cols: list[int] = []
            for row in executable_rows:
                global_id = int(compute_global[row])
                column = lookup.get(global_id)
                if column is not None:
                    rows.append(int(row))
                    cols.append(column)
            if rows:
                matrices[slot][np.asarray(rows), np.asarray(cols)] = True
        else:
            if executable_rows.size != symbolic_block.size:
                raise RuntimeError(
                    f"coordinate block {form_block.name!r} local row count "
                    "does not match its symbolic size"
                )
            if executable_rows.size:
                matrices[slot][executable_rows, symbolic_block.columns] = True

    input_deps = tuple(
        DependencySet(matrix.tocsr(), shape)
        for matrix, shape in zip(matrices, jaxpr_input_shapes, strict=True)
    )
    return symbolic_layout, input_deps, block_global_ids


def trace_local_derivatives(
    executable: LocalExecutable,
    dof_plan: LocalDofPlan,
    global_inputs: tuple[Any, ...],
    *,
    form: FormSpec | None = None,
) -> LocalDerivativeTrace:
    """Trace one lowered rank scalar form in local symbolic coordinates."""
    if len(global_inputs) != len(executable.plan.input_layouts):
        raise ValueError(
            f"expected {len(executable.plan.input_layouts)} global inputs, "
            f"got {len(global_inputs)}"
        )

    form = FormSpec.energy(input_index=0) if form is None else form
    examples = executable.pack_global_inputs(*global_inputs)
    closed_jaxpr = jax.make_jaxpr(executable)(*examples)
    require_registered_operations(closed_jaxpr.jaxpr)
    plan = analyze(closed_jaxpr.jaxpr)
    instance = materialize_plan(closed_jaxpr, examples, plan)

    symbolic_layout, input_deps, block_global_ids = _build_local_symbolic_seeds(
        form=form,
        executable=executable,
        dof_plan=dof_plan,
        global_inputs=global_inputs,
        jaxpr_input_shapes=tuple(_shape_of(var) for var in closed_jaxpr.jaxpr.invars),
    )
    derivative_trace = trace_seeded_derivatives(
        instance,
        symbolic_layout,
        input_deps,
    )

    return LocalDerivativeTrace(
        trace=derivative_trace,
        block_global_ids=block_global_ids,
        storage_global_dofs=dof_plan.storage.global_dofs,
        global_size=dof_plan.global_size,
    )
