"""Explicit, potentially expensive inspection of analyzed functionals."""

from __future__ import annotations

import functools

from tatva.tracer.api import FunctionalAnalysis
from tatva.tracer.program.concrete_resolver import ConcreteResolver
from tatva.tracer.program.contributions import ContributionBlock
from tatva.tracer.program.derivatives import DerivativeTrace, trace_form_derivatives
from tatva.tracer.program.incidence import (
    BlockDofIncidence,
    generate_contribution_blocks,
    plan_tagged_block_dof_incidence,
)
from tatva.tracer.program.materialize import JaxprInstance, materialize_plan


def contribution_blocks(
    functional: FunctionalAnalysis,
    *,
    blocks_per_root: int,
) -> tuple[ContributionBlock, ...]:
    """Construct explicit contribution blocks for inspection."""
    return generate_contribution_blocks(
        functional._contributions,
        blocks_per_root=blocks_per_root,
    )


def incidence(
    functional: FunctionalAnalysis,
    blocks: tuple[ContributionBlock, ...],
) -> BlockDofIncidence:
    """Compute sparse contribution-block-to-DOF incidence."""
    resolver, frame = ConcreteResolver.root(
        functional._captured.closed_jaxpr,
        functional._captured.flat_args,
        functional._plan,
        unavailable_inputs=functional._form.coordinate_input_indices,
    )
    return plan_tagged_block_dof_incidence(
        functional._plan,
        frame,
        resolver,
        functional._contributions,
        blocks=blocks,
    )


@functools.cache
def materialize(functional: FunctionalAnalysis) -> JaxprInstance:
    """Materialize and cache the complete invocation tree for diagnostics."""
    return materialize_plan(
        functional._captured.closed_jaxpr,
        functional._captured.flat_args,
        functional._plan,
    )


@functools.cache
def global_derivatives(functional: FunctionalAnalysis) -> DerivativeTrace:
    """Materialize and analyze global derivative sparsity."""
    return trace_form_derivatives(
        materialize(functional),
        functional._form,
    )
