from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

from jax import custom_derivatives, lax
from jax._src.lax import lax as lax_internal
from jax.extend.core import primitives

from tatva.tracer.core.nested import CallKind
from tatva.tracer.core.route_fragments import (
    gather_route_concrete_demands,
    resolve_dynamic_slice_route_fragment,
    resolve_dynamic_update_slice_route_fragment,
    resolve_gather_route_fragment,
    resolve_select_n_route_fragment,
    select_route_concrete_demands,
)
from tatva.tracer.core.routes import (
    resolve_dynamic_slice_route,
    resolve_dynamic_update_slice_route,
    resolve_gather_route,
    resolve_select_n_route,
)
from tatva.tracer.core.semantics import (
    CallAnalysisSemantics,
    CondAnalysisSemantics,
    DerivativeRule,
    LinearSolveAnalysisSemantics,
    LocalizationSemantics,
    NestedOperationSemantics,
    OperationSemantics,
    ScanAnalysisSemantics,
    custom_derivative_call_target,
    no_hessian,
)
from tatva.tracer.lowering import rules as lowerings
from tatva.tracer.rules.elementwise import (
    ELEMENTWISE_BINARY_BASIC,
    ELEMENTWISE_NARY_BASIC,
    INTEGER_POW,
    LINEAR_UNARY,
    NONLINEAR_UNARY,
)
from tatva.tracer.rules.structural import RESHAPE_LIKE

from . import (
    concrete as concrete_rules,
)
from . import (
    contributions as contribution_rules,
)
from . import (
    dot,
    elementwise,
    gather_scatter,
    indexing,
    linalg,
    opaque,
    reductions,
    structural,
    tagged,
)
from .zero_dependency import EMPTY, IOTA, NO_OP, ZERO_DEPENDENCY

if TYPE_CHECKING:
    from tatva.tracer.core.registry import PrimitiveRegistry


def _register_zero_deps_rules(reg: PrimitiveRegistry) -> None:
    reg.register(
        lax_internal.empty_p,
        replace(
            EMPTY,
            lowering=lowerings.lower_empty,
        ),
    )
    reg.register(
        lax.iota_p,
        replace(
            IOTA,
            lowering=lowerings.lower_iota,
        ),
    )
    # these have no dependency propagation, but they will have demand and contribution rules defined later
    for primitive in (
        lax.lt_p,
        lax.lt_to_p,
        lax.le_p,
        lax.le_to_p,
        lax.gt_p,
        lax.ge_p,
        lax.eq_p,
        lax.ne_p,
        lax.and_p,
        lax.or_p,
        lax.not_p,
        lax.xor_p,
        lax.is_finite_p,
        # lax.is_nan_p,  # doesn't exist, where is it defined?
        lax.argmax_p,
        lax.argmin_p,
        lax.floor_p,
        lax.ceil_p,
        lax.round_p,
        lax.sign_p,
    ):
        reg.register(primitive, ZERO_DEPENDENCY)

    for primitive in (
        lax.shift_left_p,
        lax.shift_right_arithmetic_p,
    ):
        reg.register(primitive, ZERO_DEPENDENCY)

    reg.register(
        lax.stop_gradient_p,
        replace(
            ZERO_DEPENDENCY,
            contribution=contribution_rules.transparent_unary,
        ),
    )

    try:
        from jax._src.debugging import debug_callback_p, debug_print_p
        from jax._src.lax.control_flow import platform_index_p

        reg.register(debug_print_p, NO_OP)
        reg.register(debug_callback_p, NO_OP)
        reg.register(platform_index_p, IOTA)
    except ImportError:
        pass


def _register_elementwise_unary_rules(reg: PrimitiveRegistry) -> None:
    reg.register(
        lax.neg_p,
        replace(
            LINEAR_UNARY,
            contribution=contribution_rules.negative_unary,
        ),
    )
    for primitive in (
        lax.abs_p,
        lax.device_put_p,
        lax.conj_p,
        lax.real_p,
        lax.imag_p,
    ):
        reg.register(primitive, LINEAR_UNARY)

    for primitive in (
        lax.copy_p,
        lax.convert_element_type_p,
    ):
        reg.register(
            primitive,
            replace(
                LINEAR_UNARY,
                contribution=contribution_rules.transparent_unary,
            ),
        )

    for primitive in (
        lax.sin_p,
        lax.cos_p,
        lax.tan_p,
        lax.asin_p,
        lax.acos_p,
        lax.atan_p,
        lax.exp_p,
        lax.exp2_p,
        lax.expm1_p,
        lax.log_p,
        lax.log1p_p,
        # lax.log2_p,
        lax.sqrt_p,
        lax.rsqrt_p,
        lax.cbrt_p,
        lax.tanh_p,
        lax.sinh_p,
        lax.cosh_p,
        lax.atanh_p,
        lax.asinh_p,
        lax.acosh_p,
        lax.erf_p,
        lax.erfc_p,
        # lax.erfinv_p,
        lax.lgamma_p,
        lax.digamma_p,
        lax.logistic_p,
    ):
        reg.register(primitive, NONLINEAR_UNARY)

    reg.register(lax.integer_pow_p, INTEGER_POW)


def _register_elementwise_binary_rules(reg: PrimitiveRegistry) -> None:
    reg.register(
        lax.add_p,
        replace(
            ELEMENTWISE_BINARY_BASIC,
            contribution=contribution_rules.additive_add,
        ),
    )
    reg.register(
        lax.sub_p,
        replace(
            ELEMENTWISE_BINARY_BASIC,
            contribution=contribution_rules.additive_sub,
        ),
    )
    for primitive in (
        lax.min_p,
        lax.max_p,
    ):
        reg.register(primitive, ELEMENTWISE_BINARY_BASIC)

    reg.register(lax.clamp_p, ELEMENTWISE_NARY_BASIC)

    reg.register(
        lax.mul_p,
        OperationSemantics(
            DerivativeRule(
                elementwise.prepare_elementwise_binary,
                elementwise.union_dependencies,
                elementwise.elementwise_mul_hessian,
            ),
            demand=elementwise.elementwise_demand,
            tagged_demand=tagged.elementwise,
            contribution=contribution_rules.scalar_multiply,
            regional_concrete=concrete_rules.regional_bind(
                elementwise.elementwise_demand
            ),
        ),
    )
    reg.register(
        lax.div_p,
        OperationSemantics(
            DerivativeRule(
                elementwise.prepare_elementwise_binary,
                elementwise.union_dependencies,
                elementwise.elementwise_div_hessian,
            ),
            demand=elementwise.elementwise_demand,
            tagged_demand=tagged.elementwise,
            contribution=contribution_rules.scalar_divide,
            regional_concrete=concrete_rules.regional_bind(
                elementwise.elementwise_demand
            ),
        ),
    )
    reg.register(
        lax.pow_p,
        OperationSemantics(
            DerivativeRule(
                elementwise.prepare_elementwise_binary,
                elementwise.union_dependencies,
                elementwise.elementwise_pow_hessian,
            ),
            demand=elementwise.elementwise_demand,
            tagged_demand=tagged.elementwise,
            regional_concrete=concrete_rules.regional_bind(
                elementwise.elementwise_demand
            ),
        ),
    )
    reg.register(
        lax.atan2_p,
        OperationSemantics(
            DerivativeRule(
                elementwise.prepare_elementwise_binary,
                elementwise.union_dependencies,
                elementwise.elementwise_atan2_hessian,
            ),
            demand=elementwise.elementwise_demand,
            tagged_demand=tagged.elementwise,
            regional_concrete=concrete_rules.regional_bind(
                elementwise.elementwise_demand
            ),
        ),
    )
    reg.register(
        lax.rem_p,
        OperationSemantics(
            DerivativeRule(
                elementwise.prepare_elementwise_binary,
                elementwise.union_dependencies,
                no_hessian,
            ),
            demand=elementwise.elementwise_demand,
            tagged_demand=tagged.elementwise,
            regional_concrete=concrete_rules.regional_bind(
                elementwise.elementwise_demand
            ),
        ),
    )


def _register_structural_rules(reg: PrimitiveRegistry) -> None:
    reg.register(
        lax.reshape_p,
        replace(
            RESHAPE_LIKE,
            contribution=contribution_rules.transparent_unary,
            lowering=lowerings.lower_reshape,
            regional_concrete=concrete_rules.regional_reshape(
                structural.demand_reshape_squeeze
            ),
        ),
    )
    reg.register(
        lax.squeeze_p,
        replace(
            RESHAPE_LIKE,
            contribution=contribution_rules.transparent_unary,
            regional_concrete=concrete_rules.regional_reshape(
                structural.demand_reshape_squeeze
            ),
        ),
    )
    reg.register(
        lax.broadcast_in_dim_p,
        OperationSemantics(
            DerivativeRule(
                prepare=structural.prepare_broadcast,
                dependencies=structural.unary_routed_dependencies,
                hessian=no_hessian,
            ),
            demand=structural.demand_broadcast_in_dim,
            tagged_demand=tagged.broadcast_in_dim,
            lowering=lowerings.lower_broadcast_in_dim,
            regional_concrete=concrete_rules.regional_broadcast(
                structural.demand_broadcast_in_dim
            ),
        ),
    )
    reg.register(
        lax.transpose_p,
        OperationSemantics(
            DerivativeRule(
                prepare=structural.prepare_transpose,
                dependencies=structural.unary_routed_dependencies,
                hessian=no_hessian,
            ),
            demand=structural.demand_transpose,
            tagged_demand=tagged.transpose,
            contribution=contribution_rules.transparent_unary,
            regional_concrete=concrete_rules.regional_bind(structural.demand_transpose),
        ),
    )
    reg.register(
        lax.slice_p,
        OperationSemantics(
            DerivativeRule(
                prepare=structural.prepare_slice,
                dependencies=structural.unary_routed_dependencies,
                hessian=no_hessian,
            ),
            demand=structural.demand_slice,
            tagged_demand=tagged.slice_,
            lowering=lowerings.lower_slice,
            regional_concrete=concrete_rules.regional_projected_unary(
                structural.demand_slice
            ),
        ),
    )
    reg.register(
        lax.rev_p,
        OperationSemantics(
            DerivativeRule(
                prepare=structural.prepare_rev,
                dependencies=structural.unary_routed_dependencies,
                hessian=no_hessian,
            ),
            demand=structural.demand_rev,
            tagged_demand=tagged.rev,
            contribution=contribution_rules.transparent_unary,
            regional_concrete=concrete_rules.regional_bind(structural.demand_rev),
        ),
    )
    reg.register(
        lax.concatenate_p,
        OperationSemantics(
            DerivativeRule(
                prepare=structural.prepare_concatenate,
                dependencies=structural.multi_input_routed_dependencies,
                hessian=no_hessian,
            ),
            demand=structural.demand_concatenate,
            tagged_demand=tagged.concatenate,
            lowering=lowerings.lower_concatenate,
            regional_concrete=concrete_rules.regional_bind(
                structural.demand_concatenate
            ),
        ),
    )
    reg.register(
        lax.stack_p,
        OperationSemantics(
            DerivativeRule(
                prepare=structural.prepare_stack,
                dependencies=structural.multi_input_routed_dependencies,
                hessian=no_hessian,
            ),
            demand=structural.demand_stack,
            tagged_demand=tagged.stack,
        ),
    )
    reg.register(
        lax.pad_p,
        OperationSemantics(
            DerivativeRule(
                prepare=structural.prepare_pad,
                dependencies=structural.multi_input_routed_dependencies,
                hessian=no_hessian,
            ),
            demand=structural.demand_pad,
            tagged_demand=tagged.pad,
        ),
    )
    reg.register(
        lax.split_p,
        OperationSemantics(
            DerivativeRule(
                structural.prepare_split,
                structural.multi_output_unary_routed_dependencies,
                no_hessian,
            ),
            demand=structural.demand_split,
            tagged_demand=tagged.split,
        ),
    )


def _register_routing_rules(reg: PrimitiveRegistry) -> None:
    reg.register(
        lax.gather_p,
        OperationSemantics(
            DerivativeRule(
                prepare=gather_scatter.prepare_gather,
                dependencies=gather_scatter.gather_dependencies,
                hessian=no_hessian,
            ),
            concrete_inputs=lambda _eqn: (1,),
            route=resolve_gather_route,
            route_fragment=resolve_gather_route_fragment,
            route_concrete_demands=gather_route_concrete_demands,
            demand=gather_scatter.gather_demand,
            tagged_demand=tagged.gather,
            localization=LocalizationSemantics(
                gather_scatter.localize_gather,
            ),
            lowering=lowerings.lower_gather,
        ),
    )
    for primitive in (
        lax.scatter_add_p,
        lax.scatter_sub_p,
        lax.scatter_min_p,
        lax.scatter_max_p,
    ):
        reg.register(primitive, gather_scatter.SCATTER_ACCUMULATE)

    reg.register(
        lax.scatter_p,
        replace(
            gather_scatter.SCATTER_BASIC,
            lowering=lowerings.lower_scatter_set,
        ),
    )
    reg.register(lax.scatter_mul_p, gather_scatter.SCATTER_MUL)

    # select_n
    reg.register(
        lax.select_n_p,
        OperationSemantics(
            DerivativeRule(
                indexing.prepare_select_n,
                indexing.select_n_dependencies,
                no_hessian,
            ),
            concrete_inputs=lambda _eqn: (0,),
            route=resolve_select_n_route,
            route_fragment=resolve_select_n_route_fragment,
            route_concrete_demands=select_route_concrete_demands,
            demand=indexing.select_n_demand,
            tagged_demand=tagged.select_n,
            regional_concrete=concrete_rules.regional_bind(
                elementwise.elementwise_demand
            ),
            localization=LocalizationSemantics(
                localize_route=indexing.localize_select_n,
            ),
            lowering=lowerings.lower_select_n,
        ),
    )

    # slicing dynamic
    reg.register(
        lax.dynamic_slice_p,
        OperationSemantics(
            DerivativeRule(
                prepare=indexing.prepare_dynamic_slice,
                dependencies=indexing.dynamic_slice_dependencies,
                hessian=no_hessian,
            ),
            concrete_inputs=lambda eqn: tuple(range(1, len(eqn.invars))),
            route=resolve_dynamic_slice_route,
            route_fragment=resolve_dynamic_slice_route_fragment,
            demand=indexing.dynamic_slice_demand,
            tagged_demand=tagged.dynamic_slice,
            localization=LocalizationSemantics(
                localize_route=indexing.localize_dynamic_slice,
            ),
            lowering=lowerings.lower_dynamic_slice,
        ),
    )
    reg.register(
        lax.dynamic_update_slice_p,
        OperationSemantics(
            DerivativeRule(
                indexing.prepare_dynamic_update_slice,
                indexing.dynamic_update_slice_dependencies,
                no_hessian,
            ),
            concrete_inputs=lambda eqn: tuple(range(2, len(eqn.invars))),
            route=resolve_dynamic_update_slice_route,
            route_fragment=resolve_dynamic_update_slice_route_fragment,
            demand=indexing.dynamic_update_slice_demand,
            tagged_demand=tagged.dynamic_update_slice,
        ),
    )


def _register_reduction_rules(reg: PrimitiveRegistry) -> None:
    reg.register(
        lax.reduce_sum_p,
        replace(
            reductions.REDUCE_BASIC,
            contribution=contribution_rules.reduce_sum,
            regional_concrete=concrete_rules.regional_bind(
                reductions.reduce_sum_demand
            ),
        ),
    )

    for primitive in (
        lax.reduce_max_p,
        lax.reduce_min_p,
    ):
        reg.register(
            primitive,
            replace(
                reductions.REDUCE_BASIC,
                regional_concrete=concrete_rules.regional_bind(
                    reductions.reduce_sum_demand
                ),
            ),
        )

    reg.register(lax.reduce_prod_p, reductions.REDUCE_PROD)
    reg.register(lax.reduce_and_p, reductions.ZERO_REDUCTION)
    reg.register(lax.reduce_or_p, reductions.ZERO_REDUCTION)


def _register_dot_general(reg: PrimitiveRegistry) -> None:
    reg.register(
        lax.dot_general_p,
        OperationSemantics(
            DerivativeRule(
                dot.prepare_dot_general,
                dot.dot_general_dependencies,
                dot.dot_general_hessian,
            ),
            demand=dot.dot_general_demand,
            tagged_demand=tagged.dot_general,
        ),
    )


def _register_opaque_rules(reg: PrimitiveRegistry) -> None:
    # Register rules for opaque primitives if needed
    for primitive in (
        lax.linalg.cholesky_p,
        lax.linalg.eig_p,
        lax.linalg.eigh_p,
    ):
        reg.register(
            primitive,
            OperationSemantics(
                derivatives=opaque.DERIVATIVES_OPAQUE_NONLINEAR,
            ),
        )

    reg.register(
        lax.linalg.triangular_solve_p,
        OperationSemantics(
            derivatives=opaque.DERIVATIVES_OPAQUE_NONLINEAR,
            demand=linalg.triangular_solve_demand,
            tagged_demand=tagged.triangular_solve,
        ),
    )
    reg.register(
        primitives.lu_p,
        OperationSemantics(
            derivatives=opaque.DERIVATIVES_OPAQUE_NONLINEAR,
            demand=linalg.lu_demand,
            tagged_demand=tagged.lu,
        ),
    )
    reg.register(
        lax.sort_p,
        OperationSemantics(
            derivatives=opaque.DERIVATIVES_OPAQUE_NONLINEAR,
            demand=opaque.sort_demand,
            tagged_demand=tagged.sort,
            lowering=lowerings.lower_sort,
            regional_concrete=concrete_rules.full(
                "sort requires a globally ordered invocation-local input"
            ),
        ),
    )


def _register_nested_rules(
    reg: PrimitiveRegistry,
) -> None:
    reg.register(
        primitives.jit_p,
        NestedOperationSemantics(
            analysis=CallAnalysisSemantics(
                call_kind=CallKind.JIT,
            )
        ),
    )

    reg.register(
        primitives.remat_p,
        NestedOperationSemantics(
            analysis=CallAnalysisSemantics(
                call_kind=CallKind.REMAT,
            )
        ),
    )

    reg.register(
        lax.linear_solve_p,
        NestedOperationSemantics(analysis=LinearSolveAnalysisSemantics()),
    )

    reg.register(
        primitives.scan_p,
        NestedOperationSemantics(
            analysis=ScanAnalysisSemantics(),
        ),
    )

    reg.register(
        lax.cond_p,
        NestedOperationSemantics(
            analysis=CondAnalysisSemantics(),
        ),
    )

    reg.register(
        custom_derivatives.custom_jvp_call_p,
        NestedOperationSemantics(
            analysis=CallAnalysisSemantics(
                call_kind=CallKind.CUSTOM_JVP,
                target=custom_derivative_call_target,
            )
        ),
    )

    reg.register(
        custom_derivatives.custom_vjp_call_p,
        NestedOperationSemantics(
            analysis=CallAnalysisSemantics(
                call_kind=CallKind.CUSTOM_VJP,
                target=custom_derivative_call_target,
            )
        ),
    )


def _register_unstable_api_rules(reg: PrimitiveRegistry) -> None:
    from jax._src.ad_util import add_any_p

    reg.register(
        add_any_p,
        ELEMENTWISE_BINARY_BASIC,
    )


def register_builtin_rules(reg: PrimitiveRegistry) -> None:
    _register_nested_rules(reg)

    _register_zero_deps_rules(reg)
    _register_elementwise_unary_rules(reg)
    _register_elementwise_binary_rules(reg)
    _register_structural_rules(reg)
    _register_routing_rules(reg)
    _register_reduction_rules(reg)
    _register_dot_general(reg)
    _register_opaque_rules(reg)

    # Register rules for unstable API primitives if enabled
    _register_unstable_api_rules(reg)
