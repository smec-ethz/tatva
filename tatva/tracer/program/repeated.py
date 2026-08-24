from tatva.tracer.core.nested import MapSpec
from tatva.tracer.program.analysis import EqnPlan


def map_template_requires_mapped_concrete(
    eqn_plan: EqnPlan,
    spec: MapSpec,
) -> bool:
    nested = eqn_plan.nested
    assert nested is not None, "eqn_plan must be nested"

    return any(
        input_index >= spec.num_consts for input_index in nested.body.concrete_inputs
    )
