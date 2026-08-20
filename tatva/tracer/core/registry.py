from jax.extend.core import Primitive

from tatva.tracer.core.semantics import (
    CallAnalysisSemantics,
    CondAnalysisSemantics,
    CustomJvpAnalysisSemantics,
    LinearSolveAnalysisSemantics,
    NestedOperationSemantics,
    OperationSemantics,
    RegisteredOperationSemantics,
    ScanAnalysisSemantics,
    conservative_demand,
    contribution_barrier,
    full_concrete_evaluation,
    no_route,
)
from tatva.tracer.rules.registration import register_builtin_rules


class PrimitiveRegistry:
    def __init__(self):
        self._rules: dict[Primitive, RegisteredOperationSemantics] = {}

    def register(
        self, primitive: Primitive, rule: RegisteredOperationSemantics
    ) -> None:
        if primitive in self._rules:
            raise ValueError(f"Primitive {primitive.name} is already registered.")

        self._rules[primitive] = rule

    def try_get(
        self,
        primitive: Primitive,
    ) -> RegisteredOperationSemantics | None:
        return self._rules.get(primitive)

    def get(self, primitive: Primitive) -> RegisteredOperationSemantics:
        rule = self.try_get(primitive)
        if rule is None:
            raise KeyError(f"No semantics registered for primitive {primitive.name}")
        return rule

    def get_ordinary(self, primitive: Primitive) -> OperationSemantics:
        rule = self.get(primitive)
        if not isinstance(rule, OperationSemantics):
            raise TypeError(
                f"Expected ordinary OperationSemantics for {primitive.name}, "
                f"got {type(rule).__name__}"
            )
        return rule

    def get_nested(self, primitive: Primitive) -> NestedOperationSemantics:
        rule = self.get(primitive)
        if not isinstance(rule, NestedOperationSemantics):
            raise TypeError(
                f"Expected NestedOperationSemantics for {primitive.name}, "
                f"got {type(rule).__name__}"
            )
        return rule

    def validate(self) -> None:
        errors: list[str] = []

        for primitive, rule in self._rules.items():
            if isinstance(rule, NestedOperationSemantics):
                if not isinstance(
                    rule.analysis,
                    (
                        CallAnalysisSemantics,
                        ScanAnalysisSemantics,
                        CondAnalysisSemantics,
                        LinearSolveAnalysisSemantics,
                        CustomJvpAnalysisSemantics,
                    ),
                ):
                    errors.append(
                        f"{primitive.name}: unsupported nested analysis "
                        f"{type(rule.analysis).__name__}"
                    )

                continue

            localizer = rule.localization.localize_route
            if (
                localizer is not None
                and rule.routing is not None
                and rule.routing.resolve is no_route
            ):
                errors.append(
                    f"{primitive.name}: route localizer is registered "
                    "but the operation has no route resolver"
                )

        if errors:
            formatted = "\n".join(f"  - {error}" for error in errors)
            raise ValueError(f"Invalid primitive semantics registry:\n{formatted}")

    def describe(
        self,
        primitive: Primitive,
    ) -> str:
        rule = self.get(primitive)
        if isinstance(rule, NestedOperationSemantics):
            return "\n".join(
                (
                    f"{primitive.name}: nested",
                    f"  analysis: {type(rule.analysis).__name__}",
                )
            )

        has_route = rule.routing and rule.routing.resolve is not no_route

        if not has_route:
            route_localization = "n/a"
        elif rule.localization.localize_route is None:
            route_localization = "unsupported"
        else:
            route_localization = "supported"

        return "\n".join(
            (
                f"{primitive.name}: ordinary",
                "  derivatives: supported",
                (
                    "  demand: conservative"
                    if rule.demand is conservative_demand
                    else "  demand: specialized"
                ),
                (
                    "  contribution: barrier"
                    if rule.contribution is contribution_barrier
                    else "  contribution: specialized"
                ),
                ("  routing: none" if not has_route else "  routing: supported"),
                f"  route localization: {route_localization}",
                (
                    "  lowering: generic bind"
                    if rule.lowering is None
                    else "  lowering: specialized"
                ),
                (
                    "  regional concrete: full fallback"
                    if rule.regional_concrete is full_concrete_evaluation
                    else "  regional concrete: specialized"
                ),
            )
        )

    def overview(self) -> str:
        rows: list[tuple[str, ...]] = []

        for primitive, rule in sorted(
            self._rules.items(),
            key=lambda item: item[0].name,
        ):
            if isinstance(rule, NestedOperationSemantics):
                rows.append(
                    (
                        primitive.name,
                        "nested",
                        type(rule.analysis).__name__,
                        "-",
                        "-",
                        "-",
                        "-",
                        "-",
                    )
                )
                continue

            has_route = rule.routing and rule.routing.resolve is not no_route
            has_localizer = rule.localization.localize_route is not None

            rows.append(
                (
                    primitive.name,
                    "ordinary",
                    _rule_name(rule.demand),
                    (
                        "barrier"
                        if rule.contribution is contribution_barrier
                        else _rule_name(rule.contribution)
                    ),
                    "yes" if has_route else "-",
                    ("yes" if has_localizer else "no" if has_route else "-"),
                    (
                        _rule_name(rule.lowering)
                        if rule.lowering is not None
                        else "bind"
                    ),
                    (
                        "full"
                        if rule.regional_concrete is full_concrete_evaluation
                        else _rule_name(rule.regional_concrete)
                    ),
                )
            )

        return _format_table(
            (
                "primitive",
                "kind",
                "demand/analysis",
                "contribution",
                "routing",
                "localization",
                "lowering",
                "regional concrete",
            ),
            rows,
        )


def _rule_name(rule: object) -> str:
    return getattr(rule, "__name__", type(rule).__name__)


def _format_table(
    headers: tuple[str, ...],
    rows: list[tuple[str, ...]],
) -> str:
    widths = [
        max(len(headers[i]), *(len(row[i]) for row in rows))
        for i in range(len(headers))
    ]

    def format_row(row: tuple[str, ...]) -> str:
        return "  ".join(
            value.ljust(width) for value, width in zip(row, widths, strict=True)
        )

    return "\n".join(
        (
            format_row(headers),
            format_row(tuple("-" * width for width in widths)),
            *(format_row(row) for row in rows),
        )
    )


def get_primitive_registry() -> PrimitiveRegistry:
    """Return the global primitive registry."""
    reg = PrimitiveRegistry()
    register_builtin_rules(reg)
    reg.validate()
    return reg


SEMANTICS = get_primitive_registry()
