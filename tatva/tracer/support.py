from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from jax.extend.core import ClosedJaxpr, Jaxpr, JaxprEqn

from tatva.tracer.core.registry import SEMANTICS
from tatva.tracer.core.semantics import NestedOperationSemantics
from tatva.tracer.local.plan import (
    LocalJaxprPlan,
    pending_routes,
)


class SupportCapability(Enum):
    REGISTRATION = "registration"
    ROUTE_LOCALIZATION = "route localization"


@dataclass(frozen=True, slots=True)
class SupportIssue:
    capability: SupportCapability
    primitive: str
    location: str
    detail: str


class SupportPreflightError(RuntimeError):
    def __init__(
        self,
        issues: tuple[SupportIssue, ...],
    ) -> None:
        if not issues:
            raise ValueError("SupportPreflightError requires at least one issue")

        self.issues = issues

        lines = [f"Tracer support preflight failed with {len(issues)} issue(s):"]

        for issue in issues:
            lines.append(
                f"  - [{issue.capability.value}] "
                f"{issue.location}: "
                f"{issue.primitive}: "
                f"{issue.detail}"
            )

        super().__init__("\n".join(lines))


def _nested_jaxpr(
    eqn: JaxprEqn,
) -> Jaxpr:
    value = eqn.params.get("jaxpr")

    if isinstance(value, ClosedJaxpr):
        return value.jaxpr

    if isinstance(value, Jaxpr):
        return value

    raise RuntimeError(
        f"nested primitive {eqn.primitive.name!r} does not contain a 'jaxpr' parameter"
    )


def registration_issues(
    jaxpr: Jaxpr,
) -> tuple[SupportIssue, ...]:
    issues: list[SupportIssue] = []

    def visit(
        frame: Jaxpr,
        path: tuple[int, ...],
    ) -> None:
        for eqn_index, eqn in enumerate(frame.eqns):
            eqn_path = (*path, eqn_index)
            semantics = SEMANTICS.try_get(eqn.primitive)

            if semantics is None:
                issues.append(
                    SupportIssue(
                        capability=SupportCapability.REGISTRATION,
                        primitive=eqn.primitive.name,
                        location=_format_eqn_path(eqn_path),
                        detail=("no operation semantics registered"),
                    )
                )

                # Important:
                # We do not know the semantics of this primitive,
                # so we also do not know whether any JAXPR-valued
                # params represent executable nested frames.
                continue

            if not isinstance(semantics, NestedOperationSemantics):
                continue

            child = _nested_jaxpr(eqn)
            visit(child, eqn_path)

    visit(jaxpr, ())

    return tuple(issues)


def require_registered_operations(
    jaxpr: Jaxpr,
) -> None:
    issues = registration_issues(jaxpr)
    if issues:
        raise SupportPreflightError(issues)


def route_localization_issues(
    plans: tuple[LocalJaxprPlan, ...],
) -> tuple[SupportIssue, ...]:
    # One lexical equation can occur in many ranks and repeated
    # map/scan invocations. Report it once and aggregate affected ranks.
    pending_by_eqn: dict[int, tuple[JaxprEqn, str, set[int]]] = {}

    for rank, plan in enumerate(plans):
        for local_eqn in pending_routes(plan):
            route = local_eqn.route

            if route is None:
                raise AssertionError("pending_routes returned equation without route")

            key = id(local_eqn.eqn)
            entry = pending_by_eqn.get(key)
            if entry is None:
                pending_by_eqn[key] = (
                    local_eqn.eqn,
                    type(route.global_route).__name__,
                    {rank},
                )
            else:
                entry[2].add(rank)

    issues: list[SupportIssue] = []

    for eqn, route_type, ranks in pending_by_eqn.values():
        rank_text = ", ".join(str(rank) for rank in sorted(ranks))

        issues.append(
            SupportIssue(
                capability=SupportCapability.ROUTE_LOCALIZATION,
                primitive=eqn.primitive.name,
                location=f"rank(s) {rank_text}",
                detail=(
                    f"{route_type} remained unlocalized; "
                    "no local route semantics are available"
                ),
            )
        )

    return tuple(issues)


def require_local_routes(
    plans: tuple[LocalJaxprPlan, ...],
) -> None:
    issues = route_localization_issues(plans)
    if issues:
        raise SupportPreflightError(issues)


def _format_eqn_path(
    path: tuple[int, ...],
) -> str:
    if not path:
        return "root"
    return " / ".join(f"eqn[{index}]" for index in path)


def supported_operations() -> str:
    """Return a human-readable overview of registered tracer operations."""
    return SEMANTICS.overview()
