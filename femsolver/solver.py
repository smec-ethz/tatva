from typing import Callable, NamedTuple, TypeVar

import jax
import jax.numpy as jnp

State = TypeVar("State", bound=NamedTuple)
_RT = TypeVar("_RT")


def newton_solver(
    residual: Callable[[State], _RT],
    linear_solve: Callable[[State, _RT], _RT],
    update: Callable[[State, _RT], State],
    state0: State,
    *,
    maxiter: int = 20,
    tol: float = 1e-6,
    rtol: float | None = None,
    verbose: bool = False,
    norm_function: Callable[[_RT], float] = jnp.linalg.norm,
) -> tuple[State, _RT]:
    """A Newton solver for solving nonlinear equations with jax.

    Args:
        residual ((state) -> (residual, Jacobian, norm)): A function that computes the
            residual, its Jacobian, and the norm of the residual.
        linear_solve ((Jacobian, b, state) -> dx): A function that solves the linear system T @
            dx = -residual, where T is the Jacobian returned by the residual function.
        update ((state, dx) -> state): A function that updates the state with the computed
            dx (result of linear_solve).
        state0: The initial state of the system. Must be a tuple/namedtuple of jax
            compatible types.
        maxiter: The maximum number of iterations to perform.
        tol: The tolerance for the norm of the residual. The solver will stop when the norm
            is less than this value. Relative to the initial norm.
        rtol: The relative tolerance for the norm of the residual. The solver will stop
            when the relative change in the norm is less than this value. If None, only tol is
            used.
        verbose: If True, print the iteration information.

    Returns:
        The final state of the system after the Newton iterations. And the final residual.
    """
    state = state0
    res = residual(state)
    norm0 = norm_function(res)
    norm_prev = jnp.inf

    def cond_fn(carry):
        *_, norm, norm_prev, i = carry
        if rtol is not None:
            rtol_ = abs(norm - norm_prev) / norm0
            if verbose:
                jax.debug.print(
                    "iter {}: |Δr| = {}, |r| = {} [rtol={}]",
                    i,
                    rtol_,
                    norm / norm0,
                    rtol,
                )
            return (rtol_ > rtol) & (i < maxiter)
        if verbose:
            jax.debug.print("iter {}: |res| = {} [tol={}]", i, norm, tol)
        return (norm > tol) & (i < maxiter)

    def body_fn(carry):
        state, res, norm, norm_prev, i = carry
        delta = linear_solve(state, res)  # solve T @ dx = -residual
        state = update(state, delta)
        norm_prev = norm
        res = residual(state)
        norm = norm_function(res)
        return (state, res, norm, norm_prev, i + 1)

    state, res, *_ = jax.lax.while_loop(
        cond_fn, body_fn, (state, res, norm0, norm_prev, 0)
    )
    jax.block_until_ready(state)
    return state, res
