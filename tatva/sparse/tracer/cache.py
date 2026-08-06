# Copyright (C) 2025 ETH Zurich (SMEC)
#
# This file is part of tatva.
#
# tatva is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# tatva is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with tatva.  If not, see <https://www.gnu.org/licenses/>.

import hashlib
import pickle
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

import jax
import jax.numpy as jnp
import numpy as np
import scipy.sparse as sps


def _find_project_root_or_current() -> Path:
    """Find the root of the project by looking for first, a .git directory, then for
    either pyproject.toml or uv.lock."""
    current_dir = Path.cwd()
    while current_dir != current_dir.parent:
        if (current_dir / ".git").exists():
            return current_dir
        if (current_dir / "pyproject.toml").exists() or (
            current_dir / "uv.lock"
        ).exists():
            return current_dir
        current_dir = current_dir.parent

    return Path.cwd()


class _TraceHessianSparsity(Protocol):
    def __call__(
        self,
        jaxpr: jax._src.core.ClosedJaxpr,
        concrete_vals: tuple,
        trial_test_split: int | None = None,
    ) -> sps.csr_matrix: ...


def persistent_tracer_cache(
    cache_dir: str | Path | None = None, skip_cache: bool = False
) -> Callable:
    """Decorator to cache traced sparsity patterns persistently."""
    cache_dir = (
        Path(cache_dir)
        if cache_dir
        else _find_project_root_or_current() / ".tatva" / "sparsity_cache"
    )

    def decorator(func: _TraceHessianSparsity) -> _TraceHessianSparsity:
        def wrapper(
            jaxpr: jax._src.core.ClosedJaxpr,
            concrete_vals: tuple,
            trial_test_split: int | None = None,
        ) -> sps.csr_matrix:
            if skip_cache:
                return func(jaxpr, concrete_vals, trial_test_split)

            key = _compute_cache_key(jaxpr, concrete_vals)
            cache_file = cache_dir / f"{key}.pkl"

            if cache_file.exists():
                print(f"Cache hit for energy functional, loading from {cache_file}")
                with open(cache_file, "rb") as f:
                    return pickle.load(f)

            # Cache miss: execute the function and store the result
            result = func(jaxpr, concrete_vals, trial_test_split)
            # make sure the cache directory exists
            cache_dir.mkdir(parents=True, exist_ok=True)
            with open(cache_file, "wb") as f:
                pickle.dump(result, f)

            return result

        return wrapper

    return decorator


def _hash_pytree(tree: Any) -> str:
    hasher = hashlib.sha256()

    def _update(val):
        if isinstance(val, (np.ndarray, jnp.ndarray)):
            arr = np.asarray(val)
            hasher.update(b"Array")
            hasher.update(str(arr.shape).encode())
            hasher.update(str(arr.dtype).encode())
            hasher.update(np.ascontiguousarray(arr).tobytes())
        elif isinstance(val, (int, float, bool, str, bytes)):
            hasher.update(f"{type(val).__name__}:{val}".encode())
        elif val is None:
            hasher.update(b"None")
        elif isinstance(val, (list, tuple)):
            hasher.update(type(val).__name__.encode())
            for item in val:
                _update(item)
        elif isinstance(val, dict):
            hasher.update(b"dict")
            for key, value in sorted(val.items()):
                _update(key)
                _update(value)
        else:
            try:
                hasher.update(pickle.dumps(val))
            except ValueError:
                hasher.update(repr(val).encode())

    _update(tree)
    return hasher.hexdigest()


def _compute_cache_key(
    jaxpr: jax._src.core.ClosedJaxpr,
    concrete_vals: tuple,
) -> str:
    # 3. Hash computation components
    jaxpr_graph_hash = hashlib.sha256(str(jaxpr).encode()).hexdigest()
    literals_hash = _hash_pytree(jaxpr.literals)
    args_hash = _hash_pytree(concrete_vals)

    # 4. Combine into final key
    combined = f"{jaxpr_graph_hash}_{literals_hash}_{args_hash}"
    return hashlib.sha256(combined.encode()).hexdigest()
