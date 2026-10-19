import jax.numpy as jnp
import numpy as np

from tatva.tracer.program.derivatives import tangent_pattern
from tatva.tracer.program.forms import (
    CoordinateBlock,
    CoordinateRole,
    FormSpec,
    Test,
    Trial,
)


def test_tangent_pattern_defaults_to_energy_coordinates():
    def energy(u):
        return jnp.sum((u[:-1] + u[1:]) ** 2)

    pattern = tangent_pattern(energy, (jnp.arange(5.0),), {})

    expected = np.eye(5, dtype=bool)
    for index in range(4):
        expected[index, index + 1] = True
        expected[index + 1, index] = True
    np.testing.assert_array_equal(pattern.toarray().astype(bool), expected)


def test_tangent_pattern_infers_trial_and_test_coordinates():
    def weak(v: Test, u: Trial):
        return jnp.sum(v * u)

    pattern = tangent_pattern(weak, (jnp.zeros(5), jnp.arange(5.0)), {})

    np.testing.assert_array_equal(
        pattern.toarray().astype(bool),
        np.eye(5, dtype=bool),
    )


def test_tangent_pattern_accepts_an_explicit_form_and_keyword_arguments():
    def weak(u, v):
        return jnp.sum(v * u)

    form = FormSpec(
        (
            CoordinateBlock("u", 0, CoordinateRole.COLUMN),
            CoordinateBlock("v", 1, CoordinateRole.ROW),
        )
    )
    pattern = tangent_pattern(
        weak,
        (jnp.arange(3.0),),
        {"v": jnp.zeros(3)},
        form=form,
    )

    np.testing.assert_array_equal(
        pattern.toarray().astype(bool),
        np.eye(3, dtype=bool),
    )
