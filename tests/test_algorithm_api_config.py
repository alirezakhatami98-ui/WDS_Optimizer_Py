import pickle

import numpy as np
import pytest

from wds_optimizer.algorithm_api.models import AlgorithmConfig


def make_config(**overrides):
    values = {
        "seed": 42,
        "population_size": 50,
        "max_generations": 100,
        "params": {"mutation_rate": 0.2},
    }
    values.update(overrides)
    return AlgorithmConfig(**values)


# ------------------------------------------------------------
# Valid configurations
# ------------------------------------------------------------

def test_valid_config():
    config = make_config()

    assert config.seed == 42
    assert config.population_size == 50
    assert config.max_generations == 100
    assert config.params["mutation_rate"] == 0.2


def test_config_accepts_empty_params():
    config = make_config(params={})

    assert len(config.params) == 0


def test_config_accepts_numpy_integer_values():
    config = make_config(
        seed=np.int64(42),
        population_size=np.int32(50),
        max_generations=np.int64(100),
    )

    assert config.seed == 42
    assert config.population_size == 50
    assert config.max_generations == 100


def test_config_normalizes_numpy_parameter_values():
    config = make_config(
        params={
            "count": np.int64(10),
            "rate": np.float64(0.5),
            "enabled": np.bool_(True),
        }
    )

    assert config.params["count"] == 10
    assert type(config.params["count"]) is int
    assert config.params["rate"] == 0.5
    assert type(config.params["rate"]) is float
    assert config.params["enabled"] is True


# ------------------------------------------------------------
# Integer validation
# ------------------------------------------------------------

@pytest.mark.parametrize(
    "field",
    ["seed", "population_size", "max_generations"],
)
@pytest.mark.parametrize(
    "value",
    [True, False, 1.5, "10", None],
)
def test_integer_fields_reject_invalid_types(field, value):
    with pytest.raises(TypeError, match=field):
        make_config(**{field: value})


def test_seed_must_be_non_negative():
    with pytest.raises(ValueError, match="non-negative"):
        make_config(seed=-1)


@pytest.mark.parametrize(
    "field",
    ["population_size", "max_generations"],
)
def test_positive_integer_fields_reject_zero(field):
    with pytest.raises(ValueError, match="positive"):
        make_config(**{field: 0})


@pytest.mark.parametrize(
    "field",
    ["population_size", "max_generations"],
)
def test_positive_integer_fields_reject_negative_values(field):
    with pytest.raises(ValueError, match="positive"):
        make_config(**{field: -1})


# ------------------------------------------------------------
# Parameter mapping validation
# ------------------------------------------------------------

def test_params_must_be_a_mapping():
    with pytest.raises(TypeError, match="mapping"):
        make_config(params=[("rate", 0.5)])


@pytest.mark.parametrize(
    "key",
    ["", " ", None, 123],
)
def test_parameter_keys_must_be_non_empty_strings(key):
    with pytest.raises(ValueError, match="keys"):
        make_config(params={key: 0.5})


def test_parameter_keys_are_normalized():
    config = make_config(params={" rate ": 0.5})

    assert "rate" in config.params
    assert config.params["rate"] == 0.5


def test_duplicate_normalized_parameter_keys_are_rejected():
    with pytest.raises(ValueError, match="Duplicate"):
        make_config(params={"rate": 0.5, " rate ": 0.8})


@pytest.mark.parametrize(
    "reserved",
    ["seed", "population_size", "max_generations"],
)
def test_reserved_fields_are_rejected_in_params(reserved):
    with pytest.raises(ValueError, match="reserved"):
        make_config(params={reserved: 10})


@pytest.mark.parametrize(
    "value",
    [None, [], {}, object(), complex(1, 2)],
)
def test_params_reject_non_scalar_values(value):
    with pytest.raises(TypeError, match="scalar"):
        make_config(params={"parameter": value})


@pytest.mark.parametrize(
    "value",
    [float("nan"), float("inf"), float("-inf")],
)
def test_params_reject_non_finite_floats(value):
    with pytest.raises(ValueError, match="finite"):
        make_config(params={"parameter": value})


def test_params_accept_supported_scalar_types():
    config = make_config(
        params={
            "integer": 10,
            "float": 0.5,
            "boolean": True,
            "string": "value",
            "empty_string": "",
        }
    )

    assert config.params["integer"] == 10
    assert config.params["float"] == 0.5
    assert config.params["boolean"] is True
    assert config.params["string"] == "value"
    assert config.params["empty_string"] == ""


# ------------------------------------------------------------
# Immutability and independence
# ------------------------------------------------------------

def test_config_is_frozen():
    config = make_config()

    with pytest.raises(AttributeError):
        config.seed = 100


def test_params_mapping_is_read_only():
    config = make_config()

    with pytest.raises(TypeError):
        config.params["mutation_rate"] = 0.9


def test_config_is_independent_of_input_mapping():
    original = {"mutation_rate": 0.2}

    config = make_config(params=original)

    original["mutation_rate"] = 0.9
    original["new_parameter"] = 10

    assert config.params["mutation_rate"] == 0.2
    assert "new_parameter" not in config.params


def test_config_does_not_expose_mutable_input_mapping():
    original = {"mutation_rate": 0.2}

    config = make_config(params=original)

    assert config.params is not original
    assert config.params["mutation_rate"] == 0.2


def test_algorithm_config_is_pickleable():
    config = AlgorithmConfig(
        seed=42,
        population_size=30,
        max_generations=100,
        params={
            "crossover_rate": 0.8,
            "mutation_rate": 0.1,
            "strategy": "best1bin",
        },
    )

    restored = pickle.loads(pickle.dumps(config))

    assert restored.seed == config.seed
    assert restored.population_size == config.population_size
    assert restored.max_generations == config.max_generations
    assert dict(restored.params) == dict(config.params)
    assert restored.params is not config.params


def test_unpickled_algorithm_config_params_remain_immutable():
    config = AlgorithmConfig(
        seed=42,
        population_size=30,
        max_generations=100,
        params={"mutation_rate": 0.1},
    )

    restored = pickle.loads(pickle.dumps(config))

    with pytest.raises(TypeError):
        restored.params["mutation_rate"] = 0.5