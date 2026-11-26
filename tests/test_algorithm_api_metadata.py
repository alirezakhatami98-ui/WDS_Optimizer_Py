
import copy

import pytest

from wds_optimizer.algorithm_api.metadata import (
    API_VERSION,
    ALLOWED_CONSTRAINT_POLICIES,
    validate_algorithm_metadata,
)


@pytest.fixture
def valid_metadata():
    return {
        "api_version": 2,
        "short_id": "MY_ALGORITHM",
        "display_name": "My Algorithm",
        "category": "Population-based",
        "paper_faithful": False,
        "description": "An example optimization algorithm.",
        "references": ["Example reference"],
        "params": [
            {
                "key": "learning_rate",
                "name": "lr",
                "display_name": "Learning Rate",
                "type": "float",
                "default": 0.5,
                "min": 0.0,
                "max": 1.0,
                "description": "Learning rate.",
                "maps_to": None,
            }
        ],
        "constraint_policy": "deb_prefers",
    }


def test_valid_metadata_is_accepted(valid_metadata):
    result = validate_algorithm_metadata(valid_metadata)

    assert result["api_version"] == API_VERSION
    assert result["short_id"] == "MY_ALGORITHM"
    assert result["constraint_policy"] == "deb_prefers"
    assert len(result["params"]) == 1


def test_all_allowed_constraint_policies_are_accepted(valid_metadata):
    for policy in ALLOWED_CONSTRAINT_POLICIES:
        metadata = copy.deepcopy(valid_metadata)
        metadata["constraint_policy"] = policy

        result = validate_algorithm_metadata(metadata)

        assert result["constraint_policy"] == policy


def test_optional_fields_are_preserved(valid_metadata):
    valid_metadata.update(
        {
            "author": "Example Author",
            "version": "1.0.0",
            "tags": ["discrete", "optimization"],
        }
    )

    result = validate_algorithm_metadata(valid_metadata)

    assert result["author"] == "Example Author"
    assert result["version"] == "1.0.0"
    assert result["tags"] == ["discrete", "optimization"]


def test_missing_required_metadata_field_is_rejected(valid_metadata):
    del valid_metadata["display_name"]

    with pytest.raises(ValueError, match="missing required fields"):
        validate_algorithm_metadata(valid_metadata)


def test_unknown_metadata_field_is_rejected(valid_metadata):
    valid_metadata["unexpected"] = "value"

    with pytest.raises(ValueError, match="unknown fields"):
        validate_algorithm_metadata(valid_metadata)


@pytest.mark.parametrize("value", [True, 2.0, "2", None])
def test_api_version_must_be_integer(valid_metadata, value):
    valid_metadata["api_version"] = value

    with pytest.raises((TypeError, ValueError)):
        validate_algorithm_metadata(valid_metadata)


def test_unsupported_api_version_is_rejected(valid_metadata):
    valid_metadata["api_version"] = 3

    with pytest.raises(ValueError, match="Unsupported api_version"):
        validate_algorithm_metadata(valid_metadata)


@pytest.mark.parametrize(
    "short_id",
    [
        "my_algorithm",
        "1ALGORITHM",
        "MY-ALGORITHM",
        "MY ALGORITHM",
        "",
    ],
)
def test_invalid_short_id_is_rejected(valid_metadata, short_id):
    valid_metadata["short_id"] = short_id

    with pytest.raises(ValueError, match="short_id"):
        validate_algorithm_metadata(valid_metadata)


def test_paper_faithful_must_be_boolean(valid_metadata):
    valid_metadata["paper_faithful"] = 1

    with pytest.raises(TypeError, match="paper_faithful"):
        validate_algorithm_metadata(valid_metadata)


def test_invalid_constraint_policy_is_rejected(valid_metadata):
    valid_metadata["constraint_policy"] = "custom_policy"

    with pytest.raises(ValueError, match="constraint_policy"):
        validate_algorithm_metadata(valid_metadata)


@pytest.mark.parametrize(
    "param_type,default",
    [
        ("int", 10),
        ("float", 0.25),
        ("bool", True),
        ("str", "value"),
        ("choice", "DE"),
    ],
)
def test_supported_parameter_types_are_accepted(
    valid_metadata,
    param_type,
    default,
):
    param = {
        "key": "parameter",
        "name": "parameter",
        "display_name": "Parameter",
        "type": param_type,
        "default": default,
        "description": "Test parameter.",
    }

    if param_type == "choice":
        param["choices"] = ["GA", "PSO", "DE"]

    valid_metadata["params"] = [param]

    result = validate_algorithm_metadata(valid_metadata)

    assert result["params"][0]["type"] == param_type


def test_integer_parameter_rejects_boolean(valid_metadata):
    valid_metadata["params"][0].update(
        {"type": "int", "default": True}
    )

    with pytest.raises(TypeError, match="integer"):
        validate_algorithm_metadata(valid_metadata)


def test_float_parameter_rejects_boolean(valid_metadata):
    valid_metadata["params"][0]["default"] = True

    with pytest.raises(TypeError, match="finite number"):
        validate_algorithm_metadata(valid_metadata)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_float_parameter_rejects_non_finite_values(
    valid_metadata,
    value,
):
    valid_metadata["params"][0]["default"] = value

    with pytest.raises(TypeError, match="finite number"):
        validate_algorithm_metadata(valid_metadata)


def test_default_must_respect_minimum(valid_metadata):
    valid_metadata["params"][0]["default"] = -0.1

    with pytest.raises(ValueError, match="greater than or equal"):
        validate_algorithm_metadata(valid_metadata)


def test_default_must_respect_maximum(valid_metadata):
    valid_metadata["params"][0]["default"] = 1.1

    with pytest.raises(ValueError, match="less than or equal"):
        validate_algorithm_metadata(valid_metadata)


def test_minimum_must_not_exceed_maximum(valid_metadata):
    valid_metadata["params"][0]["min"] = 0.8
    valid_metadata["params"][0]["max"] = 0.2

    with pytest.raises(ValueError, match="must not exceed"):
        validate_algorithm_metadata(valid_metadata)


def test_choice_default_must_exist_in_choices(valid_metadata):
    valid_metadata["params"][0] = {
        "key": "strategy",
        "name": "strategy",
        "display_name": "Strategy",
        "type": "choice",
        "default": "UNKNOWN",
        "choices": ["GA", "PSO", "DE"],
        "description": "Selection strategy.",
    }

    with pytest.raises(ValueError, match="must be one of"):
        validate_algorithm_metadata(valid_metadata)


def test_choice_values_must_be_unique(valid_metadata):
    valid_metadata["params"][0] = {
        "key": "strategy",
        "name": "strategy",
        "display_name": "Strategy",
        "type": "choice",
        "default": "GA",
        "choices": ["GA", "GA"],
        "description": "Selection strategy.",
    }

    with pytest.raises(ValueError, match="duplicates"):
        validate_algorithm_metadata(valid_metadata)


def test_choice_parameter_requires_non_empty_choices(valid_metadata):
    valid_metadata["params"][0] = {
        "key": "strategy",
        "name": "strategy",
        "display_name": "Strategy",
        "type": "choice",
        "default": "GA",
        "choices": [],
        "description": "Selection strategy.",
    }

    with pytest.raises(ValueError, match="non-empty list"):
        validate_algorithm_metadata(valid_metadata)


def test_bool_parameter_rejects_numeric_bounds(valid_metadata):
    valid_metadata["params"][0] = {
        "key": "enabled",
        "name": "enabled",
        "display_name": "Enabled",
        "type": "bool",
        "default": True,
        "min": 0,
        "description": "Enable option.",
    }

    with pytest.raises(ValueError, match="does not support"):
        validate_algorithm_metadata(valid_metadata)


def test_duplicate_parameter_keys_are_rejected(valid_metadata):
    valid_metadata["params"].append(
        copy.deepcopy(valid_metadata["params"][0])
    )

    with pytest.raises(ValueError, match="keys must be unique"):
        validate_algorithm_metadata(valid_metadata)


def test_unknown_parameter_field_is_rejected(valid_metadata):
    valid_metadata["params"][0]["unexpected"] = 123

    with pytest.raises(ValueError, match="unknown fields"):
        validate_algorithm_metadata(valid_metadata)


def test_references_must_be_list_of_non_empty_strings(valid_metadata):
    valid_metadata["references"] = ["Reference", ""]

    with pytest.raises(ValueError, match="must not be empty"):
        validate_algorithm_metadata(valid_metadata)


def test_tags_must_be_unique(valid_metadata):
    valid_metadata["tags"] = ["discrete", "discrete"]

    with pytest.raises(ValueError, match="duplicates"):
        validate_algorithm_metadata(valid_metadata)


def test_maps_to_must_be_string_or_none(valid_metadata):
    valid_metadata["params"][0]["maps_to"] = 123

    with pytest.raises(TypeError, match="maps_to"):
        validate_algorithm_metadata(valid_metadata)


def test_result_is_independent_from_input(valid_metadata):
    result = validate_algorithm_metadata(valid_metadata)

    valid_metadata["params"][0]["default"] = 0.9
    valid_metadata["references"].append("Another reference")

    assert result["params"][0]["default"] == 0.5
    assert result["references"] == ["Example reference"]


def test_mutating_result_does_not_change_input(valid_metadata):
    original = copy.deepcopy(valid_metadata)

    result = validate_algorithm_metadata(valid_metadata)
    result["params"][0]["default"] = 0.8
    result["references"].append("New reference")

    assert valid_metadata == original


def test_empty_optional_fields_are_valid(valid_metadata):
    valid_metadata["references"] = []
    valid_metadata["tags"] = []

    result = validate_algorithm_metadata(valid_metadata)

    assert result["references"] == []
    assert result["tags"] == []
