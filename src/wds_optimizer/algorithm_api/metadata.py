
"""Metadata validation for Algorithm Plugin API v2."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from numbers import Integral, Real
from typing import Any


API_VERSION = 2

ALLOWED_CONSTRAINT_POLICIES = frozenset(
    {
        "deb_prefers",
        "fixed_penalty_prefers",
        "simple_cost_only_prefers",
    }
)

ALLOWED_PARAM_TYPES = frozenset(
    {"int", "float", "bool", "str", "choice"}
)

_METADATA_REQUIRED = frozenset(
    {
        "api_version",
        "short_id",
        "display_name",
        "category",
        "paper_faithful",
        "description",
        "references",
        "params",
        "constraint_policy",
    }
)

_METADATA_OPTIONAL = frozenset(
    {"author", "version", "tags"}
)

_PARAM_REQUIRED = frozenset(
    {
        "key",
        "name",
        "display_name",
        "type",
        "default",
        "description",
    }
)

_PARAM_OPTIONAL = frozenset(
    {"min", "max", "choices", "maps_to"}
)

_SHORT_ID_PATTERN = re.compile(r"^[A-Z_][A-Z0-9_]*$")


def _require_mapping(value: Any, field: str) -> Mapping:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field} must be a mapping.")
    return value


def _check_keys(
    data: Mapping,
    required: frozenset[str],
    optional: frozenset[str],
    field: str,
) -> None:
    keys = set(data.keys())

    missing = required - keys
    unknown = keys - required - optional

    if missing:
        names = ", ".join(sorted(missing))
        raise ValueError(f"{field} is missing required fields: {names}.")

    if unknown:
        names = ", ".join(sorted(str(key) for key in unknown))
        raise ValueError(f"{field} contains unknown fields: {names}.")


def _require_string(
    value: Any,
    field: str,
    *,
    allow_empty: bool = False,
) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field} must be a string.")

    normalized = value.strip()

    if not allow_empty and not normalized:
        raise ValueError(f"{field} must not be empty.")

    return normalized


def _require_string_list(
    value: Any,
    field: str,
    *,
    unique: bool = False,
) -> list[str]:
    if not isinstance(value, list):
        raise TypeError(f"{field} must be a list.")

    result = [
        _require_string(item, f"{field}[{index}]")
        for index, item in enumerate(value)
    ]

    if unique and len(result) != len(set(result)):
        raise ValueError(f"{field} must not contain duplicates.")

    return result


def _is_number(value: Any) -> bool:
    return (
        isinstance(value, Real)
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _validate_numeric_value(
    value: Any,
    field: str,
    param_type: str,
) -> int | float:
    if param_type == "int":
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError(f"{field} must be an integer.")
        return int(value)

    if not _is_number(value):
        raise TypeError(f"{field} must be a finite number.")

    return float(value)


def _same_choice(a: Any, b: Any) -> bool:
    """Compare choices without treating True and 1 as identical."""
    return type(a) is type(b) and a == b


def _validate_choices(value: Any, field: str) -> list[Any]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{field} must be a non-empty list.")

    result: list[Any] = []

    for index, item in enumerate(value):
        if isinstance(item, str):
            result.append(
                _require_string(item, f"{field}[{index}]")
            )
        elif isinstance(item, bool):
            result.append(item)
        elif isinstance(item, int):
            result.append(item)
        elif isinstance(item, float) and math.isfinite(item):
            result.append(item)
        else:
            raise TypeError(
                f"{field}[{index}] must be a string, "
                "boolean, integer, or finite float."
            )

    for index, item in enumerate(result):
        if any(
            _same_choice(item, previous)
            for previous in result[:index]
        ):
            raise ValueError(f"{field} must not contain duplicates.")

    return result


def _validate_param(
    raw_param: Any,
    index: int,
) -> dict[str, Any]:
    field = f"params[{index}]"
    param = _require_mapping(raw_param, field)

    _check_keys(
        param,
        _PARAM_REQUIRED,
        _PARAM_OPTIONAL,
        field,
    )

    result: dict[str, Any] = {
        "key": _require_string(param["key"], f"{field}.key"),
        "name": _require_string(param["name"], f"{field}.name"),
        "display_name": _require_string(
            param["display_name"],
            f"{field}.display_name",
        ),
        "description": _require_string(
            param["description"],
            f"{field}.description",
        ),
    }

    param_type = _require_string(
        param["type"],
        f"{field}.type",
    )

    if param_type not in ALLOWED_PARAM_TYPES:
        raise ValueError(
            f"{field}.type must be one of "
            f"{sorted(ALLOWED_PARAM_TYPES)}."
        )

    result["type"] = param_type

    if param_type in {"int", "float"}:
        result["default"] = _validate_numeric_value(
            param["default"],
            f"{field}.default",
            param_type,
        )

        for bound in ("min", "max"):
            if bound in param:
                result[bound] = _validate_numeric_value(
                    param[bound],
                    f"{field}.{bound}",
                    param_type,
                )

        if (
            "min" in result
            and "max" in result
            and result["min"] > result["max"]
        ):
            raise ValueError(
                f"{field}.min must not exceed {field}.max."
            )

        if (
            "min" in result
            and result["default"] < result["min"]
        ):
            raise ValueError(
                f"{field}.default must be greater than or equal "
                "to min."
            )

        if (
            "max" in result
            and result["default"] > result["max"]
        ):
            raise ValueError(
                f"{field}.default must be less than or equal "
                "to max."
            )

        if "choices" in param:
            raise ValueError(
                f"{field}.choices is only allowed for type='choice'."
            )

    elif param_type == "bool":
        if not isinstance(param["default"], bool):
            raise TypeError(f"{field}.default must be a boolean.")

        result["default"] = param["default"]

        if any(key in param for key in ("min", "max", "choices")):
            raise ValueError(
                f"{field} does not support min, max, or choices."
            )

    elif param_type == "str":
        result["default"] = _require_string(
            param["default"],
            f"{field}.default",
            allow_empty=True,
        )

        if any(key in param for key in ("min", "max", "choices")):
            raise ValueError(
                f"{field} does not support min, max, or choices."
            )

    else:  # choice
        choices = _validate_choices(
            param.get("choices"),
            f"{field}.choices",
        )

        default = param["default"]

        if not any(
            _same_choice(default, choice)
            for choice in choices
        ):
            raise ValueError(
                f"{field}.default must be one of {field}.choices."
            )

        result["default"] = default
        result["choices"] = choices

        if any(key in param for key in ("min", "max")):
            raise ValueError(
                f"{field} does not support min or max."
            )

    if "maps_to" in param:
        maps_to = param["maps_to"]

        if maps_to is not None:
            maps_to = _require_string(
                maps_to,
                f"{field}.maps_to",
            )

        result["maps_to"] = maps_to

    return result


def validate_algorithm_metadata(
    metadata: Any,
) -> dict[str, Any]:
    """Validate and normalize Algorithm Plugin API v2 metadata.

    Returns an independent dictionary containing only normalized,
    supported metadata fields.

    Raises:
        TypeError: When a field has an invalid data type.
        ValueError: When metadata violates the API contract.
    """
    data = _require_mapping(metadata, "metadata")

    _check_keys(
        data,
        _METADATA_REQUIRED,
        _METADATA_OPTIONAL,
        "metadata",
    )

    api_version = data["api_version"]

    if (
        isinstance(api_version, bool)
        or not isinstance(api_version, Integral)
    ):
        raise TypeError("api_version must be an integer.")

    if api_version != API_VERSION:
        raise ValueError(
            f"Unsupported api_version: {api_version}. "
            f"Expected {API_VERSION}."
        )

    short_id = _require_string(
        data["short_id"],
        "short_id",
    )

    if not _SHORT_ID_PATTERN.fullmatch(short_id):
        raise ValueError(
            "short_id must contain only uppercase letters, "
            "digits, and underscores, and must not start with a digit."
        )

    paper_faithful = data["paper_faithful"]

    if not isinstance(paper_faithful, bool):
        raise TypeError("paper_faithful must be a boolean.")

    constraint_policy = _require_string(
        data["constraint_policy"],
        "constraint_policy",
    )

    if constraint_policy not in ALLOWED_CONSTRAINT_POLICIES:
        raise ValueError(
            "constraint_policy must be one of "
            f"{sorted(ALLOWED_CONSTRAINT_POLICIES)}."
        )

    references = _require_string_list(
        data["references"],
        "references",
    )

    if not isinstance(data["params"], list):
        raise TypeError("params must be a list.")

    params = [
        _validate_param(param, index)
        for index, param in enumerate(data["params"])
    ]

    param_keys = [param["key"] for param in params]

    if len(param_keys) != len(set(param_keys)):
        raise ValueError("Parameter keys must be unique.")

    result: dict[str, Any] = {
        "api_version": API_VERSION,
        "short_id": short_id,
        "display_name": _require_string(
            data["display_name"],
            "display_name",
        ),
        "category": _require_string(
            data["category"],
            "category",
        ),
        "paper_faithful": paper_faithful,
        "description": _require_string(
            data["description"],
            "description",
        ),
        "references": references,
        "params": params,
        "constraint_policy": constraint_policy,
    }

    if "author" in data:
        result["author"] = _require_string(
            data["author"],
            "author",
        )

    if "version" in data:
        result["version"] = _require_string(
            data["version"],
            "version",
        )

    if "tags" in data:
        result["tags"] = _require_string_list(
            data["tags"],
            "tags",
            unique=True,
        )

    return result
