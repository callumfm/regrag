"""The boot check that every configured model's provider can be reached."""

import litellm

from app.core.config import configured_models


class MissingModelKeyError(ValueError):
    """A configured model names a provider with no key in the environment."""


def check_model_keys() -> None:
    """Fail at startup when a configured model's provider key is unset, naming the variable
    to set and the models wanting it. Which key a model needs follows from its provider, so
    no settings field can express it; empty is unset, as in the env file."""
    wanted: dict[str, list[str]] = {}
    for model in configured_models():
        for key in litellm.validate_environment(model)["missing_keys"]:
            wanted.setdefault(key, []).append(model)
    if wanted:
        unset = "; ".join(f"{key} for {', '.join(models)}" for key, models in wanted.items())
        raise MissingModelKeyError(f"no provider key in the environment: {unset}")
