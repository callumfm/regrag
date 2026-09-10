"""What a role's model call is made with, and the boot check that the provider it names
can be reached."""

import litellm

from app.core.models import FrozenModel


class ModelSettings(FrozenModel):
    """One role's model call: which model answers, and the limits it answers under. A role
    that leaves temperature unset sends none, keeping the provider's own default."""

    model: str
    timeout: int
    max_tokens: int
    temperature: float | None = None


class MissingModelKeyError(ValueError):
    """A configured model names a provider with no key in the environment."""


def check_model_keys(*models: str) -> None:
    """Fail at startup when a configured model's provider key is unset, naming every model
    that cannot be reached and the variable each wants. Which key a model needs follows from
    its provider, so no settings field can express it; empty is unset, as in the env file."""
    missing = {
        model: keys
        for model in dict.fromkeys(models)
        if (keys := litellm.validate_environment(model)["missing_keys"])
    }
    if missing:
        unusable = "; ".join(f"{model} needs {', '.join(keys)}" for model, keys in missing.items())
        raise MissingModelKeyError(f"no provider key for {unusable}")
