"""A discovered document's derived values."""

from tests.conftest import discovered_document


def test_versions_try_the_consolidation_before_the_original_act():
    consolidated = discovered_document(candidates=("02015R0757-20250101",))
    assert consolidated.versions == ["02015R0757-20250101", "32015R0757"]


def test_versions_without_a_consolidation_is_the_act_alone():
    assert discovered_document("32023R2449").versions == ["32023R2449"]
