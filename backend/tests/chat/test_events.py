"""Sources events: what a retrieved chunk reads as once a stream reports it."""

import pytest

from app.chat.events import ChatSource
from tests.conftest import retrieved_chunk


@pytest.mark.parametrize(
    ("celex", "expected"),
    [("32023R1805", "Regulation (EU) 2023/1805"), ("31992L0043", "31992L0043")],
)
def test_a_source_names_the_act_it_came_from(celex: str, expected: str) -> None:
    source = ChatSource.from_result(3, retrieved_chunk(celex=celex))
    assert source.celex == celex
    assert source.act == expected
