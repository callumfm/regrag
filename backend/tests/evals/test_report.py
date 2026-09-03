"""The per-case lines `evals run --verbose` adds."""

from app.evals.judge.enums import JudgeVerdict
from app.evals.report import format_case_lines
from tests.evals.conftest import (
    eval_case,
    eval_result,
    failed_judgement,
    out_of_corpus_case,
    passed_judgement,
    refusal_judgement,
    refused_result,
)


def test_a_case_line_shows_what_it_retrieved_what_it_cited_and_how_it_ended():
    [line] = format_case_lines((eval_result(),))

    assert line.startswith("case")
    assert "raw 1.00" in line
    assert "exp 1.00" in line
    assert "cite 1.00" in line
    assert "corr    -" in line
    assert "faith    -" in line
    assert "done" in line
    assert "1000ms" in line


def test_a_case_with_nothing_to_recall_prints_dashes_rather_than_zeroes():
    """An out-of-corpus case authors no reference, so its recall is unmeasured; a 0.00
    would read as a retrieval failure on a case that has nothing to retrieve."""
    [line] = format_case_lines((refused_result(),))

    assert "raw    -" in line
    assert "cite    -" in line
    assert "refused" in line


def test_a_case_the_graph_raised_on_scores_nothing_and_is_named_with_its_error():
    """The aggregate leaves an errored case out, so its line must not show scores either."""
    [line] = format_case_lines((eval_result(eval_case(id="boom"), error="TimeoutError"),))

    assert line.startswith("boom")
    assert "raw    -" in line
    assert "error" in line
    assert line.rstrip().endswith("TimeoutError")


def test_the_case_column_is_sized_to_the_longest_id_in_the_run():
    """The ids run from 13 to 44 characters, so a fixed column either wraps or wastes."""
    lines = format_case_lines(
        (eval_result(eval_case(id="short")), eval_result(eval_case(id="a" * 40)))
    )

    assert [line.index("raw") for line in lines] == [42, 42]


def test_a_judged_case_shows_its_scores_and_a_pass_stays_on_one_line():
    [line] = format_case_lines((eval_result(judgement=passed_judgement()),))

    assert "corr 1.00" in line
    assert "faith 1.00" in line


def test_a_case_the_judge_failed_prints_why_beneath_it():
    line, correctness, faithfulness, unsupported = format_case_lines(
        (eval_result(judgement=failed_judgement()),)
    )

    assert "corr 0.00" in line
    assert "faith 0.50" in line
    assert (
        correctness
        == "    correctness fail (wrong_figure): says all of it, the reference says half"
    )
    assert faithfulness == "    faithfulness 0.50: the 5,000 GT threshold is not in the cited block"
    assert unsupported == "    unsupported: ships above 5,000 GT"


def test_an_answer_that_did_not_decline_prints_the_judges_reason():
    result = eval_result(out_of_corpus_case(), judgement=refusal_judgement(JudgeVerdict.FAIL))

    line, refusal = format_case_lines((result,))

    assert "corr    -" in line
    assert refusal == "    refusal fail: says the corpus lacks it"
