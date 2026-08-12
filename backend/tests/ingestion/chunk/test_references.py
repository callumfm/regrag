from app.ingestion.chunk.models import Reference
from app.ingestion.chunk.references import (
    extract_references,
    find_division_mentions,
    find_instrument_mentions,
    unattributed_instruments,
)


def test_extracts_internal_article_reference() -> None:
    references = extract_references("calculated in accordance with Article 6 of this Regulation")
    assert Reference(raw="Article 6", instrument=None, article="6") in references


def test_extracts_paragraph_from_internal_article_reference() -> None:
    references = extract_references("as referred to in Article 6(2)")
    assert references == (
        Reference(raw="Article 6(2)", instrument=None, article="6", paragraph="2"),
    )


def test_extracts_letter_suffixed_article_number() -> None:
    references = extract_references("the procedure in Article 11a(3) applies")
    assert references == (
        Reference(raw="Article 11a(3)", instrument=None, article="11a", paragraph="3"),
    )


def test_extracts_internal_annex_reference() -> None:
    references = extract_references("using the methods set out in Annex I")
    assert references == (Reference(raw="Annex I", instrument=None, annex="I"),)


def test_deduplicates_repeated_references() -> None:
    references = extract_references("Annex I applies. As stated in Annex I, the factor is fixed.")
    assert len(references) == 1


def test_ignores_prose_without_references() -> None:
    assert extract_references("This Regulation lays down rules on the use of fuels.") == ()


def test_resolves_regulation_to_celex() -> None:
    references = extract_references("as defined in Regulation (EU) 2015/757")
    assert references == (Reference(raw="Regulation (EU) 2015/757", instrument="32015R0757"),)


def test_resolves_directive_to_celex() -> None:
    references = extract_references("the scope of Directive 2003/87/EC")
    assert references == (Reference(raw="Directive 2003/87/EC", instrument="32003L0087"),)


def test_resolves_numbered_instrument_as_number_before_year() -> None:
    references = extract_references("repealing Regulation (EC) No 765/2008")
    assert references == (Reference(raw="Regulation (EC) No 765/2008", instrument="32008R0765"),)


def test_resolves_a_two_digit_year_to_the_twentieth_century() -> None:
    references = extract_references("as amended by Council Directive 92/43/EEC")
    assert references == (Reference(raw="Council Directive 92/43/EEC", instrument="31992L0043"),)


def test_resolves_a_numbered_instrument_with_a_two_digit_year() -> None:
    references = extract_references("referred to in Regulation (EEC) No 2913/92")
    assert references == (Reference(raw="Regulation (EEC) No 2913/92", instrument="31992R2913"),)


def test_reads_the_year_as_the_year_even_when_the_no_is_dropped() -> None:
    references = extract_references("registered under Regulation (EC) 1907/2006")
    assert references == (Reference(raw="Regulation (EC) 1907/2006", instrument="32006R1907"),)


def test_reads_a_high_act_number_as_a_number_not_a_year() -> None:
    references = extract_references("laid down in Regulation (EU) 2018/2066")
    assert references == (Reference(raw="Regulation (EU) 2018/2066", instrument="32018R2066"),)


def test_reads_a_year_shaped_act_number_as_the_number() -> None:
    references = extract_references("promoting Directive (EU) 2018/2001")
    assert references == (Reference(raw="Directive (EU) 2018/2001", instrument="32018L2001"),)


def test_drops_an_instrument_whose_citation_cannot_be_a_celex_id() -> None:
    assert extract_references("the fictional Regulation 3021/4055") == ()


def test_extracts_every_article_of_an_enumeration() -> None:
    references = extract_references("in accordance with Articles 6, 7 and 8")
    assert references == tuple(Reference(raw=f"Article {n}", article=n) for n in ("6", "7", "8"))


def test_attributes_every_article_of_an_enumeration_to_its_instrument() -> None:
    references = extract_references("under Articles 6 and 7 of Regulation (EU) 2015/757")
    assert references == tuple(
        Reference(
            raw=f"Article {n} of Regulation (EU) 2015/757",
            instrument="32015R0757",
            article=n,
        )
        for n in ("6", "7")
    )


def test_extracts_every_annex_of_an_enumeration() -> None:
    references = extract_references("set out in Annexes I and II")
    assert references == tuple(Reference(raw=f"Annex {n}", annex=n) for n in ("I", "II"))


def test_does_not_read_a_trailing_year_as_a_further_article() -> None:
    references = extract_references("Article 6, 2015 saw the adoption of the scheme")
    assert references == (Reference(raw="Article 6", article="6"),)


def test_attributes_article_to_the_instrument_it_qualifies() -> None:
    references = extract_references("verified under Article 6(2) of Regulation (EU) 2015/757")
    assert references == (
        Reference(
            raw="Article 6(2) of Regulation (EU) 2015/757",
            instrument="32015R0757",
            article="6",
            paragraph="2",
        ),
    )


def test_does_not_treat_this_regulation_as_an_external_instrument() -> None:
    references = extract_references("Article 6 of this Regulation applies")
    assert references == (Reference(raw="Article 6", instrument=None, article="6"),)


def test_find_instrument_mentions_records_the_span_and_its_celex() -> None:
    text = "as defined in Regulation (EU) 2015/757"
    mention = find_instrument_mentions(text)[0]
    assert text[mention.start : mention.end] == "Regulation (EU) 2015/757"
    assert mention.celex == "32015R0757"


def test_find_instrument_mentions_leaves_celex_none_when_it_cannot_resolve() -> None:
    assert find_instrument_mentions("the fictional Regulation 3021/4055")[0].celex is None


def test_a_division_is_qualified_by_the_instrument_that_follows_of() -> None:
    text = "under Article 6 of Regulation (EU) 2015/757"
    division = find_division_mentions(text)[0]
    instrument = find_instrument_mentions(text)[0]
    assert division.is_qualified_by(instrument, text)


def test_a_division_is_not_qualified_by_an_instrument_it_only_precedes() -> None:
    text = "Article 6 applies. Regulation (EU) 2015/757 does not."
    division = find_division_mentions(text)[0]
    instrument = find_instrument_mentions(text)[0]
    assert not division.is_qualified_by(instrument, text)


def test_an_instrument_a_division_claimed_is_not_cited_again_in_its_own_right() -> None:
    text = "under Article 6 of Regulation (EU) 2015/757"
    divisions = find_division_mentions(text)
    assert unattributed_instruments(text, divisions, find_instrument_mentions(text)) == []
    assert unattributed_instruments(text, [], find_instrument_mentions(text)) != []


def test_attributes_article_across_a_council_prefix() -> None:
    references = extract_references("pursuant to Article 3 of Council Regulation (EEC) No 3577/92")
    assert references == (
        Reference(
            raw="Article 3 of Council Regulation (EEC) No 3577/92",
            instrument="31992R3577",
            article="3",
        ),
    )


def test_attributes_article_across_a_commission_implementing_prefix() -> None:
    references = extract_references(
        "in Article 2 of Commission Implementing Regulation (EU) 2016/1927"
    )
    assert references == (
        Reference(
            raw="Article 2 of Commission Implementing Regulation (EU) 2016/1927",
            instrument="32016R1927",
            article="2",
        ),
    )


def test_attributes_article_across_a_commission_delegated_prefix() -> None:
    references = extract_references(
        "under Article 5 of Commission Delegated Regulation (EU) 2023/1640"
    )
    assert references == (
        Reference(
            raw="Article 5 of Commission Delegated Regulation (EU) 2023/1640",
            instrument="32023R1640",
            article="5",
        ),
    )


def test_attributes_article_across_a_parliament_and_council_prefix() -> None:
    references = extract_references(
        "under Article 6 of European Parliament and Council Directive 95/46/EC"
    )
    assert references == (
        Reference(
            raw="Article 6 of European Parliament and Council Directive 95/46/EC",
            instrument="31995L0046",
            article="6",
        ),
    )


def test_a_numbered_pre_2000_instrument_reads_number_then_year() -> None:
    references = extract_references("slots allocated under Regulation (EEC) No 95/93")
    assert references == (Reference(raw="Regulation (EEC) No 95/93", instrument="31993R0095"),)


def test_a_numbered_instrument_with_two_year_shaped_halves_reads_number_first() -> None:
    references = extract_references("fertilisers under Regulation (EC) No 2003/2003")
    assert references == (Reference(raw="Regulation (EC) No 2003/2003", instrument="32003R2003"),)
