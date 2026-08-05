from app.ingestion.chunk.references import Reference, extract_references


def test_extracts_internal_article_reference() -> None:
    refs = extract_references("calculated in accordance with Article 6 of this Regulation")
    assert Reference(raw="Article 6", instrument=None, article="6") in refs


def test_extracts_paragraph_from_internal_article_reference() -> None:
    refs = extract_references("as referred to in Article 6(2)")
    assert refs == (Reference(raw="Article 6(2)", instrument=None, article="6", paragraph="2"),)


def test_extracts_letter_suffixed_article_number() -> None:
    refs = extract_references("the procedure in Article 11a(3) applies")
    assert refs == (Reference(raw="Article 11a(3)", instrument=None, article="11a", paragraph="3"),)


def test_extracts_internal_annex_reference() -> None:
    refs = extract_references("using the methods set out in Annex I")
    assert refs == (Reference(raw="Annex I", instrument=None, annex="I"),)


def test_deduplicates_repeated_references() -> None:
    refs = extract_references("Annex I applies. As stated in Annex I, the factor is fixed.")
    assert len(refs) == 1


def test_ignores_prose_without_references() -> None:
    assert extract_references("This Regulation lays down rules on the use of fuels.") == ()


def test_resolves_regulation_to_celex() -> None:
    refs = extract_references("as defined in Regulation (EU) 2015/757")
    assert refs == (Reference(raw="Regulation (EU) 2015/757", instrument="32015R0757"),)


def test_resolves_directive_to_celex() -> None:
    refs = extract_references("the scope of Directive 2003/87/EC")
    assert refs == (Reference(raw="Directive 2003/87/EC", instrument="32003L0087"),)


def test_resolves_numbered_instrument_as_number_before_year() -> None:
    refs = extract_references("repealing Regulation (EC) No 765/2008")
    assert refs == (Reference(raw="Regulation (EC) No 765/2008", instrument="32008R0765"),)


def test_attributes_article_to_the_instrument_it_qualifies() -> None:
    refs = extract_references("verified under Article 6(2) of Regulation (EU) 2015/757")
    assert refs == (
        Reference(
            raw="Article 6(2) of Regulation (EU) 2015/757",
            instrument="32015R0757",
            article="6",
            paragraph="2",
        ),
    )


def test_does_not_treat_this_regulation_as_an_external_instrument() -> None:
    refs = extract_references("Article 6 of this Regulation applies")
    assert refs == (Reference(raw="Article 6", instrument=None, article="6"),)
