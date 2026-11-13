"""Deterministic, neutral, dataset-scoped pseudonyms.

A replacement is a neutral token such as ``<PERSON_1>`` or ``<PHONE_2>``: not
a realistic name that could itself look like PHI, not derived from the value,
consistent within one dataset and unrelated across datasets. Seeded generators
stand in for a property-testing library. Every value is invented.
"""

import os
import random
import sys

import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services import surrogates  # noqa: E402
from services.surrogates import SURROGATE_PATTERN, SurrogateAllocator  # noqa: E402


def test_the_same_value_keeps_one_token_and_different_values_get_different_ones():
    allocator = SurrogateAllocator()

    first = allocator.surrogate_for("PERSON", "Zelquin Vantreske")
    again = allocator.surrogate_for("PERSON", "Zelquin Vantreske")
    other = allocator.surrogate_for("PERSON", "Marbrint Morrowgale")

    assert first == again == "<PERSON_1>"
    assert other == "<PERSON_2>"


def test_two_people_with_the_same_surname_are_different_people():
    allocator = SurrogateAllocator()

    tokens = {
        allocator.surrogate_for("PERSON", name)
        for name in ("Ilse Vantreske", "Oren Vantreske", "Ilse Vantreske")
    }

    assert tokens == {"<PERSON_1>", "<PERSON_2>"}


@pytest.mark.parametrize(
    "variants",
    [
        ("José Ñandú", "JOSE NANDU", "jose nandu", "José  Ñandú"),
        ("Dr. Aurelio Quennet", "Aurelio Quennet", "AURELIO QUENNET"),
        ("Ｚｅｌｑｕｉｎ Vantreske", "Zelquin Vantreske"),
    ],
)
def test_unicode_case_spacing_and_titles_do_not_split_one_person(variants):
    allocator = SurrogateAllocator()

    assert {allocator.surrogate_for("PERSON", value) for value in variants} == {
        "<PERSON_1>"
    }


def test_structured_values_match_across_formatting():
    allocator = SurrogateAllocator()

    phones = {
        allocator.surrogate_for("PHONE_NUMBER", value)
        for value in ("555-0142", "(555) 0142", "555 0142")
    }
    records = {
        allocator.surrogate_for("MEDICAL_RECORD_NUMBER", value)
        for value in ("48-21337", "4821337", "48 21337")
    }
    emails = {
        allocator.surrogate_for("EMAIL_ADDRESS", value)
        for value in ("Clinic@Example.invalid", "clinic@example.invalid")
    }

    assert phones == {"<PHONE_1>"}
    assert records == {"<RECORD_ID_1>"}
    assert emails == {"<EMAIL_1>"}


def test_each_family_counts_on_its_own():
    allocator = SurrogateAllocator()

    assert allocator.surrogate_for("PERSON", "Ilse Vantreske") == "<PERSON_1>"
    assert allocator.surrogate_for("FACILITY", "Northgate Infirmary") == "<FACILITY_1>"
    assert allocator.surrogate_for("LOCATION", "Harrowdale") == "<LOCATION_1>"
    assert allocator.surrogate_for("PERSON", "Oren Quennet") == "<PERSON_2>"
    assert allocator.counts() == {"FACILITY": 1, "LOCATION": 1, "PERSON": 2}


def test_tokens_already_in_the_input_are_never_reused():
    allocator = SurrogateAllocator()
    allocator.reserve_existing("Seen with <PERSON_1> and <PERSON_3>; call <PHONE_1>.")

    assert allocator.surrogate_for("PERSON", "Ilse Vantreske") == "<PERSON_2>"
    assert allocator.surrogate_for("PERSON", "Oren Quennet") == "<PERSON_4>"
    assert allocator.surrogate_for("PHONE_NUMBER", "555-0142") == "<PHONE_2>"


def test_a_token_does_not_depend_on_the_value_or_cross_datasets():
    first, second = SurrogateAllocator(), SurrogateAllocator()

    assert first.surrogate_for("PERSON", "Ilse Vantreske") == "<PERSON_1>"
    assert second.surrogate_for("PERSON", "Oren Quennet") == "<PERSON_1>"
    assert second.surrogate_for("PERSON", "Ilse Vantreske") == "<PERSON_2>"
    assert "Vantreske" not in repr(first) and "Vantreske" not in str(first.counts())


def test_types_without_a_family_fall_back_to_redaction():
    allocator = SurrogateAllocator()

    assert allocator.surrogate_for("DATE_TIME", "14 March 2021") is None
    assert allocator.surrogate_for("PERSON", "   ") is None


def test_generated_tokens_never_collide_and_all_match_the_pattern():
    rng = random.Random(2024)
    families = sorted(surrogates.SURROGATE_FAMILIES)
    allocator = SurrogateAllocator()
    assigned = {}
    for _ in range(2_000):
        entity_type = rng.choice(families)
        value = "".join(
            rng.choice("abcdefghij0123456789 -.") for _ in range(rng.randint(1, 12))
        )
        token = allocator.surrogate_for(entity_type, value)
        if token is None:
            continue
        key = (
            surrogates.SURROGATE_FAMILIES[entity_type],
            surrogates.normalize_value(entity_type, value),
        )
        assert assigned.setdefault(key, token) == token
        assert SURROGATE_PATTERN.fullmatch(token)
    assert len(set(assigned.values())) == len(assigned)


def test_the_pattern_separates_adjacent_tokens_and_ignores_lookalikes():
    text = "<PERSON_1><PERSON_2> met <FACILITY_10>; not PERSON_1 or <PERSON_> or <person_1>."

    assert [m.group(0) for m in SURROGATE_PATTERN.finditer(text)] == [
        "<PERSON_1>",
        "<PERSON_2>",
        "<FACILITY_10>",
    ]


# -- the sanitizers, end to end with the test detectors --------------------------------------


def _tokens(text):
    return [match.group(0) for match in SURROGATE_PATTERN.finditer(text)]


def test_a_note_keeps_one_token_per_person_and_separates_shared_surnames():
    from services.text_anonymization import anonymize_clinical_text

    result = anonymize_clinical_text(
        "Patient Rahul Sharma attended. Rahul Sharma was discharged. "
        "Patient Priya Sharma visited."
    )

    tokens = _tokens(result["anonymized_text"])
    assert tokens[0] == tokens[1] != tokens[2]
    assert "Sharma" not in result["anonymized_text"]
    assert result["residual_phi_categories"] == {}


def test_a_token_already_in_the_note_is_kept_and_never_reused():
    from services.text_anonymization import anonymize_clinical_text

    result = anonymize_clinical_text("Patient Rahul Sharma met <PERSON_1> earlier.")

    assert "<PERSON_1>" in result["anonymized_text"]
    assert "Rahul Sharma" not in result["anonymized_text"]
    assert _tokens(result["anonymized_text"]).count("<PERSON_1>") == 1
    assert result["residual_phi_categories"] != {}
