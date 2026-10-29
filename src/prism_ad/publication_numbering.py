"""Current publication labels for unchanged scientific result identifiers.

Scientific identifiers remain stable in analysis artifacts. These mappings
control manuscript-facing table numbers and supplementary rendering aliases.
"""
SUPPLEMENTARY_TABLE_NUMBERS = {
    1: 1, 2: 2, 3: 3, 4: 5, 5: 8, 6: 10,
    7: 12, 8: 6, 9: 9, 10: 4, 11: 11, 12: 7,
}
SUPPLEMENTARY_FIGURE_IDS = {
    "optimization": "S1",
    "age": "S2",
    "prs": "S3",
}


def supplementary_table_number(scientific_id):
    """Return the display number without changing the scientific output ID."""
    value = int(scientific_id)
    if value not in SUPPLEMENTARY_TABLE_NUMBERS:
        raise ValueError("Unknown supplementary table scientific identifier")
    return SUPPLEMENTARY_TABLE_NUMBERS[value]


def supplementary_figure_kind(publication_id):
    """Return the scientific subject assigned to S1, S2 or S3."""
    matches = [key for key, value in SUPPLEMENTARY_FIGURE_IDS.items()
               if value == str(publication_id)]
    if len(matches) != 1:
        raise ValueError("Unknown supplementary figure publication identifier")
    return matches[0]
