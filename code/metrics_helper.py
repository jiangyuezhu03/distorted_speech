from jiwer import process_characters


def char_mer(reference, hypothesis):
    """
    MER = (S + D + I) / (S + D + I + C)
    where:
        S = substitutions
        D = deletions
        I = insertions
        C = correct (hits)
    """
    # If lists, join into one string (corpus-level MER)
    if isinstance(reference, list):
        reference = " ".join(reference)
    if isinstance(hypothesis, list):
        hypothesis = " ".join(hypothesis)

    result = process_characters(reference, hypothesis)

    total_errors = result.substitutions + result.deletions + result.insertions
    total_compared = total_errors + result.hits

    if total_compared == 0:
        return 0.0  # handle edge case: both ref and hyp are empty
    return total_errors / total_compared
