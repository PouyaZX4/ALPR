from typing import Dict, Any

PERSIAN_LETTERS = set("ابپتثجدسصطعقلمنوهیژآچشظغفکگ")
PERSIAN_DIGITS = "۰۱۲۳۴۵۶۷۸۹"
ENGLISH_DIGITS = "0123456789"
EN_TO_FA = str.maketrans("".join(ENGLISH_DIGITS), "".join(PERSIAN_DIGITS))


def normalize_persian_digits(text: str) -> str:
    return text.translate(EN_TO_FA)


def format_iranian_plate(corrected_text: str) -> Dict[str, Any]:
    """
    Takes true Left-to-Right plate (e.g. '۵۷ص۴۲۵۶۷')
    and outputs:
    Left: 57 | Letter: ص | Mid: 425 | Iran: 67
    Formatted: [57] [ص] [425] - ایران 67
    """
    if not corrected_text:
        return {
            "formatted": "UNKNOWN",
            "badge": "UNKNOWN",
            "left_2": "",
            "letter": "",
            "mid_3": "",
            "iran_code": "",
            "is_valid": False
        }

    cleaned = normalize_persian_digits(corrected_text.strip().replace(" ", "").replace("-", ""))

    # Locate letter
    letter_match = None
    for idx, ch in enumerate(cleaned):
        if ch in PERSIAN_LETTERS:
            letter_match = (idx, ch)
            break

    all_digits = [c for c in cleaned if c in PERSIAN_DIGITS]

    # Standard format: 2 digits + 1 letter + 3 digits + 2 digits (e.g. ۵۷ص۴۲۵۶۷)
    if letter_match and len(all_digits) >= 7:
        letter_idx, letter = letter_match
        digits_before = [c for c in cleaned[:letter_idx] if c in PERSIAN_DIGITS]
        digits_after = [c for c in cleaned[letter_idx + 1:] if c in PERSIAN_DIGITS]

        if len(digits_before) == 2 and len(digits_after) >= 5:
            left_2 = "".join(digits_before)
            mid_3 = "".join(digits_after[:3])
            iran_code = "".join(digits_after[3:5])
        else:
            left_2 = "".join(all_digits[:2])
            mid_3 = "".join(all_digits[2:5])
            iran_code = "".join(all_digits[5:7])

        # Unicode Left-to-Right Embedding (\u202A ... \u202C) prevents terminal BiDi flipping
        formatted_terminal = f"\u202A[{left_2}] [{letter}] [{mid_3}] - ایران {iran_code}\u202C"
        badge_html = f"{left_2} {letter} {mid_3} | ایران {iran_code}"

        return {
            "formatted": formatted_terminal,
            "badge": badge_html,
            "left_2": left_2,
            "letter": letter,
            "mid_3": mid_3,
            "iran_code": iran_code,
            "is_valid": True
        }

    return {
        "formatted": cleaned,
        "badge": cleaned,
        "left_2": "".join(all_digits[:2]) if len(all_digits) >= 2 else "",
        "letter": letter_match[1] if letter_match else "",
        "mid_3": "".join(all_digits[2:]) if len(all_digits) > 2 else "",
        "iran_code": "",
        "is_valid": False
    }