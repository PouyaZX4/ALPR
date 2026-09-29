import re

PERSIAN_TO_ENGLISH = str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "01234567890123456789")
ENGLISH_TO_PERSIAN = str.maketrans("0123456789", "۰۱۲۳۴۵۶۷۸۹")

def to_persian_digits(s: str) -> str:
    return str(s).translate(ENGLISH_TO_PERSIAN)

def to_english_digits(s: str) -> str:
    return str(s).translate(PERSIAN_TO_ENGLISH)

def format_iranian_plate(raw_text: str) -> dict:
    if not raw_text or raw_text.strip() == "":
        return {"formatted": "UNKNOWN", "raw": raw_text, "part1": "", "letter": "", "part2": "", "prov": ""}

    # 1. Clean unicode control marks and spaces
    cleaned = re.sub(r'[\u200B-\u200F\u202A-\u202E\uFEFF\[\]\(\)\-_]', '', raw_text).strip()
    ascii_clean = to_english_digits(cleaned)

    # 2. Extract digits and Persian letter
    # Standard format: 2 digits (part1) + 1 letter + 3 digits (part2) + 2 digits (province)
    digits = re.findall(r'\d+', ascii_clean)
    letters = re.findall(r'[^\d\s\-_]+', ascii_clean)
    
    # Remove 'ایران' or 'IR' if captured in letters
    letters = [l for l in letters if l not in ('ایران', 'IR', 'ir', 'iran')]
    letter = letters[0] if letters else ""

    p1, p2, prov = "", "", ""

    if len(digits) >= 3:
        p1 = digits[0][:2]
        p2 = digits[1][:3]
        prov = digits[2][:2]
    elif len(digits) == 1 and len(digits[0]) >= 7:
        d = digits[0]
        p1, p2, prov = d[:2], d[2:5], d[5:7]
    elif len(digits) == 2:
        p1 = digits[0][:2]
        p2 = digits[1][:3]
        prov = digits[1][3:5] if len(digits[1]) >= 5 else "--"

    if p1 and p2 and letter:
        # EXACT sequence: [ ۵۷ ] - [ ۴۲۵ ص ] - [ ۶۷ IR ]
        formatted = f"{to_persian_digits(p1)} - {to_persian_digits(p2)} {letter} - {to_persian_digits(prov)} IR"
    else:
        formatted = to_persian_digits(ascii_clean)

    return {
        "formatted": formatted,
        "raw": raw_text,
        "part1": p1,
        "letter": letter,
        "part2": p2,
        "prov": prov
    }