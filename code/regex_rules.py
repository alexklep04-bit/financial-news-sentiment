# All regular expressions used in the project are kept here so they are easy to check.
import re

# ---------------------------------------------------------------------------
# 1. finding the five firms in text
# ---------------------------------------------------------------------------
# (?:...) groups so pandas does not complain about capture groups
FIRM_PATTERNS = {
    "JPMorgan": r"\bJ\.?\s?P\.?\s?Morgan\b|\bJPMorgan\b|\bJamie Dimon\b|\$JPM\b",
    "Goldman Sachs": r"\bGoldman\b|\$GS\b|\bLloyd Blankfein\b|\bDavid Solomon\b",
    "BlackRock": r"\bBlack[Rr]ock\b|\bLarry Fink\b|\$BLK\b",
    "Vanguard": r"(?<!American )\bVanguard\b(?!\s(?:Natural|Health|Cellular|Car|Minerals))",
    "Citadel": r"\bCitadel\b|\bKen Griffin\b",
}

FIRM_NAME_RE = r"(?:J\.?\s?P\.?\s?Morgan|JPMorgan(?: Chase)?|Goldman(?: Sachs)?|BlackRock|Vanguard|Citadel)"

# ---------------------------------------------------------------------------
# 2. headlines that mention the firm but are not news ABOUT the firm
# ---------------------------------------------------------------------------
NOISE_RULES = [
    # GuruFocus style portfolio lists: "XYZ Advisors Buys iShares ..., Vanguard ..., Sells ..."
    ("portfolio_list", re.compile(r"\b(?:Buys|Sells)\b.*(?:\.\.\.|…)\s*$|\b(?:Buys|Sells)\b[^,]*,[^,]*,")),
    # the bank is acting as a broker / analyst for some other company
    ("analyst_call", re.compile(
        r"\b(?:upgrade|downgrade|initiat|reiterat|maintain|resum|assum)\w*\b"
        r"|\bprice target\b|\bPT\b"
        r"|\b(?:at|by|from)\s+" + FIRM_NAME_RE + r"\b"
        r"|^" + FIRM_NAME_RE + r"(?:'s)?:?\s+(?:sees|says|cautious|bullish|bearish|likes|positive|negative|picks"
        r"|turns|warms|swaps|expects|predicts|recommends|favors|prefers|thinks|on)\b"
        r"|" + FIRM_NAME_RE + r"(?:'s)?\s+(?:Conviction|Buy|Sell)\s+List", re.I)),
    # the firm's name is just the name of an event
    ("event", re.compile(r"\b(?:conference|forum|symposium|summit)\b", re.I)),
    # fund / ETF products (iShares is BlackRock's ETF brand)
    ("fund_product", re.compile(
        r"\biShares\b|" + FIRM_NAME_RE + r"(?:\s+(?:[A-Z0-9&][\w&\.\-']*|and|of|for)){0,7}\s+(?:Fund|Funds|ETF|ETFs|Trust|Portfolio|Index)\b")),
]


def is_noise_headline(headline, firm=None):
    """returns the name of the first rule that fires, or '' if the headline looks fine"""
    for name, rule in NOISE_RULES:
        if rule.search(headline):
            return name
    return ""


# ---------------------------------------------------------------------------
# 3. normalisation for the bag-of-words models
# ---------------------------------------------------------------------------
# PhraseBank sentences are full of amounts like "EUR 12.3 mn" or "5.2 %"
CURRENCY = r"(?:EUR|USD|SEK|GBP|NOK|DKK|RUB|JPY|CHF|\$|€|£)"
MAGNITUDE = r"(?:mn|m|million|bn|billion|b|thousand|k)"
NUM = r"\d+(?:[.,]\d+)*"

MONEY_RE = re.compile(rf"{CURRENCY}\s?{NUM}(?:\s?{MAGNITUDE})?\b|{NUM}\s?{MAGNITUDE}?\s?{CURRENCY}", re.I)
PCT_RE = re.compile(rf"{NUM}\s?(?:%|percent|pct)", re.I)
YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")
NUM_RE = re.compile(NUM)
QUARTER_RE = re.compile(r"\b(?:Q[1-4]|(?:first|second|third|fourth)[- ]quarter)\b", re.I)


def normalize(text):
    t = text
    t = PCT_RE.sub(" _pct_ ", t)
    t = MONEY_RE.sub(" _money_ ", t)
    t = YEAR_RE.sub(" _year_ ", t)
    t = QUARTER_RE.sub(" _quarter_ ", t)
    t = NUM_RE.sub(" _num_ ", t)
    t = re.sub(r"\s+", " ", t)
    return t.strip().lower()


# ---------------------------------------------------------------------------
# 4. tweets
# ---------------------------------------------------------------------------
URL_RE = re.compile(r"https?://\S+")
CASHTAG_RE = re.compile(r"\$([A-Za-z]{1,6})\b")


def clean_tweet(t):
    t = URL_RE.sub("", t)
    t = CASHTAG_RE.sub(r"\1", t)          # $JPM -> JPM
    t = re.sub(r"\s+", " ", t)
    return t.strip()


if __name__ == "__main__":
    # quick sanity checks
    print(normalize("Operating profit rose to EUR 13.1 mn from EUR 8.7 mn in the corresponding period in 2007 , up 51 % ."))
    for h in ["KBR cut to Sell at Goldman as earnings growth challenged",
              "Goldman Sachs Profit Falls 58% on Trading Slump",
              "JPMorgan sees unit revenue upside at Spirit Airlines",
              "Vanguard lowers fees on 12 ETFs",
              "Diversified Trust Co Buys iShares Core U.S. ...",
              "BlackRock Enhanced Capital and Income Fund announces $0.1 distribution",
              "Medtechs At JP Morgan Take Different Tacks"]:
        print(f"{is_noise_headline(h) or 'OK':15s} {h}")
    print(re.search(FIRM_PATTERNS["Vanguard"], "American Vanguard (AVD) Soars"))
