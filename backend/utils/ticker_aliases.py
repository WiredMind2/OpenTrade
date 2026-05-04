"""Ticker aliases used by lightweight news matching.

News providers often write company names instead of exchange symbols. Keep this
small and deterministic for the built-in popular ticker universe.
"""

TICKER_ALIASES = {
    "AAPL": ["Apple"],
    "ABBV": ["AbbVie"],
    "ABT": ["Abbott"],
    "ACN": ["Accenture"],
    "AMZN": ["Amazon"],
    "AVGO": ["Broadcom"],
    "BRK.B": ["Berkshire Hathaway", "Berkshire"],
    "COST": ["Costco"],
    "CSCO": ["Cisco"],
    "CVX": ["Chevron"],
    "GOOGL": ["Alphabet", "Google"],
    "HD": ["Home Depot"],
    "JNJ": ["Johnson & Johnson", "Johnson and Johnson"],
    "JPM": ["JPMorgan", "JPMorgan Chase"],
    "KO": ["Coca-Cola", "Coca Cola"],
    "LLY": ["Eli Lilly", "Lilly"],
    "MA": ["Mastercard"],
    "MCD": ["McDonald's", "McDonalds"],
    "META": ["Meta", "Facebook"],
    "MRK": ["Merck"],
    "MSFT": ["Microsoft"],
    "NVDA": ["Nvidia", "NVIDIA"],
    "PEP": ["PepsiCo", "Pepsi"],
    "PG": ["Procter & Gamble", "Procter and Gamble"],
    "TMO": ["Thermo Fisher"],
    "TSLA": ["Tesla"],
    "UNH": ["UnitedHealth", "UnitedHealth Group"],
    "V": ["Visa"],
    "WMT": ["Walmart", "Wal-Mart"],
    "XOM": ["Exxon", "Exxon Mobil", "ExxonMobil"],
}


def aliases_for_ticker(ticker: str, company_name: str | None = None) -> list[str]:
    aliases = list(TICKER_ALIASES.get(str(ticker).upper(), []))
    if company_name:
        aliases.append(company_name)
    return aliases
