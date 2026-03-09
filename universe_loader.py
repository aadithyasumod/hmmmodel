"""
universe_loader.py
Fetches the NSE stock universe with tiered options:
  - NIFTY 50   (~50  stocks, fastest)
  - NIFTY 500  (~500 stocks, recommended)
  - All NSE    (~1800+ stocks, slow — 10-15 min)
"""

import requests
import pandas as pd
from io import StringIO
import warnings

# ──────────────────────────────────────────────────────────────────────────────
# Static curated lists (used as fallback if NSE API is unreachable)
# ──────────────────────────────────────────────────────────────────────────────

NIFTY_50_SYMBOLS = [
    "ADANIENT", "ADANIPORTS", "APOLLOHOSP", "ASIANPAINT", "AXISBANK",
    "BAJAJ-AUTO", "BAJFINANCE", "BAJAJFINSV", "BPCL", "BHARTIARTL",
    "BRITANNIA", "CIPLA", "COALINDIA", "DIVISLAB", "DRREDDY",
    "EICHERMOT", "GRASIM", "HCLTECH", "HDFCBANK", "HDFCLIFE",
    "HEROMOTOCO", "HINDALCO", "HINDUNILVR", "ICICIBANK", "INDUSINDBK",
    "INFY", "ITC", "JSWSTEEL", "KOTAKBANK", "LT",
    "LTIM", "M&M", "MARUTI", "NESTLEIND", "NTPC",
    "ONGC", "POWERGRID", "RELIANCE", "SBILIFE", "SBIN",
    "SHRIRAMFIN", "SUNPHARMA", "TATACONSUM", "TATAMOTORS", "TATASTEEL",
    "TCS", "TECHM", "TITAN", "ULTRACEMCO", "WIPRO",
]

NIFTY_500_EXTRA = [
    "ABB", "ACC", "AIAENG", "ALKEM", "AMBUJACEM", "ANGELONE", "APLAPOLLO",
    "ASTRAL", "ATGL", "ATUL", "AUBANK", "AUROPHARMA", "BALKRISIND",
    "BANDHANBNK", "BANKINDIA", "BAYERCROP", "BERGEPAINT", "BIOCON",
    "BOSCHLTD", "BSE", "CAMS", "CANFINHOME", "CANBK", "CDSL", "CHOLAFIN",
    "COFORGE", "COLPAL", "CONCOR", "COROMANDEL", "CUMMINSIND", "CYIENT",
    "DABUR", "DEEPAKNTR", "DELHIVERY", "DIXON", "DLF", "DMART",
    "ELGIEQUIP", "EMAMILTD", "ESCORTS", "EXIDEIND", "FACT", "FEDERALBNK",
    "GAIL", "GLAND", "GMRINFRA", "GODREJCP", "GODREJPROP", "GRANULES",
    "HAL", "HAVELLS", "HINDPETRO", "HONAUT", "ICICIPRULI", "IDFCFIRSTB",
    "IGL", "INDHOTEL", "INDUSTOWER", "IOC", "IRCTC", "IRFC", "JINDALSTEL",
    "JUBLFOOD", "KALYANKJIL", "KPITTECH", "LALPATHLAB", "LICHSGFIN",
    "LICI", "LTTS", "LUPIN", "MARICO", "MCDOWELL-N", "MCX", "METROPOLIS",
    "MFSL", "MGL", "MPHASIS", "MRF", "MUTHOOTFIN", "NHPC", "NMDC",
    "OBEROIRLTY", "OFSS", "PAGEIND", "PEL", "PERSISTENT", "PFC", "PIIND",
    "PNB", "POLYCAB", "PVRINOX", "RAJESHEXPO", "RAMCOCEM", "REC",
    "SAIL", "SBICARD", "SHREECEM", "SIEMENS", "SJVN", "SRF",
    "SUPREMEIND", "SYNGENE", "TIINDIA", "TORNTPHARM", "TORNTPOWER",
    "TRENT", "TRIDENT", "TTKPRESTIG", "UNIONBANK", "UBL", "VOLTAS",
    "ZYDUSLIFE", "3MINDIA", "AARTIIND", "AAVAS", "ABBOTINDIA",
    "ABCAPITAL", "ABFRL", "AEGISLOG", "AFFLE", "AJANTPHARM",
    "AKZOINDIA", "AMARAJABAT", "AMBER", "ANGELONE", "ANURAS",
    "APTUS", "ARVINDFASN", "ASAL", "ASHOKLEY", "ASIANPAINT",
    "ATUL", "AVALON", "AVANTIFEED", "BAJAJELEC", "BAJAJHLDNG",
    "BALRAMCHIN", "BATAINDIA", "BDL", "BEML", "BHEL",
    "BIKAJI", "BLUEDART", "BLUESTARCO", "BOMBAYBURMAH", "BSOFT",
    "CEATLTD", "CENTRALBK", "CENTURYTEX", "CHAMBLFERT", "CHEMPLASTS",
    "CLEAN", "CRAFTSMAN", "DATAPATTNS", "DBSCareer", "DCMSHRIRAM",
    "EDELWEISS", "EMKAY", "ENBEEFOODS", "ENGINERSIN", "EPL",
    "EQUITASBNK", "EROSMEDIA", "ESABINDIA", "ESTER", "FINEORG",
    "FINPIPE", "FLUOROCHEM", "GANDALF", "GARFIBRES", "GEL",
    "GLENMARK", "GNFC", "GODREJIND", "GPIL", "GRINDWELL",
    "GSFC", "GSPL", "HBLPOWER", "HFCL", "HIKAL",
    "HLEGLAS", "HUDCO", "IBULHSGFIN", "ICICIGI", "IDBI",
    "IGARASHI", "IIFL", "IPCALAB", "ISEC", "JBCHEPHARM",
    "JKCEMENT", "JKPAPER", "JMFINANCIL", "JSWENERGY", "JTEKTINDIA",
    "JUBLINGREA", "KANSAINER", "KARURVYSYA", "KFINTECH", "KIRLOSENG",
    "KNRCON", "KRBL", "KSCL", "LAXMIMACH", "LEMONTREE",
    "LINDEINDIA", "LLOYDSENGG", "MAHABANK", "MANAPPURAM", "MANKIND",
    "MASFIN", "MEDANTA", "METROBRAND", "MHRIL", "MIDHANI",
    "MOLDTKPAC", "MOTILALOFS", "MSSL", "NATCOPHARM", "NAVINFLUOR",
    "NBCC", "NCC", "NEOGEN", "NLCINDIA", "NSLNISP",
    "NUVOCO", "OLECTRA", "ORIENTELEC", "PCBL", "PGHL",
    "PHOENIXLTD", "PIDILITIND", "PLSOFTWARE", "PNCINFRA", "POONAWALLA",
    "PRESTIGE", "PRINCEPIPE", "PRISM", "PSPPROJECT", "PTCIL",
    "QUESS", "RADICO", "RAILTEL", "RAIN", "RITES",
    "ROSSARI", "ROUTE", "RPGLIFE", "RPOWER", "RTNPOWER",
    "SAGCEM", "SAPPHIRE", "SCHNEIDER", "SEQUENT", "SHANKARA",
    "SHARDACROP", "SHYAMMETL", "SKFINDIA", "SOBHA", "SOLARA",
    "SONATSOFTW", "SPARC", "SPENCERS", "SPICEJET", "SSWL",
    "STARHEALTH", "STLTECH", "SUBROS", "SUMICHEM", "SUNCLAYLTD",
    "SUNDARAM", "SUNDRMBRAK", "SURYAROSNI", "SWSOLAR", "SYMPHONY",
    "TANLA", "TASTYBITE", "TATACHEM", "TATACOMM", "TATAELXSI",
    "TATAINVEST", "TATAPOWER", "TCNSBRANDS", "TEAMLEASE", "TEJASNET",
    "TIMKEN", "TINPLATE", "TIPSINDLTD", "TOBACCOIND", "TRITURBINE",
    "TVSSRICHAK", "TVSMOTOR", "TVSSUPPLY", "UJJIVANSFB", "UNIPARTS",
    "USHAMART", "UTIAMC", "UTTAMSUGAR", "VAIBHAVGBL", "VAKRANGEE",
    "VESUVIUS", "VINATIORGA", "VISHNU", "VSTIND", "WABCOINDIA",
    "WELCORP", "WELSPUNLIV", "WESTLIFE", "WINDLAS", "XCHANGING",
    "YESBANK", "ZEEL", "ZENTEC", "ZOMATO", "ZYDUSWELL",
]


def get_nifty50_tickers() -> list[str]:
    return [f"{s}.NS" for s in NIFTY_50_SYMBOLS]


def get_nifty500_tickers() -> list[str]:
    combined = NIFTY_50_SYMBOLS + [s for s in NIFTY_500_EXTRA if s not in NIFTY_50_SYMBOLS]
    return [f"{s}.NS" for s in combined]


def get_full_nse_universe() -> list[str]:
    """
    Download the complete NSE equity list from NSE's official source.
    Falls back to the static NIFTY 500 list on any network failure.
    """
    try:
        url = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Referer": "https://www.nseindia.com/",
        }
        resp = requests.get(url, headers=headers, timeout=20)
        resp.raise_for_status()

        df = pd.read_csv(StringIO(resp.text))
        # Locate symbol column robustly
        sym_col = next(
            (c for c in df.columns if "SYMBOL" in c.upper()), None
        )
        if sym_col is None:
            raise ValueError("SYMBOL column not found in NSE CSV")

        symbols = df[sym_col].dropna().str.strip().tolist()
        return [f"{s}.NS" for s in symbols if s]

    except Exception as exc:
        warnings.warn(
            f"[universe_loader] NSE fetch failed ({exc}). "
            "Falling back to NIFTY 500 static list."
        )
        return get_nifty500_tickers()


UNIVERSE_OPTIONS = {
    "NIFTY 50  (~50 stocks, ~1 min)": get_nifty50_tickers,
    "NIFTY 500 (~500 stocks, ~5 min) ★ Recommended": get_nifty500_tickers,
    "All NSE  (~1800 stocks, ~15 min)": get_full_nse_universe,
}
