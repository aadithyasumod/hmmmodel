"""
universe_loader.py
Stock universe options for both NSE (.NS) and BSE (.BO):

  NSE tier
  ─────────────────────────────────────────────────────────
  • NIFTY 50    (~50   stocks)  static list
  • NIFTY 500   (~500  stocks)  static list
  • All NSE     (~1800 stocks)  live fetch from NSE CSV

  BSE tier
  ─────────────────────────────────────────────────────────
  • SENSEX 30   (~30   stocks)  static list
  • BSE 500     (~500  stocks)  static list  (dual-listed BSE symbols)
  • All BSE     (~5000 stocks)  live fetch from BSE API → scrip-code tickers
"""

import requests
import pandas as pd
from io import StringIO
import warnings

# ──────────────────────────────────────────────────────────────────────────────
# NSE static lists
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
    "ZYDUSLIFE", "AARTIIND", "AAVAS", "ABBOTINDIA",
    "ABCAPITAL", "ABFRL", "AEGISLOG", "AFFLE", "AJANTPHARM",
    "AKZOINDIA", "AMARAJABAT", "AMBER", "ASHOKLEY",
    "BAJAJELEC", "BAJAJHLDNG", "BALRAMCHIN", "BATAINDIA", "BDL", "BEML", "BHEL",
    "BIKAJI", "BLUEDART", "BLUESTARCO", "BSOFT",
    "CEATLTD", "CENTRALBK", "CENTURYTEX", "CHAMBLFERT", "CHEMPLASTS",
    "CRAFTSMAN", "DATAPATTNS", "DCMSHRIRAM",
    "EDELWEISS", "ENGINERSIN", "EPL",
    "EQUITASBNK", "ESABINDIA", "FINEORG",
    "FLUOROCHEM", "GARFIBRES",
    "GLENMARK", "GNFC", "GODREJIND", "GPIL", "GRINDWELL",
    "GSFC", "GSPL", "HBLPOWER", "HFCL", "HIKAL",
    "HUDCO", "IBULHSGFIN", "ICICIGI", "IDBI",
    "IGARASHI", "IIFL", "IPCALAB", "ISEC", "JBCHEPHARM",
    "JKCEMENT", "JKPAPER", "JMFINANCIL", "JSWENERGY", "JTEKTINDIA",
    "KANSAINER", "KARURVYSYA", "KFINTECH", "KIRLOSENG",
    "KNRCON", "KRBL", "LAXMIMACH", "LEMONTREE",
    "LINDEINDIA", "MAHABANK", "MANAPPURAM", "MANKIND",
    "MEDANTA", "METROBRAND", "MIDHANI",
    "MOTILALOFS", "NATCOPHARM", "NAVINFLUOR",
    "NBCC", "NCC", "NLCINDIA",
    "OLECTRA", "PCBL",
    "PHOENIXLTD", "PIDILITIND", "PNCINFRA", "POONAWALLA",
    "PRESTIGE", "PRINCEPIPE", "PSPPROJECT",
    "RADICO", "RAILTEL", "RAIN", "RITES",
    "ROSSARI", "ROUTE",
    "SCHNEIDER", "SKFINDIA", "SOBHA",
    "SONATSOFTW", "SPARC", "SPICEJET",
    "STARHEALTH", "STLTECH", "SUBROS", "SUMICHEM",
    "TANLA", "TATACHEM", "TATACOMM", "TATAELXSI",
    "TATAPOWER", "TEAMLEASE",
    "TIMKEN", "TINPLATE", "TORNTPOWER",
    "TVSMOTOR", "UJJIVANSFB", "UNIPARTS",
    "UTIAMC", "VAKRANGEE",
    "VESUVIUS", "VINATIORGA", "VSTIND", "WABCOINDIA",
    "WELCORP", "WELSPUNLIV", "WESTLIFE",
    "YESBANK", "ZEEL", "ZOMATO", "ZYDUSWELL",
]


# ──────────────────────────────────────────────────────────────────────────────
# BSE static lists
# ──────────────────────────────────────────────────────────────────────────────

# SENSEX 30 components — same trading symbols, .BO suffix
SENSEX_30_SYMBOLS = [
    "RELIANCE", "TCS", "HDFCBANK", "ICICIBANK", "INFY",
    "HINDUNILVR", "ITC", "SBIN", "BHARTIARTL", "KOTAKBANK",
    "BAJFINANCE", "LT", "AXISBANK", "ASIANPAINT", "MARUTI",
    "SUNPHARMA", "TITAN", "ULTRACEMCO", "WIPRO", "NTPC",
    "M&M", "BAJAJFINSV", "POWERGRID", "INDUSINDBK", "TATAMOTORS",
    "TATASTEEL", "JSWSTEEL", "ADANIPORTS", "HCLTECH", "TECHM",
]

# BSE-only or BSE-primary symbols not in the NSE 500 list above.
# Most large-caps are dual-listed; this covers mid/small caps
# that trade primarily or only on BSE.
BSE_ONLY_EXTRA = [
    # Finance / NBFC
    "SHRIRAMCIT", "MAHINDCIE", "HGINFRA", "BGRENERGY",
    "GENESYS", "TEJASNET", "ANDHRSUGAR", "SUNFLAG",
    # Industrials
    "GOKEX", "VARDHACRLC", "WSTCSTPAPR", "GPPL",
    "KAKATCEM", "MAGNUM", "JTEKT", "KIRIINDUS",
    # Textiles
    "SPENTEX", "RSWM", "RAYMOND", "DONEAR", "GTNTEX",
    "GRASIM", "VARDHMAN", "NITIN", "ALOKTEXT",
    # Chemicals
    "GHCL", "IGPL", "POCL", "JAYAGROGN", "DHARSUGAR",
    "ORIENTCEM", "DECCANCE", "RAJRATAN",
    # Consumer
    "JYOTHYLAB", "KANSAINER", "NAVNETEDUL", "GITANJALI",
    "CAMLINFINE", "HSIL", "BAJAJCON", "JUBLPHARMA",
    # Real Estate
    "SOBHA", "MAHLIFE", "SUNTECK", "BRIGADE", "KOLTEPATIL",
    # Energy
    "TORNTPOWER", "RPOWER", "CESC", "JPPOWER",
    "KPCL", "MAHAGENCO",
    # Banking
    "DCBBANK", "LAKSHVILAS", "TMVFINANCE", "REPCO",
    "UJJIVAN", "ESAFSFB",
    # IT / Tech
    "XCHANGING", "RATEGAIN", "NEWGEN", "INTELLECT",
    "SAKSOFT", "TATAELXSI",
    # Pharma
    "SOLARA", "GRANULES", "SEQUENT", "VGUARD",
    "LAURUSLABS", "CAPLIN", "SHILPAMED",
    # Metals
    "RATNAMANI", "SANDUMA", "GMDC", "MOIL",
    "NSLNISP", "MSTEEL",
]


# ──────────────────────────────────────────────────────────────────────────────
# NSE functions
# ──────────────────────────────────────────────────────────────────────────────

def get_nifty50_tickers() -> list[str]:
    return [f"{s}.NS" for s in NIFTY_50_SYMBOLS]


def get_nifty500_tickers() -> list[str]:
    combined = NIFTY_50_SYMBOLS + [s for s in NIFTY_500_EXTRA if s not in NIFTY_50_SYMBOLS]
    return [f"{s}.NS" for s in combined]


def get_full_nse_universe() -> list[str]:
    """
    Download the complete NSE equity list from NSE's official CSV.
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
        sym_col = next((c for c in df.columns if "SYMBOL" in c.upper()), None)
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


# ──────────────────────────────────────────────────────────────────────────────
# BSE functions
# ──────────────────────────────────────────────────────────────────────────────

def get_sensex30_tickers() -> list[str]:
    return [f"{s}.BO" for s in SENSEX_30_SYMBOLS]


def get_bse500_tickers() -> list[str]:
    """
    BSE 500 tier: Sensex 30 + same companies as NIFTY 500
    (virtually all are dual-listed) + BSE-primary extras.
    """
    seen = set(SENSEX_30_SYMBOLS)
    combined = list(SENSEX_30_SYMBOLS)

    for s in NIFTY_500_EXTRA:
        if s not in seen:
            combined.append(s)
            seen.add(s)

    for s in BSE_ONLY_EXTRA:
        if s not in seen:
            combined.append(s)
            seen.add(s)

    return [f"{s}.BO" for s in combined]


def get_full_bse_universe() -> list[str]:
    """
    Download all active BSE equity scrips from BSE's API.
    Returns tickers as '{SCRIP_CODE}.BO' (most reliable format for yfinance).
    Falls back to BSE 500 static list on any failure.
    """
    try:
        url = (
            "https://api.bseindia.com/BseIndiaAPI/api/ListofScripData/w"
            "?Group=&Scripcode=&industry=&segment=Equity&status=Active"
        )
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept": "application/json, text/plain, */*",
            "Referer": "https://www.bseindia.com/",
        }
        resp = requests.get(url, headers=headers, timeout=30)
        resp.raise_for_status()

        data = resp.json()

        # BSE API returns a list of dicts; scrip code is in 'SCRIP_CD'
        codes = []
        for item in data:
            code = item.get("SCRIP_CD") or item.get("scripCd") or item.get("scrip_cd")
            if code:
                codes.append(str(int(code)))   # ensure integer string, no decimals

        if not codes:
            raise ValueError("No scrip codes found in BSE API response")

        return [f"{c}.BO" for c in codes]

    except Exception as exc:
        warnings.warn(
            f"[universe_loader] BSE API fetch failed ({exc}). "
            "Falling back to BSE 500 static list."
        )
        return get_bse500_tickers()


# ──────────────────────────────────────────────────────────────────────────────
# Registry
# ──────────────────────────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────────────────────────
# Combined NSE + BSE universe functions
# ──────────────────────────────────────────────────────────────────────────────

def get_combined_large_cap() -> list[str]:
    """NIFTY 50 (.NS) + SENSEX 30 (.BO) — ~80 tickers total."""
    return get_nifty50_tickers() + get_sensex30_tickers()


def get_combined_500() -> list[str]:
    """NSE 500 (.NS) + BSE 500 (.BO) — ~700 tickers total."""
    return get_nifty500_tickers() + get_bse500_tickers()


def get_combined_all() -> list[str]:
    """All NSE (~1800, .NS) + All BSE (~5000, .BO) — ~6800 tickers total."""
    return get_full_nse_universe() + get_full_bse_universe()


UNIVERSE_OPTIONS = {
    "Large Cap   (NIFTY 50 + SENSEX 30,  ~80 stocks,   ~2 min)":   get_combined_large_cap,
    "Mid/Large   (NSE 500 + BSE 500,    ~700 stocks,  ~10 min) ★": get_combined_500,
    "All Markets (All NSE + All BSE,   ~6800 stocks,  ~60 min)":   get_combined_all,
}
