"""港股交易日与已核实的证券实体边界。"""
import pandas as pd

from project_paths import data_dir


def hk_ifind_code(code):
    if code == "00823.HK":
        return code
    return str(int(code.split(".")[0])).zfill(4) + ".HK"


def hk_calendar():
    return pd.DatetimeIndex(pd.read_csv(data_dir("hk2") / "reference/calendar.csv", parse_dates=["trade_date"])["trade_date"])


def hk_listing_dates():
    # 本表只收录已核实的实体边界，不代表全市场上市主数据。
    identity = pd.read_csv(data_dir("hk2") / "reference/security_identity.csv", dtype={"windcode": str}, parse_dates=["listing_date"])
    return identity.set_index("windcode")["listing_date"]


def filter_hk_prices(frame):
    result = frame.copy()
    result["trade_date"] = pd.to_datetime(result["trade_date"])
    result["wind_code"] = result["wind_code"].str.split(".").str[0].str.zfill(5) + ".HK"
    listing = result["wind_code"].map(hk_listing_dates())
    valid = result["trade_date"].isin(hk_calendar())
    valid &= listing.isna() | result["trade_date"].ge(listing)
    return result.loc[valid]


def clean_hk_price_file(path):
    frame = pd.read_csv(path, dtype={"wind_code": str})
    clean = filter_hk_prices(frame)
    removed = len(frame) - len(clean)
    if removed:
        clean.to_csv(path, index=False, date_format="%Y-%m-%d")
    return removed


def filter_hk_valuation(frame, code):
    result = frame.copy()
    result["date"] = pd.to_datetime(result["date"])
    valid = result["date"].le(hk_calendar()[-1])
    listings = hk_listing_dates()
    if code in listings.index:
        valid &= result["date"].ge(listings.loc[code])
    return result.loc[valid]
