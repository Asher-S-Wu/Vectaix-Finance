"""依据逐日端点证据生成缺失收益标签补丁，原始因子与行情保持不变。"""

import argparse
import json
from pathlib import Path
from urllib.parse import quote

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from . import HORIZONS


DEVELOPMENT_END = pd.Timestamp("2023-12-31")
PATCH_COLUMNS = ["date", "security_id", "horizon", "label_end", "old_fwd_return", "new_fwd_return",
                 "reason", "source_url", "source_issue_id", "source_listing_id", "source_stock_code",
                 "signal_adj_close_hkd", "endpoint_existing_adj_close", "endpoint_fx_to_hkd",
                 "endpoint_existing_raw_close", "endpoint_source_closing", "preceding_anchor_date",
                 "retrospective_following_anchor_date", "following_anchor_used_for_label",
                 "label_available_date", "identity_support_basis", "endpoint_currency", "endpoint_fx_source_url"]


def _positive(value):
    return pd.notna(value) and np.isfinite(value) and value > 0


def _true(value):
    return pd.notna(value) and value == True


def _verified_trade(row, issue, currency):
    return (_true(row.data_valid) and _true(row.identity_date_verified)
            and pd.notna(row.identity_source_issue_id) and row.identity_source_issue_id == issue
            and _true(row.quote_present) and _positive(row.volume) and _positive(row.raw_close)
            and row.currency == currency and _positive(row.fx_to_hkd)
            and (currency != "HKD" or row.fx_to_hkd == 1.))


def endpoint_label_patches(requests, features, bars, evidence, master, calendar, as_of, fx_rates=None):
    """生成独立标签补丁和拒绝原因，收益只使用既有信号价与既有端点价。

    following_anchor_date 是事后复核信息，不参与有效性、成熟时间或收益计算。
    当日身份依据必须包含同 IssueID 的端点记录及其之前的已核验成交观察。
    """
    as_of = pd.Timestamp(as_of)
    sessions = pd.DatetimeIndex(calendar)
    if sessions.has_duplicates or not sessions.is_monotonic_increasing:
        raise ValueError("交易日历必须唯一且按日期排序")
    for frame, key in [(requests, ["date", "security_id", "horizon"]),
                       (features, ["security_id", "date"]), (bars, ["security_id", "date"]),
                       (evidence, ["security_id", "date"]), (master, ["security_id"])]:
        if frame.duplicated(key).any():
            raise ValueError("标签请求、源数据或证据存在重复键")
    feature_index = features.set_index(["security_id", "date"])
    bar_index = bars.set_index(["security_id", "date"])
    proof_index = evidence.set_index(["security_id", "date"])
    security_index = master.set_index("security_id")
    fx_index = None
    if fx_rates is not None:
        if fx_rates.duplicated(["date", "currency"]).any():
            raise ValueError("同日同币种汇率存在重复证据")
        fx_index = fx_rates.set_index(["date", "currency"])
    patches, decisions = [], []
    for request in requests.to_dict("records"):
        sid, day, end, horizon = request["security_id"], pd.Timestamp(request["date"]), pd.Timestamp(request["label_end"]), request["horizon"]
        decision = {"date": day, "security_id": sid, "horizon": horizon, "label_end": end,
                    "status": "outside_development_period"}
        decisions.append(decision)
        if day > DEVELOPMENT_END or (pd.notna(end) and end > DEVELOPMENT_END):
            continue
        if pd.isna(day) or pd.isna(end) or end > as_of:
            decision["status"] = "label_not_mature"
            continue
        if horizon not in HORIZONS or day not in sessions:
            decision["status"] = "calendar_horizon_mismatch"
            continue
        position = sessions.get_loc(day) + int(horizon)
        if position >= len(sessions) or sessions[position] != end:
            decision["status"] = "calendar_horizon_mismatch"
            continue
        key = (sid, day)
        if key not in feature_index.index:
            decision["status"] = "feature_row_missing"
            continue
        feature = feature_index.loc[key]
        old = feature[f"fwd_return_{horizon}"]
        if not pd.isna(old):
            decision["status"] = "existing_label_kept"
            continue
        if pd.isna(feature[f"label_end_{horizon}"]) or feature[f"label_end_{horizon}"] != end:
            decision["status"] = "feature_label_end_mismatch"
            continue
        endpoint_key = (sid, end)
        if endpoint_key not in bar_index.index:
            decision["status"] = "endpoint_bar_missing"
            continue
        if endpoint_key not in proof_index.index or sid not in security_index.index:
            decision["status"] = "endpoint_evidence_unverified"
            continue
        endpoint, proof, security = bar_index.loc[endpoint_key], proof_index.loc[endpoint_key], security_index.loc[sid]
        issue = security.identity_source_issue_id
        currency = security.identity_source_currency
        valid_proof = (
            pd.notna(issue) and proof.source_issue_id == issue and proof.identity_source_issue_id == issue
            and str(proof.source_stock_code) == str(security.exchange_code).removesuffix(".HK")
            and _positive(proof.source_listing_id) and proof.currency == currency
            and currency in ("HKD", "CNY") and _true(proof.identity_date_verified)
            and _true(proof.reference_price_matches) and _true(proof.data_valid)
            and _positive(proof.source_closing) and proof.source_suspended == 0 and proof.source_noclose == 0
            and proof.source_volume == 0 and proof.source_amount == 0
            and isinstance(proof.source_url, str) and bool(proof.source_url.strip())
        )
        if not valid_proof:
            decision["status"] = "endpoint_evidence_unverified"
            continue
        endpoint_fx, fx_source = 1., "HKD denomination"
        if currency == "CNY":
            if fx_index is None or (end, currency) not in fx_index.index:
                decision["status"] = "endpoint_fx_unavailable"
                continue
            rate = fx_index.loc[(end, currency)]
            if (not _positive(rate.fx_to_hkd) or pd.isna(rate.observation_date) or rate.observation_date != end
                    or not isinstance(rate.source_url, str) or not rate.source_url.strip()):
                decision["status"] = "endpoint_fx_unavailable"
                continue
            endpoint_fx, fx_source = float(rate.fx_to_hkd), rate.source_url
        endpoint_valid = (
            _true(endpoint.data_valid) and _positive(endpoint.raw_close) and _positive(endpoint.adj_close)
            and endpoint.raw_close == proof.raw_close and endpoint.adj_close == proof.adj_close
            and np.float32(endpoint.raw_close) == np.float32(proof.source_closing)
            and (pd.isna(endpoint.identity_source_issue_id) or endpoint.identity_source_issue_id == issue)
            and (pd.isna(endpoint.currency) or endpoint.currency == currency)
            and (pd.isna(endpoint.fx_to_hkd) or endpoint.fx_to_hkd == endpoint_fx)
        )
        if not endpoint_valid:
            decision["status"] = "endpoint_price_or_data_unverified"
            continue
        if key not in bar_index.index:
            decision["status"] = "signal_price_or_identity_unverified"
            continue
        signal = bar_index.loc[key]
        signal_valid = (
            feature.status == "ok" and _positive(feature.adj_close_hkd)
            and feature.adj_close_hkd == signal.adj_close_hkd and feature.raw_close == signal.raw_close
            and _verified_trade(signal, issue, currency)
            and feature.fx_to_hkd == signal.fx_to_hkd
        )
        if not signal_valid:
            decision["status"] = "signal_price_or_identity_unverified"
            continue
        preceding = pd.Timestamp(proof.preceding_anchor_date)
        if (pd.isna(preceding) or preceding >= end or (sid, preceding) not in bar_index.index
                or not _verified_trade(bar_index.loc[(sid, preceding)], issue, currency)):
            decision["status"] = "past_identity_anchor_unavailable"
            continue
        new_return = float(endpoint.adj_close * endpoint_fx / feature.adj_close_hkd - 1.)
        if not np.isfinite(new_return):
            decision["status"] = "endpoint_price_or_data_unverified"
            continue
        patches.append({"date": day, "security_id": sid, "horizon": int(horizon), "label_end": end,
                        "old_fwd_return": np.nan, "new_fwd_return": new_return,
                        "reason": "verified_zero_trade_endpoint_label", "source_url": proof.source_url,
                        "source_issue_id": int(issue), "source_listing_id": int(proof.source_listing_id),
                        "source_stock_code": proof.source_stock_code,
                        "signal_adj_close_hkd": float(feature.adj_close_hkd),
                        "endpoint_existing_adj_close": float(endpoint.adj_close), "endpoint_fx_to_hkd": endpoint_fx,
                        "endpoint_existing_raw_close": float(endpoint.raw_close), "endpoint_source_closing": float(proof.source_closing),
                        "preceding_anchor_date": preceding,
                        "retrospective_following_anchor_date": proof.following_anchor_date,
                        "following_anchor_used_for_label": False, "label_available_date": end,
                        "identity_support_basis": "same_issue_daily_reference_and_prior_verified_trade_observation",
                        "endpoint_currency": currency, "endpoint_fx_source_url": fx_source})
        decision["status"] = "patch_created"
    return pd.DataFrame(patches, columns=PATCH_COLUMNS), pd.DataFrame(decisions)


def export_endpoint_label_patches(data_root, evidence_root, as_of=DEVELOPMENT_END):
    """读取原始分区，输出补丁与审计；不写入行情或特征目录。"""
    data_root, evidence_root = Path(data_root), Path(evidence_root)
    requests = pd.read_parquet(evidence_root / "prediction_gap_review.parquet")
    requests = requests[["date", "security_id", "horizon", "label_end"]]
    evidence = pd.read_parquet(evidence_root / "verified_endpoint_identity.parquet")
    securities = sorted(requests.security_id.unique())
    if pd.to_datetime(requests.label_end).gt(DEVELOPMENT_END).any():
        raise ValueError("端点标签导出仅限开发期证据")
    columns = ["date", "security_id", "status", "raw_close", "fx_to_hkd", "adj_close_hkd"]
    columns += [column for horizon in HORIZONS for column in (f"fwd_return_{horizon}", f"label_end_{horizon}")]
    features = pd.concat([pq.read_table(data_root / "features" / f"{quote(sid, safe='.!-')}.parquet",
                                        columns=columns, filters=[("date", "<=", DEVELOPMENT_END)]).to_pandas()
                          for sid in securities], ignore_index=True)
    required_dates = pd.concat([requests.date, requests.label_end, evidence.preceding_anchor_date]).dropna()
    bar_columns = ["date", "security_id", "raw_close", "adj_close", "adj_close_hkd", "data_valid",
                   "volume", "amount", "quote_present", "currency", "fx_to_hkd", "identity_date_verified", "identity_source_issue_id"]
    bars = pd.concat([pq.read_table(data_root / "bars" / f"{year}.parquet", columns=bar_columns,
                                   filters=[("security_id", "in", securities), ("date", "<=", DEVELOPMENT_END)]).to_pandas()
                      for year in sorted(pd.to_datetime(required_dates).dt.year.unique())], ignore_index=True)
    master = pd.read_parquet(data_root / "securities.parquet")
    master = master.loc[master.security_id.isin(securities)]
    calendar = pd.read_parquet(data_root / "references/calendar.parquet")
    sessions = pd.DatetimeIndex(calendar.loc[calendar.is_open.eq(1) & calendar.cal_date.le(DEVELOPMENT_END), "cal_date"])
    fx_rates = None
    if master.identity_source_currency.eq("CNY").any():
        fx_root = data_root / "references/fx_research"
        metadata = json.loads((fx_root / "source_metadata.json").read_text(encoding="utf-8"))
        if metadata["record_schema"]["cny"]["unit"] != "HKD per 1 Chinese renminbi (CNY/RMB)":
            raise ValueError("人民币历史汇率单位未经核实")
        payload = json.loads((fx_root / "hkma_usd_cny_hkd_2010_present.json").read_text(encoding="utf-8"))
        fx_rates = pd.DataFrame(payload["records"])
        fx_rates["date"] = pd.to_datetime(fx_rates.end_of_day)
        fx_rates = fx_rates.loc[fx_rates.date.le(DEVELOPMENT_END), ["date", "cny"]].rename(columns={"cny": "fx_to_hkd"})
        fx_rates["observation_date"] = fx_rates.date
        fx_rates["currency"] = "CNY"
        fx_rates["source_url"] = metadata["api_url"]
    patches, decisions = endpoint_label_patches(requests, features, bars, evidence, master, sessions, as_of, fx_rates=fx_rates)
    patches.to_parquet(evidence_root / "label_patches.parquet", index=False)
    decisions.to_parquet(evidence_root / "label_patch_decisions.parquet", index=False)
    audit = {"scope": "audited endpoint labels, development years 2016-2023",
             "requested_securities": len(securities),
             "confirmation_used": False, "as_of": pd.Timestamp(as_of).date().isoformat(),
             "requested_labels": len(requests), "patch_rows": len(patches),
             "patches_by_security": patches.security_id.value_counts().to_dict(),
             "patches_by_horizon": {str(key): int(value) for key, value in patches.horizon.value_counts().items()},
             "patches_by_currency": patches.endpoint_currency.value_counts().to_dict(),
             "decisions": decisions.status.value_counts().to_dict(),
             "old_values_all_nan": bool(patches.old_fwd_return.isna().all()),
             "bars_modified": False, "features_modified": False, "securities_modified": False,
             "prices_created_or_modified": False, "trade_volume_created_or_modified": False,
             "following_anchor_used_for_label": False,
             "identity_support": "Endpoint source quote and prior verified trade; following anchor is retrospective review only.",
             "return_calculation": "existing endpoint adj_close * same-date HKMA FX (HKD denomination = 1) / existing signal adj_close_hkd - 1",
             "note": "补丁只替换已成熟的缺失标签；不改变信号日因子、端点价格、成交状态或成交量。"}
    (evidence_root / "label_patches_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    return audit


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--evidence-root", required=True)
    args = parser.parse_args()
    print(json.dumps(export_endpoint_label_patches(args.data_root, args.evidence_root), ensure_ascii=False))
