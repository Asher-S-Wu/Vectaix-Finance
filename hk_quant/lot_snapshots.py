"""公开历史交易单位页面：保留源证券身份，并限制到同柜台的已核验期间。"""
import re
from urllib.parse import parse_qs, urlparse

import numpy as np
import pandas as pd
from lxml import html


def _text(node):
    return ' '.join(node.itertext()).strip()


def _nonblank(value):
    return isinstance(value, str) and bool(value.strip())


def build_identity_index(master, listings, identity_mapping):
    """以源 IssueID、原柜台代码、日期共同关联，绝不继承另一柜台的历史。"""
    if master.security_id.duplicated().any():
        raise ValueError('证券主表身份重复')
    metadata = master.copy()
    for column in ('identity_valid_from', 'identity_valid_to'):
        metadata[column] = pd.to_datetime(metadata[column])
    listing_rows = listings.loc[listings.StockExID.isin([1, 20, 22, 23, 38, 71])].copy()
    for column in ('FirstTradeDate', 'DelistDate'):
        listing_rows[column] = pd.to_datetime(listing_rows[column])
    direct = {}
    for row in identity_mapping.loc[identity_mapping.mapping_status.eq('matched_unique_isin')].itertuples(index=False):
        if pd.notna(row.IssueID) and _nonblank(row.matched_isin):
            direct[row.security_id] = (int(row.IssueID), row.matched_isin)
    by_code = {code: group for code, group in metadata.groupby('exchange_code')}
    index = {}
    for row in listing_rows.itertuples(index=False):
        if not _nonblank(row.StockCode) or not re.fullmatch(r'\d{1,5}', row.StockCode):
            continue
        code = row.StockCode.zfill(5)
        candidates = by_code.get(code + '.HK')
        if candidates is None:
            continue
        for security in candidates.itertuples(index=False):
            if security.identity_status != 'verified' or security.asset_type not in ('equity', 'reit'):
                continue
            if pd.isna(security.identity_valid_from):
                continue
            # Different nonempty ISINs are a conflict requiring event evidence.
            if _nonblank(row.isin) and _nonblank(security.isin) and row.isin != security.isin:
                continue
            source_link = direct.get(security.security_id)
            isin_match = (_nonblank(security.isin) and (
                (_nonblank(row.isin) and row.isin == security.isin)
                or source_link == (int(row.IssueID), security.isin)))
            lifecycle_match = (pd.notna(row.FirstTradeDate)
                and row.FirstTradeDate == security.identity_valid_from
                and ((pd.isna(row.DelistDate) and pd.isna(security.identity_valid_to))
                     or row.DelistDate == security.identity_valid_to))
            if not (isin_match or lifecycle_match):
                continue
            begin = security.identity_valid_from
            if pd.notna(row.FirstTradeDate):
                begin = max(begin, row.FirstTradeDate)
            ends = [day for day in (row.DelistDate, security.identity_valid_to) if pd.notna(day)]
            end = min(ends) if ends else None
            if end is not None and end <= begin:
                continue
            index.setdefault((code, int(row.IssueID)), []).append(dict(
                security_id=security.security_id, start=begin, end=end,
                source_listing_id=int(row.ID),
                identity_mapping_basis='source_issue_isin_same_counter' if isin_match else 'matching_source_and_master_listing_boundaries'))
    return index


def parse_snapshot(body, day, index, source_url, source_document):
    """每行保留 IssueID 和无法使用的原因，页面日期与要求日期必须相同。"""
    day = pd.Timestamp(day).normalize()
    document = html.fromstring(body)
    selected = document.xpath('//input[@name="d"]/@value')
    if len(selected) != 1 or pd.Timestamp(str(selected[0])).normalize() != day:
        raise ValueError('页面选择日期与请求日期不一致')
    tables = []
    for table in document.xpath('//table'):
        headers = [' '.join(_text(cell).split()) for cell in table.xpath('./tr/th|./thead/tr/th')]
        if 'Stock Code' in headers and 'Board lot' in headers:
            tables.append((table, headers))
    if len(tables) != 1:
        raise ValueError('页面缺少唯一的 Board lot 表格')
    table, headers = tables[0]
    code_position, lot_position = headers.index('Stock Code'), headers.index('Board lot')
    if headers.count('Date') > 1:
        raise ValueError('Board lot 表格存在多个未区分的观察日期列')
    observation_position = headers.index('Date') if 'Date' in headers else None
    rows = []
    for tr in table.xpath('./tr|./tbody/tr'):
        if 'total' in tr.get('class', '').split():
            continue
        cells = tr.xpath('./td')
        if not cells:
            continue
        if len(cells) != len(headers):
            raise ValueError('Board lot 表格列数不一致')
        raw_code = _text(cells[code_position])
        code = raw_code.zfill(5) if re.fullmatch(r'\d{1,5}', raw_code) else raw_code
        lot_literal = _text(cells[lot_position])
        lot = pd.to_numeric(lot_literal.replace(',', ''), errors='coerce')
        lot_valid = pd.notna(lot) and np.isfinite(lot) and lot > 0 and lot % 1 == 0
        observation_literal = _text(cells[observation_position]) if observation_position is not None else ''
        issues = set()
        for href in cells[code_position].xpath('.//a/@href'):
            url = urlparse(href)
            query = parse_qs(url.query)
            if url.path.endswith('str.asp') and len(query.get('i', [])) == 1 and query['i'][0].isdigit():
                issues.add(int(query['i'][0]))
        issue = next(iter(issues)) if len(issues) == 1 else None
        candidates = [item for item in index.get((code, issue), [])
                      if day >= item['start'] and (item['end'] is None or day < item['end'])]
        unique_ids = {item['security_id'] for item in candidates}
        unique_listings = {item['source_listing_id'] for item in candidates}
        matched = len(unique_ids) == 1 and len(unique_listings) == 1
        identity = candidates[0] if matched else None
        rows.append(dict(date=day, stock_code=code, source_stock_code=code,
            source_issue_id=issue, source_listing_id=identity['source_listing_id'] if identity else None,
            security_id=identity['security_id'] if identity else None,
            identity_mapping_verified=matched,
            identity_mapping_basis=identity['identity_mapping_basis'] if identity else 'unresolved_source_identity',
            lot_literal=lot_literal, source_lot_size=int(lot) if lot_valid else np.nan,
            source_observation_date_literal=observation_literal,
            status='invalid_lot' if not lot_valid else ('ok' if matched else 'identity_unresolved'),
            source_url=source_url, source_document=source_document))
    result = pd.DataFrame(rows)
    if result.empty:
        raise ValueError('Board lot 表格没有证券行')
    if result.duplicated(['date', 'stock_code', 'source_issue_id']).any():
        raise ValueError('页面同一日期、代码和源身份出现重复')
    observed = pd.to_datetime(result.source_observation_date_literal, format='%Y-%m-%d', errors='coerce')
    contemporary = observed.notna() & observed.eq(day)
    result['source_observation_date'] = observed
    result['historical_value_date_verified'] = contemporary & result.source_lot_size.notna()
    result['lot_size'] = result.source_lot_size.where(contemporary)
    result.loc[observed.isna(), 'status'] = 'missing_observation_date'
    result.loc[observed.gt(day), 'status'] = 'future_observation'
    result.loc[observed.lt(day), 'status'] = 'stale_observation'
    return result
