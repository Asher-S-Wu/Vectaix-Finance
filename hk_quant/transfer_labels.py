"""用已核验转板事件及同日原价重叠校准收益尺度，只生成开发期标签补丁。"""

import json
import re
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pandas as pd


PATCH_KEYS = ['date', 'security_id', 'horizon', 'label_end']
PATCH_COLUMNS = PATCH_KEYS + ['old_fwd_return', 'new_fwd_return', 'reason', 'source_url']
BAR_COLUMNS = ['security_id', 'date', 'raw_close', 'cum_adjfactor', 'adj_close_hkd',
               'fx_to_hkd', 'currency', 'data_valid', 'quote_present']
GAP_REASON = 'listing_exit_requires_verified_terminal_or_transfer_outcome'


def _source_file(value):
    path = Path(value).resolve()
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f'源文件缺失或为空: {path}')
    return path


def _same_date(a, b):
    a, b = pd.Timestamp(a), pd.Timestamp(b)
    return (pd.isna(a) and pd.isna(b)) or a == b


def _in_period(day, begin, end):
    return day >= pd.Timestamp(begin) and (pd.isna(end) or day < pd.Timestamp(end))


def _verify_original_share_continuity(event, old_isin, old_stock_code):
    """当前ISIN不同的历史柜台，必须有原股证法律权属连续的原文证据。"""
    snapshot_isin = event.get('old_isin_at_master_snapshot')
    snapshot_matches = (pd.isna(snapshot_isin) and pd.isna(old_isin)) or snapshot_isin == old_isin
    if (event.get('share_continuity_verified') is not True
            or 'old_isin_at_master_snapshot' not in event or not snapshot_matches):
        raise ValueError('源证券ISIN不同，缺少原股证连续证据')
    from pypdf import PdfReader
    page_number = event.get('share_continuity_evidence_page')
    excerpt = event.get('share_continuity_evidence_excerpt')
    if not isinstance(excerpt, str) or pd.isna(page_number):
        raise ValueError('原股证连续证据缺少页码或原文')
    document = event['source_document']
    if pd.notna(event.get('share_continuity_source_document')):
        document = _source_file(event['share_continuity_source_document'])
        source_url = event.get('share_continuity_source_url')
        if not isinstance(source_url, str) or document.name != Path(urlparse(source_url).path).name:
            raise ValueError('原股证连续证据文件与URL不匹配')
        catalog = json.loads(Path(event['source_catalog']).read_text(encoding='utf-8'))['records']
        sources = [row for row in catalog if row['pdf_url'] == source_url]
        stamp = pd.Timestamp(event.get('share_continuity_published_at'))
        if (len(sources) != 1 or pd.isna(stamp) or stamp.tzinfo is None
                or pd.Timestamp(sources[0]['published_at']) != stamp
                or stamp > pd.Timestamp(event['published_at'])):
            raise ValueError('原股证连续证据公开时间不匹配或晚于最终修订')
        if old_stock_code not in sources[0]['stock_codes']:
            raise ValueError('原股证连续证据的源证券代码不匹配')
    pages = PdfReader(document).pages
    if page_number != int(page_number) or not 1 <= page_number <= len(pages):
        raise ValueError('原股证连续证据页码无效')
    compact = lambda value: re.sub(r'\s+', '', value).lower()
    actual = compact(pages[int(page_number) - 1].extract_text())
    excerpt = compact(excerpt)
    certificate_unchanged = ('willnotinvolveanytransferorexchange' in excerpt
                             or 'nochangewillbemadetothesharecertificate' in excerpt)
    if (excerpt not in actual or 'existingsharecertificates' not in excerpt
            or 'goodevidenceoflegaltitle' not in excerpt
            or not certificate_unchanged):
        raise ValueError('原股证连续证据与原文件不符')


def _validate_event(event, master, root):
    if (event['event_type'] != 'listing_transfer_existing_shares_continue'
            or event['verified'] != True or event['source_identity_verified'] != True):
        raise ValueError('转板事件或源证券身份未经核验')
    if event['continuing_shares_per_old_share'] != 1.:
        raise ValueError('本轮只支持已核验 1:1 原股份连续转板')
    if event['currency'] != 'HKD':
        raise ValueError('本轮只支持同币种 HKD 转板')
    for field in ('source_document', 'source_catalog', 'source_identity_file', 'source_quote_file'):
        _source_file(event[field])
    publication = pd.Timestamp(event['published_at'])
    if pd.isna(publication) or publication.tzinfo is None:
        raise ValueError('公告时间必须已知且包含时区')
    catalog = json.loads(Path(event['source_catalog']).read_text(encoding='utf-8'))['records']
    matches = [row for row in catalog if row['pdf_url'] == event['source_url']]
    if len(matches) != 1 or pd.Timestamp(matches[0]['published_at']) != publication:
        raise ValueError('公告时间与原始目录不匹配')
    old_identity = master.loc[master.security_id.eq(event['old_security_id'])]
    if len(old_identity) != 1:
        raise ValueError('源证券主表身份不唯一')
    if old_identity.iloc[0].exchange_code.split('.')[0] not in matches[0]['stock_codes']:
        raise ValueError('公告的源证券代码不匹配')
    transfer = pd.Timestamp(event['listing_transfer_effective_date'])
    if not (_same_date(event['old_listing_end_exclusive'], transfer)
            and _same_date(event['new_listing_start'], transfer)):
        raise ValueError('源证券转板期间不衔接')
    if Path(event['source_quote_file']).resolve() != (root / 'bars' / f'{transfer.year}.parquet').resolve():
        raise ValueError('转板源文件与行情年份不匹配')
    listings = pd.read_parquet(event['source_identity_file'])
    for prefix in ('old', 'new'):
        security_id = event[f'{prefix}_security_id']
        identity = master.loc[master.security_id.eq(security_id)]
        listing = listings.loc[listings.ID.eq(event[f'{prefix}_source_listing_id'])]
        if len(identity) != 1 or len(listing) != 1:
            raise ValueError('源证券主表或上市记录不唯一')
        identity, listing = identity.iloc[0], listing.iloc[0]
        if (int(listing.IssueID) != event['source_issue_id']
                or str(listing.StockCode).zfill(5) + '.HK' != identity.exchange_code
                or identity.identity_status != 'verified'):
            raise ValueError('源证券身份或代码复用期间不匹配')
        if identity['isin'] != event['isin']:
            if prefix != 'old':
                raise ValueError('后继源证券ISIN不匹配')
            _verify_original_share_continuity(event, identity['isin'], identity.exchange_code.split('.')[0])
        elif prefix == 'old' and pd.notna(event.get('share_continuity_source_document')):
            _verify_original_share_continuity(event, identity['isin'], identity.exchange_code.split('.')[0])
        if identity.currency != 'HKD':
            raise ValueError('源证券币种不是 HKD')
        for field, source_field, master_field in (
            ('listing_start', 'FirstTradeDate', 'identity_valid_from'),
            ('listing_end_exclusive', 'DelistDate', 'identity_valid_to'),
        ):
            value = event[f'{prefix}_{field}']
            if not (_same_date(value, listing[source_field]) and _same_date(value, identity[master_field])):
                raise ValueError('源证券上市期间不匹配')
        if prefix == 'old' and not _same_date(listing.FinalTradeDate, event['old_last_trading_date']):
            raise ValueError('源证券最后交易日不匹配')


def _valid_raw(row):
    return (row is not None and row.quote_present == True and row.data_valid == True
            and row.currency == 'HKD' and np.isfinite(row.fx_to_hkd) and row.fx_to_hkd == 1.
            and np.isfinite(row.raw_close) and row.raw_close > 0
            and np.isfinite(row.cum_adjfactor) and row.cum_adjfactor > 0)


def build_transfer_label_patches(data_root, events_path, workqueue_path):
    """源文件只读；时间或标签不合格的行记录原因，事件证据冲突则拒绝生成。"""
    root = Path(data_root).resolve()
    events_path, workqueue_path = _source_file(events_path), _source_file(workqueue_path)
    events, queue = pd.read_parquet(events_path), pd.read_parquet(workqueue_path)
    master_file = _source_file(root / 'securities.parquet')
    master = pd.read_parquet(master_file)
    queue['date'] = pd.to_datetime(queue.date)
    queue['label_end'] = pd.to_datetime(queue.label_end)
    if events.event_id.duplicated().any() or events.old_security_id.duplicated().any():
        raise ValueError('转板事件重复或旧柜台指向多个后继')
    if queue.duplicated(PATCH_KEYS).any():
        raise ValueError('标签缺口键重复')
    bar_cache = {}

    def bar(security, day):
        key = security, day.year
        if key not in bar_cache:
            path = _source_file(root / 'bars' / f'{day.year}.parquet')
            frame = pd.read_parquet(path, columns=BAR_COLUMNS, filters=[('security_id', '==', security)])
            frame['date'] = pd.to_datetime(frame.date)
            if frame.date.duplicated().any():
                raise ValueError('源行情同证券同日重复')
            bar_cache[key] = frame.set_index('date')
        frame = bar_cache[key]
        return frame.loc[day] if day in frame.index else None

    patches, exclusions = [], []
    examined = 0
    for event in events.to_dict('records'):
        _validate_event(event, master, root)
        old, new = event['old_security_id'], event['new_security_id']
        subset = queue.loc[queue.security_id.eq(old) & queue.gap_reason.eq(GAP_REASON)]
        anchor_date = pd.Timestamp(event['old_last_trading_date'])
        if not _in_period(anchor_date, event['old_listing_start'], event['old_listing_end_exclusive']):
            raise ValueError('锚点不在原柜台挂牌期间')
        old_anchor, new_anchor = bar(old, anchor_date), bar(new, anchor_date)
        if (not _valid_raw(old_anchor) or not _valid_raw(new_anchor)
                or old_anchor.raw_close != new_anchor.raw_close):
            raise ValueError('锚点必须为同日相同原价且双方原始数据有效')
        bridge = float(old_anchor.cum_adjfactor / new_anchor.cum_adjfactor)
        features_file = _source_file(root / 'features' / f'{old}.parquet')
        features = pd.read_parquet(features_file)
        if features.duplicated(['security_id', 'date']).any():
            raise ValueError('源特征同证券同日重复')
        publication = pd.Timestamp(event['published_at'])
        for gap in subset.to_dict('records'):
            examined += 1
            day, end, horizon = pd.Timestamp(gap['date']), pd.Timestamp(gap['label_end']), int(gap['horizon'])

            def exclude(reason):
                exclusions.append(dict(**{key: gap[key] for key in PATCH_KEYS}, event_id=event['event_id'], reason=reason))

            if pd.isna(end) or end > pd.Timestamp('2023-12-31') or day < pd.Timestamp('2016-01-01'):
                exclude('endpoint_outside_development')
                continue
            if not _in_period(day, event['old_listing_start'], event['old_listing_end_exclusive']):
                exclude('signal_outside_old_listing')
                continue
            if not _in_period(end, event['new_listing_start'], event['new_listing_end_exclusive']):
                exclude('endpoint_outside_new_listing')
                continue
            if publication > end.tz_localize('Asia/Hong_Kong') + pd.Timedelta(hours=19):
                exclude('event_not_public_by_label_cutoff')
                continue
            if horizon not in (1, 5, 20, 60):
                raise ValueError('预测期限无效')
            feature = features.loc[features.security_id.eq(old) & pd.to_datetime(features.date).eq(day)]
            if len(feature) != 1:
                exclude('source_feature_absent')
                continue
            feature = feature.iloc[0]
            if pd.notna(feature[f'fwd_return_{horizon}']):
                exclude('existing_label_present')
                continue
            if not _same_date(feature[f'label_end_{horizon}'], end):
                exclude('source_label_end_mismatch')
                continue
            signal, endpoint = bar(old, day), bar(new, end)
            if not _valid_raw(signal) or not _valid_raw(endpoint):
                exclude('invalid_signal_or_successor_observation')
                continue
            denominator = feature.adj_close_hkd
            expected_signal = signal.raw_close * signal.cum_adjfactor * signal.fx_to_hkd
            if (not np.isfinite(denominator) or denominator <= 0
                    or not np.isclose(denominator, expected_signal, rtol=1e-7, atol=1e-10)):
                exclude('source_signal_adjustment_mismatch')
                continue
            adjusted_endpoint = float(endpoint.raw_close * endpoint.cum_adjfactor * bridge * endpoint.fx_to_hkd)
            value = adjusted_endpoint / float(denominator) - 1.
            if not np.isfinite(value) or value <= -1:
                raise ValueError('转板收益无法表示')
            patches.append(dict(date=day, security_id=old, horizon=horizon, label_end=end,
                old_fwd_return=np.nan, new_fwd_return=value,
                reason='verified_same_share_listing_transfer_scale_bridge', source_url=event['source_url'],
                event_id=event['event_id'], successor_security_id=new, source_issue_id=event['source_issue_id'],
                old_source_listing_id=event['old_source_listing_id'], new_source_listing_id=event['new_source_listing_id'],
                event_published_at=publication, old_listing_start=event['old_listing_start'],
                old_listing_end_exclusive=event['old_listing_end_exclusive'],
                new_listing_start=event['new_listing_start'], new_listing_end_exclusive=event['new_listing_end_exclusive'],
                anchor_date=anchor_date, old_anchor_raw_close=float(old_anchor.raw_close),
                successor_anchor_raw_close=float(new_anchor.raw_close), old_anchor_factor=float(old_anchor.cum_adjfactor),
                successor_anchor_factor=float(new_anchor.cum_adjfactor), adjustment_scale_bridge=bridge,
                successor_anchor_role='scale_calibration_only_not_listed_counter',
                source_signal_adj_close_hkd=float(denominator), successor_endpoint_raw_close=float(endpoint.raw_close),
                successor_endpoint_factor=float(endpoint.cum_adjfactor), successor_endpoint_fx_to_hkd=float(endpoint.fx_to_hkd),
                bridged_endpoint_adj_close_hkd=adjusted_endpoint,
                source_document=event['source_document'], source_catalog=event['source_catalog'],
                source_identity_file=event['source_identity_file'], source_master_file=str(master_file),
                source_events_file=str(events_path), source_workqueue_file=str(workqueue_path),
                source_features_file=str(features_file),
                source_signal_bars_file=str(root / 'bars' / f'{day.year}.parquet'),
                source_anchor_bars_file=str(root / 'bars' / f'{anchor_date.year}.parquet'),
                source_endpoint_bars_file=str(root / 'bars' / f'{end.year}.parquet')))
    result = pd.DataFrame(patches) if patches else pd.DataFrame(columns=PATCH_COLUMNS)
    if result.duplicated(PATCH_KEYS).any():
        raise ValueError('生成的标签补丁重复')
    audit = dict(events=len(events), examined_rows=examined, patches=len(result), excluded_rows=len(exclusions),
        excluded=exclusions, source_data_modified=False, patches_applied=False,
        development_end='2023-12-31', return_basis='successor_raw_end * successor_factor_end * old_anchor_factor / successor_anchor_factor * HKD_fx / source_signal_adj_close_hkd - 1',
        anchor_scope='Same-day identical raw prices calibrate scales; successor prelisting observation does not establish a tradable counter.')
    return result, audit
