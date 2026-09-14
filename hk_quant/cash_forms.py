"""读取港交所标准现金股息表格的原始条款，保留条件和真实公开时间。"""
from decimal import Decimal
import re
from urllib.parse import urlparse

import pandas as pd


def _field(text, left, right, *, start=0, end=None):
    pattern = re.compile(re.escape(left) + r'\s*(.*?)\s*' + re.escape(right), re.S)
    matches = list(pattern.finditer(text, start, len(text) if end is None else end))
    if len(matches) != 1 or not matches[0][1].strip():
        raise ValueError('现金公告字段缺失或重复: ' + left)
    match = matches[0]
    return match[1].strip(), match.start(), match.end(1)


def _date(literal):
    if not re.fullmatch(r'\d{1,2} [A-Za-z]+ \d{4}', literal):
        raise ValueError('日期不是明确日历日期: ' + literal)
    return pd.to_datetime(literal, format='%d %B %Y')


def _money(literal):
    match = re.fullmatch(r'(HKD|RMB|CNY|USD) ([0-9]+(?:\.[0-9]+)?) per share', literal)
    if not match:
        raise ValueError('金额的币种、单位或数值不明确: ' + literal)
    value = Decimal(match[2])
    if not value.is_finite() or value < 0:
        raise ValueError('现金金额不合法')
    return ('CNY' if match[1] == 'RMB' else match[1]), str(value)


def parse_equity_cash_form(pages, *, source_url):
    """返回原文事实；简单条款不等于公告版本完整、股东批准或实际到账。"""
    parsed_url = urlparse(source_url)
    if parsed_url.scheme != 'https' or parsed_url.hostname != 'www1.hkexnews.hk' or not parsed_url.path.endswith('.pdf'):
        raise ValueError('现金公告原文必须绑定港交所PDF来源')
    raw = ' '.join(' '.join(page['text'].split()) for page in pages)
    forms = set(re.findall(r'\bEF\d{3}\b', raw))
    if forms != {'EF001'} or 'Cash Dividend Announcement for Equity Issuer' not in raw:
        raise ValueError('不是EF001股票现金股息表格')
    page_spans, pieces, offset = [], [], 0
    for page in pages:
        part = ' '.join(page['text'].split())
        part = re.sub(r'EF001\s+(?:vPage|Page)\s+\d+\s+of\s+\d+\s+(?:v\s*)?\d+(?:\.\d+)+', ' ', part)
        part = ' '.join(part.split())
        page_spans.append((int(page['page']), offset, offset + len(part)))
        pieces.append(part)
        offset += len(part) + 1
    text = ' '.join(pieces)
    headings = ('Information relating to the dividend', 'Information relating to Hong Kong share register',
                'Information relating to withholding tax')
    boundaries = []
    for heading in headings:
        occurrences = list(re.finditer(re.escape(heading), text))
        if len(occurrences) != 1:
            raise ValueError('现金公告章节缺失或重复: ' + heading)
        boundaries.append(occurrences[0].start())
    dividend_start, register_start, tax_start = boundaries
    if not dividend_start < register_start < tax_start:
        raise ValueError('现金公告章节顺序无效')
    dividend_section = text[dividend_start:register_start]
    combined_label = 'For the financial year / period end'
    financial_label = 'For the financial year end'
    reporting_label = 'Reporting period end for the dividend declared'
    combined = combined_label in dividend_section
    separate = financial_label in dividend_section and reporting_label in dividend_section
    if combined == separate:
        raise ValueError('现金公告财年与分派报告期栏结构缺失或冲突')
    labels = {
        'issuer': ('Issuer name', 'Stock code'),
        'stock_code': ('Stock code', 'Multi-counter stock code and currency'),
        'multi_counter': ('Multi-counter stock code and currency', 'Other related stock code(s) and name(s)'),
        'printed_date': ('Announcement date', 'Status'),
        'announcement_status': ('Status', 'Information relating to the dividend'),
        'dividend_type': ('Dividend type', 'Dividend nature'),
        'dividend_nature': ('Dividend nature', financial_label if separate else combined_label),
        'period_end': (reporting_label if separate else combined_label, 'Dividend declared'),
        'declared_amount': ('Dividend declared', "Date of shareholders' approval"),
        'approval': ("Date of shareholders' approval", 'Information relating to Hong Kong share register'),
        'payment_amount': ('the dividend will be paid', 'Exchange rate'),
        'exchange_rate': ('Exchange rate', 'Ex-dividend date'),
        'ex_date': ('Ex-dividend date', 'Latest time to lodge transfer documents'),
        'record_date': ('Record date', 'Payment date'),
        'payment_date': ('Payment date', 'Share registrar and its address'),
        'withholding_tax': ('Details of withholding tax applied to the dividend declared',
                            'Information relating to listed warrants / convertible securities issued by the issuer'),
    }
    if separate:
        labels['financial_year_end'] = (financial_label, reporting_label)
    header_fields = {'issuer','stock_code','multi_counter','printed_date','announcement_status'}
    dividend_fields = {'dividend_type','dividend_nature','period_end','declared_amount','approval','financial_year_end'}
    register_fields = {'payment_amount','exchange_rate','ex_date','record_date','payment_date'}
    fields = {}
    for key, label in labels.items():
        if key in header_fields:
            start, end = 0, dividend_start + len(headings[0])
        elif key in dividend_fields:
            start, end = dividend_start, register_start + len(headings[1])
        elif key in register_fields:
            start, end = register_start, tax_start
        else:
            start, end = tax_start, len(text)
        fields[key] = _field(text, *label, start=start, end=end)
    other = list(re.finditer(r'Other information (?:Other information )?(.*?) Directors of the issuer', text))
    if len(other) != 1:
        raise ValueError('其他条款缺失或重复')
    fields['other_information'] = (other[0][1].strip(), other[0].start(), other[0].end(1))
    facts = {key: value[0] for key, value in fields.items()}
    status_parts = facts['announcement_status'].split('Reason for the update / change', 1)
    status = status_parts[0].strip()
    update_reason = status_parts[1].strip() if len(status_parts) == 2 else None
    if not re.fullmatch(r'\d{5}', facts['stock_code']):
        raise ValueError('表格证券代码不明确')
    declared_currency, declared_amount = _money(facts['declared_amount'])
    cash_currency, cash_amount = _money(facts['payment_amount'])
    period_end = _date(facts['period_end'])
    financial_year_end = (_date(facts['financial_year_end'])
        if separate and facts['financial_year_end'] != 'Not applicable' else pd.NaT)
    dividend_type_keys = {'Final dividend':'final', 'Final':'final',
                         'Semi-annual dividend':'interim_semiannual', 'Interim (Semi-annual)':'interim_semiannual'}
    dividend_type_key = dividend_type_keys.get(facts['dividend_type'], facts['dividend_type'])
    ex, record, payment = map(_date, (facts['ex_date'], facts['record_date'], facts['payment_date']))
    if payment < ex or record < ex or record > payment:
        raise ValueError('除息、登记与派付日期的顺序不合法')
    reasons = []
    for key, reason in (
        ('approval', 'shareholder_approval_requires_evidence'),
        ('multi_counter', 'multi_counter_terms_require_review'),
        ('withholding_tax', 'withholding_tax_requires_account_terms'),
        ('other_information', 'additional_terms_require_review'),
    ):
        if facts[key] != 'Not applicable':
            reasons.append(reason)
    if status not in ('New announcement', 'Update to previous announcement'):
        reasons.append('announcement_cancelled_or_unrecognized')
    evidence = {key: dict(literal=value, pages=[number for number, left, right in page_spans
                    if start < right and end > left]) for key, (value, start, end) in fields.items()}
    return dict(stock_code=facts['stock_code'], issuer_name=facts['issuer'], source_url=source_url,
        announcement_date_printed=_date(facts['printed_date']), announcement_status=status, update_reason=update_reason,
        dividend_type=facts['dividend_type'], dividend_type_key=dividend_type_key, dividend_nature=facts['dividend_nature'],
        period_end=period_end, financial_year_end=financial_year_end,
        reporting_period_end=period_end if separate else pd.NaT,
        period_end_basis='reporting_period_end' if separate else 'combined_financial_year_or_period_end',
        ex_date=ex, record_date=record, payment_date=payment,
        cash_currency=cash_currency, cash_per_share_decimal=cash_amount,
        declared_currency=declared_currency, declared_per_share_decimal=declared_amount,
        payment_date_type='specified_payment_date', approval_literal=facts['approval'],
        simple_cash_terms=not reasons, review_reasons=reasons, evidence=evidence,
        payment_basis='issuer_final_schedule_simulated', actual_account_receipt_verified=False,
        formal_replay_eligible=False, coverage_complete=False)


def map_form_identity(form, catalog, master):
    """公告代码与目录及同一有效期证券身份同时匹配；不跨柜台或代码复用。"""
    stamp = pd.Timestamp(catalog['published_at'])
    if pd.isna(stamp) or stamp.tzinfo is None:
        raise ValueError('目录公开时间必须明确且包含时区')
    if form['source_url'] != catalog['pdf_url']:
        raise ValueError('原文PDF与公告目录URL不一致')
    if form['stock_code'] not in catalog['stock_codes']:
        raise ValueError('表格代码与公告目录不一致')
    day = stamp.tz_convert('Asia/Hong_Kong').tz_localize(None).normalize()
    code = form['stock_code'] + '.HK'
    start = pd.to_datetime(master.identity_valid_from)
    end = pd.to_datetime(master.identity_valid_to)
    matches = master.loc[master.exchange_code.eq(code) & master.asset_type.eq('equity')
        & master.identity_status.eq('verified') & start.le(day) & start.le(form['ex_date'])
        & (end.isna() | end.gt(day)) & (end.isna() | end.gt(form['ex_date']))]
    if len(matches) != 1:
        raise ValueError('原公告不能唯一对应同柜台的已核验证券身份')
    identity = matches.iloc[0]
    issue = identity.identity_source_issue_id
    event_id = '|'.join((identity.security_id, form['dividend_type_key'], form['dividend_nature'],
                         form['period_end'].strftime('%Y-%m-%d')))
    return dict(**form, security_id=identity.security_id, event_id=event_id,
        identity_issue_id=int(issue) if pd.notna(issue) else None,
        identity_valid_from=identity.identity_valid_from, identity_valid_to=identity.identity_valid_to,
        published_at=stamp, amount_known_at=stamp,
        source_fact_verified=True, identity_mapping_verified=True)
