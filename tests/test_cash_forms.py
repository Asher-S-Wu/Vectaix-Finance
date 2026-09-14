import pandas as pd
import pytest

from hk_quant.cash_forms import parse_equity_cash_form as parse_form, map_form_identity


SOURCE = 'https://www1.hkexnews.hk/listedco/example.pdf'


def parse_equity_cash_form(pages):
    return parse_form(pages, source_url=SOURCE)


FORM = '''EF001
Cash Dividend Announcement for Equity Issuer
Issuer name Example Holdings Limited
Stock code 00215
Multi-counter stock code and currency Not applicable
Other related stock code(s) and name(s) Not applicable
Title of announcement Interim dividend
Announcement date 27 July 2021
Status New announcement
Information relating to the dividend
Dividend type Semi-annual dividend
Dividend nature Ordinary
For the financial year / period end 30 June 2021
Dividend declared HKD 0.0228 per share
Date of shareholders' approval Not applicable
Information relating to Hong Kong share register
Default currency and amount in which the dividend will be paid HKD 0.0228 per share
Exchange rate HKD 1 : HKD 1
Ex-dividend date 24 August 2021
Latest time to lodge transfer documents 25 August 2021 16:30
Book close period Not applicable
Record date 25 August 2021
Payment date 03 September 2021
Share registrar and its address Example Registrar
Information relating to withholding tax
Details of withholding tax applied to the dividend declared Not applicable
Information relating to listed warrants / convertible securities issued by the issuer
Details of listed warrants / convertible securities issued by the issuer Not applicable
Other information
Other information Not applicable
Directors of the issuer Example directors
'''


def test_original_form_payment_currency_and_literal_are_the_evidence():
    record = parse_equity_cash_form([{'page': 1, 'text': FORM}])
    assert record['cash_per_share_decimal'] == '0.0228'
    assert record['cash_currency'] == 'HKD'
    assert record['ex_date'] == pd.Timestamp('2021-08-24')
    assert record['payment_date'] == pd.Timestamp('2021-09-03')
    assert record['simple_cash_terms'] is True
    assert record['formal_replay_eligible'] is False
    assert 'HKD 0.0228 per share' in record['evidence']['payment_amount']['literal']


def test_declared_currency_is_not_substituted_for_payment_currency():
    raw = FORM.replace('Dividend declared HKD 0.0228', 'Dividend declared RMB 0.0180')
    record = parse_equity_cash_form([{'page': 1, 'text': raw}])
    assert record['cash_currency'] == 'HKD'
    assert record['declared_currency'] == 'CNY'
    assert record['cash_per_share_decimal'] == '0.0228'


@pytest.mark.parametrize('before,after,reason', [
    ("approval Not applicable", "approval 20 August 2021", 'shareholder_approval_requires_evidence'),
    ('Other information Not applicable', 'Other information Subject to an amount adjustment', 'additional_terms_require_review'),
    ('dividend declared Not applicable', 'dividend declared Tax depends on investor category', 'withholding_tax_requires_account_terms'),
    ('currency Not applicable', 'currency RMB counter 80215', 'multi_counter_terms_require_review'),
    ('Status New announcement', 'Status Cancellation', 'announcement_cancelled_or_unrecognized'),
])
def test_conditional_or_complex_terms_are_not_certified_as_simple_cash(before, after, reason):
    record = parse_equity_cash_form([{'page': 1, 'text': FORM.replace(before, after)}])
    assert not record['simple_cash_terms']
    assert reason in record['review_reasons']


@pytest.mark.parametrize('before,after', [
    ('Payment date 03 September 2021', 'Payment date On or about 03 September 2021'),
    ('HKD 0.0228 per share', 'HKD 2.28 cents per share'),
    ('Payment date 03 September 2021', 'Payment date 20 August 2021'),
    ('EF001', 'EF003'),
])
def test_unclear_dates_units_or_form_type_are_rejected(before, after):
    with pytest.raises(ValueError):
        parse_equity_cash_form([{'page': 1, 'text': FORM.replace(before, after)}])


def test_publication_uses_catalog_timestamp_and_exact_counter_lifecycle():
    form = parse_equity_cash_form([{'page': 1, 'text': FORM}])
    catalog = {'stock_codes': ['00215'], 'published_at': '2021-07-28T16:31:00+08:00',
               'pdf_url': 'https://www1.hkexnews.hk/listedco/example.pdf'}
    master = pd.DataFrame([{'security_id': '00215.HK', 'exchange_code': '00215.HK',
        'identity_source_issue_id': 123, 'identity_status': 'verified', 'asset_type': 'equity',
        'identity_valid_from': pd.Timestamp('2000-01-01'), 'identity_valid_to': pd.NaT}])
    mapped = map_form_identity(form, catalog, master)
    assert mapped['amount_known_at'] == pd.Timestamp('2021-07-28T16:31:00+08:00')
    assert mapped['announcement_date_printed'] == pd.Timestamp('2021-07-27')
    assert mapped['identity_issue_id'] == 123
    wrong = master.assign(identity_valid_to=pd.Timestamp('2021-08-24'))
    with pytest.raises(ValueError, match='身份'):
        map_form_identity(form, catalog, wrong)
    with pytest.raises(ValueError, match='目录'):
        map_form_identity(form, {**catalog, 'stock_codes': ['80215']}, master)
    with pytest.raises(ValueError, match='身份'):
        map_form_identity(form, catalog, pd.concat([master, master]))


def test_naive_publication_time_is_not_inferred_from_printed_date():
    form = parse_equity_cash_form([{'page': 1, 'text': FORM}])
    catalog = {'stock_codes': ['00215'], 'published_at': '2021-07-28', 'pdf_url': 'https://www1.hkexnews.hk/example.pdf'}
    with pytest.raises(ValueError, match='时区'):
        map_form_identity(form, catalog, pd.DataFrame())


def test_form_cannot_be_bound_to_another_same_stock_announcement():
    form = parse_equity_cash_form([{'page': 1, 'text': FORM}])
    form['source_url'] = 'https://www1.hkexnews.hk/listedco/ordinary.pdf'
    catalog = {'stock_codes': ['00215'], 'published_at': '2021-07-28T16:31:00+08:00',
               'pdf_url': 'https://www1.hkexnews.hk/listedco/special.pdf'}
    master = pd.DataFrame([{'security_id': '00215.HK', 'exchange_code': '00215.HK',
        'identity_source_issue_id': 123, 'identity_status': 'verified', 'asset_type': 'equity',
        'identity_valid_from': pd.Timestamp('2000-01-01'), 'identity_valid_to': pd.NaT}])
    with pytest.raises(ValueError, match='原文'):
        map_form_identity(form, catalog, master)


def test_real_hkex_revision_status_and_reason_are_separate():
    raw = FORM.replace('Status New announcement',
        'Status Update to previous announcement Reason for the update / change Previous announcement contained typo error')
    record = parse_equity_cash_form([{'page': 1, 'text': raw}])
    assert record['announcement_status'] == 'Update to previous announcement'
    assert record['update_reason'] == 'Previous announcement contained typo error'
    assert record['simple_cash_terms']


def test_page_evidence_locates_field_label_not_unrelated_repeated_value():
    page1, page2 = FORM.split('Information relating to listed warrants', 1)
    pages = [{'page': 1, 'text': page1}, {'page': 2, 'text': 'Information relating to listed warrants' + page2}]
    record = parse_equity_cash_form(pages)
    assert record['evidence']['approval']['pages'] == [1]
    assert record['evidence']['other_information']['pages'] == [2]


@pytest.mark.parametrize('financial_year', ['31 December 2021', 'Not applicable'])
def test_split_period_form_uses_reporting_period_and_preserves_separate_fiscal_field(financial_year):
    raw = FORM.replace('For the financial year / period end 30 June 2021',
        'For the financial year end ' + financial_year +
        '\nReporting period end for the dividend declared 30 June 2021')
    record = parse_equity_cash_form([{'page': 1, 'text': raw}])
    assert record['period_end'] == pd.Timestamp('2021-06-30')
    assert record['reporting_period_end'] == pd.Timestamp('2021-06-30')
    assert record['period_end_basis'] == 'reporting_period_end'
    assert record['evidence']['financial_year_end']['literal'] == financial_year
    assert record['evidence']['financial_year_end']['pages'] == [1]
    if financial_year == 'Not applicable':
        assert pd.isna(record['financial_year_end'])
    else:
        assert record['financial_year_end'] == pd.Timestamp('2021-12-31')
    assert record['cash_per_share_decimal'] == '0.0228'
    assert not record['formal_replay_eligible']


@pytest.mark.parametrize('old_type,new_type,financial_year', [
    ('Semi-annual dividend', 'Interim (Semi-annual)', '31 December 2021'),
    ('Final dividend', 'Final', '30 June 2021'),
])
def test_schema_change_preserves_distribution_event_id_without_using_fiscal_year(old_type, new_type, financial_year):
    old_raw = FORM.replace('Semi-annual dividend', old_type)
    new_raw = old_raw.replace(old_type, new_type).replace('For the financial year / period end 30 June 2021',
        'For the financial year end ' + financial_year +
        '\nReporting period end for the dividend declared 30 June 2021')
    old = parse_equity_cash_form([{'page': 1, 'text': old_raw}])
    new = parse_equity_cash_form([{'page': 1, 'text': new_raw}])
    catalog = {'stock_codes': ['00215'], 'published_at': '2021-07-28T16:31:00+08:00', 'pdf_url': SOURCE}
    master = pd.DataFrame([dict(security_id='00215.HK', exchange_code='00215.HK', identity_source_issue_id=123,
        identity_status='verified', asset_type='equity', identity_valid_from='2000-01-01', identity_valid_to=None)])
    assert map_form_identity(old, catalog, master)['event_id'] == map_form_identity(new, catalog, master)['event_id']
    assert old['dividend_type'] == old_type and new['dividend_type'] == new_type
    assert pd.isna(old['financial_year_end']) and pd.isna(old['reporting_period_end'])
    assert old['evidence']['period_end']['literal'] == '30 June 2021'


def test_field_mentions_in_title_or_revision_reason_cannot_replace_table_values():
    raw = FORM.replace('Title of announcement Interim dividend',
        'Title of announcement Dividend declared for the six months ended June 2021')
    raw = raw.replace('Status New announcement',
        'Status Update to previous announcement Reason for the update / change Ex-dividend date corrected')
    record = parse_equity_cash_form([{'page': 1, 'text': raw}])
    assert record['cash_per_share_decimal'] == '0.0228'
    assert record['ex_date'] == pd.Timestamp('2021-08-24')
    assert record['update_reason'] == 'Ex-dividend date corrected'


@pytest.mark.parametrize('before,after', [
    ('will be paid HKD 0.0228 per share', 'will be paid HKD amount to be announced'),
    ('Payment date 03 September 2021', 'Payment date To be announced'),
])
def test_split_period_format_does_not_infer_unannounced_cash_or_payment_date(before, after):
    raw = FORM.replace('For the financial year / period end 30 June 2021',
        'For the financial year end 31 December 2021\nReporting period end for the dividend declared 30 June 2021')
    with pytest.raises(ValueError, match='金额|日期'):
        parse_equity_cash_form([{'page': 1, 'text': raw.replace(before, after)}])
