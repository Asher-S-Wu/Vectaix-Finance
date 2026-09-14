from hk_quant.training import confirmation_readiness
from test_evaluation import passing_evaluation_pair, complete_audit


def test_confirmation_requires_development_and_data_gates_without_claiming_release():
    result=confirmation_readiness(passing_evaluation_pair(),complete_audit())
    assert result['ready'] is True
    assert 'eligible' not in result
    assert not any(name.startswith('confirmation.') for name in result['checks'])


def test_missing_portfolio_or_cash_evidence_prevents_using_confirmation():
    development=passing_evaluation_pair()
    del development['portfolio']
    result=confirmation_readiness(development,complete_audit())
    assert result['ready'] is False
    assert any(name.startswith('development.portfolio.') for name in result['failed_checks'])
    audit=complete_audit();audit['corporate_action_cash_coverage_complete']=False
    result=confirmation_readiness(passing_evaluation_pair(),audit)
    assert result['ready'] is False
    assert 'data_audit.corporate_action_cash_coverage_complete' in result['failed_checks']
