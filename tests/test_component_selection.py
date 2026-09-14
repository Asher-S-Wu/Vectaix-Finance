import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hk_quant.component_selection import compare_components, choose_component_heads, main
from hk_quant.prediction_tasks import TASK_COLUMNS, TASK_STATUS_COLUMNS

KINDS=['factor','linear','lightgbm_small']
YEARS=list(range(2016,2024))


def annual_row(year,kind,h):
    actual=np.linspace(-.1,.1,24)
    good_probability=np.where(actual>0,.99,.01)
    shift=np.zeros(24);shift[[0,1,22,23]]=.03
    return pd.DataFrame({'date':pd.Timestamp(f'{year}-06-01'),'security_id':[f'S{i:02}' for i in range(24)],
        'horizon':h,'fwd_return':actual,'label_end':pd.Timestamp(f'{year}-09-01'),'status':'ok',
        **{column: 'ok' for column in TASK_STATUS_COLUMNS.values()},
        'score':actual if kind=='linear' else actual[::-1],
        'probability_up':good_probability if kind=='factor' else .5,
        'expected_return':actual if kind=='linear' else np.zeros(24),
        'q10':actual+shift-.01 if kind=='lightgbm_small' else np.full(24,-.08),
        'q50':actual+shift if kind=='lightgbm_small' else np.zeros(24),
        'q90':actual+shift+.01 if kind=='lightgbm_small' else np.full(24,.08),
        'baseline_probability':.5,'baseline_q10':-.9,'baseline_q50':0.,'baseline_q90':.9})


def write_evidence(root):
    (root/'development').mkdir()
    (root/'training_protocol.json').write_text(json.dumps({'kinds':KINDS,'development_years':YEARS}))
    for kind in KINDS:
        for year in YEARS:
            pd.concat([annual_row(year,kind,h) for h in (1,5,20,60)],ignore_index=True).to_parquet(root/'development'/f'{kind}_{year}.parquet',index=False)


def test_tasks_choose_different_heads_and_all_years_are_reported(tmp_path):
    write_evidence(tmp_path)
    metrics=compare_components(tmp_path)
    heads=choose_component_heads(metrics)
    for h in ('1','5','20','60'):
        assert heads[h]=={'score':'linear','probability_up':'factor','intervals':'lightgbm_small','expected_return':'linear'}
        for kind in KINDS:
            assert set(metrics['horizons'][h][kind]['annual'])==set(map(str,YEARS))
        assert all(metrics['coverage'][h][task]['common_samples']==192 for task in TASK_STATUS_COLUMNS)
        assert metrics['horizons'][h]['factor']['overall']['task_samples']=={task:192 for task in TASK_STATUS_COLUMNS}
        assert metrics['horizons'][h]['linear']['overall']['score']==pytest.approx(1.)
        assert metrics['horizons'][h]['factor']['overall']['probability_up']==pytest.approx(.0001)
        assert metrics['horizons'][h]['lightgbm_small']['overall']['intervals']==pytest.approx(47/18000)


def test_pairing_maturity_and_raw_scorable_counts_remain_visible(tmp_path):
    write_evidence(tmp_path)
    path=tmp_path/'development'/'factor_2023.parquet'
    frame=pd.read_parquet(path)
    failed=frame.security_id=='S00'
    frame.loc[failed,['status','interval_status']]='invalid_quantile_order'
    frame.loc[failed,TASK_COLUMNS['intervals']]=np.nan
    frame.to_parquet(path,index=False)
    for kind in KINDS:
        path=tmp_path/'development'/f'{kind}_2023.parquet'
        frame=pd.read_parquet(path)
        frame.loc[frame.security_id=='S01','label_end']=pd.Timestamp('2024-02-01')
        frame.loc[frame.security_id=='S02','fwd_return']=np.nan
        if kind=='linear':frame=frame.iloc[::-1]
        frame.to_parquet(path,index=False)
    metrics=compare_components(tmp_path)
    year=metrics['coverage']['20']['intervals']['annual']['2023']
    assert metrics['diagnostic_only'] is True
    assert metrics['comparison_scope'] == 'task_specific_common_scorable_matured_subsets'
    assert year['raw_rows']=={'factor':24,'linear':24,'lightgbm_small':24}
    assert year['scorable_rows']=={'factor':23,'linear':24,'lightgbm_small':24}
    assert year['common_samples']==21
    for kind in KINDS:
        assert metrics['horizons']['20'][kind]['annual']['2023']['task_samples']=={
            'score':22,'probability_up':22,'intervals':21,'expected_return':22}
    assert metrics['coverage']['20']['score']['annual']['2023']['common_samples']==22


@pytest.mark.parametrize('field,value',[('fwd_return',.99),('label_end',pd.Timestamp('2018-09-02'))])
def test_conflicting_labels_on_same_key_are_rejected(tmp_path,field,value):
    write_evidence(tmp_path)
    path=tmp_path/'development'/'linear_2018.parquet';frame=pd.read_parquet(path)
    frame.loc[0,field]=value;frame.to_parquet(path,index=False)
    with pytest.raises(ValueError,match='标签冲突'):
        compare_components(tmp_path)


def test_missing_development_year_prevents_comparison_and_cli_output(tmp_path,monkeypatch):
    write_evidence(tmp_path);(tmp_path/'development'/'factor_2020.parquet').unlink()
    monkeypatch.setattr('sys.argv',['component_selection','--results-root',str(tmp_path)])
    with pytest.raises(ValueError,match='2020'):
        main()
    assert not (tmp_path/'component_selection.json').exists()


def test_complex_component_needs_sustained_annual_improvement(tmp_path):
    write_evidence(tmp_path);metrics=compare_components(tmp_path)
    for h in metrics['horizons'].values():
        base=h['factor'];linear=h['linear'];tree=h['lightgbm_small']
        base['overall']['intervals']=.01;linear['overall']['intervals']=.02;tree['overall']['intervals']=.009
        for year in map(str,YEARS):
            base['annual'][year]['intervals']=.01;linear['annual'][year]['intervals']=.02
            tree['annual'][year]['intervals']=.008 if year=='2016' else .011
    assert all(heads['intervals']=='factor' for heads in choose_component_heads(metrics).values())


def test_cli_output_has_heads_comparison_and_no_release_eligibility_claim(tmp_path,monkeypatch):
    write_evidence(tmp_path)
    monkeypatch.setattr('sys.argv',['component_selection','--results-root',str(tmp_path)])
    main()
    config=json.loads((tmp_path/'component_selection.json').read_text())
    assert config['heads']['20']['probability_up']=='factor'
    assert config['as_of']=='2023-12-31'
    assert 'comparison' in config and 'eligible' not in config
    assert config['confirmation_used'] is False
    assert config['diagnostic_only'] is True
    assert config['comparison_scope'] == 'task_specific_common_scorable_matured_subsets'
    assert not (tmp_path/'frozen_model.json').exists()


def test_confirmatory_cutoff_cannot_be_used_for_component_selection(tmp_path):
    write_evidence(tmp_path)
    with pytest.raises(ValueError,match='2023'):
        compare_components(tmp_path,as_of='2024-12-31')


def test_daily_ic_is_date_average_not_size_weighted():
    from hk_quant.component_selection import _task_metrics
    first=annual_row(2016,'linear',20)
    second=pd.concat([annual_row(2016,'linear',20)]*2,ignore_index=True)
    second['security_id']=[str(i) for i in range(len(second))]
    second['date']=pd.Timestamp('2016-06-02')
    second['score']=-second.fwd_return
    data=pd.concat([first,second],ignore_index=True).set_index(['date','security_id','horizon'])
    result=_task_metrics(data,'score')
    assert result['score']==pytest.approx(0.) and result['score_dates']==2


def test_missing_annual_metric_cannot_pass_complex_stability_rule(tmp_path):
    write_evidence(tmp_path);metrics=compare_components(tmp_path)
    metrics['horizons']['20']['lightgbm_small']['annual'].pop('2020')
    with pytest.raises(ValueError,match='八年'):
        choose_component_heads(metrics)


def test_lower_brier_complex_model_with_failed_ece_is_rejected(tmp_path):
    write_evidence(tmp_path);metrics=compare_components(tmp_path)
    for horizon in metrics['horizons'].values():
        model=horizon['lightgbm_small']
        model['overall']['probability_up']=.00005;model['overall']['ece']=.0624
        for annual in model['annual'].values():annual['probability_up']=.00005
    assert all(head['probability_up']=='factor' for head in choose_component_heads(metrics).values())


def test_complex_comparison_still_uses_strongest_simple_before_feasibility(tmp_path):
    write_evidence(tmp_path);metrics=compare_components(tmp_path)
    for horizon in metrics['horizons'].values():
        for kind,loss,ece in [('linear',.00001,.08),('lightgbm_small',.00005,.01)]:
            horizon[kind]['overall'].update(probability_up=loss,ece=ece)
            for annual in horizon[kind]['annual'].values():annual['probability_up']=loss
    assert all(head['probability_up']=='factor' for head in choose_component_heads(metrics).values())


def test_best_pinball_model_with_failed_coverage_is_rejected(tmp_path):
    write_evidence(tmp_path);metrics=compare_components(tmp_path)
    for horizon in metrics['horizons'].values():horizon['lightgbm_small']['overall']['interval_coverage']=.90
    assert all(head['intervals'] in ('factor','linear') for head in choose_component_heads(metrics).values())


@pytest.mark.parametrize('column,value',[('baseline_probability',.6),('baseline_q10',-.8),('baseline_q50',.01),('baseline_q90',.8)])
def test_common_key_baseline_conflict_is_rejected(tmp_path,column,value):
    write_evidence(tmp_path)
    path=tmp_path/'development'/'linear_2016.parquet';frame=pd.read_parquet(path)
    frame.loc[0,column]=value;frame.to_parquet(path,index=False)
    with pytest.raises(ValueError,match='基准冲突'):compare_components(tmp_path)


def test_overall_ece_rebins_all_years_instead_of_averaging_annual_ece(tmp_path):
    from hk_quant.evaluation import _ece
    write_evidence(tmp_path)
    outcomes=[]
    for year in YEARS:
        actual=.05 if year%2==0 else -.05
        outcomes.extend([actual>0]*24)
        for kind in KINDS:
            path=tmp_path/'development'/f'{kind}_{year}.parquet';frame=pd.read_parquet(path)
            frame['fwd_return']=actual;frame['probability_up']=.5;frame.to_parquet(path,index=False)
    result=compare_components(tmp_path)['horizons']['20']['factor']
    expected=_ece(np.array(outcomes,dtype=float),np.full(192,.5))[0]
    assert result['overall']['ece']==pytest.approx(expected)
    assert result['overall']['ece']!=pytest.approx(np.mean([v['ece'] for v in result['annual'].values()]))
    assert result['overall']['baseline_brier']==pytest.approx(.25)


@pytest.mark.parametrize('task',['probability_up','intervals'])
def test_all_models_failing_original_gates_prevent_configuration_write(tmp_path,monkeypatch,task):
    write_evidence(tmp_path)
    for path in (tmp_path/'development').glob('*.parquet'):
        frame=pd.read_parquet(path)
        if task=='probability_up':frame['probability_up']=.5
        else:
            frame['q10']=frame.fwd_return-.01
            frame['q50']=frame.fwd_return
            frame['q90']=frame.fwd_return+.01
        frame.to_parquet(path,index=False)
    monkeypatch.setattr('sys.argv',['component_selection','--results-root',str(tmp_path)])
    with pytest.raises(ValueError,match='合格'):
        main()
    assert not (tmp_path/'component_selection.json').exists()


def test_failed_interval_does_not_remove_valid_ranking_probability_or_return(tmp_path):
    write_evidence(tmp_path)
    path=tmp_path/'development'/'factor_2023.parquet'
    frame=pd.read_parquet(path)
    failed=frame.security_id.eq('S00') & frame.horizon.eq(20)
    frame.loc[failed,['status','interval_status']]='invalid_quantile_order'
    frame.loc[failed,TASK_COLUMNS['intervals']]=np.nan
    frame.to_parquet(path,index=False)
    metrics=compare_components(tmp_path)
    for kind in KINDS:
        annual=metrics['horizons']['20'][kind]['annual']['2023']
        assert annual['task_samples']=={'score':24,'probability_up':24,'intervals':23,'expected_return':24}
        assert metrics['horizons']['20'][kind]['overall']['task_samples']=={
            'score':192,'probability_up':192,'intervals':191,'expected_return':192}
    assert metrics['coverage']['20']['score']['annual']['2023']['scorable_rows']==dict.fromkeys(KINDS,24)
    assert metrics['coverage']['20']['intervals']['annual']['2023']['scorable_rows']=={
        'factor':23,'linear':24,'lightgbm_small':24}


def test_interval_candidate_with_model_rejection_cannot_be_selected_for_freeze(tmp_path):
    write_evidence(tmp_path)
    for year in YEARS:
        path=tmp_path/'development'/f'lightgbm_small_{year}.parquet'
        frame=pd.read_parquet(path)
        failed=frame.security_id.eq('S00') & frame.horizon.eq(20)
        frame.loc[failed,['status','interval_status']]='invalid_quantile_order'
        frame.loc[failed,TASK_COLUMNS['intervals']]=np.nan
        frame.to_parquet(path,index=False)
    metrics=compare_components(tmp_path)
    assert metrics['coverage']['20']['intervals']['model_rejected_rows']['lightgbm_small']==8
    heads=choose_component_heads(metrics)
    assert heads['20']['intervals']!='lightgbm_small'


def test_old_global_status_only_predictions_require_explicit_regeneration(tmp_path):
    write_evidence(tmp_path)
    path=tmp_path/'development'/'factor_2016.parquet'
    frame=pd.read_parquet(path).drop(columns=list(TASK_STATUS_COLUMNS.values()))
    frame.to_parquet(path,index=False)
    with pytest.raises(ValueError,match='重新生成预测'):
        compare_components(tmp_path)


def test_each_overall_loss_uses_its_own_task_sample_count():
    from hk_quant.component_selection import _overall
    annual={
        '2016':{'task_samples':{'score':20,'probability_up':2,'intervals':1,'expected_return':4},
                'score_dates':1,'score':1.,'probability_up':.1,'baseline_brier':.25,
                'intervals':.5,'interval_coverage':.7,'baseline_pinball':.6,'expected_return':2.},
        '2017':{'task_samples':{'score':40,'probability_up':1,'intervals':3,'expected_return':1},
                'score_dates':1,'score':-1.,'probability_up':.4,'baseline_brier':.25,
                'intervals':.1,'interval_coverage':.9,'baseline_pinball':.2,'expected_return':10.},
    }
    probabilities=[(np.array([0.,1.]),np.array([.2,.8])),(np.array([1.]),np.array([.6]))]
    value=_overall(annual,probabilities)
    assert value['task_samples']=={'score':60,'probability_up':3,'intervals':4,'expected_return':5}
    assert value['score']==pytest.approx(0.)
    assert value['probability_up']==pytest.approx(.2)
    assert value['intervals']==pytest.approx(.2)
    assert value['interval_coverage']==pytest.approx(.85)
    assert value['baseline_pinball']==pytest.approx(.3)
    assert value['expected_return']==pytest.approx(3.6)
    assert value['ece']==pytest.approx(.8/3)
