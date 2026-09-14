"""共享输入、按任务固定选择模型的统一推理对象。"""
import numpy as np
import pandas as pd

from . import HORIZONS
from .contracts import FORECAST_RETURN_BASIS, CASH_DIVIDEND_ACCOUNTING
from .models import NUMERIC_OUTPUTS
from .prediction_tasks import (TASK_COLUMNS, TASK_STATUS_COLUMNS, PREDICTION_SCHEMA,
                               STATUS_REASONS, aggregate_task_status, validate_task_outputs)


class MultiTaskBundle:
    def __init__(self, models, heads, model_version):
        if set(map(str,heads)) != set(map(str,HORIZONS)):
            raise ValueError('固定任务配置必须覆盖四个期限')
        self.heads={str(h):dict(heads[str(h)] if str(h) in heads else heads[h]) for h in HORIZONS}
        for configuration in self.heads.values():
            if set(configuration) != set(TASK_COLUMNS):raise ValueError('任务配置不完整或包含未知任务')
        needed=set(kind for configuration in self.heads.values() for kind in configuration.values())
        if not needed <= set(models):raise ValueError('缺少配置指定的模型，不能改用其他模型')
        self.models={kind:models[kind] for kind in sorted(needed)}
        cutoffs={pd.Timestamp(model.as_of) for model in self.models.values()}
        if len(cutoffs)!=1:raise ValueError('任务模型训练截止日期必须一致')
        self.as_of=cutoffs.pop()
        self.model_version=model_version
        self.feature_columns=list(dict.fromkeys(column for model in self.models.values() for column in model.feature_columns))
        self.metadata={'kind':'multitask_bundle','model_version':model_version,'as_of':self.as_of.isoformat(),
            'horizons':list(HORIZONS),'heads':self.heads,'features':self.feature_columns,
            'components':{kind:model.metadata for kind,model in self.models.items()},
            'return_basis':FORECAST_RETURN_BASIS,'cash_dividend_accounting':CASH_DIVIDEND_ACCOUNTING,
            'prediction_schema':PREDICTION_SCHEMA,
            'missing_policy':'Each selected task retains its own status and real outputs. Failures in unselected tasks do not suppress valid selected tasks; unavailable selected tasks never switch models.'}

    def predict(self, examples, explain=False):
        out=examples.copy().reset_index(drop=True)
        predictions={kind:model.predict(out,explain=explain) for kind,model in self.models.items()}
        return self.combine_predictions(out,predictions,explain)

    def combine_predictions(self, examples, predictions, explain=False):
        """同一组合规则也用于已落盘的开发预测，避免评价与线上调用不一致。"""
        out=examples.copy().reset_index(drop=True)
        keys=['date','security_id','horizon']
        if not set(keys)<=set(out):raise ValueError('统一推理缺少日期、证券或期限')
        if set(predictions)!=set(self.models):raise ValueError('预测来源与固定任务模型不一致')
        for kind,frame in predictions.items():
            if len(frame)!=len(out) or not frame[keys].reset_index(drop=True).equals(out[keys]):
                raise ValueError(f'{kind}改变了预测行顺序或证券日期')
            validate_task_outputs(frame)
        for column in NUMERIC_OUTPUTS:out[column]=np.nan
        out['status']='insufficient_model_inputs'
        for field in TASK_STATUS_COLUMNS.values():out[field]='insufficient_model_inputs'
        out['prediction_schema']=PREDICTION_SCHEMA
        out['reasons']=''
        out['model_version']=self.model_version
        out['data_as_of']=pd.to_datetime(out.date).dt.strftime('%Y-%m-%d')
        out['model_trained_as_of']=self.as_of.isoformat()
        out['return_basis']=FORECAST_RETURN_BASIS
        if explain:
            out['explanations']=[[] for _ in range(len(out))]
            out['explanation_basis']=None
        for horizon in HORIZONS:
            mask=out.horizon.eq(horizon)
            configuration=self.heads[str(horizon)]
            for task,kind in configuration.items():
                component=predictions[kind]
                field=TASK_STATUS_COLUMNS[task]
                out.loc[mask,field]=component.loc[mask,field]
                available=component[field].eq('ok')
                failed=mask & ~available
                reason=component.loc[failed,field].map(STATUS_REASONS)
                out.loc[failed,'reasons'] += task+' ('+kind+'): '+reason+'；'
                columns=TASK_COLUMNS[task]
                out.loc[mask & available,columns]=component.loc[mask & available,columns].to_numpy()
            if explain:
                for index in out.index[mask]:
                    contributions=[]
                    for task in ('score','expected_return'):
                        if out.at[index,TASK_STATUS_COLUMNS[task]]!='ok':continue
                        kind=configuration[task]
                        for item in predictions[kind].at[index,'explanations']:
                            contribution={**item,'task':task,'component_model':kind}
                            if task=='score' and item['basis']=='predicted_arithmetic_return':
                                scale=float(self.models[kind]._return_scale(examples.iloc[[index]])[0])
                                contribution['contribution']=item['contribution']/scale
                                contribution['basis']='rank_score'
                            contributions.append(contribution)
                    out.at[index,'explanations']=contributions
                    if contributions:out.at[index,'explanation_basis']='actual_selected_rank_and_return_models'
        out['status']=aggregate_task_status(out)
        unsupported=~out.horizon.isin(HORIZONS)
        out.loc[unsupported,'reasons']='仅支持1、5、20、60个交易日'
        return out
