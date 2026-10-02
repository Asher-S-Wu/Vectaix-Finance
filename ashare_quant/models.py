"""Independent CNY artifact identity, sharing only estimator mathematics."""
from hk_quant.models import UniversalModel
from . import FORECAST_RETURN_BASIS


class AShareModel(UniversalModel):
    def fit(self,train,calibration,feature_columns,as_of):
        super().fit(train,calibration,feature_columns,as_of)
        from datetime import datetime,timezone
        self.metadata['fit_executed_at']=datetime.now(timezone.utc).isoformat()
        self.metadata['evaluation_context']='Retrospective chronological research; not a claim of live deployment at the logical training cutoff'
        self.metadata.update(market='CN',currency='CNY',return_basis=FORECAST_RETURN_BASIS,
            sigma_basis='historical unranked CNY source-adjusted daily log-price-return standard deviation',
            cash_dividend_accounting='Source-adjusted research return; no separate actual-share dividend cash entitlement implied.')
        return self

    def predict(self,examples,explain=False):
        out=super().predict(examples,explain=explain)
        out['return_basis']=FORECAST_RETURN_BASIS
        return out
