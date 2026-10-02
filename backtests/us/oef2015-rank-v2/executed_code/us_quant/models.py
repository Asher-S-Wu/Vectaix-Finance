"""US artifact identity, sharing estimator mathematics but never trained state."""
from datetime import datetime,timezone
from hk_quant.models import UniversalModel
from . import RETURN_BASIS


class USModel(UniversalModel):
    market='US'

    def fit(self,train,calibration,feature_columns,as_of):
        super().fit(train,calibration,feature_columns,as_of)
        self.metadata.update(market='US',currency='USD',return_basis=RETURN_BASIS,
            sigma_basis='unranked historical USD source-adjusted daily log-return volatility',
            cash_dividend_accounting='Source-adjusted research returns; no separate actual-share dividend cash entitlement',
            fit_executed_at=datetime.now(timezone.utc).isoformat(),
            evaluation_context='Retrospective current-vintage public-price research, conditional on observable historical cohort coverage',
            execution_validated=False,eligible=False)
        return self

    def predict(self,examples,explain=False):
        out=super().predict(examples,explain=explain)
        out['return_basis']=RETURN_BASIS
        return out
