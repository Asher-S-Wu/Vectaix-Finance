import numpy as np
import pandas as pd
import pytest

from hk_quant.training_snapshot import apply_label_patches


def fixture():
    frame = pd.DataFrame(dict(date=pd.to_datetime(['2020-01-02','2020-01-03']),
        security_id=['A','B'], factor=[3.,4.], status=['ok','data_issue'],
        fwd_return_20=[np.nan,.03], label_end_20=pd.to_datetime(['2020-01-30','2020-01-31'])))
    patch = pd.DataFrame(dict(date=pd.to_datetime(['2020-01-02']), security_id=['A'], horizon=[20],
        label_end=pd.to_datetime(['2020-01-30']), old_fwd_return=[np.nan], new_fwd_return=[-1.],
        source_url=['https://issuer.test/verified'], reason=['reviewed_outcome']))
    return frame, patch


def test_label_patch_changes_only_proven_missing_return():
    frame, patch = fixture()
    original = frame.copy(deep=True)
    result = apply_label_patches(frame, patch, '2023-12-31')
    assert result.loc[0,'fwd_return_20'] == -1.
    pd.testing.assert_frame_equal(frame, original)
    expected = original.copy(); expected.loc[0,'fwd_return_20'] = -1.
    pd.testing.assert_frame_equal(result, expected)


@pytest.mark.parametrize('change,match',[
    ('existing','已有'),('future','截止'),('endpoint','结束日'),('unknown','不存在'),
    ('duplicate','重复'),('impossible','收益')])
def test_label_patch_rejects_unproved_or_conflicting_changes(change, match):
    frame, patch = fixture()
    if change == 'existing': frame.loc[0,'fwd_return_20'] = .1
    if change == 'future': patch['label_end'] = pd.Timestamp('2024-01-01')
    if change == 'endpoint': patch['label_end'] = pd.Timestamp('2020-01-31')
    if change == 'unknown': patch['security_id'] = 'C'
    if change == 'duplicate': patch = pd.concat([patch, patch])
    if change == 'impossible': patch['new_fwd_return'] = -1.01
    with pytest.raises(ValueError, match=match):
        apply_label_patches(frame, patch, '2023-12-31')
