import pytest


@pytest.mark.skip(reason="Ticket 110: enable once feature code exists")
def test_no_feature_uses_data_after_cutoff():
    """For each feature builder, rows dated after CUTOFF_DAY must not change the output."""
