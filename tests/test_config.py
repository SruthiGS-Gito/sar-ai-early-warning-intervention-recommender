import sar
from sar import config


def test_package_imports():
    assert sar.__version__


def test_cutoff_is_day_30():
    assert config.CUTOFF_DAY == 30


def test_seven_oulad_tables_listed():
    assert len(config.OULAD_TABLES) == 7
    assert len(set(config.OULAD_TABLES)) == 7
