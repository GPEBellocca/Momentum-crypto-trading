"""
Collector of several custom Enums.

These enums are mainly used for standardization around in the code. They are frequently
used as values for config.py global variables.
"""

from enum import Enum

class BaseEnum(Enum):
    def __str__(self):
        return self.value


class StockExchangeIndex(BaseEnum):
    FTSEMIB = "FTSEMIB"
    SP500 = "SP500"
    # CRYPTO = "CryptocompareAPI" # TODO move from here


class AssetCount(BaseEnum):
    SINGLE = 'SINGLE'
    MULTI_CONCAT_ROWS = 'MULTI_CONCAT_ROWS'
    MULTI_CONCAT_COLS = 'MULTI_CONCAT_COLS'


class Classifier(BaseEnum):
    LINREG = "LINREG"
    ARIMA = "ARIMA"
    EXPSMOOTH = "EXPSMOOTH"
    VAR = "VAR"
    L3 = "L3"
    MLP = "MLP"
    RFC = "RFC"
    SVC = "SVC"
    GNB = "GNB"
    KNN = "KNN"


class ClassificationStrategy(BaseEnum):
    HOLDOUT = "HOLDOUT"
    EXPANDING = "EXPANDING"
    HOLDOUT_EXPANDING = "HOLDOUT_EXPANDING"


class Validation(BaseEnum):
    EXP_UNI = "exp-uni"
    EXP_MULTI = "exp-multi"
    HOLD_UNI = "hold-uni"
    HOLD_MULTI = "hold-multi"


class OperationAllowed(BaseEnum):
    LS = "LONG-SHORT"
    L = "LONG"
    S = "SHORT"


class OperationLength(BaseEnum):
    MD = "Multiple days"
    OD = "One day"


class TradingOperationType(BaseEnum):
    LONG = "LONG"
    SHORT = "SHORT"