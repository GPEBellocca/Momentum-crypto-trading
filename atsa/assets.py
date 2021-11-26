from enum import Enum
from atsa.config import UNKNOWN_SECTOR_NAME
import pandas as pd
from os.path import join


class AssetType(Enum):
    STOCK = 1
    ## TODO: CRYPTOCURRENCIES = 2 => use class Stock or set a new class for Crypto?!


class Asset:
    def __init__(self, asset_type: AssetType):
        self.asset_type = asset_type


class Stock(Asset):

    def __init__(self, name, openp, close, high, low, volume, sector=None):
        Asset.__init__(self, AssetType.STOCK)
        self.name = name
        self.open = openp
        self.close = close
        self.high = high
        self.low = low
        self.volume = volume

        if not sector:
            self.sector = UNKNOWN_SECTOR_NAME
        else:
            self.sector = sector        

        self.continuous_df = None   # TODO maybe it can be removed
        self.discrete_df = None     # TODO maybe it can be removed
        self.continuous_df_filtered = None
        self.discrete_df_filtered = None
        self.first_valid_row = None

    def __repr__(self):
        return f"Stock(name={self.name},sector={self.sector})"

    def __getstate__(self):
        return (self.asset_type, self.name, self.open, self.close, self.high, self.low, self.volume,
                self.first_valid_row)

    def __setstate__(self, state):
        (self.asset_type, self.name, self.open, self.close, self.high, self.low, self.volume,
         self.first_valid_row) = state

    def to_csv(self, out_dir):
        self.continuous_df = pd.concat([self.open, self.close, self.high, self.low, self.volume], axis=1).drop_duplicates().rename(columns={
            f"{self.name}_Open": "Open",
            f"{self.name}_High": "High",
            f"{self.name}_Low": "Low",
            f"{self.name}_Volume": "Volume",
            self.name: "Close" 
        })
        self.continuous_df.to_csv(join(out_dir, f"{self.name}.csv"))
