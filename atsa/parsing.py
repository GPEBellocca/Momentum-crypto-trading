import logging
import os
import pickle
from atsa.assets import Stock
import pandas as pd
from atsa.enum import AssetCount, Classifier, Validation, OperationAllowed
from atsa.config import UNKNOWN_SECTOR_NAME


def _date_parser(x):
    return pd.to_datetime(x, yearfirst=True, format='%Y-%m-%d')


def load_stocks(directory: str):
    files = [f for f in os.listdir(directory) if f != ".DS_Store"]
    stocks = list()
    for f in files:
        df = pd.read_csv(os.path.join(directory, f), index_col=0, header=0, parse_dates=True)
        stocks.append(Stock(f.split(".")[0], df.Open, df.Close, df.High, df.Low, df.Volume)) 
    return stocks


def load_stocks_by_year(directory: str, year: str):
    return [Stock(s.name,
                  s.open.loc[year],
                  s.close.loc[year],
                  s.high.loc[year],
                  s.low.loc[year],
                  s.volume.loc[year]) for s in load_stocks(directory)]


class FTSEMIBInfo:

    # These stock data present errors or wrong values hence are discarded
    STOCKS_TO_DISCARD = [
        'UBI.MI',           # contains erroneous values in Close
        'UNI.MI',           # contains erroneous values in Close
    ]

    #: Categories used for FTSE MIB index
    SECTORS = {
        'UTIL': ['A2A.MI', 'ENEL.MI', 'SRG.MI', 'TIT.MI', 'TRN.MI'],
        'CONS': ['AGL.MI', 'MONC.MI', 'CPR.MI', 'FCA.MI', 'LUX.MI', 'MS.MI', 'SFER.MI', 'TOD.MI'],
        'FINSERV': ['AZM.MI', 'BPE.MI', 'EXO.MI', 'FBK.MI', 'G.MI', 'ISP.MI', 'MB.MI', 'UBI.MI', 'UCG.MI', 'UNI.MI',
                    'US.MI'],
        'ENERGY': ['ENI.MI', 'SPM.MI', 'TEN.MI'],
        'TECH': ['PRY.MI', 'STM.MI']
    }


class SP500Info:

    STOCKS_TO_DISCARD = [
        'BHF'               # contains erroneous values in Volume
    ]

    def __init__(self):
        self.SECTORS = {}

    def read_from_file(self, path_to_file):
        df = pd.read_csv(path_to_file, sep=';', header=0)
        sector_col = df.iloc[:, 2]
        unique_sectors = sector_col.unique()

        for sector in unique_sectors:
            if type(sector) != str:
                continue

            # Remove spaces in names to avoid conflicts with file system operations
            sector_safe_name = sector.replace(" ", "_")
            self.SECTORS[sector_safe_name] = df.loc[sector_col == sector]["Symbol"].tolist()


class CRYPTOInfo:

    ## TODO: cancel => managed by filter_stocks()
    # These stock data present errors or wrong values hence are discarded
    STOCKS_TO_DISCARD = [
        'USDT', #2015, 2016
        #'DOGE', #2014
        'VEN' #2013, 2014, 2015, 2016, 2018
        #'XRP', #2015
        #'DASH'  #2014, 2015 many NaN indicators
        #'ADA'	#2018 (volume=0)
    ]

    SECTORS = {
        'BTC_LIKE': ['BTC', 'BCH', 'BTG', 'DASH', 'LTC', 'XEM', 'XMR', 'ZEC'],
        'ETH_LIKE': ['ETH', 'BNB', 'ETC', 'EOS', 'LINK', 'NEO', 'VEN', 'ZRX']
    }
    """
    # All cryptocurrencies are considered belonging to the same 'CRYPTO' sector
    SECTORS = {
        'CRYPTO': ['BTC','ADA','BCH','BNB','BTG','DASH','DOGE',
            'EOS','ETC','ETH','IOT','LINK','LTC','NEO','QTUM',
            'TRX','USDT','VEN','WAVES','XEM','XMR','XRP','ZEC','ZRX'
        ]
    }
    """


class BaseParser:
    def __init__(self):
        self.logger = logging.getLogger(__name__)


class FTSEMIBParser(BaseParser):

    def __init__(self, cdl_file: str, vol_file: str):
        super().__init__()
        self.cdl_file = cdl_file
        self.vol_file = vol_file
        self.cdl_df = None
        self.vol_df = None

    def parse_csv(self) -> (pd.DataFrame, pd.DataFrame):
        """
        Read and pre-process data:
        - 1. filter only stocks present in both the files
        - 2. drop any stock containing NaN values -> 4lice not for Crypto!
        """
        stem_sep = '_'

        cdl = pd.read_csv(self.cdl_file, header=0, index_col=0, parse_dates=True, date_parser=_date_parser)
        vol = pd.read_csv(self.vol_file, header=0, index_col=0, parse_dates=True, date_parser=_date_parser)

        self.logger.info('PRE - Candle file contains {} stocks'.format(int(len(cdl.columns) / 4)))
        self.logger.info('PRE - Volume file contains {} stocks'.format(int(len(vol.columns) / 2)))

        # Find stocks in both the files
        stem_cdl = set([x[0] for x in [x.split(stem_sep) for x in cdl.columns.tolist()]])
        stem_vol = set([x[0] for x in [x.split(stem_sep) for x in vol.columns.tolist()]])
        join_stem = stem_cdl & stem_vol

        to_drop = []
        for column in cdl.columns.values:
            stem = column.split(stem_sep)[0]
            if stem not in join_stem:
                to_drop += [column]
        self.logger.debug('Dropping {} from candle file'.format(to_drop))
        cdl = cdl.drop(columns=to_drop)

        to_drop = []
        for column in vol.columns.values:
            stem = column.split(stem_sep)[0]
            if stem not in join_stem:
                to_drop += [column]
        self.logger.debug('Dropping {} from volume file'.format(to_drop))
        vol = vol.drop(columns=to_drop)

        # drop any stock containing NaN values
        nan_cdl = set([x[0] for x in [x.split(stem_sep) for x in cdl.columns[cdl.isna().any()].tolist()]])
        nan_vol = set([x[0] for x in [x.split(stem_sep) for x in vol.columns[vol.isna().any()].tolist()]])
        if nan_cdl or nan_vol:
            join_set = nan_cdl | nan_vol
            self.logger.info('Found NaN values in {}. Discarding related columns...'.format(join_set))

            to_delete_cdl = set()
            to_delete_vol = set()
            for n in join_set:
                to_delete_cdl.add(n)
                to_delete_cdl.add(n + "_High")
                to_delete_cdl.add(n + "_Low")
                to_delete_cdl.add(n + "_Open")
                to_delete_vol.add(n)
                to_delete_vol.add(n + "_Volume")

            cdl = cdl.drop(columns=list(to_delete_cdl))
            vol = vol.drop(columns=list(to_delete_vol))

        self.logger.info('POST - Candle file contains {} stocks'.format(int(len(cdl.columns) / 4)))
        self.logger.info('POST - Volume file contains {} stocks'.format(int(len(vol.columns) / 2)))

        self.cdl_df = cdl
        self.vol_df = vol
        return cdl, vol

    def get_stocks(self, index_info) -> list:
        """
        Parse stock information from cdl and volume dataframes producing Stock objects.
        :return: a list of Stock objects.
        """
        if self.vol_df.empty or self.cdl_df.empty:
            raise RuntimeError("You must parse csv file before.")

        stocks = []
        for i in range(int(self.cdl_df.shape[1] / 4)):
            # A2A.MI_Open, A2A.MI_High, A2A.MI_Low, A2A.MI
            base = i * 4
            _open = self.cdl_df.iloc[:, base]
            _high = self.cdl_df.iloc[:, base + 1]
            _low = self.cdl_df.iloc[:, base + 2]
            _close = self.cdl_df.iloc[:, base + 3]

            self.logger.debug("Processing Stock: " + _close.name)
            # Find volume column

            df_vol = self.vol_df.filter(regex="^" + _close.name + "_Volume$")
            if df_vol.shape[1] > 1:
                raise RuntimeError("More than one volume column has been found. Investigate.")

            _volume = df_vol.iloc[:, 0]
            if _open.size == _high.size and \
                    _open.size == _low.size and \
                    _open.size == _close.size and \
                    _open.size == _volume.size:

                pick_sector = UNKNOWN_SECTOR_NAME
                for sector, items in index_info.SECTORS.items():
                    if _close.name in items:
                        pick_sector = sector
                        break
                stocks += [Stock(_close.name, _open, _close, _high, _low, _volume, pick_sector)]
            else:
                raise RuntimeError("Series associated to the same stock have different length.")

        return stocks


#UPDATE : Parser dedicated to Cryptocurrencies
class CRYPTOParser(BaseParser):

    def __init__(self, cdl_file: str, vol_file: str):
        super().__init__()
        self.cdl_file = cdl_file
        self.vol_file = vol_file
        self.cdl_df = None
        self.vol_df = None

    def parse_csv(self) -> (pd.DataFrame, pd.DataFrame):
        """
        Read and pre-process data:
        - 1. filter only stocks present in both the files
        - 2. manage NaN values
        """
        stem_sep = '_'

        cdl = pd.read_csv(self.cdl_file, header=0, index_col=0, parse_dates=True, date_parser=_date_parser)
        vol = pd.read_csv(self.vol_file, header=0, index_col=0, parse_dates=True, date_parser=_date_parser)

        self.logger.info('PRE - Candle file contains {} stocks'.format(int(len(cdl.columns) / 4)))
        self.logger.info('PRE - Volume file contains {} stocks'.format(int(len(vol.columns) / 2)))

        # Find stocks in both the files
        stem_cdl = set([x[0] for x in [x.split(stem_sep) for x in cdl.columns.tolist()]])
        stem_vol = set([x[0] for x in [x.split(stem_sep) for x in vol.columns.tolist()]])
        join_stem = stem_cdl & stem_vol

        to_drop = []
        for column in cdl.columns.values:
            stem = column.split(stem_sep)[0]
            if stem not in join_stem:
                to_drop += [column]
        self.logger.debug('Dropping {} from candle file'.format(to_drop))
        cdl = cdl.drop(columns=to_drop)

        to_drop = []
        for column in vol.columns.values:
            stem = column.split(stem_sep)[0]
            if stem not in join_stem:
                to_drop += [column]
        self.logger.debug('Dropping {} from volume file'.format(to_drop))
        vol = vol.drop(columns=to_drop)

        # manage NaN values
        #UPDATE : Series with NaN values are not discarded
        nan_cdl = set([x[0] for x in [x.split(stem_sep) for x in cdl.columns[cdl.isna().any()].tolist()]])
        nan_vol = set([x[0] for x in [x.split(stem_sep) for x in vol.columns[vol.isna().any()].tolist()]])
        if len(nan_cdl) > 0 or len(nan_vol) > 0:
            join_set = nan_cdl | nan_vol
            self.logger.info('Found NaN values in: {}'.format(join_set))

        self.logger.info('POST - Candle file contains {} stocks'.format(int(len(cdl.columns) / 4)))
        self.logger.info('POST - Volume file contains {} stocks'.format(int(len(vol.columns) / 2)))

        self.cdl_df = cdl
        self.vol_df = vol
        return cdl, vol

    def get_stocks(self, index_info) -> list:
        """
        Parse stock information from cdl and volume dataframes producing Stock objects.
        :return: a list of Stock objects.
        """
        if self.vol_df.empty or self.cdl_df.empty:
            raise RuntimeError("You must parse csv file before.")

        stocks = []

        for i in range(int(self.cdl_df.shape[1] / 4)):
            # A2A.MI_Open, A2A.MI_High, A2A.MI_Low, A2A.MI
            base = i * 4
            _open = self.cdl_df.iloc[:, base]
            _high = self.cdl_df.iloc[:, base + 1]
            _low = self.cdl_df.iloc[:, base + 2]
            _close = self.cdl_df.iloc[:, base + 3]

            self.logger.debug("Processing Stock: " + _close.name)
            # Find volume column

            df_vol = self.vol_df.filter(regex="^" + _close.name + "_Volume$")
            if df_vol.shape[1] > 1:
                raise RuntimeError("More than one volume column has been found. Investigate.")

            _volume = df_vol.iloc[:, 0]
            if _open.size == _high.size and \
                    _open.size == _low.size and \
                    _open.size == _close.size and \
                    _open.size == _volume.size:

                #pick_sector = 'CRYPTO'
                pick_sector = UNKNOWN_SECTOR_NAME
                for sector, items in index_info.SECTORS.items():
                    if _close.name in items:
                        pick_sector = sector
                        break

                stocks += [Stock(_close.name, _open, _close, _high, _low, _volume, pick_sector)]
            else:
                raise RuntimeError("Series associated to the same stock have different length.")

        return stocks

    #UPDATE : filter unconsistent datasets
    def filter_stocks(self, input_stocks, type_of_analysis):

        year = str(self.vol_df.index[(self.vol_df.shape[0]-1)]).split('-')[0] #skip rows used for technical indicators computation
        ## DEBUG: print('year {} , row[ {} ] = {} '.format(str(year), str(self.vol_df.shape[0]), str(self.vol_df.iloc[(self.vol_df.shape[0]-1),:]))
        stocks = []

        if year == '2013':
            if type_of_analysis == AssetCount.SINGLE:
                stocks = [s for s in input_stocks if s.name not in ['VEN']]
                self.logger.info('Discarded stocks: { VEN }')
            else:
                stocks = [s for s in input_stocks if s.name not in ['LTC','VEN']]
                self.logger.info('Discarded stocks: { VEN, LTC }')
        elif year == '2014':
            stocks = [s for s in input_stocks if s.name not in ['DASH','DOGE','VEN']]
            self.logger.info('Discarded stocks: { DASH, DOGE, VEN }')
        elif year == '2015':
            if type_of_analysis == AssetCount.SINGLE:
                stocks = [s for s in input_stocks if s.name not in ['DASH','USDT','VEN','XRP']]
                self.logger.info('Discarded stocks: { DASH, USDT, VEN, XRP }')
            else:
                stocks = [s for s in input_stocks if s.name not in ['DASH','ETH','USDT','VEN','XEM','XMR','XRP']]
                self.logger.info('Discarded stocks: { DASH, ETH, USDT, VEN, XEM, XMR, XRP }')
        elif year == '2016':
            if type_of_analysis == AssetCount.SINGLE:
                stocks = [s for s in input_stocks if s.name not in ['USDT','VEN']]
                self.logger.info('Discarded stocks: { USDT, VEN }')
            else:
                stocks = [s for s in input_stocks if s.name not in ['ETC','USDT','VEN','WAVES','ZEC']]
                self.logger.info('Discarded stocks: { ETC, USDT, VEN, WAVES, ZEC }')
        elif year == '2017':
            if type_of_analysis == AssetCount.SINGLE:
                stocks = [s for s in input_stocks if s.name not in ['VEN']]
                self.logger.info('Discarded stocks: { VEN }')
            else:
                stocks = [s for s in input_stocks if s.name not in ['ADA','BCH','BNB','BTG','EOS','IOT','LINK','NEO','QTUM','TRX','USDT','VEN','ZRX']]
                self.logger.info('Discarded stocks: { ADA, BCH, BNB, BTG, EOS, IOT, LINK, NEO, QTUM, TRX, USDT, VEN, ZRX }')
        elif year == '2018':
            stocks = [s for s in input_stocks if s.name not in ['ADA','VEN']]
            self.logger.info('Discarded stocks: { ADA, VEN }')
        else:
            stocks = input_stocks

        self.logger.info('FILTERED - Stock file contains {} stocks'.format(int(len(stocks))))

        return stocks


class IndexClassificationParser(BaseParser):
    def __init__(self, root_dir: str, operation_allowed: str=None):
        super().__init__()
        self.root_dir = root_dir
        self.operation_allowed = operation_allowed

    def read_index_classification_result(self, file: str, operation_allowed: OperationAllowed):
        filename = os.path.splitext(file)[0]
        tok = filename.split('_')
        params = tok[2]
        pars = str(params[1:-1]).split(',')

        if tok[0] == str(Validation.HOLD_UNI):
            validation = Validation.HOLD_UNI
        elif tok[0] == str(Validation.EXP_UNI):
            validation = Validation.EXP_UNI
        elif tok[0] == str(Validation.HOLD_MULTI):
            validation = Validation.HOLD_MULTI
        elif tok[0] == str(Validation.EXP_MULTI):
            validation = Validation.EXP_MULTI
        else:
            raise RuntimeError('Found {} as validation in file {}: not valid'.format(tok[0], file))

        if tok[1] == str(Classifier.ARIMA):
            classifier = ARIMAClassifier(pars, operation_allowed)

        elif tok[1] == str(Classifier.LINREG):
            classifier = LinearRegressionClassifier(operation_allowed)

        elif tok[1] == str(Classifier.EXPSMOOTH):
            alpha = float(pars[0])
            if pars[1] == "None":
                beta = None
            else:
                beta = float(pars[1])
            classifier = ExponentialSmoothingClassifier(operation_allowed, alpha=alpha, beta=beta)

        elif tok[1] == str(Classifier.VAR):
            order = int(pars[0])
            classifier = VARClassifier(operation_allowed, order=order)

        elif tok[1] == str(Classifier.L3):
            if len(pars[3]) > 0:
                top_count = int(pars[3])
            else:
                top_count = None

            # L3 root is not needed if not using L3Classifier for classification
            classifier = L3Classifier(operation_allowed, pars[0], pars[1], "", rule_set=pars[2], top_count=top_count)

        elif tok[1] == str(Classifier.MLP):
            hidden_layer_sizes = '{},{}'.format(pars[0], pars[1])
            solver = pars[2]
            classifier = MLPClassifier(operation_allowed, hidden_layer_sizes=hidden_layer_sizes, solver=solver)

        elif tok[1] == str(Classifier.RFC):
            n_estimators = int(pars[0])
            criterion = pars[1]
            classifier = RFClassifier(operation_allowed, n_estimators=n_estimators, criterion=criterion)

        elif tok[1] == str(Classifier.SVC):
            classifier = SVClassifier(operation_allowed, kernel=pars[0])

        elif tok[1] == str(Classifier.MNB):
            alpha = float(pars[0])
            classifier = MNBClassifier(operation_allowed, alpha=alpha)

        elif tok[1] == str(Classifier.KNN):
            n_neighbors = int(pars[0])
            classifier = KNNClassifier(operation_allowed, n_neighbors=n_neighbors)

        else:
            raise RuntimeError("Classifier not valid")

        df = pd.read_csv(os.path.join(self.root_dir, file), index_col=0)
        self.logger.debug("In index classification parser read df with shape", df.shape)
        return IndexClassificationResult(validation, classifier, df)
