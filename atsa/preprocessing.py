import logging
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

from atsa.config import LABELS, CLASS_TH, W
from atsa.enum import OperationAllowed
from atsa.assets import Stock
from l3wrapper.dictionary import RuleDictionary
from pyti.simple_moving_average import simple_moving_average
from pyti.exponential_moving_average import exponential_moving_average
from pyti.aroon import aroon_oscillator
from pyti.directional_indicators import *
from pyti.price_oscillator import price_oscillator
from pyti.relative_strength_index import relative_strength_index
from pyti.money_flow_index import money_flow_index
from pyti.true_strength_index import true_strength_index
from pyti.stochastic import percent_k
from pyti.chande_momentum_oscillator import chande_momentum_oscillator
from pyti.average_true_range_percent import average_true_range_percent
from pyti.volume_oscillator import volume_oscillator
from pyti.force_index import force_index
from pyti.accumulation_distribution import accumulation_distribution
from pyti.on_balance_volume import on_balance_volume

# UNIVAR_COL = ['SMA5-20', 'SMA8-15', 'SMA20-50',
#               'EMA5-20', 'EMA8-15', 'EMA20-50',
#               'MACD12-26', 'AO14', 'ADX14', 'WD14',
#               'PPO12_26', 'RSI14', 'MFI14', 'TSI',
#               'SO14', 'CMO14', 'ATRP14', 'PVO14',
#               'FI13', 'FI50', 'ADL', 'OBV',
#               'class']
#
# MULTIVAR_COL = ['SMA5-20', 'SMA8-15', 'SMA20-50',
#                 'EMA5-20', 'EMA8-15', 'EMA20-50',
#                 'MACD12-26', 'AO14', 'ADX14', 'WD14',
#                 'PPO12_26', 'RSI14', 'MFI14', 'TSI',
#                 'SO14', 'CMO14', 'ATRP14', 'PVO14',
#                 'class']

FEATURE_TO_DROP_MULTIVAR = ['FI13', 'FI50', 'ADL', 'OBV']


def make_discrete_single_(x, threshold):
    if np.isnan(x):
        return "nan"
    if x < threshold:
        return 1
    # elif x == threshold:
    #     return 2
    else:
        return 2


def make_discrete_double_(x, l_threshold, h_threshold):
    if np.isnan(x):
        return "nan"
    if x <= l_threshold:
        return 1
    elif l_threshold < x < h_threshold:
        return 2
    else:
        return 3


def make_class_label_(x, l_threshold, h_threshold):
    if np.isnan(x):
        return "pps"
    if x <= l_threshold:
        return LABELS['DOWN']
    elif l_threshold < x < h_threshold:
        return LABELS['HOLD']
    else:
        return LABELS['UP']


def buy_only_label_(x, threshold):
    if np.isnan(x):
        return "nan"
    if x >= threshold:
        return LABELS['UP']
    else:
        return LABELS['HOLD']


def sell_only_label_(x, threshold):
    if np.isnan(x):
        return "nan"
    if x <= threshold:
        return LABELS['DOWN']
    else:
        return LABELS['HOLD']


def find_percentage_variation(data):
    diff = np.diff(data)
    diff = np.append(diff, np.nan)
    return 100 * diff / data


def find_relative_difference(variable, fixed):
    if len(fixed) != len(variable):
        raise ValueError("Two arrays must have the same length.")
    return 100 * (variable - fixed) / fixed


def filter_initial_nan(data: pd.DataFrame):
    """
    Creates a new dataframe starting from the first row without nan values.
    :param data: input dataframe with nan values in initial rows
    :return: filtered dataframe, index of the first valid row
    """

    logger = logging.getLogger(__name__)
    data = pd.DataFrame(data)
    data.fillna('NA', inplace=True)
    data.replace('nan', 'NA', inplace=True)
    
    for i in range(data.shape[0]):
        t_row = (data.iloc[i, :]).isin(["NA"])
        if not t_row.any():
            logger.debug('First valid record found at index {}/{}: {} left'.format(i, data.shape[0], data.shape[0] - i))
            return data.iloc[i:, :], i

    # if no valid record was found there likely is an error
    raise RuntimeError("Every row is NaN => empty series.")


def create_class_labels(data: np.array,
                        operation_allowed,
                        l_threshold,
                        h_threshold):
    var = find_percentage_variation(data)
    
    if operation_allowed == OperationAllowed.LS:
        make_class_label = np.vectorize(make_class_label_)
        varc = make_class_label(var, l_threshold, h_threshold)
    elif operation_allowed == OperationAllowed.L:
        make_class_label = np.vectorize(buy_only_label_)
        varc = make_class_label(var, h_threshold)
    elif operation_allowed == OperationAllowed.S:
        make_class_label = np.vectorize(sell_only_label_)
        varc = make_class_label(var, l_threshold)
    else:
        raise ValueError("Trend parameter passed is not valid")
    return var, varc


def drop_news_features(stock: Stock):
    cols_td = [c for c in stock.continuous_df_filtered.columns.values if 'NEWS' in c]
    stock.continuous_df_filtered.drop(cols_td, axis=1, inplace=True)
    stock.discrete_df_filtered.drop(cols_td, axis=1, inplace=True)
    return stock


def normalize_columns(input_df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalizes the columns of a pandas dataframe preserving its index and columns
    :param
    :return:
    """
    scaler = StandardScaler()
    normalized_df = pd.DataFrame(scaler.fit_transform(input_df.values), columns=input_df.columns, index=input_df.index)
    # normalized_df = (input_df - input_df.mean()) / input_df.std()
    return normalized_df


def discretize_technical_columns(df):
    """Perform two types of discretization strategies."""
    classic_disc = list()
    express_disc = list()

    sep = ":"

    classic_disc.append(pd.cut(df["SMA5-20"], [-np.inf, 0, np.inf], labels=[f"(-inf{sep}0]",f"(0{sep}inf)"]))
    express_disc.append(pd.cut(df["SMA5-20"], [-np.inf, -5, 0, 5, np.inf], labels=[f"(-inf{sep}-5]",f"(-5{sep}0]",f"(0{sep}5]",f"(5{sep}inf)"]))

    classic_disc.append(pd.cut(df["SMA8-15"], [-np.inf, 0, np.inf], labels=[f"(-inf{sep}0]",f"(0{sep}inf)"]))
    express_disc.append(pd.cut(df["SMA8-15"], [-np.inf, -5, 0, 5, np.inf], labels=[f"(-inf{sep}-5]",f"(-5{sep}0]",f"(0{sep}5]",f"(5{sep}inf)"]))
    
    classic_disc.append(pd.cut(df["SMA20-50"], [-np.inf, 0, np.inf], labels=[f"(-inf{sep}0]",f"(0{sep}inf)"]))
    express_disc.append(pd.cut(df["SMA20-50"], [-np.inf, -5, 0, 5, np.inf], labels=[f"(-inf{sep}-5]",f"(-5{sep}0]",f"(0{sep}5]",f"(5{sep}inf)"]))
    
    classic_disc.append(pd.cut(df["EMA5-20"], [-np.inf, 0, np.inf], labels=[f"(-inf{sep}0]",f"(0{sep}inf)"]))
    express_disc.append(pd.cut(df["EMA5-20"], [-np.inf, -5, 0, 5, np.inf], labels=[f"(-inf{sep}-5]",f"(-5{sep}0]",f"(0{sep}5]",f"(5{sep}inf)"]))
    
    classic_disc.append(pd.cut(df["EMA8-15"], [-np.inf, 0, np.inf], labels=[f"(-inf{sep}0]",f"(0{sep}inf)"]))
    express_disc.append(pd.cut(df["EMA8-15"], [-np.inf, -5, 0, 5, np.inf], labels=[f"(-inf{sep}-5]",f"(-5{sep}0]",f"(0{sep}5]",f"(5{sep}inf)"]))
    
    classic_disc.append(pd.cut(df["EMA20-50"], [-np.inf, 0, np.inf], labels=[f"(-inf{sep}0]",f"(0{sep}inf)"]))
    express_disc.append(pd.cut(df["EMA20-50"], [-np.inf, -5, 0, 5, np.inf], labels=[f"(-inf{sep}-5]",f"(-5{sep}0]",f"(0{sep}5]",f"(5{sep}inf)"]))
    
    classic_disc.append(pd.cut(df["MACD12-26"], [-np.inf, 0, np.inf], labels=[f"(-inf{sep}0]",f"(0{sep}inf)"]))
    express_disc.append(pd.cut(df["MACD12-26"], [-np.inf, -5, -2, 0, 2, 5, np.inf], labels=[f"(-inf{sep}-5]",f"(-5{sep}-2]",f"(-2{sep}0]",f"(0{sep}2]",f"(2{sep}5]",f"(5{sep}inf)"]))
    
    classic_disc.append(pd.cut(df["AO14"], [-100, 0, 100], labels=[f"[-100{sep}0]",f"(0{sep}100]"], include_lowest=True))
    express_disc.append(pd.cut(df["AO14"], [-100, -50, 0, 50, 100], labels=[f"[-100{sep}-50]",f"(-50{sep}0]",f"(0{sep}50]",f"(50{sep}100]"], include_lowest=True))
    
    classic_disc.append(pd.cut(df["ADX14"], [-np.inf, 20, 25, np.inf], labels=[f"(-inf{sep}20]",f"(20{sep}25]",f"(25{sep}inf)"]))
    express_disc.append(pd.cut(df["ADX14"], [-np.inf, 20, 25, 40, np.inf], labels=[f"(-inf{sep}20]",f"(20{sep}25]",f"(25{sep}40]",f"(40{sep}inf)"]))
    
    classic_disc.append(pd.cut(df["WD14"], [-np.inf, 0, np.inf], labels=[f"(-inf{sep}0]",f"(0{sep}inf)"]))
    express_disc.append(pd.cut(df["WD14"], [-np.inf, -5, 0, 5, np.inf], labels=[f"(-inf{sep}-5]",f"(-5{sep}0]",f"(0{sep}5]",f"(5{sep}inf)"]))
    
    classic_disc.append(pd.cut(df["PPO12_26"], [-np.inf, 0, np.inf], labels=[f"(-inf{sep}0]",f"(0{sep}inf)"]))
    express_disc.append(pd.cut(df["PPO12_26"], [-np.inf, -5, 0, 5, np.inf], labels=[f"(-inf{sep}-5]",f"(-5{sep}0]",f"(0{sep}5]",f"(5{sep}inf)"]))
    
    classic_disc.append(pd.cut(df["RSI14"], [0, 30, 70, 100], labels=[f"[0{sep}30]",f"(30{sep}70]",f"(70{sep}100]"], include_lowest=True))
    express_disc.append(pd.cut(df["RSI14"], [0, 15, 30, 50, 70, 85, 100], labels=[f"[0{sep}15]",f"(15{sep}30]",f"(30{sep}50]",f"(50{sep}70]",f"(70{sep}85]",f"(85{sep}100]"], include_lowest=True))
    
    classic_disc.append(pd.cut(df["MFI14"], [0, 30, 70, 100], labels=[f"[0{sep}30]",f"(30{sep}70]",f"(70{sep}100]"], include_lowest=True))
    express_disc.append(pd.cut(df["MFI14"], [0, 15, 30, 50, 70, 85, 100], labels=[f"[0{sep}15]",f"(15{sep}30]",f"(30{sep}50]",f"(50{sep}70]",f"(70{sep}85]",f"(85{sep}100]"], include_lowest=True))
    
    classic_disc.append(pd.cut(df["TSI"], [-np.inf, -25, 25, np.inf], labels=[f"(-inf{sep}-25]",f"(-25{sep}25]",f"(25{sep}inf)"]))
    express_disc.append(pd.cut(df["TSI"], [-np.inf, -25, 0, 25, np.inf], labels=[f"(-inf{sep}-25]",f"(-25{sep}0]",f"(0{sep}25]",f"(25{sep}inf)"]))
    
    classic_disc.append(pd.cut(df["SO14"], [0, 20, 80, 100], labels=[f"[0{sep}20]",f"(20{sep}80]",f"(80{sep}100]"], include_lowest=True))
    express_disc.append(pd.cut(df["SO14"], [0, 10, 20, 50, 80, 90, 100], labels=[f"[0{sep}10]",f"(10{sep}20]",f"(20{sep}50]",f"(50{sep}80]",f"(80{sep}90]",f"(90{sep}100]"], include_lowest=True))

    classic_disc.append(pd.cut(df["CMO14"], [-100, -50, 50, 100], labels=[f"[-100{sep}-50]",f"(-50{sep}50]",f"(50{sep}100]"], include_lowest=True))
    express_disc.append(pd.cut(df["CMO14"], [-100, -75, -50, 0, 50, 75, 100], labels=[f"[-100{sep}-75]",f"(-75{sep}-50]",f"(-50{sep}0]",f"(0{sep}50]",f"(50{sep}75]",f"(75{sep}100]"], include_lowest=True))
    
    classic_disc.append(pd.cut(df["ATRP14"], [0, 30, 100], labels=[f"[0{sep}30]",f"(30{sep}100]"], include_lowest=True))
    express_disc.append(pd.cut(df["ATRP14"], [0, 10, 30, 40, 100], labels=[f"[0{sep}10]",f"(10{sep}30]",f"(30{sep}40]",f"(40{sep}100]"], include_lowest=True))

    classic_disc.append(pd.cut(df["PVO14"], [-100, 0, 100], labels=[f"[-100{sep}0]",f"(0{sep}100]"], include_lowest=True))
    express_disc.append(pd.cut(df["PVO14"], [-100, -40, -20, 0, 20, 40, 100], labels=[f"[-100{sep}-40]",f"(-40{sep}-20]",f"(-20{sep}0]",f"(0{sep}20]",f"(20{sep}40]",f"(40{sep}100]"], include_lowest=True))

    classic_disc.append(pd.cut(df["ADL"], [-np.inf, 0, np.inf], labels=[f"(-inf{sep}0]",f"(0{sep}inf)"]))
    express_disc.append(pd.cut(df["ADL"], [-np.inf, -1e9, 0, 1e9, np.inf], labels=[f"(-inf{sep}-1e9]",f"(-1e9{sep}0]",f"(0{sep}1e9]",f"(1e9{sep}inf)"]))
    
    classic_disc.append(pd.cut(df["OBV"], [-np.inf, 0, np.inf], labels=[f"(-inf{sep}0]",f"(0{sep}inf)"]))
    express_disc.append(pd.cut(df["OBV"], [-np.inf, -1e9, 0, 1e9, np.inf], labels=[f"(-inf{sep}-1e9]",f"(-1e9{sep}0]",f"(0{sep}1e9]",f"(1e9{sep}inf)"]))
    
    classic_disc.append(pd.cut(df["FI13"], [-np.inf, 0, np.inf], labels=[f"(-inf{sep}0]",f"(0{sep}inf)"]))
    express_disc.append(pd.cut(df["FI13"], [-np.inf, -1e7, 0, 1e7, np.inf], labels=[f"(-inf{sep}-1e7]",f"(-1e7{sep}0]",f"(0{sep}1e7]",f"(1e7{sep}inf)"]))
    
    classic_disc.append(pd.cut(df["FI50"], [-np.inf, 0, np.inf], labels=[f"(-inf{sep}0]",f"(0{sep}inf)"]))
    express_disc.append(pd.cut(df["FI50"], [-np.inf, -1e7, 0, 1e7, np.inf], labels=[f"(-inf{sep}-1e7]",f"(-1e7{sep}0]",f"(0{sep}1e7]",f"(1e7{sep}inf)"]))

    return pd.concat(classic_disc, axis=1), pd.concat(express_disc, axis=1)


class FeaturesHandler:
    """
    Class that evaluates the features and class labels of our dataset
    """

    def __init__(self, operation_allowed: OperationAllowed):
        self.operation_allowed = operation_allowed
        self.dictionary = RuleDictionary()
        self.fill_dict = True

    def compute_class_label(self, 
                            stock: Stock,
                            name: str,
                            l_threshold,
                            h_threshold):
        """
        Create class labels using the percentage variation with respect to the closing price of the next day.
        :param stock: stock to use
        :return: (pd.Series, pd.Series) -> (continuous, discrete) class labels
        """
        cc, dc = create_class_labels(stock.close, self.operation_allowed, l_threshold, h_threshold)
        cs = pd.Series(cc, index=stock.close.index, name=name)
        ds = pd.Series(dc, index=stock.close.index, name=name)
        return cs, ds

    def compute_technical_features(self, stock: Stock):
        logger = logging.getLogger(__name__)
        logger.debug('stock {}: computing technical features'.format(stock.name))

        # TODO: another NaN filter was present here (added by PETROCCHI), add it back again?

        close = stock.close.values
        high = stock.high.values
        low = stock.low.values
        volume = stock.volume.values
        new_data = pd.DataFrame()
        categorical = pd.DataFrame()

        # Functions used for discretization of continuous values
        make_discrete_single = np.vectorize(make_discrete_single_)
        make_discrete_double = np.vectorize(make_discrete_double_)

        # --- TREND INDICATORS ---

        sma5 = simple_moving_average(close, 5)
        sma8 = simple_moving_average(close, 8)
        sma15 = simple_moving_average(close, 15)
        sma20 = simple_moving_average(close, 20)
        sma50 = simple_moving_average(close, 50)
        ema5 = exponential_moving_average(close, 5)
        ema8 = exponential_moving_average(close, 8)
        ema15 = exponential_moving_average(close, 15)
        ema20 = exponential_moving_average(close, 20)
        ema50 = exponential_moving_average(close, 50)

        sma5_20 = find_relative_difference(sma5, sma20)
        sma5_20c = make_discrete_single(sma5_20, 0)
        new_data['SMA5-20'] = sma5_20
        categorical['SMA5-20'] = sma5_20c

        sma8_15 = find_relative_difference(sma8, sma15)
        sma8_15c = make_discrete_single(sma8_15, 0)
        new_data['SMA8-15'] = sma8_15
        categorical['SMA8-15'] = sma8_15c

        sma20_50 = find_relative_difference(sma20, sma50)
        sma20_50c = make_discrete_single(sma20_50, 0)
        new_data['SMA20-50'] = sma20_50
        categorical['SMA20-50'] = sma20_50c

        ema5_20 = find_relative_difference(ema5, ema20)
        ema5_20c = make_discrete_single(ema5_20, 0)
        new_data['EMA5-20'] = ema5_20
        categorical['EMA5-20'] = ema5_20c

        ema8_15 = find_relative_difference(ema8, ema15)
        ema8_15c = make_discrete_single(ema8_15, 0)
        new_data['EMA8-15'] = ema8_15
        categorical['EMA8-15'] = ema8_15c

        ema20_50 = find_relative_difference(ema20, ema50)
        ema20_50c = make_discrete_single(ema20_50, 0)
        new_data['EMA20-50'] = ema20_50
        categorical['EMA20-50'] = ema20_50c

        # MACD relative
        ema12 = exponential_moving_average(close, 12)
        ema26 = exponential_moving_average(close, 26)
        macd12_26 = find_relative_difference(ema12, ema26)
        # macd12_26 = moving_average_convergence_divergence(close, 12, 26)
        macd12_26c = make_discrete_single(macd12_26, 0)
        new_data['MACD12-26'] = macd12_26
        categorical['MACD12-26'] = macd12_26c

        ao14 = aroon_oscillator(close, 14)
        ao14c = make_discrete_single(ao14, 0)
        new_data['AO14'] = ao14
        categorical['AO14'] = ao14c
        # TODO: does create problems with some cryptocurrencies?

        adx14 = average_directional_index(close, high, low, 14)
        adx14c = make_discrete_single(adx14, 20)
        new_data['ADX14'] = adx14
        categorical['ADX14'] = adx14c

        di_pos = positive_directional_index(close, high, low, 14)
        di_neg = negative_directional_index(close, high, low, 14)
        wd14 = di_pos - di_neg
        wd14c = make_discrete_single(wd14, 0)
        new_data['WD14'] = wd14
        categorical['WD14'] = wd14c

        # --- MOMENTUM OSCILLATORS ---

        ppo12_26 = price_oscillator(close, 12, 26)
        ppo12_26c = make_discrete_single(ppo12_26, 0)
        new_data['PPO12_26'] = ppo12_26
        categorical['PPO12_26'] = ppo12_26c

        rsi14 = relative_strength_index(close, 14)
        rsi14c = make_discrete_double(rsi14, 30, 70)
        new_data['RSI14'] = rsi14
        categorical['RSI14'] = rsi14c

        mfi14 = money_flow_index(close, high, low, volume, 14)
        mfi14c = make_discrete_double(mfi14, 30, 70)
        new_data['MFI14'] = mfi14
        categorical['MFI14'] = mfi14c

        tsi = true_strength_index(close)
        tsic = make_discrete_double(tsi, -25, 25)
        new_data['TSI'] = tsi
        categorical['TSI'] = tsic

        so = percent_k(close, 14)
        soc = make_discrete_double(so, 20, 80)
        new_data['SO14'] = so
        categorical['SO14'] = soc

        # TODO inserting Price Momentum Oscillator?

        # --- VOLATILITY INDICATORS ---

        cmo = chande_momentum_oscillator(close, 14)
        cmoc = make_discrete_double(cmo, -50, 50)
        new_data['CMO14'] = cmo
        categorical['CMO14'] = cmoc

        atrp = average_true_range_percent(close, 14)
        atrpc = make_discrete_single(atrp, 30)
        new_data['ATRP14'] = atrp
        categorical['ATRP14'] = atrpc

        # --- VOLUME INDICATORS ---

        pvo = volume_oscillator(volume, 12, 26)
        pvoc = make_discrete_single(pvo, 0)
        new_data['PVO14'] = pvo
        categorical['PVO14'] = pvoc

        fi = force_index(close, volume)
        fi13 = exponential_moving_average(fi, 13)
        fi13c = make_discrete_single(fi13, 0)
        fi50 = exponential_moving_average(fi, 50)
        fi50c = make_discrete_single(fi50, 0)
        new_data['FI13'] = fi13
        categorical['FI13'] = fi13c
        new_data['FI50'] = fi50
        categorical['FI50'] = fi50c

        adl = accumulation_distribution(close, high, low, volume)
        adlc = make_discrete_single(adl, 0)
        new_data['ADL'] = adl
        categorical['ADL'] = adlc

        obv = on_balance_volume(close, volume)
        obvc = make_discrete_single(obv, 0)
        new_data['OBV'] = obv
        categorical['OBV'] = obvc

        categorical.set_index(stock.close.index, inplace=True)
        new_data.set_index(stock.close.index, inplace=True)
        return new_data, categorical

    def fill_dictionary(self):
        if self.fill_dict:
            self.dictionary.add_attribute('SMA5-20', 0)
            self.dictionary.add_attribute('SMA8-15', 0)
            self.dictionary.add_attribute('SMA20-50', 0)
            self.dictionary.add_attribute('EMA5-20', 0)
            self.dictionary.add_attribute('EMA8-15', 0)
            self.dictionary.add_attribute('EMA20-50', 0)
            self.dictionary.add_attribute('MACD12-26', 0)
            self.dictionary.add_attribute('AO14', 0)
            self.dictionary.add_attribute('ADX14', 20)
            self.dictionary.add_attribute('WD14', 0)
            self.dictionary.add_attribute('PPO12_26', 0)
            self.dictionary.add_attribute('RSI14', 30, 70)
            self.dictionary.add_attribute('MFI14', 30, 70)
            self.dictionary.add_attribute('TSI', -25, 25)
            self.dictionary.add_attribute('SO14', 20, 80)
            self.dictionary.add_attribute('CMO14', -50, 50)
            self.dictionary.add_attribute('ATRP14', 30)
            self.dictionary.add_attribute('PVO14', 0)
            self.dictionary.add_attribute('FI13', 0)
            self.dictionary.add_attribute('FI50', 0)
            self.dictionary.add_attribute('ADL', 0)
            self.dictionary.add_attribute('OBV', 0)
            for i in range(W):
                self.dictionary.add_attribute('CLOSE_VAR_{}'.format(i + 1), CLASS_TH[0], CLASS_TH[1])
            self.fill_dict = False

    def compute_temporal_features(self, stock: Stock) -> (pd.DataFrame, pd.DataFrame):
        """
        Compute W (see config.py) new features as the Rate Of Change in a window of W past time steps
        :param stock: stock to evaluate
        :return: tuple (continuous features, discrete features)
        """
        # TODO this also needs a translation of attributes names to plain english.
        #  Ignore when dictionary will be handled by L3 wrapper instead.
        #  Move to static method.

        close = stock.close.values
        # this discretization leads to string attributes, use positive integer instead
        # make_class_label = np.vectorize(make_class_label_)  # use the same discretization of class labels
        make_discrete_double = np.vectorize(make_discrete_double_)

        cont_windows = W * [np.full(W, np.nan)]
        disc_windows = W * [np.full(W, np.nan)]
        for i in range(W, len(close)):
            window = close[i - W:i + 1]                     # window of length W + 1
            cw = find_percentage_variation(window)[:-1]     # variations array of length W (drop final nan)
            # cw /= 100
            dw = make_discrete_double(cw, CLASS_TH[0], CLASS_TH[1])     # discretization
            cont_windows += [cw]
            disc_windows += [dw]

        cont_temporal = pd.DataFrame(cont_windows)
        disc_temporal = pd.DataFrame(disc_windows)
        disc_temporal = disc_temporal.fillna(0.0).astype('int32')   # because initial W rows are filled with nan
        cont_temporal.columns = ['CLOSE_VAR_{}'.format(int(x) + 1) for x in cont_temporal.columns]
        disc_temporal.columns = ['CLOSE_VAR_{}'.format(int(x) + 1) for x in disc_temporal.columns]
        cont_temporal.set_index(stock.close.index, inplace=True)
        disc_temporal.set_index(stock.close.index, inplace=True)

        return cont_temporal, disc_temporal
