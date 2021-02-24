import os
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
import time
import json
from atsa.config import *
from atsa import preprocessing
import statsmodels.api as sm
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.api import VAR
from sklearn.naive_bayes import GaussianNB
import sklearn.neural_network as neural_network
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.base import clone
import l3wrapper.classification as wrapper
from l3wrapper.dictionary import *
from atsa.enum import Classifier, Validation, OperationAllowed


def clean_l3_output(out_dir, stem):
    os.remove(os.path.join(out_dir, stem + ".data"))
    os.remove(os.path.join(out_dir, stem + ".bin"))
    os.remove(os.path.join(out_dir, stem + ".cls"))
    os.remove(os.path.join(out_dir, stem + ".diz"))
    os.remove(os.path.join(out_dir, "classificati.txt"))
    os.remove(os.path.join(out_dir, "confMatrix.txt"))
    os.remove(os.path.join(out_dir, "totali.txt"))
    os.remove(os.path.join(out_dir, "livelloI.txt"))
    os.remove(os.path.join(out_dir, "livelloII.txt"))


class BaseClassifier(ABC):
    def __init__(self, name: str, parameters: str, operation_allowed: OperationAllowed):
        self.name = name
        self.parameters = parameters
        self.operation_allowed = operation_allowed
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

    @abstractmethod
    def train_and_predict(self, train, test, columns: list = None) -> list:
        pass

    def __repr__(self):
        return '{}-{}-{}'.format(self.name, self.parameters, self.operation_allowed)


class BaseMLClassifier(BaseClassifier, ABC):
    def __init__(self, name: str, parameters: str, operation_allowed: OperationAllowed):
        super().__init__(name, parameters, operation_allowed)

    @abstractmethod
    def train(self, train, columns: list) -> object:
        pass

    @abstractmethod
    def predict(self, test, model, columns: list) -> list:
        pass


class BaseHandler(ABC):
    def __init__(self, data, classifier: BaseClassifier):
        self.data = data
        self.classifier = classifier
        self.logger = logging.getLogger(__name__)

    def __repr__(self):
        return '{}, {}, {}'.format(self.classifier.name, self.classifier.parameters, self.classifier.operation_allowed)

    @abstractmethod
    def hold_out(self):
        pass

    @abstractmethod
    def expanding_window(self):
        pass


class TimeSeriesHandler(BaseHandler):

    def hold_out(self) -> pd.Series:
        train_len = int(self.data.shape[0] * FW_TRAIN)
        test_len = self.data.shape[0] - train_len
        self.logger.debug('{}: train len={}, test len={}'.format(
            self.classifier, train_len, test_len))

        if type(self.data) == pd.Series:
            train = self.data.iloc[:train_len]
            test = self.data.iloc[train_len:]
        elif type(self.data) == np.ndarray:
            train = self.data[:train_len]
            test = self.data[train_len:]
        else:
            raise RuntimeError("Unknown data type.")

        labels = (train_len - 1) * [LABELS["TRAIN"]]
        classification = self.classifier.train_and_predict(train, test)
        return pd.Series(labels + list(classification))

    def expanding_window(self):
        train_len = int(self.data.shape[0] * EW_INIT_TRAIN)

        classification = []
        exec_times = []
        for idx in range(train_len, self.data.shape[0]):
            start_time = time.time()
            if type(self.data) == pd.Series:
                train = self.data.iloc[:idx]
                test = self.data.iloc[idx:idx + 1]
            elif type(self.data) == np.ndarray:
                train = self.data[:idx]
                test = self.data[idx:idx + 1]
            else:
                raise RuntimeError("Unknown data type.")

            labels = self.classifier.train_and_predict(train, test)
            classification += [labels[0]]

            # timing informations
            elapsed_time = time.time() - start_time
            self.logger.debug(
                'EW classification (idx: {}) took {} s'.format(idx, int(elapsed_time)))
            exec_times += [elapsed_time]
        classification += [np.nan]

        exec_times = pd.Series(exec_times)
        self.logger.info('EW classification times: min {}, max {}, avg {}'.format(exec_times.min(), exec_times.max(),
                                                                                  exec_times.mean()))

        labels = (train_len - 1) * [LABELS["TRAIN"]]
        return pd.Series(labels + classification)


def values_from_vars(initial_price: float, variations: np.array) -> list:
    values = [initial_price + variations[0]]
    for i in range(1, len(variations)):
        values += [values[i - 1] + variations[i]]
    return values


class MultivariateTimeSeriesHandler(BaseHandler):

    def __init__(self, data, classifier, univariate_classifier, index_info, out_dir):
        super().__init__(data, classifier)
        self.univariate_classifier = univariate_classifier
        self.index_info = index_info
        self.out_dir = out_dir

    def hold_out(self):
        validation = Validation.HOLD_MULTI
        dataset_size = (self.data[0]).close.shape[0]  # used as reference
        train_len = int(dataset_size * FW_TRAIN)
        test_len = dataset_size - train_len
        self.logger.debug('{}: train len={}, test len={}'.format(
            self.classifier, train_len, test_len))

        _signals = self.classify_on_diff(train_len, test_len)

        signals_dict = {}
        for stock in self.data:
            signals_dict[stock.close.name] = stock.close
            signals_dict[stock.high.name] = stock.high
            signals_dict[stock.low.name] = stock.low
            s = _signals[stock.name]
            s.name = stock.name + "_PRED"
            s.index = stock.close.index
            signals_dict[s.name] = s

        result = IndexClassificationResult(
            validation, self.classifier, pd.DataFrame(signals_dict))
        result.save(self.out_dir)

    def expanding_window(self):
        validation = Validation.EXP_MULTI
        dataset_size = (self.data[0]).close.shape[0]  # used as reference
        train_len = int(dataset_size * EW_INIT_TRAIN)
        test_len = dataset_size - train_len
        self.logger.debug('{}: train len={}, test len={}'.format(
            self.classifier, train_len, test_len))

        # For each loop round: train the model gathering records from data of the same
        # category, then classify one step for each stock.
        # At the end each session will have the complete list of classification labels.
        classification_dict = {}
        for stock in self.data:
            classification_dict[stock.name] = [LABELS["TRAIN"]] * train_len

        idx = train_len
        exec_times = []
        try:
            for idx in range(train_len, dataset_size - 1):
                start_time = time.time()
                self.logger.info('{}, {} at idx: {}'.format(
                    validation, self.classifier, idx))

                _signals = self.classify_on_diff(idx, 1)

                for k, v in classification_dict.items():
                    # TODO: this is not true if the classifier doesn't add last NaN
                    s = _signals[k].iloc[-2]
                    v.append(s)

                # timing informations
                elapsed_time = time.time() - start_time
                self.logger.debug(
                    'EW classification (idx: {}) took {} s'.format(idx, int(elapsed_time)))
                exec_times += [elapsed_time]

            exec_times = pd.Series(exec_times)
            self.logger.info(
                'EW classification times: min {}, max {}, avg {}'.format(exec_times.min(), exec_times.max(),
                                                                         exec_times.mean()))
        except ValueError as ve:
            logging.error('Value error in {}, {} at idx: {}'.format(
                validation, self.classifier, idx))
            raise ValueError() from ve

        signals_dict = {}
        for stock in self.data:
            classification_dict[stock.name] += [np.nan]
            signals = pd.Series(classification_dict[stock.name])

            signals.name = stock.name + "_PRED"
            signals.index = stock.close.index
            signals_dict[stock.close.name] = stock.close
            signals_dict[stock.high.name] = stock.high
            signals_dict[stock.low.name] = stock.low
            signals_dict[signals.name] = signals

        result = IndexClassificationResult(
            validation, self.classifier, pd.DataFrame(signals_dict))
        result.save(self.out_dir)

    def classify_on_diff(self, train_len, test_len) -> dict:
        signals_dict = {}

        # Build training set with slices of each related stock
        for sector, stock_names in self.index_info.SECTORS.items():

            # TODO: a lot of operations duplicated here (with Expanding Window validation)
            current_stocks = [s for s in self.data if s.sector == sector]

            if len(current_stocks) == 0:
                self.logger.info(
                    'No stocks found for sector {}'.format(sector))
                continue

            # UPDATE : manage case of only one stock for current sector -> unlabel stock and switch to univariate model
            if len(current_stocks) == 1:
                self.logger.info(
                    'There is only one stock found for sector {}, it will be classified using univariate model'.format(sector))
                for stock in [s for s in self.data if s.sector == sector]:
                    stock.sector = self.index_info.UNKNOWN_SECTOR
                continue

            close_prices = [s.close.diff() for s in current_stocks]

            df = pd.concat(close_prices, axis=1)
            # do not consider first row made of NaNs
            train = df.iloc[1:train_len, :]
            test = df.iloc[train_len:train_len + test_len, :]
            df_forecast = self.classifier.forecast(train, test)

            for stock in current_stocks:
                values_forecast = values_from_vars(
                    stock.close[train_len - 1], df_forecast[stock.name])
                forecast = [stock.close[train_len - 1]] + values_forecast
                _, labels = preprocessing.create_class_labels(
                    forecast, self.classifier.operation_allowed)

                train_labels = (train_len - 1) * [LABELS["TRAIN"]]
                signals = pd.concat([pd.Series(train_labels + list(labels))])
                signals_dict[stock.name] = signals

        # Then process every stock not belonging to any category:
        for stock in [s for s in self.data if s.sector == self.index_info.UNKNOWN_SECTOR]:
            logging.debug(
                '{} has no sector, classify using univariate model'.format(stock.name))
            train = stock.close.iloc[:train_len]
            test = stock.close.iloc[train_len:]

            labels = (train_len - 1) * [LABELS["TRAIN"]]
            classification = self.univariate_classifier.train_and_predict(
                train, test)
            signals = pd.Series(labels + list(classification))
            signals_dict[stock.name] = signals

        return signals_dict


class UnivariateHandler(BaseHandler):

    def hold_out(self) -> pd.Series:
        self.logger.debug('Shape: {}\n Feat: {}'.format(
            self.data.shape, self.data.columns))
        train_len = int(self.data.shape[0] * FW_TRAIN)
        test_len = self.data.shape[0] - train_len
        train = self.data.iloc[:train_len, :]
        train.name = self.data.name
        test = self.data.iloc[train_len:-1, :]
        test.name = self.data.name

        classification = self.classifier.train_and_predict(
            train, test, list(train.columns.values))
        labels = train_len * [LABELS["TRAIN"]]
        return pd.Series(labels + classification + [np.nan])

    def expanding_window(self):
        train_len = int(self.data.shape[0] * EW_INIT_TRAIN)

        main_st = time.time()
        classification = []
        exec_times = []
        for idx in range(train_len, self.data.shape[0] - 1):
            start_time = time.time()
            train = self.data.iloc[:idx, :]
            train.name = self.data.name
            test = self.data.iloc[idx:idx + 1, :]
            test.name = self.data.name
            labels = self.classifier.train_and_predict(
                train, test, list(train.columns.values))
            classification += [labels[0]]

            # timing informations
            elapsed_time = time.time() - start_time
            self.logger.debug(
                'EW classification (idx: {}) took {} s'.format(idx, int(elapsed_time)))
            exec_times += [elapsed_time]

        exec_times = pd.Series(exec_times)
        self.logger.info('EW classification times: m {:2f}, M {:2f}, A {:2f}, T {:2f}, D: {}'
                         .format(exec_times.min(),
                                 exec_times.max(),
                                 exec_times.mean(),
                                 time.time() - main_st,
                                 len(range(train_len, self.data.shape[0] - 1))))
        labels = train_len * [LABELS["TRAIN"]]
        return pd.Series(labels + classification + [np.nan])


class MultivariateHandler(BaseHandler):

    # TODO: handle self.COLUMNS missing here

    def __init__(self, data, classifier, index_info, out_dir):
        super().__init__(data, classifier)
        self.index_info = index_info
        self.out_dir = out_dir

    # TODO: optimize stock discretization by doing it once in advance
    def train_sector_models(self, train_len: int):
        sector_models = {}

        if type(self.classifier) == L3Classifier:
            initial_dir = os.getcwd()

            for sector, stock_names in self.index_info.SECTORS.items():
                print("sector: "+str(sector))

                # Build training set with slices of each related stock
                current_stocks = [s for s in self.data if s.sector == sector]
                train = pd.DataFrame(columns=self.COLUMNS)
                for stock in current_stocks:
                    df = stock.discrete_df_filtered.iloc[:train_len, :]
                    df.name = stock.discrete_df_filtered.name
                    train = train.append(df, sort=False)

                # Save each model in separate directory to avoid conflicts with files
                model_name = 'sector-{}'.format(sector)
                if not os.path.isdir(model_name):
                    os.mkdir(model_name)

                os.chdir(os.path.join(initial_dir, model_name))
                sector_models[sector] = os.path.join(initial_dir, model_name)

                train.name = model_name
                self.classifier.train(train, self.COLUMNS)
                os.chdir(initial_dir)

        elif type(self.classifier) in [MLPClassifier, MNBClassifier, RFClassifier, SVClassifier]:

            for sector, stock_names in self.index_info.SECTORS.items():

                # Build training set with slices of each related stock
                current_stocks = [s for s in self.data if s.sector == sector]

                # UPDATE: handling sectors with 0 occurences in dataset
                if not current_stocks:
                    raise RuntimeError(
                        'No stocks found for sector {}'.format(sector))

                train = pd.DataFrame(columns=self.COLUMNS)

                for stock in current_stocks:
                    df = stock.discrete_df_filtered.iloc[:train_len, :]
                    df.name = stock.discrete_df_filtered.name
                    train = train.append(df, sort=False)

                train.name = sector
                sector_models[sector] = self.classifier.train(
                    train, self.COLUMNS)

        else:
            raise RuntimeError("Unknown classifier type")

        return sector_models

    def hold_out(self):
        validation = Validation.HOLD_MULTI
        # used as reference
        dataset_size = (self.data[0]).discrete_df_filtered.shape[0]
        train_len = int(dataset_size * FW_TRAIN)
        test_len = dataset_size - train_len

        sector_models = self.train_sector_models(train_len)
        current_asset = None
        try:
            signals_dict = {}
            for asset in self.data:
                self.logger.info('{}, {}, {}'.format(
                    validation, self.classifier, asset.name))
                current_asset = asset

                train_l = int(asset.discrete_df_filtered.shape[0] * FW_TRAIN)
                test_l = asset.discrete_df_filtered.shape[0] - train_len
                # TODO check if they are the same of train len and test len of the first series

                self.logger.debug('{}: train len={}, test len={}'.format(
                    self.classifier, train_len, test_len))

                test = asset.discrete_df_filtered.iloc[train_len:-1, :]
                test.name = asset.discrete_df_filtered.name

                if asset.sector != self.index_info.UNKNOWN_SECTOR:
                    classification = self.classifier.predict(
                        test, sector_models[asset.sector], self.COLUMNS)
                else:
                    train = asset.discrete_df_filtered.iloc[:train_len, :]
                    train.name = asset.discrete_df_filtered.name
                    classification = self.classifier.train_and_predict(
                        train, test, self.COLUMNS)

                train_labels = train_len * [LABELS["TRAIN"]]
                signals = pd.concat([pd.Series([LABELS["DISCARDED"]] * asset.first_valid_row),
                                     pd.Series(train_labels + classification + [np.nan])])

                signals.name = asset.name + "_PRED"
                signals.index = asset.close.index
                signals_dict[asset.close.name] = asset.close
                signals_dict[asset.high.name] = asset.high
                signals_dict[asset.low.name] = asset.low
                signals_dict[signals.name] = signals

            result = IndexClassificationResult(
                validation, self.classifier, pd.DataFrame(signals_dict))
            result.save(self.out_dir)
        except ValueError as ve:
            logging.error('Error in {}, {}, {}'.format(
                validation, self.classifier, current_asset.name))
            raise ValueError() from ve

    def expanding_window(self):
        validation = Validation.EXP_MULTI
        # used as reference
        dataset_size = (self.data[0]).discrete_df_filtered.shape[0]
        train_len = int(dataset_size * EW_INIT_TRAIN)

        # For each loop round: train the model gathering records from data of the same
        # category, then classify one step for each stock.
        # At the end each session will have the complete list of classification labels.
        classification_dict = {}
        for stock in self.data:
            classification_dict[stock.name] = []
            classification_dict[stock.name] += [LABELS["DISCARDED"]] * stock.first_valid_row + \
                                               [LABELS["TRAIN"]] * train_len

        idx = train_len
        exec_times = []
        try:
            for idx in range(train_len, dataset_size - 1):
                start_time = time.time()
                self.logger.info('{}, {} at idx: {}'.format(
                    validation, self.classifier, idx))
                sector_models = self.train_sector_models(idx)
                for stock in self.data:
                    test = stock.discrete_df_filtered.iloc[idx:idx + 1, :]
                    test.name = stock.discrete_df_filtered.name

                    if stock.sector != self.index_info.UNKNOWN_SECTOR:
                        classification = self.classifier.predict(
                            test, sector_models[stock.sector], self.COLUMNS)
                    else:
                        train = stock.discrete_df_filtered.iloc[:idx, :]
                        train.name = stock.discrete_df_filtered.name
                        classification = self.classifier.train_and_predict(
                            train, test, self.COLUMNS)

                    classification_dict[stock.name].append(classification[0])

                # timing informations
                elapsed_time = time.time() - start_time
                self.logger.debug(
                    'EW classification (idx: {}) took {} s'.format(idx, int(elapsed_time)))
                exec_times += [elapsed_time]

            exec_times = pd.Series(exec_times)
            self.logger.info(
                'EW classification times: min {}, max {}, avg {}'.format(exec_times.min(), exec_times.max(),
                                                                         exec_times.mean()))

        except ValueError as ve:
            logging.error('Value error in {}, {} at idx: {}'.format(
                validation, self.classifier, idx))
            raise ValueError() from ve

        signals_dict = {}
        for stock in self.data:
            classification_dict[stock.name] += [np.nan]
            signals = pd.Series(classification_dict[stock.name])

            signals.name = stock.name + "_PRED"
            signals.index = stock.close.index
            signals_dict[stock.close.name] = stock.close
            signals_dict[stock.high.name] = stock.high
            signals_dict[stock.low.name] = stock.low
            signals_dict[signals.name] = signals

        result = IndexClassificationResult(
            validation, self.classifier, pd.DataFrame(signals_dict))
        result.save(self.out_dir)


class IndexClassificationResult:
    def __init__(self, validation: Validation = None,
                 classifier: BaseClassifier = None,
                 signals_df: pd.DataFrame = None):

        self.validation = validation
        self.classifier = classifier
        self.signals_df = signals_df

    def save(self, out_dir: str = None):
        filename = '{}_{}_{}_{}_signals.csv'.format(self.validation, self.classifier.name, self.classifier.parameters,
                                                    self.classifier.operation_allowed)
        if not out_dir:
            self.signals_df.to_csv(filename)
        else:
            self.signals_df.to_csv(os.path.join(out_dir, filename))

    def __repr__(self):
        return '({}, {}, {}, {})'.format(self.classifier.name, self.classifier.parameters,
                                         self.classifier.operation_allowed, self.validation)


# ---- ML CLASSIFIERS ----


def extract_np_array(df: pd.DataFrame, columns: list):
    if (not columns) or len(columns) == 0:
        raise ValueError(
            'Error extract_np_array {}: columns cannot be empty'.format(df.name))

    if 'class' in columns:
        columns.remove('class')

    labels = df['class'].values
    df_records = df[columns].apply(pd.to_numeric)
    records = df_records.values
    return records, labels


class BaseSKClassifier(BaseMLClassifier):

    def __init__(self, name: str, parameters: str, operation_allowed: OperationAllowed, classifier):
        self.classifier = classifier
        super().__init__(name, parameters, operation_allowed)

    def train_and_predict(self, train, test, columns=None) -> list:
        train_x, train_y = extract_np_array(train, columns)
        test_x, test_y = extract_np_array(test, columns)
        self.classifier.fit(train_x, train_y)
        labels = self.classifier.predict(test_x).tolist()
        return labels

    def train(self, train, columns):
        if not columns:
            raise ValueError('BaseSKClassifier error: columns is None')

        new_classifier = clone(self.classifier)
        train_x, train_y = extract_np_array(train, columns)
        new_classifier.fit(train_x, train_y)
        return new_classifier

    def predict(self, test, model, columns) -> list:
        if not columns:
            raise ValueError('BaseSKClassifier error: columns is None')

        test_x, test_y = extract_np_array(test, columns)
        labels = model.predict(test_x).tolist()
        return labels


class MLPClassifier(BaseSKClassifier):

    def __init__(self, operation_allowed: OperationAllowed,
                 hidden_layer_sizes,
                 learning_rate,
                 learning_rate_init,
                 activation,
                 solver: str = "adam"):

        parameters = '({})'.format(','.join(map(lambda x: str(x),
            [hidden_layer_sizes, learning_rate, learning_rate_init, activation, solver])))
        classifier = neural_network.MLPClassifier(hidden_layer_sizes=hidden_layer_sizes,
                                                  solver=solver,
                                                  random_state=6,
                                                  learning_rate=learning_rate,
                                                  learning_rate_init=learning_rate_init,
                                                  activation=activation)
        super().__init__(str(Classifier.MLP), parameters, operation_allowed, classifier)


class RFClassifier(BaseSKClassifier):

    def __init__(self, operation_allowed: OperationAllowed,
                 n_estimators: int,
                 criterion: str,
                 min_samples_split: int,
                 min_samples_leaf: int):
        parameters = '({})'.format(','.join(map(lambda x: str(x),
            [n_estimators, criterion, min_samples_split, min_samples_leaf])))
        classifier = RandomForestClassifier(n_estimators=n_estimators,
                                            criterion=criterion,
                                            min_samples_split=min_samples_split,
                                            min_samples_leaf=min_samples_leaf,
                                            random_state=6)
        super().__init__(str(Classifier.RFC), parameters, operation_allowed, classifier)


class GNBClassifier(BaseSKClassifier):

    def __init__(self, operation_allowed: OperationAllowed):
        parameters = '(priors=None)'
        classifier = GaussianNB()
        super().__init__(str(Classifier.GNB), parameters, operation_allowed, classifier)


class SVClassifier(BaseSKClassifier):

    def __init__(self, operation_allowed: OperationAllowed,
                 C: int,
                 kernel: str,
                 degree: int = None):
        parameters = '({})'.format(','.join(map(lambda x: str(x),[C, kernel, degree])))
        classifier = SVC(kernel=kernel, gamma='scale',
                         random_state=6, class_weight='balanced', degree=degree)
        super().__init__(str(Classifier.SVC), parameters, operation_allowed, classifier)


class KNNClassifier(BaseSKClassifier):

    def __init__(self, operation_allowed: OperationAllowed,
                 n_neighbors: int,
                 weights: str,
                 algorithm: str):
        parameters = '({})'.format(','.join(map(lambda x: str(x), [n_neighbors, weights, algorithm])))
        classifier = KNeighborsClassifier(n_neighbors=n_neighbors,
                                          n_jobs=-1,
                                          weights=weights,
                                          algorithm=algorithm)
        super().__init__(str(Classifier.KNN), parameters, operation_allowed, classifier)


# ---- TIME SERIES MODELS ----


class LinearRegressionClassifier(BaseClassifier):

    def __init__(self, operation_allowed: OperationAllowed):
        super().__init__(str(Classifier.LINREG), "(None)", operation_allowed)

    def train_and_predict(self, train, test, columns=None):
        train_values = train.values
        x = np.arange(train.size)
        xc = sm.add_constant(x)
        model = sm.OLS(train_values, xc)
        try:
            model_fit = model.fit(disp=0)
            x_to_forecast = np.arange(train.size, train.size + test.size)

            if x_to_forecast.shape[0] > 1:
                x_to_forecast = sm.add_constant(x_to_forecast)
            else:
                x_to_forecast = np.append(np.array(1.), x_to_forecast)
                x_to_forecast = x_to_forecast.reshape(1, 2)

            forecast = model_fit.predict(x_to_forecast)

            # include last training point in order to have $test_size labels and one $nan
            forecast = [train_values[-1]] + forecast.tolist()
            _, labels = preprocessing.create_class_labels(
                forecast, self.operation_allowed)

            # plt.plot(x, train_values, label="train true")
            # plt.plot(x, model_fit.fittedvalues, label="train fitted")
            # plt.plot(x_to_forecast, test.values, label="test true")
            # plt.plot(x_to_forecast, forecast[1:], label="forecast")
            # plt.legend(loc="best")
            # plt.show()

            return labels
        except (ValueError, np.linalg.linalg.LinAlgError):
            logging.error('Linear regression convergence error')
            return [LABELS["UNLABELED"]] * test.size + [np.nan]


class ARIMAClassifier(BaseClassifier):

    def __init__(self, order: list, operation_allowed: OperationAllowed):
        if len(order) != 3:
            raise ValueError("Order (p,d,q) passed is not valid")

        super().__init__(str(Classifier.ARIMA), str(order), operation_allowed)
        self.order = order

    def train_and_predict(self, train, test, columns=None):
        train_values = train.values
        model = ARIMA(train_values, order=self.order)
        try:
            model_fit = model.fit(disp=0)
            forecast, _, _ = model_fit.forecast(steps=test.size)
            forecast = [train_values[-1]] + forecast.tolist()
            _, labels = preprocessing.create_class_labels(
                forecast, self.operation_allowed)

            return labels
        except (ValueError, np.linalg.linalg.LinAlgError):
            self.logger.error('ARIMA convergence error (order {} {} {})'.format(self.order[0],
                                                                                self.order[1],
                                                                                self.order[2]))
            return [LABELS["UNLABELED"]] * test.size + [np.nan]


class ExponentialSmoothingClassifier(BaseClassifier):

    def __init__(self, operation_allowed: OperationAllowed, alpha: float, beta: float = None):
        parameters = '({},{})'.format(alpha, beta)
        super().__init__(str(Classifier.EXPSMOOTH), parameters, operation_allowed)
        self.alpha = alpha
        self.beta = beta

    def train_and_predict(self, train, test, columns=None):
        train_values = train.values
        try:
            if self.beta is not None:
                # Use Augmented Dickey-Fuller test: try to reject the null hypothesis that the series
                # has a unit root, i.e. it's stationary. Statsmodels implementation has issues with window
                # size lower than 6, so adapt it for now.
                # Using the p-value, check if it's possible to reject the null hypothesis at 5 % significance
                # level and than choose Simple Exponential Smoothing or Holt's Exponential Smoothing (additional
                # trend factor).
                r = adfuller(train_values) if train.size > 6 else adfuller(
                    train_values, maxlag=4)
                pvalue = r[1]
                if pvalue < 0.05:
                    model_fit = ExponentialSmoothing(train_values, trend=None, seasonal=None)\
                        .fit(smoothing_level=self.alpha)
                else:
                    model_fit = ExponentialSmoothing(train_values, trend='additive', seasonal='additive',
                                                     seasonal_periods=SEASONAL_PERIODS) \
                        .fit(smoothing_level=self.alpha, smoothing_slope=self.beta)
            else:
                model_fit = ExponentialSmoothing(train_values, trend=None, seasonal=None)\
                    .fit(smoothing_level=self.alpha)

            forecast = model_fit.forecast(steps=test.size)
            forecast = [train_values[-1]] + forecast.tolist()
            _, labels = preprocessing.create_class_labels(
                forecast, self.operation_allowed)
            return labels
        except (ValueError, np.linalg.linalg.LinAlgError):
            self.logger.error('Exponential Smoothing convergence error (a:{},b:{})'.format(
                self.alpha, self.beta))
            return [LABELS["UNLABELED"]] * test.size + [np.nan]


# 4l1ce : need multivariate
class VARClassifier(BaseClassifier):

    def __init__(self, operation_allowed: OperationAllowed, order: int):
        parameters = '({})'.format(order)
        super().__init__(str(Classifier.VAR), parameters, operation_allowed)
        self.order = order

    def train_and_predict(self, train, test, columns=None) -> list:
        raise NotImplementedError()

    def forecast(self, train, test):
        """
        Classifies test data by fitting a model on train data and forecasting future values. The latter are then
        converted to class labels as specified in finance.core module.
        """

        model = VAR(train)
        try:
            model_fit = model.fit(self.order)
            yhat = model_fit.forecast(model_fit.y, steps=test.shape[0])
            df_yhat = pd.DataFrame(data=yhat, index=pd.DatetimeIndex(
                test.index), columns=test.columns)
            # result = pd.DataFrame(train.iloc[-1, :]).transpose()
            # result = result.append(df_yhat)
            return df_yhat
        except (ValueError, np.linalg.linalg.LinAlgError):
            logging.error(
                'VAR convergence error with order {}'.format(self.order))
            return test.apply(lambda: LABELS["UNLABELED"])


class L3Classifier(BaseMLClassifier):

    def __init__(self,
                 operation_allowed: OperationAllowed,
                 minsup,
                 minconf,
                 l3_root: str,
                 rule_set: str = 'all',
                 top_count: int = None,
                 perc_count: int = None,
                 save_train_data_file: bool = False,
                 filter_level: str = None,
                 filtering_rules_file: str = None,
                 rule_dictionary: RuleDictionary = None):

        self.l3_classifier = wrapper.L3Classifier(minsup, minconf, l3_root)

        fourth = ''
        if top_count:
            fourth = top_count
        elif perc_count:
            fourth = perc_count

        parameters = '({},{},{},{})'.format(minsup, minconf, rule_set, fourth)
        super().__init__(str(Classifier.L3), parameters, operation_allowed)

        self.rule_set = rule_set
        self.top_count = top_count
        self.perc_count = perc_count
        self.save_train_data_file = save_train_data_file
        self.rule_dictionary = rule_dictionary
        self.filter_level = filter_level
        self.filtering_rules = None

        if filtering_rules_file:
            with open(filtering_rules_file, 'r') as fp:
                self.filtering_rules = json.load(fp)

    def train_and_predict(self, train, test, columns=None):
        labels = self.l3_classifier.train_and_predict(train, test,
                                                      columns=columns,
                                                      rule_set=self.rule_set,
                                                      top_count=self.top_count,
                                                      perc_count=self.perc_count,
                                                      filtering_rules=self.filtering_rules,
                                                      filter_level=self.filter_level,
                                                      rule_dictionary=self.rule_dictionary)
        return labels

    def train(self, train, columns):
        self.l3_classifier.train(train,
                                 columns=columns,
                                 rule_set=self.rule_set,
                                 top_count=self.top_count,
                                 perc_count=self.perc_count,
                                 save_train_data_file=self.save_train_data_file,
                                 filtering_rules=self.filtering_rules,
                                 filter_level=self.filter_level,
                                 rule_dictionary=self.rule_dictionary)

    def predict(self, data, model_dir, columns):
        labels = self.l3_classifier.predict(
            data, model_dir=model_dir, columns=columns)
        return labels

    def print_statistics(self):
        self.l3_classifier.rule_statistics.print_statistics()
