import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from atsa.config import LABELS
from atsa.enum import TradingOperationType, BaseEnum
from atsa.assets import Stock
from collections import namedtuple
from workalendar.usa import UnitedStates
from enum import Enum
from datetime import date, datetime, time
from operator import itemgetter
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()


# FIXED_FEE_PER_TRADE = 0
# VAR_FEE_PER_TRADE = 0.0015 # 0.01
# STOP_LOSS_TR = 0.01 # 0.01, 0.05, 0.1
# STOP_LIMIT_TR = 0.015


class CloseCause(BaseEnum):
    INVERSE = "INVERSE"
    STOP_LOSS = "STOPLOSS"
    MAX_LENGTH = "MAXLENGTH"
    LAST_DAY = "LASTDAY"


class TradingOperation:

    def __init__(self,
                 open_index,
                 open_price,
                 cap_invested,
                 optype: TradingOperationType,
                 stock: Stock,
                 stop_loss_tr,
                 max_duration):
        if open_price <= 0:
            raise ValueError(f"Open price zero or negative: {open_price}")
        
        self.open_index = open_index
        self.open_price = open_price
        self.optype = optype
        self.close_index = None
        self.close_price = None
        self.stock = stock
        self.length_in_days = None
        self.cap_invested = cap_invested
        self.price_var = None
        self.gross_cap_returned = None
        self.fees = None
        self.net_cap_returned = None
        self.profit = None      # could be either positive or negative
        self.return_ = None
        self.stop_loss_tr = stop_loss_tr
        self.max_duration = max_duration

        # set a stop loss value by default
        if self.optype == TradingOperationType.LONG:
            self.stop_loss_value = self.open_price * (1 - self.stop_loss_tr)
        else:
            self.stop_loss_value = self.open_price * (1 + self.stop_loss_tr)

        self.close_cause = None

    def __repr__(self):
        return f"TradingOperation({self.optype}," \
               f"{self.open_index}," \
               f"{self.open_price}," \
               f"{self.close_index}," \
               f"{self.close_price}," \
               f"{self.stock.name}," \
               f"{self.cap_invested})" 

    def close(self,
              close_index,
              close_price: float,
              close_cause,
              fixed_fee,
              percentage_fee):
        self.close_index = close_index
        self.close_price = close_price
        self.close_cause = close_cause
        self.length_in_days = int((self.close_index - self.open_index).days)
        self.price_var = (close_price - self.open_price) / self.open_price
        if self.optype == TradingOperationType.LONG:
            self.gross_cap_returned = self.cap_invested + self.cap_invested * self.price_var
        else:
            self.gross_cap_returned = self.cap_invested - self.cap_invested * self.price_var

        self.fees = fixed_fee + self.cap_invested * percentage_fee
        self.net_cap_returned = self.gross_cap_returned - self.fees
        self.profit = 100 * (self.net_cap_returned - self.cap_invested) / self.cap_invested
        self.return_ = self.net_cap_returned - self.cap_invested
        return self.net_cap_returned

    def should_close_by_stop_loss(self, day):
        high_var = (self.stock.high.loc[day] - self.open_price) / self.open_price
        low_var = (self.stock.low.loc[day] - self.open_price) / self.open_price
        if (self.optype == TradingOperationType.LONG and low_var <= -self.stop_loss_tr) or \
            (self.optype == TradingOperationType.SHORT and high_var >= self.stop_loss_tr):
            return True
        else:
            return False

    def should_close_by_max_length(self, day):
        close_index = datetime.combine(day, time(16, 59, 59))
        length_in_days = int((close_index - self.open_index).days)
        if length_in_days >= self.max_duration:
            return True
        return False

    def should_close_by_inverse(self, label):
        if (self.optype == TradingOperationType.LONG and label == LABELS["DOWN"]) or \
            (self.optype == TradingOperationType.SHORT and label == LABELS["UP"]):
            return True
        return False

    def to_dict(self):
        return {
            "stock": self.stock.name,
            'open_day': self.open_index,
            'open_price': self.open_price,
            'close_day': self.close_index,
            'close_price': self.close_price,
            'close_cause': str(self.close_cause),
            'optype': self.optype,
            'length_in_days': self.length_in_days,
            'cap_invested': self.cap_invested,
            'price_var_relative': self.price_var,
            'gross_cap_returned': self.gross_cap_returned,
            'fees': self.fees,
            'net_cap_returned': self.net_cap_returned,
            'profit_perc': self.profit,
            "return": self.return_
        }


class TradingSimulation:
    def __init__(self, initial_capital, stop_loss_tr, investment,
                 stocks,
                 fixed_fee, percentage_fee,
                 max_opened=None,
                 max_opened_per_day=None,
                 max_duration=None
                 ):
        self.initial_capital = initial_capital
        self.stop_loss_tr = stop_loss_tr
        self.investment = investment
        self.stocks = stocks
        self.fixed_fee = fixed_fee
        self.percentage_fee = percentage_fee
        self.max_opened = max_opened
        self.max_opened_per_day = max_opened_per_day
        self.max_duration = max_duration

        self.curr_cap_invested = None
        self.curr_cap_available = None
        self.equity_by_day = None

        self.current_opened = None
        self.pos_history = None
        self.budget_history = None
        self.total_opened_by_day = None
        self.investment_by_day = None
        self.investment_per_stock_by_day = None
        
        self.opened_by_day = None
        self.closed_by_day = None
    
    def get_close_by_stop_loss(self, day):
        return [p for p in self.current_opened if p.should_close_by_stop_loss(day)]

    def get_close_by_max_length(self, day):
        return [p for p in self.current_opened if p.should_close_by_max_length(day)]

    def get_close_by_inverse_signal(self, predicted_labels):
        return [p for p in self.current_opened \
            if p.should_close_by_inverse(predicted_labels.loc[p.stock.name])]

    def update_at_open(self, position, day):
        self.curr_cap_available -= position.cap_invested
        self.curr_cap_invested += position.cap_invested
        self.current_opened.append(position)
        self.opened_by_day[day] += 1

    def update_at_close(self, position, day):
        self.curr_cap_available += position.net_cap_returned
        self.curr_cap_invested -= position.cap_invested
        self.pos_history.append(position)
        self.current_opened.remove(position)
        self.closed_by_day[day] += 1

    def get_available_slots(self):
        return self.max_opened - len(self.current_opened)
        
    def get_stocks_long_short(self, row: pd.Series) -> (list, list):
        ticks_long = [k for k, v in (row == LABELS["UP"]).iteritems() if v]
        ticks_short = [k for k, v in (row == LABELS["DOWN"]).iteritems() if v]
        stocks_long = [s for s in self.stocks if s.name in ticks_long]
        stocks_short = [s for s in self.stocks if s.name in ticks_short]
        return stocks_long, stocks_short

    def remove_already_open(self, stocks_long: list, stock_short: list) -> (list, list):
        return [s for s in stocks_long if s not in self.current_opened], \
            [s for s in stock_short if s not in self.current_opened]

    def get_open_slots(self) -> int:
        if self.max_opened:
            available_slots = self.max_opened - len(self.current_opened)
            if self.max_opened_per_day:
                return min(available_slots, self.max_opened_per_day)
            else:
                return available_slots
        else:
            if self.max_opened_per_day:
                return self.max_opened_per_day
            else:
                return np.inf

    def sort_stocks_by_volume(self, day, stocks_long, stocks_short):
        stock_tuples = list()
        periods = 5

        # tuples for LONG operations come first, i.e. they are preferred
        # in case of equal volume mean and std
        for stock in stocks_long:
            rolling_vol = stock.volume.rolling(periods)
            avg_volume = rolling_vol.mean().loc[day]
            std_volume = rolling_vol.std().loc[day]
            stock_tuples.append((TradingOperationType.LONG, stock, avg_volume, std_volume))
        
        for stock in stocks_short:
            rolling_vol = stock.volume.rolling(periods)
            avg_volume = rolling_vol.mean().loc[day]
            std_volume = rolling_vol.std().loc[day]
            stock_tuples.append((TradingOperationType.SHORT, stock, avg_volume, std_volume))

        stock_tuples = sorted(stock_tuples, key=itemgetter(3)) # the lower the volume std, the better
        stock_tuples = sorted(stock_tuples, key=itemgetter(2), reverse=True) # the higher the volume mean, the better
        return stock_tuples

    def handle_last_day(self, day):
        self.total_opened_by_day[day] = 0
        self.opened_by_day[day] = 0
        self.investment_by_day[day] = 0
        self.investment_per_stock_by_day[day] = 0

        close_index = datetime.combine(day, time(16, 59, 59))
        to_close = self.current_opened.copy()
        for position in to_close:
            if position.should_close_by_stop_loss(day):
                position.close(close_index, position.stop_loss_value, CloseCause.STOP_LOSS,
                               self.fixed_fee, self.percentage_fee)
            elif position.should_close_by_max_length(day):
                position.close(close_index, position.stock.close.loc[day], CloseCause.MAX_LENGTH,
                               self.fixed_fee, self.percentage_fee)
            else:
                position.close(close_index, position.stock.close.loc[day], CloseCause.LAST_DAY,
                               self.fixed_fee, self.percentage_fee)
            self.update_at_close(position, day)

        assert np.isclose([self.curr_cap_invested], [0])
        self.equity_by_day[day] = self.curr_cap_available + self.curr_cap_invested

    def trade(self, signals_df: pd.DataFrame):
        """
        Input:
        - signals_df: pandas Dataframe with the date of the year as index and one column per Stock
        e.g. Data,AAPL,MSFT,EBAY,etc.
        """
        self.current_opened = list()
        self.curr_cap_available = self.initial_capital
        self.curr_cap_invested = 0
        self.equity_by_day = dict()

        self.pos_history = list()

        self.total_opened_by_day = dict()
        self.opened_by_day = {day: 0 for day in signals_df.index}
        self.closed_by_day = {day: 0 for day in signals_df.index}

        self.investment_by_day = dict()
        self.investment_per_stock_by_day = dict()

        self.closed_by_day[signals_df.index[0]] = 0
        for day, row in signals_df.iterrows():
            if day == signals_df.index[-1]:
                self.handle_last_day(day)
                break
            
            close_index = datetime.combine(day, time(16, 59, 59))
            # 1. Close by stop loss
            if len(self.current_opened) > 0:
                for position in self.get_close_by_stop_loss(day):
                    # 'close_index' here should be the moment, in the current day, when the stop loss
                    # value was reached. Here we simplify givin the closing of the market since we don't
                    # have that information.
                    position.close(close_index, position.stop_loss_value, CloseCause.STOP_LOSS,
                                   self.fixed_fee, self.percentage_fee)
                    self.update_at_close(position, day)

            # 2. Close by inverse signal received
            if len(self.current_opened) > 0:
                for position in self.get_close_by_inverse_signal(row):
                    position.close(close_index, position.stock.close.loc[day], CloseCause.INVERSE,
                                   self.fixed_fee, self.percentage_fee)
                    self.update_at_close(position, day)
            
            # 3. Close by max length
            if len(self.current_opened) > 0:
                for position in self.get_close_by_max_length(day):
                    position.close(close_index, position.stock.close.loc[day], CloseCause.MAX_LENGTH,
                                   self.fixed_fee, self.percentage_fee)
                    self.update_at_close(position, day)

            # 4. Detect tradable stocks
            stocks_long, stocks_short = self.get_stocks_long_short(row)
            stocks_long, stocks_short = self.remove_already_open(stocks_long, stocks_short)
            stock_tuples = self.sort_stocks_by_volume(day, stocks_long, stocks_short)   # sort stocks by exchanged volume
            
            # 5. Open new positions
            open_slots = self.get_open_slots()
            to_open_count = min(open_slots, len(stock_tuples))  # the number of operations to open today
            if to_open_count > 0:
                total_investment = (self.curr_cap_available * self.investment)  # uniform investment
                investment_per_stock = total_investment / to_open_count
                stocks_to_open = stock_tuples[:to_open_count]
                open_index = datetime.combine(day, time(16, 59, 59))
                for optype, stock, _, _ in stocks_to_open:
                    open_price = stock.close.loc[day]
                    position = TradingOperation(
                        open_index,
                        open_price,
                        investment_per_stock,
                        optype,
                        stock,
                        self.stop_loss_tr,
                        self.max_duration
                    )
                    self.update_at_open(position, day)
            else:
                total_investment = 0
                investment_per_stock = 0

            # 6. Update the state of the system at the end of each day
            self.equity_by_day[day] = self.curr_cap_available + self.curr_cap_invested
            self.investment_by_day[day] = total_investment
            self.investment_per_stock_by_day[day] = investment_per_stock
            self.total_opened_by_day[day] = len(self.current_opened)

        # 7. Collect results at the end of the year
        self.check_results(signals_df)
        columns = [
            pd.Series(self.equity_by_day),
            pd.Series(self.total_opened_by_day),
            pd.Series(self.opened_by_day),
            pd.Series(self.closed_by_day),
            pd.Series(self.investment_by_day),
            pd.Series(self.investment_per_stock_by_day)
        ]
        names = [
            "equity_by_day",
            "total_opened",
            "opened_by_day",
            "closed_by_day",
            "investment_by_day",
            "investment_per_stock_by_day"
        ]
        
        # these metrics are sampled at 5PM of each day, except for pre open budget
        results_df = pd.concat(columns, axis=1)
        results_df.columns = names
        assert (results_df.index == signals_df.index).all()
        return self.pos_history, results_df

    def check_results(self, signals_df: pd.DataFrame):
        assert not self.current_opened

        # print(signals_df.shape[0],
        #     len(self.equity_by_day),
        #     len(self.total_opened_by_day),
        #     len(self.opened_by_day),
        #     len(self.closed_by_day),
        #     len(self.investment_by_day),
        #     len(self.investment_per_stock_by_day))

        assert signals_df.shape[0] == \
            len(self.equity_by_day) == \
            len(self.total_opened_by_day) == \
            len(self.opened_by_day) == \
            len(self.closed_by_day) == \
            len(self.investment_by_day) == \
            len(self.investment_per_stock_by_day)
