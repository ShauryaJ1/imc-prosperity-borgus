from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List
import string
import math
import numpy as np

import json
from typing import Any

from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState


class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        base_length = len(
            self.to_json(
                [
                    self.compress_state(state, ""),
                    self.compress_orders(orders),
                    conversions,
                    "",
                    "",
                ]
            )
        )

        # We truncate state.traderData, trader_data, and self.logs to the same max. length to fit the log limit
        max_item_length = (self.max_log_length - base_length) // 3

        print(
            self.to_json(
                [
                    self.compress_state(state, self.truncate(state.traderData, max_item_length)),
                    self.compress_orders(orders),
                    conversions,
                    self.truncate(trader_data, max_item_length),
                    self.truncate(self.logs, max_item_length),
                ]
            )
        )

        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
        return [
            state.timestamp,
            trader_data,
            self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths),
            self.compress_trades(state.own_trades),
            self.compress_trades(state.market_trades),
            state.position,
            self.compress_observations(state.observations),
        ]

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
        compressed = []
        for listing in listings.values():
            compressed.append([listing.symbol, listing.product, listing.denomination])

        return compressed

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        compressed = {}
        for symbol, order_depth in order_depths.items():
            compressed[symbol] = [order_depth.buy_orders, order_depth.sell_orders]

        return compressed

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        compressed = []
        for arr in trades.values():
            for trade in arr:
                compressed.append(
                    [
                        trade.symbol,
                        trade.price,
                        trade.quantity,
                        trade.buyer,
                        trade.seller,
                        trade.timestamp,
                    ]
                )

        return compressed

    def compress_observations(self, observations: Observation) -> list[Any]:
        conversion_observations = {}
        for product, observation in observations.conversionObservations.items():
            conversion_observations[product] = [
                observation.bidPrice,
                observation.askPrice,
                observation.transportFees,
                observation.exportTariff,
                observation.importTariff,
                observation.sugarPrice,
                observation.sunlightIndex,
            ]

        return [observations.plainValueObservations, conversion_observations]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        compressed = []
        for arr in orders.values():
            for order in arr:
                compressed.append([order.symbol, order.price, order.quantity])

        return compressed

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        if len(value) <= max_length:
            return value

        return value[: max_length - 3] + "..."


logger = Logger()

class Strategy:
    @staticmethod
    def compute_optimal_quotes(S_t, sigma, t, gamma, k, q):
        # Time remaining
        tau = 1 - t
        # Inventory adjustment term (if any) can be added here if you track net inventory.
        risk_term = (gamma * sigma ** 2 * tau) / 2
        adjustment_term = (1 / gamma) * np.log(1 + gamma / k)

        # Optimal price offsets
        delta_bid = risk_term + adjustment_term
        delta_ask = risk_term + adjustment_term  # symmetric if no inventory imbalance

        logger.print(risk_term)
        logger.print(adjustment_term)
        logger.print(delta_bid + delta_ask)

        R_t = S_t - (q * gamma * sigma ** 2) * tau

        bid_price = R_t - delta_bid
        ask_price = R_t + delta_ask
        return bid_price, ask_price


class Trader:

    def __init__(self):
        self.RISK = 0.14
        self.PROFIT_TARGET = {"KELP": 2, "RAINFOREST_RESIN": 1}
        self.POSITION_LIMIT = {
            "KELP": 50,
            "RAINFOREST_RESIN": 50
        }
        self.VOLUMES = {
            "KELP": 5,
            "RAINFOREST_RESIN": 10
        }
        self.DEFAULT_PRICES = {
            "KELP": 2000,
            "RAINFOREST_RESIN": 10000
        }
        self.PRODUCTS = [
            "KELP",
            "RAINFOREST_RESIN"
        ]
        self.PAST_PRICES = {
            product: [] for product in self.PRODUCTS
        }

        self.MCGINLEY_PRICES = {
            product: None for product in self.PRODUCTS
        }
        # self.spreads = {
        #     product:0 for product in self.products
        # }
        self.SPREADS = {
            "KELP": 4,
            "RAINFOREST_RESIN": 7
        }
        self.IDEAL_PROFITS = {
            "KELP": 2,
            "RAINFOREST_RESIN": 4
        }

        self.MCGINLEY_PERIOD = 8

        self.AVELLANADA_PARAMS = {
            "gamma": 1,  # Risk aversion parameter (example)
            "A": 1.0,  # Baseline arrival rate (example value)
            "k": 1.0,  # Sensitivity of arrival rate (example value)
            "nu": 0
        }

        self.MARKET_ORDERS = {
            product: [] for product in self.PRODUCTS
        }

    def get_price(self, symbol, state: TradingState):
        return (max(state.order_depths[symbol].buy_orders) + min(state.order_depths[symbol].sell_orders)) / 2

    def update_spreads(self, state):
        for symbol in self.PRODUCTS:
            self.SPREADS[symbol] = min(state.order_depths[symbol].sell_orders) - max(
                state.order_depths[symbol].buy_orders)

    def update_market_orders(self, state):
        for symbol in self.PRODUCTS:
            current_price = self.get_price(symbol, state)
            dct = dict()
            actual_market = state.market_trades[symbol] if symbol in state.market_trades else []
            actual_own = state.own_trades[symbol] if symbol in state.own_trades else []
            for trade in actual_market + actual_own:
                if abs(trade.price - current_price) not in dct:
                    dct[abs(trade.price - current_price)] = 0
                dct[abs(trade.price - current_price)] += trade.quantity
            self.MARKET_ORDERS[symbol].append(dct)

    def update_k(self, symbol, state):
        master_dct = dict()
        for dct in self.MARKET_ORDERS[symbol][max(0, len(self.MARKET_ORDERS[symbol]) - 100) : ]:
            for price in dct:
                if price not in master_dct:
                    master_dct[price] = 0
                master_dct[price] += dct[price]
        #run exponential regression
        x = np.array(list(master_dct.keys()))
        y = np.array(list(master_dct.values()))

        if len(x) == 0:
            self.AVELLANADA_PARAMS["k"] = 1
            return

        log_y = np.log(y)
        slope, intercept = np.polyfit(x, log_y, 1)
        # if state.timestamp == 100000:
        #     raise Exception(f"{slope}")
        logger.print(slope)
        if slope > 0:
            self.AVELLANADA_PARAMS["k"] = 0.1
            return
        self.AVELLANADA_PARAMS["k"] = -slope

    def update_prices(self, state):
        if state.timestamp % 1 == 0:
            for symbol in self.PRODUCTS:
                self.PAST_PRICES[symbol].append(self.get_price(symbol, state))

    def update_mcginley(self, symbol, state):
        mid_price = self.get_price(symbol, state)
        if self.MCGINLEY_PRICES[symbol] is None:
            self.MCGINLEY_PRICES[symbol] = mid_price
        else:
            self.MCGINLEY_PRICES[symbol] = self.MCGINLEY_PRICES[symbol] + (mid_price - self.MCGINLEY_PRICES[symbol]) / (
                    0.6 * self.MCGINLEY_PERIOD * (mid_price / self.MCGINLEY_PRICES[symbol]) ** 4)

    def resin_trading(self, state: TradingState):
        if "RAINFOREST_RESIN" not in state.position:
            current_position = 0
        else:
            current_position = state.position["RAINFOREST_RESIN"]
        bid_volume = self.POSITION_LIMIT["RAINFOREST_RESIN"] - current_position
        ask_volume = - self.POSITION_LIMIT["RAINFOREST_RESIN"] - current_position

        orders = []
        orders.append(Order("RAINFOREST_RESIN", math.floor(
            self.DEFAULT_PRICES["RAINFOREST_RESIN"] - self.IDEAL_PROFITS["RAINFOREST_RESIN"] / 2), bid_volume))
        orders.append(Order("RAINFOREST_RESIN", math.ceil(
            self.DEFAULT_PRICES["RAINFOREST_RESIN"] + self.IDEAL_PROFITS["RAINFOREST_RESIN"] / 2), ask_volume))
        return orders

    def resin_risk_spread(self, state: TradingState):
        if "RAINFOREST_RESIN" not in state.position:
            current_position = 0
        else:
            current_position = state.position["RAINFOREST_RESIN"]
        bid_volume = self.POSITION_LIMIT["RAINFOREST_RESIN"] - current_position
        ask_volume = - self.POSITION_LIMIT["RAINFOREST_RESIN"] - current_position
        # bid_volume = self.volumes["RAINFOREST_RESIN"]
        # ask_volume = self.volumes["RAINFOREST_RESIN"]
        orders = []
        orders.append(Order("RAINFOREST_RESIN", math.floor(
            self.DEFAULT_PRICES["RAINFOREST_RESIN"] - self.IDEAL_PROFITS["RAINFOREST_RESIN"] / 2 - self.SPREADS[
                "RAINFOREST_RESIN"] / 2), bid_volume))
        orders.append(Order("RAINFOREST_RESIN", math.ceil(
            self.DEFAULT_PRICES["RAINFOREST_RESIN"] + self.IDEAL_PROFITS["RAINFOREST_RESIN"] / 2 + self.SPREADS[
                "RAINFOREST_RESIN"] / 2), ask_volume))
        return orders

    def kelp_trading_mcginley_risk_spread(self, state: TradingState):
        if "KELP" not in state.position:
            current_position = 0
        else:
            current_position = state.position["KELP"]
        bid_volume = self.POSITION_LIMIT["KELP"] - current_position
        ask_volume = - self.POSITION_LIMIT["KELP"] - current_position
        # bid_volume = self.volumes["KELP"]
        # ask_volume = self.volumes["KELP"]
        orders = []

        if current_position == 0:
            # Not long nor short
            orders.append(Order("KELP", math.floor(
                self.MCGINLEY_PRICES["KELP"] - self.IDEAL_PROFITS["KELP"] / 2 - self.SPREADS["KELP"] / 2), bid_volume))
            orders.append(Order("KELP", math.ceil(
                self.MCGINLEY_PRICES["KELP"] + self.IDEAL_PROFITS["KELP"] / 2 + self.SPREADS["KELP"] / 2), ask_volume))

        if current_position > 0:
            # Long position
            orders.append(Order("KELP", math.floor(
                self.MCGINLEY_PRICES["KELP"] - self.IDEAL_PROFITS["KELP"] - current_position * self.RISK - self.SPREADS[
                    "KELP"] / 2), bid_volume))
            orders.append(Order("KELP", math.ceil(
                self.MCGINLEY_PRICES["KELP"] + -current_position * self.RISK + self.SPREADS["KELP"] / 2), ask_volume))

        if current_position < 0:
            # Short position
            orders.append(Order("KELP", math.floor(
                self.MCGINLEY_PRICES["KELP"] - current_position * self.RISK - self.SPREADS["KELP"] / 2), bid_volume))
            orders.append(Order("KELP", math.ceil(
                self.MCGINLEY_PRICES["KELP"] + self.IDEAL_PROFITS["KELP"] - current_position * self.RISK + self.SPREADS[
                    "KELP"] / 2), ask_volume))

        return orders

    def kelp_trading_avellaneda_stoikov(self, symbol, state: TradingState):
        current_position = state.position[symbol] if symbol in state.position else 0
        S_t = self.PAST_PRICES[symbol][-1]
        sigma = np.std(np.array(   [math.log(self.PAST_PRICES[symbol][i] / self.PAST_PRICES[symbol][i-1]) for i in range(max(1, len(self.PAST_PRICES[symbol]) - 100), len(self.PAST_PRICES[symbol]) )]))
        sigma_scaled = sigma * 2000 ** 0.5
        t = state.timestamp / 200000
        best_bid, best_ask = Strategy.compute_optimal_quotes(S_t, sigma_scaled, t, self.AVELLANADA_PARAMS["gamma"], self.AVELLANADA_PARAMS["k"], current_position)
        # best_bid += self.AVELLANADA_PARAMS["nu"] * (state.position[symbol] if symbol in state.position else 0)
        # best_ask -= self.AVELLANADA_PARAMS["nu"] * (state.position[symbol] if symbol in state.position else 0)

        try:
            bid_volume = self.POSITION_LIMIT[symbol] - state.position[symbol]
            ask_volume = - self.POSITION_LIMIT[symbol] - state.position[symbol]
        except:
            bid_volume = self.POSITION_LIMIT[symbol]
            ask_volume = - self.POSITION_LIMIT[symbol]

        orders = []
        orders.append(Order(symbol, math.floor(best_bid), bid_volume))
        orders.append(Order(symbol, math.ceil(best_ask), ask_volume))
        return orders

    def kelp_trading_borgus_moment(self, state: TradingState):
        fair_bid = -1
        for bid_amt in state.order_depths["KELP"].buy_orders:
            if fair_bid == -1 or state.order_depths["KELP"].buy_orders[bid_amt] >= state.order_depths["KELP"].buy_orders[fair_bid]:
                fair_bid = bid_amt

        fair_ask = -1
        for ask_amt in state.order_depths["KELP"].sell_orders:
            if fair_ask == -1 or state.order_depths["KELP"].sell_orders[ask_amt] <= state.order_depths["KELP"].sell_orders[fair_ask]:
                fair_ask = ask_amt

        if fair_bid == -1 or fair_ask == -1:
            raise Exception(f"{state.order_depths['KELP'].buy_orders} {state.order_depths['KELP'].sell_orders}")

        if "KELP" not in state.position:
            current_position = 0
        else:
            current_position = state.position["KELP"]
        bid_volume = self.POSITION_LIMIT["KELP"] - current_position
        ask_volume = - self.POSITION_LIMIT["KELP"] - current_position
        # bid_volume = self.volumes["KELP"]
        # ask_volume = self.volumes["KELP"]
        orders = []

        fair_price = (fair_bid + fair_ask) / 2

        if -10 < current_position and current_position < 10:
            # Not long nor short
            orders.append(Order("KELP", math.floor(
                 fair_price - self.IDEAL_PROFITS["KELP"] / 2), bid_volume))
            orders.append(Order("KELP", math.ceil(
                 fair_price + self.IDEAL_PROFITS["KELP"] / 2), ask_volume))


        elif current_position > 0:
            # Long position
            orders.append(Order("KELP", math.floor(
                fair_price - self.IDEAL_PROFITS["KELP"]), bid_volume))
            orders.append(Order("KELP", math.ceil(
                fair_ask), ask_volume))

        elif current_position < 0:
            # Short position
            orders.append(Order("KELP", math.floor(
                fair_bid), bid_volume))
            orders.append(Order("KELP", math.ceil(
                fair_price + self.IDEAL_PROFITS["KELP"]), ask_volume))

        return orders



    def run(self, state: TradingState):
        # print("Beginning of run")
        # print(self.position_limit)
        # Only method required. It takes all buy and sell orders for all symbols as an input, and outputs a list of orders to be sent
        for product in self.PRODUCTS:
            self.update_mcginley(product, state)
            self.update_spreads(state)
            self.update_prices(state)
            self.update_market_orders(state)
            self.update_k("KELP", state)

        # print("Mid price of : ", self.get_price)
        orders = {}


        # orders["KELP"] = []
        # ords, pos = Strategy.arb(self, state, "KELP", self.get_price("KELP", state))
        # orders["KELP"].extend(ords)

        orders["KELP"] = self.kelp_trading_avellaneda_stoikov("KELP", state)
        # orders.extend(self.resin_trading(state    ))
        orders["RAINFOREST_RESIN"] = self.resin_trading(state)

        logger.flush(state, orders, 1, state.traderData)

        return orders, 1, state.traderData

