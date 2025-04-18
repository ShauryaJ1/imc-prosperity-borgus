from datamodel import OrderDepth, UserId, TradingState, Order, Symbol, Listing, Trade, Observation, ProsperityEncoder
from typing import List
import string
import jsonpickle
import numpy as np
import math
from typing import Any, List, Dict
from json import *
import copy

from math import log, sqrt, exp
from statistics import NormalDist


# if momentum going down short it immediately (sell everything)
# if momentum going up, buy everything (long position wow?)


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
        return dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        if len(value) <= max_length:
            return value

        return value[: max_length - 3] + "..."


logger = Logger()

class BlackScholes:
    @staticmethod
    def black_scholes_call(spot, strike, time_to_expiry, volatility):
        d1 = (
            log(spot) - log(strike) + (0.5 * volatility * volatility) * time_to_expiry
        ) / (volatility * sqrt(time_to_expiry))
        d2 = d1 - volatility * sqrt(time_to_expiry)
        call_price = spot * NormalDist().cdf(d1) - strike * NormalDist().cdf(d2)
        return call_price

    @staticmethod
    def black_scholes_put(spot, strike, time_to_expiry, volatility):
        d1 = (log(spot / strike) + (0.5 * volatility * volatility) * time_to_expiry) / (
            volatility * sqrt(time_to_expiry)
        )
        d2 = d1 - volatility * sqrt(time_to_expiry)
        put_price = strike * NormalDist().cdf(-d2) - spot * NormalDist().cdf(-d1)
        return put_price

    @staticmethod
    def delta(spot, strike, time_to_expiry, volatility):
        d1 = (
            log(spot) - log(strike) + (0.5 * volatility * volatility) * time_to_expiry
        ) / (volatility * sqrt(time_to_expiry))
        return NormalDist().cdf(d1)

    @staticmethod

    def gamma(spot, strike, time_to_expiry, volatility):
        d1 = (
            log(spot) - log(strike) + (0.5 * volatility * volatility) * time_to_expiry
        ) / (volatility * sqrt(time_to_expiry))
        return NormalDist().pdf(d1) / (spot * volatility * sqrt(time_to_expiry))

    @staticmethod
    def vega(spot, strike, time_to_expiry, volatility):
        d1 = (
            log(spot) - log(strike) + (0.5 * volatility * volatility) * time_to_expiry
        ) / (volatility * sqrt(time_to_expiry))
        # print(f"d1: {d1}")
        # print(f"vol: {volatility}")
        # print(f"spot: {spot}")
        # print(f"strike: {strike}")
        # print(f"time: {time_to_expiry}")
        return NormalDist().pdf(d1) * (spot * sqrt(time_to_expiry)) / 100

    @staticmethod
    def implied_volatility(
        call_price, spot, strike, time_to_expiry, max_iterations=200, tolerance=1e-10
    ):
        low_vol = 0.01
        high_vol = 1.0
        volatility = (low_vol + high_vol) / 2.0  # Initial guess as the midpoint
        for _ in range(max_iterations):
            estimated_price = BlackScholes.black_scholes_call(
                spot, strike, time_to_expiry, volatility
            )
            diff = estimated_price - call_price
            if abs(diff) < tolerance:
                break
            elif diff > 0:
                high_vol = volatility
            else:
                low_vol = volatility
            volatility = (low_vol + high_vol) / 2.0
        return volatility


class Product:
    RAINFOREST_RESIN = "RAINFOREST_RESIN"
    KELP = "KELP"
    SQUID_INK = "SQUID_INK"
    CROISSANTS = "CROISSANTS"
    JAMS = "JAMS"
    DJEMBES = "DJEMBES"
    PICNIC_BASKET1 = "PICNIC_BASKET1"
    PICNIC_BASKET2 = "PICNIC_BASKET2"
    SPREAD = "SPREAD"
    SPREAD2 = "SPREAD2"
    SYNTHETIC = "SYNTHETIC"
    SYNTHETIC2 = "SYNTHETIC2"
    VOLCANIC_ROCK = "VOLCANIC_ROCK"
    VOLCANIC_ROCK_VOUCHER_9500 = "VOLCANIC_ROCK_VOUCHER_9500"
    VOLCANIC_ROCK_VOUCHER_9750 = "VOLCANIC_ROCK_VOUCHER_9750"
    VOLCANIC_ROCK_VOUCHER_10000 = "VOLCANIC_ROCK_VOUCHER_10000"
    VOLCANIC_ROCK_VOUCHER_10250 = "VOLCANIC_ROCK_VOUCHER_10250"
    VOLCANIC_ROCK_VOUCHER_10500 = "VOLCANIC_ROCK_VOUCHER_10500"
    MAGNIFICENT_MACARONS = "MAGNIFICENT_MACARONS"


PARAMS = {
    Product.RAINFOREST_RESIN: {
        "fair_value": 10000,
        "take_width": 1,
        "clear_width": 0,
        # for making
        "disregard_edge": 1,  # disregards orders for joining or pennying within this value from fair
        "join_edge": 2,  # joins orders within this edge
        "default_edge": 4,
        "soft_position_limit": 35,
    },
    Product.KELP: {
        "take_width": 1,
        "clear_width": 0,
        "prevent_adverse": True,
        "adverse_volume": 15,
        "reversion_beta": -0.229,
        "disregard_edge": 1,
        "join_edge": 0,
        "default_edge": 1,
        "momentum_weight": 0.40,
        "history_window": 3,
        "momentum_cutoff": 0.1
    },
    Product.MAGNIFICENT_MACARONS: {
        "take_width": 1,
        "clear_width": 0,
        "prevent_adverse": True,
        "adverse_volume": 15,
        "reversion_beta": -0.229,
        "disregard_edge": 1,
        "join_edge": 0,
        "default_edge": 1,
        "momentum_weight": 0.40,
        "history_window": 3,
        "momentum_cutoff": 0.1
    },
    Product.SQUID_INK: {
        "take_width": 1,
        "clear_width": 1,
        "prevent_adverse": True,
        "adverse_volume": 15,
        "reversion_beta": 0.2,
        "disregard_edge": 1,
        "join_edge": 0,
        "default_edge": 1,
        "momentum_weight": 0.08,
        "small_window": 8,
        "history_window": 20,  # 3 for momentum, #20 for ema with prev=fair
        "stdev_mul": 2,
        "momentum_cutoff": 0.05,
        "ema_param": 0.7,  # 0.62 gives 5,513 with claerwidth = 4
        "fft_cutoff": 0.2,
        "entry_price": -1,  # empty right now, set it to whatever mmmid we entered at
    },
    Product.SPREAD: {
        "default_spread_mean": 48.762433333333334,
        "default_spread_std": 85.1180321401536,
        "spread_std_window": 45,
        "zscore_threshold": 3.01,
        "target_position": 60,
    },
    Product.SPREAD2: {
        "default_spread_mean_upper": 100,
        "default_spread_mean_lower": -45,
        "default_spread_mean": 30.23,
        "default_spread_std": 85.1180321401536,
        "spread_std_window": 60,
        "zscore_threshold": 6,
        "target_position": 100,
    },
    Product.VOLCANIC_ROCK_VOUCHER_9500: {
        "mean_volatility": 0.15, #0.005
        "strike": 9500,
        "starting_time_to_expiry": 247 / 250,
        "std_window": 3,
        "zscore_threshold": 21,
        "take_width": 1, 
        "clear_width": 0,
        "disregard_edge": 1,  # disregards orders for joining or pennying within this value from fair
        "join_edge": 2,  # joins orders within this edge
        "default_edge": 4, 
        "prevent_adverse": True,
        "adverse_volume": 15,
    },
    Product.VOLCANIC_ROCK_VOUCHER_9750: {
        "mean_volatility": 0.15,
        "strike": 9750,
        "starting_time_to_expiry": 247 / 250,
        "std_window": 3,
        "zscore_threshold": 21,
        "take_width": 1,
        "clear_width": 0,
        "disregard_edge": 1,  # disregards orders for joining or pennying within this value from fair
        "join_edge": 2,  # joins orders within this edge
        "default_edge": 4,
        "prevent_adverse": True,
        "adverse_volume": 15,
    },
    Product.VOLCANIC_ROCK_VOUCHER_10000: {
        "mean_volatility": 0.15,
        "strike": 10000,
        "starting_time_to_expiry": 247 / 250,
        "std_window": 3,
        "zscore_threshold": 21,
        "take_width": 1,
        "clear_width": 0,
        "disregard_edge": 1,  # disregards orders for joining or pennying within this value from fair
        "join_edge": 2,  # joins orders within this edge
        "default_edge": 4,
        "prevent_adverse": True,
        "adverse_volume": 15,
    },
    Product.VOLCANIC_ROCK_VOUCHER_10250: {
        "mean_volatility": 0.15,
        "strike": 10250,
        "starting_time_to_expiry": 247 / 250,
        "std_window": 3,
        "zscore_threshold": 21,
        "take_width": 1,
        "clear_width": 0,
        "disregard_edge": 1,  # disregards orders for joining or pennying within this value from fair
        "join_edge": 2,  # joins orders within this edge
        "default_edge": 4,
        "prevent_adverse": True,
        "adverse_volume": 15,
    },
    Product.VOLCANIC_ROCK_VOUCHER_10500: {
        "mean_volatility": 0.15,
        # "threshold": 0.00163,
        "strike": 10500,
        "starting_time_to_expiry": 247 / 250,
        "std_window": 3,
        "zscore_threshold": 21,
        "take_width": 1,
        "clear_width": 0,
        "disregard_edge": 1,  # disregards orders for joining or pennying within this value from fair
        "join_edge": 2,  # joins orders within this edge
        "default_edge": 4,
        "prevent_adverse": True,
        "adverse_volume": 15,
    },
    Product.VOLCANIC_ROCK:{
        "take_width": 1,
        "clear_width": 0,
        "prevent_adverse": True,
        "adverse_volume": 15,
        "reversion_beta": -0.229,
        "disregard_edge": 1,
        "join_edge": 0,
        "default_edge": 1,
        "momentum_weight": 0.40,
        "history_window": 3,
        "momentum_cutoff": 0.1,
    },
    Product.CROISSANTS:{
        "take_width": 1,
        "clear_width": 0,
        "prevent_adverse": True,
        "adverse_volume": 15,
        "reversion_beta": -0.229,
        "disregard_edge": 1,
        "join_edge": 0,
        "default_edge": 1,
        "momentum_weight": 0.40,
        "history_window": 3,
        "momentum_cutoff": 0.1
    },
    Product.DJEMBES:{
        "take_width": 1,
        "clear_width": 0,
        "prevent_adverse": True,
        "adverse_volume": 15,
        "reversion_beta": -0.229,
        "disregard_edge": 1,
        "join_edge": 0,
        "default_edge": 1,
        "momentum_weight": 0.40,
        "history_window": 3,
        "momentum_cutoff": 0.1
    },
    Product.JAMS:{
        "take_width": 1,
        "clear_width": 0,
        "prevent_adverse": True,
        "adverse_volume": 15,
        "reversion_beta": -0.229,
        "disregard_edge": 1,
        "join_edge": 0,
        "default_edge": 1,
        "momentum_weight": 0.40,
        "history_window": 3,
        "momentum_cutoff": 0.1
    },

}

BASKET_WEIGHTS = {
    Product.CROISSANTS: 6,
    Product.JAMS: 3,
    Product.DJEMBES: 1
}

BASKET2_WEIGHTS = {
    Product.CROISSANTS: 4,
    Product.JAMS: 2,
    Product.DJEMBES: 0
}

BASKETS = {
    1: Product.PICNIC_BASKET1,
    2: Product.PICNIC_BASKET2
}
SPREADS = {
    1: Product.SPREAD,
    2: Product.SPREAD2
}
SYNTHETICS = {
    1: Product.SYNTHETIC,
    2: Product.SYNTHETIC2
}

WEIGHTS = {
    1: BASKET_WEIGHTS,
    2: BASKET2_WEIGHTS
}

class Trader:
    def __init__(self, params=None):
        if params is None:
            params = PARAMS
        self.params = params

        self.LIMIT = {Product.RAINFOREST_RESIN: 50,
                      Product.KELP: 50,
                      Product.SQUID_INK: 50,
                      Product.CROISSANTS: 250,
                      Product.JAMS: 350,
                      Product.DJEMBES: 60,
                      Product.PICNIC_BASKET1: 60,
                      Product.PICNIC_BASKET2: 100,
                      Product.VOLCANIC_ROCK: 400,
                      Product.VOLCANIC_ROCK_VOUCHER_9500: 200,
                      Product.VOLCANIC_ROCK_VOUCHER_9750: 200,
                      Product.VOLCANIC_ROCK_VOUCHER_10000: 200,
                      Product.VOLCANIC_ROCK_VOUCHER_10250: 200,
                      Product.VOLCANIC_ROCK_VOUCHER_10500: 200,
                      Product.MAGNIFICENT_MACARONS: 75,
                      }

        self.past_prices = {
            Product.VOLCANIC_ROCK: [],
        }

        self.log_returns = {
            Product.VOLCANIC_ROCK: [],
        }

        self.trader_memory = {
            "ink_price_history": [],
            "kelp_price_history": [],
            "volitality_arr": [],
            "z_score_arr": [],
            "m_t": [],
            "v_t": [],
            "croissants_price_history": [],
            "djembes_price_history": [],
            "jams_price_history": [],
            "magnificent_macarons_price_history":[],
        }

    def take_best_orders(
            self,
            product: str,
            fair_value: int,
            take_width: float,
            orders: List[Order],
            order_depth: OrderDepth,
            position: int,
            buy_order_volume: int,
            sell_order_volume: int,
            prevent_adverse: bool = False,
            adverse_volume: int = 0,
    ) -> (int, int):
        position_limit = self.LIMIT[product]

        if len(order_depth.sell_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_ask_amount = -1 * order_depth.sell_orders[best_ask]

            if not prevent_adverse or abs(best_ask_amount) <= adverse_volume:
                # if the current best ask order isn't intentionally malicious (to topple us off of 0 ev)
                # note: since prevent_adverse isn't on, we don't really care...
                if best_ask <= fair_value - take_width:
                    # best_ask <= 9999
                    quantity = min(
                        best_ask_amount, position_limit - position
                    )  # max amt to buy

                    if quantity > 0:
                        orders.append(Order(product, best_ask, quantity))
                        buy_order_volume += quantity
                        order_depth.sell_orders[best_ask] += quantity
                        if order_depth.sell_orders[best_ask] == 0:
                            del order_depth.sell_orders[best_ask]

        if len(order_depth.buy_orders) != 0:
            best_bid = max(order_depth.buy_orders.keys())
            best_bid_amount = order_depth.buy_orders[best_bid]

            if not prevent_adverse or abs(best_bid_amount) <= adverse_volume:
                # if the current best buy order isn't intentionally malicious (to topple us off of 0 ev)
                # note: since prevent_adverse isn't on, we don't really care...
                if best_bid >= fair_value + take_width:
                    # best_bid >= 10001
                    quantity = min(
                        best_bid_amount, position_limit + position
                    )  # should be the max we can sell
                    if quantity > 0:
                        orders.append(Order(product, best_bid, -1 * quantity))
                        sell_order_volume += quantity
                        order_depth.buy_orders[best_bid] -= quantity
                        if order_depth.buy_orders[best_bid] == 0:
                            del order_depth.buy_orders[best_bid]

        return buy_order_volume, sell_order_volume

    def market_make(
            self,
            product: str,
            orders: List[Order],
            bid: int,
            ask: int,
            position: int,
            buy_order_volume: int,
            sell_order_volume: int,
    ) -> (int, int):
        buy_quantity = self.LIMIT[product] - (position + buy_order_volume)
        if buy_quantity > 0:
            orders.append(Order(product, round(bid), buy_quantity))  # Buy order

        sell_quantity = self.LIMIT[product] + (position - sell_order_volume)
        if sell_quantity > 0:
            orders.append(Order(product, round(ask), -sell_quantity))  # Sell order
        return buy_order_volume, sell_order_volume

    def clear_position_order(
            self,
            product: str,
            fair_value: float,
            width: int,
            orders: List[Order],
            order_depth: OrderDepth,
            position: int,
            buy_order_volume: int,
            sell_order_volume: int,
    ) -> List[Order]:
        position_after_take = position + buy_order_volume - sell_order_volume
        fair_for_bid = round(fair_value - width)
        fair_for_ask = round(fair_value + width)

        buy_quantity = self.LIMIT[product] - (position + buy_order_volume)
        sell_quantity = self.LIMIT[product] + (position - sell_order_volume)

        if position_after_take > 0:
            # Aggregate volume from all buy orders with price greater than or equal to fair_for_ask
            clear_quantity = sum(
                volume
                for price, volume in order_depth.buy_orders.items()
                if price >= fair_for_ask
            )
            clear_quantity = min(clear_quantity, position_after_take)
            sent_quantity = min(sell_quantity, clear_quantity)
            if sent_quantity > 0:
                orders.append(Order(product, fair_for_ask, -abs(sent_quantity)))
                sell_order_volume += abs(sent_quantity)

        if position_after_take < 0:
            # Aggregate volume from all sell orders with price lower than fair_for_bid
            clear_quantity = sum(
                abs(volume)
                for price, volume in order_depth.sell_orders.items()
                if price <= fair_for_bid
            )
            clear_quantity = min(clear_quantity, abs(position_after_take))
            sent_quantity = min(buy_quantity, clear_quantity)
            if sent_quantity > 0:
                orders.append(Order(product, fair_for_bid, abs(sent_quantity)))
                buy_order_volume += abs(sent_quantity)

        return buy_order_volume, sell_order_volume

    def kelp_fair_value(self, order_depth: OrderDepth, traderObject) -> float:
        if len(order_depth.sell_orders) != 0 and len(order_depth.buy_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())
            filtered_ask = [
                price
                for price in order_depth.sell_orders.keys()
                if abs(order_depth.sell_orders[price])
                   >= self.params[Product.KELP]["adverse_volume"]
            ]
            filtered_bid = [
                price
                for price in order_depth.buy_orders.keys()
                if abs(order_depth.buy_orders[price])
                   >= self.params[Product.KELP]["adverse_volume"]
            ]
            mm_ask = min(filtered_ask) if len(filtered_ask) > 0 else None
            mm_bid = max(filtered_bid) if len(filtered_bid) > 0 else None
            if mm_ask == None or mm_bid == None:
                if traderObject.get("kelp_last_price", None) == None:
                    mmmid_price = (best_ask + best_bid) / 2
                else:
                    mmmid_price = traderObject["kelp_last_price"]
            else:
                mmmid_price = (mm_ask + mm_bid) / 2

            # estimate with reversion
            if traderObject.get("kelp_last_price", None) != None:
                last_price = traderObject["kelp_last_price"]
                last_returns = (mmmid_price - last_price) / last_price
                pred_returns = (
                        last_returns * self.params[Product.KELP]["reversion_beta"]
                )
                fair = mmmid_price + (mmmid_price * pred_returns)
            else:
                fair = mmmid_price

            price_history: List[float] = self.trader_memory.get("kelp_price_history", [])
            # adjust with momentum?
            if len(price_history) >= 2:
                # momentum = (price_history[-1] - price_history[0])

                x = np.arange(
                    len(price_history))  # for linreg, its literally better to have a negative momentum weight?
                y = np.array(price_history)
                slope, intercept = np.polyfit(x, y, 1)
                y_pred = slope * x + intercept

                ss_res = np.sum((y - y_pred) ** 2)
                ss_tot = np.sum((y - np.mean(y)) ** 2)
                r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

                momentum = 0
                if r_squared >= self.params[Product.KELP]["momentum_cutoff"]:
                    momentum = slope

                fair += self.params[Product.KELP]["momentum_weight"] * momentum

            traderObject["kelp_last_price"] = fair

            # update price_history
            price_history.append(fair)

            if len(price_history) > self.params[Product.KELP]["history_window"]:
                price_history.pop(0)

            self.trader_memory["kelp_price_history"] = price_history

            return fair
        return None

    def volcanic_rock_fair_value(self, order_depth: OrderDepth, traderObject) -> float:
        if len(order_depth.sell_orders) != 0 and len(order_depth.buy_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())
            filtered_ask = [
                price
                for price in order_depth.sell_orders.keys()
                if abs(order_depth.sell_orders[price])
                   >= self.params[Product.VOLCANIC_ROCK]["adverse_volume"]
            ]
            filtered_bid = [
                price
                for price in order_depth.buy_orders.keys()
                if abs(order_depth.buy_orders[price])
                   >= self.params[Product.VOLCANIC_ROCK]["adverse_volume"]
            ]
            mm_ask = min(filtered_ask) if len(filtered_ask) > 0 else None
            mm_bid = max(filtered_bid) if len(filtered_bid) > 0 else None
            if mm_ask == None or mm_bid == None:
                if traderObject.get("kelp_last_price", None) == None:
                    mmmid_price = (best_ask + best_bid) / 2
                else:
                    mmmid_price = traderObject["kelp_last_price"]
            else:
                mmmid_price = (mm_ask + mm_bid) / 2

            return mmmid_price
        return None


    def ink_fair_value(self, order_depth: OrderDepth, traderObject) -> float:
        if len(order_depth.sell_orders) != 0 and len(order_depth.buy_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())
            filtered_ask = [
                price
                for price in order_depth.sell_orders.keys()
                if abs(order_depth.sell_orders[price])
                   >= self.params[Product.SQUID_INK]["adverse_volume"]
            ]
            filtered_bid = [
                price
                for price in order_depth.buy_orders.keys()
                if abs(order_depth.buy_orders[price])
                   >= self.params[Product.SQUID_INK]["adverse_volume"]
            ]
            mm_ask = min(filtered_ask) if len(filtered_ask) > 0 else None
            mm_bid = max(filtered_bid) if len(filtered_bid) > 0 else None
            if mm_ask == None or mm_bid == None:
                if traderObject.get("ink_last_price", None) == None:
                    mmmid_price = (best_ask + best_bid) / 2
                else:
                    mmmid_price = traderObject["ink_last_price"]
            else:
                mmmid_price = (mm_ask + mm_bid) / 2

            if traderObject.get("ink_last_price", None) != None:
                last_price = traderObject["ink_last_price"]
                last_returns = (mmmid_price - last_price) / last_price
                pred_returns = (
                        last_returns * self.params[Product.SQUID_INK]["reversion_beta"]
                )
                fair = mmmid_price + (mmmid_price * pred_returns)
            else:
                fair = mmmid_price

            # adjust with momentum?
            price_history: List[float] = self.trader_memory.get("ink_price_history", [])

            volitality_array: List[float] = self.trader_memory.get("volitality_arr", [])

            z_score_array: List[float] = self.trader_memory.get("z_score_arr", [])

            mean = 1960

            window = self.params[Product.SQUID_INK]["history_window"]
            if len(price_history) >= self.params[Product.SQUID_INK]["history_window"]:
                # momentum = (price_history[-1] - price_history[0])   #this gives like 3.77k somehow, ???

                x = np.arange(window)
                y = np.array(price_history[-window:])

                slope, intercept = np.polyfit(x, y, 1)

                # subtract by the line to have mean 0 over the past stuffs
                working_prices = price_history[-window:]
                working_prices = [working_prices[i] + i * slope for i in range(window)]
                tempmean = np.mean(working_prices)

                # working_prices = [i - tempmean for i in working_prices]

                curr_thing = mmmid_price - slope * window

                # x = np.arange(len(price_history))
                # y = np.array(price_history)
                # slope, intercept = np.polyfit(x, y, 1)
                # y_pred = slope * x + intercept

                # ss_res = np.sum((y - y_pred) ** 2)
                # ss_tot = np.sum((y - np.mean(y)) ** 2)
                # r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

                # momentum=0
                # if r_squared >= self.params[Product.SQUID_INK]["momentum_cutoff"]:
                #     momentum = slope

                # fair += self.params[Product.SQUID_INK]["momentum_weight"] * momentum

                #   #trying EMA
                beta = self.params[Product.SQUID_INK]["ema_param"]

                # exponential_weighted_avg = (1 - beta)/(1 - beta**len(price_history)) * (sum(beta**i * price_history[-i] for i in range(len(price_history))))
                exponential_weighted_avg = (1 - beta) * mmmid_price + (1 - beta) * sum(
                    beta ** (i + 1) * price_history[-i - 1] for i in
                    range(self.params[Product.SQUID_INK]["history_window"]))
                # short_ema = (1 - beta)
                fair = exponential_weighted_avg

                volitality_array.append(np.std(price_history[-1:-1 - window:-1]))
                z_score_array.append((curr_thing - np.mean(working_prices)) / np.std(working_prices))

            # update price history
            price_history.append(mmmid_price)

            self.trader_memory["ink_price_history"] = price_history
            self.trader_memory["volitality_arr"] = volitality_array
            self.trader_memory["z_score_arr"] = z_score_array

            # if timestamp == 958000:
            #     import plotly.graph_objects as go
            #     fig = go.Figure(
            #         data=go.Scatter(x=[2000 + i * 100 for i in range(0, 9981)], y=z_score_array, mode='lines+markers'))
            #     fig.write_image("plot.png")

            traderObject["ink_last_price"] = mmmid_price
            return mmmid_price
        return None

    def take_orders(
            self,
            product: str,
            order_depth: OrderDepth,
            fair_value: float,
            take_width: float,
            position: int,
            prevent_adverse: bool = False,
            adverse_volume: int = 0,
    ) -> (List[Order], int, int):
        orders: List[Order] = []
        buy_order_volume = 0
        sell_order_volume = 0

        buy_order_volume, sell_order_volume = self.take_best_orders(
            product,
            fair_value,
            take_width,
            orders,
            order_depth,
            position,
            buy_order_volume,
            sell_order_volume,
            prevent_adverse,
            adverse_volume,
        )
        return orders, buy_order_volume, sell_order_volume

    def clear_orders(
            self,
            product: str,
            order_depth: OrderDepth,
            fair_value: float,
            clear_width: int,
            position: int,
            buy_order_volume: int,
            sell_order_volume: int,
    ) -> (List[Order], int, int):
        orders: List[Order] = []
        buy_order_volume, sell_order_volume = self.clear_position_order(
            product,
            fair_value,
            clear_width,
            orders,
            order_depth,
            position,
            buy_order_volume,
            sell_order_volume,
        )
        return orders, buy_order_volume, sell_order_volume

    def make_orders(
            self,
            product,
            order_depth: OrderDepth,
            fair_value: float,
            position: int,
            buy_order_volume: int,
            sell_order_volume: int,
            disregard_edge: float,  # disregard trades within this edge for pennying or joining
            join_edge: float,  # join trades within this edge
            default_edge: float,  # default edge to request if there are no levels to penny or join
            manage_position: bool = False,
            soft_position_limit: int = 0,
            # will penny all other levels with higher edge
    ):

        # slope = 0
        # if product == Product.SQUID_INK:
        #     price_history: List[float] = self.trader_memory.get("ink_price_history", [])
        #     x = np.arange(len(price_history))
        #     y = np.array(price_history)
        #     slope, intercept = np.polyfit(x, y, 1)
        price_history: List[float] = self.trader_memory.get("ink_price_history", [])

        ink_stop_loss_percent = 0.005  # 0.5%
        ink_stop_loss = 10

        orders: List[Order] = []
        asks_above_fair = [
            price
            for price in order_depth.sell_orders.keys()
            if price > fair_value + disregard_edge
        ]
        bids_below_fair = [
            price
            for price in order_depth.buy_orders.keys()
            if price < fair_value - disregard_edge
        ]

        best_ask_above_fair = min(asks_above_fair) if len(asks_above_fair) > 0 else None
        best_bid_below_fair = max(bids_below_fair) if len(bids_below_fair) > 0 else None

        ask = round(fair_value + default_edge)
        if best_ask_above_fair != None:
            if abs(best_ask_above_fair - fair_value) <= join_edge:
                ask = best_ask_above_fair  # join
            else:
                ask = best_ask_above_fair - 1  # penny

        bid = round(fair_value - default_edge)
        if best_bid_below_fair != None:
            if abs(fair_value - best_bid_below_fair) <= join_edge:
                bid = best_bid_below_fair
            else:
                bid = best_bid_below_fair + 1

        if manage_position:
            if position > soft_position_limit:
                ask -= 1
            elif position < -1 * soft_position_limit:
                bid += 1

        if -10 < position < 10 and product == Product.SQUID_INK:
            self.params[Product.SQUID_INK]["entry_price"] = fair_value

        z_score_array = self.trader_memory["z_score_arr"]

        if product == Product.SQUID_INK and len(price_history) >= self.params[Product.SQUID_INK]["history_window"]:
            beta = self.params[Product.SQUID_INK]["ema_param"]

            long_ema = (1 - beta) / (1 - beta ** (self.params[Product.SQUID_INK]["history_window"] + 1)) * (
                        fair_value + sum(beta ** (i + 1) * price_history[-i - 1] for i in
                                         range(self.params[Product.SQUID_INK]["history_window"])))
            short_ema = (1 - beta) / (1 - beta ** (self.params[Product.SQUID_INK]["small_window"] + 1)) * (
                        fair_value + sum(beta ** (i + 1) * price_history[-i - 1] for i in
                                         range(self.params[Product.SQUID_INK]["small_window"])))

            if short_ema > long_ema:
                logger.logs += f"going up!\b"
            else:
                logger.logs += f"going down!\n"
            logger.logs += f"position: {position}\n"
            logger.logs += f"short_ema = {short_ema}\n long_ema = {long_ema}\n"

            # want difference of at least xxx to be significant, buy to 50 if significant, else just buy to 0
            # if short_ema > long_ema + 0.002 and position < 0: #change to 0 if its autistic
            if z_score_array and z_score_array[-1] < -4:
                # its goin up!
                # go on a long position, buy buy buy
                bid = round(fair_value)
                if best_ask_above_fair != None: ask = best_ask_above_fair + 3  # just put this super high in case anyone wants to buy for this ig
                logger.logs += f"short_ema > long_ema\nPosition: {position}\nBid:{bid}\nAsk:{ask}"

                bid = fair_value
            # elif short_ema < long_ema - 0.002 and position>0:
            elif z_score_array and z_score_array[-1] > 4:
                # its goin down!
                # go on a short position, sell sell sell
                if best_bid_below_fair != None: bid = best_bid_below_fair - 3  # just put this super high in case someone wants to sell for this lmao
                ask = round(fair_value)
                logger.logs += f"short_ema < long_ema\nPosition: {position}\nBid:{bid}\nAsk:{ask}"

            if position < 0 and fair_value < self.params[Product.SQUID_INK]["entry_price"] - ink_stop_loss and \
                    self.params[Product.SQUID_INK]["entry_price"] != -1:
                # close the long position, too risky
                # want to buy back to 0
                if best_ask_above_fair != None: bid = round(fair_value)  # just get back to 0
                if best_ask_above_fair != None: ask = best_ask_above_fair + 3
            if position > 0 and fair_value > self.params[Product.SQUID_INK]["entry_price"] + ink_stop_loss and \
                    self.params[Product.SQUID_INK]["entry_price"] != -1:
                # close the short position, too risky
                if best_bid_below_fair != None: bid = best_bid_below_fair - 3
                if best_bid_below_fair != None: ask = round(fair_value)  # just get back to 0

        buy_order_volume, sell_order_volume = self.market_make(
            product,
            orders,
            bid,
            ask,
            position,
            buy_order_volume,
            sell_order_volume,
        )

        return orders, buy_order_volume, sell_order_volume

    
    def magnificent_macarons_strategy(self, state: TradingState, traderObject, conversions):
        if Product.MAGNIFICENT_MACARONS in self.params and Product.MAGNIFICENT_MACARONS in state.order_depths:

            magnificent_macarons_position = (
                state.position[Product.MAGNIFICENT_MACARONS]
                if Product.MAGNIFICENT_MACARONS in state.position
                else 0
            )

            # if state.timestamp == 10000:
            #     raise Exception(state.observations)
            # raise Exception(state.observations.conversionObservations)
            # if state.observations.conversionObservations:
            #     raise Exception(state.observations)
            obs = state.observations.conversionObservations.get("MAGNIFICENT_MACARONS", None)

            if obs is None:
                return []
            
            buy_price = obs.askPrice + obs.transportFees + obs.importTariff


            sell_price =  (obs.askPrice + obs.bidPrice)/2 + 2 + obs.transportFees + obs.exportTariff


            #spam short, sell at buyprice + 1
            conversions -= magnificent_macarons_position

            orders = []

            buy_quantity = self.LIMIT["MAGNIFICENT_MACARONS"] - (magnificent_macarons_position)
            if buy_quantity > 0:
                orders.append(Order("MAGNIFICENT_MACARONS", round(buy_price), buy_quantity))  # Buy order

            sell_quantity = self.LIMIT["MAGNIFICENT_MACARONS"] + (magnificent_macarons_position)

            if sell_quantity > 0:
                orders.append(Order("MAGNIFICENT_MACARONS", round(sell_price), -sell_quantity))  # Sell order
            
            return orders
        return []

    def croissants_strategy(self, state: TradingState, traderObject):
        if Product.CROISSANTS in self.params and Product.CROISSANTS in state.order_depths:
            croissants_position = (
                state.position[Product.CROISSANTS]
                if Product.CROISSANTS in state.position
                else 0
            )
            croissants_fair_value = self.croissants_fair_value(
                state.order_depths[Product.CROISSANTS], traderObject
            )
            croissants_take_orders, buy_order_volume, sell_order_volume = (
                self.take_orders(
                    Product.CROISSANTS,
                    state.order_depths[Product.CROISSANTS],
                    croissants_fair_value,
                    self.params[Product.CROISSANTS]["take_width"],
                    croissants_position,
                    self.params[Product.CROISSANTS]["prevent_adverse"],
                    self.params[Product.CROISSANTS]["adverse_volume"],
                )
            )
            croissants_clear_orders, buy_order_volume, sell_order_volume = (
                self.clear_orders(
                    Product.CROISSANTS,
                    state.order_depths[Product.CROISSANTS],
                    croissants_fair_value,
                    self.params[Product.CROISSANTS]["clear_width"],
                    croissants_position,
                    buy_order_volume,
                    sell_order_volume,
                )
            )
            croissants_make_orders, _, _ = self.make_orders(
                Product.CROISSANTS,
                state.order_depths[Product.CROISSANTS],
                croissants_fair_value,
                croissants_position,
                buy_order_volume,
                sell_order_volume,
                self.params[Product.CROISSANTS]["disregard_edge"],
                self.params[Product.CROISSANTS]["join_edge"],
                self.params[Product.CROISSANTS]["default_edge"],
            )
            return croissants_take_orders + croissants_clear_orders + croissants_make_orders
        return []


    def jams_strategy(self, state: TradingState, traderObject):
        if Product.JAMS in self.params and Product.JAMS in state.order_depths:
            jams_position = (
                state.position[Product.JAMS]
                if Product.JAMS in state.position
                else 0
            )
            jams_fair_value = self.jams_fair_value(
                state.order_depths[Product.JAMS], traderObject
            )
            jams_take_orders, buy_order_volume, sell_order_volume = (
                self.take_orders(
                    Product.JAMS,
                    state.order_depths[Product.JAMS],
                    jams_fair_value,
                    self.params[Product.JAMS]["take_width"],
                    jams_position,
                    self.params[Product.JAMS]["prevent_adverse"],
                    self.params[Product.JAMS]["adverse_volume"],
                )
            )
            jams_clear_orders, buy_order_volume, sell_order_volume = (
                self.clear_orders(
                    Product.JAMS,
                    state.order_depths[Product.JAMS],
                    jams_fair_value,
                    self.params[Product.JAMS]["clear_width"],
                    jams_position,
                    buy_order_volume,
                    sell_order_volume,
                )
            )
            jams_make_orders, _, _ = self.make_orders(
                Product.JAMS,
                state.order_depths[Product.JAMS],
                jams_fair_value,
                jams_position,
                buy_order_volume,
                sell_order_volume,
                self.params[Product.JAMS]["disregard_edge"],
                self.params[Product.JAMS]["join_edge"],
                self.params[Product.JAMS]["default_edge"],
            )
            return jams_take_orders + jams_clear_orders + jams_make_orders
        return []
    

    def djembes_strategy(self, state: TradingState, traderObject):
        if Product.DJEMBES in self.params and Product.DJEMBES in state.order_depths:
            djembes_position = (
                state.position[Product.DJEMBES]
                if Product.DJEMBES in state.position
                else 0
            )
            djembes_fair_value = self.djembes_fair_value(
                state.order_depths[Product.DJEMBES], traderObject
            )
            djembes_take_orders, buy_order_volume, sell_order_volume = (
                self.take_orders(
                    Product.DJEMBES,
                    state.order_depths[Product.DJEMBES],
                    djembes_fair_value,
                    self.params[Product.DJEMBES]["take_width"],
                    djembes_position,
                    self.params[Product.DJEMBES]["prevent_adverse"],
                    self.params[Product.DJEMBES]["adverse_volume"],
                )
            )
            djembes_clear_orders, buy_order_volume, sell_order_volume = (
                self.clear_orders(
                    Product.DJEMBES,
                    state.order_depths[Product.DJEMBES],
                    djembes_fair_value,
                    self.params[Product.DJEMBES]["clear_width"],
                    djembes_position,
                    buy_order_volume,
                    sell_order_volume,
                )
            )
            djembes_make_orders, _, _ = self.make_orders(
                Product.DJEMBES,
                state.order_depths[Product.DJEMBES],
                djembes_fair_value,
                djembes_position,
                buy_order_volume,
                sell_order_volume,
                self.params[Product.DJEMBES]["disregard_edge"],
                self.params[Product.DJEMBES]["join_edge"],
                self.params[Product.DJEMBES]["default_edge"],
            )
            return djembes_take_orders + djembes_clear_orders + djembes_make_orders
        return []
    
    def croissants_fair_value(self, order_depth: OrderDepth, traderObject) -> float:
        if len(order_depth.sell_orders) != 0 and len(order_depth.buy_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())
            filtered_ask = [
                price
                for price in order_depth.sell_orders.keys()
                if abs(order_depth.sell_orders[price])
                   >= self.params[Product.CROISSANTS]["adverse_volume"]
            ]
            filtered_bid = [
                price
                for price in order_depth.buy_orders.keys()
                if abs(order_depth.buy_orders[price])
                   >= self.params[Product.CROISSANTS]["adverse_volume"]
            ]
            mm_ask = min(filtered_ask) if len(filtered_ask) > 0 else None
            mm_bid = max(filtered_bid) if len(filtered_bid) > 0 else None
            if mm_ask == None or mm_bid == None:
                if traderObject.get("croissants_last_price", None) == None:
                    mmmid_price = (best_ask + best_bid) / 2
                else:
                    mmmid_price = traderObject["croissants_last_price"]
            else:
                mmmid_price = (mm_ask + mm_bid) / 2

            # estimate with reversion
            if traderObject.get("croissants_last_price", None) != None:
                last_price = traderObject["croissants_last_price"]
                last_returns = (mmmid_price - last_price) / last_price
                pred_returns = (
                        last_returns * self.params[Product.CROISSANTS]["reversion_beta"]
                )
                fair = mmmid_price + (mmmid_price * pred_returns)
            else:
                fair = mmmid_price

            price_history: List[float] = self.trader_memory.get("croissants_price_history", [])
            # adjust with momentum?
            if len(price_history) >= 2:
                # momentum = (price_history[-1] - price_history[0])

                x = np.arange(
                    len(price_history))  # for linreg, its literally better to have a negative momentum weight?
                y = np.array(price_history)
                slope, intercept = np.polyfit(x, y, 1)
                y_pred = slope * x + intercept

                ss_res = np.sum((y - y_pred) ** 2)
                ss_tot = np.sum((y - np.mean(y)) ** 2)
                r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

                momentum = 0
                if r_squared >= self.params[Product.CROISSANTS]["momentum_cutoff"]:
                    momentum = slope

                fair += self.params[Product.CROISSANTS]["momentum_weight"] * momentum

            traderObject["croissants_last_price"] = fair

            # update price_history
            price_history.append(fair)

            if len(price_history) > self.params[Product.CROISSANTS]["history_window"]:
                price_history.pop(0)

            self.trader_memory["croissants_price_history"] = price_history

            return mmmid_price
        return None

    
    def jams_fair_value(self, order_depth: OrderDepth, traderObject) -> float:
        if len(order_depth.sell_orders) != 0 and len(order_depth.buy_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())
            filtered_ask = [
                price
                for price in order_depth.sell_orders.keys()
                if abs(order_depth.sell_orders[price])
                   >= self.params[Product.JAMS]["adverse_volume"]
            ]
            filtered_bid = [
                price
                for price in order_depth.buy_orders.keys()
                if abs(order_depth.buy_orders[price])
                   >= self.params[Product.JAMS]["adverse_volume"]
            ]
            mm_ask = min(filtered_ask) if len(filtered_ask) > 0 else None
            mm_bid = max(filtered_bid) if len(filtered_bid) > 0 else None
            if mm_ask == None or mm_bid == None:
                if traderObject.get("jams_last_price", None) == None:
                    mmmid_price = (best_ask + best_bid) / 2
                else:
                    mmmid_price = traderObject["jams_last_price"]
            else:
                mmmid_price = (mm_ask + mm_bid) / 2

            # estimate with reversion
            if traderObject.get("jams_last_price", None) != None:
                last_price = traderObject["jams_last_price"]
                last_returns = (mmmid_price - last_price) / last_price
                pred_returns = (
                        last_returns * self.params[Product.KELP]["reversion_beta"]
                )
                fair = mmmid_price + (mmmid_price * pred_returns)
            else:
                fair = mmmid_price

            price_history: List[float] = self.trader_memory.get("jams_price_history", [])
            # adjust with momentum?
            if len(price_history) >= 2:
                # momentum = (price_history[-1] - price_history[0])

                x = np.arange(
                    len(price_history))  # for linreg, its literally better to have a negative momentum weight?
                y = np.array(price_history)
                slope, intercept = np.polyfit(x, y, 1)
                y_pred = slope * x + intercept

                ss_res = np.sum((y - y_pred) ** 2)
                ss_tot = np.sum((y - np.mean(y)) ** 2)
                r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

                momentum = 0
                if r_squared >= self.params[Product.JAMS]["momentum_cutoff"]:
                    momentum = slope

                fair += self.params[Product.JAMS]["momentum_weight"] * momentum

            traderObject["jams_last_price"] = fair

            # update price_history
            price_history.append(fair)

            if len(price_history) > self.params[Product.JAMS]["history_window"]:
                price_history.pop(0)

            self.trader_memory["jams_price_history"] = price_history

            return mmmid_price
        return None

    def djembes_fair_value(self, order_depth: OrderDepth, traderObject) -> float:
        if len(order_depth.sell_orders) != 0 and len(order_depth.buy_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())
            filtered_ask = [
                price
                for price in order_depth.sell_orders.keys()
                if abs(order_depth.sell_orders[price])
                   >= self.params[Product.DJEMBES]["adverse_volume"]
            ]
            filtered_bid = [
                price
                for price in order_depth.buy_orders.keys()
                if abs(order_depth.buy_orders[price])
                   >= self.params[Product.DJEMBES]["adverse_volume"]
            ]
            mm_ask = min(filtered_ask) if len(filtered_ask) > 0 else None
            mm_bid = max(filtered_bid) if len(filtered_bid) > 0 else None
            if mm_ask == None or mm_bid == None:
                if traderObject.get("djembes_last_price", None) == None:
                    mmmid_price = (best_ask + best_bid) / 2
                else:
                    mmmid_price = traderObject["djembes_last_price"]
            else:
                mmmid_price = (mm_ask + mm_bid) / 2

            # # estimate with reversion
            # if traderObject.get("djembes_last_price", None) != None:
            #     last_price = traderObject["djembes_last_price"]
            #     last_returns = (mmmid_price - last_price) / last_price
            #     pred_returns = (
            #             last_returns * self.params[Product.KELP]["reversion_beta"]
            #     )
            #     fair = mmmid_price + (mmmid_price * pred_returns)
            # else:
            #     fair = mmmid_price

            # price_history: List[float] = self.trader_memory.get("djembes_price_history", [])
            # # adjust with momentum?
            # if len(price_history) >= 2:
            #     # momentum = (price_history[-1] - price_history[0])

            #     x = np.arange(
            #         len(price_history))  # for linreg, its literally better to have a negative momentum weight?
            #     y = np.array(price_history)
            #     slope, intercept = np.polyfit(x, y, 1)
            #     y_pred = slope * x + intercept

            #     ss_res = np.sum((y - y_pred) ** 2)
            #     ss_tot = np.sum((y - np.mean(y)) ** 2)
            #     r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

            #     momentum = 0
            #     if r_squared >= self.params[Product.DJEMBES]["momentum_cutoff"]:
            #         momentum = slope

            #     fair += self.params[Product.DJEMBES]["momentum_weight"] * momentum

            # traderObject["djembes_last_price"] = fair

            # # update price_history
            # price_history.append(fair)

            # if len(price_history) > self.params[Product.DJEMBES]["history_window"]:
            #     price_history.pop(0)

            # self.trader_memory["djembes_price_history"] = price_history

            return mmmid_price
        return None



    def resin_strategy(self, state: TradingState, traderObject):
        if Product.RAINFOREST_RESIN in self.params and Product.RAINFOREST_RESIN in state.order_depths:
            resin_position = (
                state.position[Product.RAINFOREST_RESIN]
                if Product.RAINFOREST_RESIN in state.position
                else 0
            )
            resin_take_orders, buy_order_volume, sell_order_volume = (
                self.take_orders(
                    Product.RAINFOREST_RESIN,
                    state.order_depths[Product.RAINFOREST_RESIN],
                    self.params[Product.RAINFOREST_RESIN]["fair_value"],
                    self.params[Product.RAINFOREST_RESIN]["take_width"],
                    resin_position,
                )
            )
            resin_clear_orders, buy_order_volume, sell_order_volume = (
                self.clear_orders(
                    Product.RAINFOREST_RESIN,
                    state.order_depths[Product.RAINFOREST_RESIN],
                    self.params[Product.RAINFOREST_RESIN]["fair_value"],
                    self.params[Product.RAINFOREST_RESIN]["clear_width"],
                    resin_position,
                    buy_order_volume,
                    sell_order_volume,
                )
            )
            resin_make_orders, _, _ = self.make_orders(
                Product.RAINFOREST_RESIN,
                state.order_depths[Product.RAINFOREST_RESIN],
                self.params[Product.RAINFOREST_RESIN]["fair_value"],
                resin_position,
                buy_order_volume,
                sell_order_volume,
                self.params[Product.RAINFOREST_RESIN]["disregard_edge"],
                self.params[Product.RAINFOREST_RESIN]["join_edge"],
                self.params[Product.RAINFOREST_RESIN]["default_edge"],
                True,
                self.params[Product.RAINFOREST_RESIN]["soft_position_limit"],
            )
            return resin_take_orders + resin_clear_orders + resin_make_orders
        return []

    def kelp_strategy(self, state: TradingState, traderObject):
        if Product.KELP in self.params and Product.KELP in state.order_depths:
            kelp_position = (
                state.position[Product.KELP]
                if Product.KELP in state.position
                else 0
            )
            kelp_fair_value = self.kelp_fair_value(
                state.order_depths[Product.KELP], traderObject
            )
            kelp_take_orders, buy_order_volume, sell_order_volume = (
                self.take_orders(
                    Product.KELP,
                    state.order_depths[Product.KELP],
                    kelp_fair_value,
                    self.params[Product.KELP]["take_width"],
                    kelp_position,
                    self.params[Product.KELP]["prevent_adverse"],
                    self.params[Product.KELP]["adverse_volume"],
                )
            )
            kelp_clear_orders, buy_order_volume, sell_order_volume = (
                self.clear_orders(
                    Product.KELP,
                    state.order_depths[Product.KELP],
                    kelp_fair_value,
                    self.params[Product.KELP]["clear_width"],
                    kelp_position,
                    buy_order_volume,
                    sell_order_volume,
                )
            )
            kelp_make_orders, _, _ = self.make_orders(
                Product.KELP,
                state.order_depths[Product.KELP],
                kelp_fair_value,
                kelp_position,
                buy_order_volume,
                sell_order_volume,
                self.params[Product.KELP]["disregard_edge"],
                self.params[Product.KELP]["join_edge"],
                self.params[Product.KELP]["default_edge"],
            )
            return kelp_take_orders + kelp_clear_orders + kelp_make_orders
        return []
    def ink_strategy(self, state: TradingState, traderObject):
        if Product.SQUID_INK in self.params and Product.SQUID_INK in state.order_depths:
            ink_position = (
                state.position[Product.SQUID_INK]
                if Product.SQUID_INK in state.position
                else 0
            )
            ink_fair_value = self.ink_fair_value(
                state.order_depths[Product.SQUID_INK], traderObject
            )
            ink_take_orders, buy_order_volume, sell_order_volume = (
                self.take_orders(
                    Product.SQUID_INK,
                    state.order_depths[Product.SQUID_INK],
                    ink_fair_value,
                    self.params[Product.SQUID_INK]["take_width"],
                    ink_position,
                    self.params[Product.SQUID_INK]["prevent_adverse"],
                    self.params[Product.SQUID_INK]["adverse_volume"],
                )
            )
            ink_clear_orders, buy_order_volume, sell_order_volume = (
                self.clear_orders(
                    Product.SQUID_INK,
                    state.order_depths[Product.SQUID_INK],
                    ink_fair_value,
                    self.params[Product.SQUID_INK]["clear_width"],
                    ink_position,
                    buy_order_volume,
                    sell_order_volume,
                )
            )
            ink_make_orders, _, _ = self.make_orders(
                Product.SQUID_INK,
                state.order_depths[Product.SQUID_INK],
                ink_fair_value,
                ink_position,
                buy_order_volume,
                sell_order_volume,
                self.params[Product.SQUID_INK]["disregard_edge"],
                self.params[Product.SQUID_INK]["join_edge"],
                self.params[Product.SQUID_INK]["default_edge"],
            )
            return (
                    ink_take_orders + ink_clear_orders + ink_make_orders
            )
        return []
    def volcanic_rock_strategy(self, state: TradingState, traderObject):
        if Product.VOLCANIC_ROCK in self.params and Product.VOLCANIC_ROCK in state.order_depths:
            volcanic_rock_position = (
                state.position[Product.VOLCANIC_ROCK]
                if Product.VOLCANIC_ROCK in state.position
                else 0
            )
            volcanic_rock_fair_value = self.volcanic_rock_fair_value(
                state.order_depths[Product.VOLCANIC_ROCK], traderObject
            )
            volcanic_rock_take_orders, buy_order_volume, sell_order_volume = (
                self.take_orders(
                    Product.VOLCANIC_ROCK,
                    state.order_depths[Product.VOLCANIC_ROCK],
                    volcanic_rock_fair_value,
                    self.params[Product.VOLCANIC_ROCK]["take_width"],
                    volcanic_rock_position,
                    self.params[Product.VOLCANIC_ROCK]["prevent_adverse"],
                    self.params[Product.VOLCANIC_ROCK]["adverse_volume"],
                )
            )
            volcanic_rock_clear_orders, buy_order_volume, sell_order_volume = (
                self.clear_orders(
                    Product.VOLCANIC_ROCK,
                    state.order_depths[Product.VOLCANIC_ROCK],
                    volcanic_rock_fair_value,
                    self.params[Product.VOLCANIC_ROCK]["clear_width"],
                    volcanic_rock_position,
                    buy_order_volume,
                    sell_order_volume,
                )
            )
            volcanic_rock_make_orders, _, _ = self.make_orders(
                Product.VOLCANIC_ROCK,
                state.order_depths[Product.VOLCANIC_ROCK],
                volcanic_rock_fair_value,
                volcanic_rock_position,
                buy_order_volume,
                sell_order_volume,
                self.params[Product.VOLCANIC_ROCK]["disregard_edge"],
                self.params[Product.VOLCANIC_ROCK]["join_edge"],
                self.params[Product.VOLCANIC_ROCK]["default_edge"],
            )
            return volcanic_rock_take_orders + volcanic_rock_clear_orders + volcanic_rock_make_orders
        return []
    
    def get_swmid(self, order_depth) -> float:
        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        best_bid_vol = abs(order_depth.buy_orders[best_bid])
        best_ask_vol = abs(order_depth.sell_orders[best_ask])
        return (best_bid * best_ask_vol + best_ask * best_bid_vol) / (
                best_bid_vol + best_ask_vol
        )

    def get_synthetic_basket_order_depth(
            self, order_depths: Dict[str, OrderDepth], basket_num
    ) -> OrderDepth:
        # Constants
        CROISSANTS_PER_BASKET = WEIGHTS[basket_num][Product.CROISSANTS]
        JAMS_PER_BASKET = WEIGHTS[basket_num][Product.JAMS]
        DJEMBES_PER_BASKET = WEIGHTS[basket_num][Product.DJEMBES]

        # Initialize the synthetic basket order depth
        synthetic_order_price = OrderDepth()

        # Calculate the best bid and ask for each component
        croissants_best_bid = (
            max(order_depths[Product.CROISSANTS].buy_orders.keys())
            if order_depths[Product.CROISSANTS].buy_orders
            else 0
        )
        croissants_best_ask = (
            min(order_depths[Product.CROISSANTS].sell_orders.keys())
            if order_depths[Product.CROISSANTS].sell_orders
            else float("inf")
        )
        jams_best_bid = (
            max(order_depths[Product.JAMS].buy_orders.keys())
            if order_depths[Product.JAMS].buy_orders
            else 0
        )
        jams_best_ask = (
            min(order_depths[Product.JAMS].sell_orders.keys())
            if order_depths[Product.JAMS].sell_orders
            else float("inf")
        )

        djembes_best_bid = (
            max(order_depths[Product.DJEMBES].buy_orders.keys())
            if order_depths[Product.DJEMBES].buy_orders
            else 0
        )
        djembes_best_ask = (
            min(order_depths[Product.DJEMBES].sell_orders.keys())
            if order_depths[Product.DJEMBES].sell_orders
            else float("inf")
        )

        # Calculate the implied bid and ask for the synthetic basket
        implied_bid = (
                croissants_best_bid * CROISSANTS_PER_BASKET
                + jams_best_bid * JAMS_PER_BASKET
                + djembes_best_bid * DJEMBES_PER_BASKET
        )
        implied_ask = (
                croissants_best_ask * CROISSANTS_PER_BASKET
                + jams_best_ask * JAMS_PER_BASKET
                + djembes_best_ask * DJEMBES_PER_BASKET
        )

        # Calculate the maximum number of synthetic baskets available at the implied bid and ask
        if implied_bid > 0:
            croissants_bid_volume = (
                    order_depths[Product.CROISSANTS].buy_orders[croissants_best_bid]
                    // CROISSANTS_PER_BASKET
            )
            jams_bid_volume = (
                    order_depths[Product.JAMS].buy_orders[jams_best_bid]
                    // JAMS_PER_BASKET
            )
            djembes_bid_volume = (
                    order_depths[Product.DJEMBES].buy_orders[djembes_best_bid]
                    // DJEMBES_PER_BASKET
            ) if basket_num == 1 else 1e9
            implied_bid_volume = min(
                croissants_bid_volume, jams_bid_volume, djembes_bid_volume
            )
            synthetic_order_price.buy_orders[implied_bid] = implied_bid_volume

        if implied_ask < float("inf"):
            croissants_ask_volume = (
                    -order_depths[Product.CROISSANTS].sell_orders[croissants_best_ask]
                    // CROISSANTS_PER_BASKET
            )
            jams_ask_volume = (
                    -order_depths[Product.JAMS].sell_orders[jams_best_ask]
                    // JAMS_PER_BASKET
            )
            djembes_ask_volume = (
                    -order_depths[Product.DJEMBES].sell_orders[djembes_best_ask]
                    // DJEMBES_PER_BASKET
            ) if basket_num == 1 else 1e9
            implied_ask_volume = min(
                croissants_ask_volume, jams_ask_volume, djembes_ask_volume
            )
            synthetic_order_price.sell_orders[implied_ask] = -implied_ask_volume

        return synthetic_order_price

    def convert_synthetic_basket_orders(
            self, synthetic_orders: List[Order], order_depths: Dict[str, OrderDepth], basket_num
    ) -> Dict[str, List[Order]]:
        # Initialize the dictionary to store component orders
        component_orders = {
            Product.CROISSANTS: [],
            Product.JAMS: [],
            Product.DJEMBES: [],
        }

        # Get the best bid and ask for the synthetic basket
        synthetic_basket_order_depth = self.get_synthetic_basket_order_depth(
            order_depths, basket_num
        )
        best_bid = (
            max(synthetic_basket_order_depth.buy_orders.keys())
            if synthetic_basket_order_depth.buy_orders
            else 0
        )
        best_ask = (
            min(synthetic_basket_order_depth.sell_orders.keys())
            if synthetic_basket_order_depth.sell_orders
            else float("inf")
        )

        # Iterate through each synthetic basket order
        for order in synthetic_orders:
            # Extract the price and quantity from the synthetic basket order
            price = order.price
            quantity = order.quantity

            # Check if the synthetic basket order aligns with the best bid or ask
            if quantity > 0 and price >= best_ask:
                # Buy order - trade components at their best ask prices
                croissant_price = min(
                    order_depths[Product.CROISSANTS].sell_orders.keys()
                )
                jams_price = min(
                    order_depths[Product.JAMS].sell_orders.keys()
                )
                djembes_price = min(order_depths[Product.DJEMBES].sell_orders.keys())
            elif quantity < 0 and price <= best_bid:
                # Sell order - trade components at their best bid prices
                croissant_price = max(order_depths[Product.CROISSANTS].buy_orders.keys())
                jams_price = max(
                    order_depths[Product.JAMS].buy_orders.keys()
                )
                djembes_price = max(order_depths[Product.DJEMBES].buy_orders.keys())
            else:
                # The synthetic basket order does not align with the best bid or ask
                continue

            # Create orders for each component
            croissant_order = Order(
                Product.CROISSANTS,
                croissant_price,
                quantity * WEIGHTS[basket_num][Product.CROISSANTS],
            )
            jams_order = Order(
                Product.JAMS,
                jams_price,
                quantity * WEIGHTS[basket_num][Product.JAMS],
            )
            djembes_order = Order(
                Product.DJEMBES, djembes_price, quantity * WEIGHTS[basket_num][Product.DJEMBES]
            )

            # Add the component orders to the respective lists
            component_orders[Product.CROISSANTS].append(croissant_order)
            component_orders[Product.JAMS].append(jams_order)
            component_orders[Product.DJEMBES].append(djembes_order)

        return component_orders

    def execute_spread_orders(
            self,
            target_position: int,
            basket_position: int,
            order_depths: Dict[str, OrderDepth],
            basket_num
    ):

        if target_position == basket_position:
            return {}

        target_quantity = abs(target_position - basket_position)
        basket_order_depth = order_depths[BASKETS[basket_num]]

        synthetic_order_depth = self.get_synthetic_basket_order_depth(order_depths, basket_num)

        if target_position > basket_position:
            basket_ask_price = min(basket_order_depth.sell_orders.keys())
            basket_ask_volume = abs(basket_order_depth.sell_orders[basket_ask_price])

            synthetic_bid_price = max(synthetic_order_depth.buy_orders.keys())
            synthetic_bid_volume = abs(
                synthetic_order_depth.buy_orders[synthetic_bid_price]
            )

            orderbook_volume = min(basket_ask_volume, synthetic_bid_volume)
            execute_volume = min(orderbook_volume, target_quantity)

            basket_orders = [
                Order(BASKETS[basket_num], basket_ask_price, execute_volume)
            ]
            synthetic_orders = [
                Order(SYNTHETICS[basket_num], synthetic_bid_price, -execute_volume)
            ]

            aggregate_orders = self.convert_synthetic_basket_orders(
                synthetic_orders, order_depths, basket_num
            )

            aggregate_orders[BASKETS[basket_num]] = basket_orders

            return aggregate_orders

        else:
            basket_bid_price = max(basket_order_depth.buy_orders.keys())
            basket_bid_volume = abs(basket_order_depth.buy_orders[basket_bid_price])

            synthetic_ask_price = min(synthetic_order_depth.sell_orders.keys())
            synthetic_ask_volume = abs(
                synthetic_order_depth.sell_orders[synthetic_ask_price]
            )

            orderbook_volume = min(basket_bid_volume, synthetic_ask_volume)
            execute_volume = min(orderbook_volume, target_quantity)

            basket_orders = [
                Order(BASKETS[basket_num], basket_bid_price, -execute_volume)
            ]
            synthetic_orders = [
                Order(SYNTHETICS[basket_num], synthetic_ask_price, execute_volume)
            ]

            aggregate_orders = self.convert_synthetic_basket_orders(
                synthetic_orders, order_depths, basket_num
            )

            aggregate_orders[BASKETS[basket_num]] = basket_orders

            return aggregate_orders


    def spread_orders(
            self,
            state: TradingState,
            order_depths: Dict[str, OrderDepth],
            basket_position: int,
            spread_data: Dict[str, Any],
            basket_num
    ):
        if BASKETS[basket_num] not in order_depths.keys():
            return {}

        basket_order_depth = order_depths[BASKETS[basket_num]]
        synthetic_order_depth = self.get_synthetic_basket_order_depth(order_depths, basket_num)
        basket_swmid = self.get_swmid(basket_order_depth)
        synthetic_swmid = self.get_swmid(synthetic_order_depth)
        spread = basket_swmid - synthetic_swmid
        spread_data["spread_history"].append(spread)

        if (
                len(spread_data["spread_history"])
                < self.params[SPREADS[basket_num]]["spread_std_window"]
        ):
            if state.timestamp != 0:
                return {}
            if basket_num == 1:
                # Go Short
                return self.execute_spread_orders(
                    -self.params[SPREADS[basket_num]]["target_position"],
                    basket_position,
                    order_depths,
                    basket_num
                )
            else:
               # Go long
               return self.execute_spread_orders(
                   self.params[SPREADS[basket_num]]["target_position"],
                   basket_position,
                   order_depths,
                   basket_num
               )


        elif len(spread_data["spread_history"]) > self.params[SPREADS[basket_num]]["spread_std_window"]:
            spread_data["spread_history"].pop(0)

        spread_std = np.std(spread_data["spread_history"])


        if basket_num == 2:
            moving_avg = np.mean(spread_data["spread_history"][-20:])
            diffs = [abs(moving_avg - self.params[SPREADS[basket_num]]["default_spread_mean_upper"]),
                     abs(moving_avg - self.params[SPREADS[basket_num]]["default_spread_mean"]),
                     abs(moving_avg - self.params[SPREADS[basket_num]]["default_spread_mean_lower"])]

            if min(diffs) > 20:
                return {}

            if diffs[0] > max(diffs[1], diffs[2]):
                mean = self.params[SPREADS[basket_num]]["default_spread_mean_upper"]
            elif diffs[1] > max(diffs[0], diffs[2]):
                mean = self.params[SPREADS[basket_num]]["default_spread_mean"]
            else:
                mean = self.params[SPREADS[basket_num]]["default_spread_mean_lower"]

            mean = self.params[SPREADS[basket_num]]["default_spread_mean"]

        else:
            mean = self.params[SPREADS[basket_num]]["default_spread_mean"]



        zscore = (
                         spread - mean
                 ) / spread_std

        if zscore >= self.params[SPREADS[basket_num]]["zscore_threshold"]:
            if basket_position != -self.params[SPREADS[basket_num]]["target_position"]:
                return self.execute_spread_orders(
                    -self.params[SPREADS[basket_num]]["target_position"],
                    basket_position,
                    order_depths,
                    basket_num
                )

        if zscore <= -self.params[SPREADS[basket_num]]["zscore_threshold"]:
            if basket_position != self.params[SPREADS[basket_num]]["target_position"]:
                return self.execute_spread_orders(
                    self.params[SPREADS[basket_num]]["target_position"],
                    basket_position,
                    order_depths,
                    basket_num
                )

        spread_data["prev_zscore"] = zscore
        return {}

    def spread_strategy(self, state: TradingState, traderObject):
        basket_num = 1
        if SPREADS[basket_num] not in traderObject:
            traderObject[SPREADS[basket_num]] = {
                "spread_history": [],
                "prev_zscore": 0,
                "clear_flag": False,
                "curr_avg": 0,
            }

        basket_position = (
            state.position[BASKETS[basket_num]]
            if BASKETS[basket_num] in state.position
            else 0
        )
        spread_orders = self.spread_orders(
            state,
            state.order_depths,
            basket_position,
            traderObject[SPREADS[basket_num]],
            basket_num
        )

        # if not spread_orders:
            # locally trade like squid ink around here


        return spread_orders

    def spread2_strategy(self, state: TradingState, traderObject):
        basket_num = 2

        if SPREADS[basket_num] not in traderObject:
            traderObject[SPREADS[basket_num]] = {
                "spread_history": [],
                "prev_zscore": 0,
                "clear_flag": False,
                "curr_avg": 0,
            }

        basket_position = (
            state.position[BASKETS[basket_num]]
            if BASKETS[basket_num] in state.position
            else 0
        )
        spread_orders = self.spread_orders(
            state,
            state.order_depths,
            basket_position,
            traderObject[SPREADS[basket_num]],
            basket_num
        )

        # if not spread_orders:
        #     locally trade like squid ink around here

        return spread_orders

    def get_volcanic_rock_coupon_mid_price(
        self, volcanic_rock_coupon_order_depth: OrderDepth, traderData: Dict[str, Any]
    ):
        if (
            len(volcanic_rock_coupon_order_depth.buy_orders) > 0
            and len(volcanic_rock_coupon_order_depth.sell_orders) > 0
        ):
            best_bid = max(volcanic_rock_coupon_order_depth.buy_orders.keys())
            best_ask = min(volcanic_rock_coupon_order_depth.sell_orders.keys())
            traderData["prev_coupon_price"] = (best_bid + best_ask) / 2
            return (best_bid + best_ask) / 2
        else:
            return traderData["prev_coupon_price"]

    def delta_hedge_volcanic_rock_position(
        self,
        volcanic_rock_order_depth: OrderDepth,
        volcanic_rock_coupon_position: int,
        volcanic_rock_position: int,
        volcanic_rock_buy_orders: int,
        volcanic_rock_sell_orders: int,
        delta: float,
    ) -> List[Order]:
        """
        Delta hedge the overall position in VOLCANIC_ROCK_COUPON by creating orders in VOLCANIC_ROCK.

        Args:
            volcanic_rock_order_depth (OrderDepth): The order depth for the VOLCANIC_ROCK product.
            volcanic_rock_coupon_position (int): The current position in VOLCANIC_ROCK_COUPON.
            volcanic_rock_position (int): The current position in VOLCANIC_ROCK.
            volcanic_rock_buy_orders (int): The total quantity of buy orders for VOLCANIC_ROCK in the current iteration.
            volcanic_rock_sell_orders (int): The total quantity of sell orders for VOLCANIC_ROCK in the current iteration.
            delta (float): The current value of delta for the VOLCANIC_ROCK_COUPON product.
            traderData (Dict[str, Any]): The trader data for the VOLCANIC_ROCK_COUPON product.

        Returns:
            List[Order]: A list of orders to delta hedge the VOLCANIC_ROCK_COUPON position.
        """

        target_volcanic_rock_position = -int(delta * volcanic_rock_coupon_position)
        hedge_quantity = target_volcanic_rock_position - (
            volcanic_rock_position + volcanic_rock_buy_orders - volcanic_rock_sell_orders
        )

        orders: List[Order] = []
        if hedge_quantity > 0:
            # Buy VOLCANIC_ROCK
            best_ask = min(volcanic_rock_order_depth.sell_orders.keys())
            quantity = min(
                abs(hedge_quantity), -volcanic_rock_order_depth.sell_orders[best_ask]
            )
            quantity = min(
                quantity,
                self.LIMIT[Product.VOLCANIC_ROCK] - (volcanic_rock_position + volcanic_rock_buy_orders),
            )
            if quantity > 0:
                orders.append(Order(Product.VOLCANIC_ROCK, best_ask, quantity))
        elif hedge_quantity < 0:
            # Sell VOLCANIC_ROCK
            best_bid = max(volcanic_rock_order_depth.buy_orders.keys())
            quantity = min(
                abs(hedge_quantity), volcanic_rock_order_depth.buy_orders[best_bid]
            )
            quantity = min(
                quantity,
                self.LIMIT[Product.VOLCANIC_ROCK] + (volcanic_rock_position - volcanic_rock_sell_orders),
            )
            if quantity > 0:
                orders.append(Order(Product.VOLCANIC_ROCK, best_bid, -quantity))

        return orders

    def delta_hedge_volcanic_rock_coupon_orders(
        self,
        volcanic_rock_order_depth: OrderDepth,
        volcanic_rock_coupon_orders: List[Order],
        volcanic_rock_position: int,
        volcanic_rock_buy_orders: int,
        volcanic_rock_sell_orders: int,
        delta: float,
    ) -> List[Order]:
        """
        Delta hedge the new orders for VOLCANIC_ROCK_COUPON by creating orders in VOLCANIC_ROCK.

        Args:
            volcanic_rock_order_depth (OrderDepth): The order depth for the VOLCANIC_ROCK product.
            volcanic_rock_coupon_orders (List[Order]): The new orders for VOLCANIC_ROCK_COUPON.
            volcanic_rock_position (int): The current position in VOLCANIC_ROCK.
            volcanic_rock_buy_orders (int): The total quantity of buy orders for VOLCANIC_ROCK in the current iteration.
            volcanic_rock_sell_orders (int): The total quantity of sell orders for VOLCANIC_ROCK in the current iteration.
            delta (float): The current value of delta for the VOLCANIC_ROCK_COUPON product.

        Returns:
            List[Order]: A list of orders to delta hedge the new VOLCANIC_ROCK_COUPON orders.
        """
        if len(volcanic_rock_coupon_orders) == 0:
            return None

        net_volcanic_rock_coupon_quantity = sum(
            order.quantity for order in volcanic_rock_coupon_orders
        )
        target_volcanic_rock_quantity = -int(delta * net_volcanic_rock_coupon_quantity)

        orders: List[Order] = []
        if target_volcanic_rock_quantity > 0:
            # Buy VOLCANIC_ROCK
            best_ask = min(volcanic_rock_order_depth.sell_orders.keys())
            quantity = min(
                abs(target_volcanic_rock_quantity), -volcanic_rock_order_depth.sell_orders[best_ask]
            )
            quantity = min(
                quantity,
                self.LIMIT[Product.VOLCANIC_ROCK] - (volcanic_rock_position + volcanic_rock_buy_orders),
            )
            if quantity > 0:
                orders.append(Order(Product.VOLCANIC_ROCK, best_ask, quantity))
        elif target_volcanic_rock_quantity < 0:
            # Sell VOLCANIC_ROCK
            best_bid = max(volcanic_rock_order_depth.buy_orders.keys())
            quantity = min(
                abs(target_volcanic_rock_quantity), volcanic_rock_order_depth.buy_orders[best_bid]
            )
            quantity = min(
                quantity,
                self.LIMIT[Product.VOLCANIC_ROCK] + (volcanic_rock_position - volcanic_rock_sell_orders),
            )
            if quantity > 0:
                orders.append(Order(Product.VOLCANIC_ROCK, best_bid, -quantity))

        return orders

    def volcanic_rock_hedge_orders(
        self,
        volcanic_rock_order_depth: OrderDepth,
        volcanic_rock_coupon_order_depth: OrderDepth,
        volcanic_rock_coupon_orders: List[Order],
        volcanic_rock_position: int,
        volcanic_rock_coupon_position: int,
        delta: float,
    ) -> List[Order]:
        if volcanic_rock_coupon_orders == None or len(volcanic_rock_coupon_orders) == 0:
            volcanic_rock_coupon_position_after_trade = volcanic_rock_coupon_position
        else:
            volcanic_rock_coupon_position_after_trade = volcanic_rock_coupon_position + sum(
                order.quantity for order in volcanic_rock_coupon_orders
            )

        target_volcanic_rock_position = -delta * volcanic_rock_coupon_position_after_trade

        if target_volcanic_rock_position == volcanic_rock_position:
            return None

        target_volcanic_rock_quantity = target_volcanic_rock_position - volcanic_rock_position

        orders: List[Order] = []
        if target_volcanic_rock_quantity > 0:
            # Buy VOLCANIC_ROCK
            best_ask = min(volcanic_rock_order_depth.sell_orders.keys())
            quantity = min(
                abs(target_volcanic_rock_quantity),
                self.LIMIT[Product.VOLCANIC_ROCK] - volcanic_rock_position,
            )
            if quantity > 0:
                orders.append(Order(Product.VOLCANIC_ROCK, best_ask, round(quantity)))

        elif target_volcanic_rock_quantity < 0:
            # Sell VOLCANIC_ROCK
            best_bid = max(volcanic_rock_order_depth.buy_orders.keys())
            quantity = min(
                abs(target_volcanic_rock_quantity),
                self.LIMIT[Product.VOLCANIC_ROCK] + volcanic_rock_position,
            )
            if quantity > 0:
                orders.append(Order(Product.VOLCANIC_ROCK, best_bid, -round(quantity)))

        return orders

    def volcanic_rock_coupon_orders(
        self,
        COUPON: str,
        volcanic_rock_mid_price: float,
        volcanic_rock_coupon_order_depth: OrderDepth,
        volcanic_rock_coupon_position: int,
        traderData: Dict[str, Any],
        tte: float,
        volatility: float
    ) -> List[Order]:

        coupon_mid = self.get_volcanic_rock_coupon_mid_price(volcanic_rock_coupon_order_depth, traderData)
        call_fair_price = BlackScholes.black_scholes_call(volcanic_rock_mid_price, 
            self.params[COUPON]["strike"], tte, volatility)
        

        take_orders, buy_order_volume, sell_order_volume = self.take_orders(
            COUPON,
            volcanic_rock_coupon_order_depth, call_fair_price,
            self.params[COUPON]["take_width"],
            volcanic_rock_coupon_position,
            self.params[COUPON]["prevent_adverse"],
            self.params[COUPON]["adverse_volume"],
        )

        # clear_orders, buy_order_volume, sell_order_volume = self.clear_orders(
        #     COUPON,
        #     volcanic_rock_coupon_order_depth, call_fair_price,
        #     self.params[COUPON]["clear_width"],
        #     volcanic_rock_coupon_position,
        #     buy_order_volume,
        #     sell_order_volume,
        # )

        make_orders, _, _ = self.make_orders(
            COUPON,
            volcanic_rock_coupon_order_depth, call_fair_price,
            volcanic_rock_coupon_position,
            buy_order_volume,
            sell_order_volume,
            self.params[COUPON]["disregard_edge"],
            self.params[COUPON]["join_edge"],
            self.params[COUPON]["default_edge"],
        )
        return take_orders, make_orders

    def run(self, state: TradingState):
        traderObject = {}
        if state.traderData != None and state.traderData != "":
            traderObject = jsonpickle.decode(state.traderData)
        conversions = 0


        # if state.timestamp == 10000:
        #     raise Exception(state.observations)
        
        result = {}
        result[Product.VOLCANIC_ROCK] = []
        result[Product.RAINFOREST_RESIN] = self.resin_strategy(state, traderObject)
        result[Product.KELP] = self.kelp_strategy(state, traderObject)
        result[Product.SQUID_INK] = self.ink_strategy(state, traderObject)
        result[Product.CROISSANTS] = self.croissants_strategy(state, traderObject)
        result[Product.JAMS] = self.jams_strategy(state, traderObject)
        result[Product.DJEMBES] = self.djembes_strategy(state, traderObject)
        result[Product.VOLCANIC_ROCK] = self.volcanic_rock_strategy(state, traderObject)
        result[Product.MAGNIFICENT_MACARONS] = self.magnificent_macarons_strategy(state, traderObject, conversions)

        spread_strat = self.spread_strategy(state, traderObject)
        spread2_strat = self.spread2_strategy(state, traderObject)
        # #
        for product in spread2_strat:
            if product not in result:
                result[product] = []
            if product in [Product.PICNIC_BASKET2]:
                for order in spread2_strat[product]:
                    result[product].append(order)
        #
        for product in spread_strat:
            if product not in result:
                result[product] = []
            if product in [Product.PICNIC_BASKET1]:
                for order in spread_strat[product]:
                    result[product].append(order)


        #loop through this once per coupon
        for COUPON in [Product.VOLCANIC_ROCK_VOUCHER_9500,
                       Product.VOLCANIC_ROCK_VOUCHER_9750,
                       Product.VOLCANIC_ROCK_VOUCHER_10000,
                       Product.VOLCANIC_ROCK_VOUCHER_10250,
                       Product.VOLCANIC_ROCK_VOUCHER_10500]:
            if COUPON not in traderObject:
                traderObject[COUPON] = {
                    "prev_coupon_price": 0,
                    "past_coupon_vol": [],
                }

            if (
                COUPON in self.params
                and COUPON in state.order_depths
            ):
                volcanic_rock_coupon_position = (
                    state.position[COUPON]
                    if COUPON in state.position
                    else 0
                )

                volcanic_rock_position = (
                    state.position[Product.VOLCANIC_ROCK]
                    if Product.VOLCANIC_ROCK in state.position
                    else 0
                )
                # print(f"volcanic_rock_coupon_position: {volcanic_rock_coupon_position}")
                # print(f"volcanic_rock_position: {volcanic_rock_position}")
                volcanic_rock_order_depth = state.order_depths[Product.VOLCANIC_ROCK]
                volcanic_rock_coupon_order_depth = state.order_depths[COUPON]
                volcanic_rock_mid_price = (
                    max(volcanic_rock_order_depth.buy_orders.keys())
                    + min(volcanic_rock_order_depth.sell_orders.keys())
                ) / 2


                self.past_prices[Product.VOLCANIC_ROCK].append(volcanic_rock_mid_price)
                if (len(self.past_prices[Product.VOLCANIC_ROCK]) >= 2):
                    self.log_returns[Product.VOLCANIC_ROCK].append(log(self.past_prices[Product.VOLCANIC_ROCK][-1]) - log(self.past_prices[Product.VOLCANIC_ROCK][-2]))


                volcanic_rock_coupon_mid_price = self.get_volcanic_rock_coupon_mid_price(
                    volcanic_rock_coupon_order_depth, traderObject[COUPON]
                )
                tte = (
                    (8 - 0*self.params[COUPON]["starting_time_to_expiry"]
                    - (state.timestamp) / 1000000) / ( 365)
                )
                
                
                volatility = self.params[COUPON]["mean_volatility"]

                if (len(self.log_returns[Product.VOLCANIC_ROCK]) > self.params[COUPON]["std_window"]):
                    volatility = np.std(self.log_returns[Product.VOLCANIC_ROCK][-self.params[COUPON]["std_window"]:]) * sqrt(365) + np.finfo(float).eps
                
                volatility = 0.15

                m_t = log(self.params[COUPON]["strike"]/volcanic_rock_mid_price)/sqrt(tte)

                volatility = 0.2333333 * m_t**2 + 0.147

                
    

                delta = BlackScholes.delta(
                    volcanic_rock_mid_price,
                    self.params[COUPON]["strike"],
                    tte,
                    volatility,
                )

                copied_order_depths = copy.deepcopy(state.order_depths[COUPON])
   
                implied_volatility = BlackScholes.implied_volatility(
                    volcanic_rock_coupon_mid_price,
                    volcanic_rock_mid_price,
                    self.params[COUPON]["strike"],
                    tte,
                )
                call_fair_price = BlackScholes.black_scholes_call(volcanic_rock_mid_price, 
                self.params[COUPON]["strike"], tte, volatility)

                result[COUPON] = []

                logger.logs += f"{COUPON}: {call_fair_price}\n"

                if (implied_volatility > volatility + 0.02):
                    #go long
                    buy_quantity = self.LIMIT[COUPON] - (volcanic_rock_coupon_position)
                    if buy_quantity > 0:
                        if volcanic_rock_coupon_order_depth.sell_orders.keys():
                            result[COUPON].append(Order(COUPON, min(volcanic_rock_coupon_order_depth.sell_orders.keys()), buy_quantity))  # Buy order
                        else:
                            result[COUPON].append(Order(COUPON, round(volcanic_rock_coupon_mid_price), buy_quantity))  # Buy order

                elif (implied_volatility < volatility - 0.02):
                    #go short
                    sell_quantity = -self.LIMIT[COUPON] - (volcanic_rock_coupon_position)
                    if sell_quantity < 0:
                        if volcanic_rock_coupon_order_depth.buy_orders.keys():
                            result[COUPON].append(Order(COUPON, max(volcanic_rock_coupon_order_depth.buy_orders.keys()), sell_quantity))  # Buy order
                        else: result[COUPON].append(Order(COUPON, round(volcanic_rock_coupon_mid_price), sell_quantity))  # Buy order


                

                self.trader_memory["m_t"].append(log(9500/volcanic_rock_mid_price)/sqrt(tte))
                self.trader_memory["v_t"].append(BlackScholes.implied_volatility(
                volcanic_rock_coupon_mid_price,
                volcanic_rock_mid_price,
                self.params[COUPON]["strike"],
                tte,
                ))


                


        # conversions = 1
        traderData = jsonpickle.encode(traderObject)

        logger.flush(state, result, conversions, state.traderData)

        return result, conversions, traderData