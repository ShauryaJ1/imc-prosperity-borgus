from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List
import string
import math

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


class Trader:
    def __init__(self):
        self.risk = 0.14
        self.profit_target = {"KELP": 2, "RAINFOREST_RESIN": 1}
        self.position_limit = {
            "KELP": 50,
            "RAINFOREST_RESIN": 50
        }
        self.volumes = {
            "KELP": 5,
            "RAINFOREST_RESIN": 10
        }
        self.ema_param = 0.5
        self.default_prices = {
            "KELP":2000,
            "RAINFOREST_RESIN":10000
        }
        self.products = [
            "KELP",
            "RAINFOREST_RESIN"
        ]
        self.past_prices = {
            product:[] for product in self.products
        }
        self.ema_prices = {
            product:None for product in self.products
        }
        self.mcginley_prices = {
            product:None for product in self.products
        }
        # self.spreads = {
        #     product:0 for product in self.products
        # }
        self.spreads = {
            "KELP": 4,
            "RAINFOREST_RESIN": 7
        }
        self.idealProfits = {
            "KELP": 1.1,
            "RAINFOREST_RESIN": 4
        }

    def get_price(self,symbol, state: TradingState):
        # if self.ema_prices[symbol] is None:
        #     default_price = self.default_prices[symbol]
        # else:
        #     default_price = self.ema_prices[symbol]
        # if symbol not in state.order_depths:
        #     return default_price
        # if len(state.order_depths[symbol].buy_orders) == 0 or len(state.order_depths[symbol].sell_orders) == 0:
        #     return default_price
        return (max(state.order_depths[symbol].buy_orders)+ min(state.order_depths[symbol].sell_orders))/2
    def update_spreads(self, symbol, state):
        for symbol in self.products:
            self.spreads[symbol] = min(state.order_depths[symbol].sell_orders) - max(state.order_depths[symbol].buy_orders)
    def update_ema_prices(self, symbol, state):
            mid_price = self.get_price(symbol, state)

            if self.ema_prices[symbol] is None:
                self.ema_prices[symbol] = mid_price
            else:
                self.ema_prices[symbol] = self.ema_param * mid_price + (1-self.ema_param) * self.ema_prices[symbol]
    def update_mcginley(self,symbol,state):
        mid_price = self.get_price(symbol, state)
        if self.mcginley_prices[symbol] is None:
            self.mcginley_prices[symbol] = mid_price
        else:
            self.mcginley_prices[symbol] = self.mcginley_prices[symbol]+(mid_price-self.mcginley_prices[symbol])/(0.6*10*(mid_price/self.mcginley_prices[symbol])**4)
    def resin_trading(self,state:TradingState):
        if "RAINFOREST_RESIN" not in state.position:
            current_position = 0
        else:
            current_position = state.position["RAINFOREST_RESIN"]
        bid_volume = self.position_limit["RAINFOREST_RESIN"] - current_position
        ask_volume = - self.position_limit["RAINFOREST_RESIN"] - current_position

        orders = []
        orders.append(Order("RAINFOREST_RESIN", math.floor(self.default_prices["RAINFOREST_RESIN"] - self.idealProfits["RAINFOREST_RESIN"]/2), bid_volume))
        orders.append(Order("RAINFOREST_RESIN", math.ceil(self.default_prices["RAINFOREST_RESIN"] + self.idealProfits["RAINFOREST_RESIN"]/2), ask_volume))
        return orders
    def resin_risk_spread(self,state:TradingState):
        if "RAINFOREST_RESIN" not in state.position:
            current_position = 0
        else:
            current_position = state.position["RAINFOREST_RESIN"]
        bid_volume = self.position_limit["RAINFOREST_RESIN"] - current_position
        ask_volume = - self.position_limit["RAINFOREST_RESIN"] - current_position
        # bid_volume = self.volumes["RAINFOREST_RESIN"]
        # ask_volume = self.volumes["RAINFOREST_RESIN"]
        orders = []
        orders.append(Order("RAINFOREST_RESIN", math.floor(self.default_prices["RAINFOREST_RESIN"] - self.idealProfits["RAINFOREST_RESIN"]/2 - self.spreads["RAINFOREST_RESIN"]/2), bid_volume))
        orders.append(Order("RAINFOREST_RESIN", math.ceil(self.default_prices["RAINFOREST_RESIN"] + self.idealProfits["RAINFOREST_RESIN"]/2 + self.spreads["RAINFOREST_RESIN"]/2), ask_volume))
        return orders
    def kelp_trading_combined(self,state:TradingState):
        if "KELP" not in state.position:
            current_position = 0
        else:
            current_position = state.position["KELP"]
        bid_volume = self.position_limit["KELP"] - current_position
        ask_volume = - self.position_limit["KELP"] - current_position

        orders = []
        combined_price = 1.1*self.ema_prices["KELP"] - 0.1*self.mcginley_prices["KELP"]
        if current_position == 0:
            # Not long nor short
            orders.append(Order("KELP", math.floor(combined_price - self.idealProfits["KELP"] / 2), bid_volume))
            orders.append(Order("KELP", math.ceil(combined_price + self.idealProfits["KELP"] / 2), ask_volume))
        
        if current_position > 0:
            # Long position
            orders.append(Order("KELP", math.floor(combined_price - self.idealProfits["KELP"]), bid_volume))
            orders.append(Order("KELP", math.ceil(combined_price), ask_volume))

        if current_position < 0:
            # Short position
            orders.append(Order("KELP", math.floor(combined_price), bid_volume))
            orders.append(Order("KELP", math.ceil(combined_price+ self.idealProfits["KELP"]), ask_volume))

        return orders
    def kelp_trading_mcginley(self,state:TradingState):
        if "KELP" not in state.position:
            current_position = 0
        else:
            current_position = state.position["KELP"]
        bid_volume = self.position_limit["KELP"] - current_position
        ask_volume = - self.position_limit["KELP"] - current_position

        orders = []

        if current_position == 0:
            # Not long nor short
            orders.append(Order("KELP", math.floor(self.mcginley_prices["KELP"] - self.idealProfits["KELP"] / 2), bid_volume))
            orders.append(Order("KELP", math.ceil(self.mcginley_prices["KELP"] + self.idealProfits["KELP"] / 2), ask_volume))
        
        if current_position > 0:
            # Long position
            orders.append(Order("KELP", math.floor(self.mcginley_prices["KELP"] - self.idealProfits["KELP"]), bid_volume))
            orders.append(Order("KELP", math.ceil(self.mcginley_prices["KELP"]), ask_volume))

        if current_position < 0:
            # Short position
            orders.append(Order("KELP", math.floor(self.mcginley_prices["KELP"]), bid_volume))
            orders.append(Order("KELP", math.ceil(self.mcginley_prices["KELP"] + self.idealProfits["KELP"]), ask_volume))

        return orders
    def kelp_trading_mcginley_risk_spread(self,state:TradingState):
        if "KELP" not in state.position:
            current_position = 0
        else:
            current_position = state.position["KELP"]
        bid_volume = self.position_limit["KELP"] - current_position
        ask_volume = - self.position_limit["KELP"] - current_position
        # bid_volume = self.volumes["KELP"]
        # ask_volume = self.volumes["KELP"]
        orders = []

        if current_position == 0:
            # Not long nor short
            orders.append(Order("KELP", math.floor(self.mcginley_prices["KELP"] - self.idealProfits["KELP"] / 2 -self.spreads["KELP"]/2 ), bid_volume))
            orders.append(Order("KELP", math.ceil(self.mcginley_prices["KELP"] + self.idealProfits["KELP"] / 2 + self.spreads["KELP"]/2), ask_volume))
        
        if current_position > 0:
            # Long position
            orders.append(Order("KELP", math.floor(self.mcginley_prices["KELP"] - self.idealProfits["KELP"] -current_position*self.risk - self.spreads["KELP"]/2), bid_volume))
            orders.append(Order("KELP", math.ceil(self.mcginley_prices["KELP"] + -current_position*self.risk + self.spreads["KELP"]/2), ask_volume))

        if current_position < 0:
            # Short position
            orders.append(Order("KELP", math.floor(self.mcginley_prices["KELP"]  -current_position*self.risk - self.spreads["KELP"]/2), bid_volume))
            orders.append(Order("KELP", math.ceil(self.mcginley_prices["KELP"] + self.idealProfits["KELP"] -current_position*self.risk + self.spreads["KELP"]/2), ask_volume))

        return orders
    def kelp_trading_ema(self,state:TradingState):
        if "KELP" not in state.position:
            current_position = 0
        else:
            current_position = state.position["KELP"]
        bid_volume = self.position_limit["KELP"] - current_position
        ask_volume = - self.position_limit["KELP"] - current_position

        orders = []

        if current_position == 0:
            # Not long nor short
            orders.append(Order("KELP", math.floor(self.ema_prices["KELP"] - self.idealProfits["KELP"] / 2), bid_volume))
            orders.append(Order("KELP", math.ceil(self.ema_prices["KELP"] + self.idealProfits["KELP"] / 2), ask_volume))
        
        if current_position > 0:
            # Long position
            orders.append(Order("KELP", math.floor(self.ema_prices["KELP"] - self.idealProfits["KELP"]), bid_volume))
            orders.append(Order("KELP", math.ceil(self.ema_prices["KELP"]), ask_volume))

        if current_position < 0:
            # Short position
            orders.append(Order("KELP", math.floor(self.ema_prices["KELP"]), bid_volume))
            orders.append(Order("KELP", math.ceil(self.ema_prices["KELP"] + self.idealProfits["KELP"]), ask_volume))

        return orders

    def run(self, state: TradingState):
        # print("Beginning of run")
        # print(self.position_limit)
        # Only method required. It takes all buy and sell orders for all symbols as an input, and outputs a list of orders to be sent
        for product in self.products:
            # self.update_ema_prices(product, state)
            self.update_mcginley(product, state)
            self.update_spreads(product, state)
        # print(self.mcginley_prices)
        # print(self.mcginley_prices["KELP"])
        logger.print(min(state.order_depths["KELP"].sell_orders) - max(state.order_depths["KELP"].buy_orders) + self.profit_target["KELP"])
        logger.print(min(state.order_depths["RAINFOREST_RESIN"].sell_orders) - max(state.order_depths["RAINFOREST_RESIN"].buy_orders) + self.profit_target["RAINFOREST_RESIN"])
        # print("Mid price of : ", self.get_price)
        orders = {}

        orders["KELP"] = self.kelp_trading_mcginley(state)
        # orders.extend(self.resin_trading(state))
        orders["RAINFOREST_RESIN"] = self.resin_trading(state)

        logger.flush(state, orders, 1, state.traderData)

        return orders, 1, state.traderData

