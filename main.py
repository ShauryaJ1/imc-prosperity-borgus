from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List
import string
import math
class Trader:
    def __init__(self):
        self.position_limit = {
            "KELP": 50,
            "RAINFOREST_RESIN": 50
        }
        self.ema_param = 0.55
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
    def get_price(self,symbol, state: TradingState):
        if self.ema_prices[symbol] is None:
            default_price = self.default_prices[symbol]
        else:
            default_price = self.ema_prices[symbol]
        if symbol not in state.order_depths:
            return default_price
        if len(state.order_depths[symbol].buy_orders) == 0 or len(state.order_depths[symbol].sell_orders) == 0:
            return default_price
        return (max(state.order_depths[symbol].buy_orders)+ min(state.order_depths[symbol].sell_orders))/2
    def update_ema_prices(self, symbol, state):
            mid_price = self.get_price(symbol, state)

            if self.ema_prices[symbol] is None:
                self.ema_prices[symbol] = mid_price
            else:
                self.ema_prices[symbol] = self.ema_param * mid_price + (1-self.ema_param) * self.ema_prices[symbol]
    def resin_trading(self,state:TradingState):
        if "RAINFOREST_RESIN" not in state.position:
            current_position = 0
        else:
            current_position = state.position["RAINFOREST_RESIN"]
        bid_volume = self.position_limit["RAINFOREST_RESIN"] - current_position
        ask_volume = - self.position_limit["RAINFOREST_RESIN"] - current_position

        orders = []
        orders.append(Order("RAINFOREST_RESIN", self.default_prices["RAINFOREST_RESIN"] - 2, bid_volume))
        orders.append(Order("RAINFOREST_RESIN", self.default_prices["RAINFOREST_RESIN"] + 2, ask_volume))
        return orders
    def kelp_trading(self,state:TradingState):
        if "KELP" not in state.position:
            current_position = 0
        else:
            current_position = state.position["KELP"]
        bid_volume = self.position_limit["KELP"] - current_position
        ask_volume = - self.position_limit["KELP"] - current_position

        orders = []

        if current_position == 0:
            # Not long nor short
            orders.append(Order("KELP", math.floor(self.ema_prices["KELP"] - 1), bid_volume))
            orders.append(Order("KELP", math.ceil(self.ema_prices["KELP"] + 1), ask_volume))
        
        if current_position > 0:
            # Long position
            orders.append(Order("KELP", math.floor(self.ema_prices["KELP"] - 2), bid_volume))
            orders.append(Order("KELP", math.ceil(self.ema_prices["KELP"]), ask_volume))

        if current_position < 0:
            # Short position
            orders.append(Order("KELP", math.floor(self.ema_prices["KELP"]), bid_volume))
            orders.append(Order("KELP", math.ceil(self.ema_prices["KELP"] + 2), ask_volume))

        return orders
    
    def run(self, state: TradingState):
        print("Beginning of run")
        print(self.position_limit)
        # Only method required. It takes all buy and sell orders for all symbols as an input, and outputs a list of orders to be sent
        for product in self.products:
            self.update_ema_prices(product, state)
        orders = {}        
        orders["KELP"] = self.kelp_trading(state)
        # orders.extend(self.resin_trading(state))
        orders["RAINFOREST_RESIN"] = self.resin_trading(state)
        
        return orders, 1, state.traderData

