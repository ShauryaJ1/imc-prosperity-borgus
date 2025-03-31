from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List
import string
import math
class Trader:
    def __init__(self):
        self.risk = 0.10
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
            self.spreads[symbol] = min(state.order_depths[symbol].sell_orders) - max(state.order_depths[symbol].buy_orders) + self.profit_target[symbol]
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
        orders.append(Order("RAINFOREST_RESIN", self.default_prices["RAINFOREST_RESIN"] - 2, bid_volume))
        orders.append(Order("RAINFOREST_RESIN", self.default_prices["RAINFOREST_RESIN"] + 2, ask_volume))
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
        orders.append(Order("RAINFOREST_RESIN", math.floor(self.default_prices["RAINFOREST_RESIN"] - 2 - self.spreads["RAINFOREST_RESIN"]/2), bid_volume))
        orders.append(Order("RAINFOREST_RESIN", math.ceil(self.default_prices["RAINFOREST_RESIN"] + 2 + self.spreads["RAINFOREST_RESIN"]/2), ask_volume))
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
            orders.append(Order("KELP", math.floor(combined_price - 1), bid_volume))
            orders.append(Order("KELP", math.ceil(combined_price + 1), ask_volume))
        
        if current_position > 0:
            # Long position
            orders.append(Order("KELP", math.floor(combined_price - 2), bid_volume))
            orders.append(Order("KELP", math.ceil(combined_price), ask_volume))

        if current_position < 0:
            # Short position
            orders.append(Order("KELP", math.floor(combined_price), bid_volume))
            orders.append(Order("KELP", math.ceil(combined_price+ 2), ask_volume))

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
            orders.append(Order("KELP", math.floor(self.mcginley_prices["KELP"] - 1), bid_volume))
            orders.append(Order("KELP", math.ceil(self.mcginley_prices["KELP"] + 1), ask_volume))
        
        if current_position > 0:
            # Long position
            orders.append(Order("KELP", math.floor(self.mcginley_prices["KELP"] - 2), bid_volume))
            orders.append(Order("KELP", math.ceil(self.mcginley_prices["KELP"]), ask_volume))

        if current_position < 0:
            # Short position
            orders.append(Order("KELP", math.floor(self.mcginley_prices["KELP"]), bid_volume))
            orders.append(Order("KELP", math.ceil(self.mcginley_prices["KELP"] + 2), ask_volume))

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
            orders.append(Order("KELP", math.floor(self.mcginley_prices["KELP"] - 1 -self.spreads["KELP"]/2 ), bid_volume))
            orders.append(Order("KELP", math.ceil(self.mcginley_prices["KELP"] + 1 + self.spreads["KELP"]/2), ask_volume))
        
        if current_position > 0:
            # Long position
            orders.append(Order("KELP", math.floor(self.mcginley_prices["KELP"] - 2 -current_position*self.risk - self.spreads["KELP"]/2), bid_volume))
            orders.append(Order("KELP", math.ceil(self.mcginley_prices["KELP"] + -current_position*self.risk + self.spreads["KELP"]/2), ask_volume))

        if current_position < 0:
            # Short position
            orders.append(Order("KELP", math.floor(self.mcginley_prices["KELP"]  -current_position*self.risk - self.spreads["KELP"]/2), bid_volume))
            orders.append(Order("KELP", math.ceil(self.mcginley_prices["KELP"] + 2 -current_position*self.risk + self.spreads["KELP"]/2), ask_volume))

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
        # print("Beginning of run")
        # print(self.position_limit)
        # Only method required. It takes all buy and sell orders for all symbols as an input, and outputs a list of orders to be sent
        for product in self.products:
            # self.update_ema_prices(product, state)
            self.update_mcginley(product, state)
            # self.update_spreads(product, state)
        # print(self.mcginley_prices)
        # print(self.mcginley_prices["KELP"])
        print(min(state.order_depths["KELP"].sell_orders) - max(state.order_depths["KELP"].buy_orders) + self.profit_target["KELP"])
        print(min(state.order_depths["RAINFOREST_RESIN"].sell_orders) - max(state.order_depths["RAINFOREST_RESIN"].buy_orders) + self.profit_target["RAINFOREST_RESIN"])
        # print("Mid price of : ", self.get_price)
        orders = {}        
        orders["KELP"] = self.kelp_trading_mcginley(state)
        # orders.extend(self.resin_trading(state))
        orders["RAINFOREST_RESIN"] = self.resin_trading(state)
        
        return orders, 1, state.traderData

