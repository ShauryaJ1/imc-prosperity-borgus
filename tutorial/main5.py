from datamodel import OrderDepth, UserId, TradingState, Order, Symbol, Listing, Trade, Observation, ProsperityEncoder
from typing import List
import string
import jsonpickle
import numpy as np
import math
from typing import Any
from json import *


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


class Product:
    RAINFOREST_RESIN = "RAINFOREST_RESIN"
    KELP = "KELP"
    SQUID_INK = "SQUID_INK"


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
}


class Trader:
    def __init__(self, params=None):
        if params is None:
            params = PARAMS
        self.params = params

        self.LIMIT = {Product.RAINFOREST_RESIN: 50, Product.KELP: 50, Product.SQUID_INK: 50}

        self.trader_memory = {
            "ink_price_history": [],
            "kelp_price_history": [],
            "volitality_arr": [],
            "z_score_arr": []
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

            price_history: List[float] = self.trader_memory.get("ink_price_history", [])
            volitality_array: List[float] = self.trader_memory.get("volitality_arr", [])
            z_score_array: List[float] = self.trader_memory.get("z_score_arr", [])

            if len(price_history) >= self.params[Product.SQUID_INK]["history_window"]:
                beta = self.params[Product.SQUID_INK]["ema_param"]

                volitality_array.append(np.std(price_history))
                z_score_array.append((mmmid_price - np.mean(price_history)) / np.std(price_history))

            # update price history
            price_history.append(mmmid_price)

            if len(price_history) > self.params[Product.SQUID_INK]["history_window"]:
                price_history.pop(0)

            self.trader_memory["ink_price_history"] = price_history
            self.trader_memory["volitality_arr"] = volitality_array
            self.trader_memory["z_score_arr"] = z_score_array

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

            fair = self.ink_fair_value(state.order_depths["SQUID_INK"], traderObject)
            hist = self.trader_memory.get("ink_price_history", [])
            moving_average = fair if not hist else np.mean(hist)
            exp_moving_avg = fair if not hist else (
                sum(hist[i] * self.params[Product.SQUID_INK]["ema_param"] ** (len(hist) - i) for i in range(len(hist)))
                / sum(self.params[Product.SQUID_INK]["ema_param"] ** (len(hist) - i) for i in range(len(hist)))
            )
            mcginley_moving_avg = fair if not hist else hist[0]
            for i in range(1, len(hist)):
                mcginley_moving_avg += (hist[i] - mcginley_moving_avg) / (
                        0.6 * self.params[Product.SQUID_INK]["history_window"] * (hist[i] / mcginley_moving_avg) ** 4)

            moving_average = exp_moving_avg

            ink_std = 0 if not hist else np.std(hist)
            K = self.params[Product.SQUID_INK]["stdev_mul"]

            logger.print(f"Moving Average: {moving_average}")
            logger.print(f"Ink stdev: {ink_std}")

            def spearman_corr(x, y, n_perm=33, seed=None):
                rng = np.random.default_rng(seed)

                def rankdata(a):
                    temp = a.argsort()
                    ranks = np.empty_like(temp)
                    ranks[temp] = np.arange(len(a))
                    return ranks + 1  # 1-based rank

                rx = rankdata(x)
                ry = rankdata(y)
                rho_obs = np.corrcoef(rx, ry)[0, 1]

                # Permutation test
                count = 0
                for _ in range(n_perm):
                    ry_perm = rng.permutation(ry)
                    rho_perm = np.corrcoef(rx, ry_perm)[0, 1]
                    if rho_perm >= rho_obs:
                        count += 1

                p_value = (count + 1) / (n_perm + 1)
                return rho_obs, p_value



            rho, p_val = spearman_corr(x= np.array([*range(0, len(hist))]), y= np.array(hist))
            buy_order_volume, sell_order_volume = 0, 0

            if rho > 0 and p_val < 0.03:
                #going up!
                logger.print("going up!")
                logger.print(f"hist: {hist}")
                bid = round(fair + 1)
                ask = round(bid + 2 * ink_std)

                orders = []

                orders.append(Order(Product.SQUID_INK, bid, 20 * (1 - 2.71828 ** (-1 * abs(rho)))))
                orders.append(Order(Product.SQUID_INK, ask, 20 * (1 - 2.71828 ** (-1 * abs(rho)))))

                # buy_order_volume, sell_order_volume = self.market_make(
                #     Product.SQUID_INK,
                #     orders,
                #     bid,
                #     ask,
                #     ink_position,
                #     0,
                #     0
                # )

            elif rho < 0 and 1 - p_val < 0.03:
                #going down!
                logger.print("going down!")
                logger.print(f"hist: {hist}")
                ask = round(fair - 1)
                bid = round(ask - 2 * ink_std)

                logger.print(bid, ask)

                orders = []

                orders.append(Order(Product.SQUID_INK, bid, 20 * (1 - 2.71828 ** (-1 * abs(rho)))))
                orders.append(Order(Product.SQUID_INK, ask, 20 * (1 - 2.71828 ** (-1 * abs(rho)))))

                # buy_order_volume, sell_order_volume = self.market_make(
                #     Product.SQUID_INK,
                #     orders,
                #     bid,
                #     ask,
                #     ink_position,
                #     0,
                #     0
                # )

            else:
                logger.print(f"hist: {hist}")
                orders, buy_order_volume, sell_order_volume = (
                self.take_orders(
                    Product.SQUID_INK,
                    state.order_depths[Product.SQUID_INK],
                    fair_value= moving_average,
                    take_width= round(K * ink_std),
                    position= ink_position
                ))

            ink_clear_orders, buy_order_volume, sell_order_volume = (
            self.clear_orders(
                Product.SQUID_INK,
                state.order_depths["SQUID_INK"],
                fair_value= moving_average,
                clear_width= 0,
                position= ink_position,
                buy_order_volume= buy_order_volume,
                sell_order_volume= sell_order_volume
            ))

            logger.print(orders + ink_clear_orders)
            return orders + ink_clear_orders

        return []

    def run(self, state: TradingState):
        traderObject = {}
        if state.traderData != None and state.traderData != "":
            traderObject = jsonpickle.decode(state.traderData)

        result = {}
        result[Product.RAINFOREST_RESIN] = self.resin_strategy(state, traderObject)
        result[Product.KELP] = self.kelp_strategy(state, traderObject)
        result[Product.SQUID_INK] = self.ink_strategy(state, traderObject)

        conversions = 1
        traderData = jsonpickle.encode(traderObject)

        logger.flush(state, result, 1, state.traderData)

        return result, conversions, traderData
