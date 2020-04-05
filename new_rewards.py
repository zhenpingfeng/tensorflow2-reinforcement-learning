# 複数のポジションを保有する場合、最初のポジションから処理をして行く
import numpy as np

class Reward:
    def __init__(self, spread, leverage, pip_cost, min_lots=0.01, assets=1000000, available_assets_rate=0.4):
        self.spread = spread
        self.leverage = leverage
        self.pip_cost = pip_cost
        self.min_lots = min_lots
        self.initial_assets = assets
        self.available_assets_rate = available_assets_rate

        self.reset()

    def reset(self):
        self.assets = self.initial_assets

        self.growth_rate = []
        self.positions = None
        self.total_gain = []
        self.l = []

    def reward(self, trend, high, low, action, atr, scale_atr):
        self.reset()
        old_action = None
        position = None
        gain = 0
        self.los_cut = self.max_los_cut
        old_assets = self.assets

        self.losses = []
        undetermined_assets = self.assets
        self.positions = 0
        self.lot = 0
        self.tp = 0
        self.los_cut = 0
        self.minimum_required_margin = trend[0] * self.pip_cost / self.leverage

        for action, trend, idx, atr, scale_atr in zip(action, trend, range(len(high)), atr, scale_atr):
            if old_action == None and position == None and self.assets > self.minimum_required_margin * 2 and action != 2:
                self.minimum_required_margin = trend * self.pip_cost / 500
                lot = self.assets * self.available_assets_rate / self.minimum_required_margin * self.min_lots
                self.lot = int(lot * 10 ** 2) / (10 ** 2)

                self.positions = trend + self.spread if action == 0 else trend - self.spread
                self.tp = min(abs(self.max_los_cut), atr * self.pip_cost)
                self.los_cut = max(self.max_los_cut, -atr * self.pip_cost)
                old_action = action

            elif old_action != None:
                if self.positions != None:
                    if old_action == 0:
                        gain = (trend - self.positions) * self.pip_cost * self.lot * 100
                        loss = (low[idx - 1] - self.positions) * self.pip_cost * self.lot * 100
                    else:
                        gain = (self.positions - trend) * self.pip_cost * self.lot * 100
                        loss = (self.positions - high[idx - 1]) * self.pip_cost * self.lot * 100

                    loss = min(gain, loss)
                    tp = self.tp * self.lot * 100
                    loss_cut = self.los_cut * self.lot * 100
                    gain = loss_cut if loss <= loss_cut else gain
                    undetermined_assets = max(self.assets + gain, self.assets + loss_cut)

                    confirm = gain == loss_cut
                    buy = old_action != action and action != 2
                    if confirm or buy:
                        self.assets = undetermined_assets

                        if self.assets > self.minimum_required_margin * 2:
                            if action == 2:
                                self.positions = None
                                old_action = None
                            else:
                                self.minimum_required_margin = trend * self.pip_cost / self.leverage
                                lot = self.assets * self.available_assets_rate / self.minimum_required_margin * self.min_lots
                                self.lot = int(lot * 10 ** 2) / (10 ** 2)
                                self.positions = trend + self.spread if action == 0 else trend - self.spread
                                self.tp = min(abs(self.max_los_cut), atr * self.pip_cost)
                                self.los_cut = max(self.max_los_cut, -atr * self.pip_cost)
                        else:
                            self.positions = None

                    # else:

                    old_action = action if action != 2 else old_action

                elif self.assets > self.minimum_required_margin * 2:
                    old_action = None

            self.total_gain.append(undetermined_assets)
            try:
                if self.total_gain[-1] == self.total_gain[-2]:
                    self.total_gain[-1] = undetermined_assets - undetermined_assets * 0.1
            except:
                pass

            self.losses.append(self.los_cut)

            gain = 0
            self.growth_rate.append(np.log(undetermined_assets / self.initial_assets) * 100)
        self.assets = undetermined_assets


class Reward:
    def __init__(self, spread, leverage, pip_cost, min_lots=0.01, assets=1000000, available_assets_rate=0.4):
        self.spread = spread
        self.leverage = leverage
        self.pip_cost = pip_cost
        self.min_lots = min_lots
        self.lc = 0
        self.tp = 0
        self.initial_assets = assets
        self.assets = assets
        self.available_assets_rate = available_assets_rate

        self.max_los_cut = None
        self.total_gain = []
        self.l = []
        # use rewards.max_los_cut = -np.mean(atr) * pip_cost

        self.growth_rate = []
        self.positions = None

    def reset(self):
        self.assets = self.initial_assets

        self.growth_rate = []
        self.positions = None
        self.total_gain = []
        self.l = []

    def reward(self, trend, high, low, action, atr, scale_atr):
        self.reset()
        old_action = None
        position = None
        gain = 0
        self.los_cut = self.max_los_cut
        old_assets = self.assets

        self.losses = []
        undetermined_assets = self.assets
        self.positions = 0
        self.lot = 0
        self.tp = 0
        self.los_cut = 0
        self.minimum_required_margin = trend[0] * self.pip_cost / self.leverage

        for action, trend, idx, atr, scale_atr in zip(action, trend, range(len(high)), atr, scale_atr):
            if old_action == None and position == None and self.assets > self.minimum_required_margin * 2 and action != 2:
                self.minimum_required_margin = trend * self.pip_cost / 500
                lot = self.assets * self.available_assets_rate / self.minimum_required_margin * self.min_lots
                self.lot = int(lot * 10 ** 2) / (10 ** 2)

                self.positions = trend + self.spread if action == 0 else trend - self.spread
                self.tp = min(abs(self.max_los_cut), atr * self.pip_cost)
                self.los_cut = max(self.max_los_cut, -atr * self.pip_cost)
                old_action = action

            elif old_action != None:
                if self.positions != None:
                    if old_action == 0:
                        gain = (trend - self.positions) * self.pip_cost * self.lot * 100
                        loss = (low[idx - 1] - self.positions) * self.pip_cost * self.lot * 100
                    else:
                        gain = (self.positions - trend) * self.pip_cost * self.lot * 100
                        loss = (self.positions - high[idx - 1]) * self.pip_cost * self.lot * 100

                    loss = min(gain, loss)
                    tp = self.tp * self.lot * 100
                    loss_cut = self.los_cut * self.lot * 100
                    gain = loss_cut if loss <= loss_cut else gain
                    undetermined_assets = max(self.assets + gain, self.assets + loss_cut)

                    confirm = gain == loss_cut
                    buy = old_action != action and action != 2
                    if confirm or buy:
                        self.assets = undetermined_assets

                        if self.assets > self.minimum_required_margin * 2:
                            if action == 2:
                                self.positions = None
                                old_action = None
                            else:
                                self.minimum_required_margin = trend * self.pip_cost / self.leverage
                                lot = self.assets * self.available_assets_rate / self.minimum_required_margin * self.min_lots
                                self.lot = int(lot * 10 ** 2) / (10 ** 2)
                                self.positions = trend + self.spread if action == 0 else trend - self.spread
                                self.tp = min(abs(self.max_los_cut), atr * self.pip_cost)
                                self.los_cut = max(self.max_los_cut, -atr * self.pip_cost)
                        else:
                            self.positions = None


                    old_action = action if action != 2 else old_action

                elif self.assets > self.minimum_required_margin * 2:
                    old_action = None

            self.total_gain.append(undetermined_assets)
            try:
                if self.total_gain[-1] == self.total_gain[-2]:
                    self.total_gain[-1] = undetermined_assets - undetermined_assets * 0.1
            except:
                pass

            self.losses.append(self.los_cut)

            gain = 0
            self.growth_rate.append(np.log(undetermined_assets / self.initial_assets) * 100)
        self.assets = undetermined_assets


class Reward2(Reward):
    def reward(self, trend, high, low, action, leverage, atr, scale_atr):
        self.reset()
        lot, positions, tp, lc = np.array([]), np.array([]), np.array([]), np.array([])
        self.total_gain = []

        undetermined_assets = available_assets = self.assets
        self.minimum_required_margin = trend[0] * self.pip_cost / self.leverage
        old_action = None
        sum_gain = 0

        for action, leverage, trend, idx, atr, scale_atr in zip(action, leverage, trend,
                                                                range(len(high)), atr,
                                                                scale_atr):
            if old_action == None and not positions.tolist() and available_assets > self.minimum_required_margin and action != 2:
                self.minimum_required_margin = (trend * self.pip_cost / self.leverage)[0]
                l = (self.assets - self.minimum_required_margin * sum(lot) * 100) * \
                    self.available_assets_rate / self.minimum_required_margin * self.min_lots
                l += l * leverage
                lot = np.append(lot, int(l * 10 ** 2) / (10 ** 2))
                available_assets = undetermined_assets - self.minimum_required_margin * sum(lot) * 100

                positions = np.append(positions, trend + self.spread if action == 0 else trend - self.spread)
                tp = np.append(tp, min(abs(self.max_los_cut * 100), atr * self.pip_cost * 100) * lot[-1])
                lc = np.append(lc, max(self.max_los_cut * 100, -atr * self.pip_cost * 100) * lot[-1])

                old_action = action

            elif old_action != None and positions.tolist():
                if old_action == 0:
                    gain = (trend - positions) * self.pip_cost * lot * 100
                    loss = (low[idx - 1] - positions) * self.pip_cost * lot * 100
                else:
                    gain = (positions - trend) * self.pip_cost * lot * 100
                    loss = (positions - high[idx - 1]) * self.pip_cost * lot * 100

                loss = np.minimum(gain, loss)
                gain = np.where(loss <= lc, lc, gain)
                sum_gain = sum(gain)

                d = []
                for g in range(len(gain)):
                    if gain[g] == lc[g] or gain[g] >= tp[g]:
                        self.assets += float(gain[g])
                        d.append(g)
                if d:
                    gain = np.delete(gain, d)
                    lot = np.delete(lot, d)
                    positions = np.delete(positions, d)
                    lc = np.delete(lc, d)
                    tp = np.delete(tp, d)

                undetermined_assets = self.assets + float(sum(gain))
                available_assets = undetermined_assets - self.minimum_required_margin * int(sum(lot)) * 100
                t = (old_action != action and action != 2) or available_assets > self.minimum_required_margin
                t = t[0] if type(t) == type(np.array([])) else t
                if t:
                    if old_action != action and action != 2:
                        self.assets = undetermined_assets
                        available_assets = self.assets
                        lot, positions, tp, lc = np.array([]), np.array([]), np.array([]), np.array([])
                        old_action = None

                    if action != 2:
                        a = self.assets if self.assets < undetermined_assets else undetermined_assets
                        available_assets = a - self.minimum_required_margin * int(sum(lot)) * 100
                        self.minimum_required_margin = (trend * self.pip_cost / self.leverage)[0]
                        if available_assets > self.minimum_required_margin:
                            l = available_assets * self.available_assets_rate / self.minimum_required_margin * self.min_lots
                            l += l * leverage
                            lot = np.append(lot, int(l * 10 ** 2) / (10 ** 2))
                            available_assets = undetermined_assets - self.minimum_required_margin * sum(lot) * 100

                            positions = np.append(positions, trend + self.spread if action == 0 else trend - self.spread)
                            tp = np.append(tp, min(abs(self.max_los_cut), atr * self.pip_cost) * lot[-1])
                            lc = np.append(lc, max(self.max_los_cut, -atr * self.pip_cost) * lot[-1])
                            old_action = action

                    elif action == 2:
                        if not positions.tolist() and available_assets > self.minimum_required_margin:
                            lot, positions, tp, lc = np.array([]), np.array([]), np.array([]), np.array([])
                            available_assets = self.assets
                            old_action = None

            if available_assets > self.minimum_required_margin and not positions.tolist():
                self.assets = undetermined_assets
                lot, positions, tp, lc = np.array([]), np.array([]), np.array([]), np.array([])
                old_action = None

            self.total_gain.append(undetermined_assets)
            sum_gain = 0

            self.growth_rate.append(np.log(undetermined_assets / self.initial_assets) * 100)

        self.assets = undetermined_assets
        # self.total_gain = np.clip(self.total_gain, 1, max(self.total_gain))

