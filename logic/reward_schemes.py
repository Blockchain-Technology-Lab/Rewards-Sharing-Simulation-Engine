
TOTAL_EPOCH_REWARDS_R = 1

class RSS:
    """
    Class for the Reward Sharing Scheme that is used in the simulation.
    Determines the rewards that a pool receives.
    """
    def __init__(self, k, a0):
        self._k = k #check if k is 0 here or not? for division
        self.a0 = a0 #rethink name
        self.beta = 1 / k #todo do I need beta here? maybe rename to top_saturation_threshold or sth

    @property
    def k(self):
        return self._k

    @k.setter
    def k(self, k_value):
        self._k = int(k_value)
        # whenever k changes, beta also changes
        self.beta = 1 / k_value

    def calculate_pool_reward(self, pool_pledge, pool_stake):
        raise NotImplementedError("RSS subclass must implement 'calculate_pool_reward' method.")


class CardanoRSS(RSS):
    def __init__(self, k, a0):
        super().__init__(k=k, a0=a0)

    def calculate_pool_reward(self, pool_pledge, pool_stake):
        pledge_ = min(pool_pledge, self.beta)
        stake_ = min(pool_stake, self.beta)
        r = (TOTAL_EPOCH_REWARDS_R / (1 + self.a0)) * \
            (stake_ + (pledge_ * self.a0 * ((stake_ - pledge_ * (1 - stake_ / self.beta)) / self.beta)))
        return r

class SimplifiedRSS(RSS):
    def __init__(self, k, a0):
        super().__init__(k=k, a0=a0)

    def calculate_pool_reward(self, pool_pledge, pool_stake):
        pledge_ = min(pool_pledge, self.beta)
        stake_ = min(pool_stake, self.beta)
        r = (TOTAL_EPOCH_REWARDS_R / (1 + self.a0)) * stake_ * \
            (1 + (self.a0 * pledge_ / self.beta))
        return r

class FlatPledgeBenefitRSS(RSS):
    def __init__(self, k, a0):
        super().__init__(k=k, a0=a0)

    def calculate_pool_reward(self, pool_pledge, pool_stake):
        pledge_ = min(pool_pledge, self.beta)
        stake_ = min(pool_stake, self.beta)
        r = (TOTAL_EPOCH_REWARDS_R / (1 + self.a0)) * (stake_ + self.a0 * pledge_)
        return r


class CurvePledgeBenefitRSS(RSS): #CIP-7
    def __init__(self, k, a0, crossover_factor, curve_root):
        super().__init__(k=k, a0=a0)
        self.crossover_factor = crossover_factor
        self.curve_root = curve_root

    def calculate_pool_reward(self, pool_pledge, pool_stake):
        crossover = self.beta / self.crossover_factor
        pledge_factor = (pool_pledge ** (1 / self.curve_root)) * (crossover ** ((self.curve_root - 1) / self.curve_root))
        pledge_ = min(pledge_factor, self.beta)
        stake_ = min(pool_stake, self.beta)
        r = (TOTAL_EPOCH_REWARDS_R / (1 + self.a0)) * \
            (stake_ + (pledge_ * self.a0 * ((stake_ - pledge_ * (1 - stake_ / self.beta)) / self.beta)))
        return r

RSS_MAPPING = {
    0: CardanoRSS,
    1: SimplifiedRSS,
    2: FlatPledgeBenefitRSS,
    3: CurvePledgeBenefitRSS
}