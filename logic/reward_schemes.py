TOTAL_EPOCH_REWARDS_R = 1


class RSS:
    """
    Parent class for the Reward Sharing Scheme that is used in the simulation.
    Determines the rewards that a pool receives.
    """

    def __init__(self, k, a0):
        self.k = k
        self.a0 = a0  # todo maybe rename to sth more general, e.g. pledge_influence?

    @property
    def k(self):
        return self._k

    @k.setter
    def k(self, k_value):
        if k_value == 0:
            raise ValueError('k parameter of reward scheme cannot be 0')
        self._k = int(k_value)
        # whenever k changes, the global saturation threshold also changes
        self.global_saturation_threshold = 1 / k_value

    def calculate_pool_reward(self, pool_pledge, pool_stake):
        raise NotImplementedError("RSS subclass must implement 'calculate_pool_reward' method.")

    def get_pool_saturation_threshold(self, pool_pledge):
        """
        By default, the saturation point of all pools is given by the global_saturation_threshold. However, some
        reward schemes may choose to have different saturation thresholds for different pools, depending on their pledge.
        @param pool_pledge: the pledge of the relevant pool
        @return: the saturation threshold of a pool with the given pledge
        """
        return self.global_saturation_threshold


class CardanoRSS(RSS):
    def __init__(self, k, a0):
        super().__init__(k=k, a0=a0)

    def calculate_pool_reward(self, pool_pledge, pool_stake):
        pledge_ = min(pool_pledge, self.global_saturation_threshold)
        stake_ = min(pool_stake, self.global_saturation_threshold)
        r = (TOTAL_EPOCH_REWARDS_R / (1 + self.a0)) * \
            (stake_ + (pledge_ * self.a0 * ((stake_ - pledge_ * (1 - stake_ / self.global_saturation_threshold))
                                            / self.global_saturation_threshold)))
        return r


class SimplifiedRSS(RSS):
    def __init__(self, k, a0):
        super().__init__(k=k, a0=a0)

    def calculate_pool_reward(self, pool_pledge, pool_stake):
        pledge_ = min(pool_pledge, self.global_saturation_threshold)
        stake_ = min(pool_stake, self.global_saturation_threshold)
        r = (TOTAL_EPOCH_REWARDS_R / (1 + self.a0)) * stake_ * \
            (1 + (self.a0 * pledge_ / self.global_saturation_threshold))
        return r


class FlatPledgeBenefitRSS(RSS):
    def __init__(self, k, a0):
        super().__init__(k=k, a0=a0)

    def calculate_pool_reward(self, pool_pledge, pool_stake):
        pledge_ = min(pool_pledge, self.global_saturation_threshold)
        stake_ = min(pool_stake, self.global_saturation_threshold)
        r = (TOTAL_EPOCH_REWARDS_R / (1 + self.a0)) * (stake_ + self.a0 * pledge_)
        return r


class CurvePledgeBenefitRSS(RSS):  # CIP-7
    def __init__(self, k, a0, crossover_factor, curve_root):
        super().__init__(k=k, a0=a0)
        self.crossover_factor = crossover_factor
        self.curve_root = curve_root

    def calculate_pool_reward(self, pool_pledge, pool_stake):
        crossover = self.global_saturation_threshold / self.crossover_factor
        pledge_factor = (pool_pledge ** (1 / self.curve_root)) * (
                    crossover ** ((self.curve_root - 1) / self.curve_root))
        pledge_ = min(pledge_factor, self.global_saturation_threshold)
        stake_ = min(pool_stake, self.global_saturation_threshold)
        r = (TOTAL_EPOCH_REWARDS_R / (1 + self.a0)) * \
            (stake_ + (pledge_ * self.a0 * ((stake_ - pledge_ * (1 - stake_ / self.global_saturation_threshold))
                                            / self.global_saturation_threshold)))
        return r


class CIP50RSS(RSS):
    """
    Reward scheme equivalent to that of CIP-50.
    Note that even though it's possible to use this reward scheme in the simulation, it might not give accurate results
    because our methodology was based on nuances of the original reward scheme that are not present in this one
    (e.g. the expectation of ending up with k pools). There is a separate branch for this reward scheme that uses a
    slightly different logic, which we believe is more compatible - please refer to that one for extensive
    experimentation concerning CIP-50.
    """

    def __init__(self, k, a0):
        super().__init__(k=k, a0=a0)

    def calculate_pool_reward(self, pool_pledge, pool_stake):
        pool_saturation_threshold = self.get_pool_saturation_threshold(pool_pledge)
        r = TOTAL_EPOCH_REWARDS_R * min(pool_stake, pool_saturation_threshold)
        return r

    def get_pool_saturation_threshold(self, pool_pledge):
        custom_saturation_threshold = self.a0 * pool_pledge
        return min(custom_saturation_threshold, self.global_saturation_threshold)


RSS_MAPPING = {
    0: CardanoRSS,
    1: SimplifiedRSS,
    2: FlatPledgeBenefitRSS,
    3: CurvePledgeBenefitRSS,
    4: CIP50RSS
}
