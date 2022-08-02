from logic.stakeholder import Stakeholder

class NonMyopicStakeholder(Stakeholder):
    def __init__(self, unique_id, model, stake, cost, strategy=None):
        super().__init__(unique_id=unique_id, model=model, stake=stake, cost=cost, is_myopic=False, is_abstainer=False, strategy=strategy)

class MyopicStakeholder(Stakeholder):
    def __init__(self, unique_id, model, stake, cost, strategy=None):
        super().__init__(unique_id=unique_id, model=model, stake=stake, cost=cost, is_myopic=True, is_abstainer=False, strategy=strategy)

class Abstainer(Stakeholder):
    def __init__(self, unique_id, model, stake, cost, strategy=None):
        super().__init__(unique_id=unique_id, model=model, stake=stake, cost=cost, is_myopic=False, is_abstainer=True, strategy=strategy)


profile_mapping = {
    0: NonMyopicStakeholder,
    1: MyopicStakeholder,
    2: Abstainer
}