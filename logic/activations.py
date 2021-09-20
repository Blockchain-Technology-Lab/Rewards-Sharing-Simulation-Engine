import random

from mesa.time import BaseScheduler


# todo this activation function may require changes to some logic, e.g. as to when the simulation terminates
class SemiSimultaneousActivation(BaseScheduler):
    """A scheduler to simulate the semi-simultaneous activation of some of the agents.
    A predetermined number z of agents are activated at the same time in each step.

    This scheduler requires that each agent have two methods: step and advance.
    step() activates the agent and stages any necessary changes, but does not
    apply them yet. advance() then applies the changes.

    """
    def __init__(self, model):
        super().__init__(model)
        self.z = 5

    def step(self) -> None:
        """Step z agents, then advance them."""
        agent_keys = random.choices(list(self._agents.keys()), k=self.z)
        for agent_key in agent_keys:
            self._agents[agent_key].step()
        for agent_key in agent_keys:
            self._agents[agent_key].advance()
        self.steps += 1
        self.time += 1
