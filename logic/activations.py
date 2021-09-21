import random

from mesa.time import BaseScheduler


class SemiSimultaneousActivation(BaseScheduler):
    """A scheduler to simulate the semi-simultaneous activation of some of the agents.
    A predetermined number z of agents are activated at the same time in each step.

    This scheduler requires that each agent have two methods: step and advance.
    step() activates the agent and stages any necessary changes, but does not
    apply them yet. advance() then applies the changes.

    """
    def __init__(self, model, simultaneous_moves=5, all_agents_move=True):
        super().__init__(model)
        self.simultaneous_moves = simultaneous_moves
        self.all_agents_move = all_agents_move

    def step(self) -> None:
        """
        Step a certain number of agents at a time, then advance them.
        if self.all_agents_move is True, then repeat until all agents have had the chance to make a move.
        """
        agent_keys = list(self._agents.keys())
        while len(agent_keys) > 0:
            k = self.simultaneous_moves if self.simultaneous_moves < len(agent_keys) else len(agent_keys)
            current_agent_keys = random.sample(agent_keys, k=k)
            for agent_key in current_agent_keys:
                self._agents[agent_key].step()
            for agent_key in current_agent_keys:
                self._agents[agent_key].advance()
            # Update keys to sample from
            agent_keys = [key for key in agent_keys if key not in current_agent_keys] if self.all_agents_move else []
        self.steps += 1
        self.time += 1
