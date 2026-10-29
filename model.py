import random
import math
import numpy as np
import networkx as nx
from mesa import Model
from mesa.time import RandomActivation
from agent import PersonAgent

class EpidemicModel(Model):
    def __init__(self,
                 N=1000,
                 graph=None,
                 initial_infected=10,
                 p_transmission=0.08,
                 infectious_period=14,
                 daily_capacity_fraction=0.02,
                 post_vacc_period_days=12,
                 post_vacc_infection_multiplier=0.2,
                 restriction_costs=None,
                 A_thresh=0.02):
        super().__init__()
        self.num_agents = N
        self.step_count = 0

        # graph (if not provided, create BA graph)
        if graph is None:
            # create a graph with N_nodes smaller than agents so multiple agents per node
            n_nodes = max(50, N // 10)
            self.G = nx.barabasi_albert_graph(n_nodes, max(1, n_nodes // 50))
        else:
            self.G = graph

        # params
        self.initial_infected = initial_infected
        self.p_transmission = p_transmission
        self.infectious_period = infectious_period
        self.post_vacc_period_days = post_vacc_period_days
        self.post_vacc_infection_multiplier = post_vacc_infection_multiplier
        self.daily_vaccine_capacity = max(1, int(daily_capacity_fraction * N))
        self.vaccine_stock = 10 * N
        self.cum_vaccinated = 0

        # lockdown related
        # restriction: 0=open, 1=partial, 2=full lockdown
        self.restriction = 0
        # contact reduction factors per restriction (affect transmission or movement as you like)
        self.restriction_contact_reduction = {0:0.0, 1:0.5, 2:0.9}

        # reward params
        if restriction_costs is None:
            self.restriction_costs = {0:0.0, 1:0.4, 2:1.0}
        else:
            self.restriction_costs = restriction_costs
        self.alpha = 1.0
        self.beta = 0.3
        self.gamma = 0.05
        self.delta = 5.0
        self.A_thresh = A_thresh

        # scheduler and agents
        self.schedule = RandomActivation(self)
        self.node_occupants = {n: [] for n in self.G.nodes()}
        self.agents = {}

        # distribute agents roughly uniformly over nodes
        nodes = list(self.G.nodes())
        for i in range(self.num_agents):
            node = nodes[i % len(nodes)]
            a = PersonAgent(i, self, node)
            self.agents[i] = a
            self.schedule.add(a)
            self.node_occupants[node].append(i)

        # seed initial infections
        infected_ids = random.sample(list(self.agents.keys()), min(self.initial_infected, self.num_agents))
        for aid in infected_ids:
            self.agents[aid].status = 'I'
            self.agents[aid].days_infected = 0

        # stats
        self.new_cases_today = 0
        self.cum_infected = len(infected_ids)
        self.reward = 0.0

    #vaccination
    def vaccinate_batch(self, today):
        doses = min(self.daily_vaccine_capacity, self.vaccine_stock)
        if doses <= 0:
            self.v_administered_today = 0
            return 0

        susceptibles = [aid for aid,a in self.agents.items() if a.status == 'S' and not a.vaccinated]
        if not susceptibles:
            self.v_administered_today = 0
            return 0

        selected = random.sample(susceptibles, min(len(susceptibles), doses))
        for aid in selected:
            a = self.agents[aid]
            a.vaccinated = True
            a.vaccine_day = today
        self.vaccine_stock -= len(selected)
        self.cum_vaccinated += len(selected)
        self.v_administered_today = len(selected)
        return len(selected)

    #movement
    def move_agents_batch(self):
        # agents move to a random neighbor; contact reduction could scale movement frequency
        for aid, a in self.agents.items():
            # compute movement probability reduced by restriction
            reduction = self.restriction_contact_reduction[self.restriction]
            # with probability (1 - reduction) agent moves
            if random.random() > reduction:
                neighbors = list(self.G.neighbors(a.node))
                if neighbors:
                    new = random.choice(neighbors)
                    # update occupants lists
                    if aid in self.node_occupants[a.node]:
                        self.node_occupants[a.node].remove(aid)
                    self.node_occupants[new].append(aid)
                    a.node = new

    # infection (node-wise) 
    def infection_step_nodewise(self):
        today = self.step_count
        p_contact = self.p_transmission
        self.new_cases_today = 0

        for node, occ in self.node_occupants.items():
            if not occ:
                continue
            infected_ids = [aid for aid in occ if self.agents[aid].status == 'I']
            if not infected_ids:
                continue
            susceptible_ids = [aid for aid in occ if self.agents[aid].status == 'S']
            if not susceptible_ids:
                continue

            n_I = len(infected_ids)
            # probability at least one transmission event to a susceptible from these infecteds
            p_node = 1.0 - (1.0 - p_contact)**n_I

            for aid in susceptible_ids:
                a = self.agents[aid]
                if a.vaccinated and not a.protected:
                    days = today - a.vaccine_day
                    if days < self.post_vacc_period_days:
                        p_eff = p_node * self.post_vacc_infection_multiplier
                    else:
                        # fully protected now
                        a.status = 'V'
                        a.protected = True
                        # remove from susceptible list behavior (they are already set)
                        continue
                else:
                    p_eff = p_node

                if random.random() < p_eff:
                    a.status = 'I'
                    a.days_infected = 0
                    self.new_cases_today += 1
                    self.cum_infected += 1

    # progression
    def progress_disease_batch(self):
        for a in self.agents.values():
            if a.status == 'I':
                a.days_infected += 1
                if a.days_infected >= self.infectious_period:
                    a.status = 'R'

    # reward
    def compute_reward(self):
        I = sum(1 for a in self.agents.values() if a.status == 'I')
        A_t = I / float(self.num_agents)
        vacc_admin = getattr(self, 'v_administered_today', 0)

        r = -self.alpha * A_t - self.beta * self.restriction_costs[self.restriction] - self.gamma * (vacc_admin / float(self.num_agents))
        if A_t > self.A_thresh:
            r -= self.delta * (A_t - self.A_thresh)
        self.reward = float(r)
        return r

    # step flow
    def step(self, action=None):
        # action: 0,1,2 -> restriction level
        if action is not None:
            self.restriction = int(action)
        today = self.step_count

        # vaccinate
        self.vaccinate_batch(today)

        # move
        self.move_agents_batch()

        # infection
        self.infection_step_nodewise()

        # progression
        self.progress_disease_batch()

        # reward
        self.compute_reward()

        # increment step
        self.step_count += 1
        # scheduler step is not used for agent behavior here (we operate in batch), but keep for compatibility
        self.schedule.step()

    # observation helpers 
    def get_observation(self):
        I = sum(1 for a in self.agents.values() if a.status == 'I')
        V = sum(1 for a in self.agents.values() if a.protected or a.status == 'V')
        newc = getattr(self, 'new_cases_today', 0)
        obs = np.array([
            I / float(self.num_agents),
            newc / float(self.num_agents),
            V / float(self.num_agents),
            float(self.restriction) / 2.0,
        ], dtype=np.float32)
        return obs