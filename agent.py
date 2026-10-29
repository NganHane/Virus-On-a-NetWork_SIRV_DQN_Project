from mesa import Agent

class PersonAgent(Agent):
    def __init__(self, unique_id, model, node):
        super().__init__(unique_id, model)
        self.unique_id = unique_id
        self.node = node
        self.status = 'S'   # S, I, R, V
        self.days_infected = 0

        # vaccination fields
        self.vaccinated = False
        self.vaccine_day = None
        self.protected = False

    def __repr__(self):
        return f"Agent({self.unique_id}, node={self.node}, status={self.status})"