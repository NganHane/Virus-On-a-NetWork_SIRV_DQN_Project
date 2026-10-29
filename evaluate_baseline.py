import matplotlib.pyplot as plt
from model import EpidemicModel

def run_baseline(model_params, max_days=60):
    model = EpidemicModel(**model_params)

    I_hist, V_hist = [], []

    for day in range(max_days):
        model.step(action=0)   # KHÔNG lockdown

        I = sum(1 for a in model.agents.values() if a.status == 'I')
        V = sum(1 for a in model.agents.values()
                if a.protected or a.status == 'V')

        I_hist.append(I)
        V_hist.append(V)

        if I == 0:
            break

    return I_hist, V_hist


def plot_baseline(I, V):
    plt.figure(figsize=(8,4))
    plt.plot(I, label='I (active)')
    plt.plot(V, label='V (protected)')
    plt.xlabel('days')
    plt.ylabel('agents')
    plt.title('Baseline – No RL (No lockdown)')
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    model_params = {
        'N': 1000,
        'initial_infected': 10,
        'p_transmission': 0.04,
        'infectious_period': 14,
        'daily_capacity_fraction': 0.02,
        'post_vacc_period_days': 12,
        'post_vacc_infection_multiplier': 0.2,
    }

    I, V = run_baseline(model_params)
    plot_baseline(I, V)
