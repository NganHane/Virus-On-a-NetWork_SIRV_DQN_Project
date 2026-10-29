# Virus On a Network – SIRV with Deep Q-Network (DQN)

## Introduction

This project explores the integration of Reinforcement Learning (RL) with Agent-Based Modeling (ABM) using Mesa, an open-source Python framework.

The project extends the Virus on a Network model by introducing a vaccination mechanism and integrates a Deep Q-Network (DQN) agent to learn intervention policies for epidemic control.

## Objectives

- Explore the integration of Mesa and Reinforcement Learning.
- Extend the original SIR epidemic model to SIRV (Susceptible–Infected–Recovered–Vaccinated).
- Apply DQN to learn intervention strategies in an epidemic simulation environment.
- Evaluate the learned policy through simulation and comparison with a baseline scenario.

## Technologies

- Python
- Mesa
- Reinforcement Learning
- Deep Q-Network (DQN)
- Stable-Baselines3
- NetworkX
- Gymnasium

## Model Overview

### SIRV Epidemic Model

The epidemic model includes four agent states:

- **S (Susceptible):** Agents who are susceptible to infection.
- **I (Infected):** Agents who are currently infected.
- **R (Recovered):** Agents who have recovered from infection.
- **V (Vaccinated):** Agents who have received vaccination and developed protection.

### Reinforcement Learning

A DQN agent interacts with the epidemic simulation environment to learn intervention policies.

The agent selects among three intervention actions:

- Open
- Partial intervention
- Full intervention

The learning process considers the trade-off between reducing infections and minimizing intervention costs.

## Project Structure

```text
Virus_On_a_Network_SIRV_DQN_Project/
Virus_On_a_Network_SIRV_DQN_Project/
├── agent.py
├── env.py
├── model.py
├── train_dqn.py
├── evaluate_and_plot.py
├── evaluate_baseline.py
├── saved_dqn/
│   └── DQN_1/
│       ├── dqn_sirv_model.zip
│       └── events.out.tfevents...
├── README.md
└── report.pdf

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/Virus_On_a_Network_SIRV_DQN_Project.git
cd Virus_On_a_Network_SIRV_DQN_Project
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate the environment:

Windows:

```bash
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install mesa networkx stable-baselines3 gymnasium matplotlib numpy
```

## Usage

### 1. Train the DQN Agent

Run the training script:

```bash
python train_dqn.py
```

The trained model is saved for subsequent evaluation.

### 2. Evaluate the DQN Agent

After training, run:

```bash
python evaluate_and_plot.py
```

This script evaluates the trained policy and generates simulation results and visualizations.

### 3. Run the Baseline Simulation

To evaluate the epidemic model without intervention, run:

```bash
python evaluate_baseline.py
```

### Notes

- Run the training script before evaluating the DQN agent if a trained model is not available.
- Ensure that the required dependencies are installed.
- Check the configuration and output paths in the scripts before running.

## Results

The project evaluates the learned intervention policy through epidemic simulation and compares it with a baseline scenario without intervention.

The evaluation focuses on:

- Infection dynamics over time.
- The effectiveness of intervention policies.
- Vaccination and protection dynamics.
- The trade-off between epidemic control and intervention costs.

## Author

**Lê Ngân Hà**
Software Engineering Graduate  
Interested in Artificial Intelligence, Machine Learning and Reinforcement Learning.
Github: [https://github.com/NganHane]