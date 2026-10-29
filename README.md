# Virus On a NetWork\_SIRV\_DQN\_Project

Epidemic simulation using Mesa and Deep Q-Network (DQN)


\## Introduction



This project explores the integration of Reinforcement Learning (RL) with Agent-Based Modeling (ABM) using Mesa, an open-source Python framework.



The project extends the Virus on a Network model by introducing a vaccination mechanism and integrates a Deep Q-Network (DQN) agent to learn intervention policies for epidemic control.



\## Objectives



\- Explore the integration of Mesa and Reinforcement Learning.

\- Extend the original SIR epidemic model to SIRV (Susceptible–Infected–Recovered–Vaccinated).

\- Apply DQN to learn intervention strategies in an epidemic simulation environment.

\- Evaluate the learned policy through simulation and comparison with a baseline scenario.



\## Technologies



\- Python

\- Mesa

\- Reinforcement Learning

\- Deep Q-Network (DQN)

\- Stable-Baselines3

\- NetworkX

\- Gymnasium



\## Model Overview



\### SIRV Epidemic Model



The epidemic model includes four agent states:



\- \*\*S (Susceptible):\*\* Agents who are susceptible to infection.

\- \*\*I (Infected):\*\* Agents who are currently infected.

\- \*\*R (Recovered):\*\* Agents who have recovered from infection.

\- \*\*V (Vaccinated):\*\* Agents who have received vaccination and developed protection.



\### Reinforcement Learning



A DQN agent interacts with the epidemic simulation environment to learn intervention policies.



The agent selects among three intervention actions:



\- Open

\- Partial intervention

\- Full intervention



The learning process considers the trade-off between reducing infections and minimizing intervention costs.



\## Project Structure



```text

Virus\_On\_a\_Network\_SIRV\_DQN\_Project/

├── \[source code files]

├── \[model and environment files]

├── \[training and evaluation scripts]

├── \[visualization files]

├── README.md

└── report.pdf

```



\## Installation



Install the required dependencies:



```bash

pip install mesa networkx stable-baselines3 gymnasium matplotlib numpy

```



\## Usage



\### Train the DQN Agent



```bash

python train\_dqn.py

```



\### Evaluate the Trained Agent



```bash

python evaluate\_and\_plot.py

```



\*Note: Update the commands according to the actual filenames and configuration in the project.\*



\## Results



The project evaluates the learned intervention policy through epidemic simulation and compares it with a baseline scenario without intervention.



The evaluation focuses on:



\- Infection dynamics over time.

\- The effectiveness of intervention policies.

\- Vaccination and protection dynamics.

\- The trade-off between epidemic control and intervention costs.



\## Report



The project report is available in \[`report.pdf`](report.pdf).



\## Author



Software Engineering Graduate  

Interested in Artificial Intelligence, Machine Learning, and Reinforcement Learning.



