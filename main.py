import matplotlib.pyplot as  plt 
import seaborn as sns 

from environment import Environment
from qlearning import QLearningAgent
from double_qlearning import DoubleQLearningAgent

sns.set()

env = Environment()
qlearning_agent = QLearningAgent(env)
dqlearning_agent = DoubleQLearningAgent(env)

print(qlearning_agent.soft_policy('A'))
print(qlearning_agent.generate_episode())
left_actions_ratio_a1 = qlearning_agent.update_policy()
left_actions_ratio_a2 = dqlearning_agent.update_policy()

fig, ax = plt.subplots()
ax.plot(range(len(left_actions_ratio_a1)), left_actions_ratio_a1, color="red", label="Q-Learning")
ax.plot(range(len(left_actions_ratio_a2)), left_actions_ratio_a2, color="green", label="Double Q-Learning")
ax.plot(range(len(left_actions_ratio_a1)), [5] * len(left_actions_ratio_a1), '--', color="black", label="Optimal")

ax.set_xlabel("Number of Episodes")
ax.set_ylabel("% Left Actions from State A")
ax.legend(loc="best")
plt.savefig("left_actions_ratio_a1.png")


