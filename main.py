import matplotlib.pyplot as plt
import seaborn as sns

from environment import Environment
from qlearning import QLearningAgent
from double_qlearning import DoubleQLearningAgent

sns.set()
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.2)


env = Environment()
qlearning_agent = QLearningAgent(env)
dqlearning_agent = DoubleQLearningAgent(env)

print(qlearning_agent.soft_policy('A'))
print(qlearning_agent.generate_episode())
left_actions_ratio_a1 = qlearning_agent.update_policy()
left_actions_ratio_a2 = dqlearning_agent.update_policy()

# Prettier plot
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(
	range(len(left_actions_ratio_a1)),
	left_actions_ratio_a1,
	color="red",
	label="Q-Learning",
	linewidth=2.5
)
ax.plot(
	range(len(left_actions_ratio_a2)),
	left_actions_ratio_a2,
	color="green",
	label="Double Q-Learning",
	linewidth=2.5
)
ax.plot(
	range(len(left_actions_ratio_a1)),
	[5] * len(left_actions_ratio_a1),
	'--',
	color="black",
	label="Optimal",
	linewidth=2
)

ax.set_xlabel("Number of Episodes", fontsize=14)
ax.set_ylabel("% Left Actions from State A", fontsize=14)
ax.set_title("Left Actions Ratio from State A per Episode", fontsize=16, weight="bold")
ax.legend(loc="best", fontsize=12)
ax.grid(True, which='both', linestyle='--', linewidth=0.7, alpha=0.7)
plt.tight_layout()
plt.savefig("left_actions_ratio_a1.png", dpi=150)
plt.show()


