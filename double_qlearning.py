import random
import numpy as np 

class DoubleQLearningAgent(object):
    def __init__(self, env) -> None:
        super().__init__()

        self.e = 0.1
        self.alpha = 0.1
        self.gamma = 1
        self.num_episodes = 300
        self.env = env
        self.init_state = env.start_state
        self.q1_sa = {x: 0 for x in env.state_transitions}
        self.q2_sa = {x: 0 for x in env.state_transitions}

        #print(self.q_sa)
    
    def reset_policy(self):
        self.q1_sa = {x: 0 for x in self.env.state_transitions}
        self.q2_sa = {x: 0 for x in self.env.state_transitions}
        
    def soft_policy(self, state):
        possible_actions = self.env.possible_actions[state]
        prob_explorative_action = self.e / len(possible_actions)
        prob_greedy_action = 1 - self.e + prob_explorative_action

        q1_values = []
        q2_values = []
        corresponding_actions = []
        for action in possible_actions:
            if (state, action) in self.q1_sa:
                q1_values.append(self.q1_sa[(state, action)])
                q2_values.append(self.q2_sa[(state, action)])
                corresponding_actions.append(action)
        
        q_values = np.array(q1_values) + np.array(q2_values)
        # check if all q values are the same, if so take a random action
        if all(x == q_values[0] for x in q_values):
            return random.choice(possible_actions)
        
        greedy_action = corresponding_actions[np.argmax(q_values)]
        non_greedy_actions = list(set(possible_actions) - set([greedy_action]))

        action = np.random.choice([greedy_action] + non_greedy_actions, p=[prob_greedy_action] + [prob_explorative_action for i in range(len(non_greedy_actions))])
        return action

    def generate_episode(self):
        episode = []
        state = self.init_state
        while True:
            action = self.soft_policy(state) # e-greedy approach
            reward, next_state, done = self.env.step(state, action)
            episode.append((state, action, reward, next_state))
            state = next_state
            if done:        
                break
        
        return episode  

    def update_policy(self):
        n_iters = 100
        final_left_array = []
        final_right_array = []

        for _ in range(n_iters):
            left_actions_count = []
            right_actions_count = []
            self.reset_policy()

            for i in range(self.num_episodes):
                left = 0
                right = 0

                # generate an episode using the policy defined above
                for s, a, r, s_p in self.generate_episode():
                    # use a soft policy as our behaviour policy - e-greedy
                    if (s, a) == (self.init_state, 'left'):
                        left += 1
                    elif (s, a) == (self.init_state, 'right'):
                        right += 1
                    
                    possible_future_actions = self.env.possible_actions[s_p]
                    if np.random.rand() <= 0.5:
                        q1_values = [(self.q1_sa[(s_p, action)], action) for action in possible_future_actions if (s_p, action) in self.q1_sa]
                        max_q_values = self.q2_sa[(s_p, max(q1_values)[1])] if len(q1_values) > 0 else 0
                        # update q function using a different target policy
                        self.q1_sa[(s, a)] += self.alpha * (r + self.gamma * max_q_values - self.q1_sa[(s, a)])
                    else:
                        q2_values = [(self.q2_sa[(s_p, action)], action) for action in possible_future_actions if (s_p, action) in self.q2_sa]
                        max_q_values = self.q1_sa[(s_p, max(q2_values)[1])] if len(q2_values) > 0 else 0
                        # update q function using a different target policy
                        self.q2_sa[(s, a)] += self.alpha * (r + self.gamma * max_q_values - self.q2_sa[(s, a)])

                left_actions_count.append(left)
                right_actions_count.append(right)

            left_actions_couny = np.array(left_actions_count)
            right_actions_count = np.array(right_actions_count)

            final_left_array.append(left_actions_count)
            final_right_array.append(right_actions_count)
        
        final_left_array = np.array(final_left_array)
        final_right_array = np.array(final_right_array)

        out = final_left_array.sum(axis=0) / (final_left_array.sum(axis=0) + final_right_array.sum(axis=0)) * 100

        return out

