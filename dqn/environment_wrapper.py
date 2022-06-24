import gym

class EnvironmentWrapper():
    def __init__(self, env, skip_steps):
        self.env = env
        self.skip_steps = skip_steps

    def reset(self):
        state = self.env.reset()
        return state

    def step(self, action):
        total_reward = 0
        for i in range(self.skip_steps):
            state, reward, done = self.env.step(action)
            # self.env.env.viewer.window.dispatch_events()
            total_reward += reward
            if done:
                break

        return state, total_reward, done

