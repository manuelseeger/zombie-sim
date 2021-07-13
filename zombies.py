import numpy as np
from numpy.core.fromnumeric import shape
#from gym import Env, spaces
from numpy.linalg import norm
from functools import reduce
from scenarios import scenarios
#from stable_baselines.common.env_checker import check_env
import time
import math

class CodevVsZombies(object):
    '''Code vs Zombies as an OpenAI gym'''
    metadata = {'render.modes': ['human']}
    verbose = False
    scores = []
    def __init__(self, A, humans, zombies):
        super(CodevVsZombies, self).__init__()
        self.zombies_init = np.copy(zombies)
        self.humans_init = np.copy(humans)
        self.A_init = np.copy(A)

        #self.action_space = spaces.Box(low=np.array([0,0]), high=np.array([16000, 9000]),
        #                                shape=(2,), dtype=np.float32)
        #self.observation_space = spaces.Tuple(
        #    spaces.Box(low=np.array([0,0]), high=np.array([16000, 9000]),
        #                                shape=(2,), dtype=np.float32),
        #    spaces.Box(low=0, high=16000, dtype=np.float32),
        #    spaces.Box(low=0, high=16000, dtype=np.float32)
        #)


    def step(self, action):
        done = False

        # 1 move zombies
        self.zombies[:,:2] = np.array([self.move_Z(z[:2], self.find_target(z)) for z in self.zombies])

        # 2 move Ash
        self.move_A(action)

        # 3 kill zombies
        score = self.kill()

        # 4 zombies eat
        self.eat()

        self.determine_targets()

        done = self.is_done()

        # if all humans are dead give a large negative reward equal to all rewards given so far
        if done and self.humans.shape[0] == 0:
            score += sum(self.scores) * -1

        return (self.A, self.humans, self.zombies), score, done, []


    def eat(self):
        eat_mask = np.array([self.is_food(h) for h in self.humans])
        self.humans = self.humans[~eat_mask]


    def is_food(self, H):
        for Z in self.zombies:
            if (Z[:2] == H).all():
                return True
        return False


    def is_done(self):
        done = self.humans.shape[0] == 0 or self.zombies.shape[0] == 0
        return done


    def kill(self):
        kill_mask = np.array([self.in_range(z[:2]) for z in self.zombies])
        kills = self.zombies[kill_mask]
        score = self.score(kills)
        self.zombies = self.zombies[~kill_mask]
        self.scores.append(score)
        return score


    def score(self, kills):
        if kills.shape[0] > 0:
            humans_alive = self.humans.shape[0]
            base_score = (humans_alive ** 2) * 10
            fibs = self.fibonacci(kills.shape[0])
            return np.sum(fibs * base_score)
        else:
            return 0


    def in_range(self, Z):
        return norm(self.A - Z) < 2000


    def fibonacci(self, n):
        n = n + 2
        return np.array(reduce(lambda x, _: x + [x[-2] + x[-1]], [0] * (n-2), [0, 1])[2:])


    def find_target(self, Z): 
        if np.isnan(Z[2]):
            return self.A
        else:
            return self.humans[int(Z[2])]


    def determine_targets(self):
        for iz in range(self.zombies.shape[0]):
            Z = self.zombies[iz,:2]
            d = norm(Z-self.A)
            target = np.nan
            for ih in range(self.humans.shape[0]):
                H = self.humans[ih]
                d_ = norm(Z-H)
                if d_ < d: 
                    d = d_
                    target = ih
            self.zombies[iz, 2] = target


    def move_Z(self, Z, T): 
        if norm(T - Z) < 400:
            pos = T
        else: 
            v = (T - Z) / norm(T - Z)
            pos = np.floor(Z + (v * 400))
        if self.verbose:
            print('Zombie {} moves to {}'.format(Z, pos))
        return pos


    def move_A(self, A):
        if norm(A - self.A) < 1000:
            self.A = A
        else:
            v = (A - self.A) / norm(A - self.A)
            self.A = np.floor(self.A + (v * 1000))
        if self.verbose:
            print('Ash moves to {}'.format(self.A))
        return self.A


    def render(self):
        pass


    def reset(self):
        self.scores = []
        self.A = self.A_init
        self.humans = self.humans_init
        targets_init = np.empty((self.zombies_init.shape[0], 1))
        targets_init[:] = np.nan
        self.zombies = np.concatenate((self.zombies_init, targets_init), axis=1)
        ids_init = np.empty((self.zombies_init.shape[0], 1))
        ids_init[:] = np.arange(self.zombies_init.shape[0]).reshape((self.zombies_init.shape[0],1))
        self.zombies = np.concatenate((self.zombies, ids_init), axis=1)
        self.determine_targets()
        state = np.ndarray(shape=(3,), dtype=object)
        state[:] = [self.A, self.humans, self.zombies]
        return state


class Agent:
    history = []
    policy = []
    def act(self, state, p=0):
        action = np.array([
            np.random.randint(0,16000),
            np.random.randint(0,9000)
        ])
        self.history.append(action)
        return np.clip(action, a_min=[0,0], a_max=[16000, 9000])


    def reset(self):
        self.history = []


    def predict(self, state):
        return np.clip(self.policy.pop(), a_min=[0,0], a_max=[16000, 9000])


class TargetRandomZombieAgent(Agent):
    def act(self, state, p=0):
        action = state[2][np.random.randint(state[2].shape[0])][:2]
        self.history.append(action)
        return np.clip(action, a_min=[0,0], a_max=[16000, 9000])


class TargetZombiesTargettingHumansAgent(Agent):
    def act(self, state, p=0):
        zombies = state[2]
        ash_indeces = np.where(np.isnan(zombies))[0]
        mask = np.ones(shape=zombies.shape, dtype=bool)
        mask[ash_indeces] = False
        zombies_targetting_humans = zombies[mask]
        zombies_targetting_humans = zombies_targetting_humans.reshape(
            (int(zombies_targetting_humans.shape[0]/zombies.shape[1])),
            zombies.shape[1])
        if zombies_targetting_humans.shape[0] == 0:
            action = zombies[np.random.randint(zombies.shape[0])][:2]
        else:
            action = zombies_targetting_humans[np.random.randint(zombies_targetting_humans.shape[0])][:2]
        self.history.append(action)
        return np.clip(action, a_min=[0,0], a_max=[16000, 9000])


class TargetZombiesTargettingHumansLearningAgent(Agent):
    props = None
    # how much should targetting decisions impact future targetting
    TAU = 2
    # how much should initial setup influence initial targetting
    SIGMA = 3
    # how much randomness at the beginning
    BETA = 2
    e = 1
    def act(self, state, p=0):
        '''Target a random zombie which is targetting humans,
           with the following additions: 
            - Increase the probability of a zombie becoming the target in the next 
              round with each episodes the zombie has been targetted. 
              This is meant to encourage following it's initial choice and not vascillate 
              between targets
            - Initialize the probabilities with the distance of the zombies to their 
              targets, so that zombies closer to humans are more likely to be targetted
        '''
        zombies = state[2]
        if self.e == 1:
            self.props = np.empty((zombies.shape[0], 2))
            self.props[:,0] = zombies[:,3]
            self.props[:,1] = [norm(z[:2] - state[1][int(z[2])]) if not np.isnan(z[2]) else 10000 for z in zombies]
            self.props[:,1] = np.max(self.props[:,1])+1000 - self.props[:,1]
            self.props[:,1] = self.props[:,1]**self.SIGMA
            self.props[:,1] = self.props[:,1] / np.sum(self.props[:,1])

        ash_indeces = np.where(np.isnan(zombies))[0]
        mask = np.ones(shape=zombies.shape, dtype=bool)
        mask[ash_indeces] = False
        zombies_targetting_humans = zombies[mask]
        zombies_targetting_humans = zombies_targetting_humans.reshape(
            (int(zombies_targetting_humans.shape[0] / zombies.shape[1])),
            zombies.shape[1])

        if self.BETA > self.e:
            action = np.array([
                np.random.randint(0,16000),
                np.random.randint(0,9000)
            ])
        elif zombies_targetting_humans.shape[0] == 0:
            p_mask = np.isin(self.props[:,0], zombies[:,3])
            p = self.props[p_mask,1] / np.sum(self.props[p_mask,1])
            choice = np.random.choice(np.arange(zombies.shape[0]), p=p)
            action = zombies[choice,:2]
            self.props[int(zombies[choice,3]),1] += self.TAU * (1 + 1/self.e)
        else:
            p_mask = np.isin(self.props[:,0], zombies_targetting_humans[:,3])
            p = self.props[p_mask,1] / np.sum(self.props[p_mask,1])
            choice = np.random.choice(np.arange(zombies_targetting_humans.shape[0]), p=p)
            Z = zombies_targetting_humans[choice]
            H = state[1][int(Z[2])]
            action = Z[:2] + ((H-Z[:2]) / norm(H-Z[:2])) * 400
            self.props[int(zombies_targetting_humans[choice,3]),1] += self.TAU * (1 + 1/self.e)
        self.history.append(action)
        self.e += 1
        return np.clip(action, a_min=[0,0], a_max=[16000, 9000])


def sim():

    #A, humans, zombies = scenarios['2 zombies']
    #A, humans, zombies = scenarios['3vs3']
    #A, humans, zombies = scenarios['Rescue']
    #A, humans, zombies = scenarios['SplitSecond']
    A, humans, zombies = scenarios['Rectangle']

    env = CodevVsZombies(A, humans, zombies)
    
    ash = Agent()
    ash = TargetRandomZombieAgent()
    ash = TargetZombiesTargettingHumansAgent()
    ash = TargetZombiesTargettingHumansLearningAgent()

    start_time = time.perf_counter()
    complexity = humans.shape[0] + zombies.shape[0]
    max_episodes = 60 - int(math.sqrt( 100 + complexity**2 ))
    training = []
    max_reward = 0
    for e in range(max_episodes):
        state = env.reset()
        ash.reset()

        done = False
        total_reward = 0
        reward = 0
        while not done:
            action = ash.act(state)
            state, reward, done, _ = env.step(action)
            A, humans, zombies = state
            total_reward += reward

        if e == 0:
            max_reward = total_reward
            training = ash.history
        else:
            if total_reward > max_reward:
                max_reward = total_reward
                training = ash.history
        print('Episode {}, reward {}'.format(e, total_reward))

    end_time = time.perf_counter()
    print(f"Execution Time : {end_time - start_time:0.6f}" )
    print('Best result {}'.format(max_reward))

    ash.policy = training
    ash.policy.reverse()

    state = env.reset()
    total_reward = 0
    reward = 0
    done = False
    i = 1
    while not done:
        action = ash.predict(state)
        state, reward, done, _ = env.step(action)
        A, humans, zombies = state
        total_reward += reward
        print('Round {}, Humans {}, Zombies {}, Score {}'.format(i, humans.shape[0], zombies.shape[0], total_reward))
        i += 1

    
if __name__ == '__main__':
    sim()