from stable_baselines.common.vec_env import SubprocVecEnv
import numpy as np
from sdtw import SoftDTW
from sdtw.distance import SquaredEuclidean
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances

class SimulatorModel(object):

    def __init__(self, _make_env_func, parallel_agents):
        """
        This class instantiates a dynamics model based on the pybullet simulator
        (i.e: simulates exactly the result of the actions), it can be used
        for reward tuning and verifying tasks..etc

        :param _make_env_func: (func) a function if called it will return a gym
                                      environment.
        :param parallel_agents: (int) number of parallel agents to siumulate
                                      to evaluate the actions.
        """
        self.parallel_agents = parallel_agents
        self.envs = SubprocVecEnv(
            [_make_env_func() for i in range(self.parallel_agents)])
        return

    def evaluate_trajectories(self, action_sequences):
        """
        A function to be called to evaluate the action sequences and return
        the corresponding reward for each sequence.

        :param action_sequences: (nd.array) actions to be evaluated
                                            (number of sequences, horizon length)
        :return: (nd.array) sum of rewards for each action sequence.
        """
        horizon_length = action_sequences.shape[1]
        num_of_particles = action_sequences.shape[0]
        rewards = np.zeros([num_of_particles])
        assert ((float(num_of_particles) / self.parallel_agents).is_integer())
        for j in range(0, num_of_particles, self.parallel_agents):
            self.envs.reset()
            total_reward = np.zeros([self.parallel_agents])
            for k in range(horizon_length):
                actions = action_sequences[j:j + self.parallel_agents, k]
                task_observations, current_reward, done, info = \
                    self.envs.step(actions)
                total_reward += current_reward
            rewards[j:j + self.parallel_agents] = total_reward
        return rewards

    def end_sim(self):
        """
        Closes the environments that were used for simulation.
        :return:
        """
        self.envs.close()
        return


class ExperimentingSimulatorModel(object):

    def __init__(self, _make_env_func, parallel_agents, num_environments):
        """
        This class instantiates a dynamics model based on the pybullet simulator
        (i.e: simulates exactly the result of the actions), it can be used
        for reward tuning and verifying tasks..etc

        :param _make_env_func: (func) a function if called it will return a gym
                                      environment.
        :param parallel_agents: (int) number of parallel agents to siumulate
                                      to evaluate the actions.
        :param num_environments: (int) number of different environments to siumulate
                                      to evaluate the actions.
        """
        assert (parallel_agents == num_environments) # TODO: make parallelism more flexible
        assert (float(num_environments / parallel_agents).is_integer())
        assert (float(num_environments / parallel_agents) <= 1.0)

        self.parallel_agents = parallel_agents
        self.num_environments = num_environments
        self.envs = SubprocVecEnv(
            [_make_env_func() for i in range(self.num_environments)])
        return

    def evaluate_trajectories(self, action_sequences):
        num_clusters = 2
        observations = self.simulate_trajectories(action_sequences)
        rewards = np.zeros(action_sequences.shape[0])
        #distances = np.zeros((self.num_environments,num_clusters))
        
        max_inner_iters = 100

        for i in range(rewards.size):
            #Lloyd's algorithm
            centroid_idc = np.random.choice(self.num_environments, num_clusters, replace=False)
            centroids = observations[i, centroid_idc].copy()
            cluster_memberships = np.zeros(self.num_environments)
            j = 0
            while j < max_inner_iters:
                for env in range(self.num_environments):
                    distances = np.zeros(2)
                    for k in range(centroids.shape[0]):
                        D = SquaredEuclidean(centroids[k], observations[i, env])
                        sdtw = SoftDTW(D)
                        distances[k] = sdtw.compute()
                        #distances[k] = sdtw(centroids[k], observations[i, env])
                    print(distances)
                
                
                j += 1


        return rewards

    def _evaluate_trajectories(self, action_sequences):
        observations = self.simulate_trajectories(action_sequences)
        rewards = np.zeros(action_sequences.shape[0])
        objective = 'paper' # can be "kmeans_score"
        
        for i in range(rewards.size):
            kmeans = KMeans(n_clusters=2, random_state=0, n_jobs=-1)
            predictions = kmeans.fit_predict(observations[i].reshape(self.num_environments, -1))

            if objective == 'kmeans_score':
                # reward as the kmeans score for each action.
                rewards[i] = kmeans.score(observations[i].reshape(self.num_environments, -1))
            elif objective == 'paper':
                # reward as the objective from Causal Curiosity eq. 
                D = pairwise_distances(observations[i].reshape(self.num_environments, -1))
                C1 = -np.amax(D[np.where(predictions == 0)][:,np.where(predictions == 0)])
                C2 = -np.amax(D[np.where(predictions == 1)][:,np.where(predictions == 1)])
                C3 = np.amin(D[np.where(predictions == 0)][:,np.where(predictions == 1)])
                rewards[i] = C1 + C2 + C3

        
        return rewards


    def simulate_trajectories(self, action_sequences):
        """
        A function to be called to run the action sequences and return
        the corresponding observations for each sequence.

        :param action_sequences: (nd.array) actions to be evaluated
                                            (number of sequences, horizon length)
        :return: (nd.array) observations for each action sequence.
        """
        horizon_length = action_sequences.shape[1]
        num_of_particles = action_sequences.shape[0]
        observations = np.zeros([num_of_particles, 
                                 self.num_environments,
                                 horizon_length,
                                 self.envs.observation_space.shape[0]])
        for j in range(0, num_of_particles):
            for k in range(horizon_length):
                action = action_sequences[j, k]
                task_observations, _, _, _ = self.envs.step(action)
                observations[j, :, k] = task_observations
        return observations

    def end_sim(self):
        """
        Closes the environments that were used for simulation.
        :return:
        """
        self.envs.close()
        return
