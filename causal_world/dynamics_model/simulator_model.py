from stable_baselines.common.vec_env import SubprocVecEnv
import numpy as np
from sdtw import SoftDTW
from sdtw.barycenter import sdtw_barycenter
from sdtw.distance import SquaredEuclidean
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler
from tslearn.metrics import soft_dtw # https://github.com/tslearn-team/tslearn
from tslearn.clustering import TimeSeriesKMeans, silhouette_score

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

    def __init__(self, 
                _make_env_func, 
                parallel_agents, 
                num_environments, 
                use_z_only=False, 
                metric='softdtw', 
                reward='causal_curiosity',
                num_clusters=2):
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
        :param metric: (string): the distance or discrepency measure that can be used
                                      for kmeans clustering. It can be "softdtw" for 
                                      Soft-Dynamic Time Warping or "euclidian". 
        :param reward: (string): either "causal_curiosity" or "kmeans_score".
        :param num_clusters: (int): number of clusters.
        """
        assert (parallel_agents == num_environments) # TODO: make parallelism more flexible
        assert (float(num_environments / parallel_agents).is_integer())
        assert (float(num_environments / parallel_agents) <= 1.0)

        self.parallel_agents = parallel_agents
        self.num_environments = num_environments
        self.use_z_only = use_z_only
        self.metric = metric
        if self.metric == "softdtw":
            self.evaluate_trajectories = self.evaluate_trajectories_tslearn # TODO: decide which one to use. For now use tslearn or softdtw
        elif self.metric == "euclidian":
            self.evaluate_trajectories = self.evaluate_trajectories_euclidian
        self.reward = reward
        self.num_clusters = num_clusters
        self.envs = SubprocVecEnv(
            [_make_env_func() for i in range(self.num_environments)])
        return


    def causal_curiosity(self, D, cluster_memberships):

        cluster_indices = [np.where(cluster_memberships == i) for i in range(self.num_clusters)]

        C1 = np.amin(D[cluster_indices[0]][:,cluster_indices[1]])
        C2 = np.amax(D[cluster_indices[0]][:,cluster_indices[0]])
        C3 = np.amax(D[cluster_indices[1]][:,cluster_indices[1]])

        return C1 - C2 - C3

    def evaluate_trajectories_softdtw(self, action_sequences):
        num_clusters = 2
        observations = self.simulate_trajectories(action_sequences)
        
        if self.use_z_only:
            observations = np.squeeze(observations[:,:,:,34])

        rewards = np.zeros(action_sequences.shape[0])
        max_inner_iters = 100

        for i in range(rewards.size):

            #Lloyd's algorithm
            centroid_idc = np.random.choice(self.num_environments, self.num_clusters, replace=False)
            centroids = observations[i, centroid_idc].copy()
            cluster_memberships = np.zeros(self.num_environments)
            j = 0
            dist = 0
            previous_dist = 0
            while j < max_inner_iters:
                distances = np.zeros((self.num_environments,self.num_clusters))
                for env in range(self.num_environments):
                    for k in range(centroids.shape[0]):
                        if self.use_z_only:
                            D = SquaredEuclidean(centroids[k,np.newaxis], observations[i, env, np.newaxis])
                        else:
                            D = SquaredEuclidean(centroids[k], observations[i, env])
                        sdtw = SoftDTW(D)
                        distances[env,k] = sdtw.compute()

                previous_dist = dist
                dist = np.sum(np.amin(distances,axis=1))
                if dist == previous_dist: # converged
                    break

                cluster_memberships = np.argmin(distances, axis=1)

                for k in range(centroids.shape[0]):
                    cluster_observations = np.squeeze(observations[i,np.where(cluster_memberships == k)], axis=0)

                    barycenter_init = np.sum(cluster_observations, axis=0)/len(cluster_observations)
                    if self.use_z_only:
                        centroids[k] = np.squeeze(sdtw_barycenter(cluster_observations[:,:,np.newaxis], barycenter_init[:,np.newaxis]))
                    else:
                        centroids[k] = sdtw_barycenter(cluster_observations, barycenter_init)

                j += 1

            D = np.zeros([self.num_environments, self.num_environments])
            for env_index_1 in range(self.num_environments):
                for env_index_2 in range(self.num_environments):
                    if self.use_z_only:
                        SDTW_distance = SquaredEuclidean(observations[i, env_index_1, np.newaxis], observations[i, env_index_2, np.newaxis])
                    else:
                        SDTW_distance = SquaredEuclidean(observations[i, env_index_1], observations[i, env_index_2])
                    D[env_index_1][env_index_2] = SoftDTW(SDTW_distance).compute()

            rewards[i] = self.causal_curiosity(D, cluster_memberships)

        return rewards

    def evaluate_trajectories_euclidian(self, action_sequences):
        observations = self.simulate_trajectories(action_sequences)

        if self.use_z_only:
            observations = observations[:,:,:,34]

        rewards = np.zeros(action_sequences.shape[0])
        
        for i in range(rewards.size):
            observations_flattened = observations[i].reshape(self.num_environments, -1)
            kmeans = KMeans(n_clusters=self.num_clusters, random_state=0, n_jobs=-1)
            predictions = kmeans.fit_predict(observations_flattened)

            if self.reward == "kmeans_score":
                rewards[i] = kmeans.score(observations_flattened)
            elif self.reward == "causal_curiosity":
                D = pairwise_distances(observations_flattened)
                rewards[i] = self.causal_curiosity(D, predictions)
        
        return rewards

    def evaluate_trajectories_tslearn(self, action_sequences):
        observations = self.simulate_trajectories(action_sequences)

        if self.use_z_only:
            observations = observations[:,:,:,34]
        else:
            observations = observations[:,:,:,32:35]

        rewards = np.zeros(action_sequences.shape[0])
        
        for i in range(rewards.size):
            kmeans = TimeSeriesKMeans(n_clusters=self.num_clusters, metric=self.metric, max_iter=100,
                                      max_iter_barycenter=5,
                                      metric_params={"gamma": .5},
                                      random_state=None).fit(observations[i])
            predictions = kmeans.predict(observations[i])

            if len(np.unique(predictions)) == 1:
                rewards[i] = -0.99
            else:
                rewards[i] = silhouette_score(observations[i], predictions, metric="dtw")
                '''
                D = np.zeros([self.num_environments, self.num_environments])
                for env_index_1 in range(self.num_environments):
                    for env_index_2 in range(self.num_environments):
                        D[env_index_1][env_index_2] = soft_dtw(observations[i,env_index_1],observations[i,env_index_2])
                rewards[i] = self.causal_curiosity(D, predictions)
                '''
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
