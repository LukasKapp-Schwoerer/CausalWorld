from causal_world.task_generators.base_task import BaseTask
import numpy as np


class ExperimentingTaskGenerator(BaseTask):
    def __init__(self, variables_space='space_a',
                 fractional_reward_weight=0,
                 dense_reward_weights=np.array([0,0,0,0,0,0,0,0]),
                 activate_sparse_reward=False,
                 tool_block_mass=0.02,
                 joint_positions=None,
                 tool_block_position=np.array([0, 0, 0.0325]),
                 tool_block_orientation=np.array([0, 0, 0, 1]),
                 enhanced_observations=False):
        """
        This task generates a task without goal

        :param variables_space: (str) space to be used either 'space_a' or
                                      'space_b' or 'space_a_b'
        :param fractional_reward_weight: (float) weight multiplied by the
                                                fractional volumetric
                                                overlap in the reward.
        :param dense_reward_weights: (list float) specifies the reward weights
                                                  for all the other reward
                                                  terms calculated in the
                                                  calculate_dense_rewards
                                                  function.
        :param activate_sparse_reward: (bool) specified if you want to
                                              sparsify the reward by having
                                              +1 or 0 if the volumetric
                                              fraction overlap more than 90%.
        :param tool_block_mass: (float) specifies the blocks mass.
        :param joint_positions: (nd.array) specifies the joints position to start
                                            the episode with. None if the default
                                            to be used.
        :param tool_block_position: (nd.array) specifies the cartesian position
                                               of the tool block, x, y, z.
        :param tool_block_orientation: (nd.array) specifies the euler orientation
                                               of the tool block, yaw, roll, pitch.
        :param goal_height: (float) specifies the goal height that needs to be
                                    reached.
        :param enhanced_observations: (bool) specifies if the observations should 
                                             contain tool block mass and tool block 
                                             friction.
        """
        super().__init__(task_name="experimenting",
                         variables_space=variables_space,
                         fractional_reward_weight=fractional_reward_weight,
                         dense_reward_weights=dense_reward_weights,
                         activate_sparse_reward=activate_sparse_reward)
        self._task_robot_observation_keys = ["time_left_for_task",
                                             "joint_positions",
                                             "joint_velocities",
                                             "end_effector_positions"]
        self._task_params["tool_block_mass"] = tool_block_mass
        self._task_params["joint_positions"] = joint_positions
        self._task_params["tool_block_position"] = tool_block_position
        self._task_params["tool_block_orientation"] = tool_block_orientation
        self.previous_object_position = None
        self.previous_end_effector_positions = None
        self.previous_joint_velocities = None
        self.enhanced_observations = enhanced_observations

    def get_description(self):
        """

        :return: (str) returns the description of the task itself.
        """
        return "Task where there is no goal"

    def _set_up_stage_arena(self):
        """

        :return:
        """
        creation_dict = {
            'name': "tool_block",
            'shape': "cube",
            'initial_position': self._task_params["tool_block_position"],
            'initial_orientation': self._task_params["tool_block_orientation"],
            'mass': self._task_params["tool_block_mass"]
        }
        self._stage.add_rigid_general_object(**creation_dict)
        self._task_stage_observation_keys = [
            "tool_block_type", "tool_block_size",
            "tool_block_cartesian_position", "tool_block_orientation",
            "tool_block_linear_velocity", "tool_block_angular_velocity",
        ]
        if self.enhanced_observations:
            self._task_stage_observation_keys.extend(["tool_block_mass", 
                                                      "tool_block_friction"])
        return

    def _set_intervention_space_a(self):
        """

        :return:
        """
        super(ExperimentingTaskGenerator, self)._set_intervention_space_a()
        for visual_object in self._stage.get_visual_objects():
            self._intervention_space_a[visual_object]['cylindrical_position'][
                0][-1] \
                = 0.08
            self._intervention_space_a[visual_object]['cylindrical_position'][
                1][-1] \
                = 0.20
        return

    def _set_intervention_space_b(self):
        """

        :return:
        """
        super(ExperimentingTaskGenerator, self)._set_intervention_space_b()
        for visual_object in self._stage.get_visual_objects():
            self._intervention_space_b[visual_object]['cylindrical_position'][0][
                -1] \
                = 0.20
            self._intervention_space_b[visual_object]['cylindrical_position'][1][
                -1] \
                = 0.25
        return

    def get_reward(self):
        return 0

    def _update_task_state(self, update_task_info):
        """

        :param update_task_info:

        :return:
        """
        self.previous_end_effector_positions = \
            update_task_info['current_end_effector_positions']
        self.previous_object_position = \
            update_task_info['current_tool_block_position']
        self.previous_joint_velocities = \
            update_task_info['current_velocity']
        return

    def _set_task_state(self):
        """

        :return:
        """
        self.previous_end_effector_positions = \
            self._robot.get_latest_full_state()['end_effector_positions']
        self.previous_end_effector_positions = \
            self.previous_end_effector_positions.reshape(-1, 3)
        self.previous_object_position = \
            self._stage.get_object_state('tool_block', 'cartesian_position')
        self.previous_joint_velocities = \
            self._robot.get_latest_full_state()['velocities']
        return

    def _handle_contradictory_interventions(self, interventions_dict):
        """

        :param interventions_dict:

        :return:
        """
        # for example size on goal_or tool should be propagated to the other
        if 'goal_block' in interventions_dict:
            if 'size' in interventions_dict['goal_block']:
                if 'tool_block' not in interventions_dict:
                    interventions_dict['tool_block'] = dict()
                interventions_dict['tool_block']['size'] = \
                    interventions_dict['goal_block']['size']
            if 'cylindrical_position' in interventions_dict['goal_block']:
                interventions_dict['goal_block']['cylindrical_position'][0] = 0
                interventions_dict['goal_block']['cylindrical_position'][1] = 0
        elif 'tool_block' in interventions_dict:
            if 'size' in interventions_dict['tool_block']:
                if 'goal_block' not in interventions_dict:
                    interventions_dict['goal_block'] = dict()
                interventions_dict['goal_block']['size'] = \
                    interventions_dict['tool_block']['size']
        return interventions_dict

    def sample_new_goal(self, level=None):
        """
        Used to sample new goal from the corresponding shape families.

        :param level: (int) specifying the level - not used for now.

        :return: (dict) the corresponding interventions dict that could then
                       be applied to get a new sampled goal.
        """
        intervention_dict = dict()
        intervention_dict['goal_block'] = dict()
        if self._task_params['variables_space'] == 'space_a':
            intervention_space = self._intervention_space_a
        elif self._task_params['variables_space'] == 'space_b':
            intervention_space = self._intervention_space_b
        elif self._task_params['variables_space'] == 'space_a_b':
            intervention_space = self._intervention_space_a_b
        intervention_dict['goal_block']['cylindrical_position'] = \
            np.array([0, 0, np.random.uniform(intervention_space['goal_block']['cylindrical_position']
                              [0][-1],
                              intervention_space['goal_block']['cylindrical_position']
                              [1][-1])])
        return intervention_dict

    def _adjust_variable_spaces_after_intervention(self, interventions_dict):
        spaces = [self._intervention_space_a,
                  self._intervention_space_b,
                  self._intervention_space_a_b]
        if 'tool_block' in interventions_dict:
            if 'size' in interventions_dict['tool_block']:
                for variable_space in spaces:
                    variable_space['tool_block'][
                        'cylindrical_position'][0][
                        -1] = \
                        self._stage.get_object_state('tool_block', 'size')[
                            -1] / 2.0
                    variable_space['goal_block'][
                        'cylindrical_position'][0][
                        -1] = \
                        self._stage.get_object_state('goal_block', 'size')[
                            -1] / 2.0
        return

