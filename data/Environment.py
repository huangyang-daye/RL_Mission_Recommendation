import random
from data.Action import Action
from data.DataLoader import Data
import numpy as np
from typing import List

alpha = 0.2

class Environment:
    def __init__(self, reward_type='w'):
        self._data = Data()
        self._data.get_data()
        self._reward_type = reward_type # 'w': workers, 'r': requesters(linear), 'rn1' requesters(non-linear1), 'rn2' requesters(non-linear2)
        self.is_testing = False

    def is_done(self)->bool:
        return self._done

    def get_state(self)->np.ndarray:
        return self._state

    def reset(self):
        self._done = False
        self._index = 0
        self._state = self._data.get_state_array(self._index, self.is_testing)
        self._buffered_states = []
        pass

    def sample(self)-> Action:
        # randomly sample an action from the action space
        # for n workers, the action space is [0, n-1]
        return Action(random.randint(0, len(self._data.worker_quality) -1))

    def perform(self, action)->float:
        # perform an action and return the reward
        worker_id = self._data.get_worker_id_by_index(action.get())
        project_id = self._data.get_project_id_by_index(self._index, self.is_testing)
        ret = self._data.get_standard_reward(worker_id, project_id)
        self._buffered_states.append(self._state)
        if self._reward_type == 'r':
            ret = alpha * ret + (1.0 -alpha) * self._data.get_quality_reward(worker_id)
        elif self._reward_type == 'rn1':
            ret = self._data.get_quality_reward(worker_id)
        elif self._reward_type == 'rn2':
            ret = 1.0 - ((1.0-ret)*(1.0-self._data.get_quality_reward(worker_id)))
        self._index += 1
        if self._index >= self._data.get_project_length(self.is_testing):
            self._done = True
        else:
            self._state = self._data.get_state_array(self._index, self.is_testing)
        return ret

    def get_history_states(self, n)->List[np.ndarray]:
        # get the history states of the environment, the length of the list is n, the last state is the current state
        # if n is larger than the length of the history states, return the whole history states
        if n > len(self._buffered_states):
            zero_paddings = [np.zeros(shape=(self._data.n_state,))]*(n-len(self._buffered_states))
            return zero_paddings + self._buffered_states
        else:
            return self._buffered_states[-n:]
        
    def get_output_dim(self)->int:
        return len(self._data.worker_quality)
    
    def get_state_dim(self)->int:
        return self._data.n_state

    def set_reward_type(self, reward_type):
        self._reward_type = reward_type
    
    def set_testing(self, is_testing):
        self.is_testing = is_testing
