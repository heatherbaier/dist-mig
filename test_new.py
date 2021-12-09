#!/usr/bin/env python3
import io
import os
import pprint
import sys

import torch.distributed as dist

from torch.distributed.elastic.multiprocessing.errors import record


# @record
# def main():
#     print("in here!")


import argparse
import gym
import numpy as np
import os
from itertools import count

import torch
import torch.distributed.rpc as rpc
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributed.rpc import RRef, rpc_sync, rpc_async, remote
from torch.distributions import Categorical
import torch.distributed as dist

from config import get_config
from dataloader import *
from earth_env import *


TOTAL_EPISODE_STEP = 5000
AGENT_NAME = "agent"
OBSERVER_NAME = "observer{}"

parser = argparse.ArgumentParser(description='PyTorch RPC RL example')
parser.add_argument('--world-size', type = int, default = 4, metavar = 'W',
                    help='world size for RPC, rank 0 is the agent, others are observers')
parser.add_argument('--gamma', type = float, default = 0.99, metavar = 'G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--seed', type = int, default = 543, metavar = 'N',
                    help='random seed (default: 543)')
parser.add_argument('--log-interval', type = int, default = 10, metavar = 'N',
                    help='interval between training status logs (default: 10)')
args = parser.parse_args()

torch.manual_seed(args.seed)


def _call_method(method, rref, *args, **kwargs):

    r"""
    a helper function to call a method on the given RRef
    """

    return method(rref.local_value(), *args, **kwargs)


def _remote_method(method, rref, *args, **kwargs):

    r"""
    a helper function to run method on the owner of rref and fetch back the
    result using RPC
    """

    args = [method, rref] + list(args)
    return rpc_sync(rref.owner(), _call_method, args = args, kwargs = kwargs)


class Policy(nn.Module):

    r"""
    Borrowing the ``Policy`` class from the Reinforcement Learning example.
    Copying the code to make these two examples independent.
    See https://github.com/pytorch/examples/tree/master/reinforcement_learning
    """

    def __init__(self, h, w, outputs):

        super(Policy, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size = 5, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32
        self.head = nn.Linear(linear_input_size, outputs)
        self.mig_head = nn.Linear(linear_input_size, 1)
        self.rnn = torch.nn.LSTM(input_size = linear_input_size, hidden_size = 256, num_layers = 1, batch_first = True)
        self.fc = torch.nn.Linear(256, 1)


    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x, seq = None, select = False):
        
        x = x.to(device)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        if seq is not None:
            seq = torch.cat( (seq, x.view(x.size(0), -1).unsqueeze(0)), dim = 1 )
        else:
            seq = x.view(x.size(0), -1).unsqueeze(0)

        pred = self.rnn(seq)[0][:, -1, :].unsqueeze(0)

        if select:
            return self.head(x.view(x.size(0), -1)), self.fc(pred), x.view(x.size(0), -1).unsqueeze(0)
        else:
            return self.head(x.view(x.size(0), -1)), self.fc(pred)


class Observer:

    r"""
    An observer has exclusive access to its own environment. Each observer
    captures the state from its environment, and send the state to the agent to
    select an action. Then, the observer applies the action to its environment
    and reports the reward to the agent.
    It is true that CartPole-v1 is a relatively inexpensive environment, and it
    might be an overkill to use RPC to connect observers and trainers in this
    specific use case. However, the main goal of this tutorial to how to build
    an application using the RPC API. Developers can extend the similar idea to
    other applications with much more expensive environment.
    """

    def __init__(self):

        self.id = rpc.get_worker_info().id

        print("Observer with ID: ", self.id)
        print("WORKER INFO: ", rpc.get_worker_info())


    def load_data(self, data):

        """
        Here we recieve an asynchrous call from the Agent with a list of training data batches. Select just 
        the batch that matches the current observers ID value (minus 1) and load it as the observer's data.
        Then, we load the environment's for each of the images in our dataset.
        """

        self.data = data[self.id - 1]

        print("IMAGE LOADED IN OBSERVER: ", self.id, self.data[0][0])

        self.envs = []

        for (impath, y, env_params) in self.data:

            env = EarthObs(impath = impath,
                           y_val = y,
                           num_channels = env_params[0], 
                           num_actions = env_params[1], 
                           display = env_params[2], 
                        #    batch_size = env_params[3],
                           GAMMA = env_params[3],
                           EPS_START = env_params[4],
                           EPS_END = env_params[5],
                           EPS_DECAY = env_params[6],
                           TARGET_UPDATE = env_params[7])

            self.envs.append(env)


    def run_episode(self, agent_rref, n_steps):

        r"""
        Run one episode of n_steps.
        Arguments:
            agent_rref (RRef): an RRef referencing the agent object.
            n_steps (int): number of steps in this episode
        """

        print("RUNNING EPISDE IN OBSERVER: ", self.id)

        # AKA: For each image in batch:
        for env in self.envs:

            # Reset the environment
            state, ep_reward = env.reset(), 0

            for step in range(n_steps):

                # send the state to the agent to get an action
                action = _remote_method(Agent.select_action, agent_rref, self.id, state)

                # If the action is a select:
                # 1) Get the current state and information on the number of grabs thus far
                # 2) Send that information to the Agent and make it calculate the reward
                # 3) Send the new prediction back to the environment so she can update its information
                if action == 4:
                    # print("ACTION IN OBSERVER ", self.id, " IS ", action)
                    new_state, gv, y_val, prev_pred = env.select()
                    new_pred, fc_layer = _remote_method(Agent.calculate_reward, agent_rref, self.id, action, None, new_state, gv, y_val, prev_pred)
                    done = env.end_select(new_pred, fc_layer)

                    # If the agent is done, break the training
                    if done:
                        # print("AGENT IS DONE: ", self.id)
                        break
                
                # If the action is just a move...
                # 1) Send the direction to the environment and get the new state back
                # 3) Send the state information to the Agent so she can calculate the reward
                # 4) Update the state to the new state
                else:
                    new_state, gv, y_val = env.move_box(action)
                    _remote_method(Agent.calculate_reward, agent_rref, self.id, action, state, new_state, gv, y_val, None)
                    state = new_state




class Agent:

    def __init__(self, world_size, config):

        self.ob_rrefs = []
        self.agent_rref = RRef(self)
        self.rewards = {}
        self.saved_log_probs = {}
        self.n_actions = 5
        self.policy = Policy(128, 128, self.n_actions)
        self.target_net = Policy(128, 128, self.n_actions)
        self.optimizer = optim.Adam(self.policy.parameters(), lr = 1e-2)
        self.criterion = nn.L1Loss()
        self.eps = np.finfo(np.float32).eps.item()
        self.running_reward = 0
        self.reward_threshold = gym.make('CartPole-v1').spec.reward_threshold
        self.config = config
        self.steps_done = 0
        self.eps_threshold = .9
        self.memory = []
        self.limit = 1000
        self.to_tens = transforms.ToTensor()

        # self.config, _ = get_config()

        # Load all of the data into the Agent on Rank 0
        self.train_dl = self.load_data(world_size)

        # For each of the observers in the remote world, set up their information in the agent
        for ob_rank in range(1, world_size):
            ob_info = rpc.get_worker_info(OBSERVER_NAME.format(ob_rank))
            self.ob_rrefs.append(remote(ob_info, Observer))
            self.rewards[ob_info.id] = []
            self.saved_log_probs[ob_info.id] = []

        print("TRAINING DATA: ", self.train_dl)

        # Make an RPC call to distribute a data batch to each of the Observers
        self.distribute_data()


    def distribute_data(self):

        """
        TO-DO: FOR NOW THIS IS A BLOCKING SYNC CALL SINCE WE NEED THE IMAGE TO BE LOADED IN ORDER TO 
        ACTUALLY CALL RUN_EPISODE LATER ON. TRY TO FIND A WAY TO DO THE LIKE PROCESS.JOIN THING SO WE
        CAN GET THE LOADING STARTED ON ALL OBSERVERS SIMOUTANESOULY AND THEN WAIT FOR THEM ALL TO END
        """

        print("In distribute data function!")

        futs = []

        for ob_rref in self.ob_rrefs:

            futs.append(
                rpc_async(
                    ob_rref.owner(),
                    _call_method,
                    args = (Observer.load_data, ob_rref, self.train_dl)
                )
            )

        print("About to wait!")

        for fut in futs:
            fut.wait()

        


    def load_data(self, world_size):

        """
        Function to load in the data on the Agent
        TO-DO: THIS DOES NOT CURRENLTY LOAD IN THE WHOLE IMAGE (JUST A TUPLE THAT INCLUDES THE LIST OF PARAMETERS THE 
        OBSERVERS WILL NEED TO SET UP THE ENVIRONMENTS) SO ACTUALLY I TENTATIVELY THINK THIS MIGHT BE A FINE WAY TO DO THIS
        """

        print("In load data!! WORLD SIZE: ", world_size)
            
        # Load in the data
        data = Dataset(world_size = world_size,
                        num_workers = world_size,
                        imagery_dir = self.config.imagery_dir,
                        json_path = self.config.json_path,
                        split = self.config.tv_split,
                        eps_decay = self.config.eps_decay)


        train_dl = data.train_data
        self.batch_size = data.batch_size
        self.GAMMA = data.gamma

        print("Done with loading data!")

        return train_dl


    def select_action(self, ob_id, state):
        
        r"""
        This function is mostly borrowed from the Reinforcement Learning example.
        See https://github.com/pytorch/examples/tree/master/reinforcement_learning
        The main difference is that instead of keeping all probs in one list,
        the agent keeps probs in a dictionary, one key per observer.
        NB: no need to enforce thread-safety here as GIL will serialize
        executions.
        """

        """
        TO DO: READ THE MESSAGE ABOVE AND DO THE DICTIONARY THING!!!!
        """

        # state = torch.from_numpy(state).float().unsqueeze(0)
        # probs = self.policy(state)
        # m = Categorical(probs)
        # action = m.sample()
        # self.saved_log_probs[ob_id].append(m.log_prob(action))
        # return action.item()

        # Read in the image and convert it to a tensor
        # state = self.to_tens(self.view_box.clip_image(self.image)).unsqueeze(0)
        
        # Get a random number between 0 & 1
        sample = random.random()

        # Calculate the new epsilon threshold
        # self.eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * math.exp(-1. * self.steps_done / self.EPS_DECAY)
        self.steps_done += 1

        # If the random number is larger than eps_threshold, use the trained policy to pick an action
        if sample > self.eps_threshold:
            with torch.no_grad():
                return self.policy(state)[0].max(1)[1].view(1, 1)

        # Otherwise, pick a random action
        else:
            return torch.tensor([[random.randrange(self.n_actions)]], dtype = torch.long)


    def calculate_reward(self, ob_id, action, old_state, new_state, gv, y_val, prev_pred):

        """
        Function recieves data from an observer, calculates the reward, and saves the information for later optimization
        TO-DO: AFTER  YOU HAVE THE GRAB STEP DONE, NEED TO DO THE RECORDING OF THE REWARDS PER OBSERVER AND STUFF
        """

        # Grab
        if action == 4:

            # Run the new sequence through the model
            if len(gv) == 0:
                _, mig_pred, fc_layer = self.policy(new_state, seq = None, select = True)
            else:
                seq = torch.cat(gv, dim = 1)
                _, mig_pred, fc_layer = self.policy(new_state, seq = seq, select = True)    

            try:
                # print("OPTIMIZING PRED! ", mig_pred)
                # print(mig_pred.shape, y_val.shape)
                mig_loss = self.criterion(mig_pred.squeeze(), y_val)
                # print("MIG LOSS: ", mig_loss)
                self.optimizer.zero_grad()
                mig_loss.backward()
                self.optimizer.step()
            except:
                pass

            # self.optimize_pred(mig_pred, y_val)

            print("MIG PRED: ", mig_pred)

            # If the sequence prediction is better with the new state than without, give the model a thumbs up
            if abs(y_val - prev_pred) > abs(y_val - mig_pred):
                reward = 10
            else:
                reward = 0

            # Send the new prediction back to the observer for its own records
            return mig_pred, fc_layer

        # Move
        else:

            # Run the previous state through the model
            if len(gv) == 0:
                _, mig_pred_t1 = self.policy(old_state, seq = None)
            else:
                seq = torch.cat(gv, dim = 1)
                _, mig_pred_t1 = self.policy(old_state, seq = seq)

            # Run the new state through the model
            if len(gv) == 0:
                _, mig_pred_t2 = self.policy(new_state, seq = None)
            else:
                seq = torch.cat(gv, dim = 1)
                _, mig_pred_t2 = self.policy(new_state, seq = seq)

            # If the image after the view box shift is closer to the true value than before, give the model a pat on the back
            if abs(y_val - mig_pred_t1) > abs(y_val - mig_pred_t2):
                reward = 10
            else:
                reward = 0       

            memory = (old_state, action, new_state, torch.tensor([reward])) 
            self.memory.append(memory)


    # def optimize_pred(self, mig_pred, y_val):
    #     print("OPTIMIZING PRED!")
    #     mig_loss = self.criterion(mig_pred.squeeze(0), y_val)
    #     print("MIG LOSS: ", mig_loss)
    #     self.optimizer.zero_grad()
    #     mig_loss.backward()
    #     self.optimizer.step()


    def report_reward(self, ob_id, reward):

        r"""
        Observers call this function to report rewards.
        """

        self.rewards[ob_id].append(reward)

    def run_episode(self, n_steps = 5):

        r"""
        Run one episode. The agent will tell each oberser to run n_steps.
        """

        futs = []

        for ob_rref in self.ob_rrefs:

            # Make async RPC to kick off an episode on all observers
            futs.append(
                rpc_async(
                    ob_rref.owner(),
                    _call_method,
                    args = (Observer.run_episode, ob_rref, self.agent_rref, n_steps)
                )
            )

        # Wait until all observers have finished this episode
        for fut in futs:
            fut.wait()


    def optimize_model(self):

        if len(self.memory) < self.batch_size:
            print("RETURNING BECAUSE TOO FEW MEMORY")
            return

        if len(self.memory) == self.limit:
            self.memory.pop(0)

        transitions = random.sample(list(self.memory), self.batch_size)

        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
                                                    
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to shared_model
        pnet_val, _ = self.policy(state_batch)

        state_action_values = pnet_val.gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.batch_size)
        tnet_val, _ = self.target_net(non_final_next_states)
        next_state_values[non_final_mask] = tnet_val.max(1)[0].detach()
        
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch

        # Compute Huber loss
        loss = self.criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


    def finish_episode(self):
        r"""
        This function is mostly borrowed from the Reinforcement Learning example.
        See https://github.com/pytorch/examples/tree/master/reinforcement_learning
        The main difference is that it joins all probs and rewards from
        different observers into one list, and uses the minimum observer rewards
        as the reward of the current episode.
        """

        # joins probs and rewards from different observers into lists
        R, probs, rewards = 0, [], []
        for ob_id in self.rewards:
            probs.extend(self.saved_log_probs[ob_id])
            rewards.extend(self.rewards[ob_id])

        # use the minimum observer reward to calculate the running reward
        min_reward = min([sum(self.rewards[ob_id]) for ob_id in self.rewards])
        self.running_reward = 0.05 * min_reward + (1 - 0.05) * self.running_reward

        # clear saved probs and rewards
        for ob_id in self.rewards:
            self.rewards[ob_id] = []
            self.saved_log_probs[ob_id] = []

        policy_loss, returns = [], []
        for r in rewards[::-1]:
            R = r + args.gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + self.eps)
        for log_prob, R in zip(probs, returns):
            policy_loss.append(-log_prob * R)
        self.optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()
        self.optimizer.step()
        return min_reward



@record
def run_worker(rank, world_size):

    r"""
    This is the entry point for all processes. The rank 0 is the agent. All
    other ranks are observers.
    """

    # os.environ['GLOO_SOCKET_IFNAME'] = "ib0"
    os.environ['TP_SOCKET_IFNAME'] = "ib0"

    config, _ = get_config()

    # dist.init_process_group(backend = "gloo", rank = rank, world_size = args.world_size)

    # Rank 0 is the agent
    if rank == 0:

        # print("RANK IF: ", rank)

        # Set up remote protocol on core
        # rpc.init_rpc(AGENT_NAME, rank = rank, world_size = world_size)

        rpc.init_rpc(AGENT_NAME, rank = rank, world_size = world_size, rpc_backend_options = rpc.TensorPipeRpcBackendOptions(_transports=["uv"], rpc_timeout=500))

        print("AGENT RPC INITIALIZED!")

        # Initialize the agent in rank 0
        agent = Agent(world_size, config)

        # i_episode? I think this is equivalent to epochs
        for i_episode in range(20):

            print("EPISODE: ", i_episode)

            n_steps = int(TOTAL_EPISODE_STEP / (int(os.environ['WORLD_SIZE']) - 1))

            # So here we would itereate over batches (i think...)

            # Run epsiode is basically 1 call of 'train'
            agent.run_episode(n_steps = n_steps)

            print("ABOUT TO OPTIMIZE!")

            agent.optimize_model()

        #     last_reward = agent.finish_episode()

        #     if i_episode % args.log_interval == 0:
        #         print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
        #               i_episode, last_reward, agent.running_reward))

        #     if agent.running_reward > agent.reward_threshold:
        #         print("Solved! Running reward is now {}!".format(agent.running_reward))
        #         break

    # All other ranks are observers
    else:

        # print("RANK ELSE: ", rank)

        # Set up remote protocol on core
        rpc.init_rpc(OBSERVER_NAME.format(rank), rank = rank, world_size = world_size, rpc_backend_options = rpc.TensorPipeRpcBackendOptions(_transports=["uv"], rpc_timeout=500))

        print("OBSERVER RPC INITIALIZED!")


        # observers passively waiting for instructions from agents

    rpc.shutdown()



def main():

    run_worker(dist.get_rank(), dist.get_world_size())

    # mp.spawn(
    #     run_worker,
    #     args = (args.world_size, ),
    #     nprocs = args.world_size,
    #     join = True
    # )


if __name__ == "__main__":

    print("here!")

    # main()

    # os.environ['OMP_NUM_THREADS'] = 20

    env_dict = {
        k: os.environ[k]
        for k in (
            "LOCAL_RANK",
            "RANK",
            "GROUP_RANK",
            "WORLD_SIZE",
            "MASTER_ADDR",
            "MASTER_PORT",
            "TORCHELASTIC_RESTART_COUNT",
            "TORCHELASTIC_MAX_RESTARTS",
        )
    }

    with io.StringIO() as buff:
        print("======================================================", file=buff)
        print(
            f"Environment variables set by the agent on PID {os.getpid()}:", file=buff
        )
        pprint.pprint(env_dict, stream=buff)
        print("======================================================", file=buff)
        print(buff.getvalue())
        sys.stdout.flush()

    dist.init_process_group(backend="gloo")
    dist.barrier()

    print(
        (
            f"On PID {os.getpid()}, after init process group, "
            f"rank={dist.get_rank()}, world_size = {dist.get_world_size()}\n"
        )
    )


    main()

