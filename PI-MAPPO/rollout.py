import pickle
import copy
import numpy as np
import torch
from torch.distributions import Normal

from env.mec_env import MECEnv
from agent.edge_agent import EdgeAgent
from agent.device_agent import DeviceAgent
from util.replay_buffer import ReplayBuffer
from util.utils import ObsScaling, RewardScaling, GetPolicyInputs

from util.PI_controller import PI_Controller

WARMUP_EPISODES = 50
class Rollout:
    def __init__(self, params):
        self.evaluate = params.evaluate
        if self.evaluate == False:
            torch.manual_seed(params.train_seed)
            np.random.seed(params.train_seed)
        else:
            torch.manual_seed(params.eval_seed)
            np.random.seed(params.eval_seed)

        self.device_num = params.device_num
        self.task_num = params.task_num

        # train
        self.train_time_slots = params.train_time_slots
        self.train_freq = params.train_freq
        self.save_freq = params.save_freq
        self.results_dir = params.results_dir
        self.weights_dir = params.weights_dir
        # obs scaling
        self.use_obs_scaling = params.use_obs_scaling
        if self.use_obs_scaling:
            self.obs_scaling = ObsScaling()
            if params.load_scales:
                self.load_scales()
        # reward scaling
        self.use_reward_scaling = params.use_reward_scaling
        if self.use_reward_scaling:
            self.reward_scaling = RewardScaling()

        # evaluate
        self.eval_mode = params.eval_mode
        self.eval_time_slots = params.eval_time_slots

        # MEC env
        self.mec_env = MECEnv(params)
        if not self.evaluate:
            # edge agent
            self.edge_agent = EdgeAgent(params)
            # replay buffer
            self.replay_buffer = ReplayBuffer(params)
        # device agents
        self.device_agents = []
        for i in range(self.device_num):
            self.device_agents.append(DeviceAgent(i, params))

        if not self.evaluate:
            self.pi_controllers = []
            for i in range(self.device_num):
                self.pi_controllers.append(PI_Controller(params, i))

        # initialize agents' policy networks
        if not self.evaluate:
            for i in range(self.device_num):
                self.device_agents[i].update_net(self.edge_agent.p_nets[i].state_dict())
        else:
            if self.eval_mode == "mappo":
                for i in range(self.device_num):
                    path = self.weights_dir + "p_net_params_" + str(i) + ".pkl"
                    self.device_agents[i].load_net(path)

        # criteria
        self.joint_reward = None
        self.device_rewards = None
        self.joint_cost = None
        self.device_costs = None
        self.edge_comp_ql = None
        self.device_comp_qls = None
        self.device_comp_dlys = None
        self.device_csum_engys = None
        self.device_comp_expns = None
        self.device_overtime_nums = None

        import csv
        import os
        self.all_acts_csv_path = os.path.join(self.results_dir, "all_device_actions.csv")
        os.makedirs(self.results_dir, exist_ok=True)
        if not os.path.exists(self.all_acts_csv_path):
            with open(self.all_acts_csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(["episode_id", "time_slot_id", "device_id", "action"])

    def reset(self):
        self.joint_reward = 0
        self.device_rewards = np.zeros([self.device_num], dtype=np.float32)
        self.joint_cost = 0
        self.device_costs = np.zeros([self.device_num], dtype=np.float32)
        self.edge_comp_ql = 0
        self.device_comp_qls = np.zeros([self.device_num], dtype=np.float32)
        self.device_comp_dlys = np.zeros([self.device_num], dtype=np.float32)
        self.device_csum_engys = np.zeros([self.device_num], dtype=np.float32)
        self.device_comp_expns = np.zeros([self.device_num], dtype=np.float32)
        self.device_overtime_nums = np.zeros([self.device_num], dtype=np.float32)

    def run(self, e_id):
        # reset
        self.reset()

        if not self.evaluate and self.use_reward_scaling:
            self.reward_scaling.reset()

        edge_obs, device_obss = self.mec_env.reset()
        edge_comp_ql = edge_obs[0]
        device_comp_qls = [obs[0] for obs in device_obss]
        # obs scaling
        if self.use_obs_scaling:
            self.obs_scaling(edge_obs, device_obss, self.evaluate)
        # rollout
        time_slots = self.train_time_slots + 1 if not self.evaluate else self.eval_time_slots
        for t_id in range(1, time_slots + 1):
            print("-------------time slot: " + str(t_id) + "-------------")

            device_acts = [None for i in range(self.device_num)]
            device_act_logprobs = [None for i in range(self.device_num)]

            # 简化版调用
            if not self.evaluate :
                for i in range(self.device_num):
                    act_drl, act_logprob_drl = self.device_agents[i].choose_action(
                        device_obss[i], False)

                    current_q = self.mec_env.device_envs[i].comp_ql
                    adjusted_act, was_adjusted = self.pi_controllers[i].adjust_drl_action(
                        act_drl, current_q)

                    if was_adjusted and  e_id <= WARMUP_EPISODES:
                        adjusted_act_tensor = torch.tensor(adjusted_act, dtype=torch.float32).unsqueeze(0)
                        with torch.no_grad():
                            p_inputs = GetPolicyInputs(device_obss[i])
                            mean, std = self.device_agents[i].p_net(p_inputs)
                        dist = Normal(mean, std)
                        adjusted_logprob = dist.log_prob(adjusted_act_tensor).sum(dim=-1)

                        device_acts[i] = adjusted_act
                        device_act_logprobs[i] = float(adjusted_logprob.item())
                    else:
                        device_acts[i] = act_drl
                        device_act_logprobs[i] = act_logprob_drl

                    if was_adjusted:
                        if isinstance(adjusted_act, torch.Tensor):
                            act_val = adjusted_act.cpu().detach().numpy().tolist()
                        elif isinstance(adjusted_act, np.ndarray):
                            act_val = adjusted_act.tolist()
                        else:
                            act_val = adjusted_act

                        import csv
                        with open(self.all_acts_csv_path, 'a', newline='', encoding='utf-8') as f:
                            writer = csv.writer(f)
                            writer.writerow([e_id, t_id, i, act_val])
                    else:
                        if isinstance(act_drl, torch.Tensor):
                            act_val = act_drl.cpu().detach().numpy().tolist()
                        elif isinstance(act_drl, np.ndarray):
                            act_val = act_drl.tolist()
                        else:
                            act_val = act_drl

                        import csv
                        with open(self.all_acts_csv_path, 'a', newline='', encoding='utf-8') as f:
                            writer = csv.writer(f)
                            writer.writerow([e_id, t_id, i, act_val])

            # step
            joint_reward, device_rewards, \
                joint_cost, device_costs, \
                device_comp_dlys, device_csum_engys, \
                device_comp_expns, device_overtime_nums, \
                next_edge_obs, next_device_obss = self.mec_env.step(device_acts)

            self.average(t_id, joint_reward, device_rewards,
                         joint_cost, device_costs,
                         edge_comp_ql, device_comp_qls,
                         device_comp_dlys, device_csum_engys,
                         device_comp_expns, device_overtime_nums)

            if not self.evaluate:
                if self.use_reward_scaling:
                    joint_reward = self.reward_scaling(joint_reward)

                # store sampled data
                self.replay_buffer.store(edge_obs,
                                         device_obss,
                                         device_acts,
                                         device_act_logprobs,
                                         joint_reward)

            # update computing-queue lengths
            edge_comp_ql = next_edge_obs[0]
            device_comp_qls = [obs[0] for obs in next_device_obss]
            # obs scaling
            if self.use_obs_scaling:
                self.obs_scaling(next_edge_obs, next_device_obss, self.evaluate)

            # update obs
            edge_obs = next_edge_obs
            device_obss = next_device_obss

        if not self.evaluate:
            # train
            if e_id % self.train_freq == 0:
                self.edge_agent.train_nets(self.replay_buffer)

                for i in range(self.device_num):
                    self.device_agents[i].update_net(self.edge_agent.p_nets[i].state_dict())

            # save
            if e_id % self.save_freq == 0:
                self.edge_agent.save_nets(e_id)
                self.save_scales(e_id)

        joint_reward = copy.copy(self.joint_reward)
        device_rewards = copy.copy(self.device_rewards)
        joint_cost = copy.copy(self.joint_cost)
        device_costs = copy.copy(self.device_costs)
        edge_comp_ql = copy.copy(self.edge_comp_ql)
        device_comp_qls = copy.copy(self.device_comp_qls)
        device_comp_dlys = copy.copy(self.device_comp_dlys)
        device_csum_engys = copy.copy(self.device_csum_engys)
        device_comp_expns = copy.copy(self.device_comp_expns)
        device_overtime_nums = copy.copy(self.device_overtime_nums)

        return joint_reward, device_rewards, \
            joint_cost, device_costs, \
            edge_comp_ql, device_comp_qls, \
            device_comp_dlys, device_csum_engys, \
            device_comp_expns, device_overtime_nums

    def average(self, t_id, joint_reward, device_rewards,
                joint_cost, device_costs,
                edge_comp_ql, device_comp_qls,
                device_comp_dlys, device_csum_engys,
                device_comp_expns, device_overtime_nums):
        self.joint_reward += 1 / t_id * (joint_reward - self.joint_reward)
        self.device_rewards += 1 / t_id * (device_rewards - self.device_rewards)
        self.joint_cost += 1 / t_id * (joint_cost - self.joint_cost)
        self.device_costs += 1 / t_id * (device_costs - self.device_costs)
        self.edge_comp_ql += 1 / t_id * (edge_comp_ql - self.edge_comp_ql)
        self.device_comp_qls += 1 / t_id * (device_comp_qls - self.device_comp_qls)
        self.device_comp_dlys += 1 / t_id * (device_comp_dlys - self.device_comp_dlys)
        self.device_csum_engys += 1 / t_id * (device_csum_engys - self.device_csum_engys)
        self.device_comp_expns += 1 / t_id * (device_comp_expns - self.device_comp_expns)
        self.device_overtime_nums += device_overtime_nums

    def save_scales(self, e_id):
        path = self.results_dir + "obs_scales_" + str(e_id) + ".pkl"
        with open(path, "wb") as f:
            e_mean = self.obs_scaling.e_running_ms.mean
            pickle.dump(e_mean, f)
            e_std = self.obs_scaling.e_running_ms.std
            pickle.dump(e_std, f)
            d_mean = self.obs_scaling.d_running_ms.mean
            pickle.dump(d_mean, f)
            d_std = self.obs_scaling.d_running_ms.std
            pickle.dump(d_std, f)

    def load_scales(self):
        path = self.results_dir + "obs_scales.pkl"
        with open(path, "rb") as f:
            e_mean = pickle.load(f)
            e_std = pickle.load(f)
            d_mean = pickle.load(f)
            d_std = pickle.load(f)
        self.obs_scaling.e_running_ms.mean = e_mean
        self.obs_scaling.e_running_ms.std = e_std
        self.obs_scaling.d_running_ms.mean = d_mean
        self.obs_scaling.d_running_ms.std = d_std