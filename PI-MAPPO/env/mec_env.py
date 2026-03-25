from env.device_env import DeviceEnv
from env.edge_env import EdgeEnv
import numpy as np

class MECEnv():
    def __init__(self, params):
        self.device_num = params.device_num

        # edge env
        self.edge_env = EdgeEnv(params)
        # device envs
        self.device_envs = []
        for i in range(self.device_num):
            self.device_envs.append(DeviceEnv(i, params))

        self.expense_weights = params.expense_weights
        self.energy_weights = params.energy_weights

        self.w_delay = getattr(params, "w_delay", 0.6)
        self.w_energy = getattr(params, "w_energy", 0.2)
        self.w_cost = getattr(params, "w_cost", 0.2)
        self.alpha_overtime = getattr(params, "alpha_overtime", 30.0)
        self.tau = getattr(params, "tau", 0.8)
        self.eps = 1e-8

    def reset(self):
        edge_obs = self.edge_env.reset()

        device_obss = [None for _ in range(self.device_num)]
        for i in range(self.device_num):
            device_obss[i] = self.device_envs[i].reset()

        return edge_obs, device_obss

    def step(self, device_acts):
        device_sched_tasks = [None for i in range(self.device_num)]
        for i in range(self.device_num):
            sched_tasks = self.device_envs[i].compute(device_acts[i])
            device_sched_tasks[i] = sched_tasks

        self.edge_env.compute(device_sched_tasks)

        device_rewards = [0 for _ in range(self.device_num)]
        device_costs = [0 for _ in range(self.device_num)]
        device_comp_dlys = [0 for _ in range(self.device_num)]
        device_csum_engys = [0 for _ in range(self.device_num)]
        device_comp_expns = [0 for _ in range(self.device_num)]
        device_overtime_nums = [0 for _ in range(self.device_num)]

        for i in range(self.device_num):
            sched_tasks = device_sched_tasks[i]
            task_num = len(sched_tasks)
            for j in range(task_num):
                task = sched_tasks[j]

                comp_dly = max(task.l_comp_dly, task.e_comp_dly)
                csum_engy = task.l_csum_engy + task.e_csum_engy
                comp_expn = task.comp_expn

                device_comp_dlys[i] += 1 / (j + 1) * (comp_dly - device_comp_dlys[i])
                device_csum_engys[i] += 1 / (j + 1) * (csum_engy - device_csum_engys[i])
                device_comp_expns[i] += 1 / (j + 1) * (comp_expn - device_comp_expns[i])

                device_costs[i] += self.energy_weights[i] * csum_engy + \
                                   self.expense_weights[i] * comp_expn

                lateness = comp_dly - task.dly_cons
                overtime_pen = -self.alpha_overtime * (
                    1.0 / (1.0 + np.exp(-lateness / self.tau))
                )

                delay_term = comp_dly / (task.dly_cons + self.eps)
                energy_term = (task.l_csum_engy + task.e_csum_engy) / (task.norm_csum_engy + self.eps)
                cost_term = task.comp_expn / (task.norm_comp_expn + self.eps)

                base_penalty = -(
                    self.w_delay * delay_term
                    + self.w_energy * energy_term
                    + self.w_cost * cost_term
                )

                task_reward = base_penalty + overtime_pen
                device_rewards[i] += task_reward

                if comp_dly > task.dly_cons:
                    device_overtime_nums[i] += 1

        joint_reward = sum(device_rewards)
        joint_cost = sum(device_costs)

        next_edge_obs = self.edge_env.get_obs()
        next_device_obss = [None for _ in range(self.device_num)]
        for i in range(self.device_num):
            next_device_obss[i] = self.device_envs[i].get_obs()

        return joint_reward, device_rewards, \
               joint_cost, device_costs, \
               device_comp_dlys, device_csum_engys, \
               device_comp_expns, device_overtime_nums, \
               next_edge_obs, next_device_obss
