import copy
from config import *


def train(cfg, env, agent):
    """训练"""
    print("开始训练！")
    create_directory(
        path=cfg.checkpoint_dir,
        sub_path_list=[
            "Actor",
            "Critic",
        ],
    )
    rewards = []  # 记录所有回合的奖励
    steps = []
    best_ep_reward = float("-inf")  # 记录最大回合奖励
    output_agent = None
    for i_ep in range(cfg.train_eps):  # 控制训练的回合数
        ep_reward = 0  # 记录一回合内的奖励
        ep_step = 0
        state = env.reset()
        for _ in range(cfg.max_steps):  # 控制每个回合的最大步数
            ep_step += 1
            action, log_prob = agent.sample_action(state)  # 选择动作
            # 更新环境，返回transition
            next_state, reward, done = env.step(state,action)

            agent.memory.push((state, action, log_prob, reward, done))  # 保存transition
            state = next_state  # 更新下一个状态
            # print(
            #     f"action:{action[0]:.2f}\t state:{state[2]}, {state[3]:.2f}\t reward:{reward[0]:.2f}"
            # )
            # print(action, state, reward)
            agent.update()  # 更新智能体
            ep_reward += reward  # 累加奖励
            if done:
                break
        # 每 cfg.eval_per_episode 个回合进行一次评估
        if (i_ep + 1) % cfg.eval_per_episode == 0:
            sum_eval_reward = 0
            for t in range(cfg.eval_eps):
                eval_ep_reward = 0
                state = env.reset()
                for t in range(cfg.max_steps):
                    action = agent.predict_action(state)  # 选择动作
                    # 更新环境，返回transition
                    next_state, reward, done = env.step(state, action)
                    state = next_state  # 更新下一个状态
                    eval_ep_reward += reward  # 累加奖励
                    if done:
                        break
                sum_eval_reward += eval_ep_reward
            mean_eval_reward = sum_eval_reward / cfg.eval_eps
            # 如果平均评估奖励大于等于最佳评估奖励，则更新最佳评估奖励并保存当前智能体
            # 训练过程中，agent 会继续基于当前策略和经验数据更新。
            # 训练完成后，best_agent 是训练过程中表现最好的模型，通常用于测试或部署。
            # 下次训练不会基于更新之后的 best_agent，而是基于 best_agent。
            if mean_eval_reward >= best_ep_reward:
                best_ep_reward = mean_eval_reward
                output_agent = copy.deepcopy(agent)
                print(
                    f"回合：{i_ep+1}/{cfg.train_eps}，奖励：{ep_reward:.2f}，评估奖励：{mean_eval_reward:.2f}，最佳评估奖励：{best_ep_reward:.2f}，更新模型！"
                )
            else:
                print(
                    f"回合：{i_ep+1}/{cfg.train_eps}，奖励：{ep_reward:.2f}，评估奖励：{mean_eval_reward:.2f}，最佳评估奖励：{best_ep_reward:.2f}"
                )
        steps.append(ep_step)
        rewards.append(ep_reward)
        # # 保存整个 agent 对象
        # torch.save(output_agent, "save_path/ppo_agent.pth")
        # agent= torch.load("save_path/ppo_agent.pth") # 将更新的模型继续用来训练
    print("完成训练！")
    output_agent.save_models()  # type: ignore
    return output_agent, {"rewards": rewards}


def test(cfg, env, agent):
    # 测试的每个回合是相互独立的
    print("开始测试！")
    rewards = []  # 记录所有回合的奖励
    steps = []
    for i_ep in range(cfg.test_eps):
        ep_reward = 0  # 记录一回合内的奖励
        ep_step = 0
        state = env.reset()
        for t in range(cfg.max_steps):
            ep_step += 1
            action = agent.predict_action(state)  # 选择动作
            next_state, reward, done = env.step(state, action)
            if t % 10 == 0:
                print(f"action:{action}\t ap:{state[0]:.5f}\t n:{state[2]:.5f}")
            state = next_state  # 更新下一个状态
            ep_reward += reward  # 累加奖励
            if done:    
                break
        steps.append(ep_step)
        rewards.append(ep_reward)
        print(f"回合：{i_ep+1}/{cfg.test_eps}，奖励：{ep_reward:.2f}")
    print("完成测试")
    return {"rewards": rewards}
