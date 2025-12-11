from config import Config, plot_rewards, env_agent_config
from train_test import train, test
import matplotlib.pyplot as plt

# 获取参数
cfg = Config()
# 训练
# train_env, agent = env_agent_config(cfg, "./StabilityLobeData_3D.mat")
# train_env, agent = env_agent_config(cfg, "./train_mat_files") # env_ppo
train_env, agent = env_agent_config(cfg, "./train_mat_files")


best_agent, res_dic = train(cfg, train_env, agent)
plot_rewards(res_dic["rewards"], cfg, tag="train")


# 测试
res_dic = test(cfg, train_env, best_agent)
plot_rewards(res_dic["rewards"], cfg, tag="test")  # 画出结果
plt.show()
