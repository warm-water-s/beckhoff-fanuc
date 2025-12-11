from config import Config, plot_rewards, env_agent_config
from train_test import test
import matplotlib.pyplot as plt

# 获取参数
cfg = Config()

# 加载环境和智能体
# env, agent = env_agent_config(cfg, "./test_mat_files/stability_boundary_w922.mat")
# env, agent = env_agent_config(cfg, "./StabilityLobeData_3D.mat")
env, agent = env_agent_config(cfg, "./test_mat_files")

# 加载模型
agent.load_models()

# 测试
res_dic = test(cfg, env, agent)
plot_rewards(res_dic["rewards"], cfg, tag="test")  # 画出结果
plt.show()
