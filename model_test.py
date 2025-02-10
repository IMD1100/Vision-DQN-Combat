# test_model.py
import time
from environment.game_env import GameEnv
from agent.model import CNNModel
from agent.agent import DQNAgent
from config import CONFIG
import pygetwindow


# 创建游戏环境
env = GameEnv(window_name=CONFIG["game_window_name"])

# 激活窗口
window = pygetwindow.getWindowsWithTitle(CONFIG["game_window_name"])[0]
window.activate()

# 定义模型参数
num_actions = len(env.action_map)

# 创建模型
model = CNNModel(num_actions)

# 创建 DQN Agent
agent = DQNAgent(
    model=model,
    num_actions=num_actions,
    memory_size=CONFIG["memory_size"],
    gamma=CONFIG["gamma"],
    epsilon_start=CONFIG["epsilon_start"],
    epsilon_min=CONFIG["epsilon_min"],
    epsilon_decay=CONFIG["epsilon_decay"],
    input_shape=CONFIG["input_shape"]
)

# 加载训练好的模型，最好用绝对地址
agent.model = agent.load_model('改为需要测试的模型地址')

# 测试循环
num_test_episodes = 1  # 测试轮次数量

for episode in range(num_test_episodes):
    episode_start_time = time.time()
    state = env.reset()
    done = False
    total_reward = 0

    # 在一局中持续运行，直到游戏结束
    while not done:
        # 根据当前状态选择一个动作执行
        action,_ = agent.act(state)

        # 执行选定的动作，并接收环境的反馈
        next_state, reward, done, _ = env.step(action)

        # 更新当前状态，并累计奖励
        state = next_state
        total_reward += reward

    # 打印每轮的测试结果
    print(f"第{episode + 1}轮总奖励: {total_reward:.2f}", end="  ")

    # 打印每轮时间间隔
    episode_end_time = time.time()
    episode_duration = episode_end_time - episode_start_time
    minutes = int(episode_duration // 60)
    seconds = episode_duration % 60
    print(f"第{episode + 1}轮耗时{minutes}分{seconds:.1f}秒")
    # 操作计数器
    print(f"本局游戏操作计数：{env.get_action_count()}")

print("测试完成：")
print(f"人机胜利场数：{env.cpu_total_wins}")
print(f"玩家胜利场数：{env.player_total_wins}")
