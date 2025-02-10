import sys
import time
import os
from datetime import datetime
from environment.game_env import GameEnv
from agent.model import CNNModel
from agent.agent import DQNAgent
from config import CONFIG
import pygetwindow
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 定义 Tee 类，支持同时输出到文件和控制台
class Tee:
    def __init__(self, *files):
        self.files = files
    def write(self, data):
        for file in self.files:
            file.write(data)
    def flush(self):
        for file in self.files:
            if not file.closed:
                file.flush()
    def close(self):
        for file in self.files:
            file.close()


# 创建一个简单的卷积模型来触发 cuDNN 加载
model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(64, 64, 3)),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
])
dummy_input = tf.random.normal([1, 64, 64, 3])  # 创建一个假输入
model(dummy_input)

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

# 加载已经训练好的模型，最好用绝对地址
#agent.load_model(r'替换成模型地址')

# 打开文件以保存输出
with open('datas/训练日志/训练日志.txt', 'w', encoding='utf-8') as f:
    # 将 sys.stdout 重定向到 Tee（同时输出到控制台和文件）
    sys.stdout = Tee(sys.stdout, f)

    train_start_time = time.time()
    print("开始训练")

    # 数据初始化
    num_episodes = CONFIG["num_episodes"]  # 训练轮次数量
    batch_size = CONFIG["batch_size"]  # 批次大小

    # 初始化数据存储
    q_values_log = []
    rewards_log = []

    # 开始训练循环
    for episode in range(num_episodes):
        episode_start_time = time.time()
        done = False
        total_reward = 0
        q_values_episode = []
        state = env.reset()
        last_action = 18
        # 在一局中持续运行，直到游戏结束
        while not done:
            # 根据当前状态选择一个动作执行，并获取Q值
            action, q_values = agent.act(state)
            #action =17
            # 执行选定的动作，并接收环境的反馈
            next_state, reward, done, _ = env.step(action)
            # 存储经验
            agent.remember(state, action, reward, next_state, done)

            # 当记忆中有足够的样本时，开始训练模型
            if len(agent.memory) > batch_size:
                agent.train(batch_size)

            # 如果 Q 值存在，记录其平均值
            if q_values is not None:
                avg_q_value = np.mean(q_values)
                q_values_episode.append(avg_q_value)

            # 更新当前状态，并累计奖励
            state = next_state
            total_reward += reward

        # 存储当前轮的 Q 值和奖励
        q_values_log.append(np.mean(q_values_episode))
        rewards_log.append(total_reward)

        # 每轮的训练结果
        print(f"第{episode + 1}轮总奖励: {total_reward:.2f}, 平均Q值: {np.mean(q_values_episode):.2f}", end="  ")

        # 每轮时间间隔
        episode_end_time = time.time()
        episode_duration = episode_end_time - episode_start_time
        minutes = int(episode_duration // 60)
        seconds = episode_duration % 60
        print(f"第{episode + 1}轮耗时{minutes}分{seconds:.1f}秒")

        # 操作计数器
        print(f"本局游戏操作计数：{env.get_action_count()}")

        # 每25轮保存一次模型
        if (episode + 1) % 2 == 0:
            agent.save_model(f"{CONFIG['model_filename']}", episode + 1)
            print(f"第{episode + 1}轮模型已保存", end="\n\n")

    print("本次训练中：")
    print(f"人机胜利场数：{env.cpu_total_wins}")
    print(f"玩家胜利场数：{env.player_total_wins}")

    # 本次训练时间
    train_end_time = time.time()
    train_duration = train_end_time - train_start_time
    minutes = int(train_duration // 60)
    seconds = train_duration % 60
    print(f"本次训练耗时{minutes}分{seconds:.1f}秒")
def draw():
    # 训练结束后，保存和绘制训练指标
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M")
    # 获取当前时间并格式化，用于重命名日志文件
    new_log_file_path = f'datas/训练日志/训练日志_{current_time}.txt'
    os.rename('datas/训练日志/训练日志.txt', new_log_file_path)

    # 绘制 Q值、奖励、和 Loss 曲线
    plt.figure(figsize=(12, 9))

    # 平均 Q 值曲线
    plt.subplot(3, 1, 1)
    plt.plot(q_values_log, label='Average Q-value', color='blue')
    plt.xlabel('Episode')
    plt.ylabel('Q-value')
    plt.legend()

    # 总奖励曲线
    plt.subplot(3, 1, 2)
    plt.plot(rewards_log, label='Total Reward', color='orange')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()

    # Loss 曲线
    plt.subplot(3, 1, 3)
    plt.plot(agent.loss_history, label='Loss', color='red')
    plt.xlabel('Training Step')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    metrics_plot_filename = f'datas/训练日志/训练指标变化_{current_time}.png'
    plt.savefig(metrics_plot_filename)
    plt.show()

draw()

