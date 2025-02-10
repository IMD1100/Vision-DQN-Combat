from datetime import datetime
import tensorflow as tf
import numpy as np
import random
from collections import deque

# 打印可用的GPU数量
print("GPU可用数量: ", len(tf.config.list_physical_devices('GPU')))

class ReplayMemory:

    """
    经验回放内存类
    :param capacity: 最大内存容量
    """
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)

    """
    向内存中添加经验
    :param experience: 经验元组
    """
    def push(self, experience):
        self.memory.append(experience)

    """
    从内存中随机采样一批经验
    :param batch_size: 批量大小
    :return: 经验批次
    """
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    """
    返回当前内存中的经验数量
    :return: 经验数量
    """
    def __len__(self):
        return len(self.memory)


class DQNAgent:
    """
    DQN智能体类
    :param model: 模型对象
    :param num_actions: 动作数量
    :param memory_size: 内存大小
    :param gamma: 折扣因子
    :param epsilon_start: epsilon起始值
    :param epsilon_min: epsilon最小值
    :param epsilon_decay: epsilon衰减率
    :param input_shape: 输入图像的形状 (height, width, channels)
    """
    def __init__(self, model, num_actions, memory_size, gamma, epsilon_start, epsilon_min, epsilon_decay, input_shape):
        self.model = model
        self.num_actions = num_actions
        self.memory = ReplayMemory(memory_size)
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.input_shape = input_shape

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.loss_fn = tf.keras.losses.Huber()
        self.loss_history = []  # 新增一个列表记录 loss

        # 自动构建模型
        if not self.model.built:
            self.model.build(self.input_shape)



    """
    根据当前状态选择动作
    :param state: 当前状态
    :return: 选择的动作
    """

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            action = np.random.randint(self.num_actions)
            q_values = None  # 随机选择时，Q值不可用
        else:
            state = np.expand_dims(state, axis=0)  # Add batch dimension
            state = tf.cast(state, tf.float32) / 255.0  # 转换为float32并归一化
            q_values = self.model(state).numpy()  # 转为 numpy 数组
            action = np.argmax(q_values[0])  # 选择 Q 值最大的动作
        return action, q_values

    """
    将经验添加到内存中
    :param state: 当前状态
    :param action: 当前动作
    :param reward: 当前奖励
    :param next_state: 下一个状态
    :param done: 是否完成
    """
    def remember(self, state, action, reward, next_state, done):
        self.memory.push((state, action, reward, next_state, done))

    """
    使用经验回放对智能体进行训练
    :param batch_size: 批量大小
    """
    def train(self, batch_size):
        if len(self.memory) < batch_size:
            return

        batch = self.memory.sample(batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_states = np.array(next_states)
        dones = np.array(dones)

        with tf.GradientTape() as tape:
            states = tf.cast(states, tf.float32) / 255.0
            q_values = self.model(states)
            next_states = tf.cast(next_states, tf.float32) / 255.0
            next_q_values = self.model(next_states)

            target_q_values = rewards + (1 - dones) * self.gamma * tf.reduce_max(next_q_values, axis=1)
            q_values = tf.gather_nd(q_values, tf.stack([tf.range(batch_size), actions], axis=1))

            loss = self.loss_fn(target_q_values, q_values)

        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        self.loss_history.append(loss.numpy())  # 将 loss 转为 numpy 并记录

    """
       保存模型权重
       :param filename: 文件名
       """

    def save_model(self, filename, num_episodes):
        # 获取当前的时间戳
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M")
        # 拼接文件名，添加时间戳
        model_filename = f"{filename}_episode{num_episodes}_{current_time}.h5"
        # 保存模型权重
        self.model.save_weights(model_filename)
    """
    加载模型权重
    :param filename: 文件名
    """

    def load_model(self, filename):
        # 先调用模型一次以创建变量
        dummy_input = tf.zeros((1,) + self.input_shape)
        _ = self.model(dummy_input)
        # 加载权重
        self.model.load_weights(filename)



