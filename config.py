# 配置字典，包含游戏仿真和神经网络训练的各种参数和设置
CONFIG = {
    "game_window_name": "Adobe Flash Player 17", # 游戏窗口名，用于识别游戏窗口
    "memory_size": 10000, # 记忆池大小，存储经验回放的数据量
    "gamma": 0.98, # 折扣因子，用于计算未来奖励的现值
    "epsilon_start": 1.0, # epsilon贪心策略的起始值
    "epsilon_min": 0.01, # epsilon的最小值，防止过度拟合
    "epsilon_decay": 0.98, # epsilon的衰减率，控制探索与利用的平衡
    "num_episodes": 1, # 训练的轮次数量
    "batch_size": 32, # 小批量梯度下降的批次大小
    "input_shape": (427, 596, 3), # 输入图像的形状 (height, width, channels)
    "model_filename": "datas/model", # 训练好的模型权重保存路径
}

