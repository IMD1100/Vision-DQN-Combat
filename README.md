# 基于机器视觉的自动战斗系统

一个基于卷积神经网络（CNN）和深度Q网络（DQN）的格斗游戏自动战斗系统，应用于《死神vs火影》游戏。系统通过实时捕捉游戏画面，提取特征并生成智能战斗策略。

---

## 🚀 主要技术
- **图像处理**：使用OpenCV实时截取游戏画面并进行预处理。
- **特征提取**：基于三层卷积神经网络（CNN）提取游戏状态特征。
- **决策模型**：深度Q网络（DQN）实现动作选择，采用ε-greedy策略平衡探索与利用。
- **控制执行**：通过模拟键盘/鼠标输入控制游戏角色动作。
- **强化学习框架**：采用TensorFlow框架搭建DQN模型，支持经验回放和实时训练。

---

## 📥 安装与依赖
1. **克隆仓库**：
   ```bash
   git clone https://github.com/your-repo/auto-combat-system.git
   cd auto-combat-system

2. **安装依赖**：
   ```bash
   pip install -r requirements.txt

3. **配置游戏环境**：
   - 在运行主程序之前，确保先启动《死神vs火影》3.0（bvn_3.0），并手动选择到SINGLE VS PEOPLE模式或SINGLE VS CPU模式选择角色界面，取决于实际训练需求，在environment/game_env中更改。
   - 使用游戏默认分辨率，系统分辨率推荐设置为2560x1600，如需更改，需同时更改部分窗口大小参数。

---

## 🎮 使用说明

### 1. 参数配置
在 `config.yaml` 中调整以下参数：
```yaml
# 训练参数
    "game_window_name": "Adobe Flash Player 17", # 游戏窗口名，用于识别游戏窗口
    "memory_size": 10000, # 记忆池大小，存储经验回放的数据量
    "gamma": 0.98, # 折扣因子，用于计算未来奖励的现值
    "epsilon_start": 1.0, # epsilon贪心策略的起始值
    "epsilon_min": 0.01, # epsilon的最小值，防止过度拟合
    "epsilon_decay": 0.98, # epsilon的衰减率，控制探索与利用的平衡
    "num_episodes": 10, # 训练的轮次数量
    "batch_size": 32, # 小批量梯度下降的批次大小
    "input_shape": (427, 596, 3), # 输入图像的形状 (height, width, channels)
    "model_filename": "datas/model", # 训练好的模型权重保存路径
```

### 2. 启动系统
运行主程序：
```bash
python main.py
```

### 3. 训练模式
- **静态对手训练**：  
  默认模式下，AI对战静止对手学习基础策略。
  选择SINGLE VS PEOPLE模式，默认配置。
- **动态对手训练**：  
  对战游戏内置AI，适应复杂环境。
  选择SINGLE VS CPU模式，同时在environment/game_env中更改reset模块。

### 4. 实时对战
- 实时进行游戏并提取画面进行训练。

### 5. 监控与日志
- 训练日志保存在 `datas/训练日志`。
- 模型保存在 `datas/`。
- 在训练完成时生成评估指标，包括总奖励，平均Q值，LOSS。

---

## 🛠️ 核心模块说明
1. **卷积神经网络模块**：  
   通过卷积层的前向传播机制提取画面特征。
   
2. **动作决策模块**：  
   基于DQN选择动作，支持经验回放和模型保存。

3. **奖励系统模块**：  
   根据角色血量、攻击命中率等计算即时奖励。

---

## 📊 性能优化
- 若训练速度慢，可减少`batch_size`。
- 若内存不足，可减少memor_size大小。
- 调整`epsilon_decay`参数控制探索速率。
- 使用GPU加速需安装`tensorflow-gpu`并配置CUDA环境。

---

