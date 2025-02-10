import tensorflow as tf

class CNNModel(tf.keras.Model):
    """
    定义一个卷积神经网络模型，继承自Keras的Model类。

    Attributes:
        num_actions (int): 动作的数量，用于确定输出层的神经元数量。
    """

    def __init__(self, num_actions):
        """
        初始化CNNModel类的实例。

        Parameters:
            num_actions (int): 动作的数量。
        """
        super(CNNModel, self).__init__()
        self.outputs = None
        self.dense1 = None
        self.flatten = None
        self.conv3 = None
        self.conv2 = None
        self.conv1 = None
        self.num_actions = num_actions

    def build(self, input_shape):
        """
        构建卷积神经网络的结构。

        Parameters:
            input_shape (tuple): 输入数据的形状。
        """
        # 在这里定义和初始化权重等
        # 第一个卷积层，32个8x8的卷积核，步长为4x4，使用relu激活函数
        self.conv1 = tf.keras.layers.Conv2D(32, (8, 8), strides=(4, 4), activation='relu')
        # 第二个卷积层，64个4x4的卷积核，步长为2x2，使用relu激活函数
        self.conv2 = tf.keras.layers.Conv2D(64, (4, 4), strides=(2, 2), activation='relu')
        # 第三个卷积层，64个3x3的卷积核，步长为1x1，使用relu激活函数
        self.conv3 = tf.keras.layers.Conv2D(64, (3, 3), strides=(1, 1), activation='relu')
        # 扁平化层，用于将三维的卷积层输出转换为一维
        self.flatten = tf.keras.layers.Flatten()
        # 全连接层，512个神经元，使用relu激活函数
        self.dense1 = tf.keras.layers.Dense(512, activation='relu')
        # 输出层，神经元数量由num_actions决定，用于输出每个动作的价值
        self.outputs = tf.keras.layers.Dense(self.num_actions)

    def call(self, inputs, training=None, mask=None):
        """
        实现模型的前向传播过程。

        Parameters:
            inputs: 输入数据。

        Returns:
            输出每个动作的价值。
        """
        # 通过卷积层的堆叠，实现对输入数据的特征提取和价值预测
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.dense1(x)
        return self.outputs(x)
