import time
import cv2
import numpy as np
import pyautogui
import pygetwindow
from config import CONFIG

class GameEnv:
    def __init__(self, window_name="游戏窗口标题"):
        self.window_name = window_name
        self.window_rect = None
        self.player_previous_health = None # 记录上一帧玩player（即训练模型）血量
        self.bot_previous_health = None # 记录上一帧cpu血量
        self.done_template = cv2.imread(r'environment\KO.png')
        self.pic1 = cv2.imread(r'environment\down.png')
        self.pic2 = cv2.imread(r'environment\up.png')
        self.player_total_wins = 0 # 记录一次训练中player（即训练模型）总胜场
        self.cpu_total_wins = 0 # 记录一次训练中CPU总胜场
        self.player_won = False  # 判断当前player（即训练模型）是否胜利，用来判断胜利给予的奖励
        self.plus = 0 # 记录并增加重复操作扣除的奖励值
        self.last_action = None # 记录上一次操作，奖励判断用
        self.player_win = 0 # 判断单局游戏player（即训练模型）胜场，用来判断is_done
        self.cpu_win = 0 # 判断单局游戏cpu胜场，用来判断is_done

        self.action_map = {
            0: "A",  # 左移动
            1: "D",  # 左移动
            2: "S",   # 防御
            3: "J",   # 近攻
            4: "K",   # 跳
            5: "L",   # 冲刺
            6: "U",   # 远攻
            7: "I",   # 必杀
            8: "O",   # 召唤援助
            9: "WJ",  # W+J
            10: "SJ",  # S+J
            11: "KJ",  # K+J
            12: "WU", # W+U
            13: "SU", # S+U
            14: "KU",  # K+U
            15: "WI", # W+I
            #15: "SK", # S+K
            16: "SL", # SL
            17: "WL", # WL
            18: ".", #无操作
        }

        self.action_count = {action: 0 for action in self.action_map.values()}  # 动作编号计数器

    def get_action_count(self):
        """获取动作编号计数器，并按动作编号排序，同时显示编号对应的按键"""
        sorted_action_count = dict(sorted(self.action_count.items(), key=lambda item: item[0]))
        action_details = {self.action_map[action]: count for action, count in sorted_action_count.items()}
        return action_details

    def find_window(self):
        """查找游戏窗口"""
        try:
            self.window_rect = pygetwindow.getWindowsWithTitle(self.window_name)[0]
            self.window_rect = (
                self.window_rect.left,
                self.window_rect.top,
                self.window_rect.width,
                self.window_rect.height,
            )
            return True
        except IndexError:
            print(f"未找到名为 '{self.window_name}' 的窗口。")
            return False

    def get_screenshot(self, save_path=None):
        """获取游戏窗口截图，并进行预处理"""
        if self.window_rect is None:
            if not self.find_window():
                return None

        region = (18, 118, 1210, 973)
        screenshot = pyautogui.screenshot(region=self.window_rect)
        screenshot = np.array(screenshot)
        screenshot = cv2.cvtColor(screenshot, cv2.COLOR_RGB2BGR)
        screenshot = screenshot[region[1]:region[3], region[0]:region[2]]

        # 预处理图像，进行缩放等
        screenshot = self.preprocess_image(screenshot)

        if save_path:
            cv2.imwrite(save_path, screenshot)  # 保存截图

        return screenshot

    def preprocess_image(self, image):
        """图像预处理（缩放、归一化等）"""
        return cv2.resize(image, (image.shape[1] // 2, image.shape[0] // 2))  # 将图像缩小

    def send_key(self, key):
        """模拟按键操作"""
        pyautogui.press(key, interval=0.01)

    def send_action(self, action):
        """根据动作映射发送一系列按键操作"""
        action_str = self.action_map[action]

        if len(action_str) == 2:
            pyautogui.keyDown(action_str[0])
            pyautogui.press(action_str[1])
            pyautogui.keyUp(action_str[0])
        if len(action_str) >= 3:
            for key in action_str:
                time.sleep(0.15)
                self.send_key(key)
        if 2 >= action >= 0:
            pyautogui.keyDown(action_str)
            time.sleep(0.15)
            pyautogui.keyUp(action_str)
        if 18 >= action >= 3:
            self.send_key(action_str)
        # 更新动作编号计数器
        self.action_count[action] += 1

    def reset(self):
        """重置游戏状态"""
        self.action_count = {action: 0 for action in self.action_map.keys()}  # 动作编号计数器
        self.plus = 0

        # #玩家 vs cpu模式训练
        # # 选择player人物
        # self.send_key('W')
        # self.send_key('W')
        # self.send_key('W')
        # self.send_key('J')
        # time.sleep(0.2)
        # # 选择cpu人物
        # #self.send_key('A')
        # self.send_key('J')
        # time.sleep(1)
        # # 选择player辅助
        # self.send_key('A')
        # self.send_key('J')
        # time.sleep(0.5)
        # # 选择cou辅助
        # self.send_key('D')
        # self.send_key('D')
        # self.send_key('J')
        # time.sleep(0.8)

        # 玩家vs玩家模式训练
        self.send_key('W')
        self.send_key('W')
        self.send_key('W')
        self.send_key('J')

        self.send_key('num1')
        time.sleep(1)
        self.send_key('A')
        self.send_key('J')
        self.send_key('num1')
        time.sleep(0.8)

        #选择地图
        for _ in range(4):
            self.send_key('D')
        self.send_key('J')

        self.player_previous_health = None
        self.bot_previous_health = None
        self.player_win = 0
        self.cpu_win = 0

        # 跳过动画
        time.sleep(5)
        self.send_key('J')

        return self.get_screenshot()

    def step(self, action):
        # 激活窗口
        window = pygetwindow.getWindowsWithTitle(CONFIG["game_window_name"])[0]
        window.activate()

        """执行一个动作并返回新的游戏状态、奖励、是否结束等信息"""
        next_state = self.get_screenshot()
        reward = self.get_reward(action)  # 奖励
        done = self.is_done()  # 是否结束
        if not done:
            self.send_action(action)
            #print(f"action:{self.action_map.get(action)}  reward:{reward}")
        if done:
            time.sleep(9.5)
        return next_state, reward, done, {}

    def parse_game_state(self, screenshot):
        """解析游戏状态，提取玩家和机器人血量"""
        player_health_region = (70, 3, 274, 20)
        bot_health_region = (318, 2, 520, 18)

        player_health_area = screenshot[player_health_region[1]:player_health_region[3],
                             player_health_region[0]:player_health_region[2]]
        bot_health_area = screenshot[bot_health_region[1]:bot_health_region[3],
                          bot_health_region[0]:bot_health_region[2]]

        lower_green = np.array([0, 100, 0])
        upper_green = np.array([100, 255, 100])

        player_mask = cv2.inRange(player_health_area, lower_green, upper_green)
        bot_mask = cv2.inRange(bot_health_area, lower_green, upper_green)

        player_health = np.sum(player_mask > 0)
        bot_health = np.sum(bot_mask > 0)

        return player_health, bot_health

    def get_reward(self, action):
        """计算并返回奖励值"""
        current_screenshot = self.get_screenshot()
        if current_screenshot is None:
            return 0

        player_current_health, bot_current_health = self.parse_game_state(current_screenshot)

        if self.player_previous_health is None:
            self.player_previous_health = player_current_health
            return 0

        if self.bot_previous_health is None:
            self.bot_previous_health = bot_current_health
            return 0

        player_health_diff = self.player_previous_health - player_current_health
        bot_health_diff = self.bot_previous_health - bot_current_health

        reward = 0
        if player_health_diff > 0:
            player_health_percentage = player_health_diff / 1835
            reward -= player_health_percentage * 50  # 根据血量减少比例惩罚

        if bot_health_diff > 0:
            bot_health_percentage = bot_health_diff / 1818
            reward += bot_health_percentage * 100  # 根据机器人血量减少比例奖励

        current_screenshot = self.get_screenshot()
        if current_screenshot is None:
            return False

        # 检测在没有能量时使用需要能量的操作扣分
        region1 = (175, 400, 186, 410)
        current_screenshot1 = current_screenshot[region1[1]:region1[3], region1[0]:region1[2]]

        pic1 = self.pic1[region1[1]:region1[3], region1[0]:region1[2]]
        result = cv2.matchTemplate(current_screenshot1, pic1, cv2.TM_CCOEFF_NORMED)
        min_val, max_val1, min_loc, max_loc = cv2.minMaxLoc(result)

        region2 = (200, 41, 210, 51)
        current_screenshot2 = current_screenshot[region2[1]:region2[3], region2[0]:region2[2]]

        pic2 = self.pic2[region2[1]:region2[3], region2[0]:region2[2]]
        result = cv2.matchTemplate(current_screenshot2, pic2, cv2.TM_CCOEFF_NORMED)
        min_val, max_val2, min_loc, max_loc = cv2.minMaxLoc(result)

        if max_val1 > 0.6 or max_val2 > 0.6:
            if action ==7 or action == 8 or action == 15:
                reward -= 0.5

        # 判断动作编号并扣分。重复操作越多扣分越多，否则重置。
        # 移动和防御
        if 0 <= action <= 2:
            if self.last_action == 0 or self.last_action == 1 or self.last_action == 2:
                self.plus += 0.2
            else:
                self.plus = 0
            reward -= self.plus

        # 其他操作
        if 3 <= action <= 17:
            if self.last_action == action:
                self.plus += 0.3
            else:
                self.plus = 0
            reward -= 0.1 + self.plus

        # 无操作
        if action == 18:
            if self.last_action == action:
                self.plus += 0.2
            else:
                self.plus = 0
            reward -= self.plus

        # 检查player是否胜利
        if self.player_won:
            reward += 0 # 分值
            self.player_won = False  # 重置标志

        self.player_previous_health = player_current_health
        self.bot_previous_health = bot_current_health
        self.last_action = action

        return reward

    def is_done(self):
        """检查游戏是否结束"""
        current_screenshot = self.get_screenshot()
        if current_screenshot is None:
            return False

        region = (215, 89, 235, 169)
        current_screenshot1 = current_screenshot[region[1]:region[3], region[0]:region[2]]
        done_template3 = self.done_template[region[1]:region[3], region[0]:region[2]]

        result = cv2.matchTemplate(current_screenshot1, done_template3, cv2.TM_CCOEFF_NORMED)
        min_val, max_val3, min_loc, max_loc = cv2.minMaxLoc(result)

        # 设置匹配阈值
        threshold = 0.5

        player_health, cpu_health = self.parse_game_state(current_screenshot)
        if max_val3 >= threshold:
            if player_health > cpu_health:
                if self.player_win == 0:
                    self.player_win += 1
                    print("player", end=" ")
                    time.sleep(9.5)
                else:
                    print("player胜利")
                    self.player_total_wins += 1
                    return True
            if cpu_health > player_health:
                if self.cpu_win == 0:
                    self.cpu_win += 1
                    print("cpu", end=" ")
                    time.sleep(9.5)
                else:
                    print("cpu胜利")
                    self.cpu_total_wins += 1
                    return True

        return False

