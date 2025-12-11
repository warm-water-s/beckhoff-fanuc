import random
from collections import deque


# class ReplayBufferQue:
#     """DQN的经验回放池，每次采样batch_size个样本"""

#     def __init__(self, capacity: int) -> None:
#         self.capacity = capacity
#         self.buffer = deque(maxlen=self.capacity)

#     def push(self, transitions):
#         """_summary_
#         Args:
#             trainsitions (tuple): _description_
#         """
#         self.buffer.append(transitions)

#     def sample(self, batch_size: int, sequential: bool = False):
#         if batch_size > len(self.buffer):
#             batch_size = len(self.buffer)
#         if sequential:  # sequential sampling
#             rand = random.randint(0, len(self.buffer) - batch_size)
#             batch = [self.buffer[i] for i in range(rand, rand + batch_size)]
#             return zip(*batch)
#         else:
#             batch = random.sample(self.buffer, batch_size)
#             return zip(*batch)

#     def clear(self):
#         self.buffer.clear()

#     def __len__(self):
#         return len(self.buffer)


# class PGReplay(ReplayBufferQue):
#     """PG的经验回放池，每次采样所有样本，因此只需要继承ReplayBufferQue，重写sample方法即可"""

#     def __init__(self):
#         self.buffer = deque()

#     def sample(self):
#         """sample all the transitions"""
#         batch = list(self.buffer)
#         return zip(*batch)

class PGReplay:
    """PG的经验回放池,每次采样所有样本,因此只需要重写sample方法即可"""

    def __init__(self, capacity: int = 10000):
        """
        初始化PG经验回放池
        :param capacity: 经验池的最大容量
        """
        self.capacity = capacity
        self.buffer = deque(maxlen=self.capacity)  # 用deque存储轨迹数据

    def push(self, transitions):
        """将一条轨迹数据推入缓冲池"""
        self.buffer.append(transitions)

    def sample(self, batch_size: int):
        """根据批次大小返回样本数据"""
        # 如果回放池中的样本小于batch_size，直接返回全部样本
        if batch_size > len(self.buffer):
            batch_size = len(self.buffer)
        
        # 随机抽取batch_size个样本
        batch = random.sample(self.buffer, batch_size)
        return zip(*batch)

    def clear(self):
        """清空缓冲池"""
        self.buffer.clear()

    def __len__(self):
        return len(self.buffer)