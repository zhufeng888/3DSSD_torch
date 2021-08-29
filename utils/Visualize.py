import os
import logging


class Log():
    '''
    建立本地日志输出信息
    '''

    def __init__(self, save_path):
        # 第一步，创建一个logger
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.DEBUG)  # Log等级总开关

        # 第二步，创建一个handler，用于写入日志文件
        logfile = os.path.join(save_path, 'log.txt')  # 可以换成你自己想要生成log日志的路径
        self.fh = logging.FileHandler(logfile, mode='a')
        self.fh.setLevel(logging.DEBUG)  # 用于写到file的等级开关

        # 第三步，定义handler的输出格式
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.fh.setFormatter(formatter)

        # 第四步，将logger添加到handler里面
        self.logger.addHandler(self.fh)

    def close(self):
        # 第五步，移除某个创建的handler
        self.logger.removeHandler(self.fh)

    def out(self, context: str):
        print(context)
        self.logger.debug(context)


