# -*- coding: utf-8 -*-
"""
SimplePathPlanner —— 程序入口
==============================

启动 GUI 画布窗口，并在后台线程中运行终端命令交互循环。

运行方式：
  python main.py
"""

import threading
from canvas import GridCanvas
import matplotlib.pyplot as plt


def main():
    """
    程序主入口函数。

    执行流程：
      1. 创建 GridCanvas 实例（含图形窗口初始化和首次绘制）
      2. 启动 daemon 线程，在后台运行终端命令行交互循环
         daemon=True 确保主线程退出时该线程自动终止
      3. 主线程进入 Matplotlib 事件循环（plt.show(block=True)），响应 GUI 事件
         直到用户关闭图形窗口，程序退出
    """
    app = GridCanvas()
    terminal_thread = threading.Thread(target=app.start_terminal_loop, daemon=True)
    terminal_thread.start()
    plt.show(block=True)                                    # 阻塞主线程，进入 Matplotlib 事件循环


if __name__ == "__main__":
    main()
