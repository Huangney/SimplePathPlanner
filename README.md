# SimplePathPlanner
## 快速开始
如果你是用于RC的路径规划，请在根目录运行如下命令：

### 蓝区
```bash
python ./main.py
```

### 红区
```bash
python ./main.py -p=1
```

>  如果出现缺少环境库，请自行使用pip安装

## 使用

运行后，终端会提示可运行的命令：
```
help                                        显示帮助信息
exit/q                                      退出程序
grid                                        重绘画布
addpoint x, y, theta[, vx, vy, vw]          添加路径点（网格坐标）
editpoint idx x, y, theta[, vx, vy, vw]     修改指定路径点（idx 从 1 开始）

注：坐标范围：x in [0,{GRID_HEIGHT}], y in [0,{GRID_WIDTH}]

plan                                            重新规划路径并打印摘要
density d                                       设置路径采样密度 (d >= 1.0)
speedcfg vmax=<v> amax=<a> wmax=<w> awmax=<aw>   设置全局速度约束
showpath on/off                                 切换路径曲线显示
save <文件>                                     保存当前路径点和设置到 JSON
load <文件>                                     从 JSON 加载路径点和设置
exportcpp <文件> [name=PathName] [scale=1.0]   导出 MCU C++ 路径头文件
```