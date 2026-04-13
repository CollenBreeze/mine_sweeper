# （持续更新中）这是一个通过机器学习来实现AI玩扫雷的项目,没有设置loss函数，有一个老师来指导他学什么难度的题。

## 父分支为纯手写，在这个分支我将用chatgpt优化界面和用户交互逻辑
## 如果你想体验我的项目，你需要准备：
###  0. 新建一个项目，并把该分支的全部文件放到里面
###  1.（如果不想训练可以跳过此步）：
    运行 check_gpu.py 来检查gpu环境是否配置好，如果版本不对，你需要下载安装适配你的GPU的pytorch（当然，cpu版本的也可以用，不过更慢）
    以2026年4月我的RTX5060为例，你需要在终端输入下面这行代码 :
    python -m pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu132
   
###  2.运行main.py
      大功告成了
