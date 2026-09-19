# solo-bank

账户与流水命令行工具，数据保存在工作目录下的本地文件中。

- 仅使用 Python 标准库，不联网。
- 入口：`python bank.py <子命令>`
- 金额以整数「分」存储；账户余额不允许为负。

## 测试

    python -m unittest discover
