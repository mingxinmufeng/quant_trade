"""
A_Share_Quant_Pro - 生产级 A 股量化交易框架

包入口：通过 pyproject.toml 的 package-dir 映射，本目录（src/）
在安装后对应顶层包 `a_share_quant_pro`。

公开子包：
    a_share_quant_pro.utils      公共工具（日志、重试、代码格式化等）
    a_share_quant_pro.data       数据层（行情拉取/清洗/股票池/日历）
    a_share_quant_pro.factor     因子层（技术因子/财务因子框架）
    a_share_quant_pro.strategy   策略层（基类 + 加载器 + 示例）
    a_share_quant_pro.engine     回测引擎层（撮合/组合/回测）
    a_share_quant_pro.risk       风控层（仓位/止损/手续费）
"""

__version__ = "0.1.0"
__all__ = ["__version__"]
