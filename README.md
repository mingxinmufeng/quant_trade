# A_Share_Quant_Pro

A股量化交易系统 —— 生产级分层架构实现

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## 项目架构图

```
┌─────────────────────────────────────────────────────────────┐
│                      策略层 (Strategy)                       │
│              信号生成 · 参数管理 · 超参优化                    │
├─────────────────────────────────────────────────────────────┤
│                      因子层 (Factor)                         │
│              技术指标 · 多因子计算 · 特征工程                  │
├─────────────────────────────────────────────────────────────┤
│                      回测引擎层 (Engine)                     │
│              撮合执行 · 持仓管理 · 绩效评估                   │
├─────────────────────────────────────────────────────────────┤
│                      风控层 (Risk)                           │
│              仓位控制 · 止损管理 · 费用计算                   │
├─────────────────────────────────────────────────────────────┤
│                      数据层 (Data)                           │
│              数据获取 · 清洗复权 · 股票池管理                  │
└─────────────────────────────────────────────────────────────┘
```

## 快速开始

### 1. 环境安装（使用国内镜像）

```bash
# 创建虚拟环境
python -m venv venv

# 激活环境
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# 使用国内镜像安装依赖
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 2. 配置环境变量

```bash
# 复制模板文件
copy .env.example .env

# 编辑 .env 文件，填入 Tushare Token（可选）
TUSHARE_TOKEN=your_token_here
```

### 3. 运行回测

```bash
# 更新数据
python src/main.py fetch

# 运行回测
python src/main.py backtest --start 2020-01-01 --end 2024-12-31

# 超参优化
python src/main.py optimize --strategy ma_rsi --trials 100
```

## 数据同步三步走

### 第一步：同步交易日历

交易日历自动从 akshare 获取并缓存到本地，支持离线使用。

```python
from src.data import TradingCalendar

calendar = TradingCalendar()
print(calendar.get_trading_days("2024-01-01", "2024-12-31"))
```

### 第二步：同步股票基础信息

获取全市场股票列表、上市日期、行业分类等信息，防止幸存者偏差。

```python
from src.data import Universe

universe = Universe()
# 获取2024年1月1日的可交易股票池
stocks = universe.get_tradable_stocks("2024-01-01")
```

### 第三步：同步日线数据

支持多数据源容灾（akshare → baostock → tushare），增量更新模式。

```bash
# 更新全部股票数据
python src/main.py fetch

# 更新指定股票
python src/main.py fetch --codes 000001.SZ,600519.SH
```

## A股回测避坑专栏

### T+1 规则与资金冻结

- **买入当日**：持仓标记为冻结状态，当日不可卖出
- **次交易日**：冻结自动解除，可正常交易
- **卖出资金**：默认 T+1 解冻（可通过配置关闭）

### 涨跌停 / 一字板处理

- 买入信号遇一字板涨停 → 订单顺延至次日
- 卖出信号遇一字板跌停 → 订单顺延至次日
- 支持主板(±10%)、创业板/科创板(±20%)、北交所(±30%)的涨跌停限制

### 停牌处理

- 停牌期间不可发出新订单
- 已持仓在停牌期间冻结，复牌后自动恢复交易
- 数据字段 `is_suspended` 标记停牌状态

### 复权与分红引起的持仓数量调整

- 使用**后复权价格**进行回测计算
- 通过 `adj_factor` 变化率自动检测分红/送股事件
- 送股/转增：持仓数量按比例调整
- 现金红利：自动计入可用资金

### 幸存者偏差的消除方法

- 股票池基于**历史时间点**动态计算，而非当前可交易股票
- 自动剔除退市股票、ST股票、上市未满60天的新股
- 退市股票永久保留历史数据用于回测

### 滑点和成交量限制的重要性

- 默认启用 0.1% 滑点模型
- 单笔成交量限制为当日总量的10%
- 避免过度乐观的回测结果

## 策略开发指南

### 继承 BaseStrategy 开发新策略

```python
from src.strategy import BaseStrategy, Signal
import pandas as pd
from typing import Dict

class MyStrategy(BaseStrategy):
    """自定义策略示例"""
    
    def __init__(self, fast=5, slow=20, rsi_period=14):
        super().__init__()
        self.params = {
            "fast_period": fast,
            "slow_period": slow,
            "rsi_period": rsi_period
        }
        self.strategy_name = "MyStrategy"
    
    def generate_signals(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        生成交易信号
        
        Args:
            data: {code: DataFrame} 字典，包含日线数据
        
        Returns:
            DataFrame: index=date, columns=code, values ∈ {-1, 0, 1}
            -1=卖出, 0=持有, 1=买入
        """
        signals = {}
        
        for code, df in data.items():
            # 计算信号
            signal = pd.Series(0, index=df.index)
            
            # 金叉买入
            buy_cond = df["close"] > df["close"].rolling(self.params["fast_period"]).mean()
            signal[buy_cond] = 1
            
            # 死叉卖出
            sell_cond = df["close"] < df["close"].rolling(self.params["slow_period"]).mean()
            signal[sell_cond] = -1
            
            signals[code] = signal
        
        return pd.DataFrame(signals)
```

### 运行自定义策略

```python
from src.engine import Backtester
from src.utils import get_config

config = get_config()
backtester = Backtester(config)

strategy = MyStrategy(fast=10, slow=30)
result = backtester.run(strategy, start="2020-01-01", end="2024-12-31")

print(f"总收益率: {result.total_return:.2%}")
print(f"夏普比率: {result.sharpe_ratio:.2f}")
print(f"最大回撤: {result.max_drawdown:.2%}")
```

## Optuna 超参优化使用说明

### 命令行使用

```bash
python src/main.py optimize \
    --strategy ma_rsi \
    --start 2020-01-01 \
    --end 2023-12-31 \
    --trials 100 \
    --metric sharpe_ratio
```

### 在策略中定义搜索空间

```python
from optuna import Trial

class MyStrategy(BaseStrategy):
    
    @classmethod
    def get_optuna_params(cls, trial: Trial) -> dict:
        """定义 Optuna 搜索参数空间"""
        return {
            "fast_period": trial.suggest_int("fast_period", 5, 20),
            "slow_period": trial.suggest_int("slow_period", 20, 60),
            "rsi_period": trial.suggest_int("rsi_period", 10, 20),
            "rsi_upper": trial.suggest_float("rsi_upper", 60, 80),
            "rsi_lower": trial.suggest_float("rsi_lower", 20, 40),
        }
```

### 优化结果

优化完成后，最佳参数组合将保存在 `reports/optuna_best_params.json`，可直接用于实盘回测。

## 项目结构

```
.
├── README.md                 # 项目说明
├── requirements.txt          # 依赖清单
├── config.yaml              # 配置文件
├── .env.example             # 环境变量模板
├── .gitignore               # Git忽略规则
├── src/                     # 源代码
│   ├── data/               # 数据层
│   ├── factor/             # 因子层
│   ├── strategy/           # 策略层
│   ├── engine/             # 回测引擎层
│   ├── risk/               # 风控层
│   ├── utils/              # 工具函数
│   └── main.py             # CLI入口
├── tests/                   # 测试代码
└── data_store/              # 本地数据仓库（Git忽略）
```

## 许可证

MIT License
