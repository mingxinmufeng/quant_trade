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
# 更新数据（须以模块方式运行：src 内部为相对导入，直接 python src/main.py 会报 ImportError）
python -m src.main fetch

# 运行回测
python -m src.main backtest --start 2020-01-01 --end 2024-12-31

# 超参优化
python -m src.main optimize --strategy ma_rsi --trials 100
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

### 第三步：同步行情数据（日线 / 5 分钟 / 1 分钟）

**拉取优先级**：`pytdx`（通达信本地盘）→ `akshare` → `baostock` → `tushare`。
增量更新**优先 akshare**，且 akshare 内部二次容灾（东财 → 新浪 → 腾讯）。

落盘只保留三套原始周期：`data_store/daily`、`data_store/min5`、`data_store/min1`。
其余周期**按需在加载时生成、不落盘**：

- 周/月/季/年线由**日线** resample；
- 15/30/60 分钟由 **5 分钟**生成；其它分钟周期由 **1 分钟**生成。

> **数据仓库位置（每台机器自选，保证可移植/可复现）**：代码本身**不含机器专属路径**。
> `store_path` 缺省 `data_store`（相对运行目录），每台机器按需覆盖，优先级高→低：
> `--store-path`（CLI 一次性）> `QUANT_DATA__STORE_PATH`（环境变量）> `config.private.yaml`（本机，已 gitignore）> `config.yaml` > 内置默认。
> 推荐设**绝对路径**且放 **SSD、非云同步盘**；模板见 [`config.private.yaml.example`](config.private.yaml.example)。
> 换机器跑同样代码 → 数据落到**那台机器**配置的位置（不会沿用别人的盘）；结果复现取决于**数据内容**相同，与路径无关。

```bash
# 数据源连通性自测（先跑这个确认各源可用）
python -m src.data.fetcher --selftest

# 全市场增量更新（默认日线 + 5分钟 + 1分钟）
python -m src.data.fetcher

# 指定股票 / 指定周期
python -m src.data.fetcher --codes 000001.SZ 600519.SH 920819.BJ
python -m src.data.fetcher --codes 000001.SZ --freqs daily min5
```

```python
from src.data import DataFetcher

f = DataFetcher()                       # 自动寻径通达信目录，或在 config 设 data.tdx_path
f.update(["000001.SZ"])                 # 日线+5min+1min 增量
daily = f.load_daily("000001.SZ", "2024-01-01", "2024-12-31")
weekly = f.load_daily("000001.SZ", "2024-01-01", "2024-12-31", period="weekly")
m15 = f.load_minute("000001.SZ", "15", "2024-06-01", "2024-06-30")  # 由 5min resample
```

> ⚠️ **首次使用强烈建议先下载通达信本地数据，避免触发东财风控**
>
> 1. **优先方案（推荐）**：安装通达信客户端，下载并解压「沪深京日线 + 5 分钟 + 1 分钟
>    完整包」（见 https://www.tdx.com.cn/article/vipdata.html），按说明覆盖到
>    `{通达信目录}/vipdoc`。**盘后**首次全量读取完全走本地、零网络风控。
>    本系统会用本地 lday/fzline/minline 生成所有更高周期。
> 2. **备选方案**：订阅 **Tushare**（填 `TUSHARE_TOKEN`）用于初次拉取，配额稳定、抗风控。
> 3. 若直接用 akshare 首拉全市场，请保留默认的 `--throttle` 与年分片防风控，并尽量在盘后运行。

### 复权因子（统一拉取）

所有数据源只返回**不复权 OHLCV**，复权因子由系统**统一**按股票拉取一次，并以同一张
「日期→累计后复权因子」表**同时作用于日线和分钟线**（按日期对齐），保证多周期一致。
因子源容灾：新浪 `hfq-factor` → tushare `adj_factor` → 东财后复权/不复权比值。
落盘 OHLC 为后复权价，`adj_factor` 为累计因子，回测可直接用 `close`。

### 代理说明（重要）

本系统**不再自动绕过/设置系统代理**。若运行时报「代理错误 / ProxyError / 无法连接到代理」，
说明你的系统代理或 VPN/加速器不可达，请执行任一操作后重试：

1. **关闭** Windows 系统代理 / VPN / 加速器；
2. 或运行前设置环境变量直连：
   - PowerShell：`$env:NO_PROXY="*"`
   - cmd：`set NO_PROXY=*`

发生代理类错误时程序会抛出 `ProxyConfigError` 并打印上述处置提示。

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
python -m src.main optimize \
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
├── README.md                       # 项目说明
├── pyproject.toml                  # 包配置 / 构建 / 工具链（ruff/black/mypy/pytest/coverage）
├── requirements.txt                # 依赖清单
├── config.yaml                     # 配置文件
├── .env.example                    # 环境变量模板
├── .gitignore                      # Git 忽略规则
├── Dockerfile                      # 容器化部署
│
├── src/                            # 源代码（import 名 a_share_quant_pro，见 pyproject 映射）
│   ├── main.py                     # CLI 入口（typer，console_scripts: quant）
│   │
│   ├── data/                       # 数据层
│   │   ├── trading_calendar.py     # 交易日历（akshare + 本地缓存，可离线）
│   │   ├── universe.py             # 动态股票池（防幸存者偏差）
│   │   ├── fetcher.py              # 多源拉取 + 增量更新编排（核心）
│   │   ├── processor.py            # 清洗 / 对齐 / 复权编排
│   │   ├── adjust.py               # 按需复权（前/后/不复权换算）
│   │   ├── factors.py              # 复权因子提供器（新浪/tushare/东财容灾）
│   │   ├── gbbq.py                 # 通达信本地权息读取 + 公司行为事件解析
│   │   ├── resample.py             # 周期重采样（日→周/月；5min→15/30/60min）
│   │   ├── suspend.py              # 停牌名单 Provider（整市场交易日快照）
│   │   ├── storage.py              # 本地 Parquet 数据仓库读写
│   │   └── sources/                # 数据源适配器（统一返回不复权 OHLCV）
│   │       ├── base.py             #   DataSourceBase 抽象基类
│   │       ├── pytdx_source.py     #   通达信本地盘（首选，零风控）
│   │       ├── akshare_source.py   #   akshare（东财/新浪/腾讯二次容灾）
│   │       ├── baostock_source.py  #   baostock（备用）
│   │       └── tushare_source.py   #   tushare（备用，需 token）
│   │
│   ├── factor/                     # 因子层
│   │   ├── base.py                 # FactorBase 抽象基类
│   │   ├── technical.py            # 技术指标因子（pandas-ta-classic）
│   │   └── fundamental.py          # 通用财务因子框架
│   │
│   ├── strategy/                   # 策略层（基类+示例公开，私有策略外部加载）
│   │   ├── base.py                 # BaseStrategy 抽象基类 + Signal 枚举
│   │   ├── loader.py               # 外部私有策略加载器
│   │   └── examples/
│   │       └── ma_rsi.py           # 双均线 + RSI 示例策略
│   │
│   ├── engine/                     # 回测引擎层
│   │   ├── portfolio.py            # 持仓与资金管理
│   │   ├── execution.py            # 撮合引擎（T+1/涨跌停/滑点/成交量/分红）
│   │   └── backtester.py           # 回测主引擎 + 绩效评估
│   │
│   ├── risk/                       # 风控层
│   │   └── risk_manager.py         # 仓位 / 止损 / 手续费 / 印花税
│   │
│   └── utils/                      # 公共工具
│       ├── config_loader.py        # 配置加载（config.yaml + 外部覆盖）
│       └── helpers.py              # 重试装饰器 / 日志初始化 / 工具函数
│
├── tests/                          # 测试套件（镜像 src + 数据完整性扫描）
│   ├── conftest.py
│   ├── data/                       # test_calendar / test_fetcher
│   ├── engine/                     # test_execution / test_backtester
│   ├── strategy/                   # test_loader
│   ├── test_adjust.py              # 复权换算
│   ├── test_gbbq_factor.py         # 通达信权息 / 复权因子
│   ├── test_suspend.py             # 停牌名单
│   └── test_data_integrity.py      # 本地 data_store 完整性扫描（脚本 + pytest 双用）
│
├── scripts/                        # 运维 / 数据工具脚本
├── data_store/                     # 本地数据仓库（Git 忽略）
└── logs/                           # 运行日志（Git 忽略）
```

## 许可证

MIT License
