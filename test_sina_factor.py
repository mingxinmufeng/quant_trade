"""
测试新浪后复权因子接口
"""
import time
import akshare as ak
from loguru import logger

logger.remove()
logger.add(
    lambda msg: print(msg, end=""),
    level="INFO",
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>\n"
)

logger.info("测试新浪后复权因子接口（ak.stock_zh_a_daily, adjust='hfq-factor'）...")

try:
    start = time.time()
    df = ak.stock_zh_a_daily(symbol="sz000001", start_date="20240501", end_date="20240530", adjust="hfq-factor")
    elapsed = time.time() - start
    logger.success(f"✓ 新浪后复权因子获取成功，耗时 {elapsed:.2f}秒，数据行数: {len(df)}")
    print("\n返回的列名:")
    print(df.columns.tolist())
    print("\n前5行数据:")
    print(df.head())
    print("\nhfq_factor 统计:")
    print(df['hfq_factor'].describe())
except Exception as e:
    logger.error(f"✗ 新浪后复权因子获取失败: {type(e).__name__}: {e}")
