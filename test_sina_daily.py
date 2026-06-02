"""测试新浪源日线数据接口"""
import akshare as ak
from datetime import date, timedelta

# 测试拉取平安银行最近30天数据
end_date = date.today()
start_date = end_date - timedelta(days=30)

print(f"测试新浪源日线数据：{start_date} → {end_date}")
print("=" * 60)

# 测试1：后复权日线
print("\n1. 测试后复权日线 (adjust='hfq')")
try:
    df_hfq = ak.stock_zh_a_daily(
        symbol="sz000001",
        start_date=start_date.strftime("%Y%m%d"),
        end_date=end_date.strftime("%Y%m%d"),
        adjust="hfq"
    )
    print(f"返回行数: {len(df_hfq)}")
    print(f"字段: {list(df_hfq.columns)}")
    print(df_hfq.head())
except Exception as e:
    print(f"失败: {e}")

# 测试2：不复权日线
print("\n2. 测试不复权日线 (adjust='')")
try:
    df_raw = ak.stock_zh_a_daily(
        symbol="sz000001",
        start_date=start_date.strftime("%Y%m%d"),
        end_date=end_date.strftime("%Y%m%d"),
        adjust=""
    )
    print(f"返回行数: {len(df_raw)}")
    print(f"字段: {list(df_raw.columns)}")
    print(df_raw.head())
except Exception as e:
    print(f"失败: {e}")

# 测试3：复权因子
print("\n3. 测试复权因子 (adjust='hfq-factor')")
try:
    df_factor = ak.stock_zh_a_daily(
        symbol="sz000001",
        start_date=start_date.strftime("%Y%m%d"),
        end_date=end_date.strftime("%Y%m%d"),
        adjust="hfq-factor"
    )
    print(f"返回行数: {len(df_factor)}")
    print(f"字段: {list(df_factor.columns)}")
    print(df_factor.head())
except Exception as e:
    print(f"失败: {e}")
