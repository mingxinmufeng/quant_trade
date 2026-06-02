"""
测试东方财富（akshare.stock_zh_a_hist）是否被风控
"""

from datetime import date, timedelta

def test_eastmoney_risk_control():
    """测试东财接口是否被风控"""
    import akshare as ak

    # 测试股票代码
    test_code = "000001"  # 平安银行
    today = date.today()

    print("=" * 60)
    print("东方财富风控测试 - 30天区间")
    print("=" * 60)

    # 测试：短区间（增量更新场景）
    print("\n[测试] 短区间拉取（最近30天）")
    start = today - timedelta(days=30)
    try:
        df = ak.stock_zh_a_hist(
            symbol=test_code,
            period="daily",
            start_date=start.strftime("%Y%m%d"),
            end_date=today.strftime("%Y%m%d"),
            adjust="hfq",
        )
        if df is None or df.empty:
            print("❌ 返回空数据 - 可能被风控")
        else:
            print(f"✅ 成功拉取 {len(df)} 条数据")
            print(f"   日期范围: {df['日期'].min()} ~ {df['日期'].max()}")
    except Exception as e:
        print(f"❌ 异常: {type(e).__name__}: {e}")

    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)

if __name__ == "__main__":
    test_eastmoney_risk_control()
