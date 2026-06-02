"""
测试 Baostock 数据源是否可用
"""

from datetime import date, timedelta

def test_baostock():
    """测试 Baostock 接口"""
    import baostock as bs

    # 测试股票代码
    test_code = "000001.SZ"  # 平安银行
    baostock_code = "sz.000001"  # baostock 格式

    print("=" * 60)
    print("Baostock 数据源测试")
    print("=" * 60)

    # 登录
    print("\n[步骤1] 登录 Baostock")
    lg = bs.login()
    if lg.error_code != '0':
        print(f"❌ 登录失败: {lg.error_msg}")
        return
    print("✅ 登录成功")

    today = date.today()

    # 测试1：短区间（30天）
    print("\n[测试1] 短区间拉取（最近30天）")
    start = today - timedelta(days=30)
    fields = "date,open,high,low,close,volume,amount,adjustflag,tradestatus,preclose"

    rs = bs.query_history_k_data_plus(
        baostock_code, fields,
        start_date=start.strftime("%Y-%m-%d"),
        end_date=today.strftime("%Y-%m-%d"),
        frequency="d",
        adjustflag="1",  # 1=后复权
    )

    if rs.error_code != '0':
        print(f"❌ 查询失败: {rs.error_msg}")
    else:
        rows = []
        while rs.next():
            rows.append(rs.get_row_data())

        if not rows:
            print("❌ 返回空数据")
        else:
            print(f"✅ 成功拉取 {len(rows)} 条数据")
            print(f"   日期范围: {rows[0][0]} ~ {rows[-1][0]}")
            print(f"   最新一条: {rows[-1]}")

    # 测试2：中等区间（1年）
    print("\n[测试2] 中等区间拉取（最近1年）")
    start = today - timedelta(days=365)

    rs = bs.query_history_k_data_plus(
        baostock_code, fields,
        start_date=start.strftime("%Y-%m-%d"),
        end_date=today.strftime("%Y-%m-%d"),
        frequency="d",
        adjustflag="1",
    )

    if rs.error_code != '0':
        print(f"❌ 查询失败: {rs.error_msg}")
    else:
        rows = []
        while rs.next():
            rows.append(rs.get_row_data())

        if not rows:
            print("❌ 返回空数据")
        else:
            print(f"✅ 成功拉取 {len(rows)} 条数据")
            print(f"   日期范围: {rows[0][0]} ~ {rows[-1][0]}")

    # 测试3：长区间（5年）
    print("\n[测试3] 长区间拉取（最近5年）")
    start = today - timedelta(days=365*5)

    rs = bs.query_history_k_data_plus(
        baostock_code, fields,
        start_date=start.strftime("%Y-%m-%d"),
        end_date=today.strftime("%Y-%m-%d"),
        frequency="d",
        adjustflag="1",
    )

    if rs.error_code != '0':
        print(f"❌ 查询失败: {rs.error_msg}")
    else:
        rows = []
        while rs.next():
            rows.append(rs.get_row_data())

        if not rows:
            print("❌ 返回空数据")
        else:
            print(f"✅ 成功拉取 {len(rows)} 条数据")
            print(f"   日期范围: {rows[0][0]} ~ {rows[-1][0]}")

    # 登出
    print("\n[步骤2] 登出")
    bs.logout()
    print("✅ 登出成功")

    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)

if __name__ == "__main__":
    test_baostock()
