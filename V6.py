import streamlit as st
import akshare as ak
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. 网页全局配置
st.set_page_config(page_title="量化分析工具", layout="wide")

# 处理中文字体显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def find_inflection_points(x, poly_coeff):
    """数学求导：锁定斜率为0的转折点"""
    deriv = np.polyder(poly_coeff)  # 一阶导数
    roots = np.roots(deriv)  # 令导数为0求根
    real_roots = roots[np.isreal(roots)].real
    return real_roots[(real_roots >= 0) & (real_roots <= len(x) - 1)]


def get_market_index(stock_code):
    """自动匹配大盘指数"""
    if stock_code.startswith('60') or stock_code.startswith('68'):
        return "sh000001"
    elif stock_code.startswith('00') or stock_code.startswith('30'):
        return "sz399001"
    return "sz899050" if stock_code.startswith(('8', '4')) else "sh000001"


# --- 侧边栏：交互控制 ---
st.sidebar.header("📊 核心参数")
start_date = st.sidebar.text_input("开始日期", "20251101")
end_date = st.sidebar.text_input("结束日期", "20251231")
stock_a = st.sidebar.text_input("个股代码 A", "002530")
index_b = st.sidebar.text_input("申万二级代码 B", "801074")
deg = st.sidebar.slider("拟合阶数 (平滑度调节)", 3, 15, 8)
run_btn = st.sidebar.button("✨ 执行量化拟合分析")

st.title("📈 股票相对强度与趋势拐点分析 (网页版)")
st.info("📊 **符号说明**：红色/橙色五角星 (★) 代表个股强弱势转换点；绿色三角形 (▲) 代表价格均线趋势反转点。")

if run_btn:
    try:
        with st.spinner('正在同步全量历史交易数据...'):
            # 数据抓取
            df_a = ak.stock_zh_a_hist(symbol=stock_a, start_date=start_date, end_date=end_date, adjust="hfq")
            df_a['date'] = pd.to_datetime(df_a['日期'])

            df_b = ak.index_hist_sw(symbol=index_b, period="day")
            df_b['date'] = pd.to_datetime(df_b['日期'])
            df_b = df_b[(df_b['date'] >= pd.to_datetime(start_date)) & (df_b['date'] <= pd.to_datetime(end_date))]

            idx_c = get_market_index(stock_a)
            df_c = ak.stock_zh_index_daily(symbol=idx_c)
            df_c['date'] = pd.to_datetime(df_c['date'])
            df_c = df_c[(df_c['date'] >= pd.to_datetime(start_date)) & (df_c['date'] <= pd.to_datetime(end_date))]

            # 对齐与计算
            data = pd.merge(df_a[['date', '收盘']], df_b[['date', '收盘']], on='date', suffixes=('_A', '_B'))
            data = pd.merge(data, df_c[['date', 'close']], on='date')
            data.columns = ['Date', 'Close_A', 'Close_B', 'Close_C']
            data = data.sort_values('Date').reset_index(drop=True)

            p0 = data.iloc[0]
            data['Diff_AB'] = ((data['Close_A'] - p0['Close_A']) / p0['Close_A']) - (
                        (data['Close_B'] - p0['Close_B']) / p0['Close_B'])
            data['Diff_AC'] = ((data['Close_A'] - p0['Close_A']) / p0['Close_A']) - (
                        (data['Close_C'] - p0['Close_C']) / p0['Close_C'])
            data['MA5'] = data['Close_A'].rolling(5).mean()

            # 绘图逻辑
            fig, ax1 = plt.subplots(figsize=(12, 7))
            x = np.arange(len(data))

            # --- 相对强度：A对B(行业) ---
            p_ab = np.polyfit(x, data['Diff_AB'], deg)
            f_ab = np.poly1d(p_ab)
            ax1.plot(x, f_ab(x), label=f"对行业强度({index_b})", color="#1f77b4", lw=2)
            for pt in find_inflection_points(x, p_ab):
                ax1.scatter(pt, f_ab(pt), color='red', marker='*', s=250, zorder=5)

            # --- 相对强度：A对C(大盘) ---
            p_ac = np.polyfit(x, data['Diff_AC'], deg)
            f_ac = np.poly1d(p_ac)
            ax1.plot(x, f_ac(x), label=f"对大盘强度({idx_c})", color="#ff7f0e", lw=2)
            for pt in find_inflection_points(x, p_ac):
                ax1.scatter(pt, f_ac(pt), color='darkorange', marker='*', s=250, zorder=5)

            # --- MA5趋势 (右轴) ---
            ax2 = ax1.twinx()
            ma5_clean = data.dropna(subset=['MA5'])
            x_ma = ma5_clean.index
            p_ma = np.polyfit(x_ma, ma5_clean['MA5'], deg)
            f_ma = np.poly1d(p_ma)
            ax2.plot(x_ma, f_ma(x_ma), label="MA5拟合趋势线", color="green", ls='--', alpha=0.6)
            for pt in find_inflection_points(x_ma, p_ma):
                ax2.scatter(pt, f_ma(pt), color='darkgreen', marker='^', s=150, zorder=5)

            # 注释与美化
            ax1.set_title(f"股票 {stock_a} 多维度趋势与转折点实时分析", fontsize=15)
            ax1.set_ylabel("相对增长率差值 (强弱度)")
            ax2.set_ylabel("MA5 价格参考 (趋势)", color="green")
            ax1.grid(True, linestyle=':', alpha=0.5)

            tick_idx = np.linspace(0, len(data) - 1, 10, dtype=int)
            ax1.set_xticks(tick_idx)
            ax1.set_xticklabels(data['Date'].dt.strftime('%m-%d').iloc[tick_idx], rotation=30)

            h1, l1 = ax1.get_legend_handles_labels()
            h2, l2 = ax2.get_legend_handles_labels()
            ax1.legend(h1 + h2, l1 + l2, loc='upper left', ncol=2)

            st.pyplot(fig)
            st.success("分析完成。请观察星号与三角形的交叠，判断买卖拐点。")

    except Exception as e:
        st.error(f"分析失败，请检查输入代码或网络连接: {e}")