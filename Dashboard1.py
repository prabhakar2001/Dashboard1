import streamlit as st
import yfinance as yf
from datetime import datetime, timedelta
import pandas as pd
import io
from st_aggrid import AgGrid, GridOptionsBuilder
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from sklearn.linear_model import LinearRegression
import numpy as np

# Set page config
st.set_page_config(
    page_title="Nifty 200 Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS to fix the layout
st.markdown("""
    <style>
    /* Remove extra padding at the top */
    .main .block-container {
        padding-top: 10 !important;
    }
    </style>
    """, unsafe_allow_html=True)

# Sidebar for navigation
selected_tab = st.sidebar.radio(
    "Select a Dashboard",
    options=["📈 Stock Screener", "📊 Historical dashboard"],
    index=0
)

if selected_tab == "📈 Stock Screener":
    def generate_csv(data):
        output = io.StringIO()
        data.to_csv(output, index=False)
        return output.getvalue()

    @st.cache_data(ttl=3600)
    def get_stock_data(symbol, period='1y'):
        try:
            stock = yf.download(symbol, period=period)
            if not stock.empty:
                # Ensure proper timezone localization
                if stock.index.tzinfo is None:
                    stock.index = stock.index.tz_localize('UTC')
                else:
                    stock.index = stock.index.tz_convert('UTC')
            return stock
        except Exception as e:
            st.error(f"Error fetching data for {symbol}: {e}")
            return pd.DataFrame()

    @st.cache_data(ttl=3600)  # Cache full dataset for All-Time High
    def get_full_stock_data(symbol):
        return get_stock_data(symbol, period="max")

    def calculate_rsi(data, window=14):
        delta = data['Close'].diff(1)
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=window, min_periods=1).mean()
        avg_loss = loss.rolling(window=window, min_periods=1).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def calculate_ema(data, window=200):
        ema = data['Close'].ewm(span=window, adjust=False).mean()
        std_dev = data['Close'].rolling(window=window).std()
        return ema, std_dev

    def calculate_percentage_change(current_price, previous_price):
        return ((current_price - previous_price) / previous_price) * 100

    def calculate_monthly_change(data):
        if len(data) >= 21:  # Approx 21 trading days in a month
            month_ago_price = data['Close'].iloc[-21]
            current_price = data['Close'].iloc[-1]
            monthly_change = ((current_price - month_ago_price) / month_ago_price) * 100
            return round(monthly_change, 2)
        else:
            return None

    def calculate_3month_change(data):
        if len(data) >= 63:  # Approx 63 trading days in 3 months
            three_month_ago_price = data['Close'].iloc[-63]
            current_price = data['Close'].iloc[-1]
            three_month_change = ((current_price - three_month_ago_price) / three_month_ago_price) * 100
            return round(three_month_change, 2)
        else:
            return None

    def get_nearest_trading_day(data, target_date):
        if target_date in data.index:
            return target_date
        else:
            if data.empty:
                return None
            nearest_date = min(data.index, key=lambda d: abs(d - target_date))
            return nearest_date

    def check_conditions_and_get_percentage_change(stock_data):
        if stock_data.empty:
            return 0, 0, pd.Series(dtype=float)
        stock_data_weekly = stock_data.resample('W').agg({'Open': 'first', 'Close': 'last'})
        if len(stock_data_weekly) < 2:
            return 0, 0, pd.Series(dtype=float)
        last_week = stock_data_weekly.iloc[-2]
        current_week = stock_data_weekly.iloc[-1]
        percentage_change = ((last_week['Close'] - last_week['Open']) / last_week['Open']) * 100
        week_to_week_change = ((current_week['Close'] - last_week['Close']) / last_week['Close']) * 100
        return percentage_change, week_to_week_change, last_week

    def check_52_week_condition(stock_data):
        if len(stock_data) < 252:
            return False, None, None, None, None
        high_52_week = stock_data['High'].iloc[-252:].max()
        low_52_week = stock_data['Low'].iloc[-252:].min()
        range_52_week = high_52_week - low_52_week

        lower_bound = high_52_week - 0.60 * range_52_week
        upper_bound = high_52_week - 0.50 * range_52_week

        current_price = stock_data['Close'][-1]
        condition_met = current_price <= upper_bound and current_price >= lower_bound
        return condition_met, lower_bound, upper_bound, high_52_week, low_52_week

    def check_fibonacci_condition(stock_data, fib_level_1=0.618, fib_level_2=0.786):
        if len(stock_data) < 252:
            return False, None, None, None, None
        high_52_week = stock_data['High'].iloc[-252:].max()
        low_52_week = stock_data['Low'].iloc[-252:].min()
        fib_value_2 = high_52_week - fib_level_2 * (high_52_week - low_52_week)
        fib_value_1 = high_52_week - fib_level_1 * (high_52_week - low_52_week)
        current_price = stock_data['Close'][-1]
        condition_met = fib_value_2 <= current_price <= fib_value_1
        return condition_met, current_price, fib_value_2, fib_value_1, high_52_week

    def check_fibonacci_year_condition(stock_data, fib_level_1=0.618, fib_level_2=0.786):
        if len(stock_data) < 365:
            return False, None, None, None, None
        recent_data = stock_data[-365:]
        return check_fibonacci_condition(recent_data, fib_level_1, fib_level_2)

    def check_all_time_high_condition(stock_data_full):
        if stock_data_full.empty:
            return False, None, None, None, None
        ath = stock_data_full['High'].max()
        current_price = stock_data_full['Close'].iloc[-1]
        low_limit = ath * 0.50
        high_limit = ath * 0.70
        condition_met = low_limit <= current_price <= high_limit
        return condition_met, current_price, low_limit, high_limit, ath

    def check_52_week_high_yearly_condition(stock_data):
        if len(stock_data) < 252:
            return False, None, None, None
        high_52_week = stock_data['High'].iloc[-252:].max()
        current_price = stock_data['Close'][-1]
        low_limit = high_52_week * 0.50
        high_limit = high_52_week * 0.60
        condition_met = low_limit <= current_price <= high_limit
        return condition_met, low_limit, high_limit, high_52_week

    def check_rsi_condition(stock_data):
        if stock_data.empty:
            return False, None, None
        rsi_daily = calculate_rsi(stock_data)
        stock_data_weekly = stock_data.resample('W').agg({'Open': 'first', 'Close': 'last'})
        if stock_data_weekly.empty:
            return False, rsi_daily, pd.Series(dtype=float)
        rsi_weekly = calculate_rsi(stock_data_weekly)
        condition_met = (rsi_daily.iloc[-1] < 30) or (rsi_weekly.iloc[-1] < 30)
        return condition_met, rsi_daily, rsi_weekly

    def check_ema_condition(stock_data):
        if stock_data.empty:
            return False, pd.Series(dtype=float), None, None
        ema_200, std_dev = calculate_ema(stock_data)
        if len(ema_200) == 0 or len(std_dev) == 0:
            return False, ema_200, None, None
        current_price = stock_data['Close'][-1]
        lower_bound_1_std = ema_200.iloc[-1] - std_dev.iloc[-1]  # -1 Std Dev
        lower_bound_2_std = ema_200.iloc[-1] - 2 * std_dev.iloc[-1]  # -2 Std Dev
        condition_met = current_price < ema_200.iloc[-1]
        return condition_met, ema_200, lower_bound_1_std, lower_bound_2_std

    # Define stock symbols and sector mappings
    nifty_100_symbols = [
        'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'HINDUNILVR.NS',
        'ICICIBANK.NS', 'SBIN.NS', 'BHARTIARTL.NS', 'KOTAKBANK.NS', 'ITC.NS',
        'LT.NS', 'AXISBANK.NS', 'BAJFINANCE.NS', 'HCLTECH.NS', 'WIPRO.NS',
        'ASIANPAINT.NS', 'MARUTI.NS', 'TATASTEEL.NS', 'M&M.NS', 'ULTRACEMCO.NS',
        'SUNPHARMA.NS', 'NESTLEIND.NS', 'BAJAJFINSV.NS', 'POWERGRID.NS', 'NTPC.NS',
        'JSWSTEEL.NS', 'HDFCLIFE.NS', 'DIVISLAB.NS', 'ADANIGREEN.NS', 'TATAMOTORS.NS',
        'GRASIM.NS', 'TECHM.NS', 'BAJAJ-AUTO.NS', 'ADANIPORTS.NS', 'BRITANNIA.NS',
        'CIPLA.NS', 'SBILIFE.NS', 'SHREECEM.NS', 'HEROMOTOCO.NS', 'HINDALCO.NS',
        'TATACONSUM.NS', 'ONGC.NS', 'COALINDIA.NS', 'DRREDDY.NS', 'EICHERMOT.NS',
        'INDUSINDBK.NS', 'UPL.NS', 'HDFCAMC.NS', 'SBICARD.NS', 'ADANITRANS.NS',
        'ICICIPRULI.NS', 'BPCL.NS', 'TATAPOWER.NS', 'ADANIENT.NS', 'APOLLOHOSP.NS',
        'DLF.NS', 'GODREJCP.NS', 'AMBUJACEM.NS', 'PIDILITIND.NS', 'MUTHOOTFIN.NS',
        'TORNTPHARM.NS', 'BIOCON.NS', 'GAIL.NS', 'ADANITOTALGAS.NS', 'BANDHANBNK.NS',
        'LICHSGFIN.NS', 'MPHASIS.NS', 'LTTS.NS', 'PAGEIND.NS', 'BERGEPAINT.NS',
        'UNITDSPR.NS', 'ICICIGI.NS', 'MARICO.NS', 'HAL.NS', 'TATACOMM.NS',
        'ADANIWILMAR.NS', 'ZOMATO.NS', 'NYKAA.NS', 'POLICYBZR.NS', 'PAYTM.NS',
        'JUBLFOOD.NS', 'L&TFH.NS', 'ABCAPITAL.NS', 'IGL.NS', 'GUJGASLTD.NS',
        'SONACOMS.NS', 'MAXHEALTH.NS', 'MOTHERSON.NS', 'AUBANK.NS', 'PVR.NS',
        'SRF.NS', 'TATAELXSI.NS', 'CONCOR.NS', 'RECLTD.NS', 'NHPC.NS',
        'BEL.NS', 'LTIM.NS', 'MINDTREE.NS', 'ADANIPOWER.NS', 'IOC.NS',
        'PNB.NS', 'BANKBARODA.NS', 'CANBK.NS', 'SAIL.NS', 'MTARTECH.NS',
        'CLEAN.NS', 'HAPPSTMNDS.NS'
    ]

    sector_mapping = {
        'RELIANCE.NS': 'Energy',
        'TCS.NS': 'IT',
        'HDFCBANK.NS': 'Banking',
        'INFY.NS': 'IT',
        'HINDUNILVR.NS': 'Consumer Goods',
        'ICICIBANK.NS': 'Banking',
        'SBIN.NS': 'Banking',
        'BHARTIARTL.NS': 'Telecom',
        'KOTAKBANK.NS': 'Banking',
        'ITC.NS': 'Consumer Goods',
        'LT.NS': 'Infrastructure',
        'AXISBANK.NS': 'Banking',
        'BAJFINANCE.NS': 'Finance',
        'HCLTECH.NS': 'IT',
        'WIPRO.NS': 'IT',
        'ASIANPAINT.NS': 'Consumer Goods',
        'MARUTI.NS': 'Automobile',
        'TATASTEEL.NS': 'Metals',
        'M&M.NS': 'Automobile',
        'ULTRACEMCO.NS': 'Cement',
        'SUNPHARMA.NS': 'Pharmaceuticals',
        'NESTLEIND.NS': 'Consumer Goods',
        'BAJAJFINSV.NS': 'Finance',
        'POWERGRID.NS': 'Energy',
        'NTPC.NS': 'Energy',
        'JSWSTEEL.NS': 'Metals',
        'HDFCLIFE.NS': 'Insurance',
        'DIVISLAB.NS': 'Pharmaceuticals',
        'ADANIGREEN.NS': 'Energy',
        'TATAMOTORS.NS': 'Automobile',
        'GRASIM.NS': 'Diversified',
        'TECHM.NS': 'IT',
        'BAJAJ-AUTO.NS': 'Automobile',
        'ADANIPORTS.NS': 'Logistics',
        'BRITANNIA.NS': 'Consumer Goods',
        'CIPLA.NS': 'Pharmaceuticals',
        'SBILIFE.NS': 'Insurance',
        'SHREECEM.NS': 'Cement',
        'HEROMOTOCO.NS': 'Automobile',
        'HINDALCO.NS': 'Metals',
        'TATACONSUM.NS': 'Consumer Goods',
        'ONGC.NS': 'Energy',
        'COALINDIA.NS': 'Mining',
        'DRREDDY.NS': 'Pharmaceuticals',
        'EICHERMOT.NS': 'Automobile',
        'INDUSINDBK.NS': 'Banking',
        'UPL.NS': 'Agro Chemicals',
        'HDFCAMC.NS': 'Finance',
        'SBICARD.NS': 'Finance',
        'ADANITRANS.NS': 'Energy',
        'ICICIPRULI.NS': 'Insurance',
        'BPCL.NS': 'Energy',
        'TATAPOWER.NS': 'Energy',
        'ADANIENT.NS': 'Diversified',
        'APOLLOHOSP.NS': 'Pharmaceuticals',
        'DLF.NS': 'Realty',
        'GODREJCP.NS': 'Consumer Goods',
        'AMBUJACEM.NS': 'Cement',
        'PIDILITIND.NS': 'Chemicals',
        'MUTHOOTFIN.NS': 'Finance',
        'TORNTPHARM.NS': 'Pharmaceuticals',
        'BIOCON.NS': 'Pharmaceuticals',
        'GAIL.NS': 'Energy',
        'ADANITOTALGAS.NS': 'Energy',
        'BANDHANBNK.NS': 'Banking',
        'LICHSGFIN.NS': 'Finance',
        'MPHASIS.NS': 'IT',
        'LTTS.NS': 'IT',
        'PAGEIND.NS': 'Consumer Goods',
        'BERGEPAINT.NS': 'Consumer Goods',
        'UNITDSPR.NS': 'Beverages',
        'ICICIGI.NS': 'Insurance',
        'MARICO.NS': 'Consumer Goods',
        'HAL.NS': 'Defense',
        'TATACOMM.NS': 'Telecom',
        'ADANIWILMAR.NS': 'Consumer Goods',
        'ZOMATO.NS': 'Online Services',
        'NYKAA.NS': 'Retail',
        'POLICYBZR.NS': 'Insurance',
        'PAYTM.NS': 'Financial Services',
        'JUBLFOOD.NS': 'Consumer Goods',
        'L&TFH.NS': 'Finance',
        'ABCAPITAL.NS': 'Finance',
        'IGL.NS': 'Energy',
        'GUJGASLTD.NS': 'Energy',
        'SONACOMS.NS': 'Automobile',
        'MAXHEALTH.NS': 'Healthcare',
        'MOTHERSON.NS': 'Automobile',
        'AUBANK.NS': 'Banking',
        'PVR.NS': 'Media',
        'SRF.NS': 'Chemicals',
        'TATAELXSI.NS': 'IT',
        'CONCOR.NS': 'Logistics',
        'RECLTD.NS': 'Finance',
        'NHPC.NS': 'Energy',
        'BEL.NS': 'Capital Goods',
        'LTIM.NS': 'IT',
        'MINDTREE.NS': 'IT',
        'ADANIPOWER.NS': 'Energy',
        'IOC.NS': 'Energy',
        'PNB.NS': 'Banking',
        'BANKBARODA.NS': 'Banking',
        'CANBK.NS': 'Banking',
        'SAIL.NS': 'Metals',
        'MTARTECH.NS': 'Capital Goods',
        'CLEAN.NS': 'Chemicals',
        'HAPPSTMNDS.NS': 'IT'
    }

    # Indices
    nifty_indices = [
        '^NSEI', '^CNXIT', '^CNXAUTO', '^CNXINFRA', 'NIFTY_FIN_SERVICE.NS', '^CNXMETAL', '^NSEBANK', '^CNXENERGY', '^CNXFMCG', '^CNXPHARMA'
    ]

    index_sector_mapping = {
        '^NSEI': 'Nifty 50',
        '^CNXIT': 'IT',
        '^CNXAUTO': 'Automobile',
        '^CNXMETAL': 'Metals',
        '^NSEBANK': 'Banking',
        '^CNXENERGY': 'Energy',
        '^CNXFMCG': 'Consumer Goods',
        '^CNXPHARMA': 'Pharmaceuticals',
        'NIFTY_FIN_SERVICE.NS': 'Finance',
        '^CNXINFRA': 'Infrastructure'
    }

    # Streamlit UI for this tab
    st.title("📈 Nifty 100 Stock and Indices Screener and Chart")
    st.markdown(f"### Current Date: {datetime.now().strftime('%Y-%m-%d')}")

    # Sidebar for details
    selected_stock_sidebar = st.sidebar.selectbox("Select a stock or index to view details:", nifty_100_symbols + nifty_indices)
    selected_range = st.sidebar.selectbox("Select the range for percentage change:", ['1 Month', '3 Months', '6 Months', '1 Year'])
    if selected_range == '1 Month':
        period = '1mo'
    elif selected_range == '3 Months':
        period = '3mo'
    elif selected_range == '6 Months':
        period = '6mo'
    elif selected_range == '1 Year':
        period = '1y'

    stock_data_full_sidebar = get_full_stock_data(selected_stock_sidebar)
    data_sidebar = get_stock_data(selected_stock_sidebar, period=period)

    if not data_sidebar.empty:
        current_price_sidebar = round(data_sidebar['Close'].iloc[-1], 2)
        if not stock_data_full_sidebar.empty:
            ath_sidebar = round(stock_data_full_sidebar['High'].max(), 2)
        else:
            ath_sidebar = None

        # Calculate 52-week stats
        if len(data_sidebar) >= 252:
            high_52_week_sidebar = round(data_sidebar['High'].iloc[-252:].max(), 2)
            low_52_week_sidebar = round(data_sidebar['Low'].iloc[-252:].min(), 2)
        else:
            high_52_week_sidebar = data_sidebar['High'].max()
            low_52_week_sidebar = data_sidebar['Low'].min()

        if len(data_sidebar) > 1:
            todays_change = round(calculate_percentage_change(current_price_sidebar, data_sidebar['Close'][-2]), 2)
        else:
            todays_change = 0.0

        st.sidebar.write(f"### {selected_stock_sidebar} - Details")
        st.sidebar.write(f"**Current Market Price (CMP):** ₹{current_price_sidebar}")
        st.sidebar.write(f"**All-Time High (ATH) Price:** ₹{ath_sidebar}")
        st.sidebar.write(f"**52-Week High Price:** ₹{high_52_week_sidebar}")
        st.sidebar.write(f"**52-Week Low Price:** ₹{low_52_week_sidebar}")
        st.sidebar.write(f"**Today's Change:** {todays_change}%")

        # 1-Week Calculation
        one_week_ago_date = data_sidebar.index[-1] - timedelta(weeks=1)
        one_week_ago_date = get_nearest_trading_day(data_sidebar, one_week_ago_date)
        if one_week_ago_date in data_sidebar.index:
            one_week_ago_price = data_sidebar.loc[one_week_ago_date, 'Close']
            one_week_percentage_change = round(calculate_percentage_change(current_price_sidebar, one_week_ago_price), 2)
        else:
            one_week_percentage_change = None
        st.sidebar.write(f"**1-Week Percentage Change:** {one_week_percentage_change}%")

        # Current Week Percentage Change
        monday_date = data_sidebar.index[-1] - timedelta(days=data_sidebar.index[-1].weekday())
        monday_date = get_nearest_trading_day(data_sidebar, monday_date)
        if monday_date in data_sidebar.index:
            monday_price = data_sidebar.loc[monday_date, 'Open']
            current_week_percentage_change = round(calculate_percentage_change(current_price_sidebar, monday_price), 2)
        else:
            current_week_percentage_change = None
        st.sidebar.write(f"**Current Week Percentage Change:** {current_week_percentage_change}%")

        # Percentage change over selected range
        if not data_sidebar.empty:
            start_date = data_sidebar.index[0]
            end_date = data_sidebar.index[-1]
            if start_date != end_date:
                percentage_change_sidebar = round(calculate_percentage_change(data_sidebar.loc[end_date, 'Close'], data_sidebar.loc[start_date, 'Close']), 2)
            else:
                percentage_change_sidebar = 0.0
            st.sidebar.write(f"**Percentage Change ({selected_range}):** {percentage_change_sidebar}%")
        else:
            st.sidebar.write("No data available for the selected range.")

        # Sidebar conditions summary
        st.sidebar.markdown("### 📝 Main Conditions")
        main_conditions_list = [
            "📉 52-Week Range: Price in 40-50% down from 52-Week High-low range",
            "🔢 Fibonacci 52-Week: Price between 0.618-0.786 retracement levels",
            "📊 All-Time High (ATH): Price in 30-50% down from ATH",
            "📈 52-Week High Yearly: Price in 40-50% down from 52-Week High",
            "📅 Monthly % Change < -8%",
            "📅 3-Month % Change < -10%",
            "📅 Weekly % Change < -5%"
        ]
        for condition in main_conditions_list:
            st.sidebar.markdown(f"- {condition}")

        st.sidebar.markdown("### ➕ Additional Conditions")
        additional_conditions_list = [
            "📊 RSI: Relative Strength Index below 30 (Daily or Weekly)",
            "📈 EMA: Price less than 200-Day(EMA)"
        ]
        for condition in additional_conditions_list:
            st.sidebar.markdown(f"- {condition}")

    # Run screeners for all stocks
    results_met_stocks = []
    results_all_stocks = []
    atleast_two_met_stocks = []
    unindexed_stocks = []

    main_condition_labels = [
        "52-Week Range",
        "Fibonacci 52-Week",
        "All-Time High",
        "52-Week High Yearly",
        "Monthly % Change < -8%",
        "3-Month % Change < -10%",
        "Weekly % Change < -5%"
    ]
    additional_condition_labels = ["RSI", "EMA"]

    for idx, symbol in enumerate(nifty_100_symbols, start=1):
        data = get_stock_data(symbol)
        if data.empty:
            continue

        stock_data_full = get_full_stock_data(symbol)
        cond_ath, current_price_ath, low_limit_ath, high_limit_ath, ath = check_all_time_high_condition(stock_data_full)
        percentage_change, week_to_week_change, last_week = check_conditions_and_get_percentage_change(data)
        cond_52_week, lower_bound, upper_bound, high_52_week, low_52_week = check_52_week_condition(data)
        cond_fib_52_week, current_price_fib, fib_value_2, fib_value_1, high_52_week_fib = check_fibonacci_condition(data)
        cond_fib_year, current_price_fib_year, fib_value_2_year, fib_value_1_year, high_52_week_fib_year = check_fibonacci_year_condition(data)
        cond_52_week_year, low_limit_52_week_year, high_limit_52_week_year, high_52_week_year = check_52_week_high_yearly_condition(data)
        cond_rsi, rsi_daily, rsi_weekly = check_rsi_condition(data)
        cond_ema, ema_200, upper_bound_ema, lower_bound_ema = check_ema_condition(data)

        todays_change = 0
        if len(data) > 1:
            todays_change = round(calculate_percentage_change(data['Close'].iloc[-1], data['Close'].iloc[-2]), 2)
        monthly_change = calculate_monthly_change(data)
        three_month_change = calculate_3month_change(data)

        cond_monthly_change = (monthly_change is not None and monthly_change < -8)
        cond_weekly_change = (percentage_change < -5)
        cond_3month_change = (three_month_change is not None and three_month_change < -10)

        main_conditions_met = [
            cond_52_week, cond_fib_52_week, cond_ath, cond_52_week_year, cond_monthly_change, cond_3month_change, cond_weekly_change
        ]
        additional_conditions_met = [cond_rsi, cond_ema]
        conditions_met_count = sum(main_conditions_met) + sum(additional_conditions_met)
        conditions_html = ''.join(['✓' if cond else '✗' for cond in main_conditions_met + additional_conditions_met])
        conditions_score = f"{conditions_met_count}/{len(main_conditions_met) + len(additional_conditions_met)}"
        met_main_conditions = [label for label, cond in zip(main_condition_labels, main_conditions_met) if cond]
        met_additional_conditions = [label for label, cond in zip(additional_condition_labels, additional_conditions_met) if cond]

        if last_week.empty:
            last_week_close_val = None
        else:
            last_week_close_val = round(last_week['Close'], 2)

        result = {
            "S.No": idx,
            "Symbol": symbol,
            "Conditions Met": conditions_html,
            "Met Main Conditions": ', '.join(met_main_conditions),
            "Met Additional Conditions": ', '.join(met_additional_conditions),
            "Score": conditions_score,
            "Sector": sector_mapping.get(symbol, 'Unmapped'),
            "Current Price": round(data['Close'][-1], 2),
            "Today's Change": todays_change,
            "Weekly % Change": round(percentage_change, 2),
            "Monthly % Change": monthly_change,
            "3-Month % Change": three_month_change,
            "Week-to-Week % Change": round(week_to_week_change, 2),
            "Last Week Close": last_week_close_val,
            "52-Week High": round(high_52_week, 2) if high_52_week else None,
            "52-Week Low": round(low_52_week, 2) if low_52_week else None,
            "52-Week High Range (40-50% down from 52-Week High)": f"₹{round(lower_bound, 2)} - ₹{round(upper_bound, 2)}" if lower_bound and upper_bound else None,
            "Fibonacci Range (0.618-0.786 of 52-Week High)": f"₹{round(fib_value_2, 2)} - ₹{round(fib_value_1, 2)}" if fib_value_1 and fib_value_2 else None,
            "All-Time High": round(ath, 2) if ath else None,
            "ATH Range (30-50% down from ATH)": f"₹{round(low_limit_ath, 2)} - ₹{round(high_limit_ath, 2)}" if low_limit_ath and high_limit_ath else None,
            "52-Week High Yearly Range (40-50% down from 52-Week High)": f"₹{round(low_limit_52_week_year, 2)} - ₹{round(high_limit_52_week_year, 2)}" if low_limit_52_week_year and high_limit_52_week_year else None,
            "200-Day EMA": round(ema_200.iloc[-1], 2) if len(ema_200) > 0 else None,
            "Upper Bound (-1 STD)": round(upper_bound_ema, 2) if upper_bound_ema else None,
            "Lower Bound (-2 STD)": round(lower_bound_ema, 2) if lower_bound_ema else None,
            "Daily RSI": round(rsi_daily.iloc[-1], 2) if rsi_daily is not None and len(rsi_daily) > 0 else None,
            "Weekly RSI": round(rsi_weekly.iloc[-1], 2) if rsi_weekly is not None and len(rsi_weekly) > 0 else None,
        }

        if conditions_met_count >= 2:
            atleast_two_met_stocks.append(result)
        if all(main_conditions_met):
            results_met_stocks.append(result)
        results_all_stocks.append(result)

    # Sorting results
    results_all_stocks = sorted(results_all_stocks, key=lambda x: x["Score"], reverse=True)
    atleast_two_met_stocks = sorted(atleast_two_met_stocks, key=lambda x: x["Score"], reverse=True)
    results_met_stocks = sorted(results_met_stocks, key=lambda x: x["Score"], reverse=True)

    df_met_stocks = pd.DataFrame(results_met_stocks)
    df_all_stocks = pd.DataFrame(results_all_stocks)
    df_atleast_two_met_stocks = pd.DataFrame(atleast_two_met_stocks)

    def ensure_columns(df, required_columns):
        for col in required_columns:
            if col not in df.columns:
                df[col] = None
        return df[required_columns]

    required_columns = [
        "S.No", "Symbol", "Conditions Met", "Met Main Conditions", "Met Additional Conditions",
        "Score", "Sector", "Current Price", "Today's Change", "Weekly % Change",
        "Monthly % Change", "3-Month % Change", "Week-to-Week % Change", "Last Week Close", "52-Week High",
        "52-Week Low", "52-Week High Range (40-50% down from 52-Week High)",
        "Fibonacci Range (0.618-0.786 of 52-Week High)", "All-Time High",
        "ATH Range (30-50% down from ATH)", "52-Week High Yearly Range (40-50% down from 52-Week High)",
        "200-Day EMA", "Upper Bound (-1 STD)", "Lower Bound (-2 STD)",
        "Daily RSI", "Weekly RSI"
    ]

    df_met_stocks = ensure_columns(df_met_stocks, required_columns)
    df_all_stocks = ensure_columns(df_all_stocks, required_columns)
    df_atleast_two_met_stocks = ensure_columns(df_atleast_two_met_stocks, required_columns)

    # Top gainers and losers
    if not df_all_stocks.empty:
        df_top_gainers = df_all_stocks.sort_values(by="Today's Change", ascending=False).head(5)
        df_top_losers = df_all_stocks.sort_values(by="Today's Change", ascending=True).head(5)
    else:
        df_top_gainers = pd.DataFrame()
        df_top_losers = pd.DataFrame()

    st.write("### Top Gainers (Today's Change)")
    if not df_top_gainers.empty:
        df_top_gainers = df_top_gainers[required_columns]
        gb_gainers = GridOptionsBuilder.from_dataframe(df_top_gainers)
        gb_gainers.configure_default_column(editable=False)
        gb_gainers.configure_column("S.No", pinned=True)
        gb_gainers.configure_column("Symbol", pinned=True)
        gb_gainers.configure_column("Conditions Met", pinned=True)
        gb_gainers.configure_column("Score", pinned=True)
        gb_gainers.configure_columns(
            ['Current Price', "Today's Change", 'Weekly % Change', 'Monthly % Change', '3-Month % Change', 'Week-to-Week % Change'],
            cellStyle={'color': 'black', 'backgroundColor': 'white'}
        )
        grid_options_gainers = gb_gainers.build()
        AgGrid(df_top_gainers, gridOptions=grid_options_gainers, height=min(400, len(df_top_gainers) * 35), theme='balham')
    else:
        st.write("No gainers found.")

    st.write("### Top Losers (Today's Change)")
    if not df_top_losers.empty:
        df_top_losers = df_top_losers[required_columns]
        gb_losers = GridOptionsBuilder.from_dataframe(df_top_losers)
        gb_losers.configure_default_column(editable=False)
        gb_losers.configure_column("S.No", pinned=True)
        gb_losers.configure_column("Symbol", pinned=True)
        gb_losers.configure_column("Conditions Met", pinned=True)
        gb_losers.configure_column("Score", pinned=True)
        gb_losers.configure_columns(
            ['Current Price', "Today's Change", 'Weekly % Change', 'Monthly % Change', '3-Month % Change', 'Week-to-Week % Change'],
            cellStyle={'color': 'black', 'backgroundColor': 'white'}
        )
        grid_options_losers = gb_losers.build()
        AgGrid(df_top_losers, gridOptions=grid_options_losers, height=min(400, len(df_top_losers) * 35), theme='balham')
    else:
        st.write("No losers found.")

    # Indices
    results_met_indices = []
    results_all_indices = []

    for idx, index in enumerate(nifty_indices, start=1):
        data = get_stock_data(index)
        if data.empty:
            # No data for this index
            stock_symbols = [stock for stock, sector in sector_mapping.items() if sector == index_sector_mapping.get(index, '')]
            result = {
                "S.No": idx,
                "Symbol": ", ".join(stock_symbols),
                "Conditions Met": '',
                "Score": '',
                "Sector": index_sector_mapping.get(index, ''),
                "Current Price": None,
                "Today's Change": '',
                "Weekly % Change": '',
                "Monthly % Change": '',
                "3-Month % Change": '',
                "Week-to-Week % Change": '',
                "Last Week Close": '',
                "52-Week High": '',
                "52-Week Low": '',
                "52-Week High Range (40-50% down from 52-Week High)": '',
                "Fibonacci Range (0.618-0.786 of 52-Week High)": '',
                "All-Time High": '',
                "ATH Range (30-50% down from ATH)": '',
                "52-Week High Yearly Range (40-50% down from 52-Week High)": '',
                "200-Day EMA": '',
                "Upper Bound (-1 STD)": '',
                "Lower Bound (-2 STD)": '',
                "Daily RSI": '',
                "Weekly RSI": '',
            }
            results_all_indices.append(result)
            continue

        percentage_change, week_to_week_change, last_week = check_conditions_and_get_percentage_change(data)
        cond_52_week, lower_bound, upper_bound, high_52_week, low_52_week = check_52_week_condition(data)
        cond_fib_52_week, current_price_fib, fib_2, fib_1, high_52_week_fib = check_fibonacci_condition(data)
        cond_fib_year, current_price_fib_year, fib_2_year, fib_1_year, high_52_week_fib_year = check_fibonacci_year_condition(data)
        cond_ath, current_price_ath, low_limit_ath, high_limit_ath, ath = check_all_time_high_condition(data)
        cond_52_week_year, low_limit_52_week_year, high_limit_52_week_year, high_52_week_year = check_52_week_high_yearly_condition(data)
        cond_rsi, rsi_daily, rsi_weekly = check_rsi_condition(data)
        cond_ema, ema_200, upper_bound_ema, lower_bound_ema = check_ema_condition(data)

        todays_change = 0
        if len(data) > 1:
            todays_change = round(calculate_percentage_change(data['Close'].iloc[-1], data['Close'].iloc[-2]), 2)
        monthly_change = calculate_monthly_change(data)
        three_month_change = calculate_3month_change(data)

        cond_monthly_change = (monthly_change is not None and monthly_change < -8)
        cond_weekly_change = (percentage_change < -5)
        cond_3month_change = (three_month_change is not None and three_month_change < -10)

        main_conditions_met = [
            cond_52_week, cond_fib_52_week, cond_ath, cond_52_week_year, cond_monthly_change, cond_3month_change, cond_weekly_change
        ]
        additional_conditions_met = [cond_rsi, cond_ema]
        conditions_met_count = sum(main_conditions_met) + sum(additional_conditions_met)
        conditions_html = ''.join(['✓' if cond else '✗' for cond in main_conditions_met + additional_conditions_met])
        conditions_score = f"{conditions_met_count}/{len(main_conditions_met) + len(additional_conditions_met)}"
        met_main_conditions = [label for label, cond in zip(main_condition_labels, main_conditions_met) if cond]
        met_additional_conditions = [label for label, cond in zip(additional_condition_labels, additional_conditions_met) if cond]

        stock_symbols = [stock for stock, sector in sector_mapping.items() if sector == index_sector_mapping.get(index, '')]

        if last_week.empty:
            last_week_close_val = None
        else:
            last_week_close_val = round(last_week['Close'], 2)

        result = {
            "S.No": idx,
            "Symbol": ", ".join(stock_symbols),
            "Conditions Met": conditions_html,
            "Score": conditions_score,
            "Sector": index_sector_mapping.get(index, ''),
            "Current Price": round(data['Close'][-1], 2),
            "Today's Change": todays_change,
            "Weekly % Change": round(percentage_change, 2),
            "Monthly % Change": monthly_change,
            "3-Month % Change": three_month_change,
            "Week-to-Week % Change": round(week_to_week_change, 2),
            "Last Week Close": last_week_close_val,
            "52-Week High": round(high_52_week, 2) if high_52_week else None,
            "52-Week Low": round(low_52_week, 2) if low_52_week else None,
            "52-Week High Range (40-50% down from 52-Week High)": f"₹{round(lower_bound, 2)} - ₹{round(upper_bound, 2)}" if lower_bound and upper_bound else None,
            "Fibonacci Range (0.618-0.786 of 52-Week High)": f"₹{round(fib_2, 2)} - ₹{round(fib_1, 2)}" if fib_1 and fib_2 else None,
            "All-Time High": round(ath, 2) if ath else None,
            "ATH Range (30-50% down from ATH)": f"₹{round(low_limit_ath, 2)} - ₹{round(high_limit_ath, 2)}" if low_limit_ath and high_limit_ath else None,
            "52-Week High Yearly Range (40%-50% down from 52-Week High)": f"₹{round(low_limit_52_week_year, 2)} - ₹{round(high_limit_52_week_year, 2)}" if low_limit_52_week_year and high_limit_52_week_year else None,
            "200-Day EMA": round(ema_200.iloc[-1], 2) if len(ema_200) > 0 else None,
            "Upper Bound (-1 STD)": round(upper_bound_ema, 2) if upper_bound_ema else None,
            "Lower Bound (-2 STD)": round(lower_bound_ema, 2) if lower_bound_ema else None,
            "Daily RSI": round(rsi_daily.iloc[-1], 2) if rsi_daily is not None and len(rsi_daily) > 0 else None,
            "Weekly RSI": round(rsi_weekly.iloc[-1], 2) if rsi_weekly is not None and len(rsi_weekly) > 0 else None,
        }

        if all(main_conditions_met):
            results_met_indices.append(result)
        results_all_indices.append(result)

    df_met_indices = pd.DataFrame(results_met_indices)
    df_all_indices = pd.DataFrame(results_all_indices)

    df_met_indices = ensure_columns(df_met_indices, required_columns)
    df_all_indices = ensure_columns(df_all_indices, required_columns)

    st.write("### Stocks with At Least Two Conditions Met:")
    if not df_atleast_two_met_stocks.empty:
        df_atleast_two_met_stocks = df_atleast_two_met_stocks[required_columns]
        gb = GridOptionsBuilder.from_dataframe(df_atleast_two_met_stocks)
        gb.configure_default_column(editable=False)
        gb.configure_column("S.No", pinned=True)
        gb.configure_column("Symbol", pinned=True)
        gb.configure_column("Conditions Met", pinned=True)
        gb.configure_column("Score", pinned=True)
        gb.configure_columns(
            ['Current Price', "Today's Change", 'Weekly % Change', 'Monthly % Change', '3-Month % Change', 'Week-to-Week % Change'],
            cellStyle={'color': 'black', 'backgroundColor': 'white'}
        )
        grid_options = gb.build()
        AgGrid(df_atleast_two_met_stocks, gridOptions=grid_options, height=min(400, len(df_atleast_two_met_stocks) * 35), theme='balham')
    else:
        st.write("No stocks with at least two conditions met.")

    st.download_button(
        label="Download Report for Stocks with At Least Two Conditions Met as CSV",
        data=generate_csv(df_atleast_two_met_stocks),
        file_name='stocks_with_at_least_two_conditions_met_report.csv',
        mime='text/csv'
    )

    for idx, row in df_all_indices.iterrows():
        with st.expander(f"📊 {row['Sector']} - {row['Symbol']}"):
            stocks_in_sector = [
                stock for stock, sector in sector_mapping.items()
                if sector == row['Sector']
            ]
            if stocks_in_sector:
                stocks_data = [
                    result for result in results_all_stocks
                    if result["Symbol"] in stocks_in_sector
                ]
                df_stocks_in_sector = pd.DataFrame(stocks_data)
                if not df_stocks_in_sector.empty:
                    combined_data = pd.concat(
                        [pd.DataFrame([row]), df_stocks_in_sector],
                        ignore_index=True
                    )

                    gb = GridOptionsBuilder.from_dataframe(combined_data)
                    gb.configure_default_column(editable=False, wrapHeaderText=True, autoSizeColumns=False)
                    gb.configure_grid_options(domLayout='normal', suppressHorizontalScroll=False)
                    gb.configure_column("Symbol", width=200)
                    gb.configure_columns(
                        combined_data.columns.tolist(),
                        cellStyle={'color': 'black', 'backgroundColor': '#f9f9f9'}
                    )
                    grid_options = gb.build()

                    st.markdown(f"#### Stocks in {row['Sector']} Sector")
                    AgGrid(
                        combined_data,
                        gridOptions=grid_options,
                        height=min(400, len(combined_data) * 35),
                        theme='balham',
                        enable_enterprise_modules=True
                    )
                else:
                    st.info(f"No detailed stock data available for the {row['Sector']} sector.")
            else:
                st.info(f"No stocks found in the {row['Sector']} sector.")

    st.write("### Stocks Meeting At Least 6 Conditions with 5 Main Conditions")
    qualified_stocks = []
    for stock in results_all_stocks:
        main_conditions_met_count = sum(
            1 for cond in main_condition_labels if cond in stock["Met Main Conditions"]
        )
        additional_conditions_met_count = sum(
            1 for cond in additional_condition_labels if cond in stock["Met Additional Conditions"]
        )
        total_conditions_met_count = main_conditions_met_count + additional_conditions_met_count
        if total_conditions_met_count >= 6 and main_conditions_met_count >= 5:
            missing_main_conditions = []
            missing_additional_conditions = []

            for cond in main_condition_labels:
                if cond not in stock["Met Main Conditions"]:
                    missing_main_conditions.append(cond + " (Missed)")

            for cond in additional_condition_labels:
                if cond not in stock["Met Additional Conditions"]:
                    missing_additional_conditions.append(cond + " (Missed)")

            stock["Missing Main Conditions"] = missing_main_conditions
            stock["Missing Additional Conditions"] = missing_additional_conditions
            qualified_stocks.append(stock)

    if qualified_stocks:
        stocks_by_sector = {}
        for stock in qualified_stocks:
            sector = stock.get("Sector", "Unmapped")
            if sector not in stocks_by_sector:
                stocks_by_sector[sector] = []
            stocks_by_sector[sector].append(stock)

        st.markdown(
            """
            <style>
            .grid-container {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                gap: 20px;
                margin-top: 20px;
            }
            .grid-card {
                border: 1px solid #ddd;
                border-radius: 10px;
                padding: 15px;
                background-color: #ffffff;
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            }
            .grid-card h4 {
                margin: 0;
                color: #1f4e79;
                font-size: 18px;
                font-weight: bold;
            }
            .grid-card p {
                margin: 5px 0;
                font-size: 14px;
                color: #555;
            }
            .conditions {
                display: flex;
                flex-wrap: wrap;
                gap: 5px;
                font-size: 12px;
            }
            .condition {
                padding: 2px 8px;
                background-color: #e6f7e6;
                border: 1px solid #b2d8b2;
                border-radius: 5px;
                color: #2e7d32;
                font-weight: bold;
            }
            .condition-additional {
                background-color: #f0f4ff;
                border: 1px solid #b3cde3;
                color: #1e5a91;
            }
            .missing-condition {
                background-color: #fff0f0;
                border: 1px solid #f5b1b1;
                color: #d32f2f;
                font-weight: bold;
                padding: 2px 8px;
                border-radius: 5px;
                margin: 5px 0;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

        for sector, stocks in stocks_by_sector.items():
            with st.expander(f"📂 {sector} ({len(stocks)} Stocks)", expanded=True):
                st.markdown('<div class="grid-container">', unsafe_allow_html=True)
                for stock in stocks:
                    condition_details = ""
                    card_content = f"""
                        <div class="grid-card">
                            <h4>📈 {stock['Symbol']}</h4>
                            <p><b>Current Price:</b> ₹{stock['Current Price']}</p>
                            <p><b>Score:</b> {stock['Score']}</p>
                            <p><strong>Missing Main Conditions:</strong></p>
                            {"<br>".join([f"<div class='missing-condition'>{m}</div>" for m in stock.get('Missing Main Conditions', [])])}
                            <p><strong>Missing Additional Conditions:</strong></p>
                            {"<br>".join([f"<div class='missing-condition'>{m}</div>" for m in stock.get('Missing Additional Conditions', [])])}
                        </div>
                    """
                    st.markdown(card_content, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

elif selected_tab == "📊 Historical dashboard":
    tickers = {
        'Adani Enterprises': 'ADANIENT.NS',
        'Adani Ports and SEZ': 'ADANIPORTS.NS',
        'Asian Paints': 'ASIANPAINT.NS',
        'Axis Bank': 'AXISBANK.NS',
        'Bajaj Auto': 'BAJAJ-AUTO.NS',
        'Bajaj Finance': 'BAJFINANCE.NS',
        'Bajaj Finserv': 'BAJAJFINSV.NS',
        'Bharti Airtel': 'BHARTIARTL.NS',
        'BPCL': 'BPCL.NS',
        'Britannia Industries': 'BRITANNIA.NS',
        'Cipla': 'CIPLA.NS',
        'Coal India': 'COALINDIA.NS',
        'Divi\'s Laboratories': 'DIVISLAB.NS',
        'Dr. Reddy\'s Laboratories': 'DRREDDY.NS',
        'Eicher Motors': 'EICHERMOT.NS',
        'Grasim Industries': 'GRASIM.NS',
        'HCL Technologies': 'HCLTECH.NS',
        'HDFC Bank': 'HDFCBANK.NS',
        'HDFC Life': 'HDFCLIFE.NS',
        'Hero MotoCorp': 'HEROMOTOCO.NS',
        'Hindalco Industries': 'HINDALCO.NS',
        'Hindustan Unilever': 'HINDUNILVR.NS',
        'ICICI Bank': 'ICICIBANK.NS',
        'Indian Oil Corporation': 'IOC.NS',
        'IndusInd Bank': 'INDUSINDBK.NS',
        'Infosys': 'INFY.NS',
        'ITC': 'ITC.NS',
        'JSW Steel': 'JSWSTEEL.NS',
        'Kotak Mahindra Bank': 'KOTAKBANK.NS',
        'Larsen & Toubro': 'LT.NS',
        'Mahindra & Mahindra': 'M&M.NS',
        'Maruti Suzuki': 'MARUTI.NS',
        'Nestle India': 'NESTLEIND.NS',
        'NTPC': 'NTPC.NS',
        'ONGC': 'ONGC.NS',
        'Power Grid Corporation': 'POWERGRID.NS',
        'Reliance Industries': 'RELIANCE.NS',
        'SBI Life': 'SBILIFE.NS',
        'State Bank of India': 'SBIN.NS',
        'Sun Pharmaceuticals': 'SUNPHARMA.NS',
        'Tata Consumer Products': 'TATACONSUM.NS',
        'Tata Motors': 'TATAMOTORS.NS',
        'Tata Steel': 'TATASTEEL.NS',
        'TCS': 'TCS.NS',
        'Tech Mahindra': 'TECHM.NS',
        'Titan Company': 'TITAN.NS',  # Added Titan for completeness
        'UltraTech Cement': 'ULTRACEMCO.NS',
        'UPL': 'UPL.NS',
        'Wipro': 'WIPRO.NS'
    }

    current_date = datetime.now().strftime('%Y-%m-%d')
    current_month = datetime.now().strftime('%B')

    months_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']

    @st.cache_data
    def download_stock_data(stock_symbol):
        try:
            data = yf.download(stock_symbol, start="2000-01-01", end="2024-01-01", interval="1d")
            return data
        except:
            return pd.DataFrame()

    def calculate_monthly_percentage_change(data):
        if data.empty:
            return pd.DataFrame(), pd.DataFrame()
        data = data.reset_index()
        data['Year'] = data['Date'].dt.year
        data['Month'] = data['Date'].dt.month

        monthly_data = data.groupby(['Year', 'Month']).agg(
            Opening_Price=('Open', 'first'),
            Closing_Price=('Close', 'last')
        ).reset_index()

        monthly_data['Pct Change'] = ((monthly_data['Closing_Price'] - monthly_data['Opening_Price']) / monthly_data['Opening_Price']) * 100
        heatmap_data = monthly_data.pivot(index='Year', columns='Month', values='Pct Change')
        if heatmap_data.empty:
            return pd.DataFrame(), pd.DataFrame()

        heatmap_data.loc['Average'] = heatmap_data.mean()
        heatmap_data.columns = [pd.to_datetime(f'{m}', format='%m').strftime('%B') for m in heatmap_data.columns]

        return heatmap_data, monthly_data

    def calculate_positive_ratio(monthly_data):
        if monthly_data.empty:
            return pd.DataFrame()
        positive_count = monthly_data.groupby('Month')['Pct Change'].apply(lambda x: (x > 0).sum())
        total_count = monthly_data.groupby('Month')['Pct Change'].count()
        ratio = (positive_count / total_count) * 100
        month_names = [pd.to_datetime(f'{m}', format='%m').strftime('%B') for m in total_count.index]
        ratio_data = pd.DataFrame({
            'Month': month_names,
            'Positive Months': positive_count.values,
            'Total Months': total_count.values,
            'Positive Ratio (%)': ratio.values
        }).set_index('Month')
        return ratio_data

    def calculate_yearly_percentage_change(data):
        if data.empty:
            return pd.DataFrame()
        data = data.reset_index()
        data['Year'] = data['Date'].dt.year
        yearly_data = data.groupby('Year').agg(
            Opening_Price=('Open', 'first'),
            Closing_Price=('Close', 'last')
        ).reset_index()
        yearly_data['Yearly Pct Change'] = ((yearly_data['Closing_Price'] - yearly_data['Opening_Price']) / yearly_data['Opening_Price']) * 100
        return yearly_data[['Year', 'Yearly Pct Change']].set_index('Year')

    def predict_future_trend(month_data):
        if month_data.empty:
            return None
        X = month_data['Year'].values.reshape(-1, 1)
        y = month_data['Monthly % Change'].values
        if len(X) < 2:
            return None
        model = LinearRegression()
        model.fit(X, y)
        next_year = np.array([[X[-1][0] + 1]])
        future_trend = model.predict(next_year)[0]
        return future_trend

    def process_stock_data(ticker):
        stock_data = yf.download(ticker, start="2015-01-01", end=current_date, interval="1mo")
        if stock_data.empty:
            return {
                'summary_table': pd.DataFrame(),
                'std_table': pd.DataFrame(),
                'stock_data': pd.DataFrame(),
                'current_month_data': {},
                'best_month': {},
                'worst_month': {},
                'average_monthly_change': pd.Series()
            }
        stock_data['Monthly % Change'] = ((stock_data['Close'] - stock_data['Open']) / stock_data['Open']) * 100
        stock_data['Year'] = stock_data.index.year
        stock_data['Month'] = stock_data.index.strftime('%B')

        data_until_2023 = stock_data[stock_data['Year'] <= 2023]
        average_monthly_change = data_until_2023.groupby('Month')['Monthly % Change'].mean().reindex(months_order)
        std_dev = data_until_2023.groupby('Month')['Monthly % Change'].std()
        max_monthly_change = data_until_2023.groupby('Month')['Monthly % Change'].max()
        min_monthly_change = data_until_2023.groupby('Month')['Monthly % Change'].min()

        sharpe_ratio = (average_monthly_change / std_dev).round(2)
        positive_returns = data_until_2023.pivot_table(values='Monthly % Change', index='Month', aggfunc=lambda x: (x > 0).sum()).reindex(months_order)
        negative_returns = data_until_2023.pivot_table(values='Monthly % Change', index='Month', aggfunc=lambda x: (x < 0).sum()).reindex(months_order)

        best_month = average_monthly_change.idxmax()
        best_month_avg = average_monthly_change.max()
        worst_month = average_monthly_change.idxmin()
        worst_month_avg = average_monthly_change.min()

        current_month_avg = average_monthly_change.loc[current_month] if current_month in average_monthly_change.index else None
        current_month_max = max_monthly_change.loc[current_month] if current_month in max_monthly_change.index else None
        current_month_min = min_monthly_change.loc[current_month] if current_month in min_monthly_change.index else None
        current_year_data = stock_data[(stock_data['Month'] == current_month) & (stock_data['Year'] == datetime.now().year)]
        if not current_year_data.empty:
            current_month_pct_change = current_year_data['Monthly % Change'].iloc[-1]
        else:
            current_month_pct_change = None

        std_table = pd.DataFrame({
            'Average Monthly % Change': average_monthly_change.round(2),
            'Standard Deviation': std_dev.round(2),
            '±1 Std Dev Range': (average_monthly_change - std_dev).round(2).astype(str) + " to " + (average_monthly_change + std_dev).round(2).astype(str),
            '±2 Std Dev Range': (average_monthly_change - 2 * std_dev).round(2).astype(str) + " to " + (average_monthly_change + 2 * std_dev).round(2).astype(str),
            '±3 Std Dev Range': (average_monthly_change - 3 * std_dev).round(2).astype(str) + " to " + (average_monthly_change + 3 * std_dev).round(2).astype(str)
        })

        summary_table = pd.DataFrame({
            'Average Monthly % Change': average_monthly_change.round(2),
            'Max Monthly % Change': max_monthly_change.round(2),
            'Min Monthly % Change': min_monthly_change.round(2),
            'Positive Returns (Months)': positive_returns.values.flatten(),
            'Negative Returns (Months)': negative_returns.values.flatten(),
            'Sharpe Ratio': sharpe_ratio.values.flatten()
        })

        return {
            'summary_table': summary_table,
            'std_table': std_table,
            'stock_data': stock_data,
            'current_month_data': {
                'pct_change': current_month_pct_change,
                'avg_change': current_month_avg,
                'max_change': current_month_max,
                'min_change': current_month_min
            },
            'best_month': {'name': best_month, 'avg': best_month_avg},
            'worst_month': {'name': worst_month, 'avg': worst_month_avg},
            'average_monthly_change': average_monthly_change
        }

    def plot_bar_chart(month_data, selected_month):
        if month_data.empty:
            st.write("No data to plot.")
            return
        fig, ax = plt.subplots(figsize=(12, 6))
        colors = ['#FF9999' if val < 0 else '#99CCFF' for val in month_data['Monthly % Change']]
        bars = ax.bar(month_data['Year'], month_data['Monthly % Change'], color=colors, edgecolor='black', alpha=0.9)

        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, yval + (1 if yval > 0 else -1.5), f"{yval:.2f}%", ha='center', va='bottom' if yval > 0 else 'top', fontsize=9)

        ax.set_title(f"Monthly Percentage Change for {selected_month}", fontsize=16, fontweight='bold', color='#003366')
        ax.set_xlabel("Year", fontsize=12)
        ax.set_ylabel("Monthly % Change (%)", fontsize=12)
        ax.axhline(0, color='grey', linestyle='--', linewidth=0.8)
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)

    st.title("Nifty 50 Top 5 Stocks - Monthly and Yearly Performance Heatmap")
    selected_stock = st.selectbox("Select Stock", list(tickers.keys()))

    stock_symbol = tickers[selected_stock]
    stock_data = download_stock_data(stock_symbol)
    if stock_data.empty:
        st.write("No data available for the selected stock.")
    else:
        monthly_returns, monthly_data = calculate_monthly_percentage_change(stock_data)
        positive_ratio_data = calculate_positive_ratio(monthly_data)
        yearly_returns = calculate_yearly_percentage_change(stock_data)

        if not monthly_returns.empty and not yearly_returns.empty:
            monthly_returns['Yearly Change'] = yearly_returns['Yearly Pct Change']

        st.subheader(f"Monthly and Yearly Percentage Change for {selected_stock} (2000 to 2024)")
        if not monthly_returns.empty:
            fig, ax = plt.subplots(figsize=(12, 8))
            norm = TwoSlopeNorm(vmin=monthly_returns.min().min(), vcenter=0, vmax=monthly_returns.max().max())
            cmap = sns.color_palette("RdYlGn", as_cmap=True)
            sns.heatmap(monthly_returns, cmap=cmap, norm=norm, annot=True, fmt=".1f", ax=ax, linewidths=0.5, cbar_kws={"label": "% Change"})
            ax.set_title(f"{selected_stock} Monthly Performance Heatmap (with Yearly % Change)", fontsize=16)
            ax.set_ylabel("Year", fontsize=12)
            ax.set_xlabel("Month", fontsize=12)
            st.pyplot(fig)

            monthly_averages = monthly_returns.mean(axis=0).drop('Yearly Change', errors='ignore')

            if not positive_ratio_data.empty:
                positive_ratio_data["Average % Change"] = monthly_averages.values if len(positive_ratio_data) == len(monthly_averages) else None
                st.subheader(f"Positive-to-Total Ratio and Average Monthly Performance for {selected_stock}")
                st.write(positive_ratio_data)
        else:
            st.write("Insufficient data to generate heatmap.")
