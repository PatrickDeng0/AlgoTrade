import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.arima_model import ARIMA
import datetime
import scipy.optimize as sco
import math
import re
import copy


# 可以自己import我们平台支持的第三方python模块，比如pandas、numpy等。

# 在这个方法中编写任何的初始化逻辑。context对象将会在你的算法策略的任何方法之间做传递。
def now_trade_future(context):
    total = all_instruments(type='Future').order_book_id.values
    category = set()
    pattern = re.compile(r'\D+')
    res = []
    for item in total:
        category.add(pattern.findall(item)[0])
    for name in list(category):
        ele = get_dominant_future(name)
        if ele is not None:
            res.append(ele)
    return res


def init(context):
    # context内引入全局变量s1
    # 初始化时订阅合约行情。订阅之后的合约行情会在handle_bar中进行更新。
    # 实时打印日志
    name = now_trade_future(context)
    context.s1 = [v for v in name if not v.endswith('88') and not v.startswith('IC') and not v.startswith('IH') and not v.startswith('IF') and not v.endswith('99')]
    subscribe(context.s1)
    logger.info("RunInfo: {}".format(context.run_info))


# before_trading此函数会在每天策略交易开始前被调用，当天只会被调用一次
def alpha_select(context):
    num = 0;
    ranking = [];
    top_rank = 20
    for i in context.s1:
        risk_free = context.risk_free
        num = num + 1
        try:
            future_price = pd.DataFrame(
                get_price(i, start_date=context.now - datetime.timedelta(days=context.look_back),
                          frequency='1d', fields='close'))
            lm_data = pd.merge(context.index, future_price, left_index=True, right_index=True, how='inner')
            index_ret = price2return(lm_data.close_x)
            future_ret = price2return(lm_data.close_y)
            lm_fit = np.polyfit(index_ret - risk_free, future_ret - risk_free, 1)
            ranking.append([i, lm_fit[1]])
        except:
            continue
    ranking.sort(key=lambda x: x[1], reverse=True)
    res = []
    for j in range(len(ranking)):
        res.append(ranking[j][0])
        if j == top_rank - 1:
            break
    return res


def before_trading(context):
    name = now_trade_future(context)
    context.s1 = [v for v in name if not v.endswith('88') and not v.startswith('IC') and not v.startswith('IH') and not v.startswith('IF') and not v.endswith('99')]
    context.look_back = 41
    subscribe(context.s1)
    context.risk_free = get_yield_curve(get_previous_trading_date(context.now))['0S'].values[0]
    context.index = pd.DataFrame(
        get_price('沪深300', start_date=context.now - datetime.timedelta(days=context.look_back)).close)
    context.after_alpha = alpha_select(context)
    context.buys = {}
    context.sells = {}
    context.sell_order = {}
    context.buy_order = {}


def price2return(minute_price):
    log_Return = np.array([])
    for i in range(len(minute_price)):
        if i != 0:
            log_Return = np.concatenate(
                (log_Return, np.array([np.log(minute_price[i] / minute_price[i - 1])])))
    return log_Return


def Each_futures(choice, tseries):
    def index_process(series):
        res = [series[0]]
        error_num = 0
        for i in range(1, len(series)):
            if series[i] - series[i - 1] >= 3:
                return res
            elif series[i] - series[i - 1] == 2:
                error_num += 1
                if error_num >= 2:
                    return res
                else:
                    res.append(series[i])
            elif series[i] - series[i - 1] == 1:
                res.append(series[i])
        return res

    def select_process(series):
        new_series = index_process(series)
        if len(new_series) <= 2:
            return new_series
        else:
            return new_series[-2:]

    try:
        d = 0
        for i in range(5):
            ts_test = np.diff(tseries, i)
            p = adfuller(ts_test, 1)[1]
            if p < 0.05:
                d = i
                break
                # Get def
                    
        p_series = acf(np.diff(tseries, d), nlags=5, qstat=True)[2]
        acf_index = [0]
        for i in range(len(p_series)):
            if p_series[i] > 0.05:
                acf_index.append(i + 1)
        acf_index = select_process(acf_index)
        # Get p
        
        q_series = pacf(np.diff(tseries, d), nlags=5)
        pacf_index = []
        for i in range(len(q_series)):
            if abs(q_series[i]) > 0.1:
                pacf_index.append(i)
        pacf_index = select_process(pacf_index)
        # Get quit
        
        selected_tuple = []
        for i in pacf_index:
            for j in acf_index:
                selected_tuple.append((i, d, j))
        selected_tuple.sort(key=lambda x: sum(x))
        # Get all of the possible (p,i,q)
        
        max_len = min(3, len(selected_tuple))
        selected_tuple = selected_tuple[:max_len]
        # Choose top 3 or less
        
        diction = {}
        feasible = []
        for i in range(len(selected_tuple)):
            try:
                model = ARIMA(tseries, order=selected_tuple[i]).fit()
                diction[selected_tuple[i]] = model
                ele = list(selected_tuple[i])
                a = model.aic
                ele.append(a)
                feasible.append(ele)
            except:
                continue
            
        feasible.sort(key=lambda x: x[3])
        final_model = diction[tuple(feasible[0][:3])]
        
        prediction = final_model.predict()
        return np.sum(prediction[0:choice-1])
    
    except:
        return


def future_selection(context):
    might_be_future = []
    num = 0
    for i in context.after_alpha:
        num = num + 1
        record_price = history_bars(i, bar_count=1000, frequency='1m', fields='close')
        record_ret = price2return(record_price)
        if len(record_ret) == 999:
            sumed_log_return = Each_futures(60, record_ret)
            if sumed_log_return is not None:
                might_be_future.append([i, sumed_log_return, record_price])
                
    might_be_future.sort(key=lambda x: x[1], reverse=True)
    res = []
    for i in range(len(might_be_future)):
        res.append(might_be_future[i])
        if i == 9:
            break
    return res


def opti_pf(res, context):  # res 是 might_be_future

    number_of_assets = len(res)
    # 生成随机数
    weights = np.random.random(number_of_assets)
    # 将随机数归一化，每一份就是权重，权重之和为1
    weights /= np.sum(weights)

    def statistics(weights):
        weights = np.array(weights)
        pred_ret = []
        compute_cov = []
        for i in range(len(res)):
            pred_ret.append(res[i][1])
            compute_cov.append(price2return(res[i][2]))
        pret = np.dot(np.array(pred_ret), weights) * 1000
        compute_cov = np.matrix(compute_cov)
        pvol = np.sqrt(np.dot(weights, np.dot(np.cov(compute_cov) * 1000, weights)))
        return pret / pvol

    def min_func_sharpe(weights):
        return -statistics(weights)

    bnds = tuple((0, 1) for x in range(number_of_assets))
    cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    opts = sco.minimize(min_func_sharpe, number_of_assets * [1. / number_of_assets, ], method='SLSQP', bounds=bnds,
                        constraints=cons)

    pool = context.future_account.total_value * 0.85
    cost = opts['x'].round(6) * pool
    futures = []
    for i in range(len(res)):
        unit = current_snapshot(res[i][0]).last
        if math.isnan(unit) or unit == 0:
            unit = res[i][2][-1]
        price = unit * instruments(res[i][0]).contract_multiplier * instruments(res[i][0]).margin_rate * 3.3
        if not (price==0):
            num = math.floor(cost[i] / price)
            if num >= 1:
                futures.append([res[i][0], num, unit])
    return futures


# 得到现在要的操作
def waiting_list(futures, context):
    if futures:
        buy_list = {}; sell_list = {}
        name_list = [future[0] for future in futures]
        now_hold = [key for key in context.future_account.positions.keys() if context.future_account.positions[key].buy_quantity > 0]
        for item in now_hold:
            if item not in name_list:
                sell_list[item] = [context.future_account.positions[item].buy_quantity, context.future_account.positions[item].buy_avg_open_price]
        for future in futures:
            if future[0] not in now_hold:
                buy_list[future[0]] = [future[1], future[2]]
            else:
                if future[1] > context.future_account.positions[future[0]].buy_quantity:
                    buy_list[future[0]] = [future[1] - context.future_account.positions[future[0]].buy_quantity, future[2]]
                if future[1] < context.future_account.positions[future[0]].buy_quantity:
                    sell_list[future[0]] = [context.future_account.positions[future[0]].buy_quantity - future[1], future[2]]
        return sell_list, buy_list
    else:
        return {}, {}

def opera_process(context, bar_dict):
    sell_list = {}
    for key, value in context.sells.items():
        if not value[0] == 0:
            sell_list[key] = value
    
    buy_list = {}
    for key, value in context.buys.items():
        if not value[0] == 0:
            buy_list[key] = value
    
    sell_names = list(sell_list.keys())
    on_sell = list(context.sell_order.keys())
    for name in sell_names:
        if name not in on_sell:
            once = int(sell_list[name][0]*0.2) + 1
            if once < sell_list[name][0]:
                thisorder = sell_close(name, once, LimitOrder(sell_list[name][1]))
                if thisorder is not None:
                    context.sell_order[name] = thisorder
            else:
                thisorder = sell_close(name, sell_list[name][0], LimitOrder(sell_list[name][1]))
                if thisorder is not None:
                    context.sell_order[name] = thisorder
        else:
            if not isinstance(context.sell_order[name], list):
                if context.sell_order[name].unfilled_quantity == 0:
                    sell_list[name][0] = sell_list[name][0] - context.sell_order[name].filled_quantity
                    del context.sell_order[name]
            else:
                orders = []
                for order in context.sell_order[name]:
                    if order.unfilled_quantity == 0:
                        sell_list[name][0] = sell_list[name][0] - order.filled_quantity
                    else:
                        orders.append(order)
                if orders:
                    context.sell_order[name] = orders
                else:
                    del context.sell_order[name]
                
    buy_names = list(buy_list.keys())
    on_buy = list(context.buy_order.keys())
    for name in buy_names:
        if name not in on_buy:
            if context.future_account.cash > 10000:
                once = int(buy_list[name][0]*0.2) + 1
                if once < buy_list[name][0]:
                    thisorder = buy_open(name, once, LimitOrder(buy_list[name][1]))
                    if thisorder is not None:
                        context.buy_order[name] = thisorder
                else:
                    thisorder = buy_open(name, buy_list[name][0], LimitOrder(buy_list[name][1]))
                    if thisorder is not None:
                        context.buy_order[name] = thisorder
        else:
            if context.buy_order[name].unfilled_quantity == 0:
                buy_list[name][0] = buy_list[name][0] - context.buy_order[name].filled_quantity
                del context.buy_order[name]
    return sell_list, buy_list


def cancel_old(context):
    hang_out = get_open_orders()
    for order in hang_out:
        cancel_order(order)
    context.buys = {}
    context.sells = {}
    context.sell_order = {}
    context.buy_order = {}


# 你选择的期货数据更新将会触发此段逻辑，例如日线或分钟线更新
def handle_bar(context, bar_dict):
    if (context.now.date().isoweekday() in [1, 2, 3, 4, 5]) and ((context.now.hour in [9, 10, 14] and context.now.minute == 30) or (context.now.hour in [10, 11, 14] and context.now.minute == 0)):
        # 开始编写你的主要的算法逻辑
        cancel_old(context)
        selects = future_selection(context)
        futures = opti_pf(selects, context)
        context.sells, context.buys = waiting_list(futures, context)
        # context.future_account 可以获取到当前投资组合信息
        # 使用buy_open(id_or_ins, amount)方法进行买入开仓操作

    if context.sells or context.buys:
        context.sells, context.buys = opera_process(context, bar_dict)


# after_trading函数会在每天交易结束后被调用，当天只会被调用一次
def after_trading(context):
    pass
