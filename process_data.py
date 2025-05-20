import os
import json
import dask.dataframe as dd
import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules, fpgrowth
import matplotlib.pyplot as plt


# 商品映射表路径
ITEM_MAP_PATH = "product_mapping.json"
BASE_PATH = "C:\\Users\\admin\\Downloads\\30G_data"

# 加载商品映射表
with open(ITEM_MAP_PATH) as f:
    item_data = json.load(f)['products']
    item_map = {x['id']: x['category'] for x in item_data}


# 预处理函数
def preprocess(df):
    # 解析JSON字段
    df = df.assign(
        purchase_history=df.purchase_history.apply(json.loads, meta=('purchase_history', 'object')))
    
    # 展开购买记录
    exploded = df.explode('purchase_history')
    
    # 提取字段
    def extract_fields(x):
        try:
            return pd.Series({
                'item_id': x.get('item_id'),
                'price': x.get('price'),
                'payment_status': x.get('payment_status'),
                'payment_method': x.get('payment_method'),
                'purchase_date': pd.to_datetime(x.get('purchase_date'))
            })
        except:
            return pd.Series()
    
    extracted = exploded.purchase_history.apply(extract_fields, meta={
        'item_id': 'int64',
        'price': 'float64',
        'payment_status': 'object',
        'payment_method': 'object',
        'purchase_date': 'datetime64[ns]'
    })
    
    # 合并结果
    processed = dd.concat([exploded.drop('purchase_history', axis=1), extracted], axis=1)
    
    # 添加类别信息
    processed['category'] = processed.item_id.map(item_map)
    
    # 过滤无效数据
    processed = processed[processed.category.notnull()]
    
    return processed

# 加载并预处理数据
ddf = dd.read_parquet(os.path.join(BASE_PATH, "*.parquet"), engine='pyarrow')
processed_ddf = preprocess(ddf).persist()  # 持久化预处理结果


# 商品类别关联规则
def task1_analysis():
    # 生成事务数据
    transactions = processed_ddf.groupby(['id', 'timestamp'])['category'].unique().to_frame()
    
    # 转换为Pandas（注意数据量）
    transactions_pd = transactions.compute()
    
    # 应用Apriori算法
    te = TransactionEncoder()
    te_ary = te.fit_transform(transactions_pd.category.tolist())
    df = pd.DataFrame(te_ary, columns=te.columns_)
    
    # 挖掘频繁项集
    frequent_itemsets = apriori(df, min_support=0.02, use_colnames=True)
    
    # 生成关联规则
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)
    
    # 筛选电子产品相关规则
    electronics_rules = rules[
        rules.antecedents.apply(lambda x: '电子产品' in x) |
        rules.consequents.apply(lambda x: '电子产品' in x)
    ]
    
    # 保存结果
    electronics_rules.to_parquet("task1_electronics_rules.parquet")
    rules.to_parquet("task1_all_rules.parquet")
    
    return electronics_rules

def task2_analysis():
    # 普通支付方式分析
    payment_rules = (
        processed_ddf.groupby(['payment_method', 'category'])
        .size().reset_index(name='count')
        .compute()
    )
    
    # 高价值商品分析
    high_value = processed_ddf[processed_ddf.price > 5000]
    high_value_payment = (
        high_value.groupby(['payment_method', 'category'])
        .size().reset_index(name='count')
        .compute()
    )
    
    # 支付方式关联规则
    payment_transactions = processed_ddf.groupby(['id', 'timestamp']).agg({
        'payment_method': 'first',
        'category': 'unique'
    }).reset_index()
    
    # 生成扩展特征矩阵
    payment_transactions['cat_payment'] = payment_transactions.apply(
        lambda row: [row.payment_method] + list(row.category), axis=1)
    
    # 计算关联规则
    te = TransactionEncoder()
    te_ary = te.fit_transform(payment_transactions.cat_payment.tolist())
    payment_df = pd.DataFrame(te_ary, columns=te.columns_)
    
    payment_itemsets = apriori(payment_df, min_support=0.01, use_colnames=True)
    payment_rules = association_rules(payment_itemsets, metric="confidence", min_threshold=0.6)
    
    # 保存结果
    payment_rules.to_parquet("task2_payment_rules.parquet")
    
    return payment_rules


def task3_analysis():
    # 时间特征提取
    timed_ddf = processed_ddf.assign(
        month=processed_ddf.purchase_date.dt.month,
        quarter=processed_ddf.purchase_date.dt.quarter,
        weekday=processed_ddf.purchase_date.dt.weekday
    )
    
    # 季节性分析
    seasonal = (
        timed_ddf.groupby(['category', 'quarter'])
        .size().reset_index(name='count')
        .compute()
    )
    
    # 时序模式挖掘
    sequence_data = (
        timed_ddf.groupby('id')
        .apply(lambda g: g.sort_values('purchase_date').category.tolist(), 
               meta=('category', 'object'))
        .compute()
    )
    
    # 生成序列模式
    from prefixspan import PrefixSpan
    ps = PrefixSpan(sequence_data)
    seq_patterns = ps.frequent(100, closed=True)  # 设置最小支持度计数
    
    # 保存结果
    seasonal.to_parquet("task3_seasonal.parquet")
    pd.DataFrame(seq_patterns).to_parquet("task3_sequence_patterns.parquet")
    
    return seasonal


def task4_analysis():
    # 筛选退款记录
    refund_ddf = processed_ddf[processed_ddf.payment_status.isin(['已退款', '部分退款'])]
    
    # 生成事务数据
    refund_transactions = (
        refund_ddf.groupby(['id', 'timestamp'])['category']
        .unique().reset_index()
        .compute()
    )
    
    # 关联规则挖掘
    te = TransactionEncoder()
    te_ary = te.fit_transform(refund_transactions.category)
    refund_df = pd.DataFrame(te_ary, columns=te.columns_)
    
    refund_itemsets = apriori(refund_df, min_support=0.005, use_colnames=True)
    refund_rules = association_rules(refund_itemsets, metric="confidence", min_threshold=0.4)
    
    # 保存结果
    refund_rules.to_parquet("task4_refund_rules.parquet")
    
    
    return refund_rules


if __name__ == "__main__":
    # 执行所有任务
    task1_results = task1_analysis()
    task2_results = task2_analysis()
    task3_results = task3_analysis()
    task4_results = task4_analysis()
    
    # 生成报告数据
    report_data = {
        'electronics_rules_count': len(task1_results),
        'high_value_payment': task2_results.head(10).to_dict(),
        'top_seasonal': task3_results.groupby('category')['count'].sum().nlargest(5).to_dict(),
        'refund_combinations': task4_results[['antecedents','consequents']].head(5).to_dict()
    }
    
    # 保存报告数据
    with open('analysis_report.json', 'w') as f:
        json.dump(report_data, f, ensure_ascii=False)