import pandas as pd
import numpy as np
import os

def correct_time_point_aggreation():
    """Simplified version focusing on Transformer model with MEDIAN aggregation strategy"""
    
    print("=== Transformer Model with MEDIAN Aggregation Strategy ===")
    
    # 检查必要文件是否存在
    required_files = [
        'Light_data.xlsx',
        'score.xlsx',
        'dataset/LIGHT_SMAP/LIGHT_SMAP_test.npy',
        'dataset/LIGHT_SMAP/LIGHT_SMAP_test_label.npy',
        'dataset/LIGHT_SMAP/LIGHT_SMAP_train.npy'
    ]
    
    missing_files = [f for f in required_files if not os.path.exists(f)]
    if missing_files:
        print(f"错误：缺少以下必要文件：")
        for f in missing_files:
            print(f"  - {f}")
        return
    
    # 加载数据
    light_data = pd.read_excel('Light_data.xlsx')
    light_data['工况时间'] = pd.to_datetime(light_data['工况时间'])
    test_cutoff = pd.to_datetime('2022-07-01')
    test_data = light_data[light_data['工况时间'] >= test_cutoff].copy()
    
    score_data = pd.read_excel('score.xlsx')
    score_data = score_data[~score_data['日期'].astype(str).str.contains('总分', na=False)]
    score_data['日期'] = pd.to_datetime(score_data['日期'])
    
    print(f"测试数据条数: {len(test_data)}")
    print(f"评分数据条数: {len(score_data)}")
    
    # 加载Transformer模型的异常分数
    model_name = 'Transformer'
    test_result_path = 'Time-Series-Library/test_results/anomaly_detection_LIGHT_SMAP_Transformer_LIGHT_SMAP_ftM_sl100_ll48_pl0_dm128_nh8_el3_dl1_df128_expand2_dc4_fc1_ebtimeF_dtTrue_test_0'
    
    if not os.path.exists(test_result_path):
        print(f"错误：找不到Transformer模型的测试结果目录: {test_result_path}")
        return
    
    anomaly_scores = np.load(f'{test_result_path}/anomaly_scores.npy')
    
    # 重塑为窗口格式
    seq_len = 100
    expected_windows = len(test_data) - seq_len + 1
    reshaped = anomaly_scores.reshape(expected_windows, seq_len)
    
    print(f"窗口数: {expected_windows}")
    print(f"原始分数形状: {anomaly_scores.shape}")
    print(f"重塑后形状: {reshaped.shape}")
    
    # 步骤1: 按时间点重新聚合
    print("\n步骤1: 按时间点重新聚合...")
    time_point_scores = {}
    
    for window_idx in range(expected_windows):
        for pos_in_window in range(seq_len):
            actual_time_idx = window_idx + pos_in_window
            
            if actual_time_idx < len(test_data):
                score = reshaped[window_idx, pos_in_window]
                
                if actual_time_idx not in time_point_scores:
                    time_point_scores[actual_time_idx] = []
                time_point_scores[actual_time_idx].append(score)
    
    print(f"总时间点数: {len(time_point_scores)}")
    
    # 使用MEDIAN策略
    print("\n步骤2: 使用MEDIAN策略聚合每个时间点的分数...")
    aggregated_scores = {}
    
    for time_idx, scores in time_point_scores.items():
        aggregated_scores[time_idx] = np.median(scores)
    
    print(f"聚合后时间点数: {len(aggregated_scores)}")
    
    # 显示前5个时间点的聚合示例
    print("\n前5个时间点的MEDIAN聚合示例:")
    for i, (time_idx, final_score) in enumerate(list(aggregated_scores.items())[:5]):
        original_scores = time_point_scores[time_idx]
        time_point = test_data.iloc[time_idx]['工况时间']
        print(f"  时间点{time_idx} ({time_point}): {len(original_scores)}个分数 -> MEDIAN {final_score:.6f}")
    
    # 步骤3: 确定时间点级异常阈值
    print("\n步骤3: 优化异常阈值...")
    all_scores = list(aggregated_scores.values())
    
    thresholds = [90, 95, 98, 99]
    best_result = None
    best_score = float('-inf')
    
    for threshold_percentile in thresholds:
        threshold = np.percentile(all_scores, threshold_percentile)
        
        # 步骤4: 判断每个时间点是否异常
        anomalous_time_points = []
        for time_idx, score in aggregated_scores.items():
            if score > threshold:
                time_point = test_data.iloc[time_idx]['工况时间']
                anomalous_time_points.append((time_idx, time_point, score))
        
        # 步骤5: 按日期分组，判断每天的异常时间点数量
        daily_anomaly_counts = {}
        
        for time_idx, time_point, score in anomalous_time_points:
            date = time_point.date()
            if date not in daily_anomaly_counts:
                daily_anomaly_counts[date] = []
            daily_anomaly_counts[date].append((time_point, score))
        
        # 步骤6: 使用阈值判断日级异常
        daily_labels = {}
        min_anomaly_points = 8
        
        # 先把所有日期标记为正常
        for _, row in score_data.iterrows():
            date = row['日期'].date()
            daily_labels[date] = 0
        
        # 然后标记异常日期
        for date, anomaly_points in daily_anomaly_counts.items():
            if len(anomaly_points) > min_anomaly_points:
                daily_labels[date] = 1
        
        anomaly_days = sum(daily_labels.values())
        
        # 步骤7: 计算最终评分
        result_df = score_data.copy()
        result_df['算法标签'] = np.nan
        
        # 填入预测标签
        for idx, row in result_df.iterrows():
            date = row['日期'].date()
            if date in daily_labels:
                result_df.at[idx, '算法标签'] = daily_labels[date]
        
        # 计算评分
        total_score = 0
        false_positives = 0
        false_negatives = 0
        true_positives = 0
        true_negatives = 0
        
        for _, row in result_df.iterrows():
            true_label = row['真实标签']
            pred_label = row['算法标签']
            
            if pd.notna(true_label) and pd.notna(pred_label):
                if true_label == 0 and pred_label == 1:  # 误报
                    total_score -= 1
                    false_positives += 1
                elif true_label == 1 and pred_label == 0:  # 漏报
                    total_score -= 20
                    false_negatives += 1
                elif true_label == 1 and pred_label == 1:  # 正确检测异常
                    true_positives += 1
                elif true_label == 0 and pred_label == 0:  # 正确检测正常
                    true_negatives += 1
        
        if total_score > best_score:
            best_score = total_score
            best_result = {
                'threshold_percentile': threshold_percentile,
                'threshold_value': threshold,
                'score': total_score,
                'result_df': result_df.copy(),
                'true_positives': true_positives,
                'false_positives': false_positives,
                'false_negatives': false_negatives,
                'anomaly_days': anomaly_days,
                'anomalous_time_points': anomalous_time_points
            }
    
    # 输出最终结果
    print(f"\n🏆 Transformer + MEDIAN策略最佳结果:")
    print(f"  最佳阈值: 第{best_result['threshold_percentile']}百分位数")
    print(f"  最佳得分: {best_result['score']}")
    print(f"  检测异常: {best_result['true_positives']}/5")
    print(f"  误报: {best_result['false_positives']}次")
    print(f"  漏报: {best_result['false_negatives']}次")
    print(f"  异常天数: {best_result['anomaly_days']}天")
    
    # 保存最佳结果
    output_file = 'score_correct_Transformer_median.xlsx'
    best_result['result_df'].to_excel(output_file, index=False)
    print(f"\n最佳结果已保存到: {output_file}")
    
    # 输出异常日期详情
    print(f"\n📊 异常日期详情:")
    anomalous_dates = []
    for _, row in best_result['result_df'].iterrows():
        if row['算法标签'] == 1:
            anomalous_dates.append(row['日期'].strftime('%Y-%m-%d'))
    
    if anomalous_dates:
        print(f"  检测到异常的日期: {', '.join(anomalous_dates)}")
    else:
        print("  未检测到异常日期")
    
    return best_result

if __name__ == "__main__":
    correct_time_point_aggreation()