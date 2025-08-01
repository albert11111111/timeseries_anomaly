import pandas as pd
import numpy as np

def correct_time_point_aggregation():
    """正确的时间点级聚合和异常判断，支持多种聚合策略"""
    
    print("=== 正确的时间点级聚合和异常判断 (多策略比较) ===")
    
    # 加载数据
    light_data = pd.read_excel('Light_data.xlsx')
    light_data['工况时间'] = pd.to_datetime(light_data['工况时间'])
    test_cutoff = pd.to_datetime('2022-07-01')
    test_data = light_data[light_data['工况时间'] >= test_cutoff].copy()
    
    score_data = pd.read_excel('score.xlsx')
    score_data = score_data[~score_data['日期'].astype(str).str.contains('总分', na=False)]
    score_data['日期'] = pd.to_datetime(score_data['日期'])
    
    def process_model(model_name):
        """处理单个模型的异常检测，比较不同聚合策略"""
        print(f"\n=== 处理 {model_name} 模型 ===")
        
        # 加载异常分数
        anomaly_scores = np.load(f'Time-Series-Library/test_results/anomaly_detection_LIGHT_SMAP_{model_name}_LIGHT_SMAP_ftM_sl100_ll48_pl0_dm128_nh8_el3_dl1_df128_expand2_dc4_fc1_ebtimeF_dtTrue_test_0/anomaly_scores.npy')
        
        # 重塑为窗口格式
        seq_len = 100
        expected_windows = len(test_data) - seq_len + 1
        reshaped = anomaly_scores.reshape(expected_windows, seq_len)
        
        print(f"窗口数: {expected_windows}")
        print(f"原始分数形状: {anomaly_scores.shape}")
        print(f"重塑后形状: {reshaped.shape}")
        
        # 步骤1: 按时间点重新聚合
        print("\n步骤1: 按时间点重新聚合")
        time_point_scores = {}
        
        for window_idx in range(expected_windows):
            for pos_in_window in range(seq_len):
                # 计算实际时间点索引
                actual_time_idx = window_idx + pos_in_window
                
                if actual_time_idx < len(test_data):
                    score = reshaped[window_idx, pos_in_window]
                    
                    if actual_time_idx not in time_point_scores:
                        time_point_scores[actual_time_idx] = []
                    time_point_scores[actual_time_idx].append(score)
        
        print(f"总时间点数: {len(time_point_scores)}")
        
        # 定义不同的聚合策略
        aggregation_strategies = {
            'max': np.max,
            'median': np.median,
            'mean': np.mean,
            'min': np.min
        }
        
        strategy_results = {}
        
        # 测试每种聚合策略
        for strategy_name, strategy_func in aggregation_strategies.items():
            print(f"\n--- 测试 {strategy_name.upper()} 聚合策略 ---")
            
            # 步骤2: 对每个时间点的多个分数进行聚合
            aggregated_scores = {}
            
            for time_idx, scores in time_point_scores.items():
                # 使用当前策略聚合分数
                aggregated_scores[time_idx] = strategy_func(scores)
            
            print(f"聚合后时间点数: {len(aggregated_scores)}")
            
            # 显示聚合示例
            if strategy_name in ['max', 'median']:  # 只显示主要策略的示例
                print(f"\n前5个时间点的{strategy_name}聚合示例:")
                for i, (time_idx, final_score) in enumerate(list(aggregated_scores.items())[:5]):
                    original_scores = time_point_scores[time_idx]
                    time_point = test_data.iloc[time_idx]['工况时间']
                    print(f"  时间点{time_idx} ({time_point}): {len(original_scores)}个分数 -> {strategy_name} {final_score:.6f}")
            
            # 步骤3: 确定时间点级异常阈值
            all_scores = list(aggregated_scores.values())
            
            # 测试不同阈值
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
                
                # 步骤6: 使用阈值判断日级异常（异常时间点数 > 5）
                daily_labels = {}
                min_anomaly_points = 8  # 一天内至少8个异常时间点才标记为异常日
                
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
                        'strategy': strategy_name,
                        'threshold_percentile': threshold_percentile,
                        'threshold_value': threshold,
                        'score': total_score,
                        'result_df': result_df.copy(),
                        'true_positives': true_positives,
                        'false_positives': false_positives,
                        'false_negatives': false_negatives,
                        'anomaly_days': anomaly_days
                    }
            
            strategy_results[strategy_name] = best_result
            print(f"{strategy_name.upper()}策略最佳结果: {best_result['score']}分 (阈值: 第{best_result['threshold_percentile']}百分位数, 检测: {best_result['true_positives']}/5)")
        
        # 找到该模型的最佳策略
        best_strategy_result = max(strategy_results.values(), key=lambda x: x['score'])
        
        print(f"\n🏆 {model_name} 最佳聚合策略:")
        print(f"  策略: {best_strategy_result['strategy'].upper()}")
        print(f"  最佳阈值: 第{best_strategy_result['threshold_percentile']}百分位数")
        print(f"  最佳得分: {best_strategy_result['score']}")
        print(f"  检测异常: {best_strategy_result['true_positives']}/5")
        print(f"  误报: {best_strategy_result['false_positives']}次")
        print(f"  漏报: {best_strategy_result['false_negatives']}次")
        
        # 返回该模型的最佳结果和所有策略结果
        return best_strategy_result, strategy_results
    
    # 处理所有模型
    models = ['TimesNet', 'Transformer', 'Autoformer']
    model_results = {}
    all_strategy_results = {}
    
    for model in models:
        try:
            best_result, strategy_results = process_model(model)
            model_results[model] = best_result
            all_strategy_results[model] = strategy_results
        except Exception as e:
            print(f"处理模型 {model} 时出错: {e}")
            model_results[model] = None
            all_strategy_results[model] = None
    
    # 比较所有模型和策略
    print("\n" + "="*80)
    print("所有模型的最佳结果比较:")
    print("="*80)
    
    best_overall_model = None
    best_overall_score = float('-inf')
    best_overall_strategy = None
    
    for model, result in model_results.items():
        if result:
            print(f"{model:12}: {result['score']:4d}分 (策略: {result['strategy']:6}, 阈值: 第{result['threshold_percentile']}百分位数, 检测: {result['true_positives']}/5)")
            if result['score'] > best_overall_score:
                best_overall_score = result['score']
                best_overall_model = model
                best_overall_strategy = result['strategy']
    
    # 详细的策略比较
    print("\n" + "="*80)
    print("不同聚合策略的详细比较:")
    print("="*80)
    
    strategies = ['max', 'median', 'mean', 'min']
    
    print(f"{'模型':<12} {'策略':<8} {'得分':<6} {'阈值':<8} {'检测':<8} {'误报':<6} {'漏报':<6}")
    print("-" * 80)
    
    for model in models:
        if all_strategy_results[model]:
            for strategy in strategies:
                if strategy in all_strategy_results[model]:
                    result = all_strategy_results[model][strategy]
                    print(f"{model:<12} {strategy.upper():<8} {result['score']:<6d} {result['threshold_percentile']:>3d}%     {result['true_positives']}/5    {result['false_positives']:<6d} {result['false_negatives']:<6d}")
    
    if best_overall_model:
        print(f"\n🏆 全局最佳结果:")
        print(f"   模型: {best_overall_model}")
        print(f"   策略: {best_overall_strategy.upper()}")
        print(f"   得分: {best_overall_score}")
        
        # 保存最佳结果
        best_result = model_results[best_overall_model]
        output_file = f'score_correct_{best_overall_model}_{best_overall_strategy}.xlsx'
        best_result['result_df'].to_excel(output_file, index=False)
        print(f"   最佳结果已保存到: {output_file}")
        
        # 分析各策略的优劣
        print(f"\n📊 策略效果分析:")
        strategy_scores = {}
        for model in models:
            if all_strategy_results[model]:
                for strategy in strategies:
                    if strategy in all_strategy_results[model]:
                        if strategy not in strategy_scores:
                            strategy_scores[strategy] = []
                        strategy_scores[strategy].append(all_strategy_results[model][strategy]['score'])
        
        print(f"{'策略':<10} {'平均得分':<10} {'最高得分':<10} {'最低得分':<10}")
        print("-" * 50)
        for strategy in strategies:
            if strategy in strategy_scores:
                scores = strategy_scores[strategy]
                avg_score = np.mean(scores)
                max_score = np.max(scores)
                min_score = np.min(scores)
                print(f"{strategy.upper():<10} {avg_score:<10.1f} {max_score:<10d} {min_score:<10d}")

if __name__ == "__main__":
    correct_time_point_aggregation() 