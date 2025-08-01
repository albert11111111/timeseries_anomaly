import os
import glob
import subprocess
import time
import pandas as pd
from tqdm import tqdm

def get_feature_num():
    df = pd.read_excel('Light_data.xlsx')
    exclude_cols = ['Unnamed: 0', '工况时间', 'Class']
    feature_columns = [col for col in df.columns if col not in exclude_cols]
    return len(feature_columns)

def check_model_results(model_name):
    """检查模型是否已完成训练"""
    result_pattern = f"Time-Series-Library/results/anomaly_detection/LIGHT_SMAP_{model_name}_*"
    result_dirs = glob.glob(result_pattern)
    
    if result_dirs:
        print(f"✓ 模型 {model_name} 已完成训练")
        for result_dir in result_dirs:
            result_files = glob.glob(os.path.join(result_dir, "*.npy"))
            print(f"  结果文件: {len(result_files)} 个")
            for f in result_files:
                print(f"    {os.path.basename(f)}")
        return True
    else:
        print(f"✗ 模型 {model_name} 尚未完成训练")
        return False

def run_model(model_name):
    """运行指定模型"""
    print(f"开始运行模型 {model_name}...")
    feature_num = get_feature_num()  # 先获取特征数
    os.chdir("Time-Series-Library")  # 再切换目录
    cmd = [
        "python", "-u", "run.py",
        "--task_name", "anomaly_detection",
        "--is_training", "1",
        "--root_path", "./dataset/LIGHT_SMAP",
        "--model_id", "LIGHT_SMAP",
        "--model", model_name,
        "--data", "LIGHT_SMAP",
        "--features", "M",
        "--seq_len", "100",
        "--pred_len", "0",
        "--d_model", "128",
        "--d_ff", "128",
        "--e_layers", "3",
        f"--enc_in", str(feature_num),
        f"--c_out", str(feature_num),
        "--anomaly_ratio", "1",
        "--batch_size", "128",
        "--train_epochs", "3"
    ]
    if model_name == "TimesNet":
        cmd.extend(["--top_k", "3"])
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        print(f"模型 {model_name} 训练完成")
        # 打印loss相关信息
        loss_lines = [line for line in result.stdout.split('\n') if 'loss' in line.lower()]
        if loss_lines:
            print(f"{model_name} 训练过程中的loss信息：")
            for line in loss_lines[-10:]:  # 只显示最后10条loss
                print(line)
        else:
            print("未检测到loss信息。")
        print("STDOUT:", result.stdout[-500:])
        if result.stderr:
            print("STDERR:", result.stderr[-500:])
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print(f"模型 {model_name} 训练超时")
        return False
    except Exception as e:
        print(f"运行模型 {model_name} 时出错: {e}")
        return False
    finally:
        os.chdir("..")

def main():
    """主函数"""
    models = ['TimesNet', 'Transformer', 'Autoformer']
    
    print("=== 检查模型训练状态 ===")
    completed_models = []
    pending_models = []
    
    for model in models:
        if check_model_results(model):
            completed_models.append(model)
        else:
            pending_models.append(model)
    
    print(f"\n已完成的模型: {completed_models}")
    print(f"待训练的模型: {pending_models}")
    
    # 运行待训练的模型
    if pending_models:
        print("\n=== 开始训练待训练的模型 ===")
        for model in tqdm(pending_models, desc="模型训练进度"):
            print(f"\n开始训练 {model}...")
            success = run_model(model)
            if success:
                print(f"✓ {model} 训练成功")
                completed_models.append(model)
            else:
                print(f"✗ {model} 训练失败")
    
    print(f"\n=== 最终状态 ===")
    print(f"成功训练的模型: {completed_models}")
    
    # 如果所有模型都完成，运行结果处理
    if len(completed_models) >= 1:  # 至少有一个模型完成
        print("\n=== 开始处理结果 ===")
        try:
            # 导入并运行单独模型评估
            import sys
            sys.path.append('.')
            from process_results_individual import compare_all_models
            
            best_model, best_score, all_results = compare_all_models()
            if best_model:
                print(f"异常检测完成！最佳模型: {best_model}, 得分: {best_score}")
            else:
                print("结果处理失败")
        except Exception as e:
            print(f"结果处理出错: {e}")
    else:
        print("没有模型训练成功，无法处理结果")

if __name__ == "__main__":
    main() 