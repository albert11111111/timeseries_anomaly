import pandas as pd
import numpy as np
import os
import shutil
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class LightDataProcessor:
    """处理灯杆数据并转换为SMAP格式"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        
    def load_data(self):
        """加载原始数据"""
        print("正在加载数据...")
        self.light_data = pd.read_excel('Light_data.xlsx')
        self.score_data = pd.read_excel('score.xlsx')
        
        # 过滤掉非日期行（如包含"总分"的行）
        self.score_data = self.score_data[~self.score_data['日期'].astype(str).str.contains('总分', na=False)]
        
        # 转换时间列
        self.light_data['工况时间'] = pd.to_datetime(self.light_data['工况时间'])
        self.score_data['日期'] = pd.to_datetime(self.score_data['日期'])
        
        print(f"Light_data形状: {self.light_data.shape}")
        print(f"时间范围: {self.light_data['工况时间'].min()} 到 {self.light_data['工况时间'].max()}")
        print(f"Score_data形状: {self.score_data.shape}")
        print(f"Score_data时间范围: {self.score_data['日期'].min()} 到 {self.score_data['日期'].max()}")
        
    def prepare_features(self):
        """准备特征数据 - 选择与SMAP相同数量的特征"""
        print("正在准备特征...")
        
        # 自动选择所有有效特征（去除'Unnamed: 0'、'工况时间'、'Class'）
        exclude_cols = ['Unnamed: 0', '工况时间', 'Class']
        self.feature_columns = [col for col in self.light_data.columns if col not in exclude_cols]
        print(f"最终特征列: {self.feature_columns}")
        # 提取特征数据
        self.feature_data = self.light_data[['工况时间'] + self.feature_columns].copy()
        
    def split_train_test_data(self):
        """按时间分割训练和测试数据"""
        print("正在分割训练测试数据...")
        
        # 7月1日之前的数据作为训练集
        train_cutoff = pd.to_datetime('2022-07-01')
        
        train_mask = self.feature_data['工况时间'] < train_cutoff
        test_mask = self.feature_data['工况时间'] >= train_cutoff
        
        self.train_data = self.feature_data[train_mask].copy()
        self.test_data = self.feature_data[test_mask].copy()
        
        print(f"训练数据: {len(self.train_data)} 条记录")
        print(f"测试数据: {len(self.test_data)} 条记录")
        print(f"训练数据时间范围: {self.train_data['工况时间'].min()} 到 {self.train_data['工况时间'].max()}")
        print(f"测试数据时间范围: {self.test_data['工况时间'].min()} 到 {self.test_data['工况时间'].max()}")
        
    def create_smap_format_data(self):
        """创建SMAP格式的数据"""
        print("正在创建SMAP格式数据...")
        
        # 准备训练数据
        train_features = self.train_data[self.feature_columns].values
        test_features = self.test_data[self.feature_columns].values
        
        # 标准化数据
        train_features_scaled = self.scaler.fit_transform(train_features)
        test_features_scaled = self.scaler.transform(test_features)
        
        # 创建标签 - 训练数据全部标记为正常（0）
        train_labels = np.zeros(len(train_features_scaled), dtype=bool)
        
        # 测试数据的标签需要根据每日异常情况生成
        test_labels = self.create_daily_labels()
        
        return train_features_scaled, test_features_scaled, train_labels, test_labels
    
    def create_daily_labels(self):
        """根据每日真实标签创建测试数据的点级别标签"""
        print("正在创建每日标签...")
        
        test_labels = np.zeros(len(self.test_data), dtype=bool)
        
        # 为每个测试数据点分配对应的日期标签
        for idx, row in self.test_data.iterrows():
            date = row['工况时间'].date()
            
            # 找到对应的日期标签
            score_row = self.score_data[self.score_data['日期'].dt.date == date]
            if not score_row.empty:
                daily_label = score_row.iloc[0]['真实标签']
                if daily_label == 1:
                    # 如果这一天是异常的，将这一天的所有数据点标记为异常
                    test_labels[idx - self.test_data.index[0]] = True
        
        return test_labels
    
    def save_smap_format(self, output_dir='dataset/LIGHT_SMAP'):
        """保存为SMAP格式的文件"""
        print("正在保存SMAP格式文件...")
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 获取数据
        train_features, test_features, train_labels, test_labels = self.create_smap_format_data()
        
        # 保存文件
        np.save(os.path.join(output_dir, 'LIGHT_SMAP_train.npy'), train_features)
        np.save(os.path.join(output_dir, 'LIGHT_SMAP_test.npy'), test_features)
        np.save(os.path.join(output_dir, 'LIGHT_SMAP_test_label.npy'), test_labels)
        
        print(f"训练数据保存: {train_features.shape}")
        print(f"测试数据保存: {test_features.shape}")
        print(f"测试标签保存: {test_labels.shape}")
        print(f"测试数据异常比例: {np.mean(test_labels):.4f}")
        
        return output_dir

def create_model_scripts(data_dir, feature_num):
    """创建三个模型的训练脚本"""
    print("正在创建模型训练脚本...")
    os.makedirs('scripts/anomaly_detection/LIGHT_SMAP', exist_ok=True)
    models = ['TimesNet', 'Transformer', 'Autoformer']
    for model in models:
        script_content = f"""export CUDA_VISIBLE_DEVICES=0

python -u run.py \
  --task_name anomaly_detection \
  --is_training 1 \
  --root_path ./{data_dir} \
  --model_id LIGHT_SMAP \
  --model {model} \
  --data LIGHT_SMAP \
  --features M \
  --seq_len 100 \
  --pred_len 0 \
  --d_model 128 \
  --d_ff 128 \
  --e_layers 3 \
  --enc_in {feature_num} \
  --c_out {feature_num} \
  --anomaly_ratio 1 \
  --batch_size 128 \
  --train_epochs 3 
"""
        with open(f'scripts/anomaly_detection/LIGHT_SMAP/{model}.sh', 'w') as f:
            f.write(script_content)
        print(f"创建了 {model}.sh 脚本")

def run_anomaly_detection():
    """运行异常检测"""
    print("开始运行异常检测...")
    
    # 1. 数据处理
    processor = LightDataProcessor()
    processor.load_data()
    processor.prepare_features()
    processor.split_train_test_data()
    
    # 2. 保存SMAP格式数据
    data_dir = processor.save_smap_format()
    
    # 3. 创建模型脚本，传入特征数
    create_model_scripts(data_dir, len(processor.feature_columns))
    
    # 4. 复制必要的配置文件
    print("正在复制配置文件...")
    
    # 确保Time-Series-Library目录存在
    if os.path.exists('Time-Series-Library'):
        # 复制数据到Time-Series-Library中
        dest_dir = 'Time-Series-Library/dataset/LIGHT_SMAP'
        if os.path.exists(dest_dir):
            shutil.rmtree(dest_dir)
        shutil.copytree(data_dir, dest_dir)
        print(f"数据已复制到 {dest_dir}")
        
        # 复制脚本
        script_dest = 'Time-Series-Library/scripts/anomaly_detection/LIGHT_SMAP'
        if os.path.exists(script_dest):
            shutil.rmtree(script_dest)
        shutil.copytree('scripts/anomaly_detection/LIGHT_SMAP', script_dest)
        print(f"脚本已复制到 {script_dest}")
    
    return processor

if __name__ == "__main__":
    processor = run_anomaly_detection()
    print("异常检测准备完成！")
    print("接下来需要运行Time-Series-Library中的模型进行训练和预测。") 