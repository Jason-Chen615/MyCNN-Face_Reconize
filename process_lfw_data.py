import os
import shutil
import random
from pathlib import Path
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def check_paths(lfw_path, test_pairs_path):
    """检查数据集路径是否存在"""
    if not os.path.exists(lfw_path):
        raise FileNotFoundError(f"数据集路径不存在: {lfw_path}")
    if not os.path.exists(test_pairs_path):
        raise FileNotFoundError(f"测试对文件不存在: {test_pairs_path}")
    
    # 检查数据集文件夹内容
    files = os.listdir(lfw_path)
    logging.info(f"数据集文件夹中包含 {len(files)} 个文件/文件夹")
    if len(files) > 0:
        logging.info(f"前5个文件/文件夹: {files[:5]}")

def create_directory_structure():
    """创建必要的文件夹结构"""
    directories = ['faces_my', 'faces_sxx', 'faces_wtt', 'faces_other']
    for dir_name in directories:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        else:
            # 清空文件夹
            for file in os.listdir(dir_name):
                file_path = os.path.join(dir_name, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)
        logging.info(f"创建/清空文件夹: {dir_name}")

def get_person_name(img_path):
    """从图片路径中提取人名"""
    return img_path.split('/')[0]

def process_lfw_data(lfw_path, test_pairs_path):
    """处理LFW数据集"""
    # 首先检查路径
    check_paths(lfw_path, test_pairs_path)
    
    # 创建文件夹结构
    create_directory_structure()
    
    # 用于跟踪已分配的人物
    person_assignments = {}  # 记录每个人被分配到哪个文件夹
    used_images = set()
    processed_count = 0
    error_count = 0
    
    # 遍历所有人物文件夹
    for person_dir in os.listdir(lfw_path):
        person_path = os.path.join(lfw_path, person_dir)
        if not os.path.isdir(person_path):
            continue
            
        # 获取该人物的所有图片
        person_images = [f for f in os.listdir(person_path) if f.endswith('.jpg')]
        if not person_images:
            continue
            
        # 如果这个人还没有被分配到任何文件夹
        if person_dir not in person_assignments:
            # 随机选择一个文件夹
            person_assignments[person_dir] = random.choice(['faces_my', 'faces_sxx', 'faces_wtt', 'faces_other'])
            
        # 复制该人物的所有图片
        for img_name in person_images:
            try:
                src_path = os.path.join(person_path, img_name)
                dst_path = os.path.join(person_assignments[person_dir], f"{person_dir}_{img_name}")
                
                if src_path not in used_images:
                    shutil.copy2(src_path, dst_path)
                    used_images.add(src_path)
                    processed_count += 1
                    
                if processed_count % 1000 == 0:
                    logging.info(f"已处理 {processed_count} 张图片")
                    
            except Exception as e:
                logging.error(f"处理图片时出错: {src_path}, 错误: {str(e)}")
                error_count += 1
                continue
    
    # 统计每个文件夹中的图片数量
    for dir_name in ['faces_my', 'faces_sxx', 'faces_wtt', 'faces_other']:
        count = len([f for f in os.listdir(dir_name) if os.path.isfile(os.path.join(dir_name, f))])
        logging.info(f"{dir_name} 中包含 {count} 张图片")
            
    logging.info(f"处理完成！成功处理 {processed_count} 张图片，遇到 {error_count} 个错误")

if __name__ == "__main__":
    lfw_path = r"D:\研一\研一下\数据挖掘\lfw-align-128\lfw-align-128"  # 更新后的路径
    test_pairs_path = r"D:\研一\研一下\数据挖掘\lfw_test_pair.txt"
    
    logging.info("开始处理LFW数据集...")
    process_lfw_data(lfw_path, test_pairs_path)
    logging.info("数据集处理完成！") 