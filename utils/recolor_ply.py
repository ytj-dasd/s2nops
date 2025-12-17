import numpy as np
from plyfile import PlyData, PlyElement
import argparse
import os
import glob

def recolor_ply_by_label(input_path, output_path=None):
    """
    根据pred_label重新着色PLY文件
    
    Args:
        input_path: 输入PLY文件路径
        output_path: 输出PLY文件路径，如果为None则在原文件名后加_recolored
    """
    
    # color_map = {
    #     0: [0, 255, 255],     # Ground - 青色
    #     1: [255, 255, 0],       # Road marking - 黄色   
    #     2: [0, 255, 0],       # Natural/Tree - 绿色
    #     3: [114, 98, 84],     # Person - 深紫色
    #     4: [128, 0, 128],     # Low vegetation - 橙色
    #     5: [0, 0, 255],     # Pole - 亮黄色
    #     6: [255, 0,  51],       # Car - 蓝色
    #     7: [128, 128, 128]    # Fence - 灰色
    # }
    
    color_map = {
        0: [255, 255, 255],   # unlabeled     
        1: [0, 255, 255],     # Ground - 青色
        2: [255, 255, 0],     # Road marking - 黄色   
        3: [0, 255, 0],       # Natural/Tree - 绿色
        4: [114, 98, 84],     # Person - 深紫色
        5: [128, 0, 128],     # Low vegetation - 橙色
        6: [0, 0, 255],     # Pole - 亮黄色
        7: [255, 0, 51],       # Car - 蓝色
        8: [128, 128, 128]    # Fence - 灰色
    }
    
    # 读取PLY文件
    plydata = PlyData.read(input_path)
    vertex = plydata['vertex']
    
    # 提取数据
    x = vertex['x']
    y = vertex['y'] 
    z = vertex['z']
    labels = vertex['semantics']
    
    # 根据标签映射颜色
    colors = np.array([color_map.get(label, [128, 128, 128]) for label in labels])
    
    # 创建新的顶点数据
    vertex_data = np.array([
        (x[i], y[i], z[i], colors[i, 0], colors[i, 1], colors[i, 2], labels[i])
        for i in range(len(x))
    ], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), 
              ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'), ('pred_label', 'u1')])
    
    vertex_element = PlyElement.describe(vertex_data, 'vertex')
    
    # 确定输出路径
    if output_path is None:
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}_recolored{ext}"
    
    # 保存PLY文件
    PlyData([vertex_element]).write(output_path)
    print(f"Recolored PLY saved to: {output_path}")

def process_folder(input_folder, output_folder=None):
    """
    处理文件夹中的所有PLY文件
    
    Args:
        input_folder: 输入文件夹路径
        output_folder: 输出文件夹路径，如果为None则在输入文件夹下创建recolored子文件夹
    """
    if output_folder is None:
        output_folder = os.path.join(input_folder, 'recolored')
    
    os.makedirs(output_folder, exist_ok=True)
    
    # 查找所有PLY文件
    ply_files = glob.glob(os.path.join(input_folder, '*.ply'))
    
    if not ply_files:
        print(f"No PLY files found in {input_folder}")
        return
    
    print(f"Found {len(ply_files)} PLY files to process")
    
    for ply_file in ply_files:
        filename = os.path.basename(ply_file)
        output_path = os.path.join(output_folder, filename)
        
        try:
            recolor_ply_by_label(ply_file, output_path)
        except Exception as e:
            print(f"Error processing {filename}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Recolor PLY files based on pred_label")
    parser.add_argument("input", help="Input PLY file or folder path")
    parser.add_argument("-o", "--output", help="Output PLY file or folder path")
    
    args = parser.parse_args()
    
    if os.path.isdir(args.input):
        process_folder(args.input, args.output)
    else:
        recolor_ply_by_label(args.input, args.output)
