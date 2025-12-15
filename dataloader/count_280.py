import os

def count_images_in_folder(folder_path, extensions=['.jpg', '.jpeg', '.png', '.bmp', '.gif']):
    """统计指定文件夹中图片的数量"""
    count = 0
    try:
        for file in os.listdir(folder_path):
            if any(file.lower().endswith(ext) for ext in extensions):
                count += 1
    except Exception as e:
        print(f"处理文件夹 {folder_path} 时出错: {str(e)}")
    return count

def find_folders_with_many_images(root_path, min_images=300):
    """查找包含指定数量以上图片的文件夹"""
    folders_with_many_images = []
    total_folders = 0
    
    for root, dirs, files in os.walk(root_path):
        total_folders += 1
        image_count = count_images_in_folder(root)
        
        if image_count >= min_images:
            folders_with_many_images.append({
                'path': root,
                'count': image_count
            })
            
    # 输出结果
    print(f"\n总共扫描的文件夹数量: {total_folders}")
    print(f"包含{min_images}张或更多图片的文件夹数量: {len(folders_with_many_images)}")
    print("\n具体文件夹列表:")

if __name__ == "__main__":
    # 在这里替换为你要扫描的根目录路径
    root_directory = ""
    min_image_count = 280
    
    print(f"开始扫描文件夹: {root_directory}")
    print(f"查找包含{min_image_count}张或更多图片的文件夹...")
    
    find_folders_with_many_images(root_directory, min_image_count)
