import os
import random
import shutil
from pathlib import Path

# 修正：将 raw_dir 指向你实际存放数据的路径
root_dir = Path(__file__).resolve().parent.parent
base_dir = root_dir / 'data' / 'datasets'
raw_dir = root_dir / 'data' / 'datasets' / 'raw'

# YOLO 标准目录结构
train_img_dir = base_dir / 'images/train'
val_img_dir = base_dir / 'images/val'
train_lbl_dir = base_dir / 'labels/train'
val_lbl_dir = base_dir / 'labels/val'

# 创建目录
for d in [train_img_dir, val_img_dir, train_lbl_dir, val_lbl_dir]:
    d.mkdir(parents=True, exist_ok=True)

# 搜索 raw 目录下图片
all_images = []
for ext in ['*.jpg', '*.png', '*.jpeg', '*.JPG', '*.PNG']:
    all_images.extend(list(raw_dir.rglob(ext)))

# 匹配标签并划分
matched_pairs = []
for img_path in all_images:
    txt_name = img_path.stem + '.poses'
    txt_path = img_path.parent / txt_name

    if not txt_path.exists():
        found = list(raw_dir.rglob(txt_name))
        if found:
            txt_path = found[0]

    if txt_path.exists():
        matched_pairs.append((img_path, txt_path))

print(f"✅ 成功找到 {len(matched_pairs)} 对匹配的图片和标签！")

if len(matched_pairs) > 0:
    random.shuffle(matched_pairs)
    split_idx = int(len(matched_pairs) * 0.8)
    train_pairs = matched_pairs[:split_idx]
    val_pairs = matched_pairs[split_idx:]


    def copy_files(pairs, split_type):
        img_dest = train_img_dir if split_type == 'train' else val_img_dir
        lbl_dest = train_lbl_dir if split_type == 'train' else val_lbl_dir
        for img_p, txt_p in pairs:
            shutil.copy(img_p, img_dest / img_p.name)
            shutil.copy(txt_p, lbl_dest / txt_p.name)


    copy_files(train_pairs, 'train')
    copy_files(val_pairs, 'val')
    print("🎉 数据集划分完成！图片已存入 data/datasets/images 目录。")
else:
    print("❌ 未找到有效文件，请检查 data/datasets/raw 目录下是否有图片和同名标签文件。")