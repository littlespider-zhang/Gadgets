import os
import csv
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from pathlib import Path

def calculate_otsu_threshold(gray):
    """纯 NumPy 实现的 Otsu 自动阈值（无需 OpenCV）"""
    if gray.size == 0:
        return 128
    hist, _ = np.histogram(gray, bins=256, range=(0, 256))
    hist = hist.astype(np.float32)
    total_pixels = hist.sum()
    if total_pixels == 0:
        return 128
    probs = hist / total_pixels
    cum_probs = np.cumsum(probs)
    cum_means = np.cumsum(probs * np.arange(256))
    total_mean = cum_means[-1]
    max_variance = 0
    best_threshold = 0
    for t in range(1, 256):
        w0 = cum_probs[t]
        if w0 == 0 or w0 == 1:
            continue
        w1 = 1.0 - w0
        mean0 = cum_means[t] / w0
        mean1 = (total_mean - cum_means[t]) / w1
        variance = w0 * w1 * (mean0 - mean1) ** 2
        if variance > max_variance:
            max_variance = variance
            best_threshold = t
    return best_threshold

def main():
    print("=== FRAP 荧光强度分析脚本（用户自定义第一帧圆心 + 动态追踪版） ===")
    print("原始图片格式为 JPG，按文件名排序处理时间序列")
    print("【核心更新】")
    print("   • 处理第一张图片时：自动计算 droplet 几何中心 → 让用户手动输入「计算强度的圆心坐标」")
    print("   • 根据用户输入的圆心，自动计算其相对于 droplet 中心的「相对偏移量」（rel_dx, rel_dy）")
    print("   • 后续每张图片：先计算该帧 droplet 几何中心 → 再加上第一帧的相对偏移 → 放置计算圆形区域")
    print("   • 实现精准手动指定 + 动态追踪（可应对漂移）")
    print("依赖包：Pillow（PIL） + NumPy\n")

    # 用户输入
    folder_path = input("请输入图片文件夹的完整路径（例如：C:\\data\\frap_images）： ").strip()
    if not os.path.isdir(folder_path):
        print("❌ 错误：文件夹不存在，请检查路径！")
        return

    try:
        x1 = int(input("请输入目标区域左上角 X 坐标（像素）： "))
        y1 = int(input("请输入目标区域左上角 Y 坐标（像素）： "))
        x2 = int(input("请输入目标区域右下角 X 坐标（像素）： "))
        y2 = int(input("请输入目标区域右下角 Y 坐标（像素）： "))
        radius = int(input("请输入计算荧光强度的圆形区域半径（像素，例如 20）： "))
    except ValueError:
        print("❌ 错误：坐标和半径必须是整数！")
        return

    if x1 >= x2 or y1 >= y2 or radius <= 0:
        print("❌ 错误：坐标范围或半径无效！")
        return

    # 创建标记图片输出文件夹
    marked_folder = os.path.join(folder_path, "marked_images")
    os.makedirs(marked_folder, exist_ok=True)
    print(f"✅ 标记图片将保存到：{marked_folder}\n")

    # 获取所有 JPG 文件并按文件名排序
    image_files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg'))])
    if not image_files:
        print("❌ 错误：文件夹中没有找到 JPG 图片！")
        return

    print(f"✅ 找到 {len(image_files)} 张图片，开始处理...\n")

    # 准备变量
    first_droplet_center = None
    rel_dx = 0
    rel_dy = 0
    first_calc_cx = None
    first_calc_cy = None

    # 处理所有图片
    results = []
    for idx, filename in enumerate(image_files):
        img_path = os.path.join(folder_path, filename)
        try:
            pil_img = Image.open(img_path)
        except Exception:
            print(f"⚠️  跳过无法读取的图片：{filename}")
            continue

        # 裁剪目标区域
        crop_box = (x1, y1, x2, y2)
        gray_crop = np.array(pil_img.convert('L').crop(crop_box))
        if gray_crop.size == 0:
            print(f"⚠️  跳过：{filename} 裁剪区域无效")
            continue

        height, width = gray_crop.shape

        # 计算当前图片的 droplet 几何中心（亮区质心）
        thresh = calculate_otsu_threshold(gray_crop)
        bright_mask = gray_crop > thresh
        y_coords, x_coords = np.nonzero(bright_mask)

        if len(x_coords) == 0:
            if idx == 0:
                print("❌ 错误：第一张图片的目标区域内未检测到有效亮区！")
                return
            else:
                # 漂移严重或完全漂白时，使用第一张 droplet 中心作为 fallback
                new_cx, new_cy = first_droplet_center
                print(f"⚠️  {filename} 未检测到亮区，使用第一张图片 droplet 中心作为 fallback")
        else:
            new_cx = int(np.mean(x_coords))
            new_cy = int(np.mean(y_coords))

        # === 第一张图片特殊处理：让用户输入计算强度的圆心坐标 ===
        if idx == 0:
            first_droplet_center = (new_cx, new_cy)
            print(f"✅ 第一张图片 droplet 几何中心已自动计算：({new_cx}, {new_cy})（相对目标区域）")

            # 让用户手动输入计算强度的圆心坐标
            try:
                calc_cx_input = int(input(f"请输入第一张图片【计算强度的圆心】X 坐标（相对目标区域，像素，建议范围 0~{width-1}）： "))
                calc_cy_input = int(input(f"请输入第一张图片【计算强度的圆心】Y 坐标（相对目标区域，像素，建议范围 0~{height-1}）： "))
            except ValueError:
                print("❌ 错误：圆心坐标必须是整数！")
                return

            # 边界检查
            if not (0 <= calc_cx_input < width and 0 <= calc_cy_input < height):
                print("❌ 错误：输入的圆心坐标超出目标区域范围！")
                return

            first_calc_cx = calc_cx_input
            first_calc_cy = calc_cy_input

            # 计算相对偏移量（后续帧会使用此偏移）
            rel_dx = first_calc_cx - new_cx
            rel_dy = first_calc_cy - new_cy

            print(f"✅ 已记录第一帧计算圆心：({first_calc_cx}, {first_calc_cy})")
            print(f"   相对 droplet 中心的偏移量：({rel_dx}, {rel_dy})")
            print(f"   后续图片将以此偏移动态追踪圆心位置\n")

            # 第一帧实际使用的圆心就是用户输入的值
            calc_cx = first_calc_cx
            calc_cy = first_calc_cy

        else:
            # 后续帧：droplet中心 + 相对偏移
            calc_cx = new_cx + rel_dx
            calc_cy = new_cy + rel_dy

        # 防止圆心超出裁剪区域
        calc_cx = max(0, min(calc_cx, width - 1))
        calc_cy = max(0, min(calc_cy, height - 1))

        # 创建圆形掩膜
        Y, X = np.ogrid[:height, :width]
        dist_from_center = np.sqrt((X - calc_cx)**2 + (Y - calc_cy)**2)
        mask = dist_from_center <= radius

        # 计算平均荧光强度
        if np.any(mask):
            mean_intensity = float(np.mean(gray_crop[mask]))
        else:
            mean_intensity = 0.0

        # === 可视化标记到原始图片 ===
        draw_img = pil_img.convert('RGB').copy()
        draw = ImageDraw.Draw(draw_img)

        # 1. 绿色矩形（目标区域）
        draw.rectangle([(x1, y1), (x2, y2)], outline=(0, 255, 0), width=3)

        # 2. 红色圆形（实际计算区域）
        center_full = (x1 + calc_cx, y1 + calc_cy)
        draw.ellipse(
            [(center_full[0] - radius, center_full[1] - radius),
             (center_full[0] + radius, center_full[1] + radius)],
            outline=(0, 0, 255), width=3
        )

        # 3. 文字信息（文件名 + 强度 + 当前 droplet 中心 + 计算圆心）
        text = f"{filename} | Intensity:{mean_intensity:.2f} | Droplet:({new_cx},{new_cy}) | Calc:({calc_cx},{calc_cy})"
        font = ImageFont.load_default()
        text_y = y1 - 55 if y1 > 60 else y1 + 10
        # 黑色描边
        for dx, dy in [(-1,-1), (-1,1), (1,-1), (1,1), (0,-2), (0,2), (-2,0), (2,0)]:
            draw.text((x1 + 12 + dx, text_y + dy), text, font=font, fill=(0, 0, 0))
        # 白色文字
        draw.text((x1 + 12, text_y), text, font=font, fill=(255, 255, 255))

        # 保存带标记的图片
        marked_path = os.path.join(marked_folder, f"marked_{filename}")
        draw_img.save(marked_path, quality=95)

        results.append((filename, round(mean_intensity, 4)))
        status = "（第一帧用户指定）" if idx == 0 else "（动态追踪）"
        # print(f"[{idx+1:03d}/{len(image_files)}] {filename} → 荧光强度: {mean_intensity:.4f}  |  ✅ 已生成标记图 {status}")

    # 保存结果为 CSV
    csv_path = os.path.join(folder_path, "FRAP_荧光强度结果.csv")
    with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['文件名', '平均荧光强度（灰度值）'])
        writer.writerows(results)

    print("\n🎉 处理完成！")
    print(f"   结果已保存到：{csv_path}")
    print(f"   带标记参考图片已保存到：{marked_folder}（共 {len(results)} 张）")
    print(f"   共处理 {len(results)} 张图片")
    # print("\n【本次更新亮点】")
    # print("   - 第一张图片时，用户可手动指定计算强度的精确圆心坐标")
    # print("   - 自动计算该圆心相对于 droplet 中心的偏移，后续帧自动追踪")
    # print("   - 标记图中会显示：Droplet（droplet 中心） + Calc（实际计算圆心）")
    # print("   - 若某帧无法检测亮区，会 fallback 使用第一帧 droplet 中心")
    # print("提示：如需增加背景扣除、归一化曲线图或其他功能，随时告诉我！")

if __name__ == "__main__":
    main()