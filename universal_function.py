import csv
import os

def save_downsampled_csv(downsampled_data, downsampled_labels,
                         base_dir, prefix="downsampled", sample_idx=0):
    #将下采样后的第 sample_idx 条样本保存为：base_dir/prefix-XX.csv   每行格式：SCL, SDA, Label
    os.makedirs(base_dir, exist_ok=True)
    # 自动编号文件名
    existing = [f for f in os.listdir(base_dir)
                if f.startswith(prefix+"-") and f.endswith(".csv")]
    nums = []
    for fn in existing:
        try:
            nums.append(int(fn[len(prefix)+1:fn.rfind(".csv")]))
        except:
            pass
    next_no = max(nums)+1 if nums else 1
    fname = f"{prefix}-{next_no:02d}.csv"
    path = os.path.join(base_dir, fname)

    scl = downsampled_data[sample_idx, :, 0]
    sda = downsampled_data[sample_idx, :, 1]
    lbl = downsampled_labels[sample_idx]

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["SCL", "SDA", "Label"])
        for i in range(len(scl)):
            writer.writerow([scl[i], sda[i], int(lbl[i])])

    # print(f"Saved downsampled sample {sample_idx} → {path}")