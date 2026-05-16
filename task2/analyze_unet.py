import argparse
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib"))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

def detect_threshold_from_histogram(psnr_values, bins=100):
    hist, bin_edges = np.histogram(psnr_values, bins=bins)
    peaks, properties = find_peaks(hist, prominence=0.1 * hist.max())
    if len(peaks) < 2:
        return None
    sorted_peak_indices = peaks[np.argsort(hist[peaks])[-2:]]
    valley_index = (sorted_peak_indices[0] + sorted_peak_indices[1]) // 2
    threshold = (bin_edges[valley_index] + bin_edges[valley_index + 1]) / 2
    return threshold

def main():
    parser = argparse.ArgumentParser(description="分析 PSNR/SSIM CSV 文件")
    parser.add_argument("csv_path", nargs="?", default="psnr_ssim_raw.csv",
                        help="CSV 文件路径，默认为当前目录下的 psnr_ssim_raw.csv")
    parser.add_argument("-o", "--output_dir", default="analysis_output",
                        help="输出目录，默认 analysis_output")
    parser.add_argument("--manual_thresh", type=float, default=None,
                        help="手动指定背景 PSNR 阈值（默认自动检测）")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    try:
        df = pd.read_csv(args.csv_path)
    except FileNotFoundError:
        print(f"错误：找不到文件 {args.csv_path}")
        return
    except Exception as e:
        print(f"读取 CSV 出错：{e}")
        return

    if "PSNR_dB" not in df.columns or "SSIM" not in df.columns:
        print("CSV 文件必须包含 'PSNR_dB' 和 'SSIM' 列。")
        return

    psnr = df["PSNR_dB"].values
    ssim = df["SSIM"].values
    total = len(psnr)

    stats_text = f"原始数据总切片数: {total}\n"
    stats_text += f"\n----- PSNR -----\n"
    stats_text += f"均值: {np.mean(psnr):.4f} dB\n"
    stats_text += f"标准差: {np.std(psnr):.4f}\n"
    stats_text += f"最小值: {np.min(psnr):.4f}\n"
    stats_text += f"下四分位数 (25%): {np.percentile(psnr, 25):.4f}\n"
    stats_text += f"中位数 (50%): {np.percentile(psnr, 50):.4f}\n"
    stats_text += f"上四分位数 (75%): {np.percentile(psnr, 75):.4f}\n"
    stats_text += f"最大值: {np.max(psnr):.4f}\n"
    stats_text += f"\n----- SSIM -----\n"
    stats_text += f"均值: {np.mean(ssim):.6f}\n"
    stats_text += f"标准差: {np.std(ssim):.6f}\n"
    stats_text += f"最小值: {np.min(ssim):.6f}\n"
    stats_text += f"下四分位数 (25%): {np.percentile(ssim, 25):.6f}\n"
    stats_text += f"中位数 (50%): {np.percentile(ssim, 50):.6f}\n"
    stats_text += f"上四分位数 (75%): {np.percentile(ssim, 75):.6f}\n"
    stats_text += f"最大值: {np.max(ssim):.6f}\n"
    
    bg_thresh = args.manual_thresh
    if bg_thresh is None:
        bg_thresh = detect_threshold_from_histogram(psnr)
        if bg_thresh is not None:
            stats_text += f"\n自动检测的背景 PSNR 阈值: {bg_thresh:.2f} dB\n"
        else:
            stats_text += "\n未能自动检测到双峰，无法分离背景切片。\n"
    else:
        stats_text += f"\n手动指定背景 PSNR 阈值: {bg_thresh:.2f} dB\n"
    
    if bg_thresh is not None:
        tissue_mask = psnr <= bg_thresh
        n_tissue = np.sum(tissue_mask)
        n_bg = total - n_tissue
        psnr_tissue = psnr[tissue_mask]
        ssim_tissue = ssim[tissue_mask]

        stats_text += f"背景切片数 (PSNR > {bg_thresh:.1f} dB): {n_bg}\n"
        stats_text += f"组织切片数: {n_tissue}\n"
        if n_tissue > 0:
            stats_text += f"\n----- 组织切片 PSNR -----\n"
            stats_text += f"均值: {np.mean(psnr_tissue):.4f} dB\n"
            stats_text += f"标准差: {np.std(psnr_tissue):.4f}\n"
            stats_text += f"最小值: {np.min(psnr_tissue):.4f}\n"
            stats_text += f"最大值: {np.max(psnr_tissue):.4f}\n"
            stats_text += f"\n----- 组织切片 SSIM -----\n"
            stats_text += f"均值: {np.mean(ssim_tissue):.6f}\n"
            stats_text += f"标准差: {np.std(ssim_tissue):.6f}\n"
            stats_text += f"最小值: {np.min(ssim_tissue):.6f}\n"
            stats_text += f"最大值: {np.max(ssim_tissue):.6f}\n"
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        axes[0, 0].hist(psnr, bins=80, color='steelblue', edgecolor='white', alpha=0.8)
        axes[0, 0].axvline(bg_thresh, color='red', linestyle='--', label=f'Threshold={bg_thresh:.1f} dB')
        axes[0, 0].set_title(f'PSNR Histogram (All {total} slices)')
        axes[0, 0].set_xlabel('PSNR (dB)')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].legend()
        
        axes[0, 1].hist(psnr, bins=80, color='lightgray', edgecolor='white', alpha=0.8, label='All')
        axes[0, 1].hist(psnr_tissue, bins=80, color='darkorange', edgecolor='white', alpha=0.8, label='Tissue')
        axes[0, 1].axvline(bg_thresh, color='red', linestyle='--')
        axes[0, 1].set_title(f'PSNR Histogram (Tissue slices in orange)')
        axes[0, 1].set_xlabel('PSNR (dB)')
        axes[0, 1].set_ylabel('Count')
        axes[0, 1].legend()
        
        axes[1, 0].hist(ssim, bins=80, color='steelblue', edgecolor='white', alpha=0.8)
        axes[1, 0].set_title(f'SSIM Histogram (All {total} slices)')
        axes[1, 0].set_xlabel('SSIM')
        axes[1, 0].set_ylabel('Count')
        
        axes[1, 1].hist(ssim, bins=80, color='lightgray', edgecolor='white', alpha=0.8, label='All')
        axes[1, 1].hist(ssim_tissue, bins=80, color='darkorange', edgecolor='white', alpha=0.8, label='Tissue')
        axes[1, 1].set_title(f'SSIM Histogram (Tissue slices in orange)')
        axes[1, 1].set_xlabel('SSIM')
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].legend()

        plt.tight_layout()
        hist_path = os.path.join(args.output_dir, "histograms.png")
        plt.savefig(hist_path, dpi=150)
        plt.close()
        stats_text += f"\n直方图已保存至 {hist_path}\n"
        
        plt.figure(figsize=(8, 6))
        plt.scatter(psnr[~tissue_mask], ssim[~tissue_mask], c='lightgray', alpha=0.5, label='Background')
        plt.scatter(psnr_tissue, ssim_tissue, c='darkorange', alpha=0.5, label='Tissue')
        plt.xlabel('PSNR (dB)')
        plt.ylabel('SSIM')
        plt.title('PSNR vs SSIM (Test Set)')
        plt.legend()
        plt.grid(alpha=0.3)
        scatter_path = os.path.join(args.output_dir, "scatter.png")
        plt.savefig(scatter_path, dpi=150)
        plt.close()
        stats_text += f"散点图已保存至 {scatter_path}\n"
        
        psnr_sorted = np.sort(psnr_tissue)
        cdf = np.arange(1, len(psnr_sorted)+1) / len(psnr_sorted)
        plt.figure(figsize=(8, 4))
        plt.plot(psnr_sorted, cdf, color='darkorange')
        plt.xlabel('PSNR (dB)')
        plt.ylabel('Cumulative Probability')
        plt.title('CDF of PSNR (Tissue Slices)')
        plt.grid(alpha=0.3)
        cdf_path = os.path.join(args.output_dir, "psnr_cdf.png")
        plt.savefig(cdf_path, dpi=150)
        plt.close()
        stats_text += f"CDF 图已保存至 {cdf_path}\n"

    else:      
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].hist(psnr, bins=80, color='steelblue', edgecolor='white')
        axes[0].set_title(f'PSNR Histogram (All {total} slices)')
        axes[0].set_xlabel('PSNR (dB)')
        axes[0].set_ylabel('Count')
        axes[1].hist(ssim, bins=80, color='steelblue', edgecolor='white')
        axes[1].set_title(f'SSIM Histogram (All {total} slices)')
        axes[1].set_xlabel('SSIM')
        axes[1].set_ylabel('Count')
        plt.tight_layout()
        hist_path = os.path.join(args.output_dir, "histograms.png")
        plt.savefig(hist_path, dpi=150)
        plt.close()
        stats_text += f"直方图已保存至 {hist_path}\n"

        plt.figure(figsize=(8, 6))
        plt.scatter(psnr, ssim, alpha=0.5)
        plt.xlabel('PSNR (dB)')
        plt.ylabel('SSIM')
        plt.title('PSNR vs SSIM')
        plt.grid(alpha=0.3)
        scatter_path = os.path.join(args.output_dir, "scatter.png")
        plt.savefig(scatter_path, dpi=150)
        plt.close()
        stats_text += f"散点图已保存至 {scatter_path}\n"
    
    report_path = os.path.join(args.output_dir, "analysis_report.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(stats_text)

    print(stats_text)
    print(f"完整报告已保存至 {report_path}")

if __name__ == "__main__":
    main()
