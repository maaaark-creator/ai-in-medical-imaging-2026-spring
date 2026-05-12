from __future__ import annotations

import re
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PRE_DIR = SCRIPT_DIR.parent
HTML = PRE_DIR / "presentation" / "task2_presentation.html"


NOTES = [
    (
        "开场先把 Task 2 定义成一个完整 reconstruction workflow，而不是“堆模型”。主线是：从 fully sampled T2 构造 R=5 undersampled input，再训练 image-domain 网络恢复 target。",
        "Task 2 的目标是从 R=5 欠采样 T2 图像恢复 fully sampled T2。我们完成的不是单个网络，而是一条完整流程：数据模拟、预处理、patient-level split、模型训练、PSNR/SSIM 评价和可视化分析。后面的模型比较都围绕一个问题展开：line-mask aliasing 更适合直接生成，还是更适合做 artifact correction。",
    ),
    (
        "强调数据严谨性：patient-level split 防止 leakage；p99 shared normalization 防止 input/target 各自重标定；tissue slice 是主指标，避免背景 slice 虚高。",
        "我们采用 patient-level split，训练、验证、测试病人分别是 437、62、126，避免同一病人的相邻 slice 同时出现在训练和测试里。归一化没有用每张 slice 单独 min-max，而是用 fully sampled target volume 的非零 p99 作为共享 scale，让 input 和 target 在同一强度坐标下比较。主结果报告 tissue slices，因为大量背景 slice 会让指标不真实地变高。",
    ),
    (
        "把 point-wise 和 line mask 讲成两个 artifact regimes，不说“做错了”。Point-wise 是额外探索，line mask 是主结论，因为更贴近 Cartesian MRI。",
        "我们研究了两个欠采样 regime。Point-wise random mask 更像随机纹理污染，可以帮助验证 pipeline；vertical-line mask 更接近 Cartesian MRI 中 phase-encoding line 的缺失，因此会产生方向性 aliasing。我们没有把两个 regime 的数值硬放在一起比较，而是把 point-wise 作为额外探索，把 vertical-line R=5 作为最终 Task 2 主实验。",
    ),
    (
        "主结果提前亮出，并且现在按老师要求补齐 mean、median、variance、MAE、RMSE。主表只放最关键的 mean，完整统计放下一页。",
        "这是 line-mask R=5 下、统一 nonzero/tissue-slice 口径的主结果。所有模型都使用 target nonzero fraction >= 0.001 的 17,340 张 tissue slices 计算。2D U-Net 已经把 PSNR mean 从 28.54 提升到 36.36。2.5D U-Net bf64 比 bf32 略好，但仍未超过 2D。最终 residual ResNet 不仅 PSNR/SSIM 最高，MAE 和 RMSE 也是最低，说明它在像素误差和结构相似性上都最稳。",
    ),
    (
        "这一页回应老师的指标要求：PSNR/SSIM 的均值、中位数、方差，以及 MAE/RMSE 都集中展示；右侧用分布图说明不是只看平均值。",
        "这里是完整扩展统计表。PSNR 和 SSIM 都给出 mean、median 和 variance；MAE 和 RMSE 也用同一批 nonzero slices 计算。右侧分布图帮助我们判断模型是否只是在少数样例上好，还是整体分布都改善。可以看到 ResNet 在 PSNR/SSIM 上最好，同时 MAE/RMSE 最低，因此主结论不只依赖单个指标。",
    ),
    (
        "2D U-Net 是强 baseline，作用是建立直接 image-to-image synthesis 能做到什么程度。",
        "第一个模型是 2D U-Net baseline。它非常适合 image-to-image regression：encoder 提取多尺度 aliasing pattern，decoder 恢复空间分辨率，skip connections 保留边缘和局部解剖结构。它的 feature widths 是 64、128、256、512，最终 tissue PSNR 达到 36.36 dB，说明即使不使用 slice context，image-domain baseline 已经有很强的重建能力。",
    ),
    (
        "这页重点是回答公平性问题。bf64 证明容量有影响，但提升有限，所以“2.5D context 不是自动胜利”。",
        "2.5D U-Net 的动机是利用 BraTS volume 的 slice 间连续性。输入从单张 slice 变成 z-1、z、z+1 三个通道，输出中心 slice。之前 bf32 版本和 2D U-Net 容量不完全公平，所以我们补跑了 base_features=64。结果 bf64 从 35.38 提升到 35.79 dB，validation MSE 也下降，说明容量确实有用。但它仍没有超过 2D U-Net，因此我们不能简单说 2.5D context 一定更好。",
    ),
    (
        "解释“奇怪”的 all-slice PSNR：tissue 变好，blank/background 变差，所以 all-slice 下降。用它支持 tissue-slice 主报告。",
        "bf64 的结果一开始看起来奇怪，因为 all-slice PSNR 反而比 bf32 低。但拆开看会发现，tissue-slice PSNR 是提升的，validation MSE 也是更好的。all-slice 下降主要来自 blank 或 near-blank slices：这些切片本来几乎全黑，模型只要生成一点背景纹理，PSNR 就会被严重惩罚。因此这个现象反而说明，我们把 tissue-slice metrics 作为主比较是合理的。",
    ),
    (
        "ResNet 是主模型。核心不是更深，而是 formulation 更对：预测 residual，再加回 center input。",
        "Residual ResNet 的关键想法是把 reconstruction 建模成 artifact correction，而不是从零生成整张图。欠采样输入已经包含大部分解剖结构，网络只需要学习缺失和伪影对应的 residual。我们使用 3-slice input、base channels 64、12 个 residual blocks，并通过 dilation 扩大感受野。这个设计和 line-mask 的长程方向性 aliasing 更匹配，因此取得了最好的 38.70 dB tissue PSNR。",
    ),
    (
        "DenseNet 只讲设计和 pending，不编结果。强调它是 feature reuse baseline，不是多模态或 data consistency。",
        "DenseNet 是我们后续加入的 feature-reuse baseline。它使用 single-channel undersampled T2 输入，经过初始 3x3 convolution、4 个 dense blocks 和 transition layers。Dense connectivity 的优点是复用浅层和深层特征，可能对细节恢复有帮助。不过它当前还没有完成同一 evaluation policy 下的最终 line-mask metrics，所以这页只作为 ongoing ablation，不用于支撑主结论。",
    ),
    (
        "定性图讲 ResNet 到底修了什么：减少条纹和 aliasing，保留脑结构；误差集中在边缘和纹理。",
        "从可视化上看，输入图像中有明显的 structured aliasing。ResNet 重建后，背景条纹和组织内伪影明显减少，脑组织轮廓更接近 ground truth。绝对误差图也显示，残余误差主要集中在组织边缘和高频纹理区域。这说明模型已经能恢复主要结构，但细节和边界仍然是难点。",
    ),
    (
        "Point-wise 作为额外实验：证明 pipeline 可学，也展示 artifact regime 会影响模型排序。",
        "这里展示的是 point-wise random mask 的额外实验。这个 regime 下网络也能明显学习重建，而且 2.5D U-Net 的表现非常强。但我们不把它和 line-mask 主表直接混合，因为两者的 artifact statistics 不同。它的价值在于说明：模型表现不仅取决于网络名字，还取决于欠采样方式产生的物理退化模式。",
    ),
    (
        "解释为什么两个 mask 会带来不同故事：point-wise 像 denoising，line-mask 像 de-aliasing。",
        "Point-wise mask 的缺失更随机，伪影更像噪声或纹理污染，网络行为更接近 denoising。Vertical-line mask 是整条 k-space line 缺失，会产生方向性和结构性的 aliasing，更接近 Cartesian MRI reconstruction。这个差异解释了为什么同一个模型在不同 regime 下排序可能不同，也说明我们最终把 line-mask 作为主结论更符合 MRI 物理背景。",
    ),
    (
        "四个模型的意义：2D baseline、2.5D context、bf64 capacity ablation、ResNet formulation、DenseNet pending feature reuse。",
        "把这些模型放在一起看，它们回答的是不同问题。2D U-Net 告诉我们 direct synthesis baseline 有多强；2.5D U-Net 测试 slice context 是否有帮助；bf64 版本进一步检查容量是不是限制因素；Residual ResNet 检查 residual correction 是否更符合任务；DenseNet 则是 feature reuse 的后续 baseline。最终最强证据指向的是 residual formulation，而不是单纯增加输入 context。",
    ),
    (
        "Worst-case 分析是防守页：说明我们知道失败在哪里，并能提出下一步。",
        "最差样例分析可以帮助我们理解模型边界。即使是低 PSNR 的样例，ResNet 通常仍能减少主要 aliasing，但在组织边缘、高频纹理或局部结构复杂的位置仍有残差。这也自然引出后续改进：可以加入 data consistency、residual U-Net、tissue-weighted loss，或者对困难 slice 做更系统的分析。",
    ),
    (
        "主动讲限制，显得成熟：bf64 减少容量质疑，但还不能证明所有变量都公平；没有 DC；DenseNet 未完成。",
        "我们也不想过度解读结果。bf64 ablation 回答了一部分容量问题，但不同模型仍然不是完全控制变量的比较。DenseNet 还没有最终同策略测试结果。Point-wise 和 line-mask 也不能直接比较绝对数值。更重要的是，当前 Task 2 仍是 image-domain reconstruction，没有显式 k-space data consistency。因此我们的结论限定在当前 line-mask R=5、当前训练策略下：residual correction 是最有效的建模方式。",
    ),
    (
        "Next steps 要接住已有发现：bf64 已完成；下一步不是再盲目加宽，而是 residual U-Net、tissue loss、DC layer、DenseNet same-policy eval。",
        "下一步可以分为 Task 2 ablation 和 Task 3 extension。Task 2 里，bf64 已经作为 capacity ablation 完成，接下来更有价值的是 residual U-Net，用来区分 residual learning 和 ResNet backbone 的贡献；也可以尝试 tissue-weighted loss 或 nonzero-only training。Task 3 方向则是加入 k-space data consistency，并使用 T1 结构信息作为 guidance。",
    ),
    (
        "答辩问答页：特别准备 2.5D 为什么没赢、ResNet 为什么赢、为什么 tissue-slice、为什么保留 point-wise。",
        "如果被问为什么不说 2.5D 更好，我会回答：bf64 确实改善了 bf32，但仍没有超过 2D U-Net，所以 context alone 不足够。如果被问 ResNet 为什么最好，我会说 aliased input 已经有主要解剖结构，预测 residual correction 比生成整张图更稳定。如果被问为什么用 tissue-slice metrics，我会说背景 slice 会显著影响 all-slice PSNR，tissue slice 更贴近重建目标。",
    ),
    (
        "结论收束到一句：under Cartesian line undersampling, residual artifact correction wins。",
        "总结来说，我们完成了 Task 2 的完整重建 pipeline，并在 vertical-line R=5 设置下比较了 2D U-Net、2.5D U-Net、bf64 capacity ablation、Residual ResNet，以及仍在进行的 DenseNet baseline。结果显示，单纯增加 2.5D context 有帮助但不够；在 Cartesian line undersampling 下，把任务建模为 residual artifact correction 更有效。",
    ),
    (
        "AI 声明页按课程要求保留。正式提交前可按小组实际情况改措辞。",
        "按照课程要求，我们声明 AI assistance 主要用于汇报结构整理、语言润色、图表规划和 HTML slide 生成。实验代码、训练输出、指标和可视化结果都来自我们的项目文件和实际 evaluation outputs。正式提交前，我们会根据课程模板调整这段声明。",
    ),
    (
        "Backup 用来回答 mask 追问：两个 regime 是实验视角，不是错误和修正。",
        "如果老师继续问 mask 的问题，我会强调：point-wise 和 vertical-line 是两个不同 artifact regimes。Point-wise 帮助我们验证 pipeline 和探索模型行为；vertical-line 更符合 Cartesian MRI，因此作为最终 Task 2 主结论。不同 regime 不冲突，反而帮助我们理解模型为什么会在不同退化模式下表现不同。",
    ),
]


def build_note(simple: str, script: str) -> str:
    return (
        '<aside class="notes">'
        '<div class="note-block"><strong>简单思路：</strong>'
        f"{simple}"
        '</div><div class="note-block"><strong>草拟演讲稿：</strong>'
        f"{script}"
        "</div></aside>"
    )


def main() -> None:
    text = HTML.read_text(encoding="utf-8")
    text = text.replace(";`r`n      max-height", ";\n      max-height")
    text = text.replace(";`r`n      overflow-y", ";\n      overflow-y")
    text = text.replace(" }`r`n    .note-block", " }\n    .note-block")
    text = text.replace('<div class="arrow">鈫?/div>', '<div class="arrow">&rarr;</div>')
    text = re.sub(
        r'<div class="controls">.*?</div>',
        '<div class="controls">&larr; / &rarr; or Space: navigate &middot; N: notes &middot; F: fullscreen &middot; P: print</div>',
        text,
        count=1,
        flags=re.S,
    )

    note_pattern = r'<aside\b[^>]*>.*?(?:</aside>|\?/aside>)'
    matches = re.findall(note_pattern, text, flags=re.S)
    if len(matches) != len(NOTES):
        raise RuntimeError(
            f"Expected {len(NOTES)} notes, found {len(matches)} in {HTML}. "
            f"Raw '<aside' count: {text.count('<aside')}"
        )

    iterator = iter(build_note(simple, script) for simple, script in NOTES)
    text = re.sub(note_pattern, lambda _m: next(iterator), text, flags=re.S)

    if ".note-block + .note-block" not in text:
        text = text.replace(
            "body.show-notes .slide.active .notes { display: block; }",
            "body.show-notes .slide.active .notes { display: block; }\n"
            "    .note-block + .note-block { margin-top: 10px; }\n"
            "    .note-block strong { color: #d9f99d; }",
        )
    HTML.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
