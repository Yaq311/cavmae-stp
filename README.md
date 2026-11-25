# CAV-MAE Joint STP-8

KS50 强度 5 的全维 STP-8 基准。音视频各保留 3 个 768 维 prompt token，联合搜索 4608 维。每个损坏开始时独立初始化，批内共享 prompt，批间继承上一批选中的 prompt。

每批用原始视图的平均预测熵做 8 轮正负方向搜索，最后用最优 prompt 对弱增强视图做一次推理，两视图取概率均值。每批共 18 次模型前向。历史完整评估准确率为音频 71.2422%、视频 61.3139%。

## 代码

- `stp.py` 包含 prompt 初始化、搜索和双视图融合。
- `run_stp.py` 加载权重，运行评估并输出 JSON。
- `models/` 和 `dataloader.py` 沿用历史实验的 CAV-MAE 和数据加载代码。

## 运行

使用原实验的 `adapgc` 环境，依赖版本列在 `requirements.txt`。历史运行使用 PyTorch 2.9.1 和 CUDA 12.8。将 `--checkpoint` 和 `--manifest-root` 指向实际可读的位置。

```bash
python run_stp.py \
  --corruption-modality audio \
  --checkpoint /mnt/data2/yaq/cav_mae_ks50.pth \
  --manifest-root /mnt/data2/yaq/cavmae-accuracy-tta/metadata/ks50 \
  --output-json results/audio.json
```

视频评估用 `--corruption-modality video` 和 `--output-json results/video.json`。程序按 `<manifest-root>/<模态>/<损坏>/severity_5.json` 读取索引，默认读取根目录中的 `class_labels_indices_ks50.csv`。可用 `--label-csv` 指定其他位置。索引内的音频和视频路径也需可读。

快速检查时添加 `--corruptions gaussian_noise --max-batches 1 --num-workers 0`，并为检查结果指定独立的输出文件。

## 相似度继承

在上述命令中添加 `--method similarity_transfer`，并指定独立的输出文件。每个样本用冻结参考 prompt 提取融合特征，以 `softmax(cosine / 0.07)` 加权上一批的最终 prompt，然后按自己的预测熵做 8 轮搜索。首批逐样本初始化。每种损坏开始时重置历史。

参考特征额外占 1 次前向，因此相似度继承每批共 19 次前向。默认方法仍是 `stp8_ensemble`。
