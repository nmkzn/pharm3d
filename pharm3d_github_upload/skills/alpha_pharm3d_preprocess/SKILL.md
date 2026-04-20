---
name: alpha_pharm3d_preprocess
description: 复现论文中 MLP 前的数据预处理流程，只分析和执行前处理，不启动完整训练。
---

# Alpha Pharm3D Preprocess

你的任务是复现论文《Enhancing multifunctional drug screening via artificial intelligence》中 MLP 前的数据预处理。

## 工作边界
MLP 前处理完成的标志是已经明确定位或成功生成以下中间产物：
- confs.sdf
- box_info.txt
- 5A_dist_info.npy
- tensor.pkl

## 项目根目录
- E:\pharm3d

## 论文文件
- E:\pharm3d\paper\D5DD00082C.pdf

## 优先分析的文件
- E:\pharm3d\README.md
- E:\pharm3d\app.py
- E:\pharm3d\pharm\embed_script.py
- E:\pharm3d\pharm\embed\pocket.py
- E:\pharm3d\pharm\embed\slicedMulti.py
- E:\pharm3d\pharm\embed\utils.py
- E:\pharm3d\pharm\embed\train.py

## 必查关键词
- tensor.pkl
- confs.sdf
- box_info.txt
- 5A_dist_info.npy
- split_info.json
- train_idx.npy
- val_idx.npy
- EmbedMultipleConfs
- MMFFOptimizeMolecule
- Open3DALIGN
- ChemicalFeatures
- BuildFeatureFactory
- ravel_multi_index

## 输出格式
每轮必须输出：
1. 当前定位到的步骤名
2. 对应脚本与函数
3. 输入文件
4. 输出文件
5. 是否属于 MLP 前处理
6. 运行命令
7. 预期输出
8. 当前阻塞点
9. 下一条建议命令

## 执行限制
- 未经允许，不要启动完整训练
- 优先读文件、搜索代码、运行最小闭环命令
- 所有运行日志写到 E:\pharm3d\repro_logs
- 修改代码前先解释目的，修改后给出完整文件内容
- 优先先搜索，不要一上来无限 exec
