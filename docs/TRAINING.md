# FeTARefine：独立重训与报告整理工程

当前入口按 **新实验** 使用原始数据、现有PP1、YAML配置，从头训练M0–M5及对照。`result/`保留的是历史结果，不是当前重构代码已运行的证明。当前版本通过合成CPU检查；完整GPU训练与医学结果尚未产生。

## 目录和运行边界

- `data/raw/`、`data/preprocessed/`、`result/`只读使用，不覆盖。
- 新训练写入`runs/`，新预测/连续输出写入`outputs/`，新评价写入`reports/`。
- `./run_local.sh`把Torch、Matplotlib、临时文件和缓存固定在本目录的`.runtime/`；使用当前Python解释器，未新建或修改全局环境。需要独立环境时在本目录创建`.venv`并由该解释器运行。
- 所有CLI输入/输出路径必须位于项目内，输出目录必须是新的。未实现断点续训，不得将未完成实验当作COMPLETE。大型任务的恢复、GPU吞吐和显存峰值仍需服务器验证。
- 修改前源码在`audits/retrain_repair/before/`；检查日志在同级审计目录。无需访问旧FeTA工程即可运行新流程。

## 方法与协议

M0为SwinUNETR；M1/M2为纯2D/相邻三切片DeepLabV3+；M3为M0软先验引导残差；M4为M0/M2八类raw logits等权平均。BR为35通道二维有界残差，最终M5为BR之后的17通道三维选择性门控。无界残差只取消tanh约束；Diff-UNet是独立架构对照。

基线开发：56例训练、8例完整体积原空间验证，以病例等权、reference-present前景Dice选轮。之后按SELECTION.json选定轮数，在64例上**重新初始化**训练。配置和开发源码必须匹配；不是直接续训best权重。

- M0开发上限300轮，100轮后允许早停，每5轮验证，耐心10次。
- M1/M2/M3开发上限120轮，每轮验证，耐心20次。
- Diff-UNet开发上限500轮，warmup50，每5轮验证，100轮后允许早停，耐心20次。
- 严格最高Dice保存best；min_delta=0.001仅用于早停耐心。
- BR/无界：64例×64样本÷batch4×30轮=30,720更新。
- 门控：64例×16样本÷batch2×30轮=15,360更新。
- BR/门控固定末轮，不验证选模，固定学习率。门控必须读取真实BR输出，不允许M4回退。

这是清理后的新实现，不承诺逐位复现历史实验：当前YAML明确FP32；二维采样为有效区域中心裁剪与联合翻转，未声称复刻历史旋转/缩放增强；学习率和采样随机数以当前代码/CONFIG/IDENTITY为准。不同实验必须按新协议统一报告，不能把历史指标标成新代码实测。

## 从头训练顺序

所有命令在项目根执行。正式运行前加`--preflight-only`进行输入与资源检查；预检不加载训练模型、不训练。正式GPU任务建议先前台预检，再在tmux运行相同命令。

### 1. 单模型开发与final64

```bash
./run_local.sh train.py --model m0 --phase phase1 --output_dir runs/m0_dev --preflight-only
./run_local.sh train.py --model m0 --phase phase1 --output_dir runs/m0_dev
./run_local.sh train.py --model m0 --phase phase2 --selection runs/m0_dev/SELECTION.json --output_dir runs/m0_final
```

M1、M2、Diff-UNet同样分别执行phase1/phase2，使用对应`--model m1`、`m2`、`diffunet`。默认YAML会自动读取；`--config`可指定项目内的新配置。开发和final必须使用同一配置。ImageNet初始化首次可能下载至`.runtime/torch/`，需要网络或预置权重；本轮未下载。

### 2. M3先验

```bash
./run_local.sh infer.py --model m0 --checkpoint runs/m0_dev/best.pt --group all80 --output_dir outputs/m0_dev_prior
./run_local.sh train.py --model m3 --phase phase1 --priors_dir outputs/m0_dev_prior --output_dir runs/m3_dev
./run_local.sh infer.py --model m0 --checkpoint runs/m0_final/final.pt --group all80 --output_dir outputs/m0_final
./run_local.sh train.py --model m3 --phase phase2 --selection runs/m3_dev/SELECTION.json --priors_dir outputs/m0_final --output_dir runs/m3_final
```

开发M3使用开发M0先验，final64 M3使用final64 M0先验，父权重身份分别记录。这是系统比较，不宣称严格单因素或OOF。`all80`推理只读取图像，不读取GT。

### 3. M4、BR及无界残差

```bash
./run_local.sh infer.py --model m2 --checkpoint runs/m2_final/final.pt --group all80 --output_dir outputs/m2_final
./run_local.sh infer.py --model m4 --expert_3d_dir outputs/m0_final --expert_2p5d_dir outputs/m2_final --output_dir outputs/m4_test
./run_local.sh train.py --model m5_br --phase phase2 --expert_3d_dir outputs/m0_final --expert_2p5d_dir outputs/m2_final --output_dir runs/br
./run_local.sh infer.py --model m5_br --checkpoint runs/br/final.pt --group all80 --expert_3d_dir outputs/m0_final --expert_2p5d_dir outputs/m2_final --output_dir outputs/br
./run_local.sh train.py --model unbounded --phase phase2 --expert_3d_dir outputs/m0_final --expert_2p5d_dir outputs/m2_final --output_dir runs/unbounded
```

每个缓存包含FP32连续值、病例/空间信息、模型与父权重SHA和数组SHA。BR全XY切片推理；门控使用2体素halo分块、裁去halo拼接连续输出。

### 4. 最终M5门控与评价

```bash
./run_local.sh train.py --model m5_gate --phase phase2 --expert_3d_dir outputs/m0_final --expert_2p5d_dir outputs/m2_final --lr_dir outputs/br --output_dir runs/gate
./run_local.sh infer.py --model m5 --checkpoint runs/gate/final.pt --expert_3d_dir outputs/m0_final --expert_2p5d_dir outputs/m2_final --lr_dir outputs/br --output_dir outputs/m5_test
./run_local.sh evaluate.py --pred_dir outputs/m5_test --output_dir reports/m5_test
```

其他模型同样先infer再evaluate。infer默认输出16例测试集的原空间`*_pred.nii.gz`及PP1连续数组；不接受未经预处理的原图冒充PP1。现有PP1通过`derived_affine`恢复至`original_affine`。评价默认native，禁止重复逆映射；需要评价PP1 NIfTI时明确指定`--prediction_grid pp1`。

统一输出六指标、16例逐病例和112例×组织行。表面定义为26邻域边界点、world-affine物理坐标、双向合并HD95/ASSD/NSD@1mm。缺失预测立即报错；距离失败显式记录，汇总不静默删除失败项。

评测在16例独立测试集上进行；各项指标均在患者原始物理空间下严格对齐金标准。

## 检查与资源

```bash
./run_local.sh tests/test_retraining.py
```

CPU worker按affinity/cgroup CPU、可用内存、共享内存估算，默认最多2个worker，预取1批、每worker单线程；病例缓存有界。固定阶段病例内连续采样，控制体积重复加载。保存实际资源、种子、配置、源码和输入哈希，逐轮记录数据等待与计算时间。GPU实际吞吐及资源余量需在目标机器前台验证，不根据CPU合成检查宣称GPU全链可用。
