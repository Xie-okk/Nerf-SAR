# NeRF-main 运行命令速查

默认已经进入正确 conda 环境，并在项目根目录运行：

```bash
cd /home/lpy/AICode/XZC/Nerf-main
```

Windows 同理，先进入项目目录：

```powershell
cd D:\A_master\AAA组会\AAAA研究方向\网络重建\Nerf-main
```

如果环境已经激活，命令里可以直接用：

```bash
python isar_runner.py ...
```

如果没有激活环境，就把 `python` 换成完整解释器路径，例如：

```bash
/home/lpy/.conda/envs/neus_py38/bin/python isar_runner.py ...
```

```powershell
D:\Anaconda3\anaconda\envs\neus\python.exe isar_runner.py ...
```

默认配置文件：`./confs/isar_fuyan.conf`  
默认目标：`1999JV6`

---

## 1. 从头训练

```bash
python isar_runner.py --conf ./confs/isar_fuyan.conf --mode train --case 1999JV6
```

从头训练不要加 `--is_continue`，也不要加 `--checkpoint`。

输出主要在：

```text
exp/1999JV6/checkpoints/ckpt_XXXXXX.pth
exp/1999JV6/validations/iter_XXXXXX_frames_*.png
exp/1999JV6/density_z_planes/density_z0_XXXXXX.png
exp/1999JV6/meshes/mesh_XXXXXX.ply
exp/1999JV6/logs/train_metrics.csv
exp/1999JV6/logs/loss_curves.png
```

---

## 2. 继续训练

自动读取最新 checkpoint：

```bash
python isar_runner.py --conf ./confs/isar_fuyan.conf --mode train --case 1999JV6 --is_continue
```

读取指定 checkpoint：

```bash
python isar_runner.py --conf ./confs/isar_fuyan.conf --mode train --case 1999JV6 --checkpoint ckpt_020000.pth
```

也可以给完整路径：

```bash
python isar_runner.py --conf ./confs/isar_fuyan.conf --mode train --case 1999JV6 --checkpoint /home/lpy/AICode/XZC/Nerf-main/exp/1999JV6/checkpoints/ckpt_020000.pth
```

注意：继续训练时，`train.end_iter` 必须大于 checkpoint 里的 `iter_step`。

---

## 3. 验证图像

自动读取最新 checkpoint，验证所有视角：

```bash
python isar_runner.py --conf ./confs/isar_fuyan.conf --mode validate_views --case 1999JV6 --is_continue --view_ids all
```

只验证指定视角：

```bash
python isar_runner.py --conf ./confs/isar_fuyan.conf --mode validate_views --case 1999JV6 --is_continue --view_ids 0,6,12
```

读取指定 checkpoint 验证：

```bash
python isar_runner.py --conf ./confs/isar_fuyan.conf --mode validate_views --case 1999JV6 --checkpoint ckpt_020000.pth --view_ids all
```

输出位置：

```text
exp/1999JV6/validations/iter_XXXXXX_frames_all.png
exp/1999JV6/validations/iter_XXXXXX_frames_0_6_12.png
```

---

## 4. 导出 Mesh

自动读取最新 checkpoint：

```bash
python isar_runner.py --conf ./confs/isar_fuyan.conf --mode validate_mesh --case 1999JV6 --is_continue
```

指定 checkpoint：

```bash
python isar_runner.py --conf ./confs/isar_fuyan.conf --mode validate_mesh --case 1999JV6 --checkpoint ckpt_020000.pth
```

临时指定分辨率和阈值：

```bash
python isar_runner.py --conf ./confs/isar_fuyan.conf --mode validate_mesh --case 1999JV6 --checkpoint ckpt_020000.pth --mesh_resolution 128 --mcube_threshold 0.03
```

输出位置：

```text
exp/1999JV6/meshes/mesh_XXXXXX.ply
```

NeRF 当前 mesh 阈值是相对阈值：

```text
value = min(alpha * mean_LOS(sigma)) + (max - min) * threshold_ratio
```

注意：当前代码里 `--mcube_threshold 0.0` 表示使用配置文件默认值，不一定是真 0。想接近 0 时用：

```bash
--mcube_threshold 0.001
```

---

## 5. 扫一组 Mesh 阈值

Linux：

```bash
cd /home/lpy/AICode/XZC/Nerf-main

CKPT=ckpt_020000.pth

for t in 0.001 0.01 0.03 0.05 0.10 0.20 0.30 0.40; do
    python isar_runner.py \
        --conf ./confs/isar_fuyan.conf \
        --mode validate_mesh \
        --case 1999JV6 \
        --checkpoint $CKPT \
        --mesh_resolution 128 \
        --mcube_threshold $t

    tag=$(printf "%04d" $(python -c "print(int(float('$t') * 1000))"))
    cp ./exp/1999JV6/meshes/mesh_020000.ply ./exp/1999JV6/meshes/mesh_020000_thr${tag}.ply
done
```

Windows PowerShell：

```powershell
cd D:\A_master\AAA组会\AAAA研究方向\网络重建\Nerf-main

$ckpt = "ckpt_020000.pth"
$thresholds = 0.001, 0.01, 0.03, 0.05, 0.10, 0.20, 0.30, 0.40

foreach ($t in $thresholds) {
    python isar_runner.py `
        --conf .\confs\isar_fuyan.conf `
        --mode validate_mesh `
        --case 1999JV6 `
        --checkpoint $ckpt `
        --mesh_resolution 128 `
        --mcube_threshold $t

    $tag = "{0:D4}" -f [int]($t * 1000)
    Copy-Item .\exp\1999JV6\meshes\mesh_020000.ply ".\exp\1999JV6\meshes\mesh_020000_thr$tag.ply"
}
```

---

## 6. 常用配置项

```hocon
train {
    learning_rate = 5e-4
    end_iter = 20000
    image_loss_type = "mse"
}

validate {
    mesh_resolution = 128
    nerf_density_threshold = 0.03
    view_ids = "all"
    final_view_ids = "all"
    n_height = 64
}

model {
    type = "nerf"
}
```

阈值经验：散点多就增大 `nerf_density_threshold`；主体缺失就减小。最终出细 mesh 时可以把 `mesh_resolution` 从 `128` 提到 `256`。