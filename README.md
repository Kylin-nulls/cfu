# 菌落计数工具

一个本地 Streamlit 小工具，用 OpenCV 自动定位培养皿、分割乳白/淡黄菌落，并导出 CFU 结果、标注图和 CSV 明细。

## 运行

```bash
pip install -r requirements.txt
streamlit run app.py
```

如果使用当前工作区里已有虚拟环境：

```bash
path/to/venv/bin/streamlit run app.py
```

## 桌面启动器

开发环境里也可以用桌面启动器方式运行：

```bash
python desktop_launcher.py
```

它会自动寻找可用端口，启动本地 Streamlit 服务，并打开浏览器。

## 打包 Windows 软件

Windows 上安装 Python 3.11 或 3.12 后，在项目根目录双击：

```bat
build_windows.bat
```

脚本会创建 `.venv`、安装依赖，并用 PyInstaller 生成：

```text
dist\ColonyCounter\ColonyCounter.exe
```

分发时请压缩整个 `dist\ColonyCounter` 文件夹，不要只复制单个 exe。用户解压后双击 `ColonyCounter.exe` 即可运行；程序会在本机启动服务并自动打开浏览器。

日志位置：

```text
%LOCALAPPDATA%\ColonyCounter\launcher.log
```

## 命令行处理

```bash
python colony_counter.py samples/sample_plate.jpg \
  --overlay outputs/overlay.png \
  --csv outputs/colonies.csv \
  --json outputs/summary.json
```

## 调参提示

- 计数偏低：降低 `最小面积`，降低 `颜色阈值`，或提高 `局部亮点增强`。
- 相邻菌落被合并：降低 `近邻连接`，调整 `分水岭核心阈值`。
- 满盘近景中心漏检：默认已按连通区域分别做分水岭；命令行可调 `--watershed-core-max-distance` 控制小菌落核心点保护。
- 误检偏多：提高 `最小面积`，提高 `圆度过滤`，或降低 `计数区域比例`。
- 密集粘连样本：打开 `大面积粘连补偿` 后，用 `平均单菌落面积` 控制补偿强度。
