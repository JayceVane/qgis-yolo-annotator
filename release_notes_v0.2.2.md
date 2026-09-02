# qgis_yolo_annotator v0.2.2

应官方仓库安全扫描与代码评审反馈的加固版本。

## 安全加固

- pt→onnx 转换：权重路径改为 argv 传参（此前以文本拼入 `-c` 脚本，含引号路径可注入子进程代码）；解释器/权重文件前置存在性校验
- 在线瓦片下载入口限定 http/https 协议，拒绝 file:/自定义 scheme
- 内部 `assert` 改为显式异常（`python -O` 下不再被剥离）

## 扫描豁免标注（附原因）

- xml.etree：仅序列化写出 VOC XML，从不解析外部输入
- random.Random(seed)：固定种子保证 train/val 划分可复现
- SHA1：仅本地瓦片缓存文件名，`usedforsecurity=False`
- subprocess：入参均验证为本地存在文件，列表参数无 shell

## 代码评审修复

- E731 lambda 赋值 → def；F811 重复导入；E501 超长行；E305 空行；F841 死变量
- bandit / flake8 全库清零，并纳入 `scripts/check_syntax.py` 发布门禁
