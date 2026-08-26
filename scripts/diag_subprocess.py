"""诊断：从 QGIS Python 子进程调用 ai_env python 的 import 行为。"""

import subprocess
import sys

py = r"D:\DevKit\anaconda3\envs\ai_env\python.exe"
for code in ["import sys; print(sys.executable)", "import ultralytics; print('ULTRA_OK', ultralytics.__version__)"]:
    r = subprocess.run(
        [py, "-c", code],
        capture_output=True, text=True, timeout=120,
        creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
    )
    print("CMD:", code[:40])
    print("  rc:", r.returncode)
    print("  out:", r.stdout.strip()[:200])
    print("  err:", r.stderr.strip()[-400:])
