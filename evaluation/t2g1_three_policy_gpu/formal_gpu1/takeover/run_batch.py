"""仅在新GPU1 Gate完整复核后执行原样60臂主矩阵，任何错误立即停止。"""
import json
import subprocess
import sys
from pathlib import Path

OUT=Path(__file__).resolve().parents[1]
def step(script, *args):
    subprocess.run([sys.executable,'-B',str(script),*args],check=True)

assert json.loads((OUT/'migration_cpu_precheck.json').read_text())['status']=='READY'
assert json.loads((OUT/'container_cpu_precheck.json').read_text())['returncode']==0
step(OUT/'launch_formal.py','gate')
step(OUT/'verify_formal.py','gate')
step(OUT/'takeover/verify_evidence.py','gate')
step(OUT/'launch_formal.py','main')
step(OUT/'verify_formal.py','main')
step(OUT/'takeover/verify_evidence.py','main')
step(OUT/'analyze_formal.py')
