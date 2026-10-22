"""重新调用已冻结D4a CPU回归，临时夹具及结果全部写入本轮目录。"""
import json,unittest
from pathlib import Path
from evaluation.exact_greedy_paired_gpu_fix_b5d4a import test_regression as tests
P=Path(__file__).resolve().parent
def main():
 tests.P=P;tests.ROWS.clear()
 result=unittest.TextTestRunner(verbosity=2,resultclass=tests.Recorder).run(unittest.defaultTestLoader.loadTestsFromTestCase(tests.Regression))
 (P/'cpu_regression_results.json').write_text(json.dumps({'status':'PASS' if result.wasSuccessful() else 'FAIL','tests_run':result.testsRun,'gpu_started':False,'rows':tests.ROWS,'implementation':'原D4a测试与修复代码；仅临时输出路径重定向本目录'},ensure_ascii=False,indent=2)+'\n')
 raise SystemExit(0 if result.wasSuccessful() else 1)
if __name__=='__main__':main()
