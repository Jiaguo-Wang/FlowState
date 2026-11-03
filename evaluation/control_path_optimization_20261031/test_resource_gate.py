"""收尾历史采样与真实占用分开验证，启动前仍要求零活动。"""
import unittest
from unittest.mock import patch
from . import execute_v2 as launcher


class ResourceGateTests(unittest.TestCase):
    def query(self, utilization=4, memory=14, process=''):
        def read(args):
            if '--query-gpu=uuid,memory.used,utilization.gpu' in args:
                return f'{launcher.GPU}, {memory}, {utilization}'
            if '--query-compute-apps=gpu_uuid,pid,process_name' in args:
                return process
            return ''
        return patch.object(launcher, 'readcmd', side_effect=read)

    def test_recent_sample_during_cleanup(self):
        with self.query():
            launcher.resource_gate(cleanup=True)
            with self.assertRaises(AssertionError):
                launcher.resource_gate()

    def test_memory_occupancy_rejects(self):
        with self.query(memory=1500):
            with self.assertRaises(AssertionError):
                launcher.resource_gate(cleanup=True)

    def test_active_process_rejects(self):
        with self.query(process=launcher.GPU+', 123, compute'):
            with self.assertRaises(AssertionError):
                launcher.resource_gate(cleanup=True)

    def test_clean_launch(self):
        with self.query(utilization=0):
            launcher.resource_gate()


if __name__ == '__main__':
    unittest.main()
