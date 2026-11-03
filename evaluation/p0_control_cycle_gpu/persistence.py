"""控制结果无论成功与否先落盘；没有响应时保留请求与原始异常。（G1 同款复制）"""
import json, os, socket, time, traceback
from pathlib import Path


def save_new(path, value):
    path = Path(path)
    with path.open('x') as f:
        json.dump(value, f, ensure_ascii=False, indent=2); f.flush(); os.fsync(f.fileno())


class Journal:
    """每个阶段单独文件，禁止覆盖既有Epoch、Plan或失败记录。"""
    def __init__(self, directory):
        self.directory = Path(directory); self.directory.mkdir(exist_ok=False)
        self.stage = 'start'; self.index = 0

    def record(self, stage, result):
        self.stage = stage
        save_new(self.directory / f'{self.index:03d}_{stage}.json', result)
        self.index += 1

    def fail(self, result, exc):
        record = dict(result, failed_stage=result.get('phase', self.stage), error=repr(exc),
                      traceback=traceback.format_exc())
        self.record('failure', record)
        return record


def call_saved(port, timeout, request, directory, connect=socket.create_connection):
    """与原ControlClient线协议相同，但在判断ok之前持久化原始响应。"""
    directory = Path(directory); directory.mkdir(exist_ok=False)
    save_new(directory / 'request.json', request)
    try:
        with connect(('127.0.0.1', port), timeout=timeout) as sock:
            sock.settimeout(timeout)
            sock.sendall((json.dumps(request) + '\n').encode())
            line = sock.makefile('rb').readline()
        with (directory / 'response.raw').open('xb') as f:
            f.write(line); f.flush(); os.fsync(f.fileno())
        if not line:
            raise RuntimeError('空控制响应')
        response = json.loads(line.decode())
        save_new(directory / 'response.json', response)
        if not response.get('ok'):
            raise RuntimeError('控制失败；完整响应已保存：' + str(directory / 'response.json'))
        return response
    except BaseException as exc:
        save_new(directory / 'exception.json', {'error': repr(exc), 'traceback': traceback.format_exc()})
        raise
