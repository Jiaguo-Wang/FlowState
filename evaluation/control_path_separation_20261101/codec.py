"""取证响应无损封装；仅处理JSON，不接触任何Runtime对象或设备状态。"""
import base64
import hashlib
import json
import time
import zlib

CODEC = 'json-zlib-v1'
MAX_RAW = 256 * 1024 * 1024


def require(value, message):
    if not value:
        raise ValueError(message)


def pack(value, verify=False):
    """一次编码完整响应，不裁剪字段；失败由调用方触发原隔离处理。"""
    start, cpu = time.monotonic_ns(), time.process_time_ns()
    raw = json.dumps(value, ensure_ascii=False, separators=(',', ':')).encode()
    require(len(raw) <= MAX_RAW, '完整响应超过封装上限')
    encoded = time.monotonic_ns()
    compressed = zlib.compress(raw, level=1)
    compressed_ns = time.monotonic_ns()
    result = {'ok': value.get('ok') is True, 'codec': CODEC,
              'raw_bytes': len(raw), 'raw_sha256': hashlib.sha256(raw).hexdigest(),
              'payload_b64': base64.b64encode(compressed).decode('ascii')}
    end = time.monotonic_ns()
    result['encoding'] = {'start_ns': start, 'end_ns': end, 'wall_ms': (end-start)/1e6,
                          'cpu_ms': (time.process_time_ns()-cpu)/1e6,
                          'json_ms': (encoded-start)/1e6, 'compression_ms': (compressed_ns-encoded)/1e6,
                          'compressed_bytes': len(compressed), 'raw_bytes': len(raw),
                          'gpu_calls': 0, 'full_evidence_preserved': True}
    if verify:
        check_start = time.monotonic_ns()
        decoded, _ = unpack(result)
        require(decoded == json.loads(json.dumps(value, sort_keys=True)), '封装与原线协议对象不等价')
        result['encoding']['equivalence_verified'] = True
        result['encoding']['verification_wall_ms'] = (time.monotonic_ns()-check_start)/1e6
    return result


def _unique(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, '响应JSON存在重复键')
        result[key] = value
    return result


def unpack(packet):
    """限制解压大小并认证字节与JSON，任何损坏均抛错，不回退原始响应。"""
    start, cpu = time.monotonic_ns(), time.process_time_ns()
    require(type(packet) is dict and packet.get('codec') == CODEC and packet.get('ok') is True,
            '缺少成功响应的合法封装身份')
    size = packet.get('raw_bytes')
    require(type(size) is int and 0 < size <= MAX_RAW, '封装原始长度非法')
    digest = packet.get('raw_sha256')
    require(type(digest) is str and len(digest) == 64, '响应摘要身份缺失')
    payload = packet.get('payload_b64')
    require(type(payload) is str and len(payload) <= MAX_RAW * 2, '响应载荷非法')
    compressed = base64.b64decode(payload, validate=True)
    decoder = zlib.decompressobj()
    raw = decoder.decompress(compressed, size+1)
    require(len(raw) == size and decoder.eof and not decoder.unconsumed_tail and not decoder.unused_data,
            '响应解压长度、尾部或完整性错误')
    require(hashlib.sha256(raw).hexdigest() == digest, '完整响应摘要不一致')
    value = json.loads(raw.decode('utf-8'), object_pairs_hook=_unique)
    require(type(value) is dict and value.get('ok') is True, '完整响应对象不合法')
    end = time.monotonic_ns()
    return value, {'start_ns': start, 'end_ns': end, 'wall_ms': (end-start)/1e6,
                   'cpu_ms': (time.process_time_ns()-cpu)/1e6, 'raw_bytes': size,
                   'compressed_bytes': len(compressed), 'digest_verified': True,
                   'encoding': packet['encoding'], 'gpu_calls': 0}
