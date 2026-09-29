"""为本机控制消息提供无损编码，不改变任何运行时决策信息。"""
import base64
import json
import zlib


def pack(value):
    """对完整JSON值进行无损压缩，所有字段均保留。"""
    raw=json.dumps(value,separators=(',',':'),ensure_ascii=False).encode()
    return {'codec':'json-zlib-base64-v1','payload':base64.b64encode(zlib.compress(raw,1)).decode('ascii')}


def unpack(value):
    """严格识别编码并恢复原始JSON字段，损坏消息直接失败。"""
    if not isinstance(value,dict) or set(value)!= {'codec','payload'} or value['codec']!='json-zlib-base64-v1':
        raise ValueError('控制消息编码格式错误')
    return json.loads(zlib.decompress(base64.b64decode(value['payload'],validate=True)))
