"""在冻结镜像中CPU重建目标输入，与原有数组逐Token比较，不加载权重或Engine。"""
import importlib.metadata
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from evaluation.agentic_executable_workload_cpu_gate.adapter import ROOT, PRIOR, OUT, read, require, sha, save_new, frozen_normalizer


def main():
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and os.environ.get('NVIDIA_VISIBLE_DEVICES') == 'void', 'CPU容器必须禁用GPU')
    require(not list(Path('/dev').glob('nvidia*')), 'CPU容器发现GPU设备')
    config = read(PRIOR / 'input_configuration_lock.json')
    versions = {n: importlib.metadata.version(n) for n in ('transformers', 'tokenizers', 'sglang', 'jinja2')}
    require(versions == config['versions'], '镜像依赖版本与冻结输入不同')
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(config['model_path'], local_files_only=True)
    import hashlib
    require(hashlib.sha256(str(tokenizer.chat_template).encode()).hexdigest() == config['loaded_chat_template_sha256'], '模板身份不符')
    normalize = frozen_normalizer(config)
    inputs = read(OUT / 'prepared_inputs.json')
    checks = []
    for row in inputs['history'] + inputs['pendings']:
        messages = [normalize(m) for m in row['raw_messages']]
        one = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        two = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        require(isinstance(one, list) and one == two == row['token_ids'], '逐Token重建失败：' + row['request_id'])
        checks.append({'request_id': row['request_id'], 'token_count': len(one), 'exact_token_match': True, 'repeat_deterministic': True, 'message_count': len(messages)})
    require('sglang' not in sys.modules, '不得导入SGLang运行时')
    save_new(OUT / 'token_reconstruction.json', {'status': 'PASS', 'checks': checks, 'versions': versions, 'template_sha256': config['loaded_chat_template_sha256'], 'gpu_devices': [], 'engine_started': False, 'model_weights_loaded': False, 'original_token_files_modified': False})
    print('六份输入两次重建均与冻结数组逐Token完全相同。')


if __name__ == '__main__':
    main()
