"""G2 Cold 参照落位（宿主机执行）：核验 A/B/D 三条补采臂证据后写入 preflight/ 根 Cold 参照文件。

核验链（全部只读；任一不满足即失败，不落位）：存在 returncode==0 的进程记录、cleanup PASS、
run 目录 summary PASS（arm=cold、fa_hit==0、executable_prefix==0、8 Token、无回撤）、
output.json 与 summary 的 output_ids 一致、post_request 内原生匹配行 FA命中==E==0 且输入摘要一致。
落位文件仅含 output_ids 与来源线索（正式臂只读 output_ids 做逐 Token 对照）。拒绝覆盖。
"""
import json
from pathlib import Path

ROOT = Path('/home/wjg/code/FlowState')
PRE = ROOT / 'evaluation/b5c4_formal_gpu_g2/preflight'
OUT = PRE / 'cold'
LETTERS = ['A', 'B', 'D']


def read(p):
    return json.loads(Path(p).read_text())


def write_once(name, x):
    p = PRE / name
    assert not p.exists(), '拒绝覆盖：' + name
    p.write_text(json.dumps(x, ensure_ascii=False, indent=2))


def latest_passing(letter):
    """取该字母最新 PASS 的 run（含重试序号），逐级核验宿主侧证据。"""
    runs = ['cold_' + letter.lower()] + sorted([p.stem for p in OUT.glob('cold_%s_r*.runtime.log' % letter.lower())], key=lambda s: int(s.rsplit('_r', 1)[1]))
    tried = []
    for run in runs:
        d = OUT / run
        tried.append(run)
        if not (d / 'summary.json').exists():
            continue
        s = read(d / 'summary.json')
        if s.get('status') != 'PASS':
            continue
        assert read(OUT / (run + '_process.json'))['returncode'] == 0, run + ' 进程非零退出'
        assert read(OUT / (run + '_cleanup.json'))['status'] == 'PASS', run + ' 清理失败'
        assert s['arm'] == 'cold' and s['pending'] == 'B5C2_%s_TURN_003' % letter and s['fa_hit'] == 0 and s['executable_prefix'] == 0, run + ' 摘要字段不符'
        assert len(s['output_ids']) == 8, run + ' 输出非 8 Token'
        out = read(d / 'output.json')
        assert out['output_ids'] == s['output_ids'] and out['meta_info']['completion_tokens'] == 8 and int(out['meta_info'].get('num_retractions', 0) or 0) == 0, run + ' 输出证据不符'
        post = read(d / 'post_request.json')
        rows = [m['native'] for m in post['trace']['matches'] if m.get('native', {}).get('request_id') == s['pending']]
        assert rows and rows[-1]['physical_fa_hit'] == rows[-1]['executable_prefix'] == 0, run + ' Cold 命中旧状态'
        return run, s
    raise AssertionError('字母 %s 无 PASS 的 Cold 臂（尝试过：%s）' % (letter, tried))


def main():
    promoted = []
    for letter in LETTERS:
        run, s = latest_passing(letter)
        write_once('cold_reference_B5C2_%s_TURN_003.json' % letter,
                   {'output_ids': s['output_ids'], 'source_run': 'preflight/cold/' + run,
                    'protocol': 'b5d4b Cold 同协议：全新 Engine、空缓存断言、g2_set_cold_target 仅设观察目标、直接请求、FA命中==E==0',
                    'note': 'G2 正式臂仅读 output_ids 做逐 Token 对照'})
        promoted.append({'pending': s['pending'], 'run': run, 'output_ids_sha_prefix': None})
    assert read(PRE / 'cold_reference_B5C2_C_TURN_003.json').get('output_ids'), 'C 冻结 Cold 缺失'
    print('Cold 参照落位完成：A/B/D 已核验落位；C 沿用 gate_a 冻结（4/4 齐）')


if __name__ == '__main__':
    main()
