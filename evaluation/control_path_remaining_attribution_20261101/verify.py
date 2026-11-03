"""核验派生分解及原始日志，保留被忽略的启动警告，不修改任何输入。"""
import csv
import hashlib
import json
from pathlib import Path
import re

OUT = Path(__file__).resolve().parent
SRC = OUT.parent/'control_path_separation_20261101'


def main():
    manifest = json.loads((OUT/'source_manifest.json').read_text())
    checks = []
    for rnd in range(1,6):
        for tag in ['fs','mc','kv']:
            arm=f't{rnd}_{tag}'
            base=SRC/'arms'/arm
            text=(base/'runtime.log').read_text()
            blocks=list(re.finditer(r'\[start of libtorchcodec loading traceback\].*?\[end of libtorchcodec loading traceback\]\.?',text,re.S))
            ready=text.index('Engine startup timings')
            assert all(b.end()<ready for b in blocks)
            other=re.sub(r'\[start of libtorchcodec loading traceback\].*?\[end of libtorchcodec loading traceback\]\.?','',text,flags=re.S)
            assert 'Traceback (most recent call last)' not in other,arm
            found=re.findall(r'^完成 .*控制周期 ([0-9.]+) ms 全周期 ([0-9.]+) ms TTFT ',text,re.M)
            assert len(found)==1,(arm,found)
            s=json.loads((base/'run/summary.json').read_text())
            assert abs(float(found[0][0])-s['cycle']['control_cycle_ms'])<=0.051
            assert abs(float(found[0][1])-s['cycle']['cycle_ms_full_incl_warm'])<=0.051
            manifest['files'][str(base/'runtime.log')]=hashlib.sha256((base/'runtime.log').read_bytes()).hexdigest()
            checks.append({'arm':arm,'completion_matches_summary':True,'ignored_startup_torchcodec_blocks':len(blocks),
                           'startup_warning_precedes_engine_ready':True,'other_tracebacks':False})
    for path,expected in manifest['files'].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==expected,path
    rows=list(csv.DictReader((OUT/'paired_attribution.csv').open()))
    assert len(rows)==10
    for row in rows:
        get=lambda k:float(row[k])
        for parent,parts in [
            ('audit_ms',['control_ms','rpc_tail_ms','warm_ms','requests_ms','window_gaps_ms','post_ms']),
            ('control_ms',['observation_ms','mapping_ms','epoch_ms','selection_stage_ms','freshness_ms','fa_before_ms','reconciliation_ms','fa_post_ms','journal_in_ms','control_unknown_ms']),
            ('rpc_tail_ms',['journal_after_ms','journal_flush_ms','baseline_ms','encode_ms','decode_ms','rpc_unknown_ms']),
            ('post_ms',['post_state_audit_ms','report_ms','oracle_ms','post_unknown_ms'])]:
            assert abs(get(parent)-sum(get(k) for k in parts))<1e-7
    manifest['runtime_log_checks']=checks
    (OUT/'source_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    result={'decision':'VALIDATE_SERVING_PATH','raw_arms':15,'paired_comparisons':10,
            'raw_logs_match_summary':True,'all_additive_identities_pass':True,'source_hashes_unchanged':True,
            'warnings':'可选多模态TorchCodec/FFmpeg加载失败被SGLang忽略，位于Engine就绪前，不属于T0–T5成本；原日志保留',
            'gpu_started':False,'production_modified':False}
    (OUT/'validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print('15臂原始日志/摘要一致；10组配对各层精确闭合；源SHA不变')


if __name__ == '__main__':
    main()
