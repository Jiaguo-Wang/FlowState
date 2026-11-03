"""G1 实验B复用程序装配（宿主机执行）：复制 paper_hybrid_gpu_fix_b 并施加唯一最小集成修复。

背景：g1_ck3_greedy_A 失败根因＝fix_b runner 第94行无条件查找目标绑定源节点，而 C_A 在
FS 冻结计划下 E=0 无恢复源（control.py 已有该分支）；c_formal 生产 runner（60 正式臂，
含 5 个 E=0 的 C_A 臂）对 E>0 才做源绑定/COW/别名断言。本修复把该生产语义对齐到 fix_b：
E>0 保持原断言不变；E=0 改为正向断言“无恢复源且无COW痕迹”。不触碰任何求解/协调/安全语义。
产物：pkg_greedy/{bridge,observation,persistence,protocol,control,runner}.py（runner 含修复，
其余逐字节同 fix_b）+ build_record.json（逐文件哈希与差异说明）。fix_b 原目录零改动。
"""
import hashlib,json
from pathlib import Path
ROOT=Path('/home/wjg/code/FlowState')
FIX=ROOT/'evaluation/paper_hybrid_gpu_fix_b'
PKG=ROOT/'evaluation/hybrid_solver_gpu_gate_g1/pkg_greedy'
OLD="""  target=result['target'];bound=next(x['node'] for x in result['reconciliation']['binding']['candidates'] if x['candidate_id']==target['source_checkpoint'])
  require(actual['best_match_node']==bound['node_id'] and actual['cow_source_slots']==bound['slots'],'恢复来源错误')
  require(any(c.get('returned') and c['source_slots']==bound['slots'] and c['destination_slots']==actual['working_slots'] and c['source_payload_before']==c['source_payload_after']==c['destination_payload_after'] for c in post['trace']['copies']),'COW未完成或内容错误')
  kept={s for c in result['reconciliation']['binding']['candidates'] if c['candidate_id'] in result['plan']['selected_ids'] for s in c['node']['slots']}
  require(not kept.intersection(actual['working_slots']),'工作状态与保留状态别名')
"""
NEW="""  # E=0 目标（该计划下无兼容保留状态）：无恢复源/COW/别名断言，仅有FA命中与E/G合同（对齐c_formal生产语义）
  target=result['target']
  if expected_E>0:
   bound=next(x['node'] for x in result['reconciliation']['binding']['candidates'] if x['candidate_id']==target['source_checkpoint'])
   require(actual['best_match_node']==bound['node_id'] and actual['cow_source_slots']==bound['slots'],'恢复来源错误')
   require(any(c.get('returned') and c['source_slots']==bound['slots'] and c['destination_slots']==actual['working_slots'] and c['source_payload_before']==c['source_payload_after']==c['destination_payload_after'] for c in post['trace']['copies']),'COW未完成或内容错误')
   kept={s for c in result['reconciliation']['binding']['candidates'] if c['candidate_id'] in result['plan']['selected_ids'] for s in c['node']['slots']}
   require(not kept.intersection(actual['working_slots']),'工作状态与保留状态别名')
  else:
   require(target['source_checkpoint'] is None and actual['cow_source_slots']==[] and actual['working_slots']==[] and not post['trace']['copies'],'E=0却出现恢复源/COW痕迹')
"""
OLD2="  write(OUT/'target_request.json',pending)\n"
NEW2="  write(out/'target_request.json',pending)  # 留证写入本臂run目录（无读取方；多臂复用时根目录'x'写会跨臂冲突）\n"
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    PKG.mkdir(exist_ok=True)
    (PKG/'__init__.py').write_text('')
    record={'fix_b_dir':str(FIX),'package':str(PKG),'files':{},'deviation':'仅runner.py一处：E=0目标的无源分支（对齐c_formal生产语义）；其余文件逐字节同fix_b'}
    for p in sorted(FIX.glob('*.py')):
        text=p.read_text()
        if p.name=='runner.py':
            assert text.count(OLD)==1 and text.count(OLD2)==1,'fix_b runner 源与预期修复点不匹配（上游已变？）'
            text=text.replace(OLD,NEW).replace(OLD2,NEW2)
            record['deviation_detail']=[
              {'file':'runner.py','anchor':'恢复来源错误/COW/别名三断言块','change':'包裹为 if expected_E>0 分支；E=0改为正向断言(无源/无COW/无工作槽)','reference':'c_formal_perf_gpu/runner.py 行261起 同语义（60正式臂含E=0臂验证）','reason':'g1首跑在C_A(E=0)上StopIteration'},
              {'file':'runner.py','anchor':"write(OUT/'target_request.json')",'change':"改写至本臂run目录 out/target_request.json",'reference':'全仓grep无任何读取方（纯留证写出）','reason':'多臂复用同一/b5out时根目录x写跨臂冲突（g2/g3/g4首批及r2均FileExistsError）'}]
        out=PKG/p.name
        out.write_text(text)
        record['files'][p.name]={'fix_b_sha256':sha(p),'pkg_sha256':sha(out),'identical':sha(p)==sha(out)}
        compile(text,str(out),'exec')
    (PKG/'build_record.json').write_text(json.dumps(record,ensure_ascii=False,indent=2))
    diff=[n for n,v in record['files'].items() if not v['identical']]
    assert diff==['runner.py'],'除runner外出现差异：'+repr(diff)
    print('pkg_greedy 装配完成：runner.py含E=0最小修复，其余与fix_b逐字节一致')
if __name__=='__main__':main()
