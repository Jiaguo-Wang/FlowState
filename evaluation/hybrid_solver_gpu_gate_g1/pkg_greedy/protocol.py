"""无Runtime依赖的单请求证据门禁。"""
def require(value,message):
    if not value:raise ValueError(message)

def admission_gate(baseline,current,received,expected):
    require(received==expected,'Admission实际Token输入不一致')
    require(current['semantic']==baseline['admission_baseline']['semantic'],'Admission前语义变化')
    require(current['nodes']==baseline['admission_baseline']['nodes'],'Admission前物理节点变化')
    require(current['selected_payload']==baseline['selected_payload_after'],'Admission前Selected内容变化')

def compare_outputs(strict,cold):
    for output in (strict,cold):
        require(len(output['output_ids'])==8 and output['meta_info']['completion_tokens']==8,'输出长度不满足冻结协议')
        require(int(output['meta_info'].get('num_retractions',0) or 0)==0,'请求发生回撤')
    require(strict['output_ids']==cold['output_ids'],'逐Token输出不一致，不允许放宽比较')
    require(strict['meta_info'].get('finish_reason')==cold['meta_info'].get('finish_reason'),'停止行为不一致')
    return {'status':'PASS','exact_token_match':True,'token_count':8,'strict_output_ids':strict['output_ids'],'cold_output_ids':cold['output_ids'],'finish_reason':strict['meta_info'].get('finish_reason'),'logits':'既有协议未提供Logits比较，本轮未新增容差'}
