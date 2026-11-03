#!/bin/bash
# G2-B 正式 40 臂顺序驱动：按 schedule.json 顺序执行，任一失败立即停止（失败证据保留，人工判定重试）
cd "$(dirname "$0")"
python3 - <<'PY' > schedule_order.txt
import json
print('\n'.join('%s %s %s' % (a['round'], a['solver'], a['letter']) for a in json.load(open('schedule.json'))['schedule']))
PY
while read -r R S L; do
  echo "=== $(date -Is) 启动 r${R}_${S,,}_${L,,} ==="
  if ! python3 launch.py "$R" "$S" "$L"; then
    echo "=== $(date -Is) 臂失败：r${R}_${S,,}_${L,,}，驱动停止（保留证据） ==="
    exit 1
  fi
done < schedule_order.txt
echo "=== $(date -Is) 40 臂全部完成 ==="
