/* 分片管理 Shards */
Components.init('shards');
const C = Components;

let currentJob = '';

async function render() {
  if (!currentJob) return;
  let d;
  try { d = await API.get('/api/jobs/' + currentJob + '/shards'); } catch (e) { return; }

  const inputs = d.input_shards || [];
  const inputTotal = inputs.reduce((sum, s) => sum + Number(s.count || 0), 0);
  const mapProcessed = (d.map_tasks || []).reduce((sum, t) => sum + Number(t.records_processed || 0), 0);
  const shuffledRecords = (d.shuffle && d.shuffle.total_records) || 0;
  document.getElementById('input-shards').innerHTML = inputs.length
    ? `<div class="small muted mb">共 ${inputs.length} 个分片 shards · 合计 ${C.fmtNum(inputTotal)} / 声明 ${C.fmtNum(d.input_rows)} · Map 已处理 ${C.fmtNum(mapProcessed)} · Shuffle ${C.fmtNum(shuffledRecords)}</div>` + inputs.map(s =>
        `<div class="flex between" style="padding:6px 0;border-bottom:1px solid var(--border)">
           <span class="mono">${C.esc(s.shard_id)}</span>
           <span class="muted">${C.fmtNum(s.count)} 条 records</span>
         </div>`).join('')
    : C.empty();

  document.getElementById('map-tasks').innerHTML = (d.map_tasks || []).length
    ? C.table([
        { key: 'task_id', label: 'Task', render: r => `<span class="mono">${C.esc(r.task_id)}</span>` },
        { key: 'input_shard', label: '分片 Shard', render: r => `<span class="mono">${C.esc(r.input_shard)}</span>` },
        { key: 'status', label: '状态', render: r => C.stateBadge(r.status) },
        { key: 'worker_name', label: 'Worker', render: r => C.esc(r.worker_name || '-') },
      ], d.map_tasks)
    : C.empty();

  document.getElementById('reduce-tasks').innerHTML = (d.reduce_tasks || []).length
    ? C.table([
        { key: 'task_id', label: 'Task', render: r => `<span class="mono">${C.esc(r.task_id)}</span>` },
        { key: 'partition', label: '分区 Partition', render: r => `part-${String(r.partition).padStart(4,'0')}`, num: true },
        { key: 'status', label: '状态', render: r => C.stateBadge(r.status) },
        { key: 'worker_name', label: 'Worker', render: r => C.esc(r.worker_name || '-') },
      ], d.reduce_tasks)
    : C.empty();
}

C.jobPicker('job-picker', (id) => { currentJob = id; render(); });
C.poll(render, 2000).start();
