"""Regression tests for exact record accounting and multi-job isolation."""

import os
import shutil
import tempfile
import time
import unittest
from unittest import mock

from backend.common import constants as C
from backend.common.config import ClusterConfig
from backend.common.http_client import HttpResponse
from backend.common.logbus import LogBus
from backend.common.storage import Storage
from backend.master.fault_tolerance import FaultTolerance
from backend.master.job_manager import JobManager
from backend.master.metrics import Metrics
from backend.master.registry import WorkerRegistry
from backend.master.scheduler import Scheduler
from backend.master.shuffle import ShuffleCoordinator
from backend.tasks import registry
from backend.tasks.samples import generate_input_records
from backend.worker.executor import Executor, _run_map, _run_reduce
from backend.worker.shuffle_store import ShuffleStore


def identity_mapper(records, params):
    return [(rec["key"], rec) for rec in records]


def identity_reducer(key, values, params):
    return {"key": key, "count": len(values), "values": values}


def unique_text_mapper(records, params):
    return [(rec, rec) for rec in records]


class TestExactInputAccounting(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.storage = Storage(self.tmp)
        self.jm = JobManager(self.storage, ClusterConfig(), LogBus(self.storage))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_generated_rows_shards_and_declared_rows_match(self):
        for rows in (1, 2, 7, 100, 1001):
            self.assertEqual(len(generate_input_records("kv", rows, rows)), rows)

        for rows, map_tasks, reduce_tasks in (
            (1, 1, 1),
            (7, 3, 2),
            (100, 100, 7),
            (1001, 8, 4),
        ):
            job = self.jm.submit({
                "name": "exact",
                "mapper": "kv_mapper",
                "reducer": "sum_reducer",
                "num_map_tasks": map_tasks,
                "num_reduce_tasks": reduce_tasks,
                "input_rows": rows,
                "params": {},
            })
            shards = self.jm.planner.input_shards(job)
            self.assertEqual(sum(s["count"] for s in shards), rows)
            self.assertEqual(job.input_rows, rows)
            self.assertEqual(job.stats["total_records"], rows)
            self.assertEqual(len(self.jm.tasks_for(job.job_id, "map")),
                             min(map_tasks, rows))


class TestMapReduceRecordIntegrity(unittest.TestCase):
    def setUp(self):
        registry.register_mapper("identity_mapper_test", identity_mapper)
        registry.register_reducer("identity_reducer_test", identity_reducer)
        self.tmp = tempfile.mkdtemp()
        self.records = [{"key": f"k{i % 5}", "value": i} for i in range(237)]

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_map_partition_counts_cover_every_input_exactly_once(self):
        spec = {
            "task_id": "m-0000",
            "job_id": "job-a",
            "kind": "map",
            "mapper": "identity_mapper_test",
            "reducer": "identity_reducer_test",
            "params": {},
            "partition_count": 4,
            "records": self.records,
            "spill_records": 37,
        }
        result = _run_map(spec, self.tmp, lambda progress, processed, emitted: None)
        self.assertEqual(result["records_processed"], len(self.records))
        self.assertEqual(result["records_emitted"], len(self.records))
        self.assertEqual(
            sum(result["partition_records"].values()), len(self.records)
        )
        self.assertEqual({"part-0000.jsonl", "part-0001.jsonl",
                          "part-0002.jsonl", "part-0003.jsonl"},
                         set(result["partition_sizes"]))

        store = ShuffleStore(self.tmp)
        read_back = sum(
            len(store.read_partition("job-a", "m-0000", p))
            for p in range(4)
        )
        self.assertEqual(read_back, len(self.records))

    def test_reduce_emits_last_group_and_every_group_once(self):
        map_spec = {
            "task_id": "m-0000",
            "job_id": "job-a",
            "kind": "map",
            "mapper": "identity_mapper_test",
            "reducer": "identity_reducer_test",
            "params": {},
            "partition_count": 1,
            "records": self.records,
            "spill_records": 29,
        }
        _run_map(map_spec, self.tmp, lambda progress, processed, emitted: None)
        reduce_spec = {
            "task_id": "r-0000",
            "job_id": "job-a",
            "kind": "reduce",
            "mapper": "identity_mapper_test",
            "reducer": "identity_reducer_test",
            "params": {},
            "partition": 0,
            "fetch_plan": [{
                "worker_url": "http://invalid-worker.invalid",
                "map_task_id": "m-0000",
            }],
            "spill_records": 29,
            "tmp_dir": os.path.join(self.tmp, "spill"),
        }
        with mock.patch("backend.worker.executor.HttpClient") as http_client:
            pairs = ShuffleStore(self.tmp).read_partition("job-a", "m-0000", 0)
            http_client.return_value.get_json.return_value = pairs
            result = _run_reduce(reduce_spec, lambda progress, processed, emitted: None)

        self.assertEqual(result["records_processed"], len(self.records))
        self.assertEqual(result["records_emitted"], 5)
        totals = {row["key"]: row["count"] for row in result["results"]}
        self.assertEqual(totals, {f"k{i}": 48 if i < 2 else 47 for i in range(5)})

    def test_map_retry_starts_from_clean_output(self):
        spec = {
            "task_id": "m-0000",
            "job_id": "job-a",
            "kind": "map",
            "mapper": "identity_mapper_test",
            "reducer": "identity_reducer_test",
            "params": {},
            "partition_count": 2,
            "records": self.records,
            "spill_records": 50,
        }
        first = _run_map(spec, self.tmp, lambda progress, processed, emitted: None)
        second = _run_map(spec, self.tmp, lambda progress, processed, emitted: None)
        self.assertEqual(first["records_emitted"], second["records_emitted"])
        store = ShuffleStore(self.tmp)
        total = sum(len(store.read_partition("job-a", "m-0000", p)) for p in range(2))
        self.assertEqual(total, len(self.records))


class TestSchedulerEndToEndAccounting(unittest.TestCase):
    def setUp(self):
        registry.register_mapper("identity_mapper_e2e", identity_mapper)
        registry.register_mapper("unique_text_mapper_e2e", unique_text_mapper)
        registry.register_reducer("identity_reducer_e2e", identity_reducer)
        self.tmp = tempfile.mkdtemp()
        self.storage = Storage(self.tmp)
        self.config = ClusterConfig(speculative_execution=False)
        self.logbus = LogBus(self.storage)
        self.jm = JobManager(self.storage, self.config, self.logbus)
        self.registry = WorkerRegistry(self.storage, self.config)
        self.registry.register({
            "worker_id": "w1", "name": "w1", "host": "127.0.0.1", "port": 9991,
            "cpu_cores": 4, "mem_total_mb": 1024,
        })
        self.metrics = Metrics(self.storage)
        self.shuffle = ShuffleCoordinator(self.storage, self.jm, self.registry, self.logbus)
        self.ft = FaultTolerance(self.storage, self.jm, self.config, self.logbus)
        self.scheduler = Scheduler(
            self.storage, self.jm, self.registry, self.shuffle, self.ft,
            self.metrics, self.config, self.logbus,
        )

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_identity_job_keeps_declared_count_through_all_stages(self):
        job = self.jm.submit({
            "name": "identity-e2e",
            "mapper": "unique_text_mapper_e2e",
            "reducer": "identity_reducer_e2e",
            "num_map_tasks": 3,
            "num_reduce_tasks": 2,
            "input_rows": 53,
            "params": {},
        })
        rows = self.jm.planner.load_input_shard(
            job.job_id,
            self.jm.tasks_for(job.job_id, C.TASK_MAP)[0].input_shard,
        )
        self.assertTrue(rows)
        dispatched = []

        def fake_post(url, spec=None, timeout=None):
            if url.endswith("/task/execute"):
                dispatched.append((url, spec))
                return HttpResponse(200, {"accepted": True})
            return HttpResponse(200, {})

        def complete_dispatched():
            current = list(dispatched)
            dispatched.clear()
            store = ShuffleStore(self_data_root)
            for _, spec in current:
                if spec["kind"] == C.TASK_MAP:
                    result = _run_map(spec, self_data_root, lambda *args: None)
                else:
                    def fake_fetch(self_client, url, default=None):
                        parts = url.split("/shuffle/", 1)[1].split("/")
                        partition = int(parts[2].split("-")[1].split(".")[0])
                        return store.read_partition(parts[0], parts[1], partition)

                    from backend.worker import executor as worker_executor
                    original_get_json = worker_executor.HttpClient.get_json
                    worker_executor.HttpClient.get_json = fake_fetch
                    try:
                        result = _run_reduce(spec, lambda *args: None)
                    finally:
                        worker_executor.HttpClient.get_json = original_get_json
                self.scheduler.on_task_complete({
                    "worker_id": "w1",
                    "job_id": spec["job_id"],
                    "task_id": spec["task_id"],
                    "kind": spec["kind"],
                    "status": C.TASK_SUCCEEDED,
                    "duration_ms": 1,
                    **result,
                })

        self_data_root = os.path.join(self.tmp, "workers", "w1")
        self.scheduler.client.post = fake_post
        self.scheduler.tick()  # dispatch maps
        complete_dispatched()
        self.scheduler.tick()  # build shuffle and transition to SHUFFLE
        self.assertEqual(job.status, C.JOB_SHUFFLE)
        self.scheduler.tick()  # hold; not enough time has elapsed
        time.sleep(0.45)
        self.scheduler.tick()  # transition to REDUCE
        self.scheduler.tick()  # dispatch reduces
        complete_dispatched()
        self.scheduler.tick()  # finish job

        job = self.jm.get_job(job.job_id)
        self.assertEqual(job.status, C.JOB_SUCCEEDED)
        self.assertEqual(job.stats["input_shard_records"], 53)
        self.assertEqual(job.stats["map_records_processed"], 53)
        self.assertEqual(job.stats["map_records_emitted"], 53)
        self.assertEqual(job.stats["shuffle_records_fetched"], 53)
        self.assertEqual(job.stats["result_records"], 53)
        partitions = self.scheduler._result_partitions(job)
        self.assertEqual(sum(p["count"] for p in partitions), 53)
        stored_values = sum(
            sum(1 for _ in p["records"])
            for p in partitions
        )
        self.assertEqual(stored_values, 53)


class TestExecutorMultiJobIsolation(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.executor = Executor(
            worker_id="w1",
            data_root=self.tmp,
            master_url="http://127.0.0.1:1",
            config=ClusterConfig(),
            exec_mode="thread",
        )

    def tearDown(self):
        self.executor.shutdown()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_same_task_id_from_different_jobs_has_independent_slots(self):
        self.assertTrue(self.executor.start_task({
            "task_id": "m-0000",
            "job_id": "job-a",
            "kind": "map",
        }))
        self.assertTrue(self.executor.start_task({
            "task_id": "m-0000",
            "job_id": "job-b",
            "kind": "map",
        }))
        self.assertEqual(self.executor.running_count, 2)
        self.assertTrue(self.executor.cancel("job-a", "m-0000"))
        self.assertFalse(self.executor.cancel("job-a", "m-0001"))
        # Cancelling one job's task must not affect the same task id elsewhere.
        self.assertTrue(self.executor.cancel("job-b", "m-0000"))


if __name__ == "__main__":
    unittest.main()
