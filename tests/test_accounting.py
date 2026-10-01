"""End-to-end accounting invariants for planning, reduce and concurrent jobs."""

import json
import shutil
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer

from backend.common.config import ClusterConfig
from backend.common.logbus import LogBus
from backend.common.storage import Storage
from backend.master.fault_tolerance import FaultTolerance
from backend.master.job_manager import JobManager
from backend.master.metrics import Metrics
from backend.master.registry import WorkerRegistry
from backend.master.scheduler import Scheduler
from backend.master.shuffle import ShuffleCoordinator
from backend.worker.executor import Executor, _execute_task, _run_map, _run_reduce
from backend.worker.shuffle_store import ShuffleStore


def base_spec(job_id: str, task_id: str = "m-0000") -> dict:
    return {
        "task_id": task_id,
        "job_id": job_id,
        "kind": "map",
        "mapper": "wordcount_mapper",
        "reducer": "count_reducer",
        "params": {},
        "partition_count": 1,
        "records": ["alpha beta", "alpha gamma", "beta gamma"],
        "spill_records": 1,
        "tmp_dir": "",
    }


class ShuffleHandler(BaseHTTPRequestHandler):
    pairs: list[list] = []

    def do_GET(self):
        body = json.dumps(self.pairs).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        pass


def serve_pairs(pairs: list[list]) -> tuple[HTTPServer, threading.Thread, int]:
    handler = type("BoundShuffleHandler", (ShuffleHandler,), {"pairs": pairs})
    server = HTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread, server.server_address[1]


class TestAccountingInvariants(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.storage = Storage(self.tmp)
        self.jm = JobManager(self.storage, ClusterConfig(), LogBus(self.storage))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_shard_counts_conserve_declared_input_rows(self):
        cases = [
            (1, 1, 1),
            (1, 8, 1),
            (2, 1, 1),
            (7, 3, 3),
            (10, 3, 3),
            (31, 7, 7),
            (100, 100, 100),
            (101, 100, 100),
        ]
        for rows, requested_maps, expected_maps in cases:
            with self.subTest(rows=rows, maps=requested_maps):
                job = self.jm.submit({
                    "name": "accounting",
                    "mapper": "wordcount_mapper",
                    "reducer": "count_reducer",
                    "num_map_tasks": requested_maps,
                    "num_reduce_tasks": 1,
                    "input_rows": rows,
                    "params": {},
                })
                shards = self.jm.planner.input_shards(job)
                self.assertEqual(len(job.map_task_ids), expected_maps)
                self.assertEqual(sum(s["count"] for s in shards), rows)
                self.assertEqual(job.input_rows, rows)
                self.assertEqual(job.stats["total_records"], rows)
                dispatched = sum(
                    len(self.jm.planner.load_input_shard(job.job_id, s["shard_id"]))
                    for s in shards
                )
                self.assertEqual(dispatched, rows)

    def test_invalid_input_rows_rejected(self):
        for rows in (0, -1):
            with self.subTest(rows=rows):
                with self.assertRaises(ValueError):
                    self.jm.submit({
                        "name": "bad",
                        "mapper": "wordcount_mapper",
                        "reducer": "count_reducer",
                        "num_map_tasks": 2,
                        "num_reduce_tasks": 1,
                        "input_rows": rows,
                        "params": {},
                    })

    def test_reduce_emits_every_group_including_last_key(self):
        tmp = tempfile.mkdtemp()
        job_id = "reduce-accounting"
        task_id = "m-0000"
        spec = base_spec(job_id, task_id)
        spec["tmp_dir"] = tmp
        _run_map(spec, tmp, lambda progress, processed, emitted: None)
        pairs = ShuffleStore(tmp).read_partition(job_id, task_id, 0)
        server, _, port = serve_pairs(pairs)
        try:
            result = _run_reduce({
                "job_id": job_id,
                "task_id": "r-0000",
                "reducer": "count_reducer",
                "params": {},
                "partition": 0,
                "spill_records": 1,
                "tmp_dir": tmp,
                "fetch_plan": [{
                    "worker_url": f"http://127.0.0.1:{port}",
                    "map_task_id": task_id,
                }],
            }, lambda progress, processed, emitted: None)

            counts = {row["key"]: row["count"] for row in result["results"]}
            self.assertEqual(counts, {"alpha": 2, "beta": 2, "gamma": 2})
            self.assertEqual(result["records_processed"], 6)
            self.assertEqual(result["records_emitted"], 3)
        finally:
            server.shutdown()
            server.server_close()
            shutil.rmtree(tmp, ignore_errors=True)

    def test_map_retry_replaces_intermediate_data_without_duplicates(self):
        tmp = tempfile.mkdtemp()
        try:
            job_id = "retry-reset"
            spec = base_spec(job_id)
            spec["tmp_dir"] = tmp
            noop = lambda progress, processed, emitted: None
            _run_map(spec, tmp, noop)
            _run_map(spec, tmp, noop)
            pairs = ShuffleStore(tmp).read_partition(job_id, "m-0000", 0)
            self.assertEqual(len(pairs), 6)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_simulated_failure_only_applies_to_first_attempt(self):
        tmp = tempfile.mkdtemp()
        try:
            first = base_spec("fault-job")
            first["simulate_failure"] = True
            first["attempt"] = 0
            first["tmp_dir"] = tmp
            with self.assertRaisesRegex(RuntimeError, "simulated failure"):
                _execute_task(first, tmp, lambda progress, processed, emitted: None)

            retry = dict(first)
            retry["attempt"] = 1
            result = _execute_task(retry, tmp, lambda progress, processed, emitted: None)
            self.assertEqual(result["records_processed"], 3)
            self.assertEqual(result["records_emitted"], 6)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class TestShuffleAccounting(unittest.TestCase):
    def setUp(self):
        self.master_tmp = tempfile.mkdtemp()
        self.worker_tmp = tempfile.mkdtemp()
        self.storage = Storage(self.master_tmp)
        self.config = ClusterConfig()
        self.logbus = LogBus(self.storage)
        self.jm = JobManager(self.storage, self.config, self.logbus)
        self.registry = WorkerRegistry(self.storage, self.config)

    def tearDown(self):
        shutil.rmtree(self.master_tmp, ignore_errors=True)
        shutil.rmtree(self.worker_tmp, ignore_errors=True)

    def test_shuffle_plan_uses_actual_partition_bytes(self):
        job = self.jm.submit({
            "name": "shuffle-bytes",
            "mapper": "wordcount_mapper",
            "reducer": "count_reducer",
            "num_map_tasks": 1,
            "num_reduce_tasks": 1,
            "input_rows": 3,
            "params": {},
        })
        worker = self.registry.register({
            "worker_id": "worker-bytes",
            "name": "bytes",
            "host": "127.0.0.1",
            "port": 9999,
            "cpu_cores": 1,
            "mem_total_mb": 1,
        })
        map_task = self.jm.tasks_for(job.job_id, "map")[0]
        reduce_task = self.jm.tasks_for(job.job_id, "reduce")[0]
        spec = base_spec(job.job_id, map_task.task_id)
        spec["records"] = self.jm.planner.load_input_shard(job.job_id, map_task.input_shard)
        spec["tmp_dir"] = self.worker_tmp
        result = _run_map(spec, self.worker_tmp, lambda progress, processed, emitted: None)

        self.jm.update_task(
            job.job_id, map_task.task_id,
            worker_id=worker.worker_id,
            stats={"partition_sizes": result["partition_sizes"]},
        )
        coordinator = ShuffleCoordinator(self.storage, self.jm, self.registry, self.logbus)
        coordinator.build(job)

        actual_bytes = ShuffleStore(self.worker_tmp).partition_sizes(job.job_id, map_task.task_id)
        expected_bytes = actual_bytes["part-0000"]
        doc = self.storage.read(
            "jobs", job.job_id, "shuffle", f"part-{reduce_task.partition:04d}.json"
        )
        self.assertGreater(expected_bytes, 0)
        self.assertEqual(doc["total_bytes"], expected_bytes)
        self.assertEqual(doc["sources"][0]["bytes"], expected_bytes)
        self.assertEqual(self.jm.get_task(job.job_id, reduce_task.task_id).stats["shuffle_bytes"],
                         expected_bytes)


class TestStaleAttemptReports(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.storage = Storage(self.tmp)
        self.config = ClusterConfig()
        self.logbus = LogBus(self.storage)
        self.jm = JobManager(self.storage, self.config, self.logbus)
        self.registry = WorkerRegistry(self.storage, self.config)
        self.fault_tolerance = FaultTolerance(self.storage, self.jm, self.config, self.logbus)
        self.metrics = Metrics(self.storage)
        self.shuffle = ShuffleCoordinator(self.storage, self.jm, self.registry, self.logbus)
        self.scheduler = Scheduler(
            self.storage, self.jm, self.registry, self.shuffle,
            self.fault_tolerance, self.metrics, self.config, self.logbus,
        )
        self.job = self.jm.submit({
            "name": "stale-attempt",
            "mapper": "wordcount_mapper",
            "reducer": "count_reducer",
            "num_map_tasks": 1,
            "num_reduce_tasks": 1,
            "input_rows": 1,
            "params": {},
        })
        self.task = self.jm.tasks_for(self.job.job_id, "map")[0]

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_old_attempt_completion_is_ignored_after_retry(self):
        self.fault_tolerance.handle_task_failure(
            self.job, self.task, "attempt 0 failed", "worker-old"
        )
        retried = self.jm.get_task(self.job.job_id, self.task.task_id)
        self.assertEqual(retried.attempts, 1)
        self.assertEqual(retried.status, "RETRYING")

        self.scheduler.on_task_complete({
            "job_id": self.job.job_id,
            "task_id": self.task.task_id,
            "attempt": 0,
            "worker_id": "worker-old",
            "status": "SUCCEEDED",
            "records_processed": 1,
            "records_emitted": 1,
            "partition_sizes": {},
            "results": [],
        })
        task = self.jm.get_task(self.job.job_id, self.task.task_id)
        self.assertEqual(task.status, "RETRYING")
        self.assertEqual(task.worker_id, None)

        self.scheduler.on_task_complete({
            "job_id": self.job.job_id,
            "task_id": self.task.task_id,
            "attempt": 1,
            "worker_id": "worker-new",
            "status": "SUCCEEDED",
            "records_processed": 1,
            "records_emitted": 1,
            "partition_sizes": {},
            "results": [],
        })
        task = self.jm.get_task(self.job.job_id, self.task.task_id)
        self.assertEqual(task.status, "SUCCEEDED")
        self.assertEqual(task.worker_id, "worker-new")


class TestExecutorJobIsolation(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_same_task_id_from_different_jobs_runs_concurrently(self):
        executor = Executor("worker-test", self.tmp, "http://127.0.0.1:1",
                            ClusterConfig(), exec_mode="thread")
        try:
            self.assertTrue(executor.start_task(base_spec("job-a", "m-0000")))
            self.assertTrue(executor.start_task(base_spec("job-b", "m-0000")))
            self.assertEqual(executor.running_count, 2)

            # Ambiguous cancellation without a job id must not cancel either job.
            self.assertFalse(executor.cancel("", "m-0000"))
            self.assertEqual(executor.running_count, 2)
            self.assertTrue(executor.cancel("job-a", "m-0000"))

            deadline = time.time() + 5
            while executor.running_count and time.time() < deadline:
                time.sleep(0.02)
            self.assertEqual(executor.running_count, 0)
        finally:
            executor.shutdown()


if __name__ == "__main__":
    unittest.main()
