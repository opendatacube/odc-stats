"""
Gather everything to be created on worker process,
or not be able to be serialized,
or to be cached for all plugins
"""

from dask.distributed import WorkerPlugin, get_worker
import tflite_runtime.interpreter as tflite
import tl2cgen
import threading
import logging


# worker plugin to load the TensorFlow Lite model
class TensorFlowLiteModelPlugin(WorkerPlugin):
    def __init__(self, model_path):
        self.model_path = model_path
        self._log = logging.getLogger(__name__)

    def setup(self, worker):
        worker.plugin_instance = self
        worker.interpreters = {}

    def get_interpreter(self):
        worker = get_worker()
        thread_id = threading.get_ident()
        if thread_id not in worker.interpreters:
            interpreter = tflite.Interpreter(model_path=self.model_path)
            interpreter.allocate_tensors()
            worker.interpreters[thread_id] = interpreter
            self._log.info(
                "Interpreter created on worker %s for thread %s",
                worker.address,
                thread_id,
            )
        return worker.interpreters[thread_id]


class TreeliteModelPlugin(WorkerPlugin):
    def __init__(self, model_path):
        self.model_path = model_path
        self._log = logging.getLogger(__name__)

    def setup(self, worker):
        worker.plugin_instance = self
        worker.predictors = {}
        print(f"registered worker {worker}")

    def get_predictor(self):
        worker = get_worker()
        thread_id = threading.get_ident()
        if thread_id not in worker.predictors:
            predictor = tl2cgen.Predictor(self.model_path)
            worker.predictors[thread_id] = predictor
            self._log.info(
                "Predictor created on worker %s for thread %s",
                worker.address,
                thread_id,
            )
        return worker.predictors[thread_id]
