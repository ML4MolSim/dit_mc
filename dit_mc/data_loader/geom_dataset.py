from dataclasses import dataclass
import os
import queue
import sys
import threading
import time
import gdown
import jax
import jraph
import numpy as np
import psutil
import tensorflow as tf
import multiprocessing as mp
import tensorflow_datasets as tfds
import wandb
from dit_mc.data_loader.utils import create_graph_tuples
from dit_mc.data_loader.base import DataLoader
from dit_mc.training.utils import dynamically_batch_extended
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Manager
from dit_mc.jraph_utils import get_number_of_nodes
from multiprocessing import get_context


@dataclass
class WorkerConfig:
    geom_subfolder: str
    split: str
    filter_edge_cases: bool
    cutoff: float
    max_num_graphs: int
    num_atoms: int
    shuffle_seed: int
    debug: bool # whether to take a subset of data
    n_workers: int
    worker_idx: int
    skip_loading_prefetch: bool

    
class StopToken:
    pass


class GeomDataset(DataLoader):
    bond_type_list = ["SINGLE", "DOUBLE", "TRIPLE", "AROMATIC", "misc"]
    STOP_TOKEN = StopToken()
    N_DEBUG = 10_000

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.data_dir = self.data_cfg["data_dir"]
        self.dataset = self.data_cfg["dataset"]
        self.geom_subfolder = os.path.join(self.data_dir, 'geom', self.dataset)

        # check dataset version
        builder = tfds.builder_from_directory(self.geom_subfolder)
        assert (builder.info.version == "0.1.3" or builder.info.version == "0.1.4"), f"Dataset version {builder.info.version} is out of date. Please use version 0.1.3 or 0.1.4."

        self.cutoff = self.data_cfg["cutoff"]

        if "download_key" in self.data_cfg:
            self.download_key = self.data_cfg["download_key"]
        else:
            self.download_key = None

        if "filter_edge_cases_bool" in self.data_cfg:
            self.filter_edge_cases_bool = self.data_cfg["filter_edge_cases_bool"]
        elif "filter_edge_cases" in self.data_cfg:
            # this is a legacy name, but we keep it for backward compatibility
            self.filter_edge_cases_bool = self.data_cfg["filter_edge_cases"]
        else:
            # default to False
            self.filter_edge_cases_bool = False

        # this only applies to distributed loading (some optimization)
        if "skip_loading_prefetch" in self.data_cfg:
            self.skip_loading_prefetch = self.data_cfg["skip_loading_prefetch"]
        else:
            # default to False
            self.skip_loading_prefetch = False

        if "n_proc" in self.data_cfg:
            self.n_proc = self.data_cfg["n_proc"]
        else:
            print("Warning: No number of processes specified. Defaulting to 4.")
            self.n_proc = 4

        if self.n_proc > 0:
            # multithread stuff
            ctx = get_context("spawn")
            self.manager = ctx.Manager()
            self.executor = ProcessPoolExecutor(max_workers=self.n_proc, mp_context=ctx)

            self.stop_event = threading.Event()
            # Start the monitor thread
            self.monitor_thread = threading.Thread(target=GeomDataset._monitor_parent_children, args=(1.0, self.stop_event))
            self.monitor_thread.start()


    def download(self):
        if self.download_key is None:
            raise ValueError("No download key provided. Cannot download dataset.")

        if not os.path.exists(self.geom_subfolder):
            print(f"Downloading data to: {self.geom_subfolder}")
            gdown.download_folder(id=self.download_key, output=self.geom_subfolder, quiet=False)
        else:
            print(f"Found data in: {self.geom_subfolder}. Skipping download")

    @staticmethod
    def _preprocess(dataset, filter_edge_cases, num_atoms, max_num_graphs):
        if filter_edge_cases:
            dataset = dataset.filter(lambda x: x["edge_case"] == 0)

        dataset = dataset.map(
            lambda element: create_graph_tuples(
                element, 
                cutoff=float("inf"),
                split="train",
            ), 
            num_parallel_calls=tf.data.AUTOTUNE,
        ).shuffle(
            buffer_size=10_000,
            reshuffle_each_iteration=True,  # we won't have more than one iteration so probably doesn't matter
        ).prefetch(tf.data.AUTOTUNE)

        batched_dataset = dynamically_batch_extended(
            dataset.as_numpy_iterator(),
            n_graph=max_num_graphs,
            n_node=(max_num_graphs - 1) * num_atoms + 1,
            n_edge=(max_num_graphs - 1) * num_atoms * num_atoms + 1,
            n_edge_cond=(max_num_graphs - 1) * num_atoms * 4 + 1,
            n_edge_prior=(max_num_graphs - 1) * num_atoms * num_atoms + 1)
        
        return batched_dataset

    @staticmethod
    def _safe_put(queue, item):
        try:
            queue.put(item)
        except (EOFError, BrokenPipeError, ConnectionResetError) as e:
            print(f"[!] Queue closed before item could be put: {e}")
        except Exception as e:
            print(f"[!] Unexpected error putting item to queue: {e}")

    @staticmethod
    def _worker(config, output_queue):
        """Worker function that loads data for the given indices and puts it into the queue."""

        try:
            # We need a deterministic shuffle seed, s.t. workers shuffle files in the same way
            read_config = tfds.ReadConfig(
                shuffle_seed=config.shuffle_seed,
                skip_prefetch=config.skip_loading_prefetch
            )

            # Note that tfds automatically pre-fetches after reading
            # This might be suboptimal if we prefetch later and we can try to disable it
            builder = tfds.builder_from_directory(config.geom_subfolder)
            dataset = builder.as_dataset(split=config.split, shuffle_files=True, read_config=read_config)

            if config.debug:
                dataset = dataset.take(GeomDataset.N_DEBUG)

            dataset = dataset.shard(num_shards=config.n_workers, index=config.worker_idx)

            for batch in GeomDataset._preprocess(dataset, 
                                                filter_edge_cases=config.filter_edge_cases, 
                                                num_atoms=config.num_atoms, 
                                                max_num_graphs=config.max_num_graphs):
                
                graph, graph_cond, graph_prior = batch

                # sanity check batch before storing to buffer
                if get_number_of_nodes(graph) == get_number_of_nodes(graph_cond) == get_number_of_nodes(graph_prior):
                    pass
                else:
                    n_graph, n_graph_cond, n_graph_prior = get_number_of_nodes(graph), get_number_of_nodes(graph_cond), get_number_of_nodes(graph_prior)
                    raise RuntimeError(
                        f'Number of nodes unequal after batching. '
                        f'Received {n_graph=}, {n_graph_cond=} and {n_graph_prior=}.'
                    )

                output_queue.put(batch)
        except Exception as e:
            print(f"[!] Error in worker {config.worker_idx}: {e}")
        finally:
            # Finished processing data, always put a stop token
            GeomDataset._safe_put(output_queue, GeomDataset.STOP_TOKEN)
    
    @staticmethod
    def _monitor_parent_children(interval=60.0, stop_event=None):
        """Monitor all child processes of the current process."""
        parent = psutil.Process(os.getpid())
        print("[Monitor] Started monitoring subprocesses...")

        while not stop_event.is_set():
            children = parent.children(recursive=True)
            mem = parent.memory_info().rss / (1024 * 1024 * 1024)
            for child in children:
                try:
                    if child.is_running() and child.status() != psutil.STATUS_ZOMBIE:
                        mem += child.memory_info().rss / (1024 * 1024 * 1024)  # GB
                except psutil.NoSuchProcess:
                    pass
            if wandb.run is not None:
                wandb.log({"total_memory_GB": mem})
            time.sleep(interval)

        print("[Monitor] Stopped monitoring.")

    def _generator(self, split):
        """Generator reads data from the queue."""
        shuffle_seed = np.random.randint(0, 2**31 - 1) # different shuffle seed for each epoch
        output_queue = self.manager.Queue(maxsize=10) # cache maximum of 100 batches, use a fresh queue to collect results

        # processes = []
        for i in range(self.n_proc):
            config = WorkerConfig(
                geom_subfolder=self.geom_subfolder,
                split=split,
                filter_edge_cases=self.filter_edge_cases_bool,
                cutoff=self.cutoff,
                max_num_graphs=self.max_num_graphs,
                num_atoms=self.num_atoms_mean,
                shuffle_seed=shuffle_seed,
                debug=self.debug,
                n_workers=self.n_proc,
                worker_idx=i,
                skip_loading_prefetch=self.skip_loading_prefetch
            )
            # p = mp.Process(target=GeomDataset._worker, args=(config, output_queue), daemon=True)
            # p.start()
            # processes.append(p)
            self.executor.submit(GeomDataset._worker, config, output_queue)

        n_stop_token = 0
        ctr = 0
        while True:
            ctr += 1

            try:
                batch = output_queue.get(timeout=3)
            except queue.Empty:
                # Timeout expired, try again
                continue
            except Exception as e:
                print(f"[!] Unexpected error retrieving item from queue: {e}")
                break

            if ctr % 100 == 0:
                size = output_queue.qsize()
                if wandb.run is not None:
                    wandb.log({"queue_size": size})

            if isinstance(batch, StopToken):
                n_stop_token += 1

                # wait for all processes to finish
                if n_stop_token == self.n_proc:
                    break
            else:
                yield batch

        del output_queue

    def _generator_sync(self, split):
        builder = tfds.builder_from_directory(self.geom_subfolder)
        dataset = builder.as_dataset(split=split, shuffle_files=True)

        if self.debug:
            dataset = dataset.take(GeomDataset.N_DEBUG)

        for batch in self._preprocess(dataset, 
                                 filter_edge_cases=self.filter_edge_cases_bool, 
                                 num_atoms=self.num_atoms_mean, 
                                 max_num_graphs=self.max_num_graphs):
            yield batch

    def get_len(self, split):
        builder = tfds.builder_from_directory(self.geom_subfolder)
        dataset = builder.as_dataset(split=split, shuffle_files=True)
        num_examples = len(dataset)

        if self.debug:
            num_examples = min(GeomDataset.N_DEBUG, num_examples)

        return num_examples

    def next_epoch(self, split):
        # TEST SINGLE SAMPLE (dummy)
        # sample = self.get_sample(split)

        # max_num_graphs = self.max_num_graphs
        # num_atoms = self.num_atoms_mean
        # for b in dynamically_batch_extended(
        #     [sample] * (self.max_num_graphs * 100),
        #     n_graph=max_num_graphs,
        #     n_node=(max_num_graphs - 1) * num_atoms + 1,
        #     n_edge=(max_num_graphs - 1) * num_atoms * num_atoms + 1,
        #     n_edge_cond=(max_num_graphs - 1) * num_atoms * 4 + 1,
        #     n_edge_prior=(max_num_graphs - 1) * num_atoms * num_atoms + 1):
        #     yield b

        # Loads the data for ONE epoch
        if self.n_proc > 0:
            return self._generator(split)
        else:
            return self._generator_sync(split)

    def get_sample(self, split="train"):
        builder = tfds.builder_from_directory(self.geom_subfolder)
        dataset = builder.as_dataset(split=split, shuffle_files=True)
        dataset = dataset.map(
            lambda element: create_graph_tuples(
                element, 
                cutoff=float("inf"),
                split="train",
            )
        )

        return tfds.as_numpy(next(iter(dataset.take(1))))

    def shutdown(self):
        if self.n_proc > 0:
            print("Shutting down monitor thread...")
            self.stop_event.set()
            self.monitor_thread.join()
            print("Shutting down manager...")
            self.manager.shutdown()
            time.sleep(5)  # Give some time for the workers to finish
            print("Shutting down executor...")
            self.executor.shutdown(wait=False, cancel_futures=True)

    # Legacy dataloading for compatibility
    def __call__(self, split):
        assert split in ['train', 'val', 'test', 'test_small'], f"Invalid split: {split}" 
        builder = tfds.builder_from_directory(
            os.path.join(self.data_dir, 'geom', self.dataset))
        dataset = builder.as_dataset(
            split=split,
            shuffle_files=True,
        )

        if self.debug:
            dataset = dataset.take(GeomDataset.N_DEBUG)

        num_samples = len(dataset)

        if self.filter_edge_cases_bool:
            dataset = dataset.filter(lambda x: x['edge_case'] == 0)

        dataset = dataset.map(
            lambda element: create_graph_tuples(
                element, 
                cutoff=self.cutoff,
                split=split,
            )
        )

        return dataset, num_samples
