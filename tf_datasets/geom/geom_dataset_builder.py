"""geom dataset."""

from functools import partial
import tensorflow_datasets as tfds
import tensorflow as tf
import itertools

from preprocessing import yield_dataset, yield_extra_test_set


class Builder(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for geom dataset."""

  VERSION = tfds.core.Version('0.1.4')
  RELEASE_NOTES = {
      '0.0.1': 'Small subset for testing.',
      '0.0.2': 'Small subset including all splits, fix eigenvalue masking and edge attr shape.',
      '0.0.3': 'Full dataset.',
      '0.0.4': 'Small test set including chirality and pickled mols.',
      '0.0.5': 'Small subset to test everything including drugs.',
      '0.0.6': 'Another test.',
      '0.0.7': 'Drugs with 1M samples train. 100K val / test.',
      '0.1.0': 'Complete version.',
      '0.1.1': 'Complete version. Fixed the error related to disconnected components. Fixed caching eigenvalues bug. Store cleaned smiles string.',
      '0.1.2': 'Sort w. boltzmannweight. Add ETFlow features. Keep 30 conformers for training.',
      '0.1.3': 'Add shortest hop information.',
      '0.1.4': 'QM9 & Drugs XL extrapolation.',
  }
  BUILDER_CONFIGS = [
      tfds.core.BuilderConfig(name="drugs", description="Loads drugs data"),
      tfds.core.BuilderConfig(name="qm9", description="Loads qm9 data"),
      tfds.core.BuilderConfig(name="qm9_ablation", description="Loads qm9 ablation data"),
      tfds.core.BuilderConfig(name="xl", description="Loads xl data"),
  ]

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    dataset = self.builder_config.name
    if dataset == "drugs":
      n_atom_features = 94
    if dataset == "qm9":
      n_atom_features = 64
    if dataset == "qm9_ablation":
      n_atom_features = 94
    if dataset == "xl":
      n_atom_features = 94

    return self.dataset_info_from_configs(
        features=tfds.features.FeaturesDict({
            'graph_x1': tfds.features.Tensor(shape=(None, 3), dtype=tf.float32),
            'graph_node_attr': tfds.features.Tensor(shape=(None, n_atom_features), dtype=tf.uint8),
            'graph_atomic_numbers': tfds.features.Tensor(shape=(None,), dtype=tf.uint8),
            'graph_shortest_hops': tfds.features.Tensor(shape=(None,), dtype=tf.uint16),
            'prior_node_attr': tfds.features.Tensor(shape=(None,), dtype=tf.float32),
            'prior_edge_attr': tfds.features.Tensor(shape=(None,), dtype=tf.float32),
            'prior_senders': tfds.features.Tensor(shape=(None,), dtype=tf.uint8),
            'prior_receivers': tfds.features.Tensor(shape=(None,), dtype=tf.uint8),
            'cond_edge_attr': tfds.features.Tensor(shape=(None, 4), dtype=tf.uint8),            
            'cond_senders': tfds.features.Tensor(shape=(None,), dtype=tf.uint8),
            'cond_receivers': tfds.features.Tensor(shape=(None,), dtype=tf.uint8),
            'smiles': tfds.features.Text(),
            'smiles_corrected': tfds.features.Text(),
            'smiles_index': tfds.features.Scalar(dtype=tf.int32),
            'edge_case': tfds.features.Scalar(dtype=tf.uint8),
            'chiral_nbr_index': tfds.features.Tensor(shape=(None,4), dtype=tf.uint8),
            'chiral_tag': tfds.features.Tensor(shape=(None,), dtype=tf.int8),
            'rdkit_pickle': tfds.features.Tensor(shape=(), dtype=tf.string),
        }),
        supervised_keys=None,  # Set to `None` to disable
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""
    dataset = self.builder_config.name
    if dataset == "qm9" or dataset == "drugs":
        modes = ["train", "val", "test_small"] # , "test_full"
    else:
        modes = ["test_small"]
    return {
        k: self._generate_examples(k) for k in modes
    }

  def _generate_examples(self, mode):
    """Yields examples."""

    dataset = self.builder_config.name
    if mode == "test":
      raise NotImplementedError("Choose full or small test set.")
    if mode == "test_small":
      iterator = partial(yield_extra_test_set, datadir="../data", dataset=dataset)
    elif mode == "test_full":
      iterator = partial(yield_dataset, datadir="../data", dataset=dataset, mode="test", max_confs=None)
    else:
      iterator = partial(yield_dataset, datadir="../data", dataset=dataset, mode=mode, max_confs=30)

    for idx, sample in enumerate(iterator()):
      yield f"{idx:09d}_{sample['smiles_index']}", sample
