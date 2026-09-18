"""Econometric data-access and estimation-sample infrastructure.

This layer is deliberately estimator-agnostic.  It projects raw sources onto
only the columns requested by an economic specification, plans bounded-memory
batching, provides stable categorical encoding across batches, and tracks the
monotone evolution of an estimation sample.
"""

from .source import (
    DataSource, DataFrameSource, CSVSource, StataSource, ParquetSource,
    as_data_source,
)
from .planner import DataIngestionPlan, plan_ingestion
from .sample import EstimationSampleState, SampleExclusion
from .encoding import StableCategoricalEncoder, EncodedBatch
from .dataset import (
    MaterializedData, EncodedEconometricDataset, materialize_required_data,
    materialize_model_data, prepare_encoded_dataset, prepare_repeated_dataset,
)
from .persistent import PersistentSessionStore, strict_source_fingerprint, metadata_source_fingerprint

__all__ = [
    "DataSource", "DataFrameSource", "CSVSource", "StataSource", "ParquetSource",
    "as_data_source", "DataIngestionPlan", "plan_ingestion",
    "EstimationSampleState", "SampleExclusion", "StableCategoricalEncoder",
    "EncodedBatch", "MaterializedData", "EncodedEconometricDataset", "materialize_required_data", "materialize_model_data", "prepare_encoded_dataset", "prepare_repeated_dataset",
    "PersistentSessionStore", "strict_source_fingerprint", "metadata_source_fingerprint",
]
