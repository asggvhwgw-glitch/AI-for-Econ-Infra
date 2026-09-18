from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from collections.abc import Iterator, Sequence
import pandas as pd

from ..errors import InputError, SpecificationError


def _normalize_columns(columns: Sequence[str] | None) -> tuple[str, ...] | None:
    if columns is None:
        return None
    out: list[str] = []
    seen: set[str] = set()
    for c in columns:
        name = str(c)
        if name not in seen:
            seen.add(name)
            out.append(name)
    return tuple(out)


def _reject_control_kwargs(kwargs: dict, forbidden: set[str], source: str) -> None:
    overlap = sorted(set(kwargs).intersection(forbidden))
    if overlap:
        raise SpecificationError(
            f"{source} source options {overlap} are controlled by econhdfe's ingestion planner",
            code="data.source_option", stage="data",
            details={"source": source, "options": overlap},
            suggestion="Remove projection/chunk-size options from the source constructor and let econhdfe plan them.",
        )


def _ordered_projection(frame: pd.DataFrame, columns: tuple[str, ...] | None) -> pd.DataFrame:
    if columns is None:
        return frame
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise InputError(
            "requested data column(s) are not present",
            code="data.missing_column", stage="data",
            details={"missing": missing},
            suggestion="Check the economic specification and source column names.",
        )
    return frame.loc[:, list(columns)]


class DataSource(ABC):
    """Estimator-agnostic source capable of projected, batched scans."""

    @property
    @abstractmethod
    def label(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def preview(self, columns: Sequence[str] | None = None, *, nrows: int = 2048) -> pd.DataFrame:
        raise NotImplementedError

    @abstractmethod
    def scan(self, columns: Sequence[str] | None = None, *, batch_rows: int = 250_000) -> Iterator[pd.DataFrame]:
        raise NotImplementedError

    def materialize(self, columns: Sequence[str] | None = None, *, batch_rows: int = 250_000) -> pd.DataFrame:
        cols = _normalize_columns(columns)
        chunks = list(self.scan(cols, batch_rows=batch_rows))
        if not chunks:
            return pd.DataFrame(columns=list(cols or ()))
        if len(chunks) == 1:
            return chunks[0].reset_index(drop=True)
        return pd.concat(chunks, axis=0, ignore_index=True, copy=False)


class DataFrameSource(DataSource):
    def __init__(self, frame: pd.DataFrame, *, label: str = "dataframe"):
        if not isinstance(frame, pd.DataFrame):
            raise TypeError("DataFrameSource requires a pandas DataFrame")
        self.frame = frame
        self._label = str(label)

    @property
    def label(self) -> str:
        return self._label

    def preview(self, columns=None, *, nrows=2048) -> pd.DataFrame:
        cols = _normalize_columns(columns)
        return _ordered_projection(self.frame.head(max(int(nrows), 0)), cols)

    def scan(self, columns=None, *, batch_rows=250_000):
        cols = _normalize_columns(columns)
        n = len(self.frame)
        step = max(int(batch_rows), 1)
        for lo in range(0, n, step):
            yield _ordered_projection(self.frame.iloc[lo:min(lo + step, n)], cols)

    def materialize(self, columns=None, *, batch_rows=250_000):
        # Avoid concatenating an already in-memory source.  The column projection
        # may still create a lightweight pandas manager, but it does not parse or
        # duplicate unused raw columns.
        return _ordered_projection(self.frame, _normalize_columns(columns))


class CSVSource(DataSource):
    def __init__(self, path, **read_csv_kwargs):
        _reject_control_kwargs(read_csv_kwargs, {"usecols", "chunksize", "nrows", "iterator"}, "CSV")
        self.path = Path(path)
        self.kwargs = dict(read_csv_kwargs)

    @property
    def label(self) -> str:
        return str(self.path)

    def preview(self, columns=None, *, nrows=2048):
        cols = _normalize_columns(columns)
        frame = pd.read_csv(self.path, usecols=None if cols is None else list(cols), nrows=max(int(nrows), 0), **self.kwargs)
        return _ordered_projection(frame, cols)

    def scan(self, columns=None, *, batch_rows=250_000):
        cols = _normalize_columns(columns)
        reader = pd.read_csv(
            self.path, usecols=None if cols is None else list(cols),
            chunksize=max(int(batch_rows), 1), **self.kwargs,
        )
        for frame in reader:
            yield _ordered_projection(frame, cols)

    def materialize(self, columns=None, *, batch_rows=250_000):
        cols = _normalize_columns(columns)
        frame = pd.read_csv(self.path, usecols=None if cols is None else list(cols), **self.kwargs)
        return _ordered_projection(frame, cols).reset_index(drop=True)


class StataSource(DataSource):
    def __init__(self, path, *, backend: str = "auto", **read_stata_kwargs):
        _reject_control_kwargs(read_stata_kwargs, {"columns", "chunksize", "iterator"}, "Stata")
        if backend not in {"auto", "pandas", "pyreadstat"}:
            raise SpecificationError(
                "Stata backend must be auto/pandas/pyreadstat",
                code="data.stata_backend", stage="data",
            )
        self.path = Path(path)
        self.backend = backend
        self.kwargs = dict(read_stata_kwargs)

    @property
    def label(self) -> str:
        return str(self.path)

    def _pyreadstat(self):
        try:
            import pyreadstat  # type: ignore
        except ImportError:
            return None
        return pyreadstat

    def _resolved_backend(self) -> str:
        if self.backend == "pandas":
            return "pandas"
        prs = self._pyreadstat()
        if self.backend == "pyreadstat":
            if prs is None:
                raise SpecificationError(
                    "Stata backend='pyreadstat' requires pyreadstat",
                    code="data.stata_engine", stage="data",
                    suggestion="Install econhdfe with the optional 'io' dependencies, or use backend='pandas'.",
                )
            if self.kwargs:
                raise SpecificationError(
                    "pandas read_stata options cannot be forwarded to pyreadstat",
                    code="data.stata_options", stage="data",
                    details={"options": sorted(self.kwargs)},
                    suggestion="Use backend='pandas' for pandas-specific Stata options.",
                )
            return "pyreadstat"
        # In auto mode, preserve explicit pandas reader options.  Otherwise
        # prefer ReadStat when available because its usecols/row_limit API is
        # designed for selected-column and chunked reads.
        return "pyreadstat" if prs is not None and not self.kwargs else "pandas"

    @property
    def resolved_backend(self) -> str:
        return self._resolved_backend()

    def preview(self, columns=None, *, nrows=2048):
        cols = _normalize_columns(columns)
        if self._resolved_backend() == "pyreadstat":
            prs = self._pyreadstat()
            frame, _ = prs.read_dta(
                str(self.path), usecols=None if cols is None else list(cols),
                row_limit=max(int(nrows), 0),
            )
            return _ordered_projection(frame, cols)
        reader = pd.read_stata(
            self.path, columns=None if cols is None else list(cols),
            chunksize=max(int(nrows), 1), **self.kwargs,
        )
        try:
            frame = next(iter(reader))
        except StopIteration:
            frame = pd.DataFrame(columns=list(cols or ()))
        return _ordered_projection(frame, cols)

    def scan(self, columns=None, *, batch_rows=250_000):
        cols = _normalize_columns(columns)
        if self._resolved_backend() == "pyreadstat":
            prs = self._pyreadstat()
            reader = prs.read_file_in_chunks(
                prs.read_dta, str(self.path), chunksize=max(int(batch_rows), 1),
                usecols=None if cols is None else list(cols),
            )
            for frame, _ in reader:
                yield _ordered_projection(frame, cols)
            return
        reader = pd.read_stata(
            self.path, columns=None if cols is None else list(cols),
            chunksize=max(int(batch_rows), 1), **self.kwargs,
        )
        for frame in reader:
            yield _ordered_projection(frame, cols)

    def materialize(self, columns=None, *, batch_rows=250_000):
        cols = _normalize_columns(columns)
        if self._resolved_backend() == "pyreadstat":
            prs = self._pyreadstat()
            frame, _ = prs.read_dta(
                str(self.path), usecols=None if cols is None else list(cols),
            )
            return _ordered_projection(frame, cols).reset_index(drop=True)
        frame = pd.read_stata(
            self.path, columns=None if cols is None else list(cols), **self.kwargs,
        )
        return _ordered_projection(frame, cols).reset_index(drop=True)


class ParquetSource(DataSource):
    """Column-projected Parquet source.

    Batch-native scanning will use Arrow in a later backend-specific layer.
    This conservative implementation already preserves column projection and
    keeps the optional parquet dependency out of econhdfe's core install.
    """

    def __init__(self, path, **read_parquet_kwargs):
        _reject_control_kwargs(read_parquet_kwargs, {"columns"}, "Parquet")
        self.path = Path(path)
        self.kwargs = dict(read_parquet_kwargs)

    @property
    def label(self) -> str:
        return str(self.path)

    def _read(self, columns=None):
        cols = _normalize_columns(columns)
        try:
            frame = pd.read_parquet(self.path, columns=None if cols is None else list(cols), **self.kwargs)
        except ImportError as exc:
            raise SpecificationError(
                "Parquet ingestion requires an installed parquet engine",
                code="data.parquet_engine", stage="data",
                suggestion="Install econhdfe with the optional 'io' dependencies (pyarrow/polars).",
            ) from exc
        return _ordered_projection(frame, cols)

    def preview(self, columns=None, *, nrows=2048):
        return self._read(columns).head(max(int(nrows), 0))

    def scan(self, columns=None, *, batch_rows=250_000):
        frame = self._read(columns)
        step = max(int(batch_rows), 1)
        for lo in range(0, len(frame), step):
            yield frame.iloc[lo:min(lo + step, len(frame))]

    def materialize(self, columns=None, *, batch_rows=250_000):
        return self._read(columns).reset_index(drop=True)


def as_data_source(data) -> DataSource:
    if isinstance(data, DataSource):
        return data
    if isinstance(data, pd.DataFrame):
        return DataFrameSource(data)
    if isinstance(data, (str, Path)):
        path = Path(data)
        suffix = path.suffix.lower()
        if suffix in {".csv", ".txt"}:
            return CSVSource(path)
        if suffix == ".dta":
            return StataSource(path)
        if suffix in {".parquet", ".pq"}:
            return ParquetSource(path)
        raise SpecificationError(
            f"unsupported data source extension {suffix!r}",
            code="data.source_type", stage="data",
            details={"path": str(path), "suffix": suffix},
            suggestion="Use a pandas DataFrame, CSV, Stata .dta, or Parquet source.",
        )
    raise TypeError("data source must be a pandas DataFrame, DataSource, or supported file path")
