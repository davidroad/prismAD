"""Read only the permitted development roster from controlled CSV inputs."""
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.csv as pcsv
import pyarrow.dataset as ds
from .runtime import input_path,settings,sha
FOLDS=input_path('folds')
BASELINE=input_path('baseline')
COGNITION=input_path('cognition')
FOLD_HASH=settings().get('expected_hashes',{}).get('folds')
def filtered_csv(path, columns, key, permitted):
    """Filter at Arrow query boundary; do not claim mixed-file byte isolation."""
    # Arrow6 applies scanner projection itself; repeating include_columns in
    # ConvertOptions creates duplicate field names in its dataset scanner.
    fmt = ds.CsvFileFormat(convert_options=pcsv.ConvertOptions(
        column_types={c:pa.string() for c in columns},
        strings_can_be_null=True))
    table = ds.dataset(str(path), format=fmt).to_table(
        columns=columns, filter=ds.field(key).isin([str(v) for v in permitted]),
        use_threads=False)
    out = table.to_pandas()
    if not set(out[key].astype(str)).issubset(set(str(v) for v in permitted)):
        raise AssertionError('Query returned a record outside the whitelist')
    return out

def development_folds(path=FOLDS):
    out = filtered_csv(path, ['subject_id','fold'], 'fold', range(1,10))
    out['fold'] = pd.to_numeric(out.fold, errors='raise').astype(int)
    if out.subject_id.duplicated().any() or not out.fold.isin(range(1,10)).all():
        raise AssertionError('Invalid development subject allocation')
    return out
