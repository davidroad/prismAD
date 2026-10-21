"""Local input configuration; input files are never downloaded automatically."""
from pathlib import Path
import os,json,hashlib
def config():
    value=os.environ.get('PRISM_AD_CONFIG')
    if not value: raise RuntimeError('Supply --config with locally authorized input paths')
    return Path(value).resolve()
def settings():return json.loads(config().read_text())
def input_path(name):
    value=settings()['inputs'][name]
    path=Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (config().parent/path).resolve()
def workspace():
    return Path(os.environ['PRISM_AD_WORKSPACE']).resolve()
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def source_data():
    return Path(os.environ.get('PRISM_AD_SOURCE_DATA','source_data')).resolve()
