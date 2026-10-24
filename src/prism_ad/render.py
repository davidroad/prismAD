"""Regenerate the current manuscript figures from the companion source data."""
from pathlib import Path
import os,json,platform,hashlib,importlib.metadata,time
def _sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def render(figure,source,output):
    started=time.monotonic()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    os.environ['PRISM_AD_SOURCE_DATA']=str(Path(source).resolve())
    out=Path(output).resolve()
    if out.exists()and any(out.iterdir()):raise RuntimeError('Choose an empty output directory')
    out.mkdir(parents=True,exist_ok=True)
    os.environ['PRISM_AD_RENDER_WORK']=str(out)
    from .figures_current.build import build
    fig,geometry=build(figure)
    assert not geometry.get('text_bbox_collisions')and not geometry.get('out_of_canvas'),geometry
    stem='SupplementaryFigure'+figure[1:]if figure.startswith('S')else'Figure'+figure
    for ext in ['pdf','svg','png']:
        meta={'CreationDate':None,'ModDate':None}if ext=='pdf'else{'Date':None}if ext=='svg'else{}
        if figure=='S3':
            fig.savefig(out/(stem+'.'+ext),dpi=600,metadata=meta)
        else:
            fig.savefig(out/(stem+'.'+ext),dpi=600,bbox_inches=None,pad_inches=0,metadata=meta)
    plt.close(fig)
    from PIL import Image
    image=Image.open(out/(stem+'.png'));image.thumbnail((1450,1900));image.save(out/(stem+'_preview.png'))
    versions={}
    for name in ['numpy','pandas','matplotlib','Pillow']:
        versions[name]=importlib.metadata.version(name)
    record=dict(figure=stem,python=platform.python_version(),versions=versions,
        elapsed_seconds=time.monotonic()-started,geometry=geometry,
        source_sha256={str(p.relative_to(source)):_sha(p)for sub in ['plotting','optimization','tables']for p in sorted((Path(source)/sub).glob('*'))if p.is_file()},
        code_sha256={str(p.relative_to(Path(__file__).parent)):_sha(p)for p in sorted(Path(__file__).parent.rglob('*.py'))},
        output_sha256={p.name:_sha(p)for p in sorted(out.iterdir())if p.suffix in ['.pdf','.svg','.png']},
        scope='Authorized private plotting inputs to publication figures. No model fitting or inferential testing in this rendering step; empirical inputs are not distributed.')
    (out/'execution.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(dict(figure=stem,output=str(out),geometry=geometry)))
