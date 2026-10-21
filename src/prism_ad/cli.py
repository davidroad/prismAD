"""Command-line entry points for controlled analysis and aggregate figure redraws."""
from pathlib import Path
import argparse,os,subprocess,sys,json,hashlib
def main():
    p=argparse.ArgumentParser(prog='prism-ad')
    sub=p.add_subparsers(dest='command',required=True)
    a=sub.add_parser('demo');a.add_argument('--with-model',action='store_true')
    a=sub.add_parser('render');a.add_argument('--figure',choices=['1','2','3','4','5','S1','S2','S3','all'],default='all')
    a.add_argument('--source-data',type=Path,required=True);a.add_argument('--output',type=Path,required=True)
    a=sub.add_parser('verify-source-data');a.add_argument('--data',type=Path,required=True)
    for name in ['features','groups','train','metrics','paired-inference','genetic-comparison','interpret','education','pet','export','profile-replay','profile-export','current-statistics','cognitive-strata','learning-curve-replay']:
        a=sub.add_parser(name);a.add_argument('--config',type=Path,required=True);a.add_argument('--output',type=Path,required=True)
        if name=='train':
            a.add_argument('--task',choices=['CN_MCI','MCI_AD']);a.add_argument('--preflight',action='store_true')
            a.add_argument('--without-prs',action='store_true');a.add_argument('--features',type=Path)
        if name=='metrics':a.add_argument('--bootstrap',action='store_true')
    args=p.parse_args()
    if args.command=='demo':
        from .demo import run
        run(args.with_model);return
    if args.command=='verify-source-data':
        root=args.data.resolve();count=0
        for line in (root/'SHA256SUMS.txt').read_text().splitlines():
            expected,rel=line.split('  ',1);path=root/rel
            assert not Path(rel).is_absolute() and root in path.resolve().parents,rel
            assert path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest()==expected,rel
            count+=1
        print(json.dumps(dict(verified_files=count)));return
    if args.command=='render':
        if args.figure=='all':
            for name in ['1','2','3','4','5','S1','S2','S3']:
                subprocess.run([sys.executable,'-m','prism_ad','render','--figure',name,
                    '--source-data',str(args.source_data.resolve()),'--output',str(args.output.resolve()/name)],check=True)
        else:
            from .render import render
            render(args.figure,args.source_data.resolve(),args.output)
        return
    os.environ['PRISM_AD_CONFIG']=str(args.config.resolve())
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    os.environ['PRISM_AD_WORKSPACE']=str(out)
    target={'features':'build_features','groups':'groups','train':'training','metrics':'metrics',
            'paired-inference':'paired_inference','genetic-comparison':'genetic_metrics',
            'interpret':'interpret','education':'education','pet':'pet','export':'export_metrics','profile-replay':'profile_replay',
            'profile-export':'profile_export','current-statistics':'current_statistics',
            'cognitive-strata':'cognitive_strata','learning-curve-replay':'learning_curve_replay'}[args.command]
    argv=[sys.executable,'-m','prism_ad.'+target]
    if args.command=='train':
        if not args.preflight and not args.task:p.error('train requires --task or --preflight')
        os.environ['PRISM_AD_INCLUDE_PRS']='0' if args.without_prs else '1'
        if args.features:os.environ['PRISM_AD_FEATURES']=str(args.features.resolve())
        argv+=['--preflight'] if args.preflight else ['--task',args.task]
    elif args.command=='metrics' and args.bootstrap:argv+=['--bootstrap']
    subprocess.run(argv,check=True)
if __name__=='__main__':main()
