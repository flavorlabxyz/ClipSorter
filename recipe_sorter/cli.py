import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(prog='recipe-sorter')
    parser.add_argument('--data-dir', type=Path, help='Application-owned directory, separate from sources')
    sub = parser.add_subparsers(dest='command', required=True)
    serve = sub.add_parser('serve')
    serve.add_argument('--port', type=int, default=8765)
    serve.add_argument('--demo', action='store_true')
    sub.add_parser('diagnostics')
    sub.add_parser('prepare-model')
    scan = sub.add_parser('scan')
    scan.add_argument('folder', type=Path)
    scan.add_argument('--no-recursive', action='store_true')
    analyze = sub.add_parser('analyze')
    analyze.add_argument('--backend', choices=['clip', 'features', 'hybrid'], default='features')
    analyze.add_argument('--force', action='store_true')
    analyze.add_argument('--sampling', choices=['uniform','adaptive'], default='uniform')
    export = sub.add_parser('benchmark-export')
    export.add_argument('--prepare', action='store_true')
    export.add_argument('--backends', nargs='+', choices=['clip','features'], default=['clip','features'])
    export.add_argument('--seed', type=int, default=17)
    export.add_argument('--output', default='benchmarks/labels.json')
    bench = sub.add_parser('benchmark')
    bench.add_argument('--manifest', type=Path)
    bench.add_argument('--synthetic', action='store_true')
    bench.add_argument('--output', default='benchmarks/results.json')
    bench.add_argument('--target-precision', type=float, default=.95)
    accept = sub.add_parser('braw-acceptance')
    accept.add_argument('source', type=Path)
    accept.add_argument('--backend', choices=['clip', 'features'], default='clip')
    args = parser.parse_args()
    from .storage.cache import Cache
    root = args.data_dir
    if args.command == 'serve' and args.demo:
        # Always a distinct subtree so mock and real libraries cannot mix.
        root = (root or Path(os.environ.get('RCS_DATA_DIR', Path(os.environ.get('XDG_DATA_HOME', Path.home()/'.local/share'))/'recipe-clip-sorter'))) / 'demo-library'
    cache = Cache(root)
    if args.command == 'serve':
        import uvicorn
        from .backend.app import create_app
        print(f'Open http://127.0.0.1:{args.port} — data: {cache.root}', flush=True)
        uvicorn.run(create_app(cache.root, args.demo), host='127.0.0.1', port=args.port)
        return
    if args.command == 'diagnostics':
        from .backend.diagnostics import diagnostic
        print(json.dumps(diagnostic(cache), indent=2))
        return
    if args.command == 'prepare-model':
        from .analysis.embeddings import prepare_model
        prepare_model(cache)
        print('CLIP installed. Subsequent analysis uses local files only.')
        return
    from .storage.database import Database
    from .backend.service import Service
    import fcntl
    with cache.output('app.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error('This library is open in another process. Stop the server before running CLI analysis.')
        db = Database(cache)
        service = Service(cache, db)
        service.recover_interrupted()
        if args.command == 'scan':
            service.scan(args.folder, not args.no_recursive)
        elif args.command == 'analyze':
            service.analyze(args.backend, args.force, args.sampling)
        elif args.command == 'benchmark-export':
            from .analysis.evaluation import export_labels
            report = export_labels(service, args.backends, args.prepare, args.seed)
            output = cache.output(args.output)
            output.write_text(json.dumps(report, indent=2))
            print(output)
            return
        elif args.command == 'benchmark':
            from .analysis.evaluation import benchmark, synthetic_fixture
            if args.synthetic == bool(args.manifest):
                parser.error('Choose exactly one of --synthetic or --manifest')
            if not 0 < args.target_precision <= 1:
                parser.error('--target-precision must be between 0 and 1')
            data = synthetic_fixture() if args.synthetic else json.loads(args.manifest.read_text())
            report = benchmark(data, args.target_precision)
            output = cache.output(args.output)
            output.write_text(json.dumps(report, indent=2))
            print(json.dumps(report, indent=2))
            print(f'Report: {output}')
            return
        elif args.command == 'braw-acceptance':
            from .acceptance import run_acceptance
            result = run_acceptance(args.source, service, args.backend)
            print(json.dumps(result, indent=2))
            if not result['automated_checks_passed']:
                raise SystemExit(1)
            return
        print(json.dumps(service.snapshot(), indent=2))
        if service.snapshot()['errors']:
            raise SystemExit(1)


if __name__ == '__main__':
    main()
