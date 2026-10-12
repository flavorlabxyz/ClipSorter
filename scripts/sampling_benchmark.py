#!/usr/bin/env python3
"""Compare bounded sampling on one real clip, without creating library records.
All output/log/model/temporary paths use --data-dir. Original media is read-only.
This measures sampling cost and retained coverage, not grouping accuracy.
"""
import argparse
import hashlib
import json
from pathlib import Path
import resource
import sys
import time
import tracemalloc
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from recipe_sorter.storage.cache import Cache
from recipe_sorter.media.decoders.registry import decoder_for
from recipe_sorter.analysis.frame_sampling import sample_frames, adaptive_frames, diverse_frames


def signature(path):
    stat=path.stat();digest=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):digest.update(block)
    return {'sha256':digest.hexdigest(),'size':stat.st_size,'mtime_ns':stat.st_mtime_ns}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source',type=Path);parser.add_argument('--data-dir',type=Path,required=True)
    args=parser.parse_args();source=args.source.resolve();cache=Cache(args.data_dir);cache.check_source(source)
    before=signature(source);decoder=decoder_for(source,cache);meta=decoder.probe(source);results=[]
    class Counted:
        calls=0
        def frame(self,*args,**kwargs):self.calls+=1;return decoder.frame(*args,**kwargs)
    for name,sampler in [('uniform',sample_frames),('adaptive',adaptive_frames),('diverse',diverse_frames)]:
        counted=Counted();tracemalloc.start();started=time.perf_counter();frames=sampler(counted,source,meta)
        elapsed=time.perf_counter()-started;_,peak=tracemalloc.get_traced_memory();tracemalloc.stop()
        results.append({'sampling':name,'retained_frames':len(frames),'decode_calls':counted.calls,'seconds':[s[1] for s in frames],
            'elapsed_seconds':elapsed,'peak_python_allocated_bytes':peak,'process_peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})
        del frames
    after=signature(source)
    report={'source_name':source.name,'decoder':decoder.name,'source_unchanged':before==after,'before':before,'after':after,'results':results,
        'scope':'Independent cold sampling passes; image quality/coverage require human review. No inference or recipe accuracy measured. RSS is cumulative; Python allocations exclude some native memory.'}
    output=cache.output('benchmarks/sampling.json');output.write_text(json.dumps(report,indent=2));print(output)
    if before!=after:raise RuntimeError('Source changed during benchmark')

if __name__=='__main__':main()
