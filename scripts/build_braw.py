#!/usr/bin/env python3
"""Build against locally acquired official headers/dispatch, never download SDK."""
import argparse
import os
from pathlib import Path
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument('--sdk', type=Path, required=True, help='Official SDK directory containing Linux/Include')
parser.add_argument('--output', type=Path, default=Path('native/recipe-braw'))
args = parser.parse_args()
headers = list(args.sdk.expanduser().rglob('BlackmagicRawAPI.h'))
headers = [p for p in headers if 'Linux' in p.parts]
if len(headers) != 1:
    parser.error('Expected one Linux/Include/BlackmagicRawAPI.h beneath --sdk')
include = headers[0].parent
sources = list(args.sdk.expanduser().rglob('BlackmagicRawAPIDispatch.cpp'))
sources = [p for p in sources if 'Linux' in p.parts]
if len(sources) != 1:
    parser.error('Expected one official Linux BlackmagicRawAPIDispatch.cpp')
output = args.output.expanduser().resolve()
output.parent.mkdir(parents=True, exist_ok=True)
flags = ['-DRCS_BRAW_READ_AUDIO'] if 'ReadAudioComplete' in headers[0].read_text() else []
subprocess.run([os.environ.get('CXX', 'g++'), '-std=c++17', '-O2', '-Wall', '-Wextra', *flags,
                '-I', str(include), str(Path(__file__).resolve().parents[1] / 'native/bridge.cpp'),
                str(sources[0]), '-ldl', '-pthread', '-o', str(output)], check=True)
print(f'Built {output}. Set RCS_BRAW_BRIDGE={output}')
print('Compilation is not verification. Run the Fedora real BRAW acceptance test.')
