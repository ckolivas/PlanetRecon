"""Create the project-local development/packaging environment from version locks."""
import argparse
from pathlib import Path
import subprocess
import sys
import venv

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu',action='store_true',help='install CUDA 13.2 Torch for Blackwell GPUs')
    args=parser.parse_args()
    if sys.version_info[:2]!=(3,13) or sys.platform!='linux':
        parser.error('these locks qualify Linux Python 3.13; other platforms need separate locks')
    destination=ROOT/'.venv'
    if not (destination/'pyvenv.cfg').exists():
        venv.EnvBuilder(with_pip=True).create(destination)
    python=destination/'bin/python'
    subprocess.run([str(python),'-m','pip','install','-r',str(ROOT/'requirements/venv-linux.lock')],check=True)
    if args.gpu:
        subprocess.run([str(python),'-m','pip','install','-r',str(ROOT/'requirements/gpu-cu132.lock')],check=True)
    subprocess.run([str(python),'-m','pip','install','--no-deps','-e',str(ROOT)],check=True)
    subprocess.run([str(python),'-m','pip','check'],check=True)
    print(f'Ready: {python}; activate with source .venv/bin/activate')


if __name__=='__main__':main()
