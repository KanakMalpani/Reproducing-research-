"""Isolated inference environment matching the frozen Prithvi audit."""
import json
import os
from pathlib import Path
import subprocess
import sys

A = Path(__file__).resolve().parent
R = A / 'results'
assert json.loads((R / 'c2sms_floods_download_manifest.json').read_text())['complete']
env = A / '.venv-prithvi-c2sms'
assert not env.exists(), 'STOP inference environment already exists'
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'uv'], check=True)
subprocess.run(['uv', 'python', 'install', '3.11'], check=True)
subprocess.run(['uv', 'venv', '--python', '3.11', str(env)], check=True)
python = str(env / 'bin' / 'python')
subprocess.run(['uv', 'pip', 'install', '--python', python, 'terratorch==0.99.8', 'rasterio', 'matplotlib'], check=True)
freeze = subprocess.check_output([python, '-m', 'pip', 'freeze'], text=True) if (env / 'bin' / 'pip').exists() else subprocess.check_output(['uv', 'pip', 'freeze', '--python', python], text=True)
(R / 'c2sms_prithvi_requirements.txt').write_text(freeze)
os.environ['MPLBACKEND'] = 'Agg'
subprocess.run([python, '-c', 'import torch,numpy,rasterio,terratorch; print("FROZEN C2SMS ENV READY",torch.__version__,numpy.__version__,torch.cuda.is_available(),flush=True); assert torch.cuda.is_available()'], check=True, env=os.environ.copy())
print('C2SMS INFERENCE PYTHON', python, flush=True)
