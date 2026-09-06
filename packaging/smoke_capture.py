"""Offline frozen-native AVI and geometry/layer export functional checks."""
from pathlib import Path
import os
import struct
import subprocess
import sys
import tempfile

import numpy as np
import tifffile

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from planetrecon.io.ser import write_ser
from planetrecon.result import load_snapshot


def raw_avi(path,frames):
    n,h,w,c=frames.shape
    def chunk(tag,data):return tag+struct.pack('<I',len(data))+data+b'\0'*(len(data)%2)
    sh=bytearray(56);sh[:8]=b'vidsDIB ';struct.pack_into('<4I',sh,20,1,25,0,n)
    fmt=struct.pack('<IiiHHIIiiII',40,w,h,1,24,0,0,0,0,0,0)
    header=chunk(b'LIST',b'hdrl'+chunk(b'LIST',b'strl'+chunk(b'strh',bytes(sh))+chunk(b'strf',fmt)))
    movie=b''.join(chunk(b'00db',b''.join(row.tobytes()+b'\0'*((-3*w)%4) for row in frame[::-1,:,::-1])) for frame in frames)
    path.write_bytes(chunk(b'RIFF',b'AVI '+header+chunk(b'LIST',b'movi'+movie)))


def main(executable):
    executable=str(Path(executable).resolve())
    env={**os.environ,'PATH':'/nonexistent','QT_QPA_PLATFORM':'offscreen','PLANETRECON_THREADS':'2','CUDA_VISIBLE_DEVICES':''}
    with tempfile.TemporaryDirectory(prefix='planetrecon-native-smoke-') as temp:
        root=Path(temp);y,x=np.indices((64,96))
        mono=(10+150*np.exp(-((x-47.5)**2+(y-31.5)**2)/400)).astype('u1')
        rgb=np.stack([mono,mono//2,mono//3],axis=-1)
        raw_avi(root/'rgb.avi',np.stack([rgb]*3))
        subprocess.run([executable,'--threads','2','stack','--path',str(root/'rgb.avi'),
            '--out',str(root/'avi'),'--device','cpu'],env=env,check=True,timeout=60)
        result=load_snapshot(root/'avi/stack.npz')
        np.testing.assert_allclose(result.image,rgb,atol=1e-10)
        assert result.units=='decoded-code-value'
        source=write_ser(root/'scene.ser',np.stack([mono]*3))
        for mode in ('combined','saturn'):
            args=[executable,'--threads','2','stack','--path',str(source),'--out',str(root/mode),
                '--device','cpu','--geometry',mode,'--cadence','1','--field-rate-deg-s','0',
                '--surface-rate-deg-s','0','--radius','15','--center-x','47.5','--center-y','31.5',
                '--sub-obs-lat-deg','30','--export','tiff32']
            if mode=='saturn':args+=['--ring-inner','22','--ring-outer','35']
            subprocess.run(args,env=env,check=True,timeout=60)
            result=load_snapshot(root/mode/'stack.npz')
            assert result.n_used==3 and result.validity.any()
            if mode=='saturn':
                assert set(result.layer_coverage)=={'globe','ring'}
                assert all(value.max()>0 for value in result.layer_coverage.values())
                with tifffile.TiffFile(root/mode/'stack.tif') as file:assert len(file.pages)==5
    print('PASS: frozen native AVI without external executables; combined/Saturn CPU and layered TIFF')


if __name__=='__main__':main(sys.argv[1])
