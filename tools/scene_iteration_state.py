"""Single-file atomic iteration state for exact same-stage solver continuation."""
from io import BytesIO
import hashlib
import json
from pathlib import Path
import numpy as np
from tools.experiment_stages import atomic_write,canonical


class IterationCheckpoint:
    def __init__(self,path,identity):
        self.path=Path(path)
        self.identity=json.loads(canonical(identity))

    @staticmethod
    def _digest(metadata,x,y):
        h=hashlib.sha256(metadata.encode())
        h.update(x.tobytes(order='C'));h.update(y.tobytes(order='C'))
        return h.hexdigest()

    def save(self,iteration,x,y,elapsed_s):
        x,y=np.asarray(x,dtype=np.float64),np.asarray(y,dtype=np.float64)
        if int(iteration)!=iteration or iteration<0 or not np.isfinite(elapsed_s) or elapsed_s<0:
            raise ValueError('invalid checkpoint iteration/time')
        if x.shape!=y.shape or not np.isfinite(x).all() or not np.isfinite(y).all() or np.any(x<0):
            raise ValueError('invalid checkpoint iterates')
        metadata=canonical({'schema':1,'identity':self.identity,'iteration':int(iteration),'elapsed_s':float(elapsed_s)})
        stream=BytesIO()
        np.savez_compressed(stream,metadata=metadata,x=x,y=y,sha256=self._digest(metadata,x,y))
        self.path.parent.mkdir(parents=True,exist_ok=True)
        atomic_write(self.path,stream.getvalue())

    def load(self):
        if not self.path.exists():return None
        with np.load(self.path,allow_pickle=False) as z:
            metadata=str(z['metadata']);x=z['x'].copy();y=z['y'].copy();checksum=str(z['sha256'])
        if x.dtype!=np.float64 or y.dtype!=np.float64 or self._digest(metadata,x,y)!=checksum:
            raise ValueError('corrupt iteration checkpoint')
        record=json.loads(metadata)
        if record.get('schema')!=1 or record['identity']!=self.identity:
            raise ValueError('iteration checkpoint identity mismatch')
        n,elapsed=record['iteration'],record['elapsed_s']
        if not isinstance(n,int) or n<0 or not np.isfinite(elapsed) or elapsed<0 or x.shape!=y.shape or not np.isfinite(x).all() or not np.isfinite(y).all() or np.any(x<0):
            raise ValueError('invalid iteration checkpoint')
        return {'iteration':n,'x':x,'y':y,'elapsed_s':elapsed}
