"""Independent per-thread map keys with OS entropy and explicit control injection.

Native Mac acquisition uses CCRandomGenerateBytes(16). os.urandom supplies the
portable OS-entropy contract here; this does not qualify native Windows behavior.
No original binary or captured state is read by this module.
"""
import os
import struct
import threading
MASK=(1<<64)-1

class RandomStates:
    def __init__(self,entropy=None):
        self._entropy=os.urandom if entropy is None else entropy
        self._local=threading.local()
    def new(self):
        state=getattr(self._local,'state',None)
        if state is None:
            raw=self._entropy(16)
            if not isinstance(raw,bytes) or len(raw)!=16:
                raise RuntimeError('OS entropy provider must return exactly 16 bytes')
            state=struct.unpack('<QQ',raw)
        self._local.state=((state[0]+1)&MASK,state[1])
        return state

DEFAULT_STATES=RandomStates()
