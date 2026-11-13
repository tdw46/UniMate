"""Bounded recovered integer-key hash table; caller supplies initial random state.

Process entropy acquisition remains an explicit boundary. This module has no
original binary dependency and reconstructs bucket order rather than reading it.
"""
import sys
M=(1<<64)-1
def rot(x,n):return ((x<<n)|(x>>(64-n)))&M

def _hash_key_python(b,k0,k1):
    # Expanded native integer-key hash, 0x10fa48. Every sum wraps at 64 bits.
    a=k1^0x646f72616e646f6d;c=k1^0x7065646279746573^int(b)
    d=(a+(k0^0x736f6d6570736575))&M;e=d^rot(a,13)
    a=(c+(k0^0x6c7967656e657261))&M;f=(e+a)&M
    a^=rot(c,16);e=f^rot(e,17);d=(a+rot(d,32))&M;c=d^rot(a,21)
    d=((d^(int(b)|(4<<56)))+e)&M;e=d^rot(e,13)
    a=((rot(f,32)^255)+c)&M;f=(e+a)&M;a^=rot(c,16);e=f^rot(e,17)
    d=(a+rot(d,32))&M;c=(e+d)&M;d^=rot(a,21);e=c^rot(e,13)
    a=(d+rot(f,32))&M;f=(e+a)&M;a^=rot(d,16);e=f^rot(e,17)
    d=(a+rot(c,32))&M;c=d^rot(a,21);e=((e+d)&M)^rot(e,13)
    d=(c+rot(f,32))&M;a=(e+d)&M;d^=rot(c,16)
    return (rot(e,17)^rot(d,21)^rot(a,32)^a)&M

def hash_key(b,k0,k1):
    # Public runtime loading verifies the binary; older helpers lack this symbol.
    native=getattr(sys.modules.get(__package__+'._bvt_solver'),'weight_hash_u32',None)
    b=int(b)
    if native is not None and 0<=b<=0xffffffff and 0<=k0<=M and 0<=k1<=M:
        return native(b,k0,k1)
    return _hash_key_python(b,k0,k1)

class Table:
    def __init__(self,k0,k1):self.keys=(k0,k1);self.slots=[];self.count=0;self._hashes={}
    def items(self):return [p for p in self.slots if p is not None]
    def resize(self,size):
        old=self.items();self.slots=[None]*size;self.count=0
        for b,w in old:self.set(b,w)
    def set(self,b,value):
        if not self.slots:self.resize(4)
        size=len(self.slots);mask=size-1
        hashed=self._hashes.get(b)
        if hashed is None:
            hashed=hash_key(b,*self.keys);self._hashes[b]=hashed
        start=hashed&mask;step=0
        while True:
            group=[(start+i)&mask for i in range(min(size,8))]
            for i in group:
                if self.slots[i] is not None and self.slots[i][0]==b:self.slots[i]=(b,value);return
            free=next((i for i in group if self.slots[i] is None),None)
            if free is not None:break
            step+=8;start=(start+step)&mask
        capacity=size-1 if size<8 else size*7//8
        if self.count>=capacity:self.resize(size*2);self.set(b,value);return
        self.slots[free]=(b,value);self.count+=1
    def prune(self):
        self.slots=[p if p is not None and p[1]>0 else None for p in self.slots];self.count=len(self.items())
        self._hashes={p[0]:self._hashes[p[0]] for p in self.slots if p is not None}
