"""Run with the isolated engine Python: python -m mesh_landmarks.test_contract."""
import ast
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from .inference import validate_geometry
from .portable_ops import fps,DropPath
from .skeleton import NAMES,tree

class ContractTests(unittest.TestCase):
    def test_no_blender_in_prediction(self):
        source=Path(__file__).with_name('inference.py').read_text()
        for node in ast.walk(ast.parse(source)):
            if isinstance(node,ast.Import):self.assertNotIn('bpy',[x.name for x in node.names])
            if isinstance(node,ast.ImportFrom):self.assertNotEqual(node.module,'bpy')
        self.assertNotIn('validation-only',source)
    def test_fixed_topology_without_positions(self):
        root=tree();self.assertEqual(len(root),52);self.assertEqual(len(set(NAMES)),52)
        self.assertEqual(set(root.joints_list[0].__dataclass_fields__),{'name','index','parent','children','template_joints'})
    def test_fps_matches_torch_reference(self):
        p=torch.randn(127,3,generator=torch.Generator().manual_seed(98));batch=torch.zeros(127,dtype=torch.long)
        result=fps(p,batch,.1,False).tolist();expected=[];d=torch.full((127,),float('inf'));i=0
        for _ in range(13):
            expected.append(i);d=torch.minimum(d,((p-p[i])**2).sum(1));i=int(d.argmax())
        self.assertEqual(result,expected)
    def test_dropout_only_for_inference(self):
        op=DropPath(.1);x=torch.randn(2,3)
        with self.assertRaises(RuntimeError):op(x)
        op.eval();self.assertTrue(torch.equal(op(x),x))
    def test_invalid_geometry(self):
        for v,f in [(np.zeros((1,3)),np.array([[0,0,0]])),(np.full((100,3),np.nan),np.array([[0,1,2]])),(np.ones((100,3)),np.array([[0,1,999]]))]:
            with self.assertRaises(ValueError):validate_geometry(v,f)
    def test_cancelled_worker_lock_recovery(self):
        # Zero PID is deliberately not used: it denotes a process group on POSIX.
        from .worker import claim_lock
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'worker.lock';claim_lock(p)
            with self.assertRaises(RuntimeError):claim_lock(p)

if __name__=='__main__':unittest.main()
