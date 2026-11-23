"""Separate fused hand/body surfaces without discarding the finger surface.

A straight torso slit cannot follow fingers curled inward against a garment.
Classify those surfaces using skeletal seeds and surface creases, duplicate their boundary
loops, and cap each bank independently. Existing surface coordinates and UVs
are retained; only the hidden closure geometry is added.
"""
import numpy as np


def _surface_owners(vertices, faces, cuts, prediction, runtime_root):
    import subprocess, tempfile
    from pathlib import Path
    from .jobs import runtime_python
    import bpy
    if runtime_root is None:
        runtime_root=Path(bpy.utils.user_resource('DATAFILES'))/'hallway_rig'/'mesh_landmarks'
    python=runtime_python(runtime_root)
    if not python.is_file():
        raise RuntimeError('Prepare the mesh rigging engine before separating fused hands')
    import json
    with tempfile.TemporaryDirectory(prefix='hallway-contact-') as directory:
        directory=Path(directory)
        np.savez(directory/'input.npz',vertices=vertices,faces=faces,
                 request=json.dumps(dict(cuts=cuts,prediction=prediction)))
        result=subprocess.run([str(python),'-I',str(Path(__file__).with_name('contact_graph.py')),
                               str(directory/'input.npz'),str(directory/'owners.npy')],
                              capture_output=True,text=True,timeout=90)
        if result.returncode:
            raise RuntimeError('Fused-hand surface separation failed: '+result.stderr[-2000:])
        owners=np.load(directory/'owners.npy',allow_pickle=False)
    if owners.shape!=(len(faces),) or not np.isin(owners,[-1,0,1]).all():
        raise RuntimeError('Fused-hand separation returned invalid face ownership')
    return owners


def separate(obj, cuts, prediction, height, runtime_root=None):
    import bmesh
    bm=bmesh.new()
    try:
        bm.from_mesh(obj.data)
        return _separate_mesh(obj,bm,cuts,prediction,height,runtime_root)
    finally:
        bm.free()


def _separate_mesh(obj,bm,cuts,prediction,height,runtime_root):
    import bmesh
    bmesh.ops.triangulate(bm,faces=[f for f in bm.faces if len(f.verts)>3])
    bm.verts.ensure_lookup_table();bm.faces.ensure_lookup_table()
    bm.verts.index_update();bm.faces.index_update()
    points=np.array([obj.matrix_world@v.co for v in bm.verts])
    faces=np.array([[v.index for v in f.verts] for f in bm.faces])
    owners=_surface_owners(points,faces,cuts,prediction,runtime_root)
    face_bank=bm.faces.layers.int.new('_hallway_contact_face')
    vertex_bank=bm.verts.layers.int.get('hallway_arm_bank') or bm.verts.layers.int.new('hallway_arm_bank')
    fill=bm.verts.layers.int.get('hallway_generated_fill') or bm.verts.layers.int.new('hallway_generated_fill')
    for face in bm.faces:
        face[face_bank]=int(owners[face.index])
    boundary=[edge for edge in bm.edges if len({f[face_bank] for f in edge.link_faces})>1]
    if not boundary:
        raise RuntimeError('Could not establish separate hand and garment surfaces')
    bmesh.ops.split_edges(bm,edges=boundary)
    for vertex in bm.verts:
        labels={face[face_bank] for face in vertex.link_faces}
        if len(labels)!=1:
            raise RuntimeError('Hand/body boundary still shares a surface vertex')
        vertex[vertex_bank]=next(iter(labels))
    # Nonplanar contact loops cannot reliably be filled as one n-gon.
    # Build concentric triangular caps, preserving every boundary position.
    # This also supplies skinning samples across the reconstructed interior.
    remaining={e for e in bm.edges if e.is_boundary}
    loops=[]
    while remaining:
        edge=next(iter(remaining));remaining.remove(edge)
        loop=[edge.verts[0],edge.verts[1]];first_edge=edge
        while loop[-1]!=loop[0]:
            choices=[e for e in loop[-1].link_edges if e in remaining]
            if len(choices)!=1:
                raise RuntimeError('Hand contact boundary branches; refine its surface ownership')
            edge=choices[0];remaining.remove(edge);loop.append(edge.other_vert(loop[-1]))
        loop.pop()
        # The cap must traverse each rim edge opposite its existing skin face.
        skin_loop=first_edge.link_loops[0]
        if skin_loop.vert==loop[0] and skin_loop.link_loop_next.vert==loop[1]:loop.reverse()
        loops.append(loop)
    for loop in loops:
        side=loop[0][vertex_bank]
        center=sum((v.co for v in loop),loop[0].co*0.)/len(loop)
        radius=max((v.co-center).length for v in loop)
        rings=int(np.clip(np.ceil(radius/(height*.012/max(obj.matrix_world.to_scale()))),2,16))
        original=list(loop);outer=list(loop)
        for k in range(1,rings):
            inner=[]
            for vertex in original:
                new=bm.verts.new(vertex.co.lerp(center,k/rings));new[vertex_bank]=side;new[fill]=1;inner.append(new)
            for i in range(len(loop)):
                j=(i+1)%len(loop)
                for verts in ((outer[i],outer[j],inner[j]),(outer[i],inner[j],inner[i])):
                    face=bm.faces.new(verts);face[face_bank]=side
            outer=inner
        tip=bm.verts.new(center);tip[vertex_bank]=side;tip[fill]=1
        for i in range(len(loop)):
            face=bm.faces.new((outer[i],outer[(i+1)%len(loop)],tip));face[face_bank]=side
    boundary_count=sum(e.is_boundary for e in bm.edges)
    nonmanifold=sum(not e.is_manifold for e in bm.edges)
    if boundary_count or nonmanifold:
        raise RuntimeError(f'Hand partition left {boundary_count} open and {nonmanifold} nonmanifold edges')
    if any(len({v[vertex_bank] for v in f.verts})>1 for f in bm.faces):
        raise RuntimeError('A generated cap reconnects a hand to the garment')
    bm.normal_update()
    bm.faces.layers.int.remove(face_bank)
    bm.to_mesh(obj.data);obj.data.update()
    points=np.array([obj.matrix_world@v.co for v in obj.data.vertices])
    strength=np.zeros(len(points),dtype=np.float32)
    join=np.zeros(len(points),dtype=np.float32)
    for cut in cuts:
        sign=1 if cut['side']=='Left' else -1
        side=(points[:,0]-prediction['joints']['Hips']['head'][0])*sign>=0
        t=np.clip((cut['curve'][0][1]-points[:,2])/(height*.035),0,1)
        strength[side]=t[side]*t[side]*(3-2*t[side])
        upper=prediction['joints'][cut['side']+'Arm']
        length=np.linalg.norm(np.asarray(upper['tail'])-upper['head'])
        # Only the short socket rim remains joined. Sharing the full upper
        # arm seam here recreates the fused underarm during deformation.
        reach=max(length*.12,height*.002)
        t=np.clip((cut['curve'][0][1]-points[:,2]-length*.08)/reach,0,1)
        join[side]=1-t[side]*t[side]*(3-2*t[side])
    attribute=obj.data.attributes.get('hallway_arm_bank_strength') or obj.data.attributes.new(name='hallway_arm_bank_strength',type='FLOAT',domain='POINT')
    attribute.data.foreach_set('value',strength)
    attribute=obj.data.attributes.get('hallway_arm_join') or obj.data.attributes.new(name='hallway_arm_join',type='FLOAT',domain='POINT')
    attribute.data.foreach_set('value',join)
    return dict(method='closed_surface_partition',ownership='concavity_face_cut',split_boundary_edges=len(boundary),
                boundary_edges=boundary_count,nonmanifold_edges=nonmanifold,
                original_surface_positions_retained=True)
