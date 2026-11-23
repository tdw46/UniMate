"""Open fused arm/torso bridges on generated copies, with closed cut surfaces.

A down-arm pose is not itself evidence of fusion. Horizontal surface sections
must remain connected across an interior arm/torso saddle below mid upper-arm.
Existing air gaps, raised arms and shape-key meshes are left alone. A wrist-level
fusion uses crease-aware surface ownership and separately capped banks. The
shoulder rim retains shared deformation. Source meshes are never
edited; callers own the disposable copies and rollback.
"""
import numpy as np


def _section(triangles, z):
    depth = triangles[:, :, 2]
    selected = triangles[(depth.min(1) <= z) & (depth.max(1) > z)]
    if not len(selected):
        return np.empty((0, 2, 3))
    above = selected[:, :, 2] > z
    odd = np.where(above.sum(1)[:, None] == 1, above, ~above).argmax(1)
    rows = np.arange(len(selected))[:, None]
    a = selected[rows, odd[:, None]]
    b = selected[rows, (odd[:, None] + [1, 2]) % 3]
    return a + (b-a)*((z-a[:, :, 2])/(b[:, :, 2]-a[:, :, 2]))[:, :, None]


def _thickness(segments, x):
    lo, hi = segments[:, :, 0].min(1), segments[:, :, 0].max(1)
    rows = segments[(lo <= x) & (hi > x)]
    if not len(rows):return 0.
    ys = np.sort(rows[:, 0, 1] + (rows[:, 1, 1]-rows[:, 0, 1]) *
                 (x-rows[:, 0, 0])/(rows[:, 1, 0]-rows[:, 0, 0]))
    # UV seam duplicates can produce repeated crossings. Collapse numerical
    # duplicates before applying the even/odd solid cross-section rule.
    ys = ys[np.r_[True, np.diff(ys) > 1e-8]]
    if len(ys) % 2:return float('nan')
    return float(np.sum(ys[1::2]-ys[::2]))


def plan(vertices, faces, prediction):
    v=np.asarray(vertices,dtype=float);tri=v[np.asarray(faces,dtype=int)]
    height=float(np.ptp(v[:,2]));j=prediction['joints'];cuts=[]
    origin=float(j['Hips']['head'][0]);v=v.copy();v[:,0]-=origin;tri=v[np.asarray(faces,dtype=int)]
    for side,sign in (('Left',1.),('Right',-1.)):
        shoulder=np.array(j[side+'Arm']['head']);elbow=np.array(j[side+'ForeArm']['head']);wrist=np.array(j[side+'Hand']['head'])
        for p in (shoulder,elbow,wrist):p[0]-=origin
        length=np.linalg.norm(elbow-shoulder)
        if shoulder[2]-elbow[2]<length*.65:continue
        hand_bottom=min(p['tail'][2] for n,p in j.items() if n.startswith(side+'Hand'))
        # A welded sleeve must release just below its socket, not a third
        # of an upper arm below it. The capped banks still share the root rim.
        levels=np.linspace(shoulder[2]-.12*length,hand_bottom-height*.012,42)
        rows=[]
        for z in levels:
            segments=_section(tri,z)
            if not len(segments):continue
            if z>=elbow[2]:a,b=shoulder,elbow
            else:a,b=elbow,wrist
            center=a+(b-a)*np.clip((z-a[2])/(b[2]-a[2]),0,1)
            mirrored=segments.copy();mirrored[:,:,0]*=sign
            outer=float(mirrored[:,:,0].max());cx=sign*center[0]
            radius=max(outer-cx,height*.012)
            low=max(cx*.55,cx-radius*1.5);high=cx-radius*.25
            if low>=high:continue
            xs=np.linspace(low,high,41);thickness=np.array([_thickness(mirrored,x) for x in xs])
            if not np.isfinite(thickness).all():continue
            k=int(np.argmin(thickness));width=float(thickness[k])
            rows.append(dict(z=float(z),x=float(xs[k]*sign),width=width,
                             interior=0<k<len(xs)-1))
        evidence=[r for r in rows if r['z']<shoulder[2]-.55*length and r['width']>height*.003 and r['interior']]
        if len(evidence)<2:continue
        # End at the established air gap for ordinary sleeve contacts. When
        # fusion reaches the wrist, also release the contact below the fingers.
        bottom=min(r['z'] for r in evidence)
        below=[r for r in rows if r['z']<bottom and r['width']<height*.001]
        stop=below[0]['z'] if below else levels[-1]
        selected=[r for r in rows if r['z']>=stop]
        if len(selected)<3:continue
        x=np.array([r['x'] for r in selected])
        # Suppress tessellation/trim noise in the saddle path; no vertex moves.
        for _ in range(3):x[1:-1]=(x[:-2]+2*x[1:-1]+x[2:])*.25
        curve=[(float(a)+origin,r['z']) for a,r in zip(x,selected)]
        hand_length=np.linalg.norm(np.array(j[side+'Hand']['tail'])-np.array(j[side+'Hand']['head']))
        fused_hand=bool(bottom<wrist[2]-hand_length*.10)
        if fused_hand and curve[-1][1]>levels[-1]:curve.append((curve[-1][0],float(levels[-1])))
        cuts.append(dict(side=side,curve=curve,hand_bottom=float(levels[-1]),release_hand=fused_hand,outer_x=float(v[:,0].max()+origin+height*.02) if sign>0 else float(v[:,0].min()+origin-height*.02),
                         gap=height*.0015,y_min=float(v[:,1].min()-height*.02),
                         y_max=float(v[:,1].max()+height*.02),evidence=len(evidence)))
    return cuts


def apply(meshes, prediction, runtime_root=None):
    """Cut and cap detected arm/body contacts on caller-owned mesh copies."""
    import bpy,bmesh
    from .blender_bridge import geometry
    vertices,faces=geometry(meshes)
    cuts=plan(vertices,faces,prediction)
    report=dict(applied=False,cuts=cuts,meshes=[])
    if not cuts:return report
    if any(o.data.shape_keys for o in meshes):
        report['skipped']='Shape keys require a correspondence-preserving topology workflow'
        return report
    active=bpy.context.view_layer.objects.active
    selected=list(bpy.context.selected_objects)
    try:
        for obj in meshes:
            world=np.array([obj.matrix_world @ v.co for v in obj.data.vertices])
            # Restrict surgery to a closed body spanning both shoulders. Open
            # hair cards, detached clothing and accessory meshes are not solids.
            shoulders=[prediction['joints'][s+'Arm']['head'] for s in ('Left','Right')]
            if not all(world[:,0].min()<p[0]<world[:,0].max() and world[:,2].min()<p[2]<world[:,2].max() for p in shoulders):
                continue
            before=len(obj.data.vertices)
            # glTF UV seams use duplicated position vertices. Welding here
            # preserves loop UVs and gives the exact Boolean a closed solid.
            bm=bmesh.new();bm.from_mesh(obj.data)
            bmesh.ops.remove_doubles(bm,verts=list(bm.verts),dist=np.ptp(vertices[:,2])*1e-7/max(obj.matrix_world.to_scale()))
            if any(not e.is_manifold for e in bm.edges):
                bm.free()
                report['meshes'].append(dict(name=obj.name,skipped='Open or nonmanifold source surface'))
                continue
            bm.to_mesh(obj.data);bm.free();obj.data.update()
            if any(cut.get('release_hand') for cut in cuts):
                from .contact_partition import separate
                audit=separate(obj,cuts,prediction,float(np.ptp(vertices[:,2])),runtime_root)
                report['meshes'].append(dict(name=obj.name,before_vertices=before,after_vertices=len(obj.data.vertices),**audit))
                continue
            for cut in cuts:
                curve=cut['curve'];gap=cut['gap']
                # Rounded, tapered slit tip below the shoulder attachment.
                polygon=[(curve[0][0],curve[0][1]+gap*2)]
                polygon += [(x-gap*.5,z) for x,z in curve]
                polygon += [(x+gap*.5,z) for x,z in reversed(curve)]
                n=len(polygon);verts=[(x,y,z) for y in (cut['y_min'],cut['y_max']) for x,z in polygon]
                fs=[tuple(range(n)),tuple(reversed(range(n,n*2)))]
                fs += [(i,i+n,(i+1)%n+n,(i+1)%n) for i in range(n)]
                data=bpy.data.meshes.new('Arm separation cutter');data.from_pydata(verts,[],fs);data.update()
                cutter=bpy.data.objects.new('Arm separation cutter',data);bpy.context.scene.collection.objects.link(cutter)
                mod=None;modifier_name=None
                try:
                    mod=obj.modifiers.new('Separate fused arm bridge','BOOLEAN');mod.operation='DIFFERENCE';mod.object=cutter
                    if hasattr(mod,'solver'):mod.solver='EXACT'
                    bpy.context.view_layer.objects.active=obj
                    modifier_name=mod.name
                    bpy.ops.object.modifier_apply(modifier=modifier_name)
                finally:
                    if modifier_name and obj.modifiers.get(modifier_name):
                        obj.modifiers.remove(obj.modifiers[modifier_name])
                    bpy.data.objects.remove(cutter,do_unlink=True);bpy.data.meshes.remove(data)
            bm=bmesh.new();bm.from_mesh(obj.data)
            initial_vertices={tuple(v.co) for v in bm.verts}
            fill=bm.verts.layers.int.get('hallway_generated_fill') or bm.verts.layers.int.new('hallway_generated_fill')
            cap_layer=bm.faces.layers.int.new('_hallway_fill_face')
            caps=[f for f in bm.faces if len(f.verts)>4]
            for f in caps:f[cap_layer]=1
            cap_faces=bmesh.ops.triangulate(bm,faces=caps)['faces'] if caps else []
            cap_edges=set(e for f in cap_faces for e in f.edges)
            pitch=np.ptp(vertices[:,2])*.006/max(obj.matrix_world.to_scale())
            # A closed polygon can span the entire slit. Triangulate and split
            # its interior so skinning has samples along both moving banks.
            for _ in range(8):
                long=[e for e in cap_edges if e.is_valid and e.calc_length()>pitch]
                if not long:break
                result=bmesh.ops.subdivide_edges(bm,edges=long,cuts=1,use_grid_fill=True)
                cap_edges={e for e in cap_edges if e.is_valid}
                cap_edges.update(e for e in result['geom_inner'] if isinstance(e,bmesh.types.BMEdge))
                cap_edges.update(e for e in result['geom_split'] if isinstance(e,bmesh.types.BMEdge))
            bmesh.ops.triangulate(bm,faces=[f for f in bm.faces if len(f.verts)>3])
            # Exact differences across nearly coincident source triangles can
            # leave tiny open boundaries. Only this previously closed solid is
            # eligible: weld numerical duplicates and cap remaining boundaries.
            eps=np.ptp(vertices[:,2])*1e-7/max(obj.matrix_world.to_scale())
            boundary_before=sum(e.is_boundary for e in bm.edges)
            if any(not e.is_manifold for e in bm.edges):
                bmesh.ops.remove_doubles(bm,verts=list(bm.verts),dist=eps)
                bmesh.ops.dissolve_degenerate(bm,edges=list(bm.edges),dist=eps)
                wire=[e for e in bm.edges if not e.link_faces]
                if wire:bmesh.ops.delete(bm,geom=wire,context='EDGES')
                boundary_edges=[e for e in bm.edges if e.is_boundary]
                if boundary_edges:bmesh.ops.holes_fill(bm,edges=boundary_edges,sides=0)
                bmesh.ops.recalc_face_normals(bm,faces=list(bm.faces))
            for v in bm.verts:
                if tuple(v.co) not in initial_vertices and v.link_faces and all(f[cap_layer] for f in v.link_faces):v[fill]=1
            bm.faces.layers.int.remove(cap_layer)
            bm.to_mesh(obj.data);obj.data.update()
            boundary=sum(e.is_boundary for e in bm.edges);nonmanifold=sum(not e.is_manifold for e in bm.edges);bm.free()
            if boundary or nonmanifold:raise RuntimeError(f'Arm separation did not produce a closed mesh ({boundary} boundary, {nonmanifold} nonmanifold edges); keep the source and review the detected bridge')
            # Preserve the cut's semantic banks through later surface fitting.
            # A volumetric solve can diffuse across a very narrow air gap;
            # these generated attributes prevent hands from reclaiming its body bank.
            points=np.array([obj.matrix_world @ v.co for v in obj.data.vertices])
            obj.data.calc_loop_triangles()
            triangles=points[np.array([t.vertices[:] for t in obj.data.loop_triangles])]
            maximum_gap_occupancy=0.
            for cut in cuts:
                curve=np.array(cut['curve'])
                for x,z in (curve[:-1]+curve[1:])*.5:
                    width=_thickness(_section(triangles,z),x)
                    if not np.isfinite(width) or width>cut['gap']*.1:
                        raise RuntimeError('Arm cut is still filled; verify a closed outward-facing cutter before binding')
                    maximum_gap_occupancy=max(maximum_gap_occupancy,width)
            bank=np.zeros(len(points),dtype=np.int32);strength=np.zeros(len(points),dtype=np.float32)
            for cut in cuts:
                sign=1 if cut['side']=='Left' else -1
                curve=np.array(cut['curve']);x=np.interp(points[:,2],curve[::-1,1],curve[::-1,0])
                top=curve[0,1];t=np.clip((top-points[:,2])/(np.ptp(vertices[:,2])*.035),0,1)
                side=(points[:,0]-prediction['joints']['Hips']['head'][0])*sign>=0
                contact=side & (points[:,2]>=curve[-1,1])
                strength[contact]=t[contact]*t[contact]*(3-2*t[contact])
                bank[contact & ((points[:,0]-x)*sign>0)]=sign
            for name,kind,values in [('hallway_arm_bank','INT',bank),('hallway_arm_bank_strength','FLOAT',strength)]:
                attr=obj.data.attributes.get(name) or obj.data.attributes.new(name=name,type=kind,domain='POINT')
                attr.data.foreach_set('value',values)
            report['meshes'].append(dict(name=obj.name,before_vertices=before,after_vertices=len(obj.data.vertices),boundary_edges=boundary,nonmanifold_edges=nonmanifold,repaired_boundary_edges=boundary_before,maximum_gap_occupancy=maximum_gap_occupancy))
        report['applied']=any('after_vertices' in row for row in report['meshes'])
        return report
    finally:
        bpy.context.view_layer.objects.active=active
        for o in bpy.context.selected_objects:o.select_set(False)
        for o in selected:o.select_set(True)


class PreparedMeshes:
    """Disposable topology preparation shared by interactive and batch callers.

    The selected source geometry and materials are never edited. Call close on
    success, error and cancellation, after the builder has copied the result.
    """
    def __init__(self, meshes, prediction, enabled=False, runtime_root=None):
        import bpy
        self.meshes=list(meshes)
        self.copies=[]
        self.report={'applied':False,'enabled':bool(enabled)}
        if not enabled:return
        from .blender_bridge import geometry
        if not plan(*geometry(meshes),prediction):return
        try:
            for obj in meshes:
                dup=obj.copy();dup.data=obj.data.copy()
                # A fresh preparation owns only the interiors it creates now.
                # Older generated surfaces are fixed input for this pass.
                fill=dup.data.attributes.get('hallway_generated_fill')
                if fill:dup.data.attributes.remove(fill)
                if 'hallway_arm_fill_revision' in dup:del dup['hallway_arm_fill_revision']
                self.copies.append(dup)
                matrix=obj.matrix_world.copy()
                dup.parent=None;dup.matrix_world=matrix;dup.animation_data_clear()
                for c in list(dup.constraints):dup.constraints.remove(c)
                for mod in list(dup.modifiers):dup.modifiers.remove(mod)
                dup.vertex_groups.clear()
                bpy.context.scene.collection.objects.link(dup)
            self.report=apply(self.copies,prediction,runtime_root)
            if self.report['applied']:
                self.meshes=self.copies
                for obj in self.copies:obj.hide_set(True)
            else:self.close()
        except BaseException:
            self.close()
            raise

    def close(self):
        import bpy
        for obj in self.copies:
            data=obj.data
            bpy.data.objects.remove(obj,do_unlink=True)
            if not data.users:bpy.data.meshes.remove(data)
        self.copies=[]


def enforce_banks(rig, meshes):
    """Honor generated cut topology at the binding boundary, before hand fitting."""
    from avatar_apparel_weights import weights,assign
    arm_prefixes=('UpperArm.','Forearm.','Hand.','Thumb','Index','Middle','Ring','Little')
    arm_names={b.name for b in rig.data.bones if b.name.startswith(arm_prefixes)}
    body=[b for b in rig.data.bones if b.use_deform and b.name not in arm_names and not b.name.startswith('Secondary_')]
    reports=[]
    for obj in meshes:
        bank=obj.data.attributes.get('hallway_arm_bank');strength=obj.data.attributes.get('hallway_arm_bank_strength')
        if not bank or not strength:continue
        changed=0
        for v in obj.data.vertices:
            side=bank.data[v.index].value
            allowed={n for n in arm_names if n.endswith('.L' if side>0 else '.R')} if side else {b.name for b in body}
            banned=set(weights(obj,v.index))-allowed
            candidates=[b for b in rig.data.bones if b.use_deform and b.name in allowed]
            share=strength.data[v.index].value
            if share<=0:continue
            row=weights(obj,v.index);removed=sum(w for n,w in row.items() if n in banned)*share
            if removed<1e-7:continue
            other={n:w for n,w in row.items() if n in allowed};total=sum(other.values())
            for n in row.keys()&banned:row[n]*=1-share
            if total>1e-4:
                for n,w in other.items():row[n]+=removed*w/total
            else:
                distances=[]
                for b in candidates:
                    a=b.head_local;d=b.tail_local-a;t=max(0.,min(1.,(v.co-a).dot(d)/max(d.length_squared,1e-16)))
                    distances.append((v.co-a-t*d).length)
                ids=np.argsort(distances)[:2];mass=1/np.maximum(np.array(distances)[ids],rig.data.bones['Hips'].length*.05)**4;mass/=mass.sum()
                for k,w in zip(ids,mass):row[candidates[k].name]=row.get(candidates[k].name,0)+removed*float(w)
            row={n:w for n,w in row.items() if w>1e-8};total=sum(row.values());assign(obj,[v.index],{n:w/total for n,w in row.items()});changed+=1
        reports.append(dict(object=obj.name,body_bank_vertices_corrected=changed))
    return reports


def close_shoulder_weights(rig, meshes):
    """Keep coincident split rims continuous at the anatomical shoulder join.

    The contact below the shoulder remains separated. Only vertices duplicated
    from the same surface position share rows, with a short rootward blend.
    """
    from avatar_apparel_weights import weights,assign
    from avatar_surface_attachments import normalized
    changed=0
    for obj in meshes:
        join=obj.data.attributes.get('hallway_arm_join')
        bank=obj.data.attributes.get('hallway_arm_bank')
        if not join or not bank:continue
        points=np.array([v.co[:] for v in obj.data.vertices])
        epsilon=max(float(np.ptp(points[:,2]))*1e-7,1e-10)
        groups={}
        for v in obj.data.vertices:
            if join.data[v.index].value<=0:continue
            key=tuple(np.rint(points[v.index]/epsilon).astype(np.int64))
            groups.setdefault(key,[]).append(v.index)
        for indices in groups.values():
            if len({bank.data[i].value for i in indices})<2:continue
            rows=[weights(obj,i) for i in indices]
            average={}
            for row in rows:
                for name,value in row.items():average[name]=average.get(name,0.)+value/len(rows)
            average=normalized(average)
            for i,row in zip(indices,rows):
                factor=join.data[i].value
                mixed={name:(1-factor)*row.get(name,0.)+factor*average.get(name,0.) for name in row.keys()|average.keys()}
                assign(obj,[i],normalized(mixed));changed+=1
    return dict(shoulder_join_vertices=changed)


def limit_generated_fill_weights(meshes):
    """Keep generated closures exportable independently of any shape repair."""
    from avatar_apparel_weights import weights,assign
    from avatar_surface_attachments import normalized
    changed=0
    for obj in meshes:
        flag=obj.data.attributes.get('hallway_generated_fill')
        if not flag:continue
        for i,value in enumerate(flag.data):
            if not value.value:continue
            row=weights(obj,i)
            if len(row)>4:
                assign(obj,[i],normalized(row));changed+=1
    return dict(vertices=changed,max_influences=4,geometry_changed=False)
