"""Read-only capsule spans for VRM's two-empty collider representation.

No extra objects, export geometry, solver, timer, or dependency-graph writes.
"""
import math
import threading
import bpy
from mathutils import Vector

_handle = None
_shader = None


def capsule_lines(a, b, radius, sides=16):
    axis = b-a
    if radius <= 0 or axis.length < 1e-8:
        return []
    axis.normalize()
    u = axis.orthogonal().normalized()
    v = axis.cross(u).normalized()
    lines = []
    for i in range(sides):
        angle = math.tau*i/sides
        next_angle = math.tau*(i+1)/sides
        radial = u*math.cos(angle)+v*math.sin(angle)
        other = u*math.cos(next_angle)+v*math.sin(next_angle)
        lines.extend((a+radius*radial, a+radius*other,
                      b+radius*radial, b+radius*other))
        if i % 2 == 0:
            lines.extend((a+radius*radial, b+radius*radial))
        if i % 4 == 0:
            for j in range(8):
                t=math.pi*.5*j/8;nt=math.pi*.5*(j+1)/8
                for center, sign in ((a,-1),(b,1)):
                    lines.extend((center+radius*(radial*math.cos(t)+sign*axis*math.sin(t)),
                                  center+radius*(radial*math.cos(nt)+sign*axis*math.sin(nt))))
    return lines


def visible_capsules(context):
    for rig in context.scene.objects:
        if rig.type != 'ARMATURE' or not (rig.get('hallway_generated_rig') or rig.get('unimate_secondary_generator')):
            continue
        ext = getattr(rig.data, 'vrm_addon_extension', None)
        if ext is None or ext.spec_version != '1.0':
            continue
        for collider in ext.spring_bone1.colliders:
            head = collider.bpy_object
            if (collider.shape_type != 'Capsule' or not head or not head.children
                    or not head.get('unimate_generated_collider')
                    or getattr(getattr(getattr(collider, 'extensions', None),
                                       'vrmc_spring_bone_extended_collider', None), 'enabled', False)
                    or not head.visible_get(view_layer=context.view_layer, viewport=context.space_data)):
                continue
            yield collider, head


def draw():
    global _shader
    if bpy.app.background or threading.current_thread() != threading.main_thread():
        return
    if callable(getattr(bpy.app, 'is_job_running', None)) and bpy.app.is_job_running('RENDER'):
        return
    context = bpy.context
    space = context.space_data
    if not space or space.type != 'VIEW_3D' or not space.overlay.show_overlays or not space.overlay.show_extras:
        return
    import gpu
    from gpu_extras.batch import batch_for_shader
    if _shader is None:
        _shader = gpu.shader.from_builtin('POLYLINE_UNIFORM_COLOR')
    depth, mask, blend = gpu.state.depth_test_get(), gpu.state.depth_mask_get(), gpu.state.blend_get()
    try:
        gpu.state.depth_mask_set(False)
        gpu.state.blend_set('ALPHA')
        _shader.bind()
        _shader.uniform_float('viewportSize', gpu.state.viewport_get()[2:])
        _shader.uniform_float('lineWidth', 1.25)
        for collider, head in visible_capsules(context):
            points=capsule_lines(head.matrix_world.translation,head.children[0].matrix_world.translation,
                                 collider.shape.capsule.radius)
            if not points:
                continue
            gpu.state.depth_test_set('NONE' if head.show_in_front else 'LESS_EQUAL')
            _shader.uniform_float('color', (0.25,0.85,1.,.75))
            batch_for_shader(_shader,'LINES',{'pos':points}).draw(_shader)
    finally:
        gpu.state.depth_test_set(depth)
        gpu.state.depth_mask_set(mask)
        gpu.state.blend_set(blend)


def register():
    global _handle
    if not bpy.app.background and _handle is None:
        _handle=bpy.types.SpaceView3D.draw_handler_add(draw,(),'WINDOW','POST_VIEW')


def unregister():
    global _handle, _shader
    if _handle is not None:
        bpy.types.SpaceView3D.draw_handler_remove(_handle,'WINDOW')
    _handle=None
    _shader=None
