"""Stable anatomical orientation frames for preview retargeting.

An ankle's posed shin-to-toe cross product becomes singular when the foot
points along the shin. Establish its frame once in rest space, then transport
that frame with the authored foot rotation instead of rebuilding it per pose.
"""

def transport_rest_frame(rest_frame, rest_world_matrix, pose_world_matrix):
    if rest_frame is None:
        return None
    return (pose_world_matrix.to_quaternion().normalized()
            @ rest_world_matrix.to_quaternion().normalized().inverted()
            @ rest_frame).normalized()
