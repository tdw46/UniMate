"""Public bind mesh validation; error order is independent of cache state."""
import numpy as np

def validate_mesh(vertices,faces):
    if vertices.shape[1]!=3:
        raise ValueError(f'[invalid_mesh] vertices must be shape (N, 3), got {vertices.shape}')
    if faces.shape[1]!=3:
        raise ValueError(f'[invalid_mesh] faces must be shape (N, 3), got {faces.shape}')
    if not len(vertices):raise ValueError('[invalid_mesh] empty vertex array')
    # The native whole-mesh check reports maximum overflow before negatives,
    # independently of triangle order. Per-part errors report a whole triangle.
    if faces.size:
        high=int(faces.max())
        if high>=len(vertices):raise ValueError(f'[invalid_mesh] face index {high} exceeds vertex count {len(vertices)}')
        low=int(faces.min())
        if low<0:raise ValueError(f'[invalid_mesh] negative face index: {low}')

def validate_part(index,vertices,faces,base,matrix):
    if vertices.shape[1]!=3 or faces.shape[1]!=3:
        raise ValueError(f'[invalid_mesh] mesh {index}: vertices and faces must be (N, 3)')
    if (base is None)!=(matrix is None):
        raise ValueError(f'mesh {index}: pass the base vertices and the world matrix together')
    if base is not None and (base.shape[1]!=3 or matrix.shape!=(4,4)):
        raise ValueError(f'[invalid_mesh] mesh {index}: base vertices must be (V, 3), the matrix (4, 4)')
    invalid=np.flatnonzero(np.any((faces<0)|(faces>=len(vertices)),axis=1))
    if len(invalid):
        triangle=faces[int(invalid[0])].tolist()
        raise ValueError(f'[invalid_mesh] mesh {index}: face index out of range {triangle} ({len(vertices)} vertices)')
