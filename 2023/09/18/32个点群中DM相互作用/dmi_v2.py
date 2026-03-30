import sympy
from sympy.tensor.tensor import TensorIndexType, TensorHead, tensor_indices
from typing import Generator
import itertools


def get_rot_mat(nx, ny, nz, t):
    # t-angle rotation with axis (nx, ny, nz)
    n = sympy.Matrix([nx, ny, nz])
    n_norm = n.norm()
    n = n / n_norm
    kron_nn = sympy.kronecker_product(n, n.T)
    n_cross_mat = sympy.Matrix([
        [0, -n[2], n[1]],
        [n[2], 0, -n[0]],
        [-n[1], n[0], 0],
    ])
    return sympy.cos(t) * sympy.eye(3) + (1 - sympy.cos(t)) * kron_nn + sympy.sin(t) * n_cross_mat


def rotate_z(theta):
    return get_rot_mat(0, 0, 1, theta)


class Group:
    def __init__(self):
        self.name = 'C1'

    def iter_elements(self) -> Generator[sympy.Matrix, None, None]:
        yield sympy.eye(3)


class CnGroup(Group):
    def __init__(self, n):
        super().__init__()
        self.name = f'C{n}'
        self.n = n

    def iter_elements(self) -> Generator[sympy.Matrix, None, None]:
        # cyclic group
        for i in range(self.n):
            theta = 2 * sympy.pi * i / self.n
            yield rotate_z(theta)


class CnhGroup(Group):
    def __init__(self, n):
        super().__init__()
        self.name = f'C{n}h'
        self.n = n

    def iter_elements(self) -> Generator[sympy.Matrix, None, None]:
        for s in [-1, 1]:
            for i in range(self.n):
                theta = 2 * sympy.pi * i / self.n
                yield sympy.diag(1, 1, s) * rotate_z(theta)


class CnvGroup(Group):
    def __init__(self, n):
        super().__init__()
        self.name = f'C{n}v'
        self.n = n

    def iter_elements(self) -> Generator[sympy.Matrix, None, None]:
        for s in [-1, 1]:
            for i in range(self.n):
                theta = 2 * sympy.pi * i / self.n
                yield sympy.diag(1, s, 1) * rotate_z(theta)


class CsGroup(CnhGroup):
    def __init__(self):
        super().__init__(1)
        self.name = 'Cs'


class SnGroup(Group):
    def __init__(self, n):
        super().__init__()
        self.name = f'S{n}'
        self.n = n

    def iter_elements(self) -> Generator[sympy.Matrix, None, None]:
        g = sympy.diag(1, 1, -1) * rotate_z(2 * sympy.pi / self.n)
        e0 = sympy.eye(3)
        for i in range(self.n):
            if i != 0:
                e0 *= g
            yield e0


class CiGroup(SnGroup):
    def __init__(self):
        super().__init__(2)
        self.name = 'Ci'


class C3iGroup(SnGroup):
    def __init__(self):
        super().__init__(6)
        self.name = f'C3i'


class DnGroup(Group):
    def __init__(self, n):
        super().__init__()
        self.name = f'D{n}'
        self.n = n

    def iter_elements(self) -> Generator[sympy.Matrix, None, None]:
        # dihedral group
        for s in [-1, 1]:
            for i in range(self.n):
                theta = 2 * sympy.pi * i / self.n
                yield sympy.diag(1, s, s) * rotate_z(theta)


class DnhGroup(Group):
    def __init__(self, n):
        super().__init__()
        self.name = f'D{n}h'
        self.n = n

    def iter_elements(self) -> Generator[sympy.Matrix, None, None]:
        for s1 in [-1, 1]:
            for s2 in [-1, 1]:
                for i in range(self.n):
                    theta = 2 * sympy.pi * i / self.n
                    yield sympy.diag(1, 1, s2) * sympy.diag(1, s1, s1) * rotate_z(theta)


class DndGroup(Group):
    def __init__(self, n):
        super().__init__()
        self.name = f'D{n}d'
        self.n = n

    def iter_elements(self) -> Generator[sympy.Matrix, None, None]:
        alpha = sympy.pi / self.n / 2
        cos_a = sympy.cos(alpha)
        sin_a = sympy.sin(alpha)
        mat_reflect = -get_rot_mat(-sin_a, cos_a, 0, sympy.pi)
        for s in [-1, 1]:
            for i in range(self.n):
                theta = 2 * sympy.pi * i / self.n
                yield sympy.diag(1, s, s) * rotate_z(theta)
                yield mat_reflect * sympy.diag(1, s, s) * rotate_z(theta)


class TdGroup(Group):
    def __init__(self):
        super().__init__()
        self.name = 'Td'

    def iter_elements(self) -> Generator[sympy.Matrix, None, None]:
        # symmetry group of tetrahedron
        for idx in itertools.permutations([0, 1, 2]):
            for signs in [(1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)]:
                T = sympy.zeros(3, 3)
                for i in range(3):
                    T[i, idx[i]] = signs[i]
                yield T


class TGroup(Group):
    def __init__(self):
        super().__init__()
        self.name = 'T'

    def iter_elements(self) -> Generator[sympy.Matrix, None, None]:
        # symmetry group of tetrahedron, proper rotation only
        group_td = TdGroup()
        for g in group_td.iter_elements():
            if g.det() > 0:
                yield g


class ThGroup(Group):
    def __init__(self):
        super().__init__()
        self.name = 'Th'

    def iter_elements(self) -> Generator[sympy.Matrix, None, None]:
        # direct product of T and Ci
        group_t = TGroup()
        for trans in group_t.iter_elements():
            yield trans
            yield -trans


class OhGroup(Group):
    def __init__(self):
        super().__init__()
        self.name = 'Oh'

    def iter_elements(self) -> Generator[sympy.Matrix, None, None]:
        # symmetry group of cube
        for idx in itertools.permutations([0, 1, 2]):
            for signs in itertools.product([-1, 1], [-1, 1], [-1, 1]):
                T = sympy.zeros(3, 3)
                for i in range(3):
                    T[i, idx[i]] = signs[i]
                yield T


class OGroup(Group):
    def __init__(self):
        super().__init__()
        self.name = 'O'

    def iter_elements(self) -> Generator[sympy.Matrix, None, None]:
        # symmetry group of cube, proper rotation only
        group_oh = OhGroup()
        for trans in group_oh.iter_elements():
            if trans.det() > 0:
                yield trans


t = sympy.Symbol('t')
V = TensorIndexType('V', dummy_name='V')
L = TensorHead('L', [V, V, V])
U = TensorHead('U', [V, V])
i, j, k, l, m, n = tensor_indices('i,j,k,l,m,n', V)
L_trans = U(-l, i) * U(-m, j) * U(n, -k) * L(-i, -j, k)
L_mat = sympy.tensor.MutableDenseNDimArray.zeros(3, 3, 3)
axis_name = ['x', 'y', 'z']
axis_2_name = ['yz', 'zx', 'xy']
for i1 in range(3):
    for j1 in range(3):
        Lif = sympy.Symbol(f'L{axis_2_name[i1]}{axis_name[j1]}')
        idx1 = axis_name.index(axis_2_name[i1][0])
        idx2 = axis_name.index(axis_2_name[i1][1])
        L_mat[idx1, idx2, j1] = Lif
        L_mat[idx2, idx1, j1] = -Lif


def get_dmi_terms(group: Group):
    L_sum = sympy.tensor.MutableDenseNDimArray.zeros(3, 3, 3)
    for g in group.iter_elements():
        rep = {
            U(-l, i): g,
            L(-i, -j, k): L_mat,
            V: sympy.eye(3)
        }
        L_rep = L_trans.replace_with_arrays(rep)
        L_sum += L_rep
    non_zero_terms = []
    for i1, j1 in [(1, 2), (2, 0), (0, 1)]:
        for k1 in range(3):
            term1 = L_sum[i1, j1, k1]
            # exclude zero terms
            if term1 == 0:
                continue
            # exclude equivalent terms
            is_new_term = True
            for term in non_zero_terms:
                ratio = sympy.simplify(term / term1)
                if ratio.is_constant():
                    is_new_term = False
                    break
            if is_new_term:
                non_zero_terms.append(term1)
    if len(non_zero_terms) > 0:
        print(group.name)
        print(non_zero_terms)


all_point_groups = [
    CnGroup(1), CnGroup(2), CnGroup(3), CnGroup(4), CnGroup(6),
    CnvGroup(2), CnvGroup(3), CnvGroup(4), CnvGroup(6),
    CnhGroup(1), CnhGroup(2), CnhGroup(3), CnhGroup(4), CnhGroup(6),
    SnGroup(2), SnGroup(4), SnGroup(6),
    DnGroup(2), DnGroup(3), DnGroup(4), DnGroup(6),
    DndGroup(2), DndGroup(3),
    DnhGroup(2), DnhGroup(3), DnhGroup(4), DnhGroup(6),
    TGroup(), ThGroup(), TdGroup(), OGroup(), OhGroup(),
]

for group in all_point_groups:
    get_dmi_terms(group)
