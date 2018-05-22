#!/usr/bin/python2
"""Toolkit for coordinate transformations.

    CAUTION: All angles (input and output) are in radians (unless otherwise
    stated)!

    Classes
    =======
    This module defines three classes: Transform, Vector, Rotation.
    - They work in 3D Euklidian space.
    - Vector can handle different coordinate systems: cartesian, sherical and
      cylindrical. Rotation operates always in cartesian coordinates. Transform
      uses Vector and Rotation. Vector can be in any coordinate system but upon
      application, its cartesian representation is used.
    - Vector can also represent an array of n vectors (implemented in a
      3xn-matrix).
    - Most operations of Transform, Vector and Rotation on vector work as well
      when the operand Vector is an array. These method exploit the optimized
      numpy operations. In doing so, computations can be sped up substantially.

    If you want to perform other coordinate transformations than these, have a
    look at `transformations.py` by Christoph Gohlke:
    https://www.lfd.uci.edu/~gohlke/code/transformations.py.html
    It can elegantly perform many more coordinate transformation (such as
    shear, projection, etc.) at the expense of being a bit less easily
    accessible for non-mathematicians.


    Todo / ideas
    ============
    Currently none. You got any? Let me know!


    Reliability
    ===========
    All functions and methods have been tested moderately but not exhaustively.
    So most probably, there are still undiscovered bugs.
    I recommend that you perform plausibility checks on your results. (AA)


    Coordinate Systems
    ==================
    *** CARTESIAN COORDINATES ***
        x, y, z defined in the regular (right-handed) way


    *** SPHERICAL COORDINATES ***
        right-handed: r, phi, theta

        symbol  range          interpretation                  ref. ("0 at")
        ----------------------------------------------------------------------
        r       [0..inf)       upwardness (dist. from origin)  origin
        phi     [0..2pi)       eastwardness (longitude)        positive x-axis
        theta   [-pi/2..pi/2]  northwardness (elevation, lat.) x-y-plane

        uniqueness:
        r == 0            -->  phi, theta == 0
        theta == +- pi/2  -->  phi == 0


    *** CYLINDRICAL COORDINATES ***
        right-handed: rho, phi, z


        symbol  range         interpretation                 reference ("0 at")
        -----------------------------------------------------------------------
        rho     [0...inf)     radius (distance from z-axis)  z-axis
        phi     [0...2pi)     eastwardness (azimuth)         positive x-axis
        z       (-inf...inf)  distance to x-y-plane          x-y-plane

        uniqueness:
        r == 0  -->  phi == 0


    Author
    ======
    Andreas Anhaeuser (AA) <anhaeus@meteo.uni-koeln.de>
    Institute for Geophysics and Meteorology
    University of Cologne, Germany


    History
    =======
    2017-11-22 (AA): Created.
    2017-12-12 (AA): Extension for 3xn Vectors. Clean up. Documentation.
    2018-01-10 (AA): Documentation and comments.
    2018-05-18 (AA): Made Rotation and Transform instances callable.
"""
# Computer, please:

import numpy as np
from copy import deepcopy as copy

import conversions as conv

###################################################
# CLASSES                                         #
###################################################
class Vector(object):
    """A 3D vector that is aware of its type of coordinate system.

        A Vector is a representation of a point (or shift, etc.) in 3D
        Euklidian space.
        
        It is aware of its type of coordinate system. This can be
        - 'car' (cartesian)
        - 'sph' (spherical)
        - 'cyl' (cylindrical)
        The representation of the Vector can be changed arbitrarily between
        these coordinate systems.

        Various operations are defined on a Vector:
        - unary (abs, negate, unit vector, ...)
        - arithmetics with another Vector (+, -, dot, cross)
        - arithmetics with scalars (*, /)

        Other fancy properties can be retrieved:
        - zenith and azimuth angles

        The class is 'defensive' w. r. t. input values: if they are beyond
        bounds, they are converted automatically (e. g. r < 0, phi > 2pi),
        using the interpretation that is most probably intended by the user.

        CAUTION: All angles (input and output) are in radians!


        Initialization
        ==============
        Instantiate the class with Vector(co, kind), where in the simple case,
        `co` is a 3-element array. The interpretation of the 3 element of `co`
        depends on the coordinate system, specified by `kind`, see Section
        'Coordiante Systems' in the module docstring.


        List of Vector
        ==============
        - To get one vector, instantiate the class with a 3-element array.
        - To get a Vector object that actually represents a list of N
          mathematical vectors, instantiate the class with a 3xN array.


        History
        =======
        2018-12-12 (AA): Created.
        2018-01-03 (AA): Azimuth and zenith angles.
        2018-05-22 (AA): Reduced type identifier to three characters
                         ('car[t]', 'sph[eric]'); kept backward compatible.
    """
    _valid_kinds = (None, 'car', 'sph', 'cyl')

    def __init__(self, co=np.zeros(3), kind='car'):
        """Initialize.

            See class description for details.

            CAUTION: All angles (input and output) are in radians!

            Parameters
            ----------
            co : array-like, first dimensions must be 3
                first dimension is interpreted as three coordinate axes.
            kind : {'car', 'sph', 'cyl'}
                How the coordinates are to be interpreted.
        """
        if not isinstance(kind, str):
            raise TypeError('kind must be a str.')
        kind = kind[:3].lower()

        if kind[:3] not in self._valid_kinds:
            raise ValueError('kind must be in %s.' % str(_valid_kinds))

        # normalize
        if kind == 'sph':
            co = conv.normalize_sph(*co)
        elif kind == 'cyl':
            co = conv.normalize_cyl(*co)

        co = np.array(co)
        assert np.shape(co)[0] == 3

        self.kind = kind
        self.co = co

    def __repr__(self):
        """Return string representation."""
        # kind
        kind = self.kind
        if kind == 'car':
            k = 'cartesian'
            n = '(x, y, z)'
        elif kind == 'sph':
            k = 'spherical'
            n = '(r, phi, theta)'
        elif kind == 'cyl':
            k = 'cylindrical'
            n = '(rho, phi, z)'

        # coordinates
        c = str(self.get_coords())

        # infix
        if len(np.shape(self.co)) > 1:
            i = '\n'
        else:
            i = ' '

        # compose string
        s = 'Vector in %s coordinates %s:%s%s' % (k, n, i, c)
        return s

    def __abs__(self):
        """Return absolute value as a number."""
        co = self.co
        if self.kind == 'car':
            a = np.sqrt(np.sum(co**2, 0))
        elif self.kind == 'sph':
            a = co[0]
        elif self.kind == 'cyl':
            rho = co[0]
            z = co[2]
            a = np.sqrt(rho**2 + z**2)
        return a

    def __neg__(self):
        """Return negation as a Vector."""
        kind = self.kind
        if kind == 'car':
            co = - self.co
        elif kind == 'sph':
            r = self.co[0]
            phi = self.co[1]
            theta = self.co[2]
            co = conv.normalize_sph(-r, phi, theta)
        elif kind == 'cyl':
            rho = self.co[0]
            phi = self.co[1]
            z = self.co[2]
            co = conv.normalize_cyl(-rho, phi, -z)
        return Vector(co, kind)

    def __add__(self, other):
        """Add another Vector and return result as Vector."""
        co1 = self.get_car()
        co2 = other.get_car()
        s1 = np.shape(co1)
        s2 = np.shape(co2)

        # regular case (both vectors have same shape):
        if s1 == s2:
            co = co1 + co2

        # first vector (3x1), the other (3xn):
        elif len(s1) == 1:
            co = (co1 + co2.T).T

        # first vector (3xn), the other (3x1):
        elif len(s2) == 1:
            co = (co1.T + co2).T

        # this should not happen:
        else:
            raise IndexError()

        return Vector(co, 'car')

    def __iadd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        """Substract another Vector and return result as Vector."""
        return self + (-other)

    def __isub__(self, other):
        return self.__sub__(other)

    def __mul__(self, a):
        """Multiply by a scalar and return result as Vector."""
        kind = self.kind
        co = copy(self.co)
        assert np.shape(a) == np.shape(co[0])
        if kind == 'car':
            co *=  a
        elif kind == 'sph':
            co[0] *= a
        elif kind == 'cyl':
            co[0] *= a
            co[2] *= a
        return Vector(co, kind)

    def __rmul__(self, a):
        return self.__mul__(a)

    def __imul__(self, a):
        return self.__mul__(a)

    def __div__(self, a):
        """Devide by a non-zero scalar and return result as Vector."""
        if np.shape(a) == () and a == 0:
            raise ZeroDivisionError()
        return self.__mul__(1. / a)

    def __idiv__(self, a):
        return self.__div__(a)

    def __truediv__(self, a):
        return self.__div(a)

    def __itruediv__(self, a):
        return self.__div__(a)

    def __len__(self):
        return self.length()

    def __bool__(self):
        return abs(self) != 0

    ###################################################
    # UNARY OPERATORS                                 #
    ###################################################
    def unit_vector(self):
        """Return a unit Vector pointing in the same direction."""
        return self / abs(self)

    ###################################################
    # BINARY OPERATORS                                #
    ###################################################
    def inner(self, other):
        """Return inner product with another Vector. (alias to 'dot')"""
        return self.dot(other)

    def dot(self, other):
        """Return inner product with another Vector.
        
            If both `self` and `other` are Vector lists, shapes must match.
        """
        assert isinstance(other, Vector)
        co1 = self.get_car()
        co2 = other.get_car()
        S1 = np.shape(co1)
        S2 = np.shape(co2)
        if S1 == S2:
            P = co1.dot(co2)
        elif len(S1) == 1:
            P = co1.dot(co2)
        elif len(S2) == 1:
            P = co2.dot(co1)
        else:
            raise IndexError('Vector do not have matching shapes.')
        return P

    def cross(self, other):
        """Return outer product. Only works for 3x1 Vectors."""
        assert isinstance(other, Vector)
        assert len(np.shape(self.co)) == 1
        assert len(np.shape(other.co)) == 1
        co1 = self.get_car()
        co2 = other.get_car()
        return Vector(np.cross(co1, co2))

    ###################################################
    # GETTERS: BASIC                                  #
    ###################################################
    def get_coords(self):
        """Get coordinates, unrespective of kind.

            This is merely intended as a helper to other class methods. It is
            discouraged to access it from outside as a user function.
            Use get_car(), get_sph() and get_cyl() instead unless you have
            good reason not to do so.
        
            Caution
            -------
            This returns the Vector coordinates in its native kind! Make sure
            you know which system you are working in for correct
            interpretation.
        """
        return self.co

    def get_kind(self):
        """Often useful in combination with get_coords()."""
        return self.kind

    def get_car(self):
        """Get coordinates in cartesian representation."""
        kind = self.kind
        co = self.get_coords()
        if kind == 'sph':
            co = np.array(conv.sph2car(*co))
        if kind == 'cyl':
            co = np.array(conv.cyl2car(*co))
        return co

    # alias
    def get_cart(self):
        return self.get_car()

    def get_sph(self):
        """Get coordinates in spherical representation."""
        kind = self.kind
        co = self.get_coords()
        if kind == 'car':
            co = np.array(conv.car2sph(*co))
        if kind == 'cyl':
            co = np.array(conv.cyl2sph(*co))
        return co

    # alias
    def get_spheric(self):
        return self.get_sph()

    def get_cyl(self):
        """Get coordinates in cylindrical representation."""
        kind = self.kind
        co = self.get_coords()
        if kind == 'car':
            co = np.array(conv.car2cyl(*co))
        if kind == 'sph':
            co = np.array(conv.sph2cyl(*co))
        return co

    ###################################################
    # GETTERS: ANGLES                                 #
    ###################################################
    def get_azimuth(self):
        """Return azimuth angle in rad.

            The azimuth angle is measured in the x-y-plane, starting at the
            y-axis towards the positive x-axis, i. e. the angle is measured in
            mathematically negative sense.

            Parameters
            ----------

            Returns
            -------
            float or array of such
                (rad) azimuth
                
            History
            -------
            2018-01-04 (AA): Created
        """
        co = self.get_car()
        return np.arctan2(co[0], co[1])

    def get_zenith_angle(self):
        """Return zenith angle (distance from z-axis) in rad.

            Parameters
            ----------

            Returns
            -------
            float or array of such
                (rad) zenith angle
                
            History
            -------
            2018-01-04 (AA): Created
        """
        theta = self.get_sph()[2]
        return np.pi/2. - theta

    ###################################################
    # GETTERS: ARRAY                                  #
    ###################################################
    def is_array(self):
        return len(np.shape(self.co)) > 1

    def length(self):
        if not self.is_array():
            TypeError('Not an array of Vector.')
        else:
            return np.shape(self.co)[1]

    def element(self, n):
        if not self.is_array():
            TypeError('Not an array of Vector.')
        N = self.length()
        if n > N - 1:
            raise IndexError()
        kind = self.get_kind()
        co = self.get_coords()[:, n]
        return Vector(co, kind)

    ###################################################
    # SETTERS                                         #
    ###################################################
    def set_kind(self, kind):
        """Set coordinate kind."""
        if not isinstance(kind, str):
            raise TypeError('kind must be a str.')
        kind = kind[:3].lower()
        if not kind in self._valid_kinds:
            raise ValueError('')
        self.kind = kind
        return self

    def set_coords(self, co):
        """Set coordinates, unrespective of kind.
        
            Caution
            -------
            This method is unaware of the coordinate kind! It does not perform
            any compatibility check. So make sure you are working in the
            correct system.
        """
        co = np.array(co)
        assert np.shape(co)[0] == 3

        self.co = co
        return self

    def set_car(self, co):
        """Set cartesian coordinates. Set kind accordingly."""
        self.set_kind('car')
        self.set_coords(co)
        return self

    # alias
    def set_cart(self, co):
        return self.set_car(co)

    def set_sph(self, co):
        """Set spherical coordinates. Set kind accordingly."""
        self.set_kind('sph')
        self.set_coords(co)
        return self

    # alias
    def set_spheric(self, co):
        return self.set_sph(co)

    def set_cyl(self, co):
        """Set cylindrical coordinates. Set kind accordingly."""
        self.set_kind('cyl')
        self.set_coords(co)
        return self

    ###################################################
    # CONVERTERS                                      #
    ###################################################
    def convert_to(self, kind):
        """Convert Vector to another coordinate system.

            Parameters
            ----------
            kind : {'car', 'sph', 'cyl'}

            Returns
            -------
            Vector
        """
        if not isinstance(kind, str):
            raise TypeError('kind must be a str.')
        kind = kind[:3].lower()

        if kind == 'car':
            return self.convert_to_car()
        if kind == 'sph':
            return self.convert_to_sph()
        if kind == 'cyl':
            return self.convert_to_cyl()

    # --> cartesian
    def convert_to_car(self):
        """Convert Vector to cartesian representation."""
        self.set_coords(self.get_car())
        self.set_kind('car')
        return self

    # alias
    def convert_to_cart(self):
        return self.convert_to_car()

    # alias
    def convert_to_cartesian(self):
        return self.convert_to_car()

    # --> spherical
    def convert_to_sph(self):
        """Convert Vector to spherical representation."""
        self.set_coords(self.get_sph())
        self.set_kind('sph')
        return self

    # alias
    def convert_to_spheric(self):
        return self.return_to_sph()

    # alias
    def convert_to_spherical(self):
        return self.return_to_sph()

    # --> cylindrical
    def convert_to_cyl(self):
        """Convert Vector to cylindrical representation."""
        self.set_coords(self.get_cyl())
        self.set_kind('cyl')
        return self

    # alias
    def convert_to_cylindrical(self):
        return self.return_to_cyl()

class Rotation(object):
    """A rotation in 3D Euklidian space.

        It can be applied to Vector objects by simply calling them with the
        Vector as argument.

        No coordinate conversion by the user is necessary, even if the Vector
        is not cartesian (this is done internally).

        Two Rotation objects can be merged into a resulting Rotation object
        which represents the net rotation using the methods before() and after().

        A rotation can be initialized if you know either
        - rotation axis (represented by a Vector) and rotation angle or
        - the rotation matrix.

        The Rotation is internally represented as a 3x3 matrix in cartesian
        x-y-z coordinates.


        Initialization
        ==============
        - If you know the rotation axis and rotation angle, an instance can be
          created directly by calling
             Rotatian(angle, axis)
          `axis` is in general a Vector. If you rotate about a cartesian
          coordinate axis, you can also use axis=0, axis=1 or axis=2 instead.
        - If you know the rotation matrix, instantiate as follows:
             Rotation().set_matrix(matrix)


        History
        =======
        2018-12-12 (AA): Created.
        2018-01-03 (AA): Azimuth and zenith angles.
        2018-01-10 (AA): Documentation.


        Examples
        ========
        >>> # rotate Vector `v` by angle `alpha` about `axis1`
        >>> v = Vector((2, np.pi/4, -np.pi/3))
        >>> axis1 = Vector((0, 1, 0))
        >>> alpha = np.radians(42)
        >>> rot1 = Rotation(alpha, axis1)
        >>> print(rot1(v))

        >>> # concatenate with another rotation `rot2`
        >>> axis2 = Vector((0, -1, 2))
        >>> beta = np.radians(20)
        >>> rot2 = Rotation(beta, axis2)
        >>> rot = rot2.after(rot1)
        >>> print(rot(v))
    """
    def __init__(self, angle=0, axis=0):
        """Set Rotation by angle about axis.

            Parameters
            ----------
            angle : float, optional
                (rad) rotation angle. Default: 0
            axis : Vector or int, optional
                Axis about which the rotation is performed. Can be a Vector is
                arbitraty direction or an integer indication one of the
                coordinate axes. Default: 0 (i. e. x-axis)
        """
        self.set_angle_and_axis(angle, axis)

    def __repr__(self):
        return 'Rotation with matrix\n%s' % repr(self.matrix)

    def __mul__(self, other):
        return self.dot(other)

    def __truediv__(self, other):
        if isinstance(other, Rotation):
            return self.__mul__(other.inveirse())
        else:
            raise TypeError()

    def __eq__(self, other):
        assert isinstance(other, Rotation)
        Meq = self.matrix == other.matrix
        return np.sum(Meq) == 9

    def __call__(self, v):
        return self.apply_to(v)

    ###################################################
    # UNARY OPERATORS                                 #
    ###################################################
    def inverse(self):
        """Return inverse Rotation."""
        return Rotation().set_matrix(self.matrix.T)

    def rotate(self, angle, axis):
        """Additionally rotate the Rotation.

            Parameters
            ----------
            angle : float
                (rad) rotation angle
            axis : int or Vector
                Axis about which the rotation is performed. Can be a Vector or
                the number of the coordinate axis (0:x, 1:y, 2:z).

            Returns
            -------
            Rotation
        """
        other = Rotation(angle, axis)
        new = other.after(self)
        self.set_matrix(new.get_matrix())
        return self

    ###################################################
    # BINARY OPERATORS                                #
    ###################################################
    def dot(self, other):
        """Return result of dot-product with Rotation or Vector."""
        if isinstance(other, Rotation):
            A = self.get_matrix()
            B = other.get_matrix()
            matrix = A.dot(B)
            return Rotation().set_matrix(matrix)
        elif isinstance(other, Vector):
            M = self.get_matrix()
            v = other.get_car()
            return Vector(M.dot(v))
        else:
            raise TypeError('other must be Rotation or Vector.')

    def after(self, other):
        """Return combined Rotation."""
        return self.dot(other)

    def before(self, other):
        """Return combined Rotation."""
        return other.after(self)

    ###################################################
    # SETTERS                                         #
    ###################################################
    def set_matrix(self, matrix):
        """Set rotation matrix."""
        assert isinstance(matrix, np.ndarray)
        assert matrix.shape == (3, 3)
        self.matrix = matrix
        return self

    def set_angle_and_axis(self, angle, axis):
        """Set Rotation by angle and axis.

            Parameters
            ----------
            angle : float
                (rad) rotation angle
            axis : Vector or int
                Axis about which the rotation is performed. Can be a Vector or
                the number of the coordinate axis (0:x, 1:y, 2:z).

            Returns
            -------
            Rotation
        """
        # special case: no rotation
        if angle == 0:
            self.matrix = np.eye(3)
            return self

        # special case: coordinate axis
        if axis in range(3):
            # delegate to sub-function
            co = np.zeros(3)
            co[axis] = 1.
            axis = Vector(co)
            return self.set_angle_and_axis(angle, axis)

        # input check
        if not isinstance(axis, Vector):
            raise TypeError('`axis` must be a Vector or an int between 0..2 .')

        # shortcuts
        sin = np.sin(angle)
        cos = np.cos(angle)
        
        # normalize
        norm = 1. * abs(axis)
        n = axis.get_car() / norm

        # build rotation matrix
        first = (1 - cos) * np.outer(n, n)
        diag = cos * np.eye(3)
        off =  sin * np.array([[   0., -n[2],  n[1]],
                               [ n[2],    0., -n[0]],
                               [-n[1],  n[0],   0.]])
        M = first + diag + off
        self.matrix = M
        return self

    ###################################################
    # GETTERS                                         #
    ###################################################
    def get_matrix(self):
        """Return rotation matrix."""
        return self.matrix

    def get_angle_and_axis(self):
        """Return angle (in rad) and axis.

            Returns
            -------
            angle : float
                magnitude of Rotation
            axis : Vector
                Rotation axis (right-handed). If angle == 0, positive x-axis is
                returned.
        """
        # special case: no rotation
        if self == Rotation():
            angle = 0.
            axis = Vector((1., 0., 0.))
            return angle, axis

        # construct unit vector along rotation axis
        R = self.get_matrix()
        x = R[2, 1] - R[1, 2]
        y = R[0, 2] - R[2, 0]
        z = R[1, 0] - R[0, 1]
        axis = Vector((x, y, z)).unit_vector()

        # helper Vector
        w = Vector((z, x, y))

        # vector perpendicular to u
        v = axis.cross(w)
        
        # ========== magnitude =========================== #
        cos = 0.5 * (np.trace(R) - 1)
        acos = np.arccos(cos)

        # use sine to resolve ambiguity of acos
        # The following line of code is taken from
        # http://vhm.mathematik.uni-stuttgart.de/Vorlesungen/
        # Lineare_Algebra/Folien_Drehachse_und_Drehwinkel.pdf
        # (join the above two lines to get the proper URL)
        sin = v.dot(self.dot(axis))
        if sin < 0:
            angle = - acos
        else:
            angle = acos
        # ================================================ #

        return angle, axis

    def get_angle(self):
        """Return rotation angle (in rad)."""
        return self.get_angle_and_axis()[0]

    def get_axis(self):
        """Return rotation axis as a Vector."""
        return self.get_angle_and_axis()[1]

    ###################################################
    # APPLY                                           #
    ###################################################
    def apply_to(self, v):
        """Return rotated Vector."""
        assert isinstance(v, Vector)
        return self.dot(v)

class Transform(object):
    """A coordinate transform from one frame of reference to another.

        A Transform is a coordinate transform in 3D Euklidian space. It
        includes:
        - shift and 
        - rotation

        It does NOT include: stretching, shearing, reflection, projection.

        It can be easily applied to Vector by calling it with the Vector.
        No coordinate conversion by the user is necessary, even if the Vector
        is not cartesian (this is done internally).

        The Transform is represented by a Rotation and a shift (Vector object).


        Initialization
        ==============
        Tranform(shift, rotation)
        It is interpreted as a rotation before a shift. The optional
        `shift_first` parameter can be set to True to inverse this order.
    """
    def __init__(self, shift=Vector(), rotation=Rotation(), shift_first=False):
        """Initialize.

            Parameters
            ----------
            shift : Vector, optional
                Default: null-vector
            rotation : Rotation, optional
                Default: identity rotation
            shift_first : bool, optional
                If True, `shift` is applied before `rotation`, otherwise inverse.
                Default: False.
        """
        if not isinstance(shift, Vector):
            raise TypeError('shift must be a Vector.')
        if not isinstance(rotation, Rotation):
            raise TypeError('rotation must be a Rotation.')

        self.rotation = rotation
        if not shift_first:
            self.shift = shift
        else:
            self.shift = rotation.dot(shift)

    def __repr__(self):
        """Return string representation."""
        s = repr(self.shift)
        R = repr(self.rotation)
        return 'Transformation with shift by %s after rotation by\n%s' % (s, R)

    def __call__(self, v):
        return self.apply_to(v)

    ###################################################
    # UNARY OPERATOR                                  #
    ###################################################
    def inverse(self):
        """Return inverse Transform."""
        s = self.shift
        R = self.rotation
        return Transform(-s, R.inverse(), shift_first=True)
            
    ###################################################
    # BINARY OPERATIONS                               #
    ###################################################
    def after(self, other):
        """Return combined Transfrom."""
        # x'  = Bx + b
        # x'' = Ax' + a
        #     = A(Bx + b) + a
        #     = ABx + (Ab + a)
        #     = Rot(x) + shift
        A = self.rotation
        a = self.shift
        B = other.rotation
        b = other.shift
        
        shift = A.dot(b) + a
        rotation = A.dot(B)
        return Transform(shift, rotation)
    
    def before(self, other):
        """Return combined Transfrom."""
        return other.after(self)

    ###################################################
    # GETTERS                                         #
    ###################################################
    def get_shift(self):
        """Return shift Vector."""
        return self.shift

    def get_rotation(self):
        """Return Rotation object."""
        return self.rotation

    def apply_to(self, v):
        """Return transformed Vector."""
        assert isinstance(v, Vector)
        s = self.shift
        R = self.rotation
        return R.dot(v) + s

###################################################
# TESTING                                         #
###################################################
# The functions have all been tested.
#
if __name__ == '__main__':
    v1 = Vector((10, 0, 0))
    v2 = Vector((0, 2, 0))
    v3 = Vector((-4, 0, 0))
    v4 = Vector((0, -3, 15))

    v = [v1, v2, v3, v4]

    R = Rotation(45, v4)
    v5 = R(v1)

    T = Transform(v2, R)
    v6 = T(v4)

    print R
    print v1
    print v5
    print '-' * 12
    print T
    print v4
    print v6
