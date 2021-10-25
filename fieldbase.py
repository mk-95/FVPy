import numpy
import numpy as np
from numbers import Number


class FieldBase(numpy.lib.mixins.NDArrayOperatorsMixin):

    def __init__(self, shape, ghostcells=1):
        if len(shape) != 2:
            raise ValueError("the shape {} has to be formed of two values (ny, nx), other dimensions are not supported".format(shape))
        for val in shape:
            if (val <=0) or (type(val)==float):
                raise ValueError("values contained in shape, {}, has to be a positive integer".format(val))
        if (type(ghostcells)!=int) or ghostcells<=0:
            raise ValueError("the value of ghostcells, {}, has to be an integer greater than zero".format(ghostcells))

        self.__original_shape = shape
        self.__ghostcells = ghostcells
        self.__shape = [n + 2 * ghostcells for n in list(shape)]

        self.__array = np.ndarray(self.__shape,dtype=np.float32,order="c")
        self.__array.fill(0)

    @property
    def shape(self):
        return self.__original_shape

    @property
    def ghostcells(self):
        return self.__ghostcells

    @property
    def array(self):
        return self.__array

    @property
    def interior(self):
        return self.__array[self.__ghostcells:-self.__ghostcells, self.__ghostcells:-self.__ghostcells]

    @interior.setter
    def interior(self,other):
        self.__array[self.__ghostcells:-self.__ghostcells, self.__ghostcells:-self.__ghostcells] = other

    @property
    def xn(self):
        return self.__array[self.__ghostcells:-self.__ghostcells, self.__ghostcells - 1: (self.__shape[1] - self.__ghostcells) - 1]

    @property
    def xp(self):
        return self.__array[self.__ghostcells:-self.__ghostcells, self.__ghostcells + 1: (self.__shape[1] - self.__ghostcells) + 1]

    @property
    def yn(self):
        return self.__array[self.__ghostcells - 1: (self.__shape[0] - self.__ghostcells) - 1, self.__ghostcells:-self.__ghostcells]

    @property
    def yp(self):
        return self.__array[self.__ghostcells + 1: (self.__shape[0] - self.__ghostcells) + 1, self.__ghostcells:-self.__ghostcells]

    @property
    def xgp(self):
        return self.__array[:, -self.__ghostcells:]

    @xgp.setter
    def xgp(self,other):
        self.__array[:, -self.__ghostcells:] = other

    @property
    def xgn(self):
        return self.__array[:, :self.__ghostcells]

    @xgn.setter
    def xgn(self,other):
        self.__array[:, :self.__ghostcells] = other

    @property
    def ygp(self):
        return self.__array[-self.__ghostcells:, :]

    @ygp.setter
    def ygp(self,other):
        self.__array[-self.__ghostcells:, :] = other

    @property
    def ygn(self):
        return self.__array[:self.__ghostcells, :]

    @ygn.setter
    def ygn(self,other):
        self.__array[:self.__ghostcells, :] = other

    @property
    def Ixgp(self):
        return self.__array[:, -2 * self.__ghostcells: -self.__ghostcells]

    @property
    def Ixgn(self):
        return self.__array[:, self.__ghostcells: 2 * self.__ghostcells]

    @property
    def Iygp(self):
        return self.__array[-2 * self.__ghostcells: -self.__ghostcells, :]
    @property
    def Iygn(self):
        return self.__array[self.__ghostcells: 2 * self.__ghostcells, :]

    def is_similar_to(self,other):
        if other.__class__.__name__ in self.__class__.allowed_fields:
            if (other.shape == self.shape) and (other.ghostcells == self.ghostcells):
                return True
            else:
                return False
        else:
            return False

    def __getitem__(self, item):
        return self.interior[item]

    def __setitem__(self, key, value):
        self.interior[key] = value

    def __repr__(self):

        return f"{self.__class__.__name__}\n {self.__array}"

    def __eq__(self, other):
        if self.is_similar_to(other):
            return np.equal(self.interior,other.interior)
        else:
            raise ValueError("{} and {} are not similar, thus cannot be compared.".format(self,other))

    def copy(self):
        return self.__copy__()

    def __copy__(self):
        new_copy = self.__class__(self.__original_shape,self.__ghostcells)
        new_copy.interior = self.interior
        new_copy.xgp = self.xgp
        new_copy.xgn = self.xgn
        new_copy.ygp = self.ygp
        new_copy.ygn = self.ygn

        return new_copy

    def __array__(self):

        return self.__array

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):

        if method == '__call__':
            all_inputs = []
            for input in  inputs:
                if self.is_similar_to(input):
                    all_inputs.append(input.interior)
                elif isinstance(input,Number):
                    all_inputs.append(input)
                elif isinstance(input,np.ndarray):
                    if self.shape == input.shape:
                        all_inputs.append(input)
                    else:
                        raise ValueError("Dimension mismatch: trying to assign {} to {}".format(input.shape,self.shape))
                else:
                    raise ValueError("operation between {} and {} are not allowed".format(self,input))
            result = self.__class__(self.__original_shape,self.__ghostcells)

            result.interior = ufunc(*all_inputs,**kwargs)
            return result
        else:
            return NotImplemented

    def __ilshift__(self, other):
        if self.is_similar_to(other):
            self.__array = other.__array__().copy()
            return self
        elif isinstance(other,Number):
            self.interior = other
            return self
        elif isinstance(other,np.ndarray):
            if self.shape == other.shape:
                self.interior = other.copy()
                return self
            else:
                raise ValueError("Dimension mismatch: trying to assign {} into {}".format(other.shape,self.shape))
        else:
            raise ValueError("operation between {} and {} is not allowed".format(self, other))
