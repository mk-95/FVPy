import numpy
import numpy as np
from numbers import Number


class FieldBase(numpy.lib.mixins.NDArrayOperatorsMixin):

    def __init__(self, shape, ghostcells=1):

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

    def __repr__(self):

        return f"{self.__class__.__name__}\n {self.__array}"

    def __array__(self):

        return self.__array

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):

        if method == '__call__':
            all_inputs = []
            for input in  inputs:
                if input.__class__.__name__ in self.__class__.allowed_fields:
                    all_inputs.append(input.interior)
                elif isinstance(input,Number):
                    all_inputs.append(input)
                elif isinstance(input,np.ndarray):
                    all_inputs.append(input)
                else:
                    return NotImplemented
            result = self.__class__(self.__original_shape)

            kwargs_keys = kwargs.keys()

            if ('xgp' in kwargs_keys) and (kwargs['xgp']==True):
                result.interior = all_inputs[0]
                result.xgp =  ufunc(inputs[0].xgp,*all_inputs[1:],**{})
            elif ('xgn' in kwargs_keys) and (kwargs['xgn']==True):
                result.interior = all_inputs[0]
                result.xgn =  ufunc(inputs[0].xgn,*all_inputs[1:],**{})
            elif ('ygp' in kwargs_keys) and (kwargs['ygp']==True):
                result.interior = all_inputs[0]
                result.ygp =  ufunc(inputs[0].ygp,*all_inputs[1:],**{})
            elif ('ygn' in kwargs_keys) and (kwargs['ygn']==True):
                result.interior = all_inputs[0]
                result.ygn =  ufunc(inputs[0].ygn,*all_inputs[1:],**{})

            else:
                result.interior = ufunc(*all_inputs,**kwargs)
            return result
        else:
            return NotImplemented

    def __ilshift__(self, other):
        if other.__class__.__name__ in self.__class__.allowed_fields:
            self.__array = other.__array__()
        elif isinstance(other,Number) or isinstance(other,np.ndarray):
            self.interior = other
        return self