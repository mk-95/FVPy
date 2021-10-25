import numpy
import numpy as np
from numbers import Number
from fields import SVolField,XVolField,YVolField,XFSVolField,YFSVolField,XFXVolField,YFXVolField,XFYVolField,YFYVolField

class FieldsContainer(numpy.lib.mixins.NDArrayOperatorsMixin):
    def __init__(self, shape, FieldClass, number: int, ghostcells=1):
        supported_types =  (SVolField,XVolField,YVolField,XFSVolField,YFSVolField,XFXVolField,YFXVolField,XFYVolField,YFYVolField)
        if not (FieldClass in supported_types):
            raise ValueError("{} is not a type of the supported fields {}".format(FieldClass,supported_types))
        if (number <= 1) or (not isinstance(number,int)) :
            raise ValueError("the number of fields {} has to be a positive integer greater than 1 ".format(number))

        self.__shape = shape
        self.__ghostcells = ghostcells
        self.__fields_type = FieldClass
        self.__num = number

        self.__fields = np.array([self.__fields_type(self.__shape,self.__ghostcells) for _ in range(self.__num)],dtype=object)

    @property
    def type(self):
        return self.__fields_type.__name__
    @property
    def shape(self):
        return self.__shape

    @property
    def ghostcells(self):
        return self.__ghostcells

    @property
    def size(self):
        return self.__num

    @property
    def fields(self):
        return self.__fields

    @property
    def interior(self):
        return np.array([field.interior for field in self.fields])

    @interior.setter
    def interior(self, other) :
        if isinstance(other, list) or (isinstance(other,np.ndarray) and (other.shape == (self.size,))):
            for num in range(self.__num):
                self.fields[num].interior = other[num]
        else:
            for num in range(self.__num):
                self.fields[num].interior = other

    @property
    def xn(self):
        return np.array([field.xn for field in self.fields], dtype=object)

    @property
    def xp(self):
        return np.array([field.xp for field in self.fields], dtype=object)

    @property
    def yn(self):
        return np.array([field.yn for field in self.fields], dtype=object)

    @property
    def yp(self):
        return np.array([field.yp for field in self.fields], dtype=object)

    @property
    def xgp(self):
        return np.array([field.xgp for field in self.fields], dtype=object)

    @xgp.setter
    def xgp(self, other):

        if isinstance(other, list) or (isinstance(other,np.ndarray) and (other.shape == (self.size,))):
            for num in range(self.__num):
                self.fields[num].xgp = other[num]
        else:
            for num in range(self.__num):
                self.fields[num].xgp = other

    @property
    def xgn(self):
        return np.array([field.xgn for field in self.fields], dtype=object)

    @xgn.setter
    def xgn(self, other):
        if isinstance(other, list) or (isinstance(other,np.ndarray) and (other.shape == (self.size,))):
            for num in range(self.__num):
                self.fields[num].xgn = other[num]
        else:
            for num in range(self.__num):
                self.fields[num].xgn = other

    @property
    def ygp(self):
        return np.array([field.ygp for field in self.fields], dtype=object)

    @ygp.setter
    def ygp(self, other):
        if isinstance(other, list) or (isinstance(other, np.ndarray) and (other.shape == (self.size,))):
            for num in range(self.__num):
                self.fields[num].ygp = other[num]
        else:
            for num in range(self.__num):
                self.fields[num].ygp = other

    @property
    def ygn(self):
        return np.array([field.ygn for field in self.fields], dtype=object)

    @ygn.setter
    def ygn(self, other):
        if isinstance(other, list) or (isinstance(other, np.ndarray) and (other.shape == (self.size,))):
            for num in range(self.__num):
                self.fields[num].ygn = other[num]
        else:
            for num in range(self.__num):
                self.fields[num].ygn = other

    @property
    def Ixgp(self):
        return np.array([field.Ixgp for field in self.fields], dtype=object)

    @property
    def Ixgn(self):
        return np.array([field.Ixgn for field in self.fields], dtype=object)

    @property
    def Iygp(self):
        return np.array([field.Iygp for field in self.fields], dtype=object)

    @property
    def Iygn(self):
        return np.array([field.Iygn for field in self.fields], dtype=object)

    def point_reshape(self):
        return np.array([field.interior.ravel() for field in self.fields]).transpose()

    def flatten(self):
        return np.array([field.interior.ravel().copy() for field in self.fields]).ravel()

    def is_similar_to(self,other):
        if isinstance(other,self.__class__):
            if (other.type in self.__fields_type.allowed_fields):
                if (other.size == self.size) and (other.shape == self.shape) and (other.ghostcells == self.ghostcells):
                    return True
                else:
                    return False
            else:
                return False
        else:
            return False

    def __eq__(self, other):
        if self.is_similar_to(other):
            return np.equal(self.interior,other.interior)
        else:
            raise ValueError("{} and {} are not similar, thus cannot be compared.".format(self,other))


    def __copy__(self):
        new_copy = self.__class__(self.shape,self.__fields_type,self.__num,self.__ghostcells)
        for num in range(self.__num):
            new_copy.__array__()[num] <<= self.fields[num]
            new_copy.__array__()[num].xgp = self.__array__()[num].xgp
            new_copy.__array__()[num].xgn = self.__array__()[num].xgn
            new_copy.__array__()[num].ygp = self.__array__()[num].ygp
            new_copy.__array__()[num].ygn = self.__array__()[num].ygn

        return new_copy

    def copy(self):
        return self.__copy__()


    def __getitem__(self, item):
        return self.__fields[item]

    def __setitem__(self, key, value):
        self.__fields[key] <<= value

    def __repr__(self):

        return f"{self.__class__.__name__}: {len(self.__fields)} {self.__fields_type.__name__}\n"

    def __array__(self):
        return self.__fields

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):

        if method == '__call__':
            all_inputs = []
            for input in inputs:
                if self.is_similar_to(input):
                    all_inputs.append(input.fields)
                else:
                    all_inputs.append(input)

            result = self.__class__(self.shape, self.__fields_type, self.__num)
            result <<= ufunc(*all_inputs, **kwargs)
            return result
        else:
            return NotImplemented

    def __ilshift__(self, other):
        # allows a list of ndarrays, numpy array of arrays, number, list of number or numpy array of numbers and fieldcontainer that is similar
        if self.is_similar_to(other):
            for num, other_field in enumerate(other.fields):
                self.__fields[num] <<= other_field

        elif isinstance(other, Number):
            for num in range(self.__num):
                self.__fields[num] <<= other

        elif isinstance(other, list):
            if len(other) != self.__num:
                raise ValueError("Dimension mismatch: trying to assign a list of size {} to a fieldsContainer of size {}".format(len(other),self.__num))
            else:
                for num, val in enumerate(other):
                    self.__fields[num] <<=val

        elif isinstance(other,np.ndarray):
            shape = other.shape
            if shape != (self.__num,self.shape[0],self.shape[1]):
                if  (shape == (self.__num,)) and (other.dtype==object):
                    for num, val in enumerate(other):
                        self.__fields[num] <<= val
                elif (shape == (self.shape[0]*self.shape[1],self.__num)):
                    # assign point_value form
                    field_val = other.transpose()
                    for num, val in enumerate(field_val):
                        self.__fields[num] <<= val.reshape(self.shape)
                else:
                    raise ValueError(
                        "Dimension mismatch: trying to assign a numpy array of shape {} to a fieldsContainer of size {}".format(
                            shape, (self.__num,)))
            else:
                for num, val in enumerate(other):
                    self.__fields[num] <<= val

        else:
            raise NotImplementedError
        return self
