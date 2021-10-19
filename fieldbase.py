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

    def __getitem__(self, item):
        return self.interior[item]

    def __setitem__(self, key, value):
        self.interior[key] = value

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


class FieldsContainer(numpy.lib.mixins.NDArrayOperatorsMixin):
    def __init__(self,domaininfo,FieldClass:FieldBase, number:int,initialization=None):
        self.__domain_info = domaininfo
        self.__fields_type = FieldClass
        self.__num = number
        self.__initialization = initialization

        self.__fields = np.array([self.__fields_type(self.__domain_info.shape) for _ in range(self.__num)],dtype=object)
        if self.__initialization !=None:
            for field, value in enumerate(self.__initialization):
                self.__fields[field]<<=value

    @property
    def type(self):
        return self.__fields_type.__name__
    @property
    def size(self):
        return self.__num

    @property
    def fields(self):
        return self.__fields
    @fields.setter
    def fields(self,value):
        self.__fields = value

    @property
    def interior(self):
        return np.array([field.interior for field in self.fields],dtype=object)

    @interior.setter
    def interior(self, other):
        if isinstance(other, list):
            for num in range(self.__num):
                self.fields[num].interior = other[num]
        else:
            for num in range(self.__num):
                self.fields[num].interior = other

    @property
    def xn(self):
        return np.array([field.xn for field in self.fields],dtype=object)

    @property
    def xp(self):
        return np.array([field.xp for field in self.fields],dtype=object)

    @property
    def yn(self):
        return np.array([field.yn for field in self.fields],dtype=object)

    @property
    def yp(self):
        return np.array([field.yp for field in self.fields],dtype=object)

    @property
    def xgp(self):
        return np.array([field.xgp for field in self.fields],dtype=object)

    @xgp.setter
    def xgp(self, other):
        result = np.zeros_like(self.xgp)
        result+=other
        for num in range(self.__num):
            self.fields[num].xgp = other[num]

    @property
    def xgn(self):
        return np.array([field.xgn for field in self.fields],dtype=object)

    @xgn.setter
    def xgn(self, other):
        result = np.zeros_like(self.xgn)
        result += other
        for num in range(self.__num):
            self.fields[num].xgn = other[num]

    @property
    def ygp(self):
        return np.array([field.ygp for field in self.fields],dtype=object)

    @ygp.setter
    def ygp(self, other):
        result = np.zeros_like(self.ygp)
        result += other
        for num in range(self.__num):
            self.fields[num].ygp = other[num]

    @property
    def ygn(self):
        return np.array([field.ygn for field in self.fields],dtype=object)

    @ygn.setter
    def ygn(self, other):
        result = np.zeros_like(self.ygn)
        result += other
        for num in range(self.__num):
            self.fields[num].ygn = other[num]

    @property
    def Ixgp(self):
        return np.array([field.Ixgp for field in self.fields],dtype=object)

    @property
    def Ixgn(self):
        return np.array([field.Ixgn for field in self.fields],dtype=object)

    @property
    def Iygp(self):
        return np.array([field.Iygp for field in self.fields],dtype=object)

    @property
    def Iygn(self):
        return np.array([field.Iygn for field in self.fields],dtype=object)


    def __getitem__(self, item):
        return self.__fields[item]

    def __setitem__(self, key, value):
        self.__fields[key] = value

    def __repr__(self):

        return f"{self.__class__.__name__}: {len(self.__fields)} {self.__fields_type.__name__}\n"

    def __array__(self):
        return self.__fields

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):

        if method == '__call__':
            all_inputs = []
            for input in inputs:
                if input.__class__.__name__ in self.__class__.__name__:
                    all_inputs.append(input.fields)
                else:
                    all_inputs.append(input)

            result = self.__class__(self.__domain_info,self.__fields_type,self.__num,None)
            result.fields = ufunc(*all_inputs, **kwargs)
            return result
        else:
            return NotImplemented

    def __ilshift__(self, other):
        # print('other:',other.type)
        # print('self:',self.__fields_type.allowed_fields)
        if isinstance(other,Number) :
            for num in range(self.__num):
                self.__fields[num] <<= other
        elif isinstance(other,list)or isinstance(other,np.ndarray):
            if len(other)==self.__num:
                for num in range(self.__num):
                    self.__fields[num] <<= other[num]
            else:
                raise ValueError("dimension mismatch {} into {}".format(len(other),self.__num))
        elif (other.type in self.__fields_type.allowed_fields) and (other.size == self.size) :
            self.__fields = other.__array__()

        return self
