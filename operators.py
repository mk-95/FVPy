import numpy as np
from fieldbase import FieldBase
from fields import SVolField, XVolField, YVolField
from fields import XFSVolField, YFSVolField
from fields import XFXVolField, YFXVolField
from fields import XFYVolField, YFYVolField

from domain_description import DomainDescription
class Operators:
    def __init__(self, domainDescription:DomainDescription):
        self.__domain_description = domainDescription

    @property
    def domain_description(self):
        return self.__domain_description

class SpatialOpt(Operators):
    def Interp(self,direction:str)->FieldBase:
        if direction=="e":
            class InterpEast:
                def __init__(self):
                    pass
                def __call__(self,field:FieldBase)->FieldBase:
                    field_type_name = field.__class__.__name__
                    if field_type_name == SVolField.__name__:
                        result = XFSVolField(field.shape, field.ghostcells)
                    elif field_type_name == XVolField.__name__:
                        result = XFXVolField(field.shape, field.ghostcells)
                    elif field_type_name == YVolField.__name__:
                        result = XFYVolField(field.shape, field.ghostcells)
                    else:
                        return NotImplementedError

                    result.interior = 0.5 * (field.xp + field.interior)
                    return result

            return InterpEast()

        elif direction=="w":
            class InterpWest:
                def __init__(self):
                    pass
                def __call__(self,field:FieldBase)->FieldBase:
                    field_type_name = field.__class__.__name__
                    if field_type_name == SVolField.__name__:
                        result = XFSVolField(field.shape, field.ghostcells)
                    elif field_type_name == XVolField.__name__:
                        result = XFXVolField(field.shape, field.ghostcells)
                    elif field_type_name == YVolField.__name__:
                        result = XFYVolField(field.shape, field.ghostcells)
                    else:
                        return NotImplementedError

                    result.interior = 0.5 * (field.xn + field.interior)
                    return result

            return InterpWest()

        elif direction == "n":
            class InterpNorth:
                def __init__(self):
                    pass
                def __call__(self, field: FieldBase) -> FieldBase:
                    field_type_name = field.__class__.__name__
                    if field_type_name == SVolField.__name__:
                        result = YFSVolField(field.shape, field.ghostcells)
                    elif field_type_name == XVolField.__name__:
                        result = YFXVolField(field.shape, field.ghostcells)
                    elif field_type_name == YVolField.__name__:
                        result = YFYVolField(field.shape, field.ghostcells)
                    else:
                        return NotImplementedError

                    result.interior = 0.5 * (field.yp + field.interior)
                    return result

            return InterpNorth()

        elif direction == "s":
            class InterpSouth:
                def __init__(self):
                    pass
                def __call__(self, field: FieldBase) -> FieldBase:
                    field_type_name = field.__class__.__name__
                    if field_type_name == SVolField.__name__:
                        result = YFSVolField(field.shape, field.ghostcells)
                    elif field_type_name == XVolField.__name__:
                        result = YFXVolField(field.shape, field.ghostcells)
                    elif field_type_name == YVolField.__name__:
                        result = YFYVolField(field.shape, field.ghostcells)
                    else:
                        return NotImplementedError

                    result.interior = 0.5 * (field.yn + field.interior)
                    return result

            return InterpSouth()

        else:
            return NotImplementedError



    def OneSidedGrad(self,direction:str): #-> should create interface for these operators
        if direction=="e":
            class GradEast:
                def __init__(self, domain_description):
                    self.domain_description = domain_description
                def __call__(self,field:FieldBase)->FieldBase:
                    dx = self.domain_description.dx
                    field_type_name = field.__class__.__name__
                    if field_type_name == SVolField.__name__:
                        result = XVolField(field.shape, field.ghostcells)
                    elif field_type_name == XVolField.__name__:
                        result = SVolField(field.shape, field.ghostcells)
                    elif field_type_name == YVolField.__name__:
                        result = XFYVolField(field.shape, field.ghostcells)
                    else:
                        return NotImplementedError

                    result.interior = (field.xp - field.interior) / dx
                    return result

            return GradEast(self.domain_description)

        elif direction=="w":
            class GradWest:
                def __init__(self, domain_description):
                    self.domain_description = domain_description
                def __call__(self,field:FieldBase)->FieldBase:
                    dx = self.domain_description.dx
                    field_type_name = field.__class__.__name__
                    if field_type_name == SVolField.__name__:
                        result = XVolField(field.shape, field.ghostcells)
                    elif field_type_name == XVolField.__name__:
                        result = SVolField(field.shape, field.ghostcells)
                    elif field_type_name == YVolField.__name__:
                        result = XFYVolField(field.shape, field.ghostcells)
                    else:
                        return NotImplementedError

                    result.interior = (field.xn - field.interior) / dx
                    return result

            return GradWest(self.domain_description)

        elif direction == "n":
            class GradNorth:
                def __init__(self, domain_description):
                    self.domain_description = domain_description

                def __call__(self, field: FieldBase) -> FieldBase:
                    dy = self.domain_description.dy
                    field_type_name = field.__class__.__name__
                    if field_type_name == SVolField.__name__:
                        result = YVolField(field.shape, field.ghostcells)
                    elif field_type_name == XVolField.__name__:
                        result = YFXVolField(field.shape, field.ghostcells)
                    elif field_type_name == YVolField.__name__:
                        result = SVolField(field.shape, field.ghostcells)
                    else:
                        return NotImplementedError

                    result.interior = (field.yp - field.interior)/dy
                    return result

            return GradNorth(self.domain_description)

        elif direction == "s":
            class GradSouth:
                def __init__(self, domain_description):
                    self.domain_description = domain_description

                def __call__(self, field: FieldBase) -> FieldBase:
                    dy = self.domain_description.dy
                    field_type_name = field.__class__.__name__
                    if field_type_name == SVolField.__name__:
                        result = YVolField(field.shape, field.ghostcells)
                    elif field_type_name == XVolField.__name__:
                        result = YFXVolField(field.shape, field.ghostcells)
                    elif field_type_name == YVolField.__name__:
                        result = SVolField(field.shape, field.ghostcells)
                    else:
                        return NotImplementedError

                    result.interior = (field.yn - field.interior)/dy
                    return result

            return GradSouth(self.domain_description)

        else:
            return NotImplementedError
