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
    def Interp(self,direction:str):
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


class BCOpt(Operators):
    # todo: generalize as a function of the ghost cells
    # the shape of the dict has to be as follows:
    # dict = {"dirichlet":{"x+":{phi:2,press:3},
    #                      "x-":{phi:2,press:-1},
    #                      "y+":{},
    #                      "y-":{}
    #                      }
    #         }

    def __parse_Bcs(self,bc_dict:dict):
        self.__bcs = bc_dict

    def apply_bcs(self,bc_dict:dict):
        self.__parse_Bcs(bc_dict)
        # deal with periodicity

        bcs_types = self.__bcs.keys()
        if "dirichlet" in bcs_types:
            self.__apply_dirichlet()
        if "neuman" in bcs_types:
            self.__apply_neuman()

    def __apply_dirichlet(self):
        dirichlet_data = self.__bcs["dirichlet"]
        for direction in dirichlet_data.keys():
            if direction == "x+":
                dirichlet_xp_data = dirichlet_data["x+"]
                for tuple in dirichlet_xp_data:
                    field, bcvalue = tuple
                    field_type_name = field.__class__.__name__
                    if field_type_name == 'FieldsContainer':
                        field_type_name = field.type
                    if field_type_name == SVolField.__name__:
                        field.xgp = 2*bcvalue - field.Ixgp
                    elif field_type_name == XVolField.__name__:
                        field.xgp = bcvalue
                    elif field_type_name == YVolField.__name__:
                        field.xgp = 2*bcvalue - field.Ixgp
                    else:
                        return NotImplementedError
            elif direction == "x-":
                dirichlet_xn_data = dirichlet_data["x-"]
                for tuple in dirichlet_xn_data:
                    field, bcvalue = tuple
                    field_type_name = field.__class__.__name__
                    if field_type_name == 'FieldsContainer':
                        field_type_name = field.type
                    if field_type_name == SVolField.__name__:
                        field.xgn = 2*bcvalue - field.Ixgn
                    elif field_type_name == XVolField.__name__:
                        field.xgn = bcvalue # need to fix this by accessing the boundary value  directly.
                    elif field_type_name == YVolField.__name__:
                        field.xgn = 2*bcvalue - field.Ixgn
                    else:
                        return NotImplementedError

            elif direction == "y+":
                dirichlet_yp_data = dirichlet_data["y+"]
                for tuple in dirichlet_yp_data:
                    field, bcvalue = tuple
                    field_type_name = field.__class__.__name__
                    if field_type_name == 'FieldsContainer':
                        field_type_name = field.type
                    if field_type_name == SVolField.__name__:
                        field.ygp = 2*bcvalue - field.Iygp
                    elif field_type_name == XVolField.__name__:
                        field.ygp = 2*bcvalue - field.Iygp
                    elif field_type_name == YVolField.__name__:
                        field.xgn = bcvalue
                    else:
                        return NotImplementedError

            elif direction == "y-":
                dirichlet_yn_data = dirichlet_data["y-"]
                for tuple in dirichlet_yn_data:
                    field, bcvalue = tuple
                    field_type_name = field.__class__.__name__
                    if field_type_name == 'FieldsContainer':
                        field_type_name = field.type
                    if field_type_name == SVolField.__name__:
                        field.ygn = 2*bcvalue - field.Iygn
                    elif field_type_name == XVolField.__name__:
                        field.ygn = 2*bcvalue - field.Iygn
                    elif field_type_name == YVolField.__name__:
                        field.xgn = bcvalue # need to fix this by accessing the boundary value  directly.
                    else:
                        return NotImplementedError

    def __apply_neuman(self):
        # I am using one sided gradient to the east
        neuman_data = self.__bcs["neuman"]
        for direction in neuman_data.keys():
            if direction == "x+":
                dx = self.domain_description.dx
                neuman_xp_data = neuman_data["x+"]
                for tuple in neuman_xp_data:
                    field, bcvalue = tuple
                    field_type_name = field.__class__.__name__
                    if field_type_name == 'FieldsContainer':
                        field_type_name = field.type
                    if field_type_name == SVolField.__name__:
                        field.xgp = dx * bcvalue + field.Ixgp
                    elif field_type_name == XVolField.__name__:
                        field.xgp = dx * bcvalue + field.Ixgp
                    elif field_type_name == YVolField.__name__:
                        field.xgp = dx * bcvalue + field.Ixgp
                    else:
                        return NotImplementedError
            elif direction == "x-":
                dx = self.domain_description.dx
                neuman_xn_data = neuman_data["x-"]
                for tuple in neuman_xn_data:
                    field, bcvalue = tuple
                    field_type_name = field.__class__.__name__
                    if field_type_name == 'FieldsContainer':
                        field_type_name = field.type
                    if field_type_name == SVolField.__name__:
                        field.xgn = field.Ixgn -  dx * bcvalue
                    elif field_type_name == XVolField.__name__:
                        field.xgn = field.Ixgn -  dx * bcvalue
                    elif field_type_name == YVolField.__name__:
                        field.xgn = field.Ixgn -  dx * bcvalue
                    else:
                        return NotImplementedError

            elif direction == "y+":
                dy = self.domain_description.dy
                neuman_yp_data = neuman_data["y+"]
                for tuple in neuman_yp_data:
                    field, bcvalue = tuple
                    field_type_name = field.__class__.__name__
                    if field_type_name == 'FieldsContainer':
                        field_type_name = field.type
                    if field_type_name == SVolField.__name__:
                        field.ygp = dy * bcvalue + field.Iygp
                    elif field_type_name == XVolField.__name__:
                        field.ygp = dy * bcvalue + field.Iygp
                    elif field_type_name == YVolField.__name__:
                        field.xgn = dy * bcvalue + field.Iygp
                    else:
                        return NotImplementedError

            elif direction == "y-":
                dy = self.domain_description.dy
                neuman_yn_data = neuman_data["y-"]
                for tuple in neuman_yn_data:
                    field, bcvalue = tuple
                    field_type_name = field.__class__.__name__
                    if field_type_name == 'FieldsContainer':
                        field_type_name = field.type
                    if field_type_name == SVolField.__name__:
                        field.ygn = field.Iygn - dy * bcvalue
                    elif field_type_name == XVolField.__name__:
                        field.ygn = field.Iygn - dy * bcvalue
                    elif field_type_name == YVolField.__name__:
                        field.xgn = field.Iygn - dy * bcvalue
                    else:
                        return NotImplementedError