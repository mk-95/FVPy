from fieldbase import FieldBase

class SVolField(FieldBase):
    allowed_fields = ("XFXVolField","YFYVolField","SVolField")
    def __init__(self, shape, ghostcells=1):
        super().__init__( shape=shape, ghostcells=ghostcells)

class XVolField(FieldBase):
    allowed_fields = ("XVolField", "XFSVolField")
    def __init__(self, shape, ghostcells=1):
        super().__init__( shape=shape, ghostcells=ghostcells)

class YVolField(FieldBase):
    allowed_fields = ("YVolField", "YFSVolField")
    def __init__(self, shape, ghostcells=1):
        super().__init__( shape=shape, ghostcells=ghostcells)

class XFSVolField(FieldBase):
    allowed_fields = ("XFSVolField", "XVolField")
    def __init__(self, shape, ghostcells=1):
        super().__init__( shape=shape, ghostcells=ghostcells)

class YFSVolField(FieldBase):
    allowed_fields = ("YFSVolField", "YVolField")
    def __init__(self, shape, ghostcells=1):
        super().__init__( shape=shape, ghostcells=ghostcells)

class XFXVolField(FieldBase):
    allowed_fields = ("XFXVolField", "YFYVolField", "SVolField")
    def __init__(self, shape, ghostcells=1):
        super().__init__( shape=shape, ghostcells=ghostcells)

class YFXVolField(FieldBase):
    allowed_fields = ("YFXVolField", "XFYVolField")
    def __init__(self, shape, ghostcells=1):
        super().__init__( shape=shape, ghostcells=ghostcells)

class XFYVolField(FieldBase):
    allowed_fields = ("XFYVolField","YFXVolField")
    def __init__(self, shape, ghostcells=1):
        super().__init__( shape=shape, ghostcells=ghostcells)

class YFYVolField(FieldBase):
    allowed_fields = ("YFYVolField", "SVolField")
    def __init__(self, shape, ghostcells=1):
        super().__init__( shape=shape, ghostcells=ghostcells)
