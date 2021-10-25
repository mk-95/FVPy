import unittest
from ddt import ddt, data
from fieldcontainer import FieldsContainer
from fields import SVolField,XVolField,YVolField,XFSVolField,YFSVolField,XFXVolField,YFXVolField,XFYVolField,YFYVolField
import numpy as np
import random

# we will conduct the testing of FieldBase with the
# self.FieldType type

@ddt
class TestSVolContainer(unittest.TestCase):
    FieldType = SVolField
    NonSimilarFieldsTypes = [XVolField,YVolField,XFSVolField,YFSVolField,YFXVolField,XFYVolField]
    SimilarFieldsTypes = [XFXVolField,YFYVolField,SVolField]

    @data((1, 3, 1,1), (3, -1, 1,10), (3, 3, -1,10), (1, 3.5, 2,10), (3.5, 1, 2,10), (3, 3, 2.5,10), (3, 3, (2, 3),10))
    def test_init_exceptions(self, value):
        ny, nx, ghostcell, number = value
        with self.assertRaises(Exception) as exception:
            test_field_container = FieldsContainer(shape=(ny, nx),FieldClass=self.FieldType,number=number, ghostcells=ghostcell)

        self.assertTrue(type(exception.exception) in [ValueError],
                        "{} was not thrown with the following input ({})".format(ValueError, value))

    @data((1, 3, 1,2), (3, 1, 1,3), (3, 3, 1,4), (1, 3, 2,4), (3, 1, 2, 2), (3, 3, 2,3))
    def test_init(self, value):
        ny, nx, ghostcell, fields_num = value
        test_field_container = FieldsContainer(shape=(ny, nx),FieldClass=self.FieldType,number=fields_num, ghostcells=ghostcell)

        # check the shape of the field
        expected_size = fields_num
        actual_size = test_field_container.__array__().shape[0]

        self.assertEqual(expected_size, actual_size,
                         "Expected size {} is different from the actual size {}.".format(
                             expected_size, actual_size))

        # check that every single field is of the FieldClass type
        for field in test_field_container.__array__():
            self.assertTrue(isinstance(field,self.FieldType),"Expected field type {}, and the actual field type {}".format(field,self.FieldType))


    @data((1, 3, 1,3), (3, 1, 1,2), (3, 3, 1,4), (1, 3, 2,3), (3, 1, 2,5), (3, 3, 2,2))
    def test_size(self, value):
        ny, nx, ghostcell, num_fields = value
        test_field = FieldsContainer(shape=(ny, nx),FieldClass=self.FieldType,number=num_fields, ghostcells=ghostcell)
        expected_size_value = num_fields
        actual_size_value = test_field.size

        self.assertEqual(expected_size_value, actual_size_value,
                         "Expected size {} is different from the actual size {}".format(expected_size_value,
                                                                                          actual_size_value))


    @data((1, 3, 1,3), (3, 1, 1,2), (3, 3, 1,4), (1, 3, 2,3), (3, 1, 2,5), (3, 3, 2,2))
    def test_fields(self, value):
        # returns an array of objects of all the field
        ny, nx, ghostcell, num_fields = value
        test_field_container = FieldsContainer(shape=(ny, nx),FieldClass=self.FieldType,number=num_fields, ghostcells=ghostcell)

        expected_array = np.array([self.FieldType(shape=(ny, nx),ghostcells=ghostcell) for _ in range(num_fields)],dtype=object)
        actual_array = test_field_container.fields

        for expected, actual in zip(expected_array,actual_array):
            self.assertTrue(np.all(expected==actual),"the expected array {} is different from the actual array {}".format(
                                          expected_array, actual_array))

    @data(1.12560, 2.01452, 3, 0.0, -1, -3.5, -0.112)
    def test_interior(self, value):
        ny, nx, ghostcells, fields_num = (3, 3, 1, 2)

        decimal = 7  # compare to how many decimal value

        test_field_container = FieldsContainer(shape=(ny, nx), FieldClass=self.FieldType, number=fields_num,
                                               ghostcells=ghostcells)
        test_empty_field = self.FieldType(shape=(ny, nx), ghostcells=ghostcells)

        # check if the initial fields are empty
        for field in test_field_container.fields:
            self.assertTrue(np.all(test_empty_field==field), "the initial value of the fields in {} are different from {}".format(test_field_container,test_empty_field))

        # testing the setter on different value
        expected_interior_array = np.array([value * np.ones((ny,nx)) for _ in range(fields_num)])
        # set the interior value of the inner array
        test_field_container.interior = value
        actual_interiro_array = test_field_container.interior
        np.testing.assert_almost_equal(expected_interior_array, actual_interiro_array, decimal=decimal, err_msg=
        "the expected interior initial array {} is "
        "different from the actual initial array {}"
                                       .format(expected_interior_array, actual_interiro_array))
        # # make sure that the ghost cells still have the value of zero
        expected_xgp = np.zeros(shape=(ny + 2 * ghostcells, ghostcells))
        expected_xgn = np.zeros(shape=(ny + 2 * ghostcells, ghostcells))
        expected_ygn = np.zeros(shape=(ghostcells, nx + 2 * ghostcells))
        expected_ygp = np.zeros(shape=(ghostcells, nx + 2 * ghostcells))
        for field in test_field_container.fields:
            actual_xgp = field.xgp
            actual_xgn = field.xgn
            actual_ygn = field.ygn
            actual_ygp = field.ygp
            np.testing.assert_almost_equal(expected_xgn, actual_xgn, decimal=decimal,
                                           err_msg="expected xgn {} is different from actual xgn {}".format(expected_xgn,
                                                                                                            actual_xgn))
            np.testing.assert_almost_equal(expected_xgp, actual_xgp, decimal=decimal,
                                           err_msg="expected xgp {} is different from actual xgp {}".format(expected_xgp,
                                                                                                            actual_xgp))
            np.testing.assert_almost_equal(expected_ygn, actual_ygn, decimal=decimal,
                                           err_msg="expected ygn {} is different from actual ygn {}".format(expected_ygn,
                                                                                                            actual_ygn))
            np.testing.assert_almost_equal(expected_ygp, actual_ygp, decimal=decimal,
                                           err_msg="expected ygp {} is different from actual ygp {}".format(expected_ygp,
                                                                                                            actual_ygp))

        # testing with a list of inputs
        # testing the setter on different value
        expected_interior_array = np.array([value * np.ones((ny, nx)) for _ in range(fields_num)])
        # set the interior value of the inner array
        test_field_container.interior = [value,value,value]
        actual_interiro_array = test_field_container.interior
        np.testing.assert_almost_equal(expected_interior_array, actual_interiro_array, decimal=decimal, err_msg=
        "the expected interior initial array {} is "
        "different from the actual initial array {}"
                                       .format(expected_interior_array, actual_interiro_array))

        # testing the setter on different value
        expected_interior_array = np.array([i*value * np.ones((ny, nx)) for i in range(fields_num)])
        # set the interior value of the inner array
        test_field_container.interior = [i*value for i in range(fields_num)]
        actual_interiro_array = test_field_container.interior
        np.testing.assert_almost_equal(expected_interior_array, actual_interiro_array, decimal=decimal, err_msg=
        "the expected interior initial array {} is "
        "different from the actual initial array {}"
                                       .format(expected_interior_array, actual_interiro_array))

    @data(1, -2.0, 3.562, 0.0, -1, -3.5, -0.112)
    def test_xn(self, value):
        interior_value = value
        decimal = 7
        ny, nx, ghostcells, fields_num = (3, 3, 1,4)
        test_field = self.FieldType(shape=(ny, nx), ghostcells=ghostcells)
        test_field.interior = interior_value

        test_field_container = FieldsContainer(shape=(ny,nx),FieldClass=self.FieldType,number=fields_num,ghostcells=ghostcells)
        test_field_container.interior = interior_value
        expected_xn = np.array([test_field.xn for _ in range(fields_num)])
        actual_xn = test_field_container.xn
        np.testing.assert_almost_equal(expected_xn, actual_xn, decimal=decimal,
                                       err_msg="expected xn {} is different from actual xn {}".format(expected_xn,
                                                                                                      actual_xn))

    @data(1, -2.0, 3.562, 0.0, -1, -3.5, -0.112)
    def test_xp(self, value):
        interior_value = value
        decimal = 7
        ny, nx, ghostcells, fields_num = (3, 3, 1,4)
        test_field = self.FieldType(shape=(ny, nx), ghostcells=ghostcells)
        test_field.interior = interior_value

        test_field_container = FieldsContainer(shape=(ny, nx), FieldClass=self.FieldType, number=fields_num,
                                               ghostcells=ghostcells)
        test_field_container.interior = interior_value

        expected_xp = np.array([test_field.xp for _ in range(fields_num)])
        actual_xp = test_field_container.xp

        np.testing.assert_almost_equal(expected_xp, actual_xp, decimal=decimal,
                                       err_msg="expected xp {} is different from actual xp {}".format(expected_xp,
                                                                                                      actual_xp))

    @data(1, -2.0, 3.562, 0.0, -1, -3.5, -0.112)
    def test_yn(self, value):
        interior_value = value
        decimal = 7
        ny, nx, ghostcells, fields_num = (3, 3, 1,4)
        test_field = self.FieldType(shape=(ny, nx), ghostcells=ghostcells)
        test_field.interior = interior_value

        test_field_container = FieldsContainer(shape=(ny, nx), FieldClass=self.FieldType, number=fields_num,
                                               ghostcells=ghostcells)
        test_field_container.interior = interior_value

        expected_yn = np.array([test_field.yn for _ in range(fields_num)])
        actual_yn = test_field_container.yn
        np.testing.assert_almost_equal(expected_yn, actual_yn, decimal=decimal,
                                       err_msg="expected yn {} is different from actual yn {}".format(expected_yn,
                                                                                                      actual_yn))

    @data(1, -2.0, 3.562, 0.0, -1, -3.5, -0.112)
    def test_yp(self, value):
        interior_value = value
        decimal = 7
        ny, nx, ghostcells, fields_num = (3, 3, 1,4)
        test_field = self.FieldType(shape=(ny, nx), ghostcells=ghostcells)
        test_field.interior = interior_value

        test_field_container = FieldsContainer(shape=(ny, nx), FieldClass=self.FieldType, number=fields_num,
                                               ghostcells=ghostcells)
        test_field_container.interior = interior_value

        expected_yp =  np.array([test_field.yp for _ in range(fields_num)])
        actual_yp = test_field_container.yp
        np.testing.assert_almost_equal(expected_yp, actual_yp, decimal=decimal,
                                       err_msg="expected yp {} is different from actual yp {}".format(expected_yp,
                                                                                                      actual_yp))

    @data(1, -2.0, 3.562, 0.0, -1, -3.5, -0.112)
    def test_xgn(self, value):
        ghost_value = value
        decimal = 7
        ny, nx, ghostcells, fields_num = (3, 3, 1,4)
        test_field = self.FieldType(shape=(ny, nx), ghostcells=ghostcells)

        test_field_container = FieldsContainer(shape=(ny, nx), FieldClass=self.FieldType, number=fields_num,
                                               ghostcells=ghostcells)

        # make sure that the initial ghost value is 0
        expected_ghostvalue_xgn = np.array([test_field.xgn for _ in range(fields_num)])
        actual_ghostvalue_xgn = test_field_container.xgn
        np.testing.assert_almost_equal(expected_ghostvalue_xgn, actual_ghostvalue_xgn, decimal=decimal,
                                       err_msg="expected xgn {} is different from actual xgn {}".format(
                                           expected_ghostvalue_xgn,
                                           actual_ghostvalue_xgn))

        # set the ghost value using a single value
        test_field.xgn = ghost_value
        test_field_container.xgn = ghost_value
        expected_ghostvalue_xgn = np.array([test_field.xgn for _ in range(fields_num)])
        actual_ghostvalue_xgn = test_field_container.xgn

        np.testing.assert_almost_equal(expected_ghostvalue_xgn, actual_ghostvalue_xgn, decimal=decimal,
                                       err_msg="expected xgn {} is different from actual xgn {}".format(
                                           expected_ghostvalue_xgn,
                                           actual_ghostvalue_xgn))

        # set the ghost value using a numpy array
        test_field.xgn = ghost_value
        test_field_container.xgn = np.array([ghost_value for _ in range(fields_num)])
        expected_ghostvalue_xgn = np.array([test_field.xgn for _ in range(fields_num)])
        actual_ghostvalue_xgn = test_field_container.xgn

        np.testing.assert_almost_equal(expected_ghostvalue_xgn, actual_ghostvalue_xgn, decimal=decimal,
                                       err_msg="expected xgn {} is different from actual xgn {}".format(
                                           expected_ghostvalue_xgn,
                                           actual_ghostvalue_xgn))

        # set the ghost value using a list of value
        test_field.xgn = ghost_value
        test_field_container.xgn = [ghost_value for _ in range(fields_num)]
        expected_ghostvalue_xgn = np.array([test_field.xgn for _ in range(fields_num)])
        actual_ghostvalue_xgn = test_field_container.xgn

        np.testing.assert_almost_equal(expected_ghostvalue_xgn, actual_ghostvalue_xgn, decimal=decimal,
                                       err_msg="expected xgn {} is different from actual xgn {}".format(
                                           expected_ghostvalue_xgn,
                                           actual_ghostvalue_xgn))
    @data(1, -2.0, 3.562, 0.0, -1, -3.5, -0.112)
    def test_xgp(self, value):
        ghost_value = value
        decimal = 7
        ny, nx, ghostcells, fields_num = (3, 3, 1, 4)
        test_field = self.FieldType(shape=(ny, nx), ghostcells=ghostcells)

        test_field_container = FieldsContainer(shape=(ny, nx), FieldClass=self.FieldType, number=fields_num,
                                               ghostcells=ghostcells)

        # make sure that the initial ghost value is 0
        expected_ghostvalue_xgp = np.array([test_field.xgp for _ in range(fields_num)])
        actual_ghostvalue_xgp = test_field_container.xgp
        np.testing.assert_almost_equal(expected_ghostvalue_xgp, actual_ghostvalue_xgp, decimal=decimal,
                                       err_msg="expected xgp {} is different from actual xgp {}".format(
                                           expected_ghostvalue_xgp,
                                           actual_ghostvalue_xgp))

        # set the ghost value using a single value
        test_field.xgp = ghost_value
        test_field_container.xgp = ghost_value
        expected_ghostvalue_xgp = np.array([test_field.xgp for _ in range(fields_num)])
        actual_ghostvalue_xgp = test_field_container.xgp

        np.testing.assert_almost_equal(expected_ghostvalue_xgp, actual_ghostvalue_xgp, decimal=decimal,
                                       err_msg="expected xgp {} is different from actual xgp {}".format(
                                           expected_ghostvalue_xgp,
                                           actual_ghostvalue_xgp))

        # set the ghost value using a numpy array
        test_field.xgp = ghost_value
        test_field_container.xgp = np.array([ghost_value for _ in range(fields_num)])
        expected_ghostvalue_xgp = np.array([test_field.xgp for _ in range(fields_num)])
        actual_ghostvalue_xgp = test_field_container.xgp

        np.testing.assert_almost_equal(expected_ghostvalue_xgp, actual_ghostvalue_xgp, decimal=decimal,
                                       err_msg="expected xgp {} is different from actual xgp {}".format(
                                           expected_ghostvalue_xgp,
                                           actual_ghostvalue_xgp))

        # set the ghost value using a list of value
        test_field.xgp = ghost_value
        test_field_container.xgp = [ghost_value for _ in range(fields_num)]
        expected_ghostvalue_xgp = np.array([test_field.xgp for _ in range(fields_num)])
        actual_ghostvalue_xgp = test_field_container.xgp

        np.testing.assert_almost_equal(expected_ghostvalue_xgp, actual_ghostvalue_xgp, decimal=decimal,
                                       err_msg="expected xgp {} is different from actual xgp {}".format(
                                           expected_ghostvalue_xgp,
                                           actual_ghostvalue_xgp))
    #
    @data(1, -2.0, 3.562, 0.0, -1, -3.5, -0.112)
    def test_ygn(self, value):
        ghost_value = value
        decimal = 7
        ny, nx, ghostcells, fields_num = (3, 3, 1, 4)
        test_field = self.FieldType(shape=(ny, nx), ghostcells=ghostcells)

        test_field_container = FieldsContainer(shape=(ny, nx), FieldClass=self.FieldType, number=fields_num,
                                               ghostcells=ghostcells)

        # make sure that the initial ghost value is 0
        expected_ghostvalue_ygn = np.array([test_field.ygn for _ in range(fields_num)])
        actual_ghostvalue_ygn = test_field_container.ygn
        np.testing.assert_almost_equal(expected_ghostvalue_ygn, actual_ghostvalue_ygn, decimal=decimal,
                                       err_msg="expected ygn {} is different from actual ygn {}".format(
                                           expected_ghostvalue_ygn,
                                           actual_ghostvalue_ygn))

        # set the ghost value using a single value
        test_field.ygn = ghost_value
        test_field_container.ygn = ghost_value
        expected_ghostvalue_ygn = np.array([test_field.ygn for _ in range(fields_num)])
        actual_ghostvalue_ygn = test_field_container.ygn

        np.testing.assert_almost_equal(expected_ghostvalue_ygn, actual_ghostvalue_ygn, decimal=decimal,
                                       err_msg="expected ygn {} is different from actual ygn {}".format(
                                           expected_ghostvalue_ygn,
                                           actual_ghostvalue_ygn))

        # set the ghost value using a numpy array
        test_field.ygn = ghost_value
        test_field_container.ygn = np.array([ghost_value for _ in range(fields_num)])
        expected_ghostvalue_ygn = np.array([test_field.ygn for _ in range(fields_num)])
        actual_ghostvalue_ygn = test_field_container.ygn

        np.testing.assert_almost_equal(expected_ghostvalue_ygn, actual_ghostvalue_ygn, decimal=decimal,
                                       err_msg="expected ygn {} is different from actual ygn {}".format(
                                           expected_ghostvalue_ygn,
                                           actual_ghostvalue_ygn))

        # set the ghost value using a list of value
        test_field.ygn = ghost_value
        test_field_container.ygn = [ghost_value for _ in range(fields_num)]
        expected_ghostvalue_ygn = np.array([test_field.ygn for _ in range(fields_num)])
        actual_ghostvalue_ygn = test_field_container.ygn

        np.testing.assert_almost_equal(expected_ghostvalue_ygn, actual_ghostvalue_ygn, decimal=decimal,
                                       err_msg="expected ygn {} is different from actual ygn {}".format(
                                           expected_ghostvalue_ygn,
                                           actual_ghostvalue_ygn))

    @data(1, -2.0, 3.562, 0.0, -1, -3.5, -0.112)
    def test_ygp(self, value):
        ghost_value = value
        decimal = 7
        ny, nx, ghostcells, fields_num = (3, 3, 1, 4)
        test_field = self.FieldType(shape=(ny, nx), ghostcells=ghostcells)

        test_field_container = FieldsContainer(shape=(ny, nx), FieldClass=self.FieldType, number=fields_num,
                                               ghostcells=ghostcells)

        # make sure that the initial ghost value is 0
        expected_ghostvalue_ygp = np.array([test_field.ygp for _ in range(fields_num)])
        actual_ghostvalue_ygp = test_field_container.ygp
        np.testing.assert_almost_equal(expected_ghostvalue_ygp, actual_ghostvalue_ygp, decimal=decimal,
                                       err_msg="expected ygp {} is different from actual ygp {}".format(
                                           expected_ghostvalue_ygp,
                                           actual_ghostvalue_ygp))

        # set the ghost value using a single value
        test_field.ygp = ghost_value
        test_field_container.ygp = ghost_value
        expected_ghostvalue_ygp = np.array([test_field.ygp for _ in range(fields_num)])
        actual_ghostvalue_ygp = test_field_container.ygp

        np.testing.assert_almost_equal(expected_ghostvalue_ygp, actual_ghostvalue_ygp, decimal=decimal,
                                       err_msg="expected ygp {} is different from actual ygp {}".format(
                                           expected_ghostvalue_ygp,
                                           actual_ghostvalue_ygp))

        # set the ghost value using a numpy array
        test_field.ygp = ghost_value
        test_field_container.ygp = np.array([ghost_value for _ in range(fields_num)])
        expected_ghostvalue_ygp = np.array([test_field.ygp for _ in range(fields_num)])
        actual_ghostvalue_ygp = test_field_container.ygp

        np.testing.assert_almost_equal(expected_ghostvalue_ygp, actual_ghostvalue_ygp, decimal=decimal,
                                       err_msg="expected ygp {} is different from actual ygp {}".format(
                                           expected_ghostvalue_ygp,
                                           actual_ghostvalue_ygp))

        # set the ghost value using a list of value
        test_field.ygp = ghost_value
        test_field_container.ygp = [ghost_value for _ in range(fields_num)]
        expected_ghostvalue_ygp = np.array([test_field.ygp for _ in range(fields_num)])
        actual_ghostvalue_ygp = test_field_container.ygp

        np.testing.assert_almost_equal(expected_ghostvalue_ygp, actual_ghostvalue_ygp, decimal=decimal,
                                       err_msg="expected ygp {} is different from actual ygp {}".format(
                                           expected_ghostvalue_ygp,
                                           actual_ghostvalue_ygp))

    @data(1, -2.0, 3.562, 0.0, -1, -3.5, -0.112)
    def test_Ixgn(self, value):
        # note: Ixgn includes the values of the ghostcells.
        internal_field_value = value
        decimal = 7
        ny, nx, ghostcells, fields_num = (3, 3, 1, 4)
        test_field = self.FieldType(shape=(ny, nx), ghostcells=ghostcells)

        test_field_container = FieldsContainer(shape=(ny, nx), FieldClass=self.FieldType, number=fields_num,
                                               ghostcells=ghostcells)


        # check that the interior value is zero
        # set the value of the interior
        test_field.interior = internal_field_value
        test_field_container.interior = internal_field_value

        expected_Ixgn = np.array([test_field.Ixgn for _ in range(fields_num)])
        actual_Ixgn = test_field_container.Ixgn
        np.testing.assert_almost_equal(expected_Ixgn, actual_Ixgn, decimal=decimal,
                                       err_msg="expected Ixgn {} is different from actual Ixgn {}".format(
                                           expected_Ixgn,
                                           actual_Ixgn))


    @data(1, -2.0, 3.562, 0.0, -1, -3.5, -0.112)
    def test_Ixgp(self, value):
        # note: Ixgp includes the values of the ghostcells.
        internal_field_value = value
        decimal = 7
        ny, nx, ghostcells, fields_num = (3, 3, 1, 4)
        test_field = self.FieldType(shape=(ny, nx), ghostcells=ghostcells)

        test_field_container = FieldsContainer(shape=(ny, nx), FieldClass=self.FieldType, number=fields_num,
                                               ghostcells=ghostcells)

        # check that the interior value is zero
        # set the value of the interior
        test_field.interior = internal_field_value
        test_field_container.interior = internal_field_value

        expected_Ixgp = np.array([test_field.Ixgp for _ in range(fields_num)])
        actual_Ixgp = test_field_container.Ixgp
        np.testing.assert_almost_equal(expected_Ixgp, actual_Ixgp, decimal=decimal,
                                       err_msg="expected Ixgp {} is different from actual Ixgp {}".format(
                                           expected_Ixgp,
                                           actual_Ixgp))

    @data(1, -2.0, 3.562, 0.0, -1, -3.5, -0.112)
    def test_Iygn(self, value):
        # note: Iygn includes the values of the ghostcells.
        internal_field_value = value
        decimal = 7
        ny, nx, ghostcells, fields_num = (3, 3, 1, 4)
        test_field = self.FieldType(shape=(ny, nx), ghostcells=ghostcells)

        test_field_container = FieldsContainer(shape=(ny, nx), FieldClass=self.FieldType, number=fields_num,
                                               ghostcells=ghostcells)

        # check that the interior value is zero
        # set the value of the interior
        test_field.interior = internal_field_value
        test_field_container.interior = internal_field_value

        expected_Iygn = np.array([test_field.Iygn for _ in range(fields_num)])
        actual_Iygn = test_field_container.Iygn
        np.testing.assert_almost_equal(expected_Iygn, actual_Iygn, decimal=decimal,
                                       err_msg="expected Iygn {} is different from actual Iygn {}".format(
                                           expected_Iygn,
                                           actual_Iygn))

    @data(1, -2.0, 3.562, 0.0, -1, -3.5, -0.112)
    def test_Iygp(self, value):
        # note: Iygp includes the values of the ghostcells.
        internal_field_value = value
        decimal = 7
        ny, nx, ghostcells, fields_num = (3, 3, 1, 4)
        test_field = self.FieldType(shape=(ny, nx), ghostcells=ghostcells)

        test_field_container = FieldsContainer(shape=(ny, nx), FieldClass=self.FieldType, number=fields_num,
                                               ghostcells=ghostcells)

        # check that the interior value is zero
        # set the value of the interior
        test_field.interior = internal_field_value
        test_field_container.interior = internal_field_value

        expected_Iygp = np.array([test_field.Iygp for _ in range(fields_num)])
        actual_Iygp = test_field_container.Iygp
        np.testing.assert_almost_equal(expected_Iygp, actual_Iygp, decimal=decimal,
                                       err_msg="expected Iygp {} is different from actual Iygp {}".format(
                                           expected_Iygp,
                                           actual_Iygp))

    @data(1, -2.0, 3.562, 0.0, -1, -3.5, -0.112)
    def test__getitem__(self, value):
        internal_field_value = value
        decimal = 7
        ny, nx, ghostcells, fields_num = (3, 3, 1, 4)
        test_field = self.FieldType(shape=(ny, nx), ghostcells=ghostcells)

        test_field_container = FieldsContainer(shape=(ny, nx), FieldClass=self.FieldType, number=fields_num,
                                               ghostcells=ghostcells)
        # check that all the fields are zero fields

        for num, field in enumerate(test_field_container.fields):
            expected_value = test_field.array
            actual_value = field.array
            np.testing.assert_almost_equal(actual_value, expected_value, decimal=decimal,
                                   err_msg="the value {} field {} is supposed to be {}".format(num, actual_value,expected_value))

        # set the interior to a value
        test_field.interior = internal_field_value
        test_field_container.interior = internal_field_value
        for num, field in enumerate(test_field_container.fields):
            expected_value = test_field.array
            actual_value = field.array
            np.testing.assert_almost_equal(actual_value, expected_value, decimal=decimal,
                                           err_msg="the value {} field {} is supposed to be {}".format(num,
                                                                                                       actual_value,
                                                                                                       expected_value))

    @data(1, -2.0, 3.562, 0.0, -1, -3.5, -0.112)
    def test__setitem__(self, value):
        internal_field_value = value
        decimal = 7
        ny, nx, ghostcells, fields_num = (3, 3, 1, 4)
        test_fields = [ self.FieldType(shape=(ny, nx), ghostcells=ghostcells) for _ in range(fields_num)]
        for field in test_fields:
            # initialize the field
            field <<= internal_field_value

        test_field_container = FieldsContainer(shape=(ny, nx), FieldClass=self.FieldType, number=fields_num,
                                               ghostcells=ghostcells)

        for num in range(fields_num):
            test_field_container[num]<<=test_fields[num]


        for num in range(fields_num):

            expected_value = test_fields[num]
            actual_value = test_field_container[num]

            self.assertTrue(np.all(actual_value== expected_value),
                                   msg="the value {} field {} is supposed to be {}".format(num,actual_value,expected_value))

    @data((1, 3, 1,2), (3, 1, 1,3), (3, 3, 1,4), (1, 3, 2,5), (3, 1, 2, 2), (3, 3, 2,2))
    def test__array__(self, value):
        # returns the entire array with ghostcells padding in both directions
        ny, nx, ghostcells, fields_num = value
        test_field_container = FieldsContainer(shape=(ny, nx), FieldClass=self.FieldType, number=fields_num,
                                               ghostcells=ghostcells)

        expected_array = np.array([self.FieldType(shape=(ny, nx), ghostcells=ghostcells) for _ in range(fields_num)],dtype=object)
        actual_array = test_field_container.__array__()

        for expected, actual in zip(expected_array,actual_array):
            self.assertTrue(np.all(expected== actual),
                                          "the expected array {} is different from the actual array {}".format(
                                              expected, actual))

    def test_is_similar_to(self):
        test_containerfields1 = [FieldsContainer(shape=(3, 3),FieldClass=self.FieldType,number=3,ghostcells=1),
                        FieldsContainer(shape=(1, 3),FieldClass=self.FieldType,number=2,ghostcells=1),
                        FieldsContainer(shape=(3, 1),FieldClass=self.FieldType,number=4,ghostcells=1),
                        FieldsContainer(shape=(2, 4),FieldClass=self.FieldType,number=2,ghostcells=1)]
        other_similar_to_containerfields1 = [FieldsContainer(shape=(3, 3),FieldClass=self.FieldType,number=3,ghostcells=1),
                        FieldsContainer(shape=(1, 3),FieldClass=self.FieldType,number=2,ghostcells=1),
                        FieldsContainer(shape=(3, 1),FieldClass=self.FieldType,number=4,ghostcells=1),
                        FieldsContainer(shape=(2, 4),FieldClass=self.FieldType,number=2,ghostcells=1)]

        test_containerfields2 = [FieldsContainer(shape=(4, 4),FieldClass=self.FieldType,number=3,ghostcells=1),
                        FieldsContainer(shape=(6, 2),FieldClass=self.FieldType,number=6,ghostcells=1),
                        FieldsContainer(shape=(1, 5),FieldClass=self.FieldType,number=3,ghostcells=1),
                        FieldsContainer(shape=(3, 4),FieldClass=self.FieldType,number=5,ghostcells=1)]
        other_similar_to_containerfields2 = [FieldsContainer(shape=(4, 4),FieldClass=self.FieldType,number=3,ghostcells=1),
                        FieldsContainer(shape=(6, 2),FieldClass=self.FieldType,number=6,ghostcells=1),
                        FieldsContainer(shape=(1, 5),FieldClass=self.FieldType,number=3,ghostcells=1),
                        FieldsContainer(shape=(3, 4),FieldClass=self.FieldType,number=5,ghostcells=1)]

        for container_1, container_2, other_1, other_2 in zip(test_containerfields1, test_containerfields2, other_similar_to_containerfields1,
                                                      other_similar_to_containerfields2):
            self.assertTrue(container_1.is_similar_to(other_1),
                            "field {} is supposed to be similar in dimensions and type to {}".format(container_1, other_1))
            self.assertFalse(container_1.is_similar_to(other_2),
                             "field {} is not supposed to be similar in dimensions and type to {}".format(container_1,
                                                                                                          other_2))

            self.assertFalse(container_2.is_similar_to(other_1),
                             "field {} is not supposed to be similar in dimensions and type to {}".format(container_2,
                                                                                                          other_1))
            self.assertTrue(container_2.is_similar_to(other_2),
                            "field {} is supposed to be similar in dimensions and type to {}".format(container_2, other_2))

        test_fieldcontainer = FieldsContainer(shape=(3, 3),FieldClass=self.FieldType,number=3,ghostcells=1)
        other_similar_containerfields_types = [FieldsContainer(shape=(3, 3),FieldClass=fieldtype,number=3,ghostcells=1) for fieldtype in self.SimilarFieldsTypes]
        other_non_similar_containerfields_types = [FieldsContainer(shape=(3, 3),FieldClass=fieldtype,number=3,ghostcells=1) for fieldtype in self.NonSimilarFieldsTypes]

        for similar_fieldcontainer in other_similar_containerfields_types:
            self.assertTrue(test_fieldcontainer.is_similar_to(similar_fieldcontainer),
                            "{} is not similar to {}".format(test_fieldcontainer, similar_fieldcontainer))

        for non_similar_fieldcontainer in other_non_similar_containerfields_types:
            self.assertFalse(test_fieldcontainer.is_similar_to(non_similar_fieldcontainer),
                             "{} is similar to {}".format(test_fieldcontainer, non_similar_fieldcontainer))

    def test__ilshift__(self):
        ny, nx, ghostcells, fields_num = (5, 5, 1,3)
        # allows a list of ndarrays, numpy array of arrays, number, list of number or numpy array of numbers and fieldcontainer that is similar
        test_cases_that_works = [np.array([np.zeros(shape=(5, 5)),np.zeros(shape=(5, 5)),np.zeros(shape=(5, 5))]),
                                 [np.zeros(shape=(5, 5)),np.zeros(shape=(5, 5)),np.zeros(shape=(5, 5))],
                                 [1,1,1],
                                 [self.FieldType(shape=(5,5)),self.FieldType(shape=(5,5)),self.FieldType(shape=(5,5))],
                                 1,
                                 FieldsContainer(shape=(5,5),FieldClass=self.FieldType,number=fields_num,ghostcells=1)]

        test_cases_that_doesnt_works = [np.array([np.zeros(shape=(1, 5)), np.zeros(shape=(1, 5)), np.zeros(shape=(1, 5))]),
                                         [np.zeros(shape=(4, 4)), np.zeros(shape=(4, 4)), np.zeros(shape=(4, 4))],
                                         [1, 1],
                                         [self.FieldType(shape=(2, 5)), self.FieldType(shape=(3, 5)),
                                          self.FieldType(shape=(4, 5))],
                                         FieldsContainer(shape=(5, 2), FieldClass=self.FieldType, number=fields_num,
                                                         ghostcells=1),
                                        FieldsContainer(shape=(5, 5), FieldClass=self.NonSimilarFieldsTypes[0], number=fields_num,
                                                        ghostcells=1)]


        test_field_container = FieldsContainer(shape=(ny,nx),FieldClass=self.FieldType,number=fields_num,ghostcells=1)

        for case in test_cases_that_works:
            try:
                test_field_container <<= case
            except Exception as e:
                print(e)
                self.fail("operation between {} and {} is supposed to be allowed".format(test_field_container, case))

        for case in test_cases_that_doesnt_works:
            with self.assertRaises(Exception) as exception:
                test_field_container <<= case
            self.assertTrue(type(exception.exception) in [ValueError,NotImplementedError],
                            "operation between {} and {} is not supposed to be allowed".format(test_field_container, case))

        # test the assignment of point reshape values
        ny, nx, ghostcells, fields_num = (3, 3, 1, 2)
        expected_field_containter = FieldsContainer(shape=(ny, nx), FieldClass=self.FieldType, number=fields_num,
                                                ghostcells=ghostcells)
        actual_field_containter = FieldsContainer(shape=(ny, nx), FieldClass=self.FieldType, number=fields_num,
                                                    ghostcells=ghostcells)
        expected_field_containter <<= np.array([np.arange(1,10,1).reshape(ny,nx),np.arange(1,10,1).reshape(ny,nx)])
        point_reshaped = np.array([[1., 1.], [2., 2.], [3., 3.], [4., 4.], [5., 5.], [6., 6.], [7., 7.], [8., 8.], [9., 9.]])
        actual_field_containter <<= point_reshaped

        self.assertTrue(np.all(expected_field_containter == actual_field_containter),
                        "the expected field {} and the actual field constructed from pointreshape view"
                        " {} are supposed to be equal point wise".format(expected_field_containter,
                                                                         actual_field_containter))

    def test__point_reshape(self):
        ny,nx,ghostcells,fields_num = (3,3,1,2)
        test_field_containter = FieldsContainer(shape=(ny,nx),FieldClass=self.FieldType,number=fields_num,ghostcells=ghostcells)
        test_field_containter<<=[1,2]
        expected_point_reshaped = np.array([[1., 2.],[1., 2.],[1., 2.],[1., 2.],[1., 2.],[1., 2.],[1., 2.],[1., 2.],[1., 2.]])
        actual_point_reshaped = test_field_containter.point_reshape()
        self.assertTrue(np.all(expected_point_reshaped==actual_point_reshaped),
                        "the expected point reshape {} and the actual point reshape"
                        " {} are supposed to be equal point wise".format(expected_point_reshaped,actual_point_reshaped))

    @data((2, 3, 1, 2),(3, 2, 1, 2),(1, 3, 1, 2),(3, 1, 1, 2),(5, 5, 1, 2))
    def test__flatten(self,value):
        ny, nx, ghostcells, fields_num = value
        test_field_containter = FieldsContainer(shape=(ny, nx), FieldClass=self.FieldType, number=fields_num,
                                                ghostcells=ghostcells)
        test_field_containter <<= np.array([np.arange(1, nx*ny+1, 1).reshape(ny, nx), np.arange(1, nx*ny+1, 1).reshape(ny, nx)])
        expected_flatten = np.array(list(range(1, nx*ny+1, 1))+list(range(1, nx*ny+1, 1)))
        actual_flattened = test_field_containter.flatten()
        self.assertTrue(np.all(expected_flatten == actual_flattened),
                        "the expected flattened fieldcontainer {} and the actual flattened fieldcontainer"
                        " {} are supposed to be equal point wise".format(expected_flatten,
                                                                         actual_flattened))
    def test__eq__(self):
        test_fieldcontainer = FieldsContainer(shape=(4,4),FieldClass=self.FieldType, number=4, ghostcells=1)
        similar_fields = [FieldsContainer(shape=test_fieldcontainer.shape,FieldClass=field,number=test_fieldcontainer.size,ghostcells=test_fieldcontainer.ghostcells) for field in self.SimilarFieldsTypes]
        non_similar_fields = [FieldsContainer(shape=test_fieldcontainer.shape,FieldClass=field,number=test_fieldcontainer.size,ghostcells=test_fieldcontainer.ghostcells) for field in self.NonSimilarFieldsTypes]

        for field in similar_fields:
            self.assertTrue(np.all(test_fieldcontainer == field),
                            "the interior of {} and {} should be equal element wise".format(test_fieldcontainer, field))

        for field in non_similar_fields:
            with self.assertRaises(Exception) as exception:
                comarison = test_fieldcontainer == field
            self.assertTrue(type(exception.exception) in [ValueError],
                            "the interior of {} and {} should not be comparable".format(test_fieldcontainer, field))

    def test__copy__(self):
        test_fieldcontainer = FieldsContainer(shape=(4,4),FieldClass=self.FieldType, number=4, ghostcells=1)
        test_fieldcontainer <<= 1.0
        copied = test_fieldcontainer.__copy__()
        self.assertTrue(np.all(test_fieldcontainer == copied), "{} and {} should be equal".format(test_fieldcontainer, copied))
        copied_with_explicit_func = test_fieldcontainer.copy()
        self.assertTrue(np.all(test_fieldcontainer == copied_with_explicit_func),
                        "{} and {} should be equal".format(test_fieldcontainer, copied_with_explicit_func))

    def test__array_ufunc__(self):
        ny, nx, ghostcells, fields_num = (4, 2, 1,3)
        allowed_data = [1, 2.5, -5.0, [np.ones(shape=(ny, nx)),np.ones(shape=(ny, nx)),np.ones(shape=(ny, nx))],
                        [3*np.ones(shape=(ny, nx)),3*np.ones(shape=(ny, nx)),3*np.ones(shape=(ny, nx))]]
        allowed_fieldcontainers = [FieldsContainer(shape=(ny, nx), FieldClass=self.FieldType, number=fields_num, ghostcells=ghostcells) for _ in range(len(allowed_data))]

        for fieldcontainer, data in zip(allowed_fieldcontainers, allowed_data):
            fieldcontainer <<= data

        def recursive_numpy_operation(field, list, operation):
            if not list:
                return field
            else:
                return operation(field, recursive_numpy_operation(list[0], list[1:], operation))

        # check the results computed using numpy and data arrays
        test_fieldcontainer = FieldsContainer(shape=(ny,nx),FieldClass=self.FieldType,number=fields_num, ghostcells=ghostcells)
        test_fieldcontainer <<= recursive_numpy_operation(allowed_data[0], allowed_data[1:], np.add)
        np.testing.assert_almost_equal(test_fieldcontainer.interior, 2.5 * np.ones_like(test_fieldcontainer.interior), decimal=7,
                                       err_msg="failed ufunc addition")

        test_fieldcontainer <<= recursive_numpy_operation(allowed_data[0], allowed_data[1:], np.subtract)
        np.testing.assert_almost_equal(test_fieldcontainer.interior, -4.5 * np.ones_like(test_fieldcontainer.interior), decimal=7,
                                       err_msg="failed ufunc subtract")

        # check the results computed using compatible Fields of data
        test_fieldcontainer = FieldsContainer(shape=(ny,nx),FieldClass=self.FieldType,number=fields_num, ghostcells=ghostcells)
        test_fieldcontainer <<= recursive_numpy_operation(allowed_fieldcontainers[0], allowed_fieldcontainers[1:], np.add)
        np.testing.assert_almost_equal(test_fieldcontainer.interior, 2.5 * np.ones_like(test_fieldcontainer.interior), decimal=7,
                                       err_msg="failed ufunc addition on Fields")

        test_fieldcontainer <<= recursive_numpy_operation(allowed_fieldcontainers[0], allowed_fieldcontainers[1:], np.subtract)
        np.testing.assert_almost_equal(test_fieldcontainer.interior, -4.5 * np.ones_like(test_fieldcontainer.interior), decimal=7,
                                       err_msg="failed ufunc subtract on Fields")


class TestXVolField(TestSVolContainer):
    FieldType = XVolField
    NonSimilarFieldsTypes = [SVolField,YVolField,YFSVolField,XFXVolField,YFXVolField,XFYVolField,YFYVolField]
    SimilarFieldsTypes = [XFSVolField,XVolField]


class TestYVolField(TestSVolContainer):
    FieldType = YVolField
    NonSimilarFieldsTypes = [SVolField,XVolField,XFSVolField,XFXVolField,YFXVolField,XFYVolField,YFYVolField]
    SimilarFieldsTypes = [YVolField, YFSVolField]

class TestXFSVolField(TestSVolContainer):
    FieldType = XFSVolField
    NonSimilarFieldsTypes = [SVolField,YVolField,YFSVolField,XFXVolField,YFXVolField,XFYVolField,YFYVolField]
    SimilarFieldsTypes = [XFSVolField, XVolField]

class TestYFSVolField(TestSVolContainer):
    FieldType = YFSVolField
    NonSimilarFieldsTypes = [SVolField,XVolField,XFSVolField,XFXVolField,YFXVolField,XFYVolField,YFYVolField]
    SimilarFieldsTypes = [YFSVolField, YVolField]

class TestXFXVolField(TestSVolContainer):
    FieldType = XFXVolField
    NonSimilarFieldsTypes = [XVolField,YVolField,XFSVolField,YFSVolField,YFXVolField,XFYVolField]
    SimilarFieldsTypes = [XFXVolField, YFYVolField, SVolField]

class TestYFXVolField(TestSVolContainer):
    FieldType = YFXVolField
    NonSimilarFieldsTypes = [SVolField,XVolField,YVolField,XFSVolField,YFSVolField,XFXVolField,YFYVolField]
    SimilarFieldsTypes = [YFXVolField, XFYVolField]

class TestXFYVolField(TestSVolContainer):
    FieldType = XFYVolField
    NonSimilarFieldsTypes = [SVolField,XVolField,YVolField,XFSVolField,YFSVolField,XFXVolField,YFYVolField]
    SimilarFieldsTypes = [XFYVolField,YFXVolField]

class TestYFYVolField(TestSVolContainer):
    FieldType = YFYVolField
    NonSimilarFieldsTypes = [XVolField,YVolField,XFSVolField,YFSVolField,XFXVolField,YFXVolField,XFYVolField]
    SimilarFieldsTypes = [YFYVolField, SVolField]

if __name__ == '__main__':
    unittest.main()