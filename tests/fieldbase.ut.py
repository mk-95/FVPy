import unittest
from ddt import ddt, data
from fieldbase import FieldBase
from fields import SVolField,XVolField,YVolField,XFSVolField,YFSVolField,XFXVolField,YFXVolField,XFYVolField,YFYVolField
import numpy as np
import random

# we will conduct the testing of FieldBase with the
# self.FieldType type

@ddt
class TestSVolField(unittest.TestCase):
    FieldType = SVolField
    NonSimilarFieldsTypes = [XVolField,YVolField,XFSVolField,YFSVolField,YFXVolField,XFYVolField]
    SimilarFieldsTypes = [XFXVolField,YFYVolField,SVolField]

    @data((-1,3,1),(3,-1,1),(3,3,-1),(1,3.5,2),(3.5,1,2),(3,3,2.5),(3,3,(2,3)))
    def test_init_exceptions(self,value):
        ny, nx, ghostcell = value
        with self.assertRaises(Exception) as exception:
            test_field = self.FieldType(shape=(ny, nx), ghostcells=ghostcell)

        self.assertTrue(type(exception.exception) in [ValueError],"{} was not thrown with the following input ({})".format(ValueError,value))

    @data((1,3,1),(3,1,1),(3,3,1),(1,3,2),(3,1,2),(3,3,2))
    def test_init(self,value):
        ny,nx,ghostcell = value
        test_field = self.FieldType(shape=(ny,nx),ghostcells=ghostcell)

        # check the shape of the field
        expected_padded_shape = (ny+ 2*ghostcell,nx + 2*ghostcell)
        actual_padded_shape = test_field.array.shape

        self.assertEqual(expected_padded_shape,actual_padded_shape,
                         "Expected padded shape {} is different from the actual padded shape {}.".format(expected_padded_shape,actual_padded_shape))

        expected_unpadded_shape = (ny, nx)
        actual_unpadded_shape = test_field.array[ghostcell:-ghostcell, ghostcell:-ghostcell].shape

        self.assertEqual(expected_unpadded_shape, actual_unpadded_shape,
                         "Expected unpadded shape {} is different from the actual unpadded shape {}.".format(
                             expected_unpadded_shape, actual_unpadded_shape))

        expected_initially_filled_value = np.zeros_like(test_field.array)
        actual_initially_filled_value = test_field.array

        np.testing.assert_array_equal(expected_initially_filled_value,actual_initially_filled_value,
                                      "Expected initially filled value 0.0 is different from the actual initially filled value {}.".format(
                             actual_initially_filled_value))

    @data((1, 3, 1), (3, 1, 1), (3, 3, 1), (1, 3, 2), (3, 1, 2), (3, 3, 2))
    def test_shape(self,value):
        ny,nx,ghostcell = value
        test_field = self.FieldType(shape=(ny,nx),ghostcells=ghostcell)
        expected_shape_value = (ny,nx)
        actual_shape_value = test_field.shape

        self.assertEqual(expected_shape_value,actual_shape_value,
                         "Expected shape {} is different from the actual shape {}".format(expected_shape_value,actual_shape_value))

    @data((1, 3, 1), (3, 1, 1), (3, 3, 1), (1, 3, 2), (3, 1, 2), (3, 3, 2))
    def test_ghostcells(self,value):
        ny, nx, ghostcell = value
        test_field = self.FieldType(shape=(ny, nx), ghostcells=ghostcell)
        expected_ghostcells_value = ghostcell
        actual_ghostcells_value = test_field.ghostcells

        self.assertEqual(expected_ghostcells_value, actual_ghostcells_value,
                         "Expected number of ghostcells {} is different from the actual number of ghostcells {}".format(expected_ghostcells_value,
                                                                                          actual_ghostcells_value))

    @data((1, 3, 1), (3, 1, 1), (3, 3, 1), (1, 3, 2), (3, 1, 2), (3, 3, 2))
    def test_array(self,value):
        # returns the entire array with ghostcells padding in both directions
        ny,nx,ghostcells = value
        test_field = self.FieldType(shape=(ny,nx),ghostcells=ghostcells)

        expected_array = np.zeros(shape=(ny+2*ghostcells,nx+2*ghostcells))
        actual_array = test_field.array

        np.testing.assert_array_equal(expected_array,actual_array,"the expected array {} is different from the actual array {}".format(expected_array,actual_array))

    @data(1.12560,2.01452,3,0.0,-1,-3.5,-0.112)
    def test_interior(self,value):
        ny,nx,ghostcells = (3,3,1)

        decimal = 7 # compare to how many decimal value

        test_field = self.FieldType(shape=(ny,nx),ghostcells=ghostcells)

        interior_field_shape = test_field.shape
        # testing the getter on the initial field value
        expected_initial_interior_array = np.zeros(interior_field_shape)
        actual_initial_interior_array = test_field.interior
        np.testing.assert_almost_equal(expected_initial_interior_array,actual_initial_interior_array,decimal=decimal, err_msg=
                                      "the expected interior initial array {} is "
                                      "different from the actual initial array {}"
                                      .format(expected_initial_interior_array,actual_initial_interior_array))

        # testing the setter on different value
        expected_interior_array = value * np.ones(interior_field_shape)
        # set the interior value of the inner array
        test_field.interior = value
        actual_interiro_array = test_field.interior
        np.testing.assert_almost_equal(expected_interior_array, actual_interiro_array,decimal=decimal, err_msg=
                                      "the expected interior initial array {} is "
                                      "different from the actual initial array {}"
                                      .format(expected_interior_array, actual_interiro_array))
        # make sure that the ghost cells still have the value of zero
        expected_xgp = np.zeros(shape=(ny+2*ghostcells,ghostcells))
        expected_xgn = np.zeros(shape=(ny+2*ghostcells,ghostcells))
        expected_ygn = np.zeros(shape=(ghostcells,nx+2*ghostcells))
        expected_ygp = np.zeros(shape=(ghostcells,nx+2*ghostcells))
        actual_xgp = test_field.xgp
        actual_xgn = test_field.xgn
        actual_ygn = test_field.ygn
        actual_ygp = test_field.ygp
        np.testing.assert_almost_equal(expected_xgn,actual_xgn,decimal=decimal, err_msg="expected xgn {} is different from actual xgn {}".format(expected_xgn,actual_xgn))
        np.testing.assert_almost_equal(expected_xgp,actual_xgp,decimal=decimal, err_msg="expected xgp {} is different from actual xgp {}".format(expected_xgp,actual_xgp))
        np.testing.assert_almost_equal(expected_ygn,actual_ygn,decimal=decimal, err_msg="expected ygn {} is different from actual ygn {}".format(expected_ygn,actual_ygn))
        np.testing.assert_almost_equal(expected_ygp,actual_ygp,decimal=decimal, err_msg="expected ygp {} is different from actual ygp {}".format(expected_ygp,actual_ygp))

    @data(1, -2.0, 3.562, 0.0, -1, -3.5, -0.112)
    def test_xn(self,value):
        interior_value = value
        decimal = 7
        ny, nx, ghostcells = (3, 3, 1)
        test_field = self.FieldType(shape=(ny,nx),ghostcells=ghostcells)
        test_field.interior = interior_value

        expected_xn = np.zeros(test_field.shape)
        expected_xn[:,1:] = test_field.interior[:,1:]
        actual_xn = test_field.xn
        np.testing.assert_almost_equal(expected_xn,actual_xn,decimal=decimal,err_msg="expected xn {} is different from actual xn {}".format(expected_xn,actual_xn))

    @data(1, -2.0, 3.562, 0.0, -1, -3.5, -0.112)
    def test_xp(self,value):
        interior_value = value
        decimal = 7
        ny, nx, ghostcells = (3, 3, 1)
        test_field = self.FieldType(shape=(ny, nx), ghostcells=ghostcells)
        test_field.interior = interior_value

        expected_xp = np.zeros(test_field.shape)
        expected_xp[:, :-1] = test_field.interior[:, :-1]
        actual_xp = test_field.xp
        np.testing.assert_almost_equal(expected_xp, actual_xp, decimal=decimal,
                                       err_msg="expected xp {} is different from actual xp {}".format(expected_xp,
                                                                                                      actual_xp))

    @data(1, -2.0, 3.562, 0.0, -1, -3.5, -0.112)
    def test_yn(self,value):
        interior_value = value
        decimal = 7
        ny, nx, ghostcells = (3, 3, 1)
        test_field = self.FieldType(shape=(ny, nx), ghostcells=ghostcells)
        test_field.interior = interior_value

        expected_yn = np.zeros(test_field.shape)
        expected_yn[1:, :] = test_field.interior[1:, :]
        actual_yn = test_field.yn
        np.testing.assert_almost_equal(expected_yn, actual_yn, decimal=decimal,
                                       err_msg="expected yn {} is different from actual yn {}".format(expected_yn,
                                                                                                      actual_yn))

    @data(1, -2.0, 3.562, 0.0, -1, -3.5, -0.112)
    def test_yp(self,value):
        interior_value = value
        decimal = 7
        ny, nx, ghostcells = (3, 3, 1)
        test_field = self.FieldType(shape=(ny, nx), ghostcells=ghostcells)
        test_field.interior = interior_value

        expected_yp = np.zeros(test_field.shape)
        expected_yp[:-1, :] = test_field.interior[:-1, :]
        actual_yp = test_field.yp
        np.testing.assert_almost_equal(expected_yp, actual_yp, decimal=decimal,
                                       err_msg="expected yp {} is different from actual yp {}".format(expected_yp,
                                                                                                      actual_yp))
    @data(1, -2.0, 3.562, 0.0, -1, -3.5, -0.112)
    def test_xgn(self,value):
        ghost_value = value
        decimal = 7
        ny, nx, ghostcells = (3, 3, 1)
        test_field = self.FieldType(shape=(ny, nx), ghostcells=ghostcells)

        #make sure that the initial ghost value is 0
        expected_ghostvalue_xgn = np.zeros(shape=(ny+2*ghostcells,ghostcells))
        actual_ghostvalue_xgn = test_field.xgn
        np.testing.assert_almost_equal(expected_ghostvalue_xgn, actual_ghostvalue_xgn, decimal=decimal,
                                       err_msg="expected xgn {} is different from actual xgn {}".format(expected_ghostvalue_xgn,
                                                                                                      actual_ghostvalue_xgn))

        #set the ghost value and test if the value was set properly
        test_field.xgn = ghost_value
        expected_ghostvalue_xgn = ghost_value * np.ones(shape=(ny+2*ghostcells,ghostcells))
        actual_ghostvalue_xgn = test_field.xgn

        np.testing.assert_almost_equal(expected_ghostvalue_xgn, actual_ghostvalue_xgn, decimal=decimal,
                                       err_msg="expected xgn {} is different from actual xgn {}".format(
                                           expected_ghostvalue_xgn,
                                           actual_ghostvalue_xgn))

    @data(1, -2.0, 3.562, 0.0, -1, -3.5, -0.112)
    def test_xgp(self,value):
        ghost_value = value
        decimal = 7
        ny, nx, ghostcells = (3, 3, 1)
        test_field = self.FieldType(shape=(ny, nx), ghostcells=ghostcells)

        # make sure that the initial ghost value is 0
        expected_ghostvalue_xgp = np.zeros(shape=(ny + 2 * ghostcells, ghostcells))
        actual_ghostvalue_xgp = test_field.xgp
        np.testing.assert_almost_equal(expected_ghostvalue_xgp, actual_ghostvalue_xgp, decimal=decimal,
                                       err_msg="expected xgp {} is different from actual xgp {}".format(
                                           expected_ghostvalue_xgp,
                                           actual_ghostvalue_xgp))

        # set the ghost value and test if the value was set properly
        test_field.xgp = ghost_value
        expected_ghostvalue_xgp = ghost_value * np.ones(shape=(ny + 2 * ghostcells, ghostcells))
        actual_ghostvalue_xgp = test_field.xgp

        np.testing.assert_almost_equal(expected_ghostvalue_xgp, actual_ghostvalue_xgp, decimal=decimal,
                                       err_msg="expected xgp {} is different from actual xgp {}".format(
                                           expected_ghostvalue_xgp,
                                           actual_ghostvalue_xgp))

    @data(1, -2.0, 3.562, 0.0, -1, -3.5, -0.112)
    def test_ygn(self,value):
        ghost_value = value
        decimal = 7
        ny, nx, ghostcells = (3, 3, 1)
        test_field = self.FieldType(shape=(ny, nx), ghostcells=ghostcells)

        # make sure that the initial ghost value is 0
        expected_ghostvalue_ygn = np.zeros(shape=(ghostcells,nx + 2 * ghostcells))
        actual_ghostvalue_ygn = test_field.ygn
        np.testing.assert_almost_equal(expected_ghostvalue_ygn, actual_ghostvalue_ygn, decimal=decimal,
                                       err_msg="expected ygn {} is different from actual ygn {}".format(
                                           expected_ghostvalue_ygn,
                                           actual_ghostvalue_ygn))

        # set the ghost value and test if the value was set properly
        test_field.ygn = ghost_value
        expected_ghostvalue_ygn = ghost_value * np.ones(shape=(ghostcells,nx + 2 * ghostcells))
        actual_ghostvalue_ygn = test_field.ygn

        np.testing.assert_almost_equal(expected_ghostvalue_ygn, actual_ghostvalue_ygn, decimal=decimal,
                                       err_msg="expected ygn {} is different from actual ygn {}".format(
                                           expected_ghostvalue_ygn,
                                           actual_ghostvalue_ygn))

    @data(1, -2.0, 3.562, 0.0, -1, -3.5, -0.112)
    def test_ygp(self,value):
        ghost_value = value
        decimal = 7
        ny, nx, ghostcells = (3, 3, 1)
        test_field = self.FieldType(shape=(ny, nx), ghostcells=ghostcells)

        # make sure that the initial ghost value is 0
        expected_ghostvalue_ygp = np.zeros(shape=(ghostcells, nx + 2 * ghostcells))
        actual_ghostvalue_ygp = test_field.ygp
        np.testing.assert_almost_equal(expected_ghostvalue_ygp, actual_ghostvalue_ygp, decimal=decimal,
                                       err_msg="expected ygn {} is different from actual ygn {}".format(
                                           expected_ghostvalue_ygp,
                                           actual_ghostvalue_ygp))

        # set the ghost value and test if the value was set properly
        test_field.ygp = ghost_value
        expected_ghostvalue_ygp = ghost_value * np.ones(shape=(ghostcells, nx + 2 * ghostcells))
        actual_ghostvalue_ygp = test_field.ygp

        np.testing.assert_almost_equal(expected_ghostvalue_ygp, actual_ghostvalue_ygp, decimal=decimal,
                                       err_msg="expected ygn {} is different from actual ygn {}".format(
                                           expected_ghostvalue_ygp,
                                           actual_ghostvalue_ygp))

    @data(1, -2.0, 3.562, 0.0, -1, -3.5, -0.112)
    def test_Ixgn(self,value):
        # note: Ixgn includes the values of the ghostcells.
        internal_field_value = value
        decimal = 7
        ny, nx, ghostcells = (3, 3, 1)
        test_field = self.FieldType(shape=(ny, nx), ghostcells=ghostcells)

        #check that the interior value is zero
        expected_Ixgn = np.zeros(shape=(ny + 2* ghostcells, ghostcells))
        actual_Ixgn = test_field.Ixgn
        np.testing.assert_almost_equal(expected_Ixgn, actual_Ixgn, decimal=decimal,
                                       err_msg="expected Ixgn {} is different from actual Ixgn {}".format(
                                           expected_Ixgn,
                                           actual_Ixgn))
        # set the value of the interior
        test_field.interior = internal_field_value
        expected_Ixgn = internal_field_value*np.ones(shape=(ny + 2* ghostcells, ghostcells))
        expected_Ixgn[0] = 0.0
        expected_Ixgn[-1] = 0.0
        actual_Ixgn = test_field.Ixgn
        np.testing.assert_almost_equal(expected_Ixgn, actual_Ixgn, decimal=decimal,
                                       err_msg="expected Ixgn {} is different from actual Ixgn {}".format(
                                           expected_Ixgn,
                                           actual_Ixgn))

    @data(1, -2.0, 3.562, 0.0, -1, -3.5, -0.112)
    def test_Ixgp(self,value):
        # note: Ixgp includes the values of the ghostcells.
        internal_field_value = value
        decimal = 7
        ny, nx, ghostcells = (3, 3, 1)
        test_field = self.FieldType(shape=(ny, nx), ghostcells=ghostcells)

        # check that the interior value is zero
        expected_Ixgp = np.zeros(shape=(ny + 2 * ghostcells, ghostcells))
        actual_Ixgp = test_field.Ixgp
        np.testing.assert_almost_equal(expected_Ixgp, actual_Ixgp, decimal=decimal,
                                       err_msg="expected Ixgp {} is different from actual Ixgp {}".format(
                                           expected_Ixgp,
                                           actual_Ixgp))
        # set the value of the interior
        test_field.interior = internal_field_value
        expected_Ixgp = internal_field_value * np.ones(shape=(ny + 2 * ghostcells, ghostcells))
        expected_Ixgp[0] = 0.0
        expected_Ixgp[-1] = 0.0
        actual_Ixgp = test_field.Ixgp
        np.testing.assert_almost_equal(expected_Ixgp, actual_Ixgp, decimal=decimal,
                                       err_msg="expected Ixgp {} is different from actual Ixgp {}".format(
                                           expected_Ixgp,
                                           actual_Ixgp))

    @data(1, -2.0, 3.562, 0.0, -1, -3.5, -0.112)
    def test_Iygn(self,value):
        # note: Iygn includes the values of the ghostcells.
        internal_field_value = value
        decimal = 7
        ny, nx, ghostcells = (3, 3, 1)
        test_field = self.FieldType(shape=(ny, nx), ghostcells=ghostcells)

        # check that the interior value is zero
        expected_Iygn = np.zeros(shape=(ghostcells, nx + 2 * ghostcells))
        actual_Iygn = test_field.Iygn
        np.testing.assert_almost_equal(expected_Iygn, actual_Iygn, decimal=decimal,
                                       err_msg="expected Iygn {} is different from actual Iygn {}".format(
                                           expected_Iygn,
                                           actual_Iygn))
        # set the value of the interior
        test_field.interior = internal_field_value
        expected_Iygn = internal_field_value * np.ones(shape=(ghostcells, nx + 2 * ghostcells))
        expected_Iygn[0,0] = 0.0
        expected_Iygn[0,-1] = 0.0
        actual_Iygn = test_field.Iygn
        np.testing.assert_almost_equal(expected_Iygn, actual_Iygn, decimal=decimal,
                                       err_msg="expected Iygn {} is different from actual Iygn {}".format(
                                           expected_Iygn,
                                           actual_Iygn))

    @data(1, -2.0, 3.562, 0.0, -1, -3.5, -0.112)
    def test_Iygp(self,value):
        # note: Iygn includes the values of the ghostcells.
        internal_field_value = value
        decimal = 7
        ny, nx, ghostcells = (3, 3, 1)
        test_field = self.FieldType(shape=(ny, nx), ghostcells=ghostcells)

        # check that the interior value is zero
        expected_Iygp = np.zeros(shape=(ghostcells, nx + 2 * ghostcells))
        actual_Iygp = test_field.Iygn
        np.testing.assert_almost_equal(expected_Iygp, actual_Iygp, decimal=decimal,
                                       err_msg="expected Iygp {} is different from actual Iygp {}".format(
                                           expected_Iygp,
                                           actual_Iygp))
        # set the value of the interior
        test_field.interior = internal_field_value
        expected_Iygp = internal_field_value * np.ones(shape=(ghostcells, nx + 2 * ghostcells))
        expected_Iygp[0, 0] = 0.0
        expected_Iygp[0, -1] = 0.0
        actual_Iygp = test_field.Iygp
        np.testing.assert_almost_equal(expected_Iygp, actual_Iygp, decimal=decimal,
                                       err_msg="expected Iygp {} is different from actual Iygp {}".format(
                                           expected_Iygp,
                                           actual_Iygp))

    @data(1, -2.0, 3.562, 0.0, -1, -3.5, -0.112)
    def test__getitem__(self,value):
        internal_field_value = value
        decimal = 7
        ny, nx, ghostcells = (3, 3, 1)
        test_field = self.FieldType(shape=(ny, nx), ghostcells=ghostcells)
        # check that a random value in the interior is zero
        j = np.random.randint(0,ny)
        i = np.random.randint(0,nx)
        expected_value = 0.0
        actual_value = test_field[j,i]
        self.assertAlmostEqual(actual_value,expected_value,places=decimal,msg="the value {} located at ({},{}) is supposed to be {}".format(actual_value,j,i,expected_value))
        #set the interior to a value
        test_field.interior=internal_field_value
        expected_value = internal_field_value
        actual_value = test_field[j,i]
        self.assertAlmostEqual(actual_value,expected_value,places=decimal,msg="the value {} located at ({},{}) is supposed to be {}".format(actual_value,j,i,expected_value))

    @data(1, -2.0, 3.562, 0.0, -1, -3.5, -0.112)
    def test__setitem__(self,value):
        internal_field_value = value
        decimal = 7
        ny, nx, ghostcells = (4, 4, 1)
        test_field = self.FieldType(shape=(ny, nx), ghostcells=ghostcells)
        # check that a random value in the interior is zero
        j = np.random.randint(1, ny-1)
        i = np.random.randint(1, nx-1)
        expected_value = 0.0
        actual_value = test_field[j, i]
        self.assertAlmostEqual(actual_value, expected_value, places=decimal,
                               msg="the value {} located at ({},{}) is supposed to be {}".format(actual_value, j, i,
                                                                                                 expected_value))
        # set the interior to a value
        test_field[j,i] = internal_field_value
        expected_value = internal_field_value
        expected_near_cell_value = 0.0
        actual_value = test_field[j, i]
        actual_value_xp = test_field[j, i+1]
        actual_value_xn = test_field[j, i-1]
        actual_value_yn = test_field[j-1, i]
        actual_value_yp = test_field[j+1, i]

        self.assertAlmostEqual(actual_value, expected_value, places=decimal,
                               msg="the value {} located at ({},{}) is supposed to be {}".format(actual_value, j, i,
                                                                                                 expected_value))

        self.assertAlmostEqual(actual_value_xp, expected_near_cell_value, places=decimal,
                               msg="the value {} located at ({},{}) is supposed to be {}".format(actual_value, j, i+1,
                                                                                                 expected_value))
        self.assertAlmostEqual(actual_value_xn, expected_near_cell_value, places=decimal,
                               msg="the value {} located at ({},{}) is supposed to be {}".format(actual_value, j, i-1,
                                                                                                 expected_value))

        self.assertAlmostEqual(actual_value_yn, expected_near_cell_value, places=decimal,
                               msg="the value {} located at ({},{}) is supposed to be {}".format(actual_value, j- 1, i ,
                                                                                                 expected_value))

        self.assertAlmostEqual(actual_value_yp, expected_near_cell_value, places=decimal,
                               msg="the value {} located at ({},{}) is supposed to be {}".format(actual_value, j + 1, i,
                                                                                                 expected_value))

    @data((1, 3, 1), (3, 1, 1), (3, 3, 1), (1, 3, 2), (3, 1, 2), (3, 3, 2))
    def test__array__(self,value):
        # returns the entire array with ghostcells padding in both directions
        ny, nx, ghostcells = value
        test_field = self.FieldType(shape=(ny, nx), ghostcells=ghostcells)

        expected_array = np.zeros(shape=(ny + 2 * ghostcells, nx + 2 * ghostcells))
        actual_array = test_field.__array__()

        np.testing.assert_array_equal(expected_array, actual_array,
                                      "the expected array {} is different from the actual array {}".format(
                                          expected_array, actual_array))

    def test_is_similar_to(self):
        test_fields1 = [self.FieldType(shape=(3,3)),self.FieldType(shape=(3,2)),self.FieldType(shape=(1,3)),self.FieldType(shape=(3,1)),self.FieldType(shape=(3,3),ghostcells=2)]
        other_similar_to_1_fields = [self.FieldType(shape=(3,3)),self.FieldType(shape=(3,2)),self.FieldType(shape=(1,3)),self.FieldType(shape=(3,1)),self.FieldType(shape=(3,3),ghostcells=2)]

        test_fields2 = [self.FieldType(shape=(1, 2)), self.FieldType(shape=(4, 2)), self.FieldType(shape=(5, 3)),
                        self.FieldType(shape=(4, 3)), self.FieldType(shape=(6, 1), ghostcells=2)]
        other_similar_to_2_fields = [self.FieldType(shape=(1, 2)), self.FieldType(shape=(4, 2)), self.FieldType(shape=(5, 3)),
                        self.FieldType(shape=(4, 3)), self.FieldType(shape=(6, 1), ghostcells=2)]

        for field_1, field_2, other_1,other_2 in zip(test_fields1,test_fields2,other_similar_to_1_fields,other_similar_to_2_fields):
            self.assertTrue(field_1.is_similar_to(other_1),"field {} is supposed to be similar in dimensions and type to {}".format(field_1,other_1))
            self.assertFalse(field_1.is_similar_to(other_2),"field {} is not supposed to be similar in dimensions and type to {}".format(field_1,other_2))

            self.assertFalse(field_2.is_similar_to(other_1),
                            "field {} is not supposed to be similar in dimensions and type to {}".format(field_2, other_1))
            self.assertTrue(field_2.is_similar_to(other_2),
                             "field {} is supposed to be similar in dimensions and type to {}".format(field_2, other_2))

        test_field = self.FieldType(shape=(4,4))
        other_similar_fields_types = [field(shape=test_field.shape) for field in self.SimilarFieldsTypes]
        other_non_similar_fields_types = [field(shape=test_field.shape) for field in self.NonSimilarFieldsTypes]

        for similar_field in other_similar_fields_types:
            self.assertTrue(test_field.is_similar_to(similar_field),"{} is not similar to {}".format(test_field,similar_field))

        for non_similar_field in other_non_similar_fields_types:
            self.assertFalse(test_field.is_similar_to(non_similar_field),"{} is similar to {}".format(test_field,non_similar_field))

    def test__ilshift__(self):
        ny,nx,ghostcells = (5,5,1)

        test_cases_that_works = [np.ndarray(shape=(5, 5)), 1, 2.4, self.SimilarFieldsTypes[0](shape=(ny,nx),ghostcells=ghostcells),self.SimilarFieldsTypes[-1](shape=(ny,nx),ghostcells=ghostcells) ]
        test_cases_that_doesnt_works = [np.ndarray(shape=(3, 2)), self.NonSimilarFieldsTypes[0](shape=(ny,nx),ghostcells=ghostcells),self.NonSimilarFieldsTypes[-1](shape=(ny,nx),ghostcells=ghostcells) ]

        test_field = self.FieldType(shape=(ny,nx),ghostcells=ghostcells)

        for case in test_cases_that_works:
            try:
                test_field <<= case
            except Exception as e:
                print(e)
                self.fail("operation between {} and {} is supposed to be allowed".format(test_field,case))

        for case in test_cases_that_doesnt_works:
            with self.assertRaises(Exception) as exception:
                test_field <<= case
            self.assertTrue(type(exception.exception) in [ValueError],"operation between {} and {} is not supposed to be allowed".format(test_field,case))

    def test__eq__(self):
        test_field = self.FieldType(shape=(4,4),ghostcells=1)
        similar_fields = [field(test_field.shape,test_field.ghostcells) for field in self.SimilarFieldsTypes]
        non_similar_fields = [field(test_field.shape,test_field.ghostcells) for field in self.NonSimilarFieldsTypes]

        for field in similar_fields:
            self.assertTrue(np.all(test_field == field),"the interior of {} and {} should be equal element wise".format(test_field,field))

        for field in non_similar_fields:
            with self.assertRaises(Exception) as exception:
                comarison = test_field == field
            self.assertTrue(type(exception.exception) in [ValueError],
                            "the interior of {} and {} should not be comparable".format(test_field, field))

    def test__copy__(self):
        test_field = self.FieldType(shape=(4, 4), ghostcells=1)
        test_field <<= 1.0
        copied = test_field.__copy__()
        self.assertTrue(np.all(test_field==copied),"{} and {} should be equal".format(test_field,copied))
        copied_with_explicit_func = test_field.copy()
        self.assertTrue(np.all(test_field == copied_with_explicit_func), "{} and {} should be equal".format(test_field, copied_with_explicit_func))

    def test__array_ufunc__(self):
        ny,nx,ghostcells = (4,2,1)
        allowed_data = [1,2.5,-5.0,np.ones(shape=(ny,nx)),3*np.ones(shape=(ny,nx))]
        allowed_fields = [field(shape=(ny,nx),ghostcells=ghostcells) for field in self.SimilarFieldsTypes]
        for field,data in zip(allowed_fields,allowed_data):
            field<<= data

        def recursive_numpy_operation(field,list,operation):
            if not list:
                return field
            else:
                return operation(field,recursive_numpy_operation(list[0],list[1:],operation))

        # check the results computed using numpy and data arrays
        test_field = self.FieldType(shape=(ny,nx),ghostcells=ghostcells)
        test_field <<= recursive_numpy_operation(allowed_data[0],allowed_data[1:],np.add)
        np.testing.assert_almost_equal(test_field.interior,2.5*np.ones_like(test_field.interior),decimal=7, err_msg="failed ufunc addition")

        test_field <<= recursive_numpy_operation(allowed_data[0], allowed_data[1:], np.subtract)
        np.testing.assert_almost_equal(test_field.interior, -4.5 * np.ones_like(test_field.interior), decimal=7,
                                       err_msg="failed ufunc subtract")

        # check the results computed using compatible Fields of data
        test_field = self.FieldType(shape=(ny, nx), ghostcells=ghostcells)
        test_field <<= recursive_numpy_operation(allowed_fields[0], allowed_data[1:], np.add)
        np.testing.assert_almost_equal(test_field.interior, 2.5 * np.ones_like(test_field.interior), decimal=7,
                                       err_msg="failed ufunc addition on Fields")

        test_field <<= recursive_numpy_operation(allowed_fields[0], allowed_data[1:], np.subtract)
        np.testing.assert_almost_equal(test_field.interior, -4.5 * np.ones_like(test_field.interior), decimal=7,
                                       err_msg="failed ufunc subtract on Fields")
        # self.fail("{} not implemented yet!".format(self.test__array_ufunc__.__name__))



class TestXVolField(TestSVolField):
    FieldType = XVolField
    NonSimilarFieldsTypes = [SVolField,YVolField,YFSVolField,XFXVolField,YFXVolField,XFYVolField,YFYVolField]
    SimilarFieldsTypes = [XFSVolField,XVolField]


class TestYVolField(TestSVolField):
    FieldType = YVolField
    NonSimilarFieldsTypes = [SVolField,XVolField,XFSVolField,XFXVolField,YFXVolField,XFYVolField,YFYVolField]
    SimilarFieldsTypes = [YVolField, YFSVolField]

class TestXFSVolField(TestSVolField):
    FieldType = XFSVolField
    NonSimilarFieldsTypes = [SVolField,YVolField,YFSVolField,XFXVolField,YFXVolField,XFYVolField,YFYVolField]
    SimilarFieldsTypes = [XFSVolField, XVolField]

class TestYFSVolField(TestSVolField):
    FieldType = YFSVolField
    NonSimilarFieldsTypes = [SVolField,XVolField,XFSVolField,XFXVolField,YFXVolField,XFYVolField,YFYVolField]
    SimilarFieldsTypes = [YFSVolField, YVolField]

class TestXFXVolField(TestSVolField):
    FieldType = XFXVolField
    NonSimilarFieldsTypes = [XVolField,YVolField,XFSVolField,YFSVolField,YFXVolField,XFYVolField]
    SimilarFieldsTypes = [XFXVolField, YFYVolField, SVolField]

class TestYFXVolField(TestSVolField):
    FieldType = YFXVolField
    NonSimilarFieldsTypes = [SVolField,XVolField,YVolField,XFSVolField,YFSVolField,XFXVolField,YFYVolField]
    SimilarFieldsTypes = [YFXVolField, XFYVolField]

class TestXFYVolField(TestSVolField):
    FieldType = XFYVolField
    NonSimilarFieldsTypes = [SVolField,XVolField,YVolField,XFSVolField,YFSVolField,XFXVolField,YFYVolField]
    SimilarFieldsTypes = [XFYVolField,YFXVolField]

class TestYFYVolField(TestSVolField):
    FieldType = YFYVolField
    NonSimilarFieldsTypes = [XVolField,YVolField,XFSVolField,YFSVolField,XFXVolField,YFXVolField,XFYVolField]
    SimilarFieldsTypes = [YFYVolField, SVolField]

if __name__ == '__main__':
    unittest.main()