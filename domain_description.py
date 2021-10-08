import numpy as np

class DomainDescription:
    def __init__(self, N, L):
        '''
        :param N: list of int [ny, nx]
        :param L: list of float [ly, lx]
        '''
        ## define some boiler plate
        self.__dim_number = len(N)
        self.__grid_points = N[::-1]
        self.__domain_size = L[::-1]
        self.__cell_dimensions = [l / n for l, n in zip(self.__domain_size, self.__grid_points)]

        self.__cell_centerted_coordinates_2D()
        self.__x_staggered_coordinates_2D()
        self.__y_staggered_coordinates_2D()

    @property
    def shape(self):
        return (self.__grid_points[1],self.__grid_points[0])

    @property
    def nx(self):
        return self.__grid_points[0]

    @property
    def ny(self):
        return self.__grid_points[1]

    @property
    def dx(self):
        return self.__cell_dimensions[0]

    @property
    def dy(self):
        return self.__cell_dimensions[1]

    @property
    def lx(self):
        return self.__domain_size[0]

    @property
    def ly(self):
        return self.__domain_size[1]

    @property
    def XSVol(self):
        return self.__XSVol

    @property
    def YSVol(self):
        return self.__YSVol

    @property
    def XXVol(self):
        return self.__XXVol

    @property
    def YXVol(self):
        return self.__YXVol

    @property
    def XYVol(self):
        return self.__XYVol

    @property
    def YYVol(self):
        return self.__YYVol

    def __cell_centerted_coordinates_2D(self):
        # cell centered coordinates
        xx = np.linspace(self.__cell_dimensions[0] / 2.0, self.__domain_size[0] - self.__cell_dimensions[0] / 2.0, self.__grid_points[0], endpoint=True)
        yy = np.linspace(self.__cell_dimensions[1] / 2.0, self.__domain_size[1] - self.__cell_dimensions[1] / 2.0, self.__grid_points[1], endpoint=True)
        self.__XSVol, self.__YSVol = np.meshgrid(xx, yy)

    def __x_staggered_coordinates_2D(self):
        # x-staggered coordinates
        yy = np.linspace(self.__cell_dimensions[1] / 2.0, self.__domain_size[1] - self.__cell_dimensions[1] / 2.0, self.__grid_points[1], endpoint=True)
        xxs = np.linspace(0, self.__domain_size[0], self.__grid_points[0] + 1, endpoint=True)
        self.__XXVol, self.__YXVol = np.meshgrid(xxs, yy)

    def __y_staggered_coordinates_2D(self):
        # y-staggered coordinates
        xx = np.linspace(self.__cell_dimensions[0] / 2.0, self.__domain_size[0] - self.__cell_dimensions[0] / 2.0, self.__grid_points[0], endpoint=True)
        yys = np.linspace(0, self.__domain_size[1], self.__grid_points[1] + 1, endpoint=True)
        self.__XYVol, self.__YYVol = np.meshgrid(xx, yys)
