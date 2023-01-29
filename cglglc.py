import numpy as np
import math
import cv2


class Coordinate(object):
    def __init__(self, *args):
        super(Coordinate, self).__init__(*args)

    def calSparkCenterCoordianteFromImg(self, img, options, threshold):
        '''
        作者：lgl；
        日期：2023.1.29；
        功能：将一张图片中的光斑分别用质心法，轮廓法和BLOB方法识别出，并计算光斑质心坐标，返回光斑质心坐标和识别到的光斑数；
        输入：img：原始图像；option：'Contours'—轮廓，‘Blob’—Blob，‘Centroid’,threshold:二值化阈值；
        输出：CenterXY：质心图像坐标，以list形式输出（允许有多个），num：识别到的光斑个数；
        '''
        ###Please code here...
        CenterXY = 0.0
        num = 0
        ###
        return CenterXY, num

    def calCoordinatesFrom2SpatialLines(self, P1, v1, P2, v2):
        '''
        作者：lgl；
        日期：2023.1.29；
        功能：计算两条空间直线之间公垂线与两条直线交点的坐标与公垂线段的距离；
        输入：直线1、2上的定点和单位方向矢量；
        输出：D：两条公垂线之间的距离；PM：公垂线和直线1的交点坐标；PN：公垂线和直线二的交点坐标；
        '''
        l = np.cross(v2, v1)
        l = l / np.linalg.norm(l)
        A = np.array([[v1[1], -v1[0], 0, 0],
                      [0, v1[2], -v1[1], 0],
                      [v2[1], -v2[0], 0, v2[1] * l[0] - v2[0] * l[1]],
                      [0, v2[2], -v2[1], v2[2] * l[1] - v2[1] * l[2]]])
        B = np.array([[v1[1] * P1[0] - v1[0] * P1[1]],
                      [v1[2] * P1[1] - v1[1] * P1[2]],
                      [v2[1] * P2[0] - v2[0] * P2[1]],
                      [v2[2] * P2[1] - v2[1] * P2[2]]])

        XYZD = np.dot(np.linalg.inv(A), B)
        print('XYZD：', XYZD)
        PM = XYZD[0:3, :]
        D = XYZD[3, 0]
        print('l:', np.transpose(l))
        PN = PM + np.array([[l[0]], [l[1]], [l[2]]]) * D
        return PM, PN, abs(D)

    def calCoordinateFrom2LinesByLS(self, P1, v1, P2, v2):
        '''
        作者：lgl；
        日期：2023.1.29；
        功能：计算两条空间直线的交点，先判断有无交点，如有，则输出解析解，如无，则输出最小二乘解；
        输入：直线1、2上的定点和方向矢量；
        输出：ret:0表示有交点，非零表示无交点；X_LS:两条直线的最小二乘解；
        '''
        D = np.array([[P2[0] - P1[0], v1[0], v2[0]],
                      [P2[1] - P1[1], v1[1], v2[1]],
                      [P2[2] - P1[2], v1[2], v2[2]]])
        ret = np.linalg.det(D)
        # 解算最小二乘解：
        A = np.array([[v1[1], -v1[0], 0],
                      [0, v1[2], -v1[1]],
                      [v2[1], -v2[0], 0],
                      [0, v2[2], -v2[1]]])
        B = np.array([[v1[1] * P1[0] - v1[0] * P1[1]],
                      [v1[2] * P1[1] - v1[1] * P1[2]],
                      [v2[1] * P2[0] - v2[0] * P2[1]],
                      [v2[2] * P2[1] - v2[1] * P2[2]]])
        X_LS = np.dot(np.dot(np.linalg.inv(np.dot(A.T, A)), A.T), B)
        Err = np.dot(A, X_LS) - B
        X_dict = {'ret:': ret,
                  'X_LS:': X_LS,
                  'Err:': Err}
        return X_dict

    def calCoordinateFrom2Lines(self, P_1, s_1, Q_1, s_2):
        '''
        作者：lgl；
        日期：2023.1.29；
        功能：计算空间两直线的交点；
        输入：利用空间直线的“点向式”方程，分别输入直线一点与该直线的方向向量P_1(x_1,y_1,z_1),s_1(m_1,n_1,p_1),Q_1(x_2,y_2,z_2),s_2(m_2,n_2,p_2);
        输出：两直线的交点M(x,y,z)；
        '''
        m_1 = np.array([s_1[0]])
        n_1 = np.array([s_1[1]])
        p_1 = np.array([s_1[2]])
        m_2 = np.array([s_2[0]])
        n_2 = np.array([s_2[1]])
        x_1 = np.array([P_1[0]])
        y_1 = np.array([P_1[1]])
        z_1 = np.array([P_1[2]])
        x_2 = np.array([Q_1[0]])
        y_2 = np.array([Q_1[1]])
        t = (m_2 * (y_2 - y_1) + n_2 * (x_1 - x_2)) / ((m_2 * n_1) - (m_1 * n_2))
        x = x_1 + m_1 * t
        y = y_1 + n_1 * t
        z = z_1 + p_1 * t
        return x, y, z

    def calCoordinateFrom2PointsAndPlane(self, P1, P2, PlaneParams):
        '''
        作者：lgl；
        日期：2023.1.29；
        功能：直线和平面计算交点坐标；
        输入：空间直线任意两点坐标P1(x1,y1,z1),P2(x2,y2,z2)，平面方程的参数PlaneParams(a,b,c,d);
        输出：空间直线与平面的交点E(x,y,z)，输出格式为数组（[x y z]）;
        '''
        a = np.array([PlaneParams[0]])
        b = np.array([PlaneParams[1]])
        c = np.array([PlaneParams[2]])
        d = np.array([PlaneParams[3]])

        LineVector = np.array(P1 - P2)
        m = LineVector[0]
        n = LineVector[1]
        p = LineVector[2]
        x1 = np.array(P1[0])
        y1 = np.array(P1[1])
        z1 = np.array(P1[2])
        t = (-a * x1 - b * y1 - c * z1 - d) / (a * m + b * n + c * p)
        x = m * t + x1
        y = n * t + y1
        z = p * t + z1
        return x, y, z


class Distance():
    def __init__(self, *args):
        super(Distance, self).__init__(*args)

    def calDistanceFrom2Points(self, P_1, P_2):
        '''
        作者：lgl；
        日期：2023.1.29；
        功能：计算空间两点距离；
        输入：空间内任意两点坐标P_1(x1,y1,z1),P_2(x2,y2,z2);
        输出：两点的距离Distance；
        '''
        x1 = np.array(P_1[0])
        y1 = np.array(P_1[1])
        z1 = np.array(P_1[2])
        x2 = np.array(P_2[0])
        y2 = np.array(P_2[1])
        z2 = np.array(P_2[2])
        X = x1 - x2
        Y = y1 - y2
        Z = z1 - z2
        #欧几里得范数是从原点到给定坐标的距离
        Dis = math.hypot(X, Y, Z)
        return Dis


class Angle():
    pass


class Plane():
    def calPlaneFrom3Points(self, Point1, Point2, Point3):
        '''
        作者：lgl；
        日期：2023.1.29；
        功能 ： 通过三点求解平面方程参数；
        输入 ： 平面上任意三点坐标 Point1(xo1,yo1,zo1),Point2(xo2,yo2,zo2),Point3(xo3,yo3,zo3);
        输出 ： 平面方程的参数 PlaneParams = (a,b,c,d)，输出格式为数组（[a b c d]）;
        '''
        xo1 = Point1[0]
        yo1 = Point1[1]
        zo1 = Point1[2]
        xo2 = Point2[0]
        yo2 = Point2[1]
        zo2 = Point2[2]
        xo3 = Point3[0]
        yo3 = Point3[1]
        zo3 = Point3[2]
        a = (yo2 - yo1) * (zo3 - zo1) - (yo3 - yo1) * (zo2 - zo1)
        b = (zo2 - zo1) * (xo3 - xo1) - (xo3 - zo1) * (xo2 - xo1)
        c = (xo2 - xo1) * (yo3 - yo1) - (xo3 - xo1) * (yo2 - yo1)
        d = -(a * xo1 + b * yo1 + c * zo1)
        return a, b, c, d


class Vector():

    def calVectorFrom2Points(self, Point_1, Point_2):
        '''
        作者：lgl；
        日期：2023.1.29；
        功能：利用空间两点计算单位方向向量；
        输入：空间内任意两点坐标Point_1(x1,y1,z1),Point_2(x2,y2,z2);
        输出：空间方向向量s；
        '''
        x1 = np.array(Point_1[0])
        y1 = np.array(Point_1[1])
        z1 = np.array(Point_1[2])
        x2 = np.array(Point_2[0])
        y2 = np.array(Point_2[1])
        z2 = np.array(Point_2[2])
        m = x2 - x1
        n = y2 - y1
        p = z2 - z1
        d = math.sqrt(m * m + n * n + p * p)
        return np.array([m / d, n / d, p / d])

    def calVectorFrom2Planes(self, Params_1, Params_2):
        '''
        作者：lgl；
        日期：2023.1.29；
        功能：计算空间两平面交线的方向向量；
        输入：两空间平面方程的参数Params_1(a_1,b_1,c_1,d_1),Params_2(a_2,b_2,c_2,d_2);
        输出：空间两平面交线的方向向量s；
        '''
        a_1 = np.array(Params_1[0])
        b_1 = np.array(Params_1[1])
        c_1 = np.array(Params_1[2])
        a_2 = np.array(Params_2[0])
        b_2 = np.array(Params_2[1])
        c_2 = np.array(Params_2[2])
        m = b_1 * c_2 - b_2 * c_1
        n = c_1 * a_2 - a_1 * c_2
        p = a_1 * b_2 - a_2 * b_1
        return m, n, p


class Pose():
    def calPoseFrom3Points(self, Oab, Pxb, Pyb):
        '''
        作者：lgl；
        日期：2023.1.29；
        功能：已知坐标系a的原点、x轴正半轴上任一点和y轴正半轴上任一点在坐标系b下的坐标，
            求解坐标系a到坐标系b的旋转矩阵R和平移矩阵T；
        输入：坐标系a的原点在坐标系b下的坐标:Oab(x1,y1,z1);
            坐标系a的x轴正半轴上任一点在坐标系b下的坐标:Pxb(x2,y2,z2);
            坐标系a的y轴正半轴上任一点在坐标系b下的坐标:Pyb(x3,y3,z3);
        输出：坐标系n到坐标系s的旋转矩阵Rns，输出格式为矩阵;
        '''
        x = (Pxb - Oab) / np.linalg.norm(Pxb - Oab)
        y = (Pyb - Oab) / np.linalg.norm(Pyb - Oab)
        z = np.cross(x, y)
        length = np.linalg.norm(z)
        z = z / length
        Rab = np.matrix([x, y, z]).transpose()
        Tab = np.matrix(Oab).transpose()
        return Rab, Tab

    def calOrientationFrom2Vectors(self, Vs1, Vs2, Vn1, Vn2):
        '''
        作者：lgl；
        日期：2023.1.29；
        功能：根据两个不共线的矢量分别在两个坐标系下的矢量坐标求解这两个坐标系之间的旋转矩阵；
        输入：两个矢量在坐标系s下的坐标:Vs1(xs1,ys1,zs1),Vs2(xs2,ys2,zs2);
            在坐标系n下的坐标:Vn1(xn1,yn1,zn1),Vn2(xn2,yn2,zn2);
        输出：坐标系n到坐标系s的旋转矩阵Rns，输出格式为矩阵;
        '''
        # frame s
        a = Vs1
        b = np.cross(Vs1, Vs2) / np.linalg.norm(np.cross(Vs1, Vs2))
        c = np.cross(a, b)
        # 参考坐标系frame d
        Rsd = np.array([a, b, c])

        # frame n
        A = Vn1
        B = np.cross(Vn1, Vn2) / np.linalg.norm(np.cross(Vn1, Vn2))
        C = np.cross(A, B)
        Rnd = np.array([A, B, C])
        Rns = np.dot(np.linalg.inv(Rsd), Rnd)

        return Rns

    def oltomatrix(self, ol):
        '''
        作者：lgl；
        日期：2023.1.29；
        功能：欧拉角转换成旋转矩阵；
        输入：欧拉角
        输出：旋转矩阵;
        '''
        ax = ol[0]
        by = ol[1]
        cz = ol[2]
        sax = math.sin(2 * math.pi / 360 * ax)
        cax = math.cos(2 * math.pi / 360 * ax)
        sby = math.sin(2 * math.pi / 360 * by)
        cby = math.cos(2 * math.pi / 360 * by)
        scz = math.sin(2 * math.pi / 360 * cz)
        ccz = math.cos(2 * math.pi / 360 * cz)
        matrix = np.array([[cax * ccz - sax * cby * scz, -sax * ccz - cax * cby * scz, sby * scz],
                           [cax * scz + sax * cby * ccz, -sax * scz + cax * cby * ccz, -sby * ccz],
                           [sax * sby, cax * sby, cby]])
        return matrix

    def vectormatrix(self,v1,R1):
        '''
        作者：lgl；
        日期：2023.1.29；
        功能：向量与旋转矩阵相乘；
        输入：向量，旋转矩阵
        输出：向量;
        '''
        return np.array([v1[0]*R1[0][0]+v1[1]*R1[0][1]+v1[2]*R1[0][2],v1[0]*R1[1][0]+v1[1]*R1[1][1]+v1[2]*R1[1][2],v1[0]*R1[2][0]+v1[1]*R1[2][1]+v1[2]*R1[2][2]])

    def makeVectorfromarray(self,arrayv):
        '''
        作者：lgl；
        日期：2023.1.29；
        功能：向量转单位向量；
        输入：向量
        输出：向量;
        '''
        d = arrayv[0]+arrayv[1]+arrayv[2]
        return np.array([arrayv[0]/d,arrayv[1]/d,arrayv[2]/d])

    def dot_product_angle(self, v1, v2):
        '''
        作者：lgl；
        日期：2023.1.29；
        功能：计算两个向量夹角；
        输入：向量，向量
        输出：夹角角度;
        '''
        if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
            print("Zero magnitude vector!")
        else:
            vector_dot_product = np.dot(v1, v2)
            arccos = np.arccos(vector_dot_product / (np.linalg.norm(v1) * np.linalg.norm(v2)))
            angle = np.degrees(arccos)
            return angle
        return 0
    def jisuan(self,v1,v2):
        '''
        作者：lgl；
        日期：2023.1.29；
        功能：计算两个直线夹角；
        输入：向量，向量
        输出：夹角角度;
        '''
        dx1 = v1[2] - v1[0]
        dy1 = v1[3] - v1[1]
        dx2 = v2[2] - v2[0]
        dy2 = v2[3] - v2[1]
        angle1 = math.atan2(dy1, dx1)
        angle1 = int(angle1 * 180 / math.pi)
        angle2 = math.atan2(dy2, dx2)
        angle2 = int(angle2 * 180 / math.pi)
        if angle1 * angle2 >= 0:
            included_angle = abs(angle1 - angle2)
        else:
            included_angle = abs(angle1) + abs(angle2)
            if included_angle > 180:
                included_angle = 360 - included_angle
        return included_angle

    def eulerAnglesToRotationMatrix(self, angles1):
        '''
        作者：lgl；
        日期：2023.1.29；
        功能：欧拉角转旋转矩阵；
        输入：欧拉角
        输出：旋转矩阵;
        '''
        theta = np.zeros((3, 1), dtype=np.float64)
        theta[0] = angles1[0] * 3.141592653589793 / 180.0
        theta[1] = angles1[1] * 3.141592653589793 / 180.0
        theta[2] = angles1[2] * 3.141592653589793 / 180.0
        R_x = np.array([[1, 0, 0],
                        [0, math.cos(theta[0]), -math.sin(theta[0])],
                        [0, math.sin(theta[0]), math.cos(theta[0])]
                        ])
        R_y = np.array([[math.cos(theta[1]), 0, math.sin(theta[1])],
                        [0, 1, 0],
                        [-math.sin(theta[1]), 0, math.cos(theta[1])]
                        ])
        R_z = np.array([[math.cos(theta[2]), -math.sin(theta[2]), 0],
                        [math.sin(theta[2]), math.cos(theta[2]), 0],
                        [0, 0, 1]
                        ])
        R = np.dot(R_z, np.dot(R_y, R_x))
        sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
        singular = sy < 1e-6
        if not singular:
            x = math.atan2(R[2, 1], R[2, 2])
            y = math.atan2(-R[2, 0], sy)
            z = math.atan2(R[1, 0], R[0, 0])
        else:
            x = math.atan2(-R[1, 2], R[1, 1])
            y = math.atan2(-R[2, 0], sy)
            z = 0
        # print('dst:', R)
        x = x * 180.0 / 3.141592653589793
        y = y * 180.0 / 3.141592653589793
        z = z * 180.0 / 3.141592653589793
        rvecstmp = np.zeros((1, 1, 3), dtype=np.float64)
        rvecs, _ = cv2.Rodrigues(R, rvecstmp)
        # print()
        return R, rvecs, x, y, z

    def isRotationMatrix(self,R):
        '''
        作者：lgl；
        日期：2023.1.29；
        功能：判断是不是旋转矩阵；
        输入：旋转矩阵
        输出：True or False;
        '''
        Rt = np.transpose(R)
        shouldBeIdentity = np.dot(Rt, R)
        I = np.identity(3, dtype=R.dtype)
        n = np.linalg.norm(I - shouldBeIdentity)
        return n < 1e-6

    def rotationMatrixToEulerAngles(self,R):
        '''
        作者：lgl；
        日期：2023.1.29；
        功能：旋转矩阵转欧拉角；
        输入：旋转矩阵
        输出：欧拉角;
        '''
        assert (self.isRotationMatrix(R))

        sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

        singular = sy < 1e-6

        if not singular:
            x = math.atan2(R[2, 1], R[2, 2])
            y = math.atan2(-R[2, 0], sy)
            z = math.atan2(R[1, 0], R[0, 0])
        else:
            x = math.atan2(-R[1, 2], R[1, 1])
            y = math.atan2(-R[2, 0], sy)
            z = 0
        x = x * 180.0 / 3.141592653589793
        y = y * 180.0 / 3.141592653589793
        z = z * 180.0 / 3.141592653589793
        return np.array([x, -y, z])
    def oljisuan(self,Point_1, Point_2, Point_3, Point_4):
        '''
        作者：lgl；
        日期：2023.1.29；
        功能：根据四个点计算欧拉角；
        输入：四个点三维坐标
        输出：欧拉角;
        '''

        A = Vector().calVectorFrom2Points(Point_1, Point_2)

        B = Vector().calVectorFrom2Points(Point_3, Point_4)
        # print(A)
        # print(B)
        origin_vector = np.array(A)
        location_vector = np.array(B)
        #点积
        c = np.dot(origin_vector, location_vector)
        #计算两个向量（向量数组）的叉乘。叉乘返回的数组既垂直于a，又垂直于b
        n_vector = np.cross(origin_vector, location_vector)
        #linalg本意为linear(线性) + algebra(代数)，norm则表示范数
        s = np.linalg.norm(n_vector)
        # print(c, s)

        n_vector_invert = np.array((
            [0, -n_vector[2], n_vector[1]],
            [n_vector[2], 0, -n_vector[0]],
            [-n_vector[1], n_vector[0], 0]
        ))
        #生成单位矩阵
        I = np.eye(3)
        R_w2c = I + n_vector_invert + np.dot(n_vector_invert, n_vector_invert) / (1 + c)
        # print('R_w2c',R_w2c)
        ola = self.rotationMatrixToEulerAngles(R_w2c)
        # olb = self.rotateMatrixToEulerAngles2(R_w2c,R_w2c)
        # print('ola',ola)
        return ola

    def TowVectorToMatrix(self,A, B):
        '''
        作者：lgl；
        日期：2023.1.29；
        功能：计算两个向量之间的旋转矩阵；
        输入：向量，向量
        输出：旋转矩阵;
        '''
        origin_vector = np.array(A)
        location_vector = np.array(B)
        #点积
        c = np.dot(origin_vector, location_vector)
        #计算两个向量（向量数组）的叉乘。叉乘返回的数组既垂直于a，又垂直于b
        n_vector = np.cross(origin_vector, location_vector)
        #linalg本意为linear(线性) + algebra(代数)，norm则表示范数
        s = np.linalg.norm(n_vector)
        # print(c, s)

        n_vector_invert = np.array((
            [0, -n_vector[2], n_vector[1]],
            [n_vector[2], 0, -n_vector[0]],
            [-n_vector[1], n_vector[0], 0]
        ))
        #生成单位矩阵
        I = np.eye(3)
        R_w2c = I + n_vector_invert + np.dot(n_vector_invert, n_vector_invert) / (1 + c)
        return R_w2c

    def oljisuanliang(self,A, B):
        '''
        作者：lgl；
        日期：2023.1.29；
        功能：两个向量计算欧拉角；
        输入：向量，向量
        输出：欧拉角;
        '''
        origin_vector = np.array(A)
        location_vector = np.array(B)
        #点积
        c = np.dot(origin_vector, location_vector)
        #计算两个向量（向量数组）的叉乘。叉乘返回的数组既垂直于a，又垂直于b
        n_vector = np.cross(origin_vector, location_vector)
        #linalg本意为linear(线性) + algebra(代数)，norm则表示范数
        s = np.linalg.norm(n_vector)
        # print(c, s)

        n_vector_invert = np.array((
            [0, -n_vector[2], n_vector[1]],
            [n_vector[2], 0, -n_vector[0]],
            [-n_vector[1], n_vector[0], 0]
        ))
        #生成单位矩阵
        I = np.eye(3)
        R_w2c = I + n_vector_invert + np.dot(n_vector_invert, n_vector_invert) / (1 + c)
        # print('R_w2c',R_w2c)
        ola = self.rotationMatrixToEulerAngles(R_w2c)
        # olb = self.rotateMatrixToEulerAngles2(R_w2c,R_w2c)
        # print('ola',ola)
        return ola

    def findcenter(self, land23, land24):
        '''
        作者：lgl；
        日期：2023.1.29；
        功能：找到两点中点坐标；
        输入：坐标点，坐标点
        输出：坐标点;
        '''
        return [(land23[0] + land24[0]) / 2.0, (land23[1] + land24[1]) / 2.0, (land23[2] + land24[2]) / 2.0]
