import numpy as np
import math


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
        DEMO:
            import sys
            import numpy as np
            import math
            #根据电脑实际路径添加Geomeas库
            sys.path.append('C:\\Users\\Administrator\\Desktop\\geo-meas-master\\src')
            #确认路径是否添加
            import geomeas as gm
            P3 = np.array([-121.015239,-264.506911,-189.283544])
            P4 = np.array([-75.884264,-165.862683,-59.067218])
            P5 = np.array([-7.196526,-262.243481,11.932582])
            P6 = np.array([-450.731169,-243.407134,-573.947270])
            v3 = gm.Vector().calVectorFrom2Points(P4,P3)
            v4 = gm.Vector().calVectorFrom2Points(P5,P6)
            PM, PN, d = gm.Coordinate().calCoordinatesFrom2SpatialLines(P1=P4, v1=v3, P2=P5, v2=v4)
            print('PM:', PM)
            print('PN:', PN)
            print('d:', d)
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
        DEMO:
            import sys
            import numpy as np
            #根据电脑实际路径添加Geomeas库
            sys.path.append('C:\\Users\\Administrator\\Desktop\\geo-meas-master\\src')
            #确认路径是否添加
            import geomeas as gm
            #相交直线
            P0 = np.array([228.578593,-69.260001,-143.548547])#交点坐标
            P1 = np.array([-31.687322,-69.260001,-143.548547])
            P2 = np.array([265.412025,11.248018,182.180592])
            v1 = gm.Vector().calVectorFrom2Points(P1,P0)
            v2 = gm.Vector().calVectorFrom2Points(P2,P0)
            retdict = gm.Coordinate().calCoordinateFrom2LinesByLS(P1=P1, v1=v1, P2=P2, v2=v2)
            print('相交直线求解：', retdict)
            #不相交直线
            P3 = np.array([-121.015239,-264.506911,-189.283544])
            P4 = np.array([-75.884264,-165.862683,-59.067218])
            P5 = np.array([-642.494851,-297.194175,-143.548547])
            P6 = np.array([88.684371,-221.630072,-143.548547])
            X_Center = np.array([[-122.457462],[-252.096252],[-159.575206]])#公垂线中点坐标
            v3 = gm.Vector().calVectorFrom2Points(P3,P4)
            v4 = gm.Vector().calVectorFrom2Points(P5,P6)
            retdict = gm.Coordinate().calCoordinateFrom2LinesByLS(P1=P3, v1=v3, P2=P6, v2=v4)
            print('不相交直线求解：', retdict)
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
        DEMO:
                import geomeas as gm
                import numpy as np
                P_1 = np.array([-108.45, 174.45, 0])
                s_1 = np.array([-335.6, -77.27, 0])
                Q_1 = np.array([227.15, 174.45, 0])
                s_2 = np.array([335.6, -77.27, 0])
                print(gm.Coordinate().calCoordinateFrom2Lines(P_1, s_1, Q_1, s_2))
        参考资料：邱维声. 解析几何（第三版）[M]. 北京：北京大学出版社，2015.60-63.
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
        DEMO：

                P1 = np.array([-670.13, 1477.30, -1576.88])
                P2 = np.array([-761.34, 914.65, -1576.88])
                PlaneParams = np.array([0.00000000e+00, 5.54216347e+05, 0.00000000e+00, -3.57990507e+08])
                print(calCoordinateFrom2PointsAndPlane(P1, P2, PlaneParams))
        参考资料：邱维声. 解析几何（第三版）[M]. 北京：北京大学出版社，2015.60-65.
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
        DEMO:   import geomeas as gm
                import numpy as np
                import math
                P_1 = np.array([13.66, 121.55, 0])
                P_2 = np.array([101.84, 121.55, 136.68])
                print(gm.Distance().calDistanceFrom2Points(P_1, P_2))
        参考资料：邱维声. 解析几何（第三版）[M]. 北京：北京大学出版社，2015.71.
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
        DEMO ：
                Point1 = np.array([-2919.13, 654.94, 1100])
                Point2 = np.array([-2919.13, 0, 1100])
                Point3 = np.array([-2438.31, 0, 0])
                print(calPlaneEquationFrom3Points(Point1, Point2, Point3))
        参考资料：邱维声. 解析几何（第三版）[M]. 北京：北京大学出版社，2015.48-51.
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
        DEMO:   import geomeas as gm
                import numpy as np
                Point_1 = np.array([227.15, 174.45, 0])
                Point_2 = np.array([-108.45, 251.72, 0])
                print(gm.Vector().calVectorFrom2Points(Point_1, Point_2))
        参考资料：邱维声. 解析几何（第三版）[M]. 北京：北京大学出版社，2015.6-10.
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
        DEMO:   import geomeas as gm
                import numpy as np
                Param_1 = np.array([0, 0, -4361.9337, 362040.4971])
                Param_2 = np.array([-2180.41, 6939.63, 2165.499, -283785.5822])
                print(gm.Vector().calLineVictorFrom2Planes(Param_1, Param_2))
        参考资料：邱维声. 解析几何（第三版）[M]. 北京：北京大学出版社，2015.30-36,48-51.
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
    '''
    Calculation Pose Matrix from other geometric elements.
    '''

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
        DEMO：
                import geomeas as gm
                import numpy as np

                Oab = np.array([-37.84381632, 152.36389864, 41.68600167])
                Pxb = np.array([-19.59820338, 139.58818292, 45.55380309])
                Pyb = np.array([-38.23270656, 157.3130709, 59.86810327])

                print(gm.Pose().calPoseFrom3Points(Oab, Pxb, Pyb))
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
        DEMO：
                import geomeas as gm
                import numpy as np

                Vs1 = np.array([0.55397988, 0.82791962, -0.08749517])
                Vs2 = np.array([0.02063334, -0.26258813, -0.96468738])
                Vn1 = np.array([0.97066373, 0.20744552, 0.12156592])
                Vn2 = np.array([0, 0, -1])

                print(gm.Pose().calOrientationFrom2Vectors(Vs1, Vs2, Vn1, Vn2))
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
        DEMO：
                import cglgl as cg
                import numpy as np

                ol=np.array([10,20,60])
                print(cg.Pose().oltomatrix(ol))
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
