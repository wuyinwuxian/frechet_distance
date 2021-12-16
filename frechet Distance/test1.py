import numpy as np
from typing import Callable, Dict
from numba import jit
from timeit import default_timer as timer
from distance_lib import euclidean, haversine, earth_haversine


class DiscreteFrechet(object):
    """
    Calculates the discrete Fréchet distance between two poly-lines using the
    original recursive algorithm
    """

    def __init__(self, dist_func):
        """
        Initializes the instance with a pairwise distance function.
        :param dist_func: The distance function. It must accept two NumPy
        arrays containing the point coordinates (x, y), (lat, long)
        """
        self.dist_func = dist_func
        self.ca = np.array([0.0])

    def distance(self, p: np.ndarray, q: np.ndarray) -> float:
        """
        Calculates the Fréchet distance between poly-lines p and q
        This function implements the algorithm described by Eiter & Mannila
        :param p: Poly-line p
        :param q: Poly-line q
        :return: Distance value
        """

        def calculate(i: int, j: int) -> float:
            """
            Calculates the distance between p[i] and q[i]
            :param i: Index into poly-line p
            :param j: Index into poly-line q
            :return: Distance value
            """
            if self.ca[i, j] > -1.0:
                return self.ca[i, j]

            d = self.dist_func(p[i], q[j])
            if i == 0 and j == 0:
                self.ca[i, j] = d
            elif i > 0 and j == 0:
                self.ca[i, j] = max(calculate(i-1, 0), d)
            elif i == 0 and j > 0:
                self.ca[i, j] = max(calculate(0, j-1), d)
            elif i > 0 and j > 0:
                self.ca[i, j] = max(min(calculate(i-1, j),
                                        calculate(i-1, j-1),
                                        calculate(i, j-1)), d)
            else:
                self.ca[i, j] = np.infty
            return self.ca[i, j]

        n_p = p.shape[0]
        n_q = q.shape[0]
        self.ca = np.zeros((n_p, n_q))
        self.ca.fill(-1.0)
        return calculate(n_p - 1, n_q - 1)


@jit(nopython=True)
def _get_linear_frechet(p: np.ndarray, q: np.ndarray,
                        dist_func: Callable[[np.ndarray, np.ndarray], float]) \
        -> np.ndarray:
    n_p = p.shape[0]
    n_q = q.shape[0]
    ca = np.zeros((n_p, n_q), dtype=np.float64)

    for i in range(n_p):
        for j in range(n_q):
            d = dist_func(p[i], q[j])

            if i > 0 and j > 0:
                ca[i, j] = max(min(ca[i - 1, j],
                                   ca[i - 1, j - 1],
                                   ca[i, j - 1]), d)
            elif i > 0 and j == 0:
                ca[i, j] = max(ca[i - 1, 0], d)
            elif i == 0 and j > 0:
                ca[i, j] = max(ca[0, j - 1], d)
            elif i == 0 and j == 0:
                ca[i, j] = d
            else:
                ca[i, j] = np.infty
    return ca


class LinearDiscreteFrechet(DiscreteFrechet):

    def __init__(self, dist_func):
        DiscreteFrechet.__init__(self, dist_func)
        # JIT the numba code
        self.distance(np.array([[0.0, 0.0], [1.0, 1.0]]),
                      np.array([[0.0, 0.0], [1.0, 1.0]]))

    def distance(self, p: np.ndarray, q: np.ndarray) -> float:
        n_p = p.shape[0]
        n_q = q.shape[0]
        self.ca = _get_linear_frechet(p, q, self.dist_func)
        return self.ca[n_p - 1, n_q - 1]


@jit(nopython=True)
def distance_matrix(p: np.ndarray,
                    q: np.ndarray,
                    dist_func: Callable[[np.array, np.array], float]) \
        -> np.ndarray:
    n_p = p.shape[0]
    n_q = q.shape[0]
    dist = np.zeros((n_p, n_q), dtype=np.float64)
    for i in range(n_p):
        for j in range(n_q):
            dist[i, j] = dist_func(p[i], q[j])
    return dist


class VectorizedDiscreteFrechet(DiscreteFrechet):

    def __init__(self, dist_func):
        DiscreteFrechet.__init__(self, dist_func)
        self.dist = np.array([0.0])

    def distance(self, p: np.ndarray, q: np.ndarray) -> float:
        """
        Calculates the Fréchet distance between poly-lines p and q
        This function implements the algorithm described by Eiter & Mannila
        :param p: Poly-line p
        :param q: Poly-line q
        :return: Distance value
        """

        def calculate(i: int, j: int) -> float:
            """
            Calculates the distance between p[i] and q[i]
            :param i: Index into poly-line p
            :param j: Index into poly-line q
            :return: Distance value
            """
            if self.ca[i, j] > -1.0:
                return self.ca[i, j]

            d = self.dist[i, j]
            if i == 0 and j == 0:
                self.ca[i, j] = d
            elif i > 0 and j == 0:
                self.ca[i, j] = max(calculate(i-1, 0), d)
            elif i == 0 and j > 0:
                self.ca[i, j] = max(calculate(0, j-1), d)
            elif i > 0 and j > 0:
                self.ca[i, j] = max(min(calculate(i-1, j),
                                        calculate(i-1, j-1),
                                        calculate(i, j-1)), d)
            else:
                self.ca[i, j] = np.infty
            return self.ca[i, j]

        n_p = p.shape[0]
        n_q = q.shape[0]
        self.ca = np.zeros((n_p, n_q))
        self.ca.fill(-1.0)
        self.dist = distance_matrix(p, q, dist_func=self.dist_func)
        return calculate(n_p - 1, n_q - 1)



@jit(nopython=True)
def compression_get_linear_frechet(p: np.ndarray, q: np.ndarray,
                        dist_func: Callable[[np.ndarray, np.ndarray], float]) \
        -> np.ndarray:
    n_p = p.shape[0]
    n_q = q.shape[0]
    ca = np.zeros((n_q), dtype=np.float64)
    for j in range(n_q):
        ca[j] = dist_func(p[0], q[j])

    for i in range(1,n_p):
        temp = ca[0]
        # 初始化最开始的
        d = dist_func(p[i], q[0])
        ca[0] = max(ca[0], d)
        for j in range(1,n_q):
            left_up = temp
            d = dist_func(p[i], q[j])
            temp = ca[j]
            ca[j] = max(min(ca[j],left_up,ca[j - 1]), d)
    return ca


class compressionLinearDiscreteFrechet(DiscreteFrechet):

    def __init__(self, dist_func):
        DiscreteFrechet.__init__(self, dist_func)
        # JIT the numba code
        self.distance(np.array([[0.0, 0.0], [1.0, 1.0]]),
                      np.array([[0.0, 0.0], [1.0, 1.0]]))

    def distance(self, p: np.ndarray, q: np.ndarray) -> float:
        self.ca = compression_get_linear_frechet(p, q, self.dist_func)
        return self.ca[-1]


def main():
    np.set_printoptions(precision=4)
    distance_fun = earth_haversine

    linear_frechet = LinearDiscreteFrechet(distance_fun)
    slow_frechet = DiscreteFrechet(distance_fun)
    VDF = VectorizedDiscreteFrechet(distance_fun)
    compression_linear = compressionLinearDiscreteFrechet(distance_fun)

    p = np.array([
       [ 42.31644861, -83.77730583],
       [ 42.31631778, -83.77754194],
       [ 42.31619611, -83.77816083],
       [ 42.31604194, -83.77894   ],
       [ 42.31589861, -83.7797925 ],
       [ 42.31565222, -83.78029194],
       [ 42.31527111, -83.780575  ],
       [ 42.31491194, -83.78094639],
       [ 42.31462333, -83.78133361],
       [ 42.31411778, -83.78156806],
       [ 42.31361861, -83.78157389],
       [ 42.31300028, -83.78156694],
       [ 42.31235028, -83.78153222],
       [ 42.31171806, -83.78150528],
       [ 42.31097083, -83.78147556],
       [ 42.31028861, -83.78144833],
       [ 42.30959611, -83.78140667],
       [ 42.30890083, -83.78137667],
       [ 42.30821972, -83.78134028],
       [ 42.30762528, -83.78131917],
       [ 42.30691111, -83.781295  ],
       [ 42.30626583, -83.78123278],
       [ 42.30579611, -83.78121444],
       [ 42.30541694, -83.78127667],
       [ 42.30485833, -83.78123083],
       [ 42.30413889, -83.78119778],
       [ 42.30312917, -83.78116222],
       [ 42.30247667, -83.78113083],
       [ 42.30210861, -83.78116667],
       [ 42.30192611, -83.78127389],
       [ 42.30147583, -83.78119139],
       [ 42.30077167, -83.78109194],
       [ 42.30028778, -83.7810625 ],
       [ 42.29992   , -83.7812025 ],
       [ 42.29948083, -83.78110111],
       [ 42.29883806, -83.78105444],
       [ 42.29824833, -83.78105167],
       [ 42.29765167, -83.78105667],
       [ 42.29700667, -83.78105194],
       [ 42.29634194, -83.78104889],
       [ 42.29573611, -83.7810525 ],
       [ 42.295395  , -83.78100861],
       [ 42.29534111, -83.78095944],
       [ 42.29527   , -83.78094   ],
       [ 42.29494278, -83.78086833],
       [ 42.29426028, -83.78081528],
       [ 42.29358333, -83.78078778],
       [ 42.29294528, -83.78075722],
       [ 42.29230917, -83.78073306],
       [ 42.29162361, -83.78072333],
       [ 42.29089417, -83.78071083],
       [ 42.29000861, -83.78067528],
       [ 42.28929556, -83.78065222],
       [ 42.28855944, -83.78064944],
       [ 42.28785778, -83.78065806],
       [ 42.28723694, -83.78065444],
       [ 42.28653778, -83.78063944],
       [ 42.28624111, -83.78061833],
       [ 42.286215  , -83.78061333],
       [ 42.2862275 , -83.78057944],
       [ 42.28612   , -83.78060611],
       [ 42.28583833, -83.78057194],
       [ 42.28545444, -83.7805525 ],
       [ 42.2850025 , -83.78053639],
       [ 42.28455472, -83.78052861],
       [ 42.283995  , -83.78050056],
       [ 42.28351194, -83.78048444],
       [ 42.28297472, -83.78047333],
       [ 42.28236417, -83.78045944],
       [ 42.28184889, -83.78045667],
       [ 42.28155556, -83.78045972],
       [ 42.28151444, -83.78046194],
       [ 42.2815175 , -83.78046   ],
       [ 42.2813275 , -83.78045028],
       [ 42.28095361, -83.78043917],
       [ 42.28050833, -83.78044917],
       [ 42.280135  , -83.78044611],
       [ 42.27977472, -83.78042778],
       [ 42.27919556, -83.78023417],
       [ 42.27864194, -83.77985167],
       [ 42.278055  , -83.77948111],
       [ 42.27750944, -83.77912028],
       [ 42.276865  , -83.77870917],
       [ 42.27639083, -83.77840111],
       [ 42.27569194, -83.77795083],
       [ 42.27524083, -83.77764583],
       [ 42.27509861, -83.77755361],
       [ 42.27511028, -83.77751083],
       [ 42.27509667, -83.77747278],
       [ 42.27495222, -83.7773675 ],
       [ 42.27458694, -83.77713028]])
    q = np.array([
       [ 42.31661889, -83.77759694],
       [ 42.31651444, -83.77805778],
       [ 42.31639167, -83.77864528],
       [ 42.31627694, -83.77921639],
       [ 42.31612361, -83.77979778],
       [ 42.31601167, -83.78022333],
       [ 42.31587083, -83.78046167],
       [ 42.31573444, -83.78056583],
       [ 42.31549833, -83.78067222],
       [ 42.3152775 , -83.780815  ],
       [ 42.31496194, -83.78111833],
       [ 42.31475   , -83.78141778],
       [ 42.31445972, -83.78161111],
       [ 42.31412833, -83.78162167],
       [ 42.31372361, -83.78162611],
       [ 42.31325667, -83.78158194],
       [ 42.31257306, -83.78156139],
       [ 42.31207222, -83.78153778],
       [ 42.3114975 , -83.78147417],
       [ 42.3110025 , -83.78147611],
       [ 42.310545  , -83.78148778],
       [ 42.30996028, -83.78151278],
       [ 42.30964889, -83.78154667],
       [ 42.30927861, -83.78154222],
       [ 42.30889583, -83.78153222],
       [ 42.30830583, -83.78148917],
       [ 42.30787444, -83.78151389],
       [ 42.30739361, -83.78149083],
       [ 42.30688833, -83.78149306],
       [ 42.30645611, -83.78140167],
       [ 42.30617917, -83.78143528],
       [ 42.30593583, -83.7815    ],
       [ 42.30538944, -83.78145306],
       [ 42.3048975 , -83.78143833],
       [ 42.30437361, -83.78140833],
       [ 42.30387111, -83.78138111],
       [ 42.30336361, -83.78134694],
       [ 42.30279222, -83.78133222],
       [ 42.3024175 , -83.78134167],
       [ 42.30200111, -83.78141333],
       [ 42.30142417, -83.78128833],
       [ 42.30099889, -83.78121667],
       [ 42.30059   , -83.78115861],
       [ 42.30020167, -83.78121778],
       [ 42.29977306, -83.78127833],
       [ 42.29939833, -83.78119194],
       [ 42.29899917, -83.78111944],
       [ 42.29854361, -83.78108361],
       [ 42.29815611, -83.78103944],
       [ 42.29781111, -83.78100278],
       [ 42.29738556, -83.78100111],
       [ 42.29699972, -83.78100667],
       [ 42.296655  , -83.78100444],
       [ 42.29633833, -83.78099861],
       [ 42.29605778, -83.78096972],
       [ 42.29586694, -83.78095194],
       [ 42.29575806, -83.78089528],
       [ 42.29561083, -83.78085861],
       [ 42.29547028, -83.78088333],
       [ 42.29539   , -83.78079861],
       [ 42.295295  , -83.78078528],
       [ 42.29517639, -83.78075333],
       [ 42.2949275 , -83.78072667],
       [ 42.29467417, -83.78067917],
       [ 42.29445778, -83.78064917],
       [ 42.29409972, -83.78061444],
       [ 42.293735  , -83.78059833],
       [ 42.29332611, -83.78060528],
       [ 42.29283   , -83.78062111],
       [ 42.29163472, -83.78061194],
       [ 42.291115  , -83.78057333],
       [ 42.29058611, -83.78053583],
       [ 42.28991556, -83.78050139],
       [ 42.28944694, -83.78049   ],
       [ 42.28895194, -83.7804875 ],
       [ 42.28846889, -83.78049167],
       [ 42.28796361, -83.78048278],
       [ 42.28730917, -83.78049444],
       [ 42.28680806, -83.78053278],
       [ 42.28629917, -83.78051222],
       [ 42.28586972, -83.78049972],
       [ 42.28529083, -83.78052472],
       [ 42.28475306, -83.78054   ],
       [ 42.28422444, -83.78052833],
       [ 42.28372139, -83.78050028],
       [ 42.28336   , -83.78042   ],
       [ 42.28331167, -83.78041917],
       [ 42.28296583, -83.78043139],
       [ 42.28257361, -83.78042306],
       [ 42.28213528, -83.780415  ],
       [ 42.28167583, -83.78038389],
       [ 42.28125306, -83.78035861],
       [ 42.27966083, -83.78032361],
       [ 42.27943167, -83.78029611],
       [ 42.27922417, -83.78011056],
       [ 42.27854833, -83.77966083],
       [ 42.27816278, -83.77938583],
       [ 42.27776444, -83.77910083],
       [ 42.27735694, -83.77880583],
       [ 42.27690417, -83.7785075 ],
       [ 42.27637639, -83.77812556],
       [ 42.27598167, -83.77782778],
       [ 42.27558917, -83.77756861],
       [ 42.27521528, -83.77733417],
       [ 42.27487861, -83.77709444],
       [ 42.274515  , -83.77680611],
       [ 42.27439389, -83.77672333]])

    start = timer()
    distance = slow_frechet.distance(p, q)
    end = timer()
    slow_time = end - start
    print("Slow time and ferechet distance:\n {:.8f}".format(slow_time),end="     ")
    print(distance)

    start = timer()
    distance = linear_frechet.distance(p, q)
    end = timer()
    linear_time = end - start
    print("Linear time and ferechet distance:\n {:.8f}".format(linear_time),end="     ")
    print(distance)

    start = timer()
    distance = VDF.distance(p, q)
    end = timer()
    VDF_time = end - start
    print("VDF time and ferechet distance:\n {:.8f}".format(VDF_time),end="     ")
    print(distance)

    start = timer()
    distance = compression_linear.distance(p, q)
    end = timer()
    compression_linear_time = end - start
    print("compression_linear time and ferechet distance :\n {:.8f}".format(compression_linear_time),end="     ")
    print(distance)

    print()
    print("time ratio(Based on linear):")
    print("Linear : compression_linear : :slow : VDA ")
    print("{:.2f} : {:.2f} : {:.2f} : {:.2f} ".format(linear_time / linear_time,compression_linear_time / linear_time,slow_time / linear_time,VDF_time / linear_time))


if __name__ == "__main__":
    main()
