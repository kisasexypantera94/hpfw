import ctypes

import numpy as np


class FilenameHashprintPair(ctypes.Structure):
    _fields_ = [('filename', ctypes.c_char_p),
                ('hashprint', ctypes.POINTER(ctypes.c_uint64)),
                ('hp_size', ctypes.c_int)]


class ParallelCollector:
    def __init__(self):
        self.lib = ctypes.CDLL(
            '/Users/chingachgook/dev/QtProjects/hpfw/cmake-build-debug/modules/python/libpyhpfw.dylib')

        # ParallelCollector()
        self.lib.par_collector_new.restype = ctypes.c_void_p

        # prepare()
        self.lib.par_collector_prepare.restype = ctypes.POINTER(FilenameHashprintPair)
        self.lib.par_collector_prepare.argtypes = [ctypes.c_void_p,
                                                   ctypes.POINTER(ctypes.c_char_p),
                                                   ctypes.c_int,
                                                   ctypes.POINTER(ctypes.c_int)]
        # calc_hashprint()
        self.lib.par_collector_calc_hashprint.restype = ctypes.POINTER(ctypes.c_uint64)
        self.lib.par_collector_calc_hashprint.argtypes = [ctypes.c_void_p,
                                                          ctypes.c_char_p,
                                                          ctypes.POINTER(ctypes.c_int)]

        # ~ParallelCollector()
        self.lib.par_collector_del.argtypes = [ctypes.c_void_p]

        self.lib.prepare_result_free.argtypes = [ctypes.c_void_p, ctypes.c_int]
        self.lib.calc_hashprint_result_free.argtypes = [ctypes.POINTER(ctypes.c_uint64)]

        self.collector = self.lib.par_collector_new()

    def prepare(self, filenames):
        pyarr = [f.encode('utf-8') for f in filenames]
        arr = (ctypes.c_char_p * len(pyarr))(*pyarr)
        got = ctypes.c_int(0)
        hps = self.lib.par_collector_prepare(self.collector, arr, len(pyarr), ctypes.byref(got))
        hashprints = [
            (np.array([hps[h].hashprint[i] for i in range(hps[h].hp_size)], dtype=np.uint64).copy(), hps[h].filename)
            for h in range(got.value)
        ]

        self.lib.prepare_result_free(hps, got)

        return hashprints

    def calc_hashprint(self, filename):
        size = ctypes.c_int(0)
        hp = self.lib.par_collector_calc_hashprint(self.collector,
                                                   ctypes.c_char_p(filename.encode('utf-8')),
                                                   ctypes.byref(size))
        hashprint = np.array([hp[i] for i in range(size.value)], dtype=np.uint64).copy()

        self.lib.calc_hashprint_result_free(hp)

        return hashprint

    def __del__(self):
        self.lib.par_collector_del(self.collector)
