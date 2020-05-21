import ctypes
from typing import List, Tuple

import numpy as np


class FilenameHashprintPair(ctypes.Structure):
    _fields_ = [('filename', ctypes.c_char_p),
                ('hashprint', ctypes.POINTER(ctypes.c_uint64)),
                ('hp_size', ctypes.c_int)]


class ParallelCollector:
    def __init__(self):
        self.__lib = ctypes.CDLL(
            '/Users/chingachgook/dev/QtProjects/hpfw/cmake-build-debug/modules/python/libpyhpfw.dylib')

        # ParallelCollector()
        self.__lib.par_collector_new.restype = ctypes.c_void_p

        # prepare()
        self.__lib.par_collector_prepare.restype = ctypes.POINTER(FilenameHashprintPair)
        self.__lib.par_collector_prepare.argtypes = [ctypes.c_void_p,
                                                     ctypes.POINTER(ctypes.c_char_p),
                                                     ctypes.c_int,
                                                     ctypes.POINTER(ctypes.c_int)]
        # calc_hashprint()
        self.__lib.par_collector_calc_hashprint.restype = ctypes.POINTER(ctypes.c_uint64)
        self.__lib.par_collector_calc_hashprint.argtypes = [ctypes.c_void_p,
                                                            ctypes.c_char_p,
                                                            ctypes.POINTER(ctypes.c_int)]

        # ~ParallelCollector()
        self.__lib.par_collector_del.argtypes = [ctypes.c_void_p]

        self.__lib.prepare_result_free.argtypes = [ctypes.c_void_p, ctypes.c_int]
        self.__lib.calc_hashprint_result_free.argtypes = [ctypes.POINTER(ctypes.c_uint64)]

        # load()
        self.__lib.par_collector_load.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        # save()
        self.__lib.par_collector_save.argtypes = [ctypes.c_void_p, ctypes.c_char_p]

        self.__collector = self.__lib.par_collector_new()

    def prepare(self, filenames: str) -> List[Tuple[str, np.ndarray]]:
        pyarr = [f.encode('utf-8') for f in filenames]
        arr = (ctypes.c_char_p * len(pyarr))(*pyarr)
        got = ctypes.c_int(0)
        hps = self.__lib.par_collector_prepare(self.__collector, arr, len(pyarr), ctypes.byref(got))
        hashprints = [
            (
                np.array([hps[h].hashprint[i] for i in range(hps[h].hp_size)], dtype=np.uint64).copy(),
                hps[h].filename.decode('utf-8')
            )
            for h in range(got.value)
        ]

        self.__lib.prepare_result_free(hps, got)

        return hashprints

    def calc_hashprint(self, filename: str) -> np.ndarray:
        size = ctypes.c_int(0)
        hp = self.__lib.par_collector_calc_hashprint(self.__collector,
                                                     ctypes.c_char_p(filename.encode('utf-8')),
                                                     ctypes.byref(size))
        hashprint = np.array([hp[i] for i in range(size.value)], dtype=np.uint64).copy()

        self.__lib.calc_hashprint_result_free(hp)

        return hashprint

    def load(self, cache: str):
        self.__lib.par_collector_load(self.__collector, ctypes.c_char_p(cache.encode('utf-8')))

    def save(self, cache: str):
        self.__lib.par_collector_save(self.__collector, ctypes.c_char_p(cache.encode('utf-8')))

    def __del__(self):
        self.__lib.par_collector_del(self.__collector)
