import ctypes


class FilenameHashprintPair(ctypes.Structure):
    _fields_ = [('filename', ctypes.c_char_p),
                ('hashprint', ctypes.POINTER(ctypes.c_uint64)),
                ('hp_size', ctypes.c_int)]


class ParallelCollector:
    def __init__(self):
        self.lib = ctypes.CDLL(
            '/Users/chingachgook/dev/QtProjects/hpfw/cmake-build-debug/modules/python/libpyhpfw.dylib')

        self.lib.par_collector_new.restype = ctypes.c_void_p
        self.lib.par_collector_prepare.restype = ctypes.POINTER(FilenameHashprintPair)
        self.lib.par_collector_prepare.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_char_p), ctypes.c_int]
        self.lib.par_collector_del.argtypes = [ctypes.c_void_p]

        self.collector = self.lib.par_collector_new()

    def prepare(self, filenames) -> FilenameHashprintPair:
        pyarr = [f.encode('utf-8') for f in filenames]
        arr = (ctypes.c_char_p * len(pyarr))(*pyarr)
        return self.lib.par_collector_prepare(self.collector, arr, len(pyarr))

    def __del__(self):
        self.lib.par_collector_del(self.collector)


collector = ParallelCollector()
filenames = ['/Users/chingachgook/dev/QtProjects/hpfw/original/New Order - Regret.mp3', 'kek']
hashprints = collector.prepare(filenames)
print(hashprints.contents.hp_size)
