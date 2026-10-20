"""Bridge CSF checkpoint scales to the original external TensorFold loader.

No TensorFold implementation is vendored. Read-ahead is disabled to bound
temporary residency; the existing packer and layout remain unchanged.
"""
import ctypes


def reader_class(base, catalog, library, progress, guard):
    import torch
    from tensorfold.cuda.direct_read import Reader
    lib = ctypes.CDLL(str(library))
    lib.csf_decode.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t,
                              ctypes.c_size_t, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t]
    lib.csf_decode.restype = ctypes.c_int

    class CSFReader(base):
        instances = []

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.host = Reader()
            self.consumed = set()
            self.decoded = 0
            self.python_scale_checks = 0
            self.layer_count = 0
            self.instances.append(self)

        def has(self, name):
            return super().has(name) or (name.endswith('.weight_scale')
                                        and name + '.nvfp4_csf_fixed' in self.where
                                        and name + '.nvfp4_csf_exceptions' in self.where)

        def queue(self, names):
            guard()

        def get(self, name):
            if name in self.where:
                self.consumed.add(name)
                return super().get(name)
            if not name.endswith('.weight_scale') or not self.has(name):
                raise KeyError(name)
            weight = catalog.entries[name.removesuffix('_scale')]
            if weight['dtype'] != 'U8' or len(weight['shape']) != 2:
                raise ValueError('invalid CSF weight')
            n, k2 = weight['shape']
            blobs = []
            for suffix, dtype in [('.nvfp4_csf_fixed', 'U8'), ('.nvfp4_csf_exceptions', 'U32')]:
                key = name + suffix
                e = catalog.entries[key]
                if e['dtype'] != dtype:
                    raise ValueError('invalid CSF component dtype')
                blobs.append(self.host.read(catalog.root / e['shard'], e['offset'], e['bytes']))
                self.consumed.add(key)
            fixed, exc = blobs
            result = torch.empty((n, k2 * 2 // 16), dtype=torch.uint8)
            code = lib.csf_decode(fixed.data_ptr(), fixed.numel(), exc.data_ptr(), exc.numel(),
                                  n, k2 * 2, result.data_ptr(), result.numel())
            if code:
                raise ValueError(f'CSF decoder rejected data ({code})')
            if self.decoded % 1536 in (0, 511, 512, 1024, 1535):
                from e2_extract_expert import decode_python
                expected = decode_python(bytes(fixed.numpy()), bytes(exc.numpy()), n, k2 * 2)
                if bytes(result.numpy()) != expected:
                    raise ValueError('independent CSF scale check failed')
                self.python_scale_checks += 1
            self.decoded += 1
            return result.to(self.device)

        def drop(self, names):
            super().drop(names)
            self.layer_count += 1
            torch.cuda.synchronize()
            progress(self.layer_count, self)

        def close(self):
            self.host.close()
            super().close()

    return CSFReader
