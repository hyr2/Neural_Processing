import numpy as np
import struct


def writemda(X, fname, dtype="int16"):
    '''Python implementation of .mda file write function
    About MDA files: http://magland.github.io//articles/mda-format/
    Currently only supporsts <int16> format export
    '''
    
    # confirm number of dimensions of array
    # TODO handle case of, eg, 10x10x0
    num_dims = len(X.shape)

    f = open(fname, "wb")
    
    if dtype=="int16":
        # first write a 4-byte word (dtype code)
        code = -4
        f.write(struct.pack('<i', code))
        f.write(struct.pack('<i', 2))
        f.write(struct.pack('<i', num_dims))
        dimprod = 1
        for dd in range(num_dims):
            f.write(struct.pack('<i', X.shape[dd]))
            dimprod *= X.shape[dd]
        Y = X.flatten(order='F')
        Y = Y.astype(np.int16)
        f.write(Y.tobytes())

def writemda16i(X, fname):
    writemda(X, fname, dtype='int16')


########### generate testbench to compare against matlab ground truth write_mda
if __name__ == '__main__':
    x = np.zeros((15,20))
    for i in range(300):
        x[i//20, i%20] = i
    print(x)
    writemda16i(x, "./test_data.mda")