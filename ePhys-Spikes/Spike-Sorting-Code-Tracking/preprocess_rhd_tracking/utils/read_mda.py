import numpy as np
import struct

def readmda(fname):
    f = open(fname, "rb")
    try:
        code=struct.unpack('i', f.read(4))[0]
    except:
        raise IOError("Problem reading file: %s" % (fname))
    if code > 0:
        num_dims = code
        code = -1
    else:
        f.read(4)
        num_dims = struct.unpack('i', f.read(4))[0]
    
    S = np.zeros(abs(num_dims))
    num_dims_abs = abs(num_dims)
    if num_dims < 0:
        # dim_type is int64
        for j in range(num_dims_abs):
            S[j] = struct.unpack('q', f.read(8))[0]
    else:
        # dim_type is int32
        for j in range(num_dims_abs):
            S[j] = struct.unpack('i', f.read(4))[0]
    S = S.astype(np.int64)
    N = np.prod(S)
    # A = np.zeros(S)
    # print("read_mda dtype code:", code)
    if code==-1:
        # complex float32
        M = np.zeros(N*2)
        mda_buff = f.read(N*8)
        M[:] = np.frombuffer(mda_buff, dtype=np.float32)
        # A.flat[:] = M[0::2] + 1j * M[1::2]
        A = M[0::2] + 1j * M[1::2]
        A = A.reshape(S[::-1]).T
    elif code==-2:
        # uchar
        mda_buff = f.read(N)
        # A.flat[:] = np.frombuffer(mda_buff, dtype=np.uint8)
        A = np.frombuffer(mda_buff, dtype=np.uint8)
        A = A.reshape(S[::-1]).T
    elif code==-3:
        # float32
        mda_buff = f.read(N*4)
        # A.flat[:] = np.frombuffer(mda_buff, dtype=np.float32)
        A = np.frombuffer(mda_buff, dtype=np.float32)
        A = A.reshape(S[::-1]).T
    elif code==-4:
        # int16
        mda_buff = f.read(N*2)
        # A.flat[:] = np.frombuffer(mda_buff, dtype=np.int16)
        A = np.frombuffer(mda_buff, dtype=np.int16)
        A = A.reshape(S[::-1]).T
    elif code==-5:
        # int32
        mda_buff = f.read(N*4)
        # A.flat[:] = np.frombuffer(mda_buff, dtype=np.int32)
        A = np.frombuffer(mda_buff, dtype=np.int32)
        A = A.reshape(S[::-1]).T
    elif code==-6:
        # uint16
        mda_buff = f.read(N*2)
        # A.flat[:] = np.frombuffer(mda_buff, dtype=np.uint16)
        A = np.frombuffer(mda_buff, dtype=np.uint16)
        A = A.reshape(S[::-1]).T
    elif code==-7:
        # double
        mda_buff = f.read(N*8)
        # A.flat[:] = np.frombuffer(mda_buff, dtype=np.float64)
        A = np.frombuffer(mda_buff, dtype=np.float64)
        A = A.reshape(S[::-1]).T
    elif code==-8:
        # uint32
        mda_buff = f.read(N*4)
        # A.flat[:] = np.frombuffer(mda_buff, dtype=np.uint32)
        A = np.frombuffer(mda_buff, dtype=np.uint32)
        A = A.reshape(S[::-1]).T
    else:
        raise IOError("Unsupported data type code: %d", code)

    f.close()
    return A.copy(order="C")


##### test write_mda
if __name__=='__main__':
    a=readmda("./test_data.mda")
    # a_gold=readmda("./test_data_matlab.mda")
    # errcnt=0
    # for i in range(300):
    #     tmp = a[i//20, i%20],
    #     tmp_gold = a_gold[i//20, i%20]
    #     if (tmp!=tmp_gold):
    #         errcnt+=1
    # print(errcnt)