import pickle
import copyreg as copy_reg

def dump_file(*dps):
    '''
    dump file.
    dps: [data, path]s.
    '''
    for dp in dps:
        if len(dp) != 2:
            print ("issue:" + str(dp))
            continue
        dfile = open(dp[1], 'wb')
        pickle.dump(dp[0], dfile)
        dfile.close()
    print ("dump file done.")


def load_file(*ps):
    '''
    load file.
    ps: [path,...]s
    '''
    ret = []
    for p_s in ps:
        with open(p_s, 'rb') as dfile:
            print(dfile)
            ret.append(pickle.load(dfile))
    return ret
