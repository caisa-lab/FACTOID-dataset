import argparse

def monthToIdx(date):
    ''' Function to compute the index of the date in a given list using month granularity. Assumes that the range of months is from 2020-2021
    '''
    add = 0
    if date.year < 2020:
        return 0
    
    if date.year == 2021 and date.month > 4:
        return -1
    
    if date.year == 2021:
        add = 12

    idx = date.month - 1 + add
    return idx

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def convert_value(value):
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value
    if isinstance(value, list):
        return [convert_value(val) for val in value]
    else:
        return str(value)


def read_userids(filename, root):
    ids = None
    with open(os.path.join(root, filename), 'r') as f: 
        ids = [str(line.strip()) for line in f]
    return ids