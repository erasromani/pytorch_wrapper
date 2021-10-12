def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError(f'{s} is not a valid boolean string')
    return s == 'True'