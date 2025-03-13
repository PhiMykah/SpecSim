import sys

# ------------------------ Printing to Standard Error ------------------------ #

def errPrint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)