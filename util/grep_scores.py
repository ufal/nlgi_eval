#!/usr/bin/env python3

import json
from argparse import ArgumentParser

def rgb_code(r, g, b):
    # Get the bash 256 colors number given RGB (with values in the range 0-6)
    return "\033[38;5;%dm" % ((16 + (36 * r) + (6 * g) + b))

def rg_gradient(lo, hi, val, swap=False):
    # Return red-green gradient rgb code
    r = int(6 - ((val - lo) / (hi - lo) * 6))
    g = int(0 + ((val - lo) / (hi - lo) * 6))
    r = 5 if  r > 5  else r
    r = 0 if  r < 0  else r
    g = 5 if  g > 5  else g
    g = 0 if  g < 0  else g
    if swap:
        g, r = r, g
    return rgb_code(r, g, 0)


def process_file(fname, acc_lo, acc_hi, corr_lo, corr_hi):
    output = ''
    with open(fname, 'r', encoding='UTF-8') as fh:
        data = json.load(fh)
    # WebNLG
    if 'metrics @ 2.5' in data:
        acc = data['metrics @ 2.5']['accuracy']
        output += 'A2.5:' + rg_gradient(acc_lo, acc_hi, acc) + "%.4f\033[0m " % acc
    if 'metrics @ 2.0' in data:
        acc = data['metrics @ 2.0']['accuracy']
        output += 'A2.0:' + rg_gradient(acc_lo, acc_hi, acc) + "%.4f\033[0m " % acc
    if 'OK_correlation' in data:
        ok_corr = data['OK_correlation']['rho']
        output += 'OKr:' + rg_gradient(corr_lo, corr_hi, ok_corr) + "%.4f\033[0m " % ok_corr

    # E2E
    if 'metrics_fine' in data:
        acc = data['metrics_fine']['accuracy']
        output += 'Af:' + rg_gradient(acc_lo, acc_hi, acc) + "%.4f\033[0m " % acc
    if 'metrics_rough' in data:
        acc = data['metrics_rough']['accuracy']
        output += 'Ar:' + rg_gradient(acc_lo, acc_hi, acc) + "%.4f\033[0m " % acc

    return output



if __name__ == '__main__':
    ap = ArgumentParser()
    ap.add_argument('--acc-range', '--acc', '-a', type=str, help='Color range for accuracy (comma-separated pair of values)', default='0.5,1.0')
    ap.add_argument('--corr-range', '--corr', '-c', type=str, help='Color range for correlation (comma-separated pair of values)', default='0.5,1.0')
    ap.add_argument('input_files', nargs='+', help='JSON files to check for results')

    args = ap.parse_args()

    acc_lo, acc_hi = [float(x) for x in args.acc_range.split(',')]
    corr_lo, corr_hi = [float(x) for x in args.corr_range.split(',')]
    output = ''
    for fname in args.input_files:
        output += process_file(fname, acc_lo, acc_hi, corr_lo, corr_hi)
    print(output, end='')
