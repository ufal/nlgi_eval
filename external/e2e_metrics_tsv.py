#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import csv

# CSV headers
HEADER_SRC = r'(mr|src|source|meaning(?:[_ .-]rep(?:resentation)?)?|da|dial(?:ogue)?[_ .-]act)s?'
HEADER_SYS = r'(out(?:put)?|ref(?:erence)?|sys(?:tem)?(?:[_ .-](?:out(?:put)?|ref(?:erence)?))?)s?'
HEADER_REF = r'(trg|tgt|target|ref(?:erence)?|human(?:[_ .-](?:ref(?:erence)?))?)s?'


def read_lines(file_name, multi_ref=False):
    """Read one instance per line from a text file. In multi-ref mode, assumes multiple lines
    (references) per instance & instances separated by empty lines."""
    buf = [[]] if multi_ref else []
    with open(file_name, 'r', encoding='UTF-8') as fh:
        for line in fh:
            line = line.strip()
            if multi_ref:
                if not line:
                    buf.append([])
                else:
                    buf[-1].append(line)
            else:
                buf.append(line)
    if multi_ref and not buf[-1]:
        del buf[-1]
    return buf


def read_tsv(tsv_file, header_src=HEADER_SRC, header_ref=HEADER_SYS):
    """Read a TSV file, check basic integrity."""
    tsv_data = read_lines(tsv_file)
    tsv_data[0] = re.sub(u'\ufeff', '', tsv_data[0])  # remove unicode BOM
    tsv_data = [line.replace(u'Ł', u'£') for line in tsv_data]  # fix Ł
    tsv_data = [line.replace(u'Â£', u'£') for line in tsv_data]  # fix Â£
    tsv_data = [line.replace(u'Ã©', u'é') for line in tsv_data]  # fix Ã©
    tsv_data = [line.replace(u'ã©', u'é') for line in tsv_data]  # fix ã©
    tsv_data = [line for line in tsv_data if line]  # ignore empty lines
    reader = csv.reader(tsv_data, delimiter=("\t" if "\t" in tsv_data[0] else ","))  # parse CSV/TSV
    tsv_data = [row for row in reader]  # convert back to list

    # check which columns are which (if headers are present)
    src_match_cols = [idx for idx, field in enumerate(tsv_data[0]) if re.match(header_src, field, re.I)]
    ref_match_cols = [idx for idx, field in enumerate(tsv_data[0]) if re.match(header_ref, field, re.I)]

    # we need to find exactly 1 column of each desired type, or exactly 0 of each
    if not ((len(src_match_cols) == len(ref_match_cols) == 0) or (len(src_match_cols) == len(ref_match_cols) == 1)):
        raise ValueError(("Strange column arrangement in %s: columns [%s] match src pattern `%s`, "
                          + "columns [%s] match ref pattern `%s`")
                         % (tsv_file, ','.join(src_match_cols), header_src,
                            ','.join(ref_match_cols), header_ref))

    num_cols = len(tsv_data[0])  # this should be the number of columns in the whole file
    # if we didn't find any headers, the number of columns must be 2
    if src_match_cols == ref_match_cols == 0:
        src_col = 0
        ref_col = 1
        if num_cols != 2:
            raise ValueError("File %s can't have no header and more than 2 columns" % tsv_file)

    # if we did find headers, just strip them and remember which columns to extract
    else:
        src_col = src_match_cols[0]
        ref_col = ref_match_cols[0]
        tsv_data = tsv_data[1:]

    # check the correct number of columns throughout the file
    errs = [line_no for line_no, item in enumerate(tsv_data, start=1) if len(item) != num_cols]
    if errs:
        print("%s -- weird number of columns" % tsv_file)
        raise ValueError('%s -- Weird number of columns on lines: %s' % (tsv_file, str(errs)))

    # extract the data
    srcs = []
    refs = []
    for row in tsv_data:
        srcs.append(row[src_col])
        refs.append(row[ref_col])
    return srcs, refs
