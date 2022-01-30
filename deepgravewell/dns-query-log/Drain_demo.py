#!/usr/bin/env python
import sys
sys.path.append('../')
import Drain

input_dir  = '../dns-query-log'  # The input directory of log file
output_dir = 'result/'  # The output directory of parsing results
log_file   = 'messages'  # The input log file name
log_format = '<Date>-<Month>-<Year> <Time> <Type>: <Level>: <Name> @<Component> <IP>#<Port> (\(<Domain>\))?: <St>: <Content>'  # DNS log format
# Regular expression list for optional preprocessing (default: [])
regex      = [r'(\d+\.){3}\d+', r'\d{2}:\d{2}:\d{2}']
st         = 0.6  # Similarity threshold
depth      = 4  # Depth of all leaf nodes

parser = Drain.LogParser(log_format, indir=input_dir, outdir=output_dir,  depth=depth, st=st, rex=regex)
parser.parse(log_file)