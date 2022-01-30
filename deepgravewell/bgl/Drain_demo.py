#!/usr/bin/env python
import sys
sys.path.append('../')
import Drain

input_dir  = '../bgl'  # The input directory of log file
output_dir = 'Drain_result/'  # The output directory of parsing results
log_file   = 'bgl2_100k'  # The input log file name
log_format = '<Label> <Timestamp> <Date> <Node> <Time> <NodeRepeat> <Type> <Component> <Level> <Content>'  # HDFS log format
# Regular expression list for optional preprocessing (default: [])
regex      = [r'(?<=[^A-Za-z0-9])(\-?\+?\d+)(?=[^A-Za-z0-9])|[0-9]+$']
st         = 0.5  # Similarity threshold
depth      = 4  # Depth of all leaf nodes

parser = Drain.LogParser(log_format, indir=input_dir, outdir=output_dir,  depth=depth, st=st, rex=regex)
parser.parse(log_file)