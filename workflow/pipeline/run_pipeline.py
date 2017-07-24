import sys
sys.path.append(
    '/export/home/amatskev/nature_methods_multicut_pipeline/software/')
from time import time

start_time=time()

print '####################################################################'
print 'Initializing ...'
print '####################################################################'
execfile('init_datasets.py')

time_after_init=time()
print "Initialization took ", (time_after_init-start_time)/60 ," minutes"

print '####################################################################'
print 'Running multicut ...'
print '####################################################################'
execfile('run_mc_all.py')

time_after_mc=time()
print "Multicut took ", (time_after_mc-time_after_init)/3600 ," hours"

print '####################################################################'
print 'Detecting merges ...'
print '####################################################################'
execfile('detect_merges_all.py')

time_after_dm=time()
print "Detecting merges took ", (time_after_dm-time_after_mc)/3600 ," hours"

print '####################################################################'
print 'Resolving segmentation ...'
print '####################################################################'
execfile('resolve_all.py')

time_after_rs=time()
print "Initialization took ", (time_after_init-start_time)/60 ," minutes"
print "Multicut took ", (time_after_mc-time_after_init)/3600 ," hours"
print "Detecting merges took ", (time_after_dm-time_after_mc)/3600 ," hours"
print "Resolving segmentation took ", (time_after_rs-time_after_dm)/3600 ," hours"

