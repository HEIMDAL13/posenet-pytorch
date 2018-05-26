import time
import os
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer
from util import html
import numpy

opt = TestOptions().parse()
opt.nThreads = 1   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()


results_dir = os.path.join(opt.results_dir, opt.name)
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

testepoch = 495
opt.which_epoch = testepoch
model = create_model(opt)
model.load_network(model.netG, 'G', testepoch)
err = []
for i, data in enumerate(dataset):
    model.set_input(data)
    model.test()
    img_path = model.get_image_paths()[0]
    print('\t%04d/%04d: process image... %s' % (i, len(dataset), img_path), end='\r')
    image_path = img_path.split('/')[-2] + '/' + img_path.split('/')[-1]
    pose = model.get_current_pose()
    err_p, err_o = model.get_current_errors()
    err.append([err_p, err_o])
median_pos = numpy.median(err, axis=0)
print("\tmedian wrt pos.: {0:.2f}m {1:.2f}°".format(median_pos[0], median_pos[1]))
