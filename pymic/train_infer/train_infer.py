# -*- coding: utf-8 -*-
from __future__ import print_function, division

import os 
import sys
import time
import scipy
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.image as mp
import matplotlib
matplotlib.use("Agg")
na = os.path.realpath(__file__)
pna = os.path.dirname(na)
ppna = os.path.dirname(pna)
pppna = os.path.dirname(ppna)
sys.path.append(pppna)

from scipy import special
from datetime import datetime
from tensorboardX import SummaryWriter
from pymic.io.image_read_write import save_nd_array_as_image
from pymic.io.nifty_dataset import NiftyDataset
from pymic.io.transform3d import get_transform
from pymic.train_infer.net_factory import get_network
from pymic.train_infer.infer_func import volume_infer
from pymic.train_infer.loss import *
from pymic.train_infer.get_optimizer import get_optimiser
from pymic.util.image_process import convert_label
from pymic.util.parse_config import parse_config


class TrainInferAgent():
    def __init__(self, config, stage = 'train'):
        self.config = config
        self.stage  = stage
        assert(stage in ['train', 'inference', 'test'])

    def __create_dataset(self):
        root_dir  = self.config['dataset']['root_dir']
        train_csv = self.config['dataset'].get('train_csv', None)
        valid_csv = self.config['dataset'].get('valid_csv', None)
        test_csv  = self.config['dataset'].get('test_csv', None)
        modal_num = self.config['dataset']['modal_num']
        if(self.stage == 'train'):
            transform_names = self.config['dataset']['train_transform']
        else:
            transform_names = self.config['dataset']['test_transform']
            
        self.transform_list = [get_transform(name, self.config['dataset']) \
            for name in transform_names if name != 'RegionSwop']
            
        if('RegionSwop' in transform_names):
            self.region_swop = get_transform('RegionSwop', self.config['dataset']) 
        else:
            self.region_swop = None
        if(self.stage == 'train'):
            train_dataset = NiftyDataset(root_dir=root_dir,
                                csv_file  = train_csv,
                                modal_num = modal_num,
                                with_label= True,
                                transform = transforms.Compose(self.transform_list))
            valid_dataset = NiftyDataset(root_dir=root_dir,
                                csv_file  = valid_csv,
                                modal_num = modal_num,
                                with_label= True,
                                transform = transforms.Compose(self.transform_list))
            batch_size = self.config['training']['batch_size']
            self.train_loader = torch.utils.data.DataLoader(train_dataset, 
                batch_size = batch_size, shuffle=True, num_workers=4)
            self.valid_loader = torch.utils.data.DataLoader(valid_dataset, 
                batch_size = batch_size, shuffle=False, num_workers=4)
        else:
            test_dataset = NiftyDataset(root_dir=root_dir,
                                csv_file  = test_csv,
                                modal_num = modal_num,
                                with_label= False,
                                transform = transforms.Compose(self.transform_list))
            batch_size = 1
            self.test_loder = torch.utils.data.DataLoader(test_dataset, 
                batch_size=batch_size, shuffle=False, num_workers=4)

    def __create_network(self):
        self.net = get_network(self.config['network'])
        self.net.double()

    def __create_optimizer(self):
        self.optimizer = get_optimiser(self.config['training']['optimizer'],
                self.net.parameters(), 
                self.config['training'])
        last_iter = -1
        if(self.checkpoint is not None):
            self.optimizer.load_state_dict(self.checkpoint['optimizer_state_dict'])
            last_iter = self.checkpoint['iteration'] - 1
        self.schedule = optim.lr_scheduler.MultiStepLR(self.optimizer,
                self.config['training']['lr_milestones'],
                self.config['training']['lr_gamma'],
                last_epoch = last_iter)

    def __train(self):
        device = torch.device(self.config['training']['device_name'])
        self.net.to(device)

        summ_writer = SummaryWriter(self.config['training']['summary_dir'])
        chpt_prefx  = self.config['training']['checkpoint_prefix']
        loss_func   = self.config['training']['loss_function']
        iter_start  = self.config['training']['iter_start']
        iter_max    = self.config['training']['iter_max']
        iter_valid  = self.config['training']['iter_valid']
        iter_save   = self.config['training']['iter_save']
        class_num   = self.config['network']['class_num']

        if(iter_start > 0):
            checkpoint_file = "{0:}_{1:}.pt".format(chpt_prefx, iter_start)
            self.checkpoint = torch.load(checkpoint_file)
            assert(self.checkpoint['iteration'] == iter_start)
            self.net.load_state_dict(self.checkpoint['model_state_dict'])
        else:
            self.checkpoint = None
        self.__create_optimizer()

        train_loss      = 0
        train_dice_list = []
        loss_obj = SegmentationLossCalculator(loss_func, True)
        trainIter = iter(self.train_loader)
        print("{0:} training start".format(str(datetime.now())[:-7]))
        for it in range(iter_start, iter_max):
            try:
                data = next(trainIter)
            except StopIteration:
                trainIter = iter(self.train_loader)
                data = next(trainIter)
            if(self.region_swop is not None):
                data = self.region_swop(data)
            # get the inputs
            dataPath, inputs, labels_prob = data['pathName'], data['image'].double(), data['label_prob'].double()

            inputs, labels_prob = inputs.to(device), labels_prob.to(device)
            if it%20 ==0:
                Ic1 = np.array(inputs.cpu()[0,0])
                Ic1 = np.asarray(Ic1,np.uint8)
                Lc1 = np.array(labels_prob.cpu()[0,0])*255.
                Lc1 = np.asarray(Lc1, np.uint8)
                Lc2 = np.array(labels_prob.cpu()[0,1])*255.
                Lc2 = np.asarray(Lc2, np.uint8)

                # mp.imsave('Ic1.png',Ic1,cmap='gray')
                # mp.imsave('Lc1.png', Lc1, cmap='gray')
                # mp.imsave('Lc2.png', Lc2, cmap='gray')
                plt.figure(1)
                plt.subplot(311)
                plt.imshow(Ic1,cmap='gray')
                plt.subplot(312)
                plt.imshow(Lc1, cmap='gray')
                plt.subplot(313)
                plt.imshow(Lc2, cmap='gray')
                plt.show()
                plt.close()
                plt.figure(2)
                print(f'path:{dataPath[0]}')
                pathOfMask = dataPath[0]
                pathOfImg = pathOfMask.replace('mask','image')
                OrigMask = np.load(pathOfMask)
                OrigImg = np.load(pathOfImg)
                print(pathOfImg)
                plt.subplot(211)
                plt.imshow(OrigImg, cmap='gray')
                plt.subplot(212)
                plt.imshow(OrigMask, cmap='gray')
                plt.show()
                plt.close()

                summ_writer.add_image('input Image',Ic1,it+1,dataformats='HW')
                summ_writer.add_image('lableImg1', Lc1, it+1,dataformats='HW')
                summ_writer.add_image('lableImg2', Lc2,it+1, dataformats='HW')
            # zero the parameter gradients
            self.optimizer.zero_grad()
            self.schedule.step()
                
            # forward + backward + optimize
            outputs = self.net(inputs)
            
            loss_input_dict = {'prediction':outputs, 'ground_truth':labels_prob}
            if ('label_distance' in data):
                label_distance = data['label_distance'].double()
                loss_input_dict['label_distance'] = label_distance.to(device)
            loss   = loss_obj.get_loss(loss_input_dict)
            # if (self.config['training']['use'])
            loss.backward()
            self.optimizer.step()

            # get dice evaluation for each class
            if(isinstance(outputs, tuple) or isinstance(outputs, list)):
                outputs = outputs[0] 
            outputs_argmax = torch.argmax(outputs, dim = 1, keepdim = True)
            soft_out       = get_soft_label(outputs_argmax, class_num)
            soft_out, labels_prob = reshape_prediction_and_ground_truth(soft_out, labels_prob) 
            dice_list = get_classwise_dice(soft_out, labels_prob)
            train_dice_list.append(dice_list.cpu().numpy())

            # evaluate performance on validation set
            train_loss = train_loss + loss.item()
            if (it % iter_valid == iter_valid - 1):
                train_avg_loss = train_loss / iter_valid
                train_cls_dice = np.asarray(train_dice_list).mean(axis = 0)
                train_avg_dice = train_cls_dice.mean()
                train_loss = 0.0
                train_dice_list = []

                valid_loss = 0.0
                valid_dice_list = []
                with torch.no_grad():
                    for data in self.valid_loader:
                        inputs, labels_prob = data['image'].double(), data['label_prob'].double()
                        inputs, labels_prob = inputs.to(device), labels_prob.to(device)
                        outputs = self.net(inputs)
                        loss_input_dict = {'prediction':outputs, 'ground_truth':labels_prob}
                        if ('label_distance' in data):
                            label_distance = data['label_distance'].double()
                            loss_input_dict['label_distance'] = label_distance.to(device)
                        loss   = loss_obj.get_loss(loss_input_dict)
                        valid_loss = valid_loss + loss.item()

                        if(isinstance(outputs, tuple) or isinstance(outputs, list)):
                            outputs = outputs[0] 
                        outputs_argmax = torch.argmax(outputs, dim = 1, keepdim = True)
                        soft_out  = get_soft_label(outputs_argmax, class_num)
                        soft_out, labels_prob = reshape_prediction_and_ground_truth(soft_out, labels_prob) 
                        dice_list = get_classwise_dice(soft_out, labels_prob)
                        valid_dice_list.append(dice_list.cpu().numpy())

                valid_avg_loss = valid_loss / len(self.valid_loader)
                valid_cls_dice = np.asarray(valid_dice_list).mean(axis = 0)
                valid_avg_dice = valid_cls_dice.mean()
                loss_scalers = {'train': train_avg_loss, 'valid': valid_avg_loss}
                summ_writer.add_scalars('loss', loss_scalers, it + 1)
                dice_scalers = {'train': train_avg_dice, 'valid': valid_avg_dice}
                summ_writer.add_scalars('class_avg_dice', dice_scalers, it + 1)
                print('train cls dice', train_cls_dice.shape, train_cls_dice)
                print('valid cls dice', valid_cls_dice.shape, valid_cls_dice)
                for c in range(class_num):
                    dice_scalars = {'train':train_cls_dice[c], 'valid':valid_cls_dice[c]}
                    summ_writer.add_scalars('class_{0:}_dice'.format(c), dice_scalars, it + 1)
                
                print("{0:} it {1:}, loss {2:.4f}, {3:.4f}".format(
                    str(datetime.now())[:-7], it + 1, train_avg_loss, valid_avg_loss))
            if (it % iter_save ==  iter_save - 1):
                save_dict = {'iteration': it + 1,
                             'model_state_dict': self.net.state_dict(),
                             'optimizer_state_dict': self.optimizer.state_dict()}
                save_name = "{0:}_{1:}.pt".format(chpt_prefx, it + 1)
                torch.save(save_dict, save_name)    
        summ_writer.close()
    
    def __infer(self):
        device = torch.device(self.config['testing']['device_name'])
        self.net.to(device)
        # laod network parameters and set the network as evaluation mode
        self.checkpoint = torch.load(self.config['testing']['checkpoint_name'], map_location = device)
        self.net.load_state_dict(self.checkpoint['model_state_dict'])
        
        if(self.config['testing']['evaluation_mode'] == True):
            self.net.eval()
            if(self.config['testing']['test_time_dropout'] == True):
                def test_time_dropout(m):
                    if(type(m) == nn.Dropout):
                        print('dropout layer')
                        m.train()
                self.net.apply(test_time_dropout)
        output_dir       = self.config['testing']['output_dir']
        save_probability = self.config['testing']['save_probability']
        label_source = self.config['testing']['label_source']
        label_target = self.config['testing']['label_target']
        class_num    = self.config['network']['class_num']
        mini_batch_size     = self.config['testing']['mini_batch_size']
        mini_patch_inshape  = self.config['testing']['mini_patch_shape']
        mini_patch_stride   = self.config['testing']['mini_patch_stride']
        filename_replace_source = self.config['testing']['filename_replace_source']
        filename_replace_target = self.config['testing']['filename_replace_target']
        mini_patch_outshape = None
        # automatically infer outupt shape
        if(mini_patch_inshape is not None):
            patch_inshape = [1, self.config['dataset']['modal_num']] + mini_patch_inshape
            testx = np.random.random(patch_inshape)
            testx = torch.from_numpy(testx)
            testx = torch.tensor(testx)
            testx = testx.to(device)
            testy = self.net(testx)
            if(isinstance(testy, tuple) or isinstance(testy, list)):
                testy = testy[0] 
            testy = testy.detach().cpu().numpy()
            mini_patch_outshape = testy.shape[2:]
            print('mini patch in shape', mini_patch_inshape)
            print('mini patch out shape', mini_patch_outshape)
        start_time = time.time()
        with torch.no_grad():
            iOfli=0
            for data in self.test_loder:
                images = data['image'].double()
                names  = data['names']

                data['predict'] = volume_infer(images, self.net, device, class_num, 
                    mini_batch_size, mini_patch_inshape, mini_patch_outshape, mini_patch_stride)

                for i in reversed(range(len(self.transform_list))):
                    if (self.transform_list[i].inverse):
                        data = self.transform_list[i].inverse_transform_for_prediction(data) 
                output = np.argmax(data['predict'][0], axis = 0)
                output = np.asarray(output, np.uint8)

                # if((label_source is not None) and (label_target is not None)):
                    # output = convert_label(output, label_source, label_target)

                # save the output and (optionally) probability predictions
                root_dir  = self.config['dataset']['root_dir']
                save_name = names[0].split('/')[-1]
                names = names[0].split('/')[-4:]
                save_name_ = ''
                for name in names:
                    save_name_ += '/'
                    save_name_ += name
                print(f"save_name_----------->{save_name_}")
                save_name = save_name_
                # save_name= '_'.join(names)
                if((filename_replace_source is  not None) and (filename_replace_target is not None)):
                    save_name = save_name.replace(filename_replace_source, filename_replace_target)

                save_name = "{0:}/{1:}".format(output_dir, save_name)
                iOfli+=1
                pname = os.path.dirname(save_name)
                if not os.path.exists(pname):
                    os.makedirs(pname)
                print(f"save_name---------->{save_name}")
                save_nd_array_as_image(output, save_name, root_dir + '/' + names[0])
                if(save_probability):
                    save_name_split = save_name.split('.')
                    if('.nii.gz' in save_name):
                        save_prefix = '.'.join(save_name_split[:-2])
                        save_format = 'nii.gz'
                    else:
                        save_prefix = '.'.join(save_name_split[:-1])
                        save_format = save_name_split[-1]
                    prob = scipy.special.softmax(data['predict'][0],axis = 0)
                    class_num = prob.shape[0]
                    for c in range(0, class_num):
                        temp_prob = prob[c]
                        prob_save_name = "{0:}_prob_{1:}.{2:}".format(save_prefix, c, save_format)
                        if(len(temp_prob.shape) == 2):
                            temp_prob = np.asarray(temp_prob * 255, np.uint8)
                        save_nd_array_as_image(temp_prob, prob_save_name, root_dir + '/' + names[0])

        avg_time = (time.time() - start_time) / len(self.test_loder)
        print("average testing time {0:}".format(avg_time))

    def run(self):
        self.__create_dataset()
        self.__create_network()
        if(self.stage == 'train'):
            self.__train()
        elif (self.stage == 'test'):
            self.__infer()
        elif (self.stage == 'IOInfer'):
            self.__infer()
        else :
            raise("Don't BB")

if __name__ == "__main__":
    # if(len(sys.argv) < 3):
    #     print('Number of arguments should be 3. e.g.')
    #     print('    python train_infer.py train config.cfg')
    #     exit()
    # stage    = str(sys.argv[1])
    # cfg_file = str(sys.argv[2])
    stage    = "train"
    cfg_file = '/data1/wanglonglong/01workspace/PyMIC/examples/lymph/config/train_test.cfg'
    # cfg_file = "/data1/wanglonglong/01workspace/PyMIC/examples/prostate/config/train_test.cfg"
    config   = parse_config(cfg_file)
    # print(config)
    agent    = TrainInferAgent(config, stage)
    agent.run()

    fig, ax = plt.subplots(3, 2, figsize=(20, 20))
