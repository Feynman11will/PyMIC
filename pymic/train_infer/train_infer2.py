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

sys.path.append('../../')

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
from IPython import display

class TrainInferAgent():
    def __init__(self, config, stage='train'):
        self.config = config
        self.stage = stage
        assert (stage in ['train', 'inference', 'test'])

    def __create_dataset(self):
        root_dir = self.config['dataset']['root_dir']
        train_csv = self.config['dataset'].get('train_csv', None)
        valid_csv = self.config['dataset'].get('valid_csv', None)
        test_csv = self.config['dataset'].get('test_csv', None)
        modal_num = self.config['dataset']['modal_num']
        if (self.stage == 'train'):
            transform_names = self.config['dataset']['train_transform']
        else:
            transform_names = self.config['dataset']['test_transform']

        self.transform_list = [get_transform(name, self.config['dataset']) \
                               for name in transform_names if name != 'RegionSwop']

        if ('RegionSwop' in transform_names):
            self.region_swop = get_transform('RegionSwop', self.config['dataset'])
        else:
            self.region_swop = None
        if (self.stage == 'train'):
            train_dataset = NiftyDataset(root_dir=root_dir,
                                         csv_file=train_csv,
                                         modal_num=modal_num,
                                         with_label=True,
                                         transform=transforms.Compose(self.transform_list))
            valid_dataset = NiftyDataset(root_dir=root_dir,
                                         csv_file=valid_csv,
                                         modal_num=modal_num,
                                         with_label=True,
                                         transform=transforms.Compose(self.transform_list))
            batch_size = self.config['training']['batch_size']
            self.train_loader = torch.utils.data.DataLoader(train_dataset,
                                                            batch_size=batch_size, shuffle=True, num_workers=4)
            self.valid_loader = torch.utils.data.DataLoader(valid_dataset,
                                                            batch_size=batch_size, shuffle=False, num_workers=4)
        else:
            test_dataset = NiftyDataset(root_dir=root_dir,
                                        csv_file=test_csv,
                                        modal_num=modal_num,
                                        with_label=True,
                                        transform=transforms.Compose(self.transform_list))
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
        if (self.checkpoint is not None):
            self.optimizer.load_state_dict(self.checkpoint['optimizer_state_dict'])
            last_iter = self.checkpoint['iteration'] - 1
        self.schedule = optim.lr_scheduler.MultiStepLR(self.optimizer,
                                                       self.config['training']['lr_milestones'],
                                                       self.config['training']['lr_gamma'],
                                                       last_epoch=last_iter)

    def __train(self):
        device = torch.device(self.config['training']['device_name'])
        self.net.to(device)

        summ_writer = SummaryWriter(self.config['training']['summary_dir'])
        chpt_prefx = self.config['training']['checkpoint_prefix']
        loss_func = self.config['training']['loss_function']
        iter_start = self.config['training']['iter_start']
        iter_max = self.config['training']['iter_max']
        iter_valid = self.config['training']['iter_valid']
        iter_save = self.config['training']['iter_save']
        class_num = self.config['network']['class_num']

        if (iter_start > 0):
            checkpoint_file = "{0:}_{1:}.pt".format(chpt_prefx, iter_start)
            self.checkpoint = torch.load(checkpoint_file)
            assert (self.checkpoint['iteration'] == iter_start)
            self.net.load_state_dict(self.checkpoint['model_state_dict'])
        else:
            self.checkpoint = None
        self.__create_optimizer()

        train_loss = 0
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
            if (self.region_swop is not None):
                data = self.region_swop(data)
            # get the inputs
            dataPath, inputs, labels_prob = data['pathName'], data['image'].double(), data['label_prob'].double()

            inputs, labels_prob = inputs.to(device), labels_prob.to(device)

            # zero the parameter gradients
            self.optimizer.zero_grad()
            self.schedule.step()

            # forward + backward + optimize
            outputs = self.net(inputs)

            loss_input_dict = {'prediction': outputs, 'ground_truth': labels_prob}
            if ('label_distance' in data):
                label_distance = data['label_distance'].double()
                loss_input_dict['label_distance'] = label_distance.to(device)
            loss = loss_obj.get_loss(loss_input_dict)
            # if (self.config['training']['use'])
            loss.backward()
            self.optimizer.step()

            # get dice evaluation for each class
            if (isinstance(outputs, tuple) or isinstance(outputs, list)):
                outputs = outputs[0]
            outputs_argmax = torch.argmax(outputs, dim=1, keepdim=True)
            wocao = np.array(outputs_argmax.cpu()[0, 0, :, :])* 255.

            if (it % 20 == 0):
                Ic1 = np.array(inputs.cpu()[0, 0])
                Ic1fanwei = Ic1.max() - Ic1.min()
                Ic1 = (Ic1 - Ic1.min())/Ic1fanwei*255.*0.3
                Lc1 = np.array(labels_prob.cpu()[0, 0]) * 255.
                Lc1 = np.asarray(Lc1, np.uint8)
                Lc2 = np.array(labels_prob.cpu()[0, 1]) * 255.
                Lc2 = np.asarray(Lc2, np.uint8)


                fig, ax = plt.subplots(2, 3, figsize=(20, 20))
                # cmap = 'gray'
                ax[0, 0].imshow(Ic1)
                ax[0, 1].imshow(Lc1+Ic1)
                ax[0, 2].imshow(Lc2+Ic1)

                pathOfMask = dataPath[0]
                pathOfImg = pathOfMask.replace('mask', 'image')
                OrigMask = np.load(pathOfMask)

                OrigImg = np.load(pathOfImg)
                OrigImgfanwei = OrigImg.max() - OrigImg.min()
                OrigImg = (OrigImg - OrigImg.min()) / OrigImgfanwei * 255. * 0.3
                OrigMask = OrigMask * 255.
                ax[1, 0].imshow(OrigImg)
                ax[1, 1].imshow(OrigMask+OrigImg)
                ax[1, 2].imshow(wocao+Ic1)
                plt.show()
                display.clear_output(wait=True)
            soft_out = get_soft_label(outputs_argmax, class_num)
            soft_out, labels_prob = reshape_prediction_and_ground_truth(soft_out, labels_prob)
            dice_list = get_classwise_dice(soft_out, labels_prob)
            train_dice_list.append(dice_list.cpu().numpy())

            # evaluate performance on validation set
            train_loss = train_loss + loss.item()
            if (it % iter_valid == iter_valid - 1):
                train_avg_loss = train_loss / iter_valid
                train_cls_dice = np.asarray(train_dice_list).mean(axis=0)
                train_avg_dice = train_cls_dice.mean()
                train_loss = 0.0
                train_dice_list = []

                valid_loss = 0.0
                valid_dice_list = []
                with torch.no_grad():
                    lsi = 0
                    for data in self.valid_loader:
                        inputs, labels_prob = data['image'].double(), data['label_prob'].double()
                        inputs, labels_prob = inputs.to(device), labels_prob.to(device)
                        outputs = self.net(inputs)
                        loss_input_dict = {'prediction': outputs, 'ground_truth': labels_prob}
                        if ('label_distance' in data):
                            label_distance = data['label_distance'].double()
                            loss_input_dict['label_distance'] = label_distance.to(device)
                        loss = loss_obj.get_loss(loss_input_dict)
                        valid_loss = valid_loss + loss.item()

                        if (isinstance(outputs, tuple) or isinstance(outputs, list)):
                            outputs = outputs[0]
                        outputs_argmax = torch.argmax(outputs, dim=1, keepdim=True)
                        soft_out = get_soft_label(outputs_argmax, class_num)

                        soft_out, labels_prob = reshape_prediction_and_ground_truth(soft_out, labels_prob)
                        dice_list = get_classwise_dice(soft_out, labels_prob)
                        valid_dice_list.append(dice_list.cpu().numpy())

                valid_avg_loss = valid_loss / len(self.valid_loader)
                valid_cls_dice = np.asarray(valid_dice_list).mean(axis=0)
                valid_avg_dice = valid_cls_dice.mean()
                loss_scalers = {'train': train_avg_loss, 'valid': valid_avg_loss}
                summ_writer.add_scalars('loss', loss_scalers, it + 1)
                dice_scalers = {'train': train_avg_dice, 'valid': valid_avg_dice}
                summ_writer.add_scalars('class_avg_dice', dice_scalers, it + 1)
                print('train cls dice', train_cls_dice.shape, train_cls_dice)
                print('valid cls dice', valid_cls_dice.shape, valid_cls_dice)
                for c in range(class_num):
                    dice_scalars = {'train': train_cls_dice[c], 'valid': valid_cls_dice[c]}
                    summ_writer.add_scalars('class_{0:}_dice'.format(c), dice_scalars, it + 1)

                print("{0:} it {1:}, loss {2:.4f}, {3:.4f}".format(
                    str(datetime.now())[:-7], it + 1, train_avg_loss, valid_avg_loss))
            if (it % iter_save == iter_save - 1):
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
        self.checkpoint = torch.load(self.config['testing']['checkpoint_name'], map_location=device)
        self.net.load_state_dict(self.checkpoint['model_state_dict'])

        if (self.config['testing']['evaluation_mode'] == True):
            self.net.eval()
            if (self.config['testing']['test_time_dropout'] == True):
                def test_time_dropout(m):
                    if (type(m) == nn.Dropout):
                        print('dropout layer')
                        m.train()

                self.net.apply(test_time_dropout)
        output_dir = self.config['testing']['output_dir']
        save_probability = self.config['testing']['save_probability']
        label_source = self.config['testing']['label_source']
        label_target = self.config['testing']['label_target']
        class_num = self.config['network']['class_num']
        mini_batch_size = self.config['testing']['mini_batch_size']
        mini_patch_inshape = self.config['testing']['mini_patch_shape']
        mini_patch_stride = self.config['testing']['mini_patch_stride']
        filename_replace_source = self.config['testing']['filename_replace_source']
        filename_replace_target = self.config['testing']['filename_replace_target']
        mini_patch_outshape = None
        # automatically infer outupt shape

        if (mini_patch_inshape is not None):
            patch_inshape = [1, self.config['dataset']['modal_num']] + mini_patch_inshape
            testx = np.random.random(patch_inshape)
            testx = torch.from_numpy(testx)
            testx = torch.tensor(testx)
            testx = testx.to(device)
            testy = self.net(testx)
            if (isinstance(testy, tuple) or isinstance(testy, list)):
                testy = testy[0]
            testy = testy.detach().cpu().numpy()
            mini_patch_outshape = testy.shape[2:]
            print('mini patch in shape', mini_patch_inshape)
            print('mini patch out shape', mini_patch_outshape)
        start_time = time.time()
        with torch.no_grad():
            iOfli = 0
            showImg = 10
            fig, ax = plt.subplots(showImg, 4, figsize=(300, 300))
            for data in self.test_loder:
                labels_prob = data['label_prob'].double()
                images ,labels_prob = data['image'].double(),labels_prob.to(device)
                names = data['names']

                data['predict'] = volume_infer(images, self.net, device, class_num,
                                               mini_batch_size, mini_patch_inshape, mini_patch_outshape,
                                               mini_patch_stride)
                output = np.argmax(data['predict'][0], axis=0)
                print(f"output.shape---->:{output.shape}")
                if iOfli <showImg:
                    inputImg = np.array(images.cpu()[0, 0])*255
                    ax[iOfli, 0].imshow(inputImg)
                    inputImg = inputImg*0.3
                    maskImg1 = np.array(labels_prob.cpu()[0, 0])

                    outImg1 = output*255.
                    # outImg = np.asarray(outImg1, np.uint8)
                    outImg = outImg1 + inputImg

                    ax[iOfli, 1].imshow(maskImg1,cmap='gray')
                    ax[iOfli, 2].imshow(outImg,cmap='gray')
                    ax[iOfli, 3].imshow(outImg1, cmap='gray')
                    # display.clear_output(wait=True)

                if iOfli==showImg-1:

                    display.clear_output(wait=True)
                    plt.show()
                    break
                iOfli += 1

        avg_time = (time.time() - start_time) / len(self.test_loder)
        print("average testing time {0:}".format(avg_time))

    def run(self):
        self.__create_dataset()
        self.__create_network()
        if (self.stage == 'train'):
            self.__train()
        elif (self.stage == 'test'):
            self.__infer()
        elif (self.stage == 'IOInfer'):
            self.__infer()
        else:
            raise ("Don't BB")


