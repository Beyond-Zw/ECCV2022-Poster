import numpy as np
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torch.nn.init as init
import torch.utils.data as data
import torch.utils.data.dataset as dataset
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.autograd import Variable
import torchvision.utils as v_utils
import matplotlib.pyplot as plt
import cv2
import math
from collections import OrderedDict
import copy
import time
from model.utils import DataLoader
from model.base_model_A_14 import *    # ped
from model.base_model_A_13 import *    # avenue
from model.base_model_A_18 import *    # shangahi
# from model.Reconstruction import *
from sklearn.metrics import roc_auc_score
from utils import *
import random
import glob
import pickle
from os import path
from PIL import Image

import argparse


parser = argparse.ArgumentParser(description="MNAD")
parser.add_argument('--gpus', nargs='+', type=str, default='0', help='gpus')
parser.add_argument('--batch_size', type=int, default=4, help='batch size for training')
parser.add_argument('--test_batch_size', type=int, default=1, help='batch size for test')
parser.add_argument('--epochs', type=int, default=250, help='number of epochs for training')
parser.add_argument('--epochs_som_cluster', type=int, default=20, help='number of epochs for training')
parser.add_argument('--h', type=int, default=256, help='height of input images')
parser.add_argument('--w', type=int, default=256, help='width of input images')
parser.add_argument('--c', type=int, default=3, help='channel of input images')
parser.add_argument('--method', type=str, default='pred', help='The target task for anoamly detection')
parser.add_argument('--loss_gd', type=float, default=1.00, help='weight of the gradient loss')
parser.add_argument('--loss_compact', type=float, default=0.01, help='weight of the feature compactness loss')
parser.add_argument('--loss_separate', type=float, default=0.01, help='weight of the feature separateness loss')
parser.add_argument('--lr_D', type=float, default=1e-4, help='initial learning rate for parameters')
parser.add_argument('--lr_step_size', type=int, default=20, help='learning rate step size for parameters')
parser.add_argument('--lr_gamma', type=float, default=0.8, help='learning rate gamma for parameters')
parser.add_argument('--t_length', type=int, default=5, help='length of the frame sequences')
parser.add_argument('--fdim', type=int, default=512, help='channel dimension of the features')
parser.add_argument('--mdim', type=int, default=512, help='channel dimension of the memory items')
parser.add_argument('--msize', type=int, default=10, help='number of the memory items')
parser.add_argument('--alpha', type=float, default=0.6, help='weight for the anomality score')
parser.add_argument('--th', type=float, default=0.01, help='threshold for test updating')
parser.add_argument('--num_workers', type=int, default=8, help='number of workers for the train loader')
parser.add_argument('--num_workers_test', type=int, default=8, help='number of workers for the test loader')
parser.add_argument('--dataset_type', type=str, default='ped2', help='type of dataset: ped2, avenue, shanghai')
parser.add_argument('--dataset_path', type=str, default='./data', help='directory of data')
parser.add_argument('--exp_dir', type=str, default='log', help='directory of log')
parser.add_argument('--resume', type=str, default='', help='file path of resume pth')
parser.add_argument('--log_name', type=str, default='log_A_16', help='directory of log')
parser.add_argument('--model_dir', type=str, help='directory of model')
parser.add_argument('--m_items_dir', type=str, help='directory of model')
parser.add_argument('--debug', type=bool, default=False, help='if debug')
parser.add_argument('--som_cluster', type=bool, default=False, help='if som_cluster')
parser.add_argument('--som_side', type=int, default=5, help='side length of SOM')
parser.add_argument('--som_sigma', type=float, default=0.1, help='sigma of SOM')
parser.add_argument('--som_lr', type=float, default=0.5, help='learning rate of SOM')
parser.add_argument('--som_log_name', type=str, default='log_1', help='directory of som log')
parser.add_argument('--som_neighborhood_function', type=str, default='gaussian', help='som_neighborhood_function of som')

args = parser.parse_args()


class SaveConvFeatures():

    def __init__(self, m):  # module to hook
        self.hook = m.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.features = output.data

    def remove(self):
        self.hook.remove()


def seed_torch(seed=2022):
    seed = int(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True

seed_torch(2022)

def _init_fn():
    np.random.seed(2022)

def auc_cal(epoch_num):
    # log_dir = os.path.join('./exp', args.dataset_type, args.exp_dir)
    # if not os.path.exists(log_dir):
    #     os.makedirs(log_dir)
    #
    # if not args.debug:
    #     orig_stdout_ = sys.stdout
    #     f_ = open(os.path.join(log_dir, 'log_auc.txt'), 'w')
    #     sys.stdout = f_


    # args.model_dir = f'exp/ped2/{args.exp_dir}/model_'+str(epoch_num)+'.pth'
    args.model_dir = f'exp/ped2/log_A_16/model_' + str(epoch_num) + '.pth'
    # args.model_dir = f'exp/avenue/model_' + str(epoch_num) + '.pth'
    # args.model_dir = f'exp/shanghai/log_A_16/model_' + str(epoch_num) + '.pth'
    seed_torch(2022)

    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    if args.gpus is None:
        gpus = "0"
        os.environ["CUDA_VISIBLE_DEVICES"]= gpus
    else:
        gpus = ""
        for i in range(len(args.gpus)):
            gpus = gpus + args.gpus[i] + ","
        os.environ["CUDA_VISIBLE_DEVICES"]= gpus[:-1]


    test_folder = args.dataset_path+"/"+args.dataset_type+"/testing/frames"

    # Loading dataset
    test_dataset = DataLoader(test_folder, transforms.Compose([
                 transforms.ToTensor(),
                 ]), resize_height=args.h, resize_width=args.w, time_step=args.t_length-1)

    test_size = len(test_dataset)

    test_batch = data.DataLoader(test_dataset, batch_size = args.test_batch_size,
                                 shuffle=False, num_workers=args.num_workers_test, drop_last=False, worker_init_fn=_init_fn())

    loss_func_mse = nn.MSELoss(reduction='none')

    # Loading the trained model
    model = torch.load(args.model_dir)['state_dict']
    model.cuda()
    # m_items = torch.load(args.m_items_dir)
    labels = np.load('./data/frame_labels_'+args.dataset_type+'.npy')
    vis_model = model
    # hook_ref = SaveConvFeatures(vis_model.decoder)


    videos = OrderedDict()
    videos_list = sorted(glob.glob(os.path.join(test_folder, '*')))
    for video in videos_list:
        video_name = video.split('/')[-1]
        videos[video_name] = {}
        videos[video_name]['path'] = video
        videos[video_name]['frame'] = glob.glob(os.path.join(video, '*.jpg'))
        videos[video_name]['frame'].sort()
        videos[video_name]['length'] = len(videos[video_name]['frame'])

    labels_list = []
    label_length = 0
    psnr_list = {}
    feature_distance_list = {}

    # print('Evaluation of', args.dataset_type)

    # Setting for video anomaly detection
    for video in sorted(videos_list):
        video_name = video.split('/')[-1]
        if args.dataset_type =='shanghai':
            if args.method == 'pred':
                labels_list = np.append(labels_list, labels[4+label_length:videos[video_name]['length']+label_length])
            else:
                labels_list = np.append(labels_list, labels[label_length:videos[video_name]['length']+label_length])
        else:
            if args.method == 'pred':
                labels_list = np.append(labels_list,
                                        labels[0][4 + label_length:videos[video_name]['length'] + label_length])
            else:
                labels_list = np.append(labels_list,
                                        labels[0][label_length:videos[video_name]['length'] + label_length])
        label_length += videos[video_name]['length']
        psnr_list[video_name] = []
        feature_distance_list[video_name] = []

    label_length = 0
    video_num = 0
    label_length += videos[videos_list[video_num].split('/')[-1]]['length']
    # m_items_test = m_items.clone()

    model.eval()
    fps = 0
    # torch.cuda.synchronize()
    with torch.no_grad():
        for k,(imgs) in enumerate(test_batch):

            if args.method == 'pred':
                if k == label_length-4*(video_num+1):
                    video_num += 1
                    label_length += videos[videos_list[video_num].split('/')[-1]]['length']
            else:
                if k == label_length:
                    video_num += 1
                    label_length += videos[videos_list[video_num].split('/')[-1]]['length']

            imgs = Variable(imgs).cuda()

            if args.method == 'pred':
                # if k==1411:
                #     # outputs, feas, updated_feas, m_items_test, softmax_score_query, softmax_score_memory, _, _, _, compactness_loss = model.forward(imgs[:,0:3*4], m_items_test, False)
                #     outputs, _, _, m_items, softmax_score_query, softmax_score_memory, separateness_loss, compactness_loss = model.forward(
                #         imgs[:, 0:3 * 4], None, True, False)  # Todo:将Ture改为False
                #     # outputs = model.forward(imgs[:,0:3*4], None, True)
                #     mse_imgs = torch.mean(loss_func_mse((outputs[0] + 1) / 2, (imgs[0, 3 * 4:] + 1) / 2)).item()
                #     print(k, mse_imgs)
                #     if mse_imgs == "nan":
                #         print(mse_imgs)
                    # mse_feas = compactness_loss.item()

                    # Calculating the threshold for updating at the test time
                    # point_sc = point_score(outputs, imgs[:, 3 * 4:])
                # outputs, feas, updated_feas, m_items_test, softmax_score_query, softmax_score_memory, _, _, _, compactness_loss = model.forward(imgs[:,0:3*4], m_items_test, False)
                #*********************ped2******************************
                torch.cuda.synchronize()
                temp = time.time()
                #***************train***************
                # outputs, _, _, m_items, softmax_score_query, softmax_score_memory, separateness_loss, compactness_loss = model.forward(args, None, None,  imgs[:,0:3*4], True)# Todo:将Ture改为False
                # ******************test*********************
                hook_ref = SaveConvFeatures(vis_model.decoder)
                outputs = model.forward(args, None, None, imgs[:, 0:3 * 4], False)  # Todo:将Ture改为False

                # *********************avenue******************************
                # outputs, _, _, m_items, softmax_score_query, softmax_score_memory, separateness_loss, compactness_loss = model.forward(
                #     args, None,  imgs[:, 0:3 * 4], True)  # Todo:将Ture改为False

                end = time.time()
                # if k > 1:  # Compute fps by calculating the time used in one completed iteration, this is more accurate.
                fps = 1 / (end - temp)

                print(f'\rDetecting: {fps:.2f} fps.', end='')



                ## =======可视化特征映射图========#
                #
                # img1 = outputs
                # img2 = np.array(img1.cpu())
                # img3 = np.squeeze(img2)
                # b = img3.shape
                # img = ((img3 + 1) * 127.5).transpose(1, 2, 0).astype('uint8')
                #
                # conv_features = hook_ref.features  # [1,2048,7,7]
                # print('特征图输出维度：', conv_features.shape)  # 其实得到特征图之后可以自己编写绘图程序
                # hook_ref.remove()
                #
                # heat = conv_features.squeeze(0)
                # heat_mean = torch.mean(heat, dim=0)
                # heatmap = heat_mean.cpu().numpy()
                # heatmap /= np.max(heatmap)
                # heatmap = cv2.resize(heatmap, (img.shape[0], img.shape[1]))
                # heatmap = np.uint8(255*heatmap)
                # heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                # # plt.imshow(heatmap)
                # # plt.show()
                # cv2.imshow('heatmap',heatmap)
                # cv2.waitKey(1)
                # if not path.exists(f'results/{args.dataset_type}/features map'):
                #     os.mkdir(f'results/{args.dataset_type}/features map')
                # cv2.imwrite(f'results/{args.dataset_type}/features map/{k+4}_fea.jpg', heatmap)

                # superimg = heatmap*0.4 + np.array(img[:,:,::-1])
                # cv2.imwrite('./superimg.jpg',superimg)
                #
                # img_ = np.array(Image.open('./superimg.jpg')) #.convert('RGB'))
                # # plt.imshow(img_)
                # # plt.show()
                # cv2.imshow('img_', img_)
                # cv2.waitKey(1)


                # =======保存预测图片========= #
                # img1 = outputs
                # img2 = np.array(img1.cpu())
                # img3 = np.squeeze(img2)
                # b = img3.shape
                # # img = np.transpose(img3)
                # img = ((img3 + 1) * 127.5).transpose(1, 2, 0).astype('uint8')
                # # img = cv2.resize(img, (640, 360)).astype('float32')
                # # plt.imshow(img)
                # # plt.savefig(str(i)+'.png')
                # cv2.imshow("predict",img)
                # cv2.waitKey(1)  # show video
                # # name = folder.split('/')[-1]
                # if not path.exists(f'results/{args.dataset_type}/predict_frame'):
                #     os.mkdir(f'results/{args.dataset_type}/predict_frame')
                # cv2.imwrite(f'results/{args.dataset_type}/predict_frame/{k+4}_pre.jpg', img)

                ## =======保存真实图片========= #
                # img1_true = imgs[:,3*4:15]
                # img2_true = np.array(img1_true.cpu())
                # img3_true = np.squeeze(img2_true)
                # b = img3_true.shape
                # # img = np.transpose(img3)
                # img_true = ((img3_true + 1) * 127.5).transpose(1, 2, 0).astype('uint8')
                # # # img = cv2.resize(img, (640, 360)).astype('float32')
                # # # plt.imshow(img)
                # # # plt.savefig(str(i)+'.png')
                # cv2.imshow("img true",img_true)
                # cv2.waitKey(1)  # show video
                # # name = folder.split('/')[-1]
                # if not path.exists(f'results/{args.dataset_type}/true_frame'):
                #     os.mkdir(f'results/{args.dataset_type}/true_frame')
                # cv2.imwrite(f'results/{args.dataset_type}/true_frame/{k + 4}_gt.jpg', img_true)

                ## =======保存错误映射图========= #
                # G_frame = np.squeeze(imgs[:,3*4:15])
                # target_frame = np.squeeze(outputs)
                # b = target_frame.shape
                # # diff_map = torch.sum(torch.abs(G_frame - target_frame).squeeze(), 0)
                # diff_map = torch.sum(torch.abs(G_frame - target_frame), 0)
                # # diff_map = torch.sum(G_frame, 0)
                # diff_map -= diff_map.min()  # Normalize to 0 ~ 255.
                # diff_map /= diff_map.max()
                # diff_map *= 255
                # diff_map = diff_map.cpu().detach().numpy().astype('uint8')
                # # heat_map = cv2.applyColorMap(diff_map, cv2.COLORMAP_JET)
                # heat_map = cv2.applyColorMap(diff_map, cv2.COLORMAP_JET)
                # # heat_map = cv2.applyColorMap(diff_map, cv2.COLORMAP_HOT)
                #
                # cv2.imshow('difference map', heat_map)
                # cv2.imwrite(f'results/{args.dataset_type}/difference map/{k + 4}_dp.jpg',  heat_map)
                # cv2.waitKey(1)

                mse_imgs = torch.mean(loss_func_mse((outputs[0]+1)/2, (imgs[0,3*4:]+1)/2)).item()
                # print(k, mse_imgs)
                # mse_feas = compactness_loss.item()

                # Calculating the threshold for updating at the test time
                # point_sc = point_score(outputs, imgs[:,3*4:])

            else:
                # outputs, feas, updated_feas, m_items_test, softmax_score_query, softmax_score_memory, compactness_loss = model.forward(imgs, m_items_test, False)
                outputs = model.forward(imgs, m_items, False)
                mse_imgs = torch.mean(loss_func_mse((outputs[0]+1)/2, (imgs[0]+1)/2)).item()
                mse_feas = compactness_loss.item()

                # Calculating the threshold for updating at the test time
                # point_sc = point_score(outputs, imgs)

            # if  point_sc < args.th:
            #     query = F.normalize(feas, dim=1)
            #     query = query.permute(0,2,3,1) # b X h X w X d
            #     m_items_test = model.memory.update(query, m_items_test, False)
            psnr_list[videos_list[video_num].split('/')[-1]].append(psnr(mse_imgs))
            # feature_distance_list[videos_list[video_num].split('/')[-1]].append(mse_feas)
            # torch.cuda.synchronize()
            # end = time.time()
            # if k > 1:  # Compute fps by calculating the time used in one completed iteration, this is more accurate.
            #     fps = 1 / (end - temp)
            # temp = end
            # print(f'\rDetecting: [ {fps:.2f} fps.', end='')

    # result_dict = {'psnr': psnr_list}
    # pickle_path = f'./PSNR/{epoch_num}.pkl'
    # with open(pickle_path, 'wb') as writer:
    #     pickle.dump(result_dict, writer, pickle.HIGHEST_PROTOCOL)
    # with open(pickle_path, 'rb') as reader:
    #     results = pickle.load(reader)
    # psnr_list = results['psnr']

    # Measuring the abnormality score and the AUC
    anomaly_score_total_list = []
    for video in sorted(videos_list):
        video_name = video.split('/')[-1]
        # anomaly_score_total_list += score_sum(anomaly_score_list(psnr_list[video_name]),
        #                                  anomaly_score_list_inv(feature_distance_list[video_name]), args.alpha)
        anomaly_score_total_list += anomaly_score_list(psnr_list[video_name])

    anomaly_score_total_list = np.asarray(anomaly_score_total_list)

    accuracy = AUC(anomaly_score_total_list, np.expand_dims(1-labels_list, 0))

    # print(args.model_dir)
    # print('The result of ', args.dataset_type)
    print(f'epoch{epoch_num}-AUC: ', round(accuracy*100,3), '%')
    # if not args.debug:
    #     sys.stdout = orig_stdout_
    #     f_.close()
    auc = round(accuracy*100,3)
    if auc >= 80.0:
        result_dict = {'psnr': psnr_list}
        pickle_path = f'./PSNR/model_{epoch_num}_{args.log_name}_{auc}_.pkl'
        with open(pickle_path, 'wb') as writer:
            pickle.dump(result_dict, writer, pickle.HIGHEST_PROTOCOL)

    return round(accuracy*100,3)



if __name__ == "__main__":
    # auc_cal('66_log_A_16_53_96.434_')
    auc_cal(0)
    # auc_cal('13_log_A_13_11_88.578')
    # auc_cal('0_log_A_18_21_72.413_')
