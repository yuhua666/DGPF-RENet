# -*- coding: utf-8 -*-

import os
from functions import loop_train_test
from data import image_size_dict as dims
from data import draw_false_color, draw_gt, draw_bar

# remove abundant output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

## global constants
verbose = 1 # whether or not print redundant info (1 if and only if in debug mode, 0 in run mode)
run_times = 3 # random run times, recommend at least 10
output_map = True # whether or not output classification map
only_draw_label = True # whether or not only predict labeled samples
disjoint = False # whether or not train and test on spatially disjoint samples

lr = 1e-3 # init learing rate
decay = 1e-3 # exponential learning rate decay
ws = 19 # window size
epochs = 64 # epoch
batch_size = 32   # batch size
model_type = 'demo'  # model type

def pavia_university_experiment():
    hp = {
        'pc': dims['1'][2],
        'w': ws,
        'decay': decay,
        'bs': batch_size,
        'lr': lr,
        'epochs': epochs,
        'disjoint': disjoint,
        'model_type': model_type,
    }
    num_list = [7, 17, 2, 3, 1, 5, 1, 4, 1]
    loop_train_test(dataID=1, num_list=num_list, verbose=verbose, run_times=run_times,
                    hyper_parameters=hp, output_map=output_map, only_draw_label=only_draw_label, model_save=True)

def indian_pine_experiment():
    hp = {
        'pc': dims['2'][2],
        'w': ws,
        'decay': decay,
        'bs': batch_size,
        'lr': lr,
        'epochs': epochs,
        'disjoint': disjoint,
        'model_type': model_type
    }
    #num_list = [2, 71, 41, 11, 24, 36, 1, 23, 1, 48, 122, 29, 10, 63, 19, 4] # 5%
    #num_list = [10*7, 143*7, 83*7, 24*7, 48*7, 73*7, 3*7, 48*7, 2*7, 97*7, 246*7, 59*7, 21*7, 127*7, 39*7, 9*7]  # 70%
    num_list = [45, 1414, 822, 235, 478, 723 ,27, 473, 19, 964, 2430, 587, 203, 1252, 382, 91] #99%
    # num_list = [2, 71, 41, 11, 24, 36, 2, 23, 2, 48, 122, 29, 10, 63, 19, 4] # 5%
    # num_list = [40, 1420, 820, 230, 480, 720, 20, 470, 10, 970, 2400, 581, 200, 1250, 370, 85]  # 99%%
    # num_list = [4, 142, 82, 23, 48, 72, 2, 47, 1, 97, 24, 58, 20, 125, 37, 8]  # temp
    #num_list = [13, 473, 270, 73, 160, 240, 7, 153, 3, 323, 800, 194, 67, 416, 123, 28]  # 30%
    # num_list = [10, 355, 205, 57, 120, 180, 5, 117, 2, 282, 600, 140, 50, 312, 92, 21]  # 25%%
    # num_list = [8, 284, 164, 46, 98, 140, 4, 81, 2, 19, 480, 136, 40, 250, 74, 17]    #20%
    # num_list = [5, 177, 102, 29, 60, 90, 2, 58, 1, 121, 300, 72, 25, 156, 46, 10]
    # num_list = [4, 142, 82, 23, 48, 72, 2, 47, 1, 97, 24, 58, 20, 125, 37, 8]
    # num_list = [3, 95, 55, 15, 32, 48, 1, 31, 1, 64, 163, 39, 13, 84, 25, 6]  #划分15份

    # num_list = [30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30]  # 0.1%训练样本

    #num_list = [5, 143, 83, 24, 48, 73, 3, 48, 2, 97, 246, 59, 21, 127, 39, 9]  # 10%
    loop_train_test(dataID=2, num_list=num_list, verbose=verbose, run_times=run_times,
                        hyper_parameters=hp, output_map=output_map, only_draw_label=only_draw_label, model_save=True)

def Salinas_experiment():
    hp = {
        'pc': dims['3'][2],
        'w': ws,
        'decay': decay,
        'bs': batch_size,
        'lr': lr,
        'epochs': epochs,
        'disjoint': disjoint,
        'model_type': model_type,
    }
    num_list = [40, 100, 13, 17, 8, 25, 9, 20, 8]
    loop_train_test(dataID=4, num_list=num_list, verbose=verbose, run_times=run_times,
                    hyper_parameters=hp, output_map=output_map, only_draw_label=only_draw_label, model_save=False)


def WHU_Hi_HanChuan_experiment():
    hp = {
        'pc': dims['5'][2],
        'w': ws,
        'decay': decay,
        'bs': batch_size,
        'lr': lr,
        'epochs': epochs,
        'disjoint': disjoint,
        'model_type': model_type,
    }


    # num_list =   [13, 13, 7, 12, 12, 3, 13, 12, 13, 12, 12, 12, 5, 4, 7, 8] #0.1%训练样本
    # num_list = [10, 19, 10, 7, 13, 20, 18, 56, 31, 16, 5, 10, 5, 5, 36, 9] #0.5%训练样本
    num_list = [100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100]  # 100训练样本
    # num_list = [200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200] #0.1%训练样本
    #num_list = [50] * 15  #每类固定使用50个训练样本
    loop_train_test(dataID=5, num_list=num_list, verbose=verbose, run_times=run_times,
                        hyper_parameters=hp, output_map=output_map, only_draw_label=only_draw_label, model_save=True)


def WHU_Hi_HongHu_experiment():
    hp = {
        'pc': dims['6'][2],
        'w': ws,
        'decay': decay,
        'bs': batch_size,
        'lr': lr,
        'epochs': epochs,
        'disjoint': disjoint,
        'model_type': model_type,
    }


    # num_list =   [13, 13, 7, 12, 12, 3, 13, 12, 13, 12, 12, 12, 5, 4, 7, 8] #0.1%训练样本
    #num_list = [2, 71, 41, 11, 24, 36, 2, 23, 2, 48, 122, 29, 10, 63, 19, 4] # 5%
    #num_list = [10, 19, 10, 7, 13, 20, 18, 56, 31, 16, 5, 10, 5, 5, 36, 9] #0.5%训练样本
    #num_list = [20, 37, 19, 13, 26, 39, 35, 112, 62, 32, 10, 19, 9, 10, 72, 18]  # 1%训练样本
    # num_list = [200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200] #0.1%训练样本
    #num_list = [50] * 15  #每类固定使用50个训练样本
    #num_list = [25] * 22  #每类固定使用25个训练样本
    num_list = [50] * 22  #每类固定使用50个训练样本
    loop_train_test(dataID=6, num_list=num_list, verbose=verbose, run_times=run_times,
                        hyper_parameters=hp, output_map=output_map, only_draw_label=only_draw_label, model_save=True)


def WHU_Hi_LongKow_experiment():
    hp = {
        'pc': dims['7'][2],
        'w': ws,
        'decay': decay,
        'bs': batch_size,
        'lr': lr,
        'epochs': epochs,
        'disjoint': disjoint,
        'model_type': model_type,
    }


    # num_list =   [13, 13, 7, 12, 12, 3, 13, 12, 13, 12, 12, 12, 5, 4, 7, 8] #0.1%训练样本
    # num_list = [10, 19, 10, 7, 13, 20, 18, 56, 31, 16, 5, 10, 5, 5, 36, 9] #0.5%训练样本
    #num_list = [20, 37, 19, 13, 26, 39, 35, 112, 62, 32, 10, 19, 9, 10, 72, 18]  # 1%训练样本
    # num_list = [200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200] #0.1%训练样本
    num_list = [25] * 9  #每类固定使用25个训练样本
    loop_train_test(dataID=7, num_list=num_list, verbose=verbose, run_times=run_times,
                        hyper_parameters=hp, output_map=output_map, only_draw_label=only_draw_label, model_save=True)


def houston_university_experiment():
    hp = {
        'pc': dims['3'][2],
        'w': ws,
        'decay': decay,
        'bs': batch_size,
        'lr': lr,
        'epochs': epochs,
        'disjoint': disjoint,
        'model_type': model_type,
    }
     # num_list =   [13, 13, 7, 12, 12, 3, 13, 12, 13, 12, 12, 12, 5, 4, 7, 8] #0.1%训练样本
    #num_list = [2, 71, 41, 11, 24, 36, 2, 23, 2, 48, 122, 29, 10, 63, 19, 4] # 5%
    #num_list = [10, 19, 10, 7, 13, 20, 18, 56, 31, 16, 5, 10, 5, 5, 36, 9] #0.5%训练样本
    #num_list = [20, 37, 19, 13, 26, 39, 35, 112, 62, 32, 10, 19, 9, 10, 72, 18]  # 1%训练样本
    # num_list = [200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200] #0.1%训练样本
    #num_list = [25] * 20  #每类固定使用50个训练样本
    #num_list = [392, 1300, 27, 543, 202, 181, 11, 1590, 8948, 1833, 1360, 61, 1854, 394, 277, 459, 6, 263, 215, 273] # %1
    #num_list = [392*5, 1300*5, 27*5, 543*5, 202*5, 181*5, 11*5, 1590*5, 8948*5, 1833*5, 1360*5, 61*5, 1854*5, 394*5, 277*5, 459*5, 6*5, 263*5, 215*5, 273*5] # %5
    #num_list = [392*10, 1300*10, 27*10, 543*10, 202*10, 181*10, 11*10, 1590*10, 8948*10, 1833*10, 1360*10, 61*10, 1854*10, 394*10, 277*10, 459*10, 6*10, 263*10, 215*10, 273*10] # %10
    #num_list = [392*20, 1300*20, 27*20, 543*20, 202*20, 181*20, 11*20, 1590*20, 8948*20, 1833*20, 1360*20, 61*20, 1854*20, 394*20, 277*20, 459*20, 6*20, 263*20, 215*20, 273*20] # %20
    #num_list = [392*40, 1300*40, 27*40, 543*40, 202*40, 181*40, 11*40, 1590*40, 8948*40, 1833*40, 1360*40, 61*40, 1854*40, 394*40, 277*40, 459*40, 6*40, 263*40, 215*40, 273*40] # %40
    #num_list = [392*50, 1300*50, 27*50, 543*50, 202*50, 181*50, 11*50, 1590*50, 8948*50, 1833*50, 1360*50, 61*50, 1854*50, 394*50, 277*50, 459*50, 6*50, 263*50, 215*50, 273*50] # %50
    #num_list = [392*60, 1300*60, 27*60, 543*60, 202*60, 181*60, 11*60, 1590*60, 8948*60, 1833*60, 1360*60, 61*60, 1854*60, 394*60, 277*60, 459*60, 6*60, 263*60, 215*60, 273*60] # %60
    #num_list = [392*70, 1300*70, 27*70, 543*70, 202*70, 181*70, 11*70, 1590*70, 8948*70, 1833*70, 1360*70, 61*70, 1854*70, 394*70, 277*70, 459*70, 6*70, 263*70, 215*70, 273*70] # %70
    #num_list = [392*80, 1300*80, 27*80, 543*80, 202*80, 181*80, 11*80, 1590*80, 8948*80, 1833*80, 1360*80, 61*80, 1854*80, 394*80, 277*80, 459*80, 6*80, 263*80, 215*80, 273*80] # %80
    #num_list = [392*99, 1300*99, 27*99, 543*99, 202*99, 181*99, 11*99, 1590*99, 8948*99, 1833*99, 1360*99, 61*99, 1854*99, 394*99, 277*99, 459*99, 6*99, 263*99, 215*99, 273*99] # %99
    num_list = [78, 260, 260, 108, 40, 36, 2, 318, 1790, 367, 272, 12, 370, 79, 55, 92, 1, 53, 43, 55] # %0.2
    loop_train_test(dataID=3, num_list=num_list, verbose=verbose, run_times=run_times,
                        hyper_parameters=hp, output_map=output_map, only_draw_label=only_draw_label, model_save=True)



#实验
#pavia_university_experiment()
#indian_pine_experiment()
houston_university_experiment()
#Salinas_experiment()
#WHU_Hi_HanChuan_experiment()
#WHU_Hi_HongHu_experiment()
#WHU_Hi_LongKow_experiment()

#draw_false_color(dataID=1)
#draw_false_color(dataID=2)
#draw_false_color(dataID=3)
#draw_false_color(dataID=4)
#draw_false_color(dataID=5)
#draw_false_color(dataID=6)
#draw_false_color(dataID=7)

#
# draw_bar(dataID=1)
# draw_bar(dataID=2)
#draw_bar(dataID=3)

# draw_gt(dataID=1, fixed=disjoint)
# draw_gt(dataID=2, fixed=disjoint)
# draw_gt(dataID=3, fixed=disjoint)

