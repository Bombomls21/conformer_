import torch
import time
import os
import random
import torch
import torch.nn.functional as F
from utils.Logger import AverageMeter
import torch.nn as nn
import numpy as np

import wandb
from utils.metrics import get_pedestrian_metrics

#
def adjust_learning_rate(optimizer, iter, opt):

    if iter % 5 == 0 and iter < opt.max_iter:
        new_lr = opt.learning_rate*(1 - float(iter) / opt.max_iter) ** opt.power
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr

def train_conformer(epoch, data_loader, model, criterion, optimizer, opt,
                epoch_logger, batch_logger):
    print('train at epoch {}'.format(epoch))

    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses_BCE = AverageMeter()
    losses_cnn = AverageMeter()
    losses_trans = AverageMeter()
    # losses_suppression = AverageMeter()
    total_loss = AverageMeter()

    end_time = time.time()
    for i, (imgs, gt_label, imgname) in enumerate(data_loader):
        data_time.update(time.time() - end_time)

        imgs = imgs.cuda()
        gt_label = gt_label.float().cuda()

        # Tạo variable pytorch để tính gradient
        inputs = torch.autograd.Variable(imgs)
        targets = torch.autograd.Variable(gt_label)

        ##########################################
        ##########################################
        random.seed()
        # Dự đoán, lấy các output
        output_cnn, output_trans = model(inputs)

        # Tính loss BCE riêng cho từng output
        loss_cnn = nn.BCEWithLogitsLoss(size_average=False)(output_cnn, targets)
        loss_cnn = loss_cnn/opt.batch_size

        loss_trans = nn.BCEWithLogitsLoss(size_average=False)(output_trans, targets)
        loss_trans = loss_trans/opt.batch_size



        ###############################   pcb   ##############################################
        # loss_cnn_suppression = nn.BCEWithLogitsLoss(size_average=False)(output_suppresion[0], targets)
        # loss_cnn_suppression = loss_cnn_suppression/opt.batch_size
        # loss_trans_suppression = nn.BCEWithLogitsLoss(size_average=False)(output_suppresion[1], targets)
        # loss_trans_suppression = loss_trans_suppression/opt.batch_size
        
        # loss_suppression = (loss_cnn_suppression + loss_trans_suppression)/2

        ################  total 
        loss = loss_cnn + loss_trans  # + loss_suppression

        ################ Cập nhật các loại loss vào AverageMeter để lưu trung bình

        losses_cnn.update(loss_cnn.item(), inputs.size(0))   #update
        losses_trans.update(loss_trans.item(), inputs.size(0))
        # losses_suppression.update(loss_suppression.item(), inputs.size(0))
        total_loss.update(loss.item(), 1)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end_time)
        end_time = time.time()
        #
        iter = (epoch - 1) * len(data_loader) + (i + 1)
        adjust_learning_rate(optimizer, iter, opt)

        batch_logger.log({
            'epoch': epoch,
            'batch': i + 1,
            'iter': (epoch - 1) * len(data_loader) + (i + 1),
            'loss': total_loss.val,
            'lr': optimizer.param_groups[len(optimizer.param_groups) - 1]['lr']
        })
        if i % opt.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Total_loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Loss_CNN {loss_cnn.val:.4f} ({loss_cnn.avg:.4f})\t'
                  'Loss_Trans {loss_trans.val:.4f} ({loss_trans.avg:.4f})\t'
                  .format(epoch, i + 1, len(data_loader), batch_time=batch_time,
                    data_time=data_time, loss=total_loss, loss_cnn=losses_cnn,
                    loss_trans=losses_trans))
            
    # 'Loss_Suppression {loss_suppression.val:.4f} ({loss_suppression.avg:.4f})\t'
    # loss_suppression=losses_suppression
     
    epoch_logger.log({
        'epoch': epoch,
        'loss': total_loss.avg,
        'lr': optimizer.param_groups[len(optimizer.param_groups) - 2]['lr']
    })
    #if not os.path.exists(opt.pretrain_path):
        #os.mkdir(opt.pretrain_path)
    if epoch % 1 == 0:
        save_file_path = os.path.join(opt.result_path, 'save_{}.pth'.format(epoch))
        states = {
            'epoch': epoch + 1,
            'arch': opt.arch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        torch.save(states, save_file_path)
    
    return total_loss.avg, losses_cnn.avg, losses_trans.avg
        
        
def evaluate_conformer(epoch, data_loader, model, opt):
    print('--------------------------------------')
    print('TEST')
    model.eval()

    output_buffer = [[],[],[],[],[],[]] # lưu trữ đầu ra
    gt_buffer = [] # groundtruth

    # Tính 6 buffer
    for i, (imgs, gt_label, imgname) in enumerate(data_loader):
        with torch.no_grad():
            if not opt.no_cuda:
                inputs = imgs.cuda()
                targets = gt_label.cuda()
                
        for m in range(6):
            inputs = torch.autograd.Variable(inputs)
            targets = torch.autograd.Variable(targets)

            output = model(inputs)
            output = (output[0] + output[1])/2
            # output = F.sigmoid(output)
            output_buffer[m] += list(output.data.cpu().numpy())


        gt_buffer += list(targets.data.cpu().numpy())

    gt_results = np.stack(gt_buffer, axis=0) # ground truth
    gt_results = np.where(gt_results > 0.5, 1, 0)
    # print(gt_results.shape)
    ######################################
    results_ensemble = np.zeros(gt_results.shape)
    # print(results_ensemble.shape)
    
    # tổng hợp thông tin từ 6 buffer
    for m in range(6):
        output_results = np.stack(output_buffer[m], axis=0) # 39
        if m!=0:
            results_ensemble += output_results # 40
    ####################################################
    attr = np.where(results_ensemble > 2, 1, 0)
    result = get_pedestrian_metrics(attr, gt_results)
    
    print(f'mA: {result.ma} | Instance_acc: {result.instance_acc} | Instance_prec: {result.instance_prec} | instance_recall: {result.instance_recall}')
    print('--------------------------------------')
    return result.ma, result.instance_acc, result.instance_prec, result.instance_recall
    
# def evaluate_conformer(epoch, data_loader, model, opt):
#     print('test')
#     model.eval()

#     for i, (imgs, gt_label, imgname) in enumerate(data_loader):
#         with torch.no_grad():
#             if not opt.no_cuda:
#                 inputs = imgs.cuda()
#                 targets = gt_label.float().cuda()

#         for m in range(6):
#             inputs = torch.autograd.Variable(inputs)
#             targets = torch.autograd.Variable(targets)

#             output = model(inputs)
#             output = (output[0] + output[1])/2
#             output = F.sigmoid(output)
#             output_buffer[m] += list(output.data.cpu().numpy())


#         gt_buffer += list(targets.data.cpu().numpy())

#     gt_results = np.stack(gt_buffer, axis=0)
#     gt_results = np.where(gt_results > 0.5, 1, 0)
#     ######################################
#     results_ensemble = np.zeros(gt_results.shape)

#     for m in range(6):
#         output_results = np.stack(output_buffer[m], axis=0)
#         if m!=0:
#             results_ensemble += output_results
#         attr = np.where(output_results > 0.5, 1, 0)
#     ####################################################
#     attr = np.where(results_ensemble > 2, 1, 0)
#     result = get_pedestrian_metrics(attr, gt_results)
#     wandb.log({'epoch': epoch, 
#             'mA': result.ma,
#             'instance_acc': result.instance_acc,
#             'instance_prec': result.instance_prec,
#             'instance_recall': result.instance_recall})