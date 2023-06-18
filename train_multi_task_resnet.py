import os
import argparse
import copy
from configs import cfg

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from data.imdb_loader import IMDBDataset

from models.nddr_net import NDDRNet
from models.resnet_model import ResnetModel

from utils.losses import get_normal_loss

from eval import evaluate

import datetime
from tensorboardX import SummaryWriter
from utils.visualization import process_image, process_seg_label, process_normal_label
import torch.nn.functional as F

def expected_age(vector):
    res = [(i+1)*v for i, v in enumerate(vector)]
    return sum(res)

def main():
    parser = argparse.ArgumentParser(description="PyTorch NDDR Training")
    parser.add_argument(
        "--config-file",
        default="configs/vgg16_nddr_pret.yaml",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()
    print('-'*50)
    print('======> Args <======')
    print(args)
    print('-'*50)

    cfg.merge_from_file(args.config_file)
    if cfg.EXPERIMENT_NAME is None:
        cfg.EXPERIMENT_NAME = args.config_file.split('/')[-1][:-5]
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    if not os.path.exists(os.path.join(cfg.SAVE_DIR, cfg.EXPERIMENT_NAME)):
        os.makedirs(os.path.join(cfg.SAVE_DIR, cfg.EXPERIMENT_NAME))

    ## Datasets
    train_loader = torch.utils.data.DataLoader(
        IMDBDataset(
            mode='TRAIN',
            data_dir=cfg.DATA_DIR,
            output_size=cfg.TRAIN.OUTPUT_SIZE,
            random_scale=cfg.TRAIN.RANDOM_SCALE,
            random_mirror=cfg.TRAIN.RANDOM_MIRROR,
            random_crop=cfg.TRAIN.RANDOM_CROP,
        ),
        batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True)

    if cfg.TRAIN.EVAL_CKPT:
        test_loader = torch.utils.data.DataLoader(
            IMDBDataset(
                mode='EVAL',
                data_dir=cfg.DATA_DIR,
                output_size=cfg.TEST.OUTPUT_SIZE,
                random_scale=cfg.TEST.RANDOM_SCALE,
                random_mirror=cfg.TEST.RANDOM_MIRROR,
                random_crop=cfg.TEST.RANDOM_CROP,
            ),
            batch_size=cfg.TEST.BATCH_SIZE, shuffle=False)
    print('-'*50)
    print('======> Datasets <======')
    print('cfg.TRAIN.BATCH_SIZE:', cfg.TRAIN.BATCH_SIZE)
    print('cfg.TEST.BATCH_SIZE:', cfg.TEST.BATCH_SIZE)
    print('cfg.EXPERIMENT_NAME:', cfg.EXPERIMENT_NAME)
    print('-'*50)

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d~%H:%M:%S")
    experiment_log_dir = os.path.join(cfg.LOG_DIR, cfg.EXPERIMENT_NAME, timestamp)
    if not os.path.exists(experiment_log_dir):
        os.makedirs(experiment_log_dir)
    writer = SummaryWriter(logdir=experiment_log_dir)

    ## Single Net
    print('-'*50)
    print('===> Single Network 1: Resnet Gender Classifier <===')
    net1 = ResnetModel(mode='gender_classifier', weights=cfg.TRAIN.WEIGHT_1)
    if cfg.CUDA:
        net1 = net1.cuda()
    net1.train()
    # print(net1)
    print('-'*50)

    # print('-'*50)
    # print('===> Single Network 2: Resnet Age Regressor <===')
    # net2 = ResnetModel(mode='age_regressor', weights=cfg.TRAIN.WEIGHT_2)
    # if cfg.CUDA:
    #     net2 = net2.cuda()
    # net2.train()
    # # print(net2)
    # print('-'*50)

    print('-'*50)
    print('===> Single Network 2: Resnet Age Classifier <===')
    net2 = ResnetModel(mode='age_classifier', weights=cfg.TRAIN.WEIGHT_2)
    if cfg.CUDA:
        net2 = net2.cuda()
    net2.train()
    # print(net2)
    print('-'*50)

    ## NDDR
    print('-'*50)
    print('======> NDDRNet <======')
    model = NDDRNet(copy.deepcopy(net1), copy.deepcopy(net2),
                    dataset='imdb-clean',
                    shortcut=cfg.MODEL.SHORTCUT,
                    bn_before_relu=cfg.MODEL.BN_BEFORE_RELU)
    # print(model)
    print('-'*50)

    if cfg.CUDA:
        model = model.cuda()
    model.train()
    steps = 0

    # Loss function
    entropy_loss = nn.CrossEntropyLoss()
    # mse_loss = nn.MSELoss()
    mae_loss = nn.L1Loss()

    # hacky way to pick params
    nddr_params = []
    fc8_weights = []
    fc8_bias = []
    base_params = []
    for k, v in model.named_parameters():
        if 'nddrs' in k:
            nddr_params.append(v)
        elif cfg.MODEL.FC8_ID in k:
            if 'weight' in k:
                fc8_weights.append(v)
            else:
                assert 'bias' in k
                fc8_bias.append(v)
        else:
            base_params.append(v)
    assert len(nddr_params) > 0 and len(fc8_weights) > 0 and len(fc8_bias) > 0

    # parameter_dict = [
    #     {'params': base_params},
    #     {'params': fc8_weights},
    #     {'params': fc8_bias},
    #     {'params': nddr_params}
    # ]
    parameter_dict = [
        {'params': base_params},
        {'params': fc8_weights, 'lr': cfg.TRAIN.LR * cfg.TRAIN.FC8_WEIGHT_FACTOR},
        {'params': fc8_bias, 'lr': cfg.TRAIN.LR * cfg.TRAIN.FC8_BIAS_FACTOR},
        {'params': nddr_params, 'lr': cfg.TRAIN.LR * cfg.TRAIN.NDDR_FACTOR}
    ]
    optimizer = optim.SGD(parameter_dict, lr=cfg.TRAIN.LR, momentum=cfg.TRAIN.MOMENTUM,
                          weight_decay=cfg.TRAIN.WEIGHT_DECAY)

    if cfg.TRAIN.SCHEDULE == 'Poly':
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda step: (1 - float(step) / cfg.TRAIN.STEPS)**cfg.TRAIN.POWER, last_epoch=-1)
    elif cfg.TRAIN.SCHEDULE == 'Cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg.TRAIN.STEPS)
    else:
        raise NotImplementedError

    print('-'*50)
    print('======> Model & Train Params <======')
    print('optimizer:', optimizer)
    print('scheduler:', scheduler)
    print('cfg.TRAIN.FC8_WEIGHT_FACTOR:', cfg.TRAIN.FC8_WEIGHT_FACTOR)
    print('cfg.TRAIN.FC8_BIAS_FACTOR:', cfg.TRAIN.FC8_BIAS_FACTOR)
    print('cfg.TRAIN.NDDR_FACTOR:', cfg.TRAIN.NDDR_FACTOR)
    print('cfg.TRAIN.NORMAL_FACTOR:', cfg.TRAIN.NORMAL_FACTOR)
    print('-'*50)

    nddr_train_loss_list = []
    nddr_gender_loss_list = []
    nddr_age_loss_list = []
    net1_train_loss_list = []
    net2_train_loss_list = []

    best_nddr_gender_val_acc = 0
    best_nddr_avg_sub_age = 100
    best_nddr_mean_age_val_loss = 100

    nddr_gender_val_acc_best_steps = 0
    nddr_avg_sub_age_best_steps = 0
    nddr_mean_age_val_loss_best_steps = 0

    while steps < cfg.TRAIN.STEPS:
        for batch_idx, (image, gender, age) in enumerate(train_loader):
            if cfg.CUDA:
                image, gender, age = image.cuda(), gender.cuda(), age.cuda()
                label_1 = gender
                # label_2 = age.view(-1, 1)
                label_2 = age
            
            optimizer.zero_grad()
            
            out1, out2, _, out2_softmax = model(image)

            with torch.no_grad():
                orin_net1_out, _ = net1(image)
                orin_net2_out, orin_net2_out_softmax = net2(image)
                orin_net1_loss = entropy_loss(orin_net1_out, label_1)
                # orin_net2_loss = mae_loss(orin_net2_out, label_2)
                orin_net2_loss = entropy_loss(orin_net2_out, label_2)

                net1_train_loss_list.append(float(orin_net1_loss.detach().cpu().numpy()))
                net2_train_loss_list.append(float(orin_net2_loss.detach().cpu().numpy()))

            loss_gender_classification = entropy_loss(out1, label_1)
            # loss_age_regression = mae_loss(out2, label_2)
            loss_age_classification = entropy_loss(out2, label_2)

            # loss = loss_gender_classification + cfg.TRAIN.NORMAL_FACTOR * loss_age_regression
            loss = loss_gender_classification + cfg.TRAIN.NORMAL_FACTOR * loss_age_classification

            loss.backward()

            optimizer.step()
            scheduler.step()

            nddr_train_loss_list.append(float(loss.detach().cpu().numpy()))
            nddr_gender_loss_list.append(float(loss_gender_classification.detach().cpu().numpy()))
            # nddr_age_loss_list.append(float(loss_age_regression.detach().cpu().numpy()))
            nddr_age_loss_list.append(float(loss_age_classification.detach().cpu().numpy()))

            # Print out the loss periodically.
            if steps % cfg.TRAIN.LOG_INTERVAL == 0:
                
                nddr_mean_total_train_loss = np.mean(np.array(nddr_train_loss_list))
                nddr_mean_gender_train_loss = np.mean(np.array(nddr_gender_loss_list))
                nddr_mean_age_train_loss = np.mean(np.array(nddr_age_loss_list))

                print('-'*30)
                print(' ==> NDDR Model <==')
                print('image.shape:', image.shape)
                
                ## Task 1
                print('out1.shape:', out1.shape)
                print('out1:', out1)
                print('label_1.shape:', label_1.shape)
                print('label_1:', label_1)      
                
                ## Task 2          
                print('out2.shape:', out2.shape)
                # print('out2:', out2)
                print('label_2.shape:', label_2.shape)
                # print('label_2:', label_2)
                preds_age_list = []
                for o in out2_softmax:
                    o = o.detach().cpu().numpy().squeeze()
                    preds_age = expected_age(o)
                    preds_age_list.append(preds_age)
                print('preds_age_list:', preds_age_list)
                age_label = label_2.argmax(-1)
                print('age_label:', age_label)

                print('Train Step: {} [{}/{} ({:.0f}%)]\tMean Loss: {:.6f}\tMean Loss1: {:.6f}\tMean Loss2: {:.6f}'.format(
                    steps, batch_idx * len(image), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), nddr_mean_total_train_loss,
                    nddr_mean_gender_train_loss, nddr_mean_age_train_loss))
                print('-'*30)

                nddr_train_loss_list = []
                nddr_gender_loss_list = []
                nddr_age_loss_list = []

                net1_mean_total_train_loss = np.mean(np.array(net1_train_loss_list))
                net2_mean_total_train_loss = np.mean(np.array(net2_train_loss_list))

                print('-'*30)
                print(' ==> Each Single Model <==')

                ## Task 1
                print('out1.shape:', orin_net1_out.shape)
                print('out1:', orin_net1_out)
                print('label_1.shape:', label_1.shape)
                print('label_1:', label_1)           

                ## Task 2     
                print('out2.shape:', orin_net2_out.shape)
                # print('out2:', orin_net2_out)
                print('label_2.shape:', label_2.shape)
                # print('label_2:', label_2)
                preds_age_list = []
                for o in orin_net2_out_softmax:
                    o = o.detach().cpu().numpy().squeeze()
                    preds_age = expected_age(o)
                    preds_age_list.append(preds_age)
                print('preds_age_list:', preds_age_list)
                age_label = label_2.argmax(-1)
                print('age_label:', age_label)

                print('orin_net1_loss (gender classification): {:.6f}'.format(net1_mean_total_train_loss))
                # print('orin_net2_loss (age regression): {:.6f}'.format(net2_mean_total_train_loss))
                print('orin_net2_loss (age classification): {:.6f}'.format(net2_mean_total_train_loss))
                print('-'*30)

                net1_train_loss_list = []
                net2_train_loss_list = []

                # Log to tensorboard
                writer.add_scalar('lr', scheduler.get_lr()[0], steps)
                writer.add_scalar('loss/overall', loss.data.item(), steps)
                writer.add_scalar('loss/gender_classification', loss_gender_classification.data.item(), steps)
                # writer.add_scalar('loss/age_regression', loss_age_regression.data.item(), steps)
                writer.add_scalar('loss/age_classification', loss_age_classification.data.item(), steps)
                # writer.add_image('image', process_image(image[0]), steps)

            if steps % cfg.TRAIN.SAVE_INTERVAL == 0:
                checkpoint = {
                    'cfg': cfg,
                    'step': steps,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'loss': loss,
                    'loss_gender_classification': loss_gender_classification,
                    # 'loss_age_regression': loss_age_regression,
                    'loss_age_classification': loss_age_classification,
                    'gender_classification_acc': None,
                    # 'age_regression_val_loss': None
                    'age_classification_val_loss': None,
                    'avg_pred_age': None,
                    'avg_label_age': None,
                    'avg_sub_age': None,
                }

                if cfg.TRAIN.EVAL_CKPT:
                    
                    model.eval()

                    with torch.no_grad():

                        gender_list = []
                        nddr_gender_preds_list = []
                        single_gender_preds_list = []

                        age_list = []
                        nddr_age_val_loss_list = []
                        single_age_val_loss_list = []
                        nddr_age_preds_list = []
                        single_age_preds_list = []
                        nddr_age_subtract_list = []
                        single_age_subtract_list = []

                        for batch_idx, (image, gender, age) in enumerate(test_loader):
                            if cfg.CUDA:
                                image, gender, age = image.cuda(), gender.cuda(), age.cuda()

                            # Inference
                            out1, out2, _, out2_softmax = model(image)
                            gender_single_out, _ = net1(image)
                            age_single_out, age_single_out_softmax = net2(image)

                            # Get classification loss
                            val_loss_gender_classification = entropy_loss(out1, gender)
                            
                            # Get gender class
                            preds = out1.argmax(-1)
                            single_preds = gender_single_out.argmax(-1)

                            gender_label = gender.argmax(-1)

                            gender_list.extend(gender_label.tolist())
                            nddr_gender_preds_list.extend(preds.tolist())
                            single_gender_preds_list.extend(single_preds.tolist())

                            # # Get regression loss
                            # age = age.view(-1, 1)
                            # nddr_val_loss_age_regression = mae_loss(out2, age)
                            # single_val_loss_age_regression = mae_loss(age_single_out, age)

                            # age_nddr_val_loss_list.append(float(nddr_val_loss_age_regression.detach().cpu().numpy()))
                            # age_single_val_loss_list.append(float(single_val_loss_age_regression.detach().cpu().numpy()))

                            # Get age classification loss
                            nddr_val_loss_age_classification = entropy_loss(out2, age)
                            single_val_loss_age_classification = entropy_loss(age_single_out, age)

                            nddr_age_val_loss_list.append(float(nddr_val_loss_age_classification.detach().cpu().numpy()))
                            single_age_val_loss_list.append(float(single_val_loss_age_classification.detach().cpu().numpy()))

                            age_label = age.argmax(-1)
                            age_list.extend(age_label.tolist())

                            # Get preds age (nddr)
                            sub_preds_list = []
                            for o in out2_softmax:
                                o = o.cpu().numpy().squeeze()
                                preds_age = expected_age(o)
                                sub_preds_list.append(preds_age)
                            nddr_age_preds_list.extend(sub_preds_list)

                            # Sub (nddr)
                            for item1, item2 in zip(age_label.tolist(), sub_preds_list):
                                item = abs(item1 - item2)
                                nddr_age_subtract_list.append(item)

                            # Get preds age (single)
                            sub_preds_list = []
                            for o in age_single_out_softmax:
                                o = o.cpu().numpy().squeeze()
                                preds_age = expected_age(o)
                                sub_preds_list.append(preds_age)
                            single_age_preds_list.extend(sub_preds_list)

                            # Sub (single)
                            for item1, item2 in zip(age_label.tolist(), sub_preds_list):
                                item = abs(item1 - item2)
                                single_age_subtract_list.append(item)

                            if batch_idx % 1000 == 0:
                                print(f' -- {batch_idx}')

                        assert len(nddr_gender_preds_list) == len(single_gender_preds_list) == len(gender_list), "Length of both lists should be same!"
                        assert len(nddr_age_preds_list) == len(single_age_preds_list) == len(age_list), "Length of both lists should be same!"

                        # Count the number of matches
                        nddr_matches = sum([int(true == predicted) for true, predicted in zip(gender_list, nddr_gender_preds_list)])
                        single_matches = sum([int(true == predicted) for true, predicted in zip(gender_list, single_gender_preds_list)])

                        # Calculate the accuracy
                        nddr_gender_val_acc = nddr_matches / len(gender_list)     
                        single_gender_val_acc = single_matches / len(gender_list)     

                        nddr_mean_age_val_loss = np.mean(np.array(nddr_age_val_loss_list))           
                        single_mean_age_val_loss = np.mean(np.array(single_age_val_loss_list))                      

                        avg_label_age = sum(age_list) / float(len(age_list))

                        nddr_avg_pred_age = sum(nddr_age_preds_list) / float(len(nddr_age_preds_list))
                        nddr_avg_sub_age = sum(nddr_age_subtract_list) / float(len(nddr_age_subtract_list))

                        single_avg_pred_age = sum(single_age_preds_list) / float(len(single_age_preds_list))
                        single_avg_sub_age = sum(single_age_subtract_list) / float(len(single_age_subtract_list))

                        # Set best performance
                        if best_nddr_gender_val_acc < nddr_gender_val_acc:
                            best_nddr_gender_val_acc = nddr_gender_val_acc
                            nddr_gender_val_acc_best_steps = steps
                        if best_nddr_avg_sub_age > nddr_avg_sub_age:
                            best_nddr_avg_sub_age = nddr_avg_sub_age
                            nddr_avg_sub_age_best_steps = steps                        
                        if best_nddr_mean_age_val_loss > nddr_mean_age_val_loss:
                            best_nddr_mean_age_val_loss = nddr_mean_age_val_loss
                            nddr_mean_age_val_loss_best_steps = steps   

                        print('-'*50)
                        print(' ===> Gender classification validation accuracy (NDDR):', nddr_gender_val_acc)
                        print(' ===> Gender classification validation accuracy (Single model):', single_gender_val_acc)
                        print('-'*50)
                        print(' ===> Age classification validation loss (NDDR):', nddr_mean_age_val_loss)
                        print(' ===> Age classification validation loss (Single model):', single_mean_age_val_loss)
                        print(' ===> Average of age_list:', avg_label_age)
                        print(' ===> Average of age_preds_list (NDDR):', nddr_avg_pred_age)
                        print(' ===> Average of age_preds_list (Single model):', single_avg_pred_age)
                        print(' ===> Average of age_subtract_list (NDDR):', nddr_avg_sub_age)
                        print(' ===> Average of age_subtract_list (Single model):', single_avg_sub_age) 
                        print('-'*50)

                        print('-'*70)
                        print(f' ===> Best gender classification validation accuracy ({nddr_gender_val_acc_best_steps} steps):', best_nddr_gender_val_acc)
                        print(f' ===> Best age classification validation loss ({nddr_mean_age_val_loss_best_steps} steps):', best_nddr_mean_age_val_loss)
                        print(f' ===> Best average of age_subtract_list ({nddr_avg_sub_age_best_steps} steps):', best_nddr_avg_sub_age)
                        print('-'*70)

                        writer.add_scalar('eval/NDDR val gender classification accuracy', nddr_gender_val_acc, steps)
                        writer.add_scalar('eval/NDDR val age classification loss', nddr_mean_age_val_loss, steps)
                        writer.add_scalar('eval/NDDR val average of label_age', avg_label_age, steps)
                        writer.add_scalar('eval/NDDR val average of pred_age', nddr_avg_pred_age, steps)
                        writer.add_scalar('eval/NDDR val average of sub_age', nddr_avg_sub_age, steps)

                        writer.add_scalar('eval/Best NDDR val gender classification accuracy', best_nddr_gender_val_acc, steps)
                        writer.add_scalar('eval/Best NDDR val age classification loss', best_nddr_mean_age_val_loss, steps)
                        writer.add_scalar('eval/Best NDDR val average of sub_age', best_nddr_avg_sub_age, steps)

                        checkpoint['gender_classification_acc'] = nddr_gender_val_acc
                        # checkpoint['age_regression_val_loss'] = nddr_mean_age_val_loss
                        checkpoint['age_classification_val_loss'] = nddr_mean_age_val_loss
                        checkpoint['avg_pred_age'] = avg_label_age
                        checkpoint['avg_label_age'] = nddr_avg_pred_age
                        checkpoint['avg_sub_age'] = nddr_avg_sub_age

                ckpt_path =  os.path.join(cfg.SAVE_DIR, cfg.EXPERIMENT_NAME, f'nddr ckpt-%s.pth' % str(steps).zfill(5))
                torch.save(checkpoint, ckpt_path)
                print(f' ===> Save ckpt: {ckpt_path}')

            steps += 1
            if steps >= cfg.TRAIN.STEPS:
                break

if __name__ == '__main__':
    main()
