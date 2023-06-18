import os
import argparse
import copy
from configs import cfg
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from data.imdb_loader import IMDBDataset

from models.resnet_model import ResnetModel

from utils.losses import get_normal_loss

from eval import evaluate

import datetime
from tensorboardX import SummaryWriter
from utils.visualization import process_image, process_seg_label, process_normal_label
import torch.nn.functional as F

# Ref: https://github.com/siriusdemon/pytorch-DEX/blob/master/dex/api.py
def expected_age(vector):
    res = [(i+1)*v for i, v in enumerate(vector)]
    return sum(res)

def NLLLoss(logs, targets):
    out = torch.zeros_like(targets, dtype=torch.float)
    for i in range(len(targets)):
        out[i] = logs[i][targets[i]]
    return -out.sum()/len(out)

def train(cfg, writer, mode, train_loader, test_loader, model, optimizer, loss_fn):

    steps = 0

    if cfg.TRAIN.SCHEDULE == 'Poly':
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda step: (1 - float(step) / cfg.TRAIN.STEPS)**cfg.TRAIN.POWER, last_epoch=-1)
    elif cfg.TRAIN.SCHEDULE == 'Cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg.TRAIN.STEPS)
    else:
        raise NotImplementedError

    print('-'*50)
    print(f'{mode} train start!')
    print('======> Model & Train Params <======')
    print('optimizer:', optimizer)
    print('scheduler:', scheduler)
    print('cfg.TRAIN.NORMAL_FACTOR:', cfg.TRAIN.NORMAL_FACTOR)
    print('-'*50)

    train_loss_list = []

    while steps < cfg.TRAIN.STEPS:
        for batch_idx, (image, gender, age) in enumerate(train_loader):
            if cfg.CUDA:
                image, gender, age = image.cuda(), gender.cuda(), age.cuda()
            
            optimizer.zero_grad()
            
            out, out_softmax = model(image)

            if mode == "gender_classifier":
                label = gender
            elif mode == "age_regressor":
                label = age
                label = label.view(-1, 1)
            elif mode == "age_classifier":
                label = age

            loss = loss_fn(out, label)
            loss.backward()

            optimizer.step()
            scheduler.step()

            train_loss_list.append(float(loss.detach().cpu().numpy()))

            # Print out the loss periodically.
            if steps % cfg.TRAIN.LOG_INTERVAL == 0:
                
                mean_train_loss = np.mean(np.array(train_loss_list))   

                print('-'*30)
                print('image.shape:', image.shape)
                print('out.shape:', out.shape)
                print('label.shape:', label.shape)
                if mode == "age_classifier":
                    preds_age_list = []
                    for o in out_softmax:
                        o = o.detach().cpu().numpy().squeeze()
                        preds_age = expected_age(o)
                        preds_age_list.append(preds_age)
                    print('preds_age_list:', preds_age_list)
                    age_label = label.argmax(-1)
                    print('age_label:', age_label)
                else:
                    print('out:', out)
                    print('label:', label)
                print('Train Step: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, Mean Loss: {:.6f}'.format(
                    steps, batch_idx * len(image), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.data.item(), mean_train_loss))
                print('-'*30)

                train_loss_list = []

                # Log to tensorboard
                writer.add_scalar('lr', scheduler.get_lr()[0], steps)
                writer.add_scalar(f'loss/{mode}', mean_train_loss, steps)

            if steps % cfg.TRAIN.SAVE_INTERVAL == 0:
                if mode == "gender_classifier":
                    checkpoint = {
                        'cfg': cfg,
                        'step': steps,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'loss': loss,
                        'acc': None
                    }
                elif mode == "age_regressor":
                    checkpoint = {
                        'cfg': cfg,
                        'step': steps,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'loss': loss,
                        'val_loss': None
                    }                    
                elif mode == "age_classifier":
                    checkpoint = {
                        'cfg': cfg,
                        'step': steps,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'loss': loss,
                        'val_loss': None,
                        'avg_pred_age': None,
                        'avg_label_age': None,
                        'avg_sub_age': None
                    }     

                if cfg.TRAIN.EVAL_CKPT:
                    model.eval()
                    
                    with torch.no_grad():
                        if mode == "gender_classifier":
                            
                            gender_list = []
                            preds_list = []

                            for batch_idx, (image, gender, age) in enumerate(test_loader):
                                if cfg.CUDA:
                                    image, gender, age = image.cuda(), gender.cuda(), age.cuda()

                                # Inference
                                out, _ = model(image)

                                # Get loss
                                # val_loss = loss_fn(out, gender)
                                
                                # Get class
                                preds = out.argmax(-1)
                                preds_list.extend(preds.tolist())

                                gender_label = gender.argmax(-1)
                                gender_list.extend(gender_label.tolist())
                                # gender_list.extend(gender.tolist())         
                                
                                if batch_idx % 1000 == 0:
                                    print(f' -- {batch_idx}')

                            assert len(preds_list) == len(gender_list), "Length of both lists should be same!"

                            # Count the number of matches
                            matches = sum([int(true == predicted) for true, predicted in zip(gender_list, preds_list)])

                            # Calculate the accuracy
                            val_acc = matches / len(gender_list)                                  

                            print('-'*50)
                            print(' ===> val_acc:', val_acc)
                            print('-'*50)
                            writer.add_scalar('eval/Val accuracy', val_acc, steps)
                            checkpoint['acc'] = val_acc

                        elif mode == "age_regressor":
                            
                            val_loss_list = []

                            for batch_idx, (image, gender, age) in enumerate(test_loader):
                                if cfg.CUDA:
                                    image, gender, age = image.cuda(), gender.cuda(), age.cuda()

                                # Inference
                                out = model(image)

                                # Get loss
                                age = age.view(-1, 1)
                                val_loss = loss_fn(out, age)
                                
                                val_loss_list.append(float(val_loss.detach().cpu().numpy()))

                                if batch_idx % 1000 == 0:
                                    print(f'-- {batch_idx}')

                            mean_val_loss = np.mean(np.array(val_loss_list))                                       
                            
                            print('-'*50)
                            print(' ===> mean_val_loss:', mean_val_loss)
                            print('-'*50)
                            writer.add_scalar('eval/Val loss', mean_val_loss, steps)
                            checkpoint['val_loss'] = mean_val_loss

                        elif mode == "age_classifier":
                            
                            age_list = []
                            preds_list = []
                            val_loss_list = []
                            age_subtract_list = []

                            for batch_idx, (image, gender, age) in enumerate(test_loader):
                                if cfg.CUDA:
                                    image, gender, age = image.cuda(), gender.cuda(), age.cuda()

                                # Inference
                                out, out_softmax = model(image)

                                # Get loss
                                val_loss = loss_fn(out, age)
                                val_loss_list.append(float(val_loss.detach().cpu().numpy()))
                                
                                age_label = age.argmax(-1)
                                age_list.extend(age_label.tolist())

                                # preds = out.argmax(-1)
                                # preds_list.extend(preds.tolist())
                                
                                # Get age
                                sub_preds_list = []
                                for o in out_softmax:
                                    o = o.cpu().numpy().squeeze()
                                    preds_age = expected_age(o)
                                    sub_preds_list.append(preds_age)
                                preds_list.extend(sub_preds_list)

                                # age_label = age.argmax(-1)
                                # age_list.extend(age_label.tolist())
                                # age_list.extend(age)

                                # Sub
                                for item1, item2 in zip(age_label.tolist(), sub_preds_list):
                                    item = abs(item1 - item2)
                                    age_subtract_list.append(item)
                                
                                if batch_idx % 1000 == 0:
                                    print(f' -- {batch_idx}')

                            assert len(preds_list) == len(age_list) == len(age_subtract_list), "Length of both lists should be same!"

                            # Count the number of matches
                            # matches = sum([int(true == predicted) for true, predicted in zip(age_list, preds_list)])

                            # Calculate the accuracy
                            # val_acc = matches / len(age_list)
                            
                            avg_pred_age = sum(preds_list) / float(len(preds_list))
                            avg_label_age = sum(age_list) / float(len(age_list))
                            avg_sub_age = sum(age_subtract_list) / float(len(age_subtract_list))

                            mean_val_loss = np.mean(np.array(val_loss_list))           

                            print('-'*50)
                            print(' ===> mean_val_loss:', mean_val_loss)
                            print(' ===> avg_label_age:', avg_label_age)
                            print(' ===> avg_pred_age:', avg_pred_age) 
                            print(' ===> avg_sub_age:', avg_sub_age)
                            print('-'*50)

                            writer.add_scalar('eval/Val loss', mean_val_loss, steps)
                            writer.add_scalar('eval/Val average of label_age', avg_label_age, steps)
                            writer.add_scalar('eval/Val average of pred_age', avg_pred_age, steps)
                            writer.add_scalar('eval/Val average of sub_age', avg_sub_age, steps)
                            
                            checkpoint['val_loss'] = mean_val_loss                           
                            checkpoint['avg_pred_age'] = avg_pred_age
                            checkpoint['avg_label_age'] = avg_label_age
                            checkpoint['avg_sub_age'] = avg_sub_age

                    model.train()
                
                ckpt_path =  os.path.join(cfg.SAVE_DIR, cfg.EXPERIMENT_NAME, f'{mode} ckpt-%s.pth' % str(steps).zfill(5))
                torch.save(checkpoint, ckpt_path)
                print(f' ===> Save ckpt: {ckpt_path}')
            
            steps += 1
            if steps >= cfg.TRAIN.STEPS:
                break

def main():
    parser = argparse.ArgumentParser(description="PyTorch Single Taks (Gender Classification & Age Regression/Classification) Training")
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
    parser.add_argument(
        "--train_mode",
        default='gender_classifier',
        choices=['gender_classifier', 'age_regressor', 'age_classifier']
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

    ## Train net1
    if args.train_mode == 'gender_classifier':

        ## Single Net
        print('-'*50)
        print('===> Single Network 1: Resnet Gender Classifier <===')
        net1 = ResnetModel(mode=args.train_mode)
        if cfg.CUDA:
            net1 = net1.cuda()
        net1.train()
        # print(net1)

        ## Loss function
        entropy_loss = nn.CrossEntropyLoss()
        # nll_loss = NLLLoss #nn.NLLLoss()

        ## Opti of net1
        optimizer_net1 = optim.SGD(net1.parameters(), lr=cfg.TRAIN.LR, momentum=cfg.TRAIN.MOMENTUM,
                            weight_decay=cfg.TRAIN.WEIGHT_DECAY)

        train(cfg=cfg, writer=writer, mode=args.train_mode, \
            train_loader=train_loader, test_loader=test_loader, \
            model=net1, optimizer=optimizer_net1, loss_fn=entropy_loss)

    ## Train net2
    if args.train_mode == 'age_regressor':

        ## Single Net
        print('===> Single Network 2: Resnet Age Regressor <===')
        net2 = ResnetModel(mode=args.train_mode)
        if cfg.CUDA:
            net2 = net2.cuda()
        net2.train()
        # print(net2)
        print('-'*50)

        ## Loss function
        # mse_loss = nn.MSELoss()
        mae_loss = nn.L1Loss()

        ## Opti of net2
        optimizer_net2 = optim.SGD(net2.parameters(), lr=cfg.TRAIN.LR, momentum=cfg.TRAIN.MOMENTUM,
                            weight_decay=cfg.TRAIN.WEIGHT_DECAY)
    
        train(cfg=cfg, writer=writer, mode=args.train_mode, \
            train_loader=train_loader, test_loader=test_loader, \
            model=net2, optimizer=optimizer_net2, loss_fn=mae_loss)

    ## Train net3
    if args.train_mode == 'age_classifier':

        ## Single Net
        print('-'*50)
        print('===> Single Network 3: Resnet Age Classifier <===')
        net3 = ResnetModel(mode=args.train_mode)
        if cfg.CUDA:
            net3 = net3.cuda()
        net3.train()
        # print(net3)

        ## Loss function
        entropy_loss = nn.CrossEntropyLoss()
        # nll_loss = NLLLoss #nn.NLLLoss()

        ## Opti of net3
        optimizer_net3 = optim.SGD(net3.parameters(), lr=cfg.TRAIN.LR, momentum=cfg.TRAIN.MOMENTUM,
                            weight_decay=cfg.TRAIN.WEIGHT_DECAY)

        train(cfg=cfg, writer=writer, mode=args.train_mode, \
            train_loader=train_loader, test_loader=test_loader, \
            model=net3, optimizer=optimizer_net3, loss_fn=entropy_loss)

if __name__ == '__main__':
    main()
