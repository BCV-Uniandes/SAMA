# -*- coding: utf-8 -*-
import os
import time
import argparse
import json
import copy
import yaml
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import precision_recall_curve, roc_auc_score,\
    average_precision_score

def create_parser():
    # SET THE PARAMETERS
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, default='config.yaml',
                        help='Name of the configuration yaml file')
    parser.add_argument('--image', action='store_true', default=False,
                        help='Model is multimodal or not (default: False)')
    parser.add_argument('--desc', action='store_true', default=False,
                        help='Model trained with the descriptor (default: False)')
    parser.add_argument('--fold', type=int, default=0,
                        help='Fold selected for start training. If 0, all folds'
                        'are run (default: 0).')
    parser.add_argument('--name', type=str, default='Test',
                        help='Name of the current test (default: Test)')
    parser.add_argument('--load_model', type=str, default='best_metric',
                        help='Weights to load (default: best_metric)')
    parser.add_argument('--resume', action='store_true', default=False,
                        help='Continue training a model')
    parser.add_argument('--load_path', type=str, default=None,
                        help='Name of the folder with the pretrained model')
    parser.add_argument('--ft', action='store_true', default=False,
                        help='Fine-tune a model')
    parser.add_argument('--freeze', action='store_false', default=True,
                        help='Freeze weights of the model')

    parser.add_argument('--gpu', type=str, default='0',
                        help='GPU(s) to use (default: 0)') 
    
    args = parser.parse_args()
    
    return args

args = create_parser()
with open(args.config_file, 'r') as file:
    config = yaml.load(file, Loader=yaml.FullLoader)
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


from models import Model
from data import CTdataset, DescDataset, collate
from utils import AverageMeter, save_graphs, get_lr, max_f_measure, Logger, \
     LossConstructor, OptimizerConstructor, seed_all, seed_worker


def main():
    training = True

    # DATA
    root = config['root']
    image_file = config['image_file'] 
    des_file = config['des_file'] 

    # SEEDS
    seed_all(config['seed'])

    # CREATE THE NETWORK ARCHITECTURE
    model = Model(image=args.image, desc=args.desc, **config['model_params'])
    model = model.cuda()
    intitial_weights = copy.deepcopy(model.state_dict())
    #Load Folds 
    with open(config['folds_file'], 'r') as file:
        folds = json.loads(file.read())
    pbar_folds = tqdm(folds)
    
    f = 0
    for train_idx, val_idx in pbar_folds:
        f += 1
        if f < args.fold: 
            continue #Skip folds to start in selected fold from args
        # Created once at the beginning of training
        if args.resume:
            writer = SummaryWriter(log_dir=f'runs/{config["load_path"][f]}')
            model_name = writer.get_logdir().split(os.path.sep)[-1]
            if not os.path.exists(os.path.join('Saved_Models', model_name, 'Resumed')):
                os.makedirs(os.path.join('Saved_Models', model_name, 'Resumed'))
            save_path = os.path.join('Saved_Models', f'{model_name}', 'Resumed')
        else:
            writer = SummaryWriter(log_dir=None, comment=f'_Fold_{f}_{args.name}')
            model_name = writer.get_logdir().split(os.path.sep)[-1]
            os.makedirs(os.path.join('Saved_Models', model_name))
            save_path = os.path.join('Saved_Models', f'{model_name}')
        best_train_dir = {}
        best_val_dir = {}
        logger = Logger(os.path.join(save_path, 'Log.txt'))
        model.load_state_dict(intitial_weights)
        torch.save(model.state_dict(),
                        os.path.join(save_path, 'first_model.pt'))
        n_params = sum([p.data.nelement() for p in model.parameters()])
        logger.write(f'Number of params: {n_params}')

        optimizer_init = OptimizerConstructor(config['optimizer'], 
                                         config['optim_args_dict'])
        optimizer = optimizer_init(model.parameters())
        
        scaler = torch.cuda.amp.GradScaler(enabled=config['amp'])
        
        if config['annealing'] == 1:
            annealing = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, patience=config['patience'], verbose=True) 
        elif config['annealing'] == 2:
            annealing = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                            config['ann_epochs'])
        elif config['annealing'] == 3:
            annealing = torch.optim.lr_scheduler.ExponentialLR(optimizer, 
                                                            gamma=config['ann_gamma'])
        
        criterion = LossConstructor(config['criterion'], 
                                    config['loss_args_dict'])
    
        # LOAD A MODEL IF NEEDED
        args.epoch = 0
        best_metric = 0
        
        if args.resume:
            checkpoint = torch.load(
                os.path.join(os.path.join('Saved_Models',config["load_path"][f],
                                          args.load_model)),
                map_location=lambda storage, loc: storage)
                
            if 'state_dict' in checkpoint:
                model_weights = checkpoint['state_dict']
                best_metric = checkpoint['best_metric']
                args.epoch = checkpoint['epoch']
                optimizer.load_state_dict(checkpoint['optimizer'])
            else:
                model_weights = checkpoint
                
            model.load_state_dict(model_weights, strict=(not args.ft))
            
        if args.ft:
            model_dict = model.state_dict()
            if args.load_path is not None:
                checkpoint = torch.load(
                    os.path.join(os.path.join('Saved_Models',args.load_path,
                                            args.load_model)),
                    map_location=lambda storage, loc: storage)
            else:
                checkpoint = torch.load(
                os.path.join(os.path.join(config["load_path"][f],
                                          args.load_model)),
                map_location=lambda storage, loc: storage)
                
            logger.write(f"Loading model and optimizer {checkpoint.get('epoch', '')}.")
            
            if 'state_dict' in checkpoint:
                backbone_weights = {
                    k: v for k, v in checkpoint['state_dict'].items() 
                    if k in model_dict}
                 
            else:
                backbone_weights = {
                    k:v for (k,v) in checkpoint.items() 
                    if k in model_dict}
                
            model_dict.update(backbone_weights)     
            model.load_state_dict(backbone_weights, strict=(not args.ft))

            if args.freeze:
                logger.write('- Frozen Backbone -')
                for param in model.backbone.parameters():
                    param.requires_grad = False

        # DATALOADERS
        if args.image:
            train_data = CTdataset(train_idx, image_file, des_file, root, 
                                config['image_size'])
            val_data = CTdataset(val_idx, image_file, des_file, root, 
                                config['image_size'])
            my_collate = collate(config['image_size'])#, config[data_aug]) 
            
        elif args.desc:
            train_data = DescDataset(train_idx, image_file, des_file, root, 
                                config['image_size'])
            val_data = DescDataset(val_idx, image_file, des_file, root, 
                                config['image_size'])
            my_collate = None
                
        train_loader = DataLoader(train_data, sampler=None, shuffle=True,
                                batch_size=config['batch'], num_workers=10,
                                worker_init_fn=seed_worker,
                                collate_fn=my_collate)
        val_loader = DataLoader(val_data, shuffle=False, sampler=None,
                                batch_size=config['batch'], num_workers=10,
                                collate_fn=my_collate)

        # TRAIN THE MODEL        
        
        try:
            if args.resume:
                pbar_epoch = tqdm(range(args.epoch+1, args.epoch+config['epochs']+1))
            else:
                pbar_epoch = tqdm(range(config['epochs']))
            if training:
                torch.cuda.empty_cache()
                for epoch in pbar_epoch:
                    torch.cuda.empty_cache()
                    lr = get_lr(optimizer)
                    pbar_epoch.set_description(f'Epoch {epoch}, Lr {lr}')
                    pbar_epoch.set_postfix({'T': time.strftime("%H:%M:%S")})
                    
                    train_dir = train(args, model, train_loader, optimizer, 
                                    criterion, scaler, epoch)
                    
                    logger.write(f'Epoch: {epoch} | Train | loss: {train_dir["loss"]:.4f}, '+ 
                                 f'F1: {train_dir["f1"]:.4f}, AP: {train_dir["ap"]:.4f}')
                    
                    val_dir = val(args, model, val_loader, criterion, epoch)
                    logger.write(f'Epoch: {epoch} | Val | loss: {val_dir["loss"]:.4f}, '+ 
                                 f'F1: {val_dir["f1"]:.4f}, AP: {val_dir["ap"]:.4f}')
                    
                    if config['annealing'] == 1:
                        annealing.step(val_dir['loss'])
                    else:
                        annealing.step()
                    
                    logger.write(f'Learning rate {lr}')
                    #Save checkpoint models every 10 epochs
                    if (epoch % 10) == 0:
                        state = {
                        'epoch': epoch,
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'loss': [train_dir['loss'], val_dir['loss']],
                        'lr': lr,
                        'metric': val_dir['f1'],
                        'best_metric': best_metric}
                        
                        logger.write(f'Checkpoint saved')
                        torch.save(state,
                        os.path.join(save_path,f'epoch_{epoch}_model.pt'))
                   
                    if best_metric < val_dir['f1']: 
                        best_metric = max(best_metric, val_dir['f1'])
                        best_train_dir = train_dir
                        best_val_dir = val_dir
                        
                        state = {
                        'epoch': epoch,
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'loss': [train_dir['loss'], val_dir['loss']],
                        'lr': lr,
                        'metric': val_dir['f1'],
                        'best_metric': best_metric}
                        
                        logger.write(f'New best model')
                        torch.save(state,
                        os.path.join(save_path,'best_model.pt'))
                        
                        scores_dict = {'ID': best_val_dir['patients'], 
                                        'Scores': best_val_dir['scores']}
                        best_val_scores = pd.DataFrame.from_dict(scores_dict)
                        best_val_scores.to_csv(os.path.join(save_path,
                                                'best_val_scores.csv'),
                                                index=False)
                                
                    loss_dict = {'Train': train_dir['loss'],
                                    'Val': val_dir['loss']}
                    f1_dict = {'Train': train_dir['f1'],
                                'Val': val_dir['f1']}
                    ap_dict = {'Train': train_dir['ap'],
                                'Val': val_dir['ap']}
                    roc_dict = {'Train_ROC': train_dir['roc'], 
                                'Val_ROC': val_dir['roc']}
                    
                    writer.add_scalars('Loss', loss_dict, epoch)
                    writer.add_scalars('F1', f1_dict, epoch)
                    writer.add_scalars('AP', ap_dict, epoch)
                    writer.add_scalars('ROC', roc_dict, epoch)
                    writer.flush()
                    writer.close()
                    
                    if config['amp']:
                        state['scaler'] = scaler.state_dict()
                
            writer.flush()
            writer.close()

            status = 'Succesfully completed'
            
        except KeyboardInterrupt:
            status = 'Terminating run early'
            raise
        
        except Exception as e:
            status = 'Unexpected error encountered'
            logger.write(e)
            raise

        finally:
            logger.write(status)
            if (best_metric != 0):
                logger.write('Saving last model')
                state = {
                        'epoch': epoch,
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'loss': [train_dir['loss'], val_dir['loss']],
                        'lr': lr,
                        'metric': val_dir['f1'],
                        'best_metric': best_metric}
                torch.save(state,
                           os.path.join(save_path, 'last_model.pt'))
                writer.add_hparams(hparam_dict={'Final lr': lr,
                                            'Batch size': config['batch'],
                                            'Patch size': f'{config["image_size"]}',
                                            },
                                   metric_dict={'F1_Train': train_dir['f1'],
                                            'F1_Val': val_dir['f1'],
                                            'AP_train': train_dir['ap'],
                                            'AP_val': val_dir['ap'],
                                            'ROC_train': train_dir['roc'],
                                            'ROC_val': val_dir['roc'],
                                            'Best_F1_Train': best_train_dir['f1'],
                                            'Best_F1_Val': best_val_dir['f1'],
                                            'Best_AP_train': best_train_dir['ap'],
                                            'Best_AP_val': best_val_dir['ap'],
                                            'Best_ROC_train': best_train_dir['roc'],
                                            'Best_ROC_val': best_val_dir['roc']})
                save_graphs(best_train_dir, best_val_dir, writer)
                
                writer.flush()
                writer.close()

def train(args, model, loader, optimizer, criterion, scaler, epoch):
    model.train()
    pbar_train = tqdm(loader)
    epoch_loss = AverageMeter()
    f1, ap = 0, 0
    labels, patients, scores, predictions = [], [], [], []
    
    pbar_train.set_description(f'Train| Loss: {epoch_loss.avg: .4f} '+
                               f'F1: {f1: .4f} '+
                               f'AP: {ap: .4f}')
    pbar_train.set_postfix({'T': time.strftime("%H:%M:%S")})
    
    for batch_idx, sample in enumerate(pbar_train):
        data = sample['data'].float().cuda()
        target = sample['target'].float().cuda()
        target = target.unsqueeze(1)
        if config['model_params']['num_classes']>1:
            target = torch.cat([1-target, target], dim=1)            
        
        labels.extend(sample['target'].tolist())
    
        if args.desc:
            descriptor = sample['descriptor'].float().cuda()
        optimizer.zero_grad()
        
        #Mixed precision torch 
        with torch.cuda.amp.autocast(enabled=config['amp']):
            if args.image and args.desc:
                out = model(data, descriptor)
            elif args.desc:
                out = model(descriptor)
            else:
                out = model(data)

        if config['model_params']['num_classes']>1:
            loss = criterion(out, target, epoch)
        else:
            loss = criterion(out, target, epoch)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # Output logits 
        if config['model_params']['num_classes']>1:
            softmax = torch.nn.Softmax(dim=1)
            pred = softmax(out)[:,1]
        else:
            pred = torch.sigmoid(out)
        
        scores.extend(pred.tolist())
        epoch_loss.update(loss.item())
        
        
        ap = average_precision_score(labels, scores)
        f1 = max_f_measure(labels, scores) 
        pbar_train.set_description(f'Train| Loss: {epoch_loss.avg: .4f} '+
                                f'F1: {f1: .4f} '+
                                f'AP: {ap: .4f}')
        pbar_train.set_postfix({'T': time.strftime("%H:%M:%S")})
    
    # Calculate metrics using sklearn functions
    roc = roc_auc_score(labels, scores)
    precision, recall, thresholds = precision_recall_curve(
        labels, scores)
    ap = average_precision_score(labels, scores)
    f1 = max_f_measure(labels, scores) 
    
    #Define output dictionary with metrics and loss
    output_dict = {'loss':epoch_loss.avg, 'f1': f1, 'scores': scores,
                   'roc': roc, 'precision': precision, 'recall': recall, 
                   'tresholds': thresholds, 'ap': ap}

    return output_dict


def val(args, model, loader, criterion, epoch):
    model.eval()
    pbar_val = tqdm(loader)
    epoch_loss = AverageMeter()
    f1, ap = 0, 0
    labels, patients, scores, predictions = [], [], [], []
    
    pbar_val.set_description(f'Val| Loss: {epoch_loss.avg: .4f} '+
                               f'F1: {f1: .4f} '+
                               f'AP: {ap: .4f}')
    pbar_val.set_postfix({'T': time.strftime("%H:%M:%S")})

    for batch_idx, sample in enumerate(pbar_val):
        data = sample['data'].float().cuda()
        target = sample['target'].float().cuda()
        target = target.unsqueeze(1)
        if config['model_params']['num_classes']>1:
            target = torch.cat([1-target, target], dim=1) 
        patients.extend(sample['id'].tolist())
        labels.extend(sample['target'].tolist())
        
        if args.desc:
            descriptor = sample['descriptor'].float().cuda()
        
        with torch.no_grad():
            #Mixed precision 
            with torch.cuda.amp.autocast():
                if args.image and args.desc:
                    out = model(data, descriptor)
                elif args.desc:
                    out = model(descriptor)
                else:
                    out = model(data)       
        
        if config['model_params']['num_classes']>1:
            loss = criterion(out, target, epoch)
        else:
            loss = criterion(out, target, epoch)
        
        epoch_loss.update(loss.item())

        # Output logits 
        if config['model_params']['num_classes']>1:
            softmax = torch.nn.Softmax(dim=1)
            pred = softmax(out)[:,1]
        else:
            pred = torch.sigmoid(out)
        scores.extend(pred.tolist())
        
        epoch_loss.update(loss.item())
        
        ap = average_precision_score(labels, scores)
        f1 = max_f_measure(labels, scores) 
        pbar_val.set_description(f'Val| Loss: {epoch_loss.avg: .4f} '+
                                 f'F1: {f1: .4f} '+
                                 f'AP: {ap: .4f}')
        pbar_val.set_postfix({'T': time.strftime("%H:%M:%S")})
    
    # Calculate metrics using sklearn functions
    roc = roc_auc_score(labels, scores)
    precision, recall, thresholds = precision_recall_curve(
        labels, scores)
    ap = average_precision_score(labels, scores)
    f1 = max_f_measure(labels, scores)
  
    #Define output dictionary with metrics and loss
    output_dict = {'loss':epoch_loss.avg, 'f1': f1, 'roc': roc, 
                   'precision': precision, 'recall': recall, 'scores': scores,
                   'patients': patients, 'tresholds': thresholds, 'ap': ap}
    
    return output_dict


if __name__ == '__main__':
    main()
