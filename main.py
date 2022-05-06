import warnings
warnings.filterwarnings('ignore',category=FutureWarning)
import os
import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from data_util import MimicFullDataset, my_collate_fn
from tqdm import tqdm
import shutil
import json
import ipdb
import sys
import numpy as np
from constant import MIMIC_2_DIR, MIMIC_3_DIR
from evaluation import all_metrics, print_metrics
from torch.utils.data import DataLoader
from train_parser import generate_parser
from train_utils import generate_output_folder_name, generate_model
from find_threshold import find_threshold_micro
from accelerate import DistributedDataParallelKwargs, Accelerator


def run(args):
    kwargs_handlers = [DistributedDataParallelKwargs(find_unused_parameters=True)]
    accelerator = Accelerator(kwargs_handlers=kwargs_handlers)    
    
    output_basename = generate_output_folder_name(args)
    accelerator.print(output_basename)
    output_path = os.path.join(args.output_base_dir, output_basename)

    try:
        os.system(f"mkdir -p {output_path}")
    except BaseException:
        pass
   
    with open(os.path.join(output_path, 'args.json'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    word_embedding_path = args.word_embedding_path         
    accelerator.print(f"Use word embedding from {word_embedding_path}")
    
    train_dataset = MimicFullDataset(args.version, "train", word_embedding_path, args.truncate_length, args.label_truncate_length, args.term_count, args.sort_method)
    dev_dataset = MimicFullDataset(args.version, "dev", word_embedding_path, args.truncate_length)
    test_dataset = MimicFullDataset(args.version, "test", word_embedding_path, args.truncate_length)

    if args.knowledge_distill:
        raise NotImplementedError
        # teacher_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=my_collate_fn, shuffle=False, num_workers=1)
    else:
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=my_collate_fn, shuffle=True, num_workers=8, pin_memory=True)
    eval_batch_size = args.eval_batch_size if args.eval_batch_size > 0 else args.batch_size
    dev_dataloader = DataLoader(dev_dataset, batch_size=eval_batch_size, collate_fn=my_collate_fn, shuffle=False, num_workers=8, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=eval_batch_size, collate_fn=my_collate_fn, shuffle=False, num_workers=8, pin_memory=True)
    
    model = generate_model(args, train_dataset).to(accelerator.device)
    accelerator.print(model)
    optimizer, scheduler_step = model.configure_optimizers(train_dataloader)
    optimizer = optimizer[0]
    scheduler_step = scheduler_step[0]
    
    # prepare label input feature
    model.c_input_word = train_dataset.c_input_word.to(accelerator.device)
    model.c_word_mask = train_dataset.c_word_mask.to(accelerator.device)
    model.c_word_sent = train_dataset.c_word_sent.to(accelerator.device)
    #model.mc_input_word = train_dataset.mc_input_word.to(accelerator.device)
    #model.mc_word_mask = train_dataset.mc_word_mask.to(accelerator.device)
    #model.mc_word_sent = train_dataset.mc_word_sent.to(accelerator.device)
    
    model, optimizer, train_dataloader = \
        accelerator.prepare(model, optimizer, train_dataloader)

    # adversarial_training
    if args.adv_training:
        raise NotImplementedError
    
    steps = 0
    best_dev_metric = {}
    best_test_metric = {}
    early_stop_count = 0
    best_epoch_idx = 0
   
    if accelerator.is_local_main_process and args.debug:
        dev_metric, _ = eval_func(model, dev_dataloader, args.device, args.prob_threshold, True, args)
        print_metrics(dev_metric, 'DEBUG')

    for epoch_idx in range(1, args.train_epoch + 1):
        epoch_dev_metric, epoch_test_metric, steps = train_one_epoch(model, steps, train_dataloader, dev_dataloader, test_dataloader, optimizer, scheduler_step, args, accelerator)
        accelerator.wait_for_everyone()
        if accelerator.is_local_main_process:
            # torch.save(model, os.path.join(output_path, f"epoch{epoch_idx}.pth"))
            accelerator.save(accelerator.unwrap_model(model), os.path.join(output_path, f"epoch{epoch_idx}.pth"))
            print_metrics(epoch_dev_metric, 'Dev_Epoch' + str(epoch_idx))
            print_metrics(epoch_dev_metric, 'Dev_Epoch' + str(epoch_idx), os.path.join(output_path, 'metric_log'))
            print_metrics(epoch_test_metric, 'Test_Epoch' + str(epoch_idx))
            print_metrics(epoch_test_metric, 'Test_Epoch' + str(epoch_idx), os.path.join(output_path, 'metric_log'))
    
        # Early Stop
        if not best_dev_metric:
            best_dev_metric = epoch_dev_metric
            best_test_metric = epoch_test_metric
            best_epoch_idx = epoch_idx
        else:
            if args.early_stop_metric in epoch_dev_metric:
                if epoch_dev_metric[args.early_stop_metric] >= best_dev_metric[args.early_stop_metric]:
                    best_dev_metric = epoch_dev_metric
                    best_test_metric = epoch_test_metric
                    best_epoch_idx = epoch_idx
                    early_stop_count = 0
                else:
                    early_stop_count += 1

        if args.early_stop_epoch > 0 and early_stop_count >= args.early_stop_epoch:
            accelerator.print(f"Early Stop at Epoch {epoch_idx}, \
                    metric {args.early_stop_metric} not improve on dev set for {early_stop_count} epoch.")
            break
        
    if accelerator.is_local_main_process:
        best_train_metric, _ = eval_func(model, train_dataloader, accelerator.device, args.prob_threshold, True, args)
        print_metrics(best_train_metric, 'Best_Train_Epoch' + str(best_epoch_idx))
        print_metrics(best_train_metric, 'Best_Train_Epoch' + str(best_epoch_idx), os.path.join(output_path, 'metric_log'))
        print_metrics(best_dev_metric, 'Best_Dev_Epoch' + str(best_epoch_idx))
        print_metrics(best_dev_metric, 'Best_Dev_Epoch' + str(best_epoch_idx), os.path.join(output_path, 'metric_log'))
        print_metrics(best_test_metric, 'Best_Test_Epoch' + str(best_epoch_idx))
        print_metrics(best_test_metric, 'Best_Test_Epoch' + str(best_epoch_idx), os.path.join(output_path, 'metric_log'))
        best_path = os.path.join(output_path, f"epoch{best_epoch_idx}.pth")
        new_path = os.path.join(output_path, "best_epoch.pth")
        os.system(f'cp {best_path} {new_path}')

    
def train_one_epoch(model, steps, train_dataloader, dev_dataloader, test_dataloader, optimizer, scheduler, args, accelerator=None):
    model.train()
    epoch_loss = 0.
    # epoch_mc_loss = 0.
    epoch_kl_loss = 0.
    epoch_c_loss = 0.
    
    # if args.knowledge_distill:
    #     epoch_teacher = 0.
    #     epoch_total = 0.
    # if abs(args.code_penalty) > 0.0:
    #     code_penalty = CodePenalty(args.code_penalty, ind2c)
        
    epoch_iterator = tqdm(train_dataloader, desc="Iteration", ascii=True, disable=not accelerator.is_local_main_process)
    for batch_idx, batch in enumerate(epoch_iterator):
        batch_gpu = tuple([x.to(accelerator.device) for x in batch])
        #if args.rdrop_alpha > 0.0:
        #    ori_loss = model.forward_rdrop(batch_gpu)
        #else:
        ori_loss = model(batch_gpu, rdrop=args.rdrop_alpha > 0.0)
        if isinstance(ori_loss, dict):
            loss = ori_loss['loss']
        else:
            loss = ori_loss
            
        batch_loss = float(loss.item())
        epoch_loss += batch_loss
        
        # batch_mc_loss = float(ori_loss['mc_loss'].item())
        # epoch_mc_loss += batch_mc_loss
        batch_c_loss = float(ori_loss['c_loss'].item())
        epoch_c_loss += batch_c_loss
        
        if args.rdrop_alpha > 0.0:
            batch_kl_loss = float(ori_loss['kl_loss'].item())
            epoch_kl_loss += batch_kl_loss
                
        if args.gradient_accumulation_steps > 1:
            loss = loss / args.gradient_accumulation_steps

        # loss.backward()
        accelerator.backward(loss)
            
        # if args.adv_training:
        #     adv.adversarial_training(args, inputs, optimizer)

        if not args.knowledge_distill:
            if args.rdrop_alpha > 0.0:
                epoch_iterator.set_description("Epoch: %0.4f/%0.4f/%0.4f, Batch: %0.4f/%0.4f/%0.4f" % \
                                           (epoch_loss / (batch_idx + 1), epoch_kl_loss/(batch_idx + 1), epoch_c_loss/(batch_idx + 1), \
                                            batch_loss, batch_kl_loss, batch_c_loss)
                                          )
            else:
                epoch_iterator.set_description("Epoch: %0.4f, Batch: %0.4f" % (epoch_loss / (batch_idx + 1), batch_loss))
        else:
            epoch_iterator.set_description("E_loss: %0.4f, B_loss: %0.4f, E_teach: %0.4f, B_teach: %0.4f, E_total: %0.4f, B_total: %0.4f" % \
                                           (epoch_loss / (batch_idx + 1), batch_loss, epoch_teacher / (batch_idx + 1), batch_teacher, epoch_total / (batch_idx + 1), batch_total))

        if (steps + 1) % args.gradient_accumulation_steps == 0:
            # torch.nn.utils.clip_grad_norm_(
            #     model.parameters(), args.max_grad_norm)
            accelerator.clip_grad_norm_(
                 model.parameters(), args.max_grad_norm)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()  # Update learning rate schedule
            model.zero_grad()

        steps += 1

    tqdm_bar = False
    accelerator.wait_for_everyone()
    if accelerator.is_local_main_process:
        dev_metric, _, threshold = eval_func(model, dev_dataloader, accelerator.device, None, tqdm_bar, args)
        print('Threshold find on dev:', threshold)
        test_metric, _, _ = eval_func(model, test_dataloader, accelerator.device, threshold, tqdm_bar, args)
    else:
        dev_metric = None
        test_metric = None
    return dev_metric, test_metric, steps

def predict(model, dataloader, device, threshold=None, tqdm_bar=None, args=None):
    model.eval()
    outputs = []
    device = args.device if args is not None else device
    it = tqdm(dataloader) if tqdm_bar else dataloader
    with torch.no_grad():
        if isinstance(model, DistributedDataParallel):
                model.module.calculate_label_hidden()
        else:
            model.calculate_label_hidden()
        for batch in it:
            batch_gpu = tuple([x.to(device) for x in batch])
            if isinstance(model, DistributedDataParallel):
                now_res = model.module.predict(batch_gpu, threshold)
            else:
                now_res = model.predict(batch_gpu, threshold)
            outputs.append({key:value.cpu().detach() for key, value in now_res.items()})
            
    yhat = torch.cat([output['yhat'] for output in outputs]).cpu().detach().numpy()
    yhat_raw = torch.cat([output['yhat_raw'] for output in outputs]).cpu().detach().numpy()
    y = torch.cat([output['y'] for output in outputs]).cpu().detach().numpy()
    return yhat, y, yhat_raw

def eval_func(model, dataloader, device, threshold=None, tqdm_bar=False, args=None):
    yhat, y, yhat_raw = predict(model, dataloader, device, threshold, tqdm_bar, args)
    if threshold is None:
        threshold = find_threshold_micro(yhat_raw, y)
    yhat = np.where(yhat_raw > threshold, 1, 0)
    metric = all_metrics(yhat=yhat, y=y, yhat_raw=yhat_raw)
    return metric, (yhat, y, yhat_raw), threshold

def main():
    parser = generate_parser()
    args = parser.parse_args()
    run(args)

if __name__ == "__main__":
    main()
