import sys
import os.path
import argparse
import math
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from tqdm import tqdm

import config
import data
import model
import utils

from tensorboardX import SummaryWriter
exp_setting = input("What's new in this experiment......")
print('Experiment Setting: ', exp_setting)
writer = SummaryWriter('runs/'+exp_setting)

def run(net, loader, optimizer, scheduler, tracker, train=False, has_answers=True, prefix='', epoch=0):
    """ Run an epoch over the given loader """
    assert not (train and not has_answers)
    if train:
        net.train()
        tracker_class, tracker_params = tracker.MovingMeanMonitor, {'momentum': 0.99}
    else:
        net.eval()
        tracker_class, tracker_params = tracker.MeanMonitor, {}
        answ = []
        idxs = []
        accs = []

    loader = tqdm(loader, desc='{} E{:03d}'.format(prefix, epoch), ncols=0)
    loss_tracker = tracker.track('{}_loss'.format(prefix), tracker_class(**tracker_params))
    acc_tracker = tracker.track('{}_acc'.format(prefix), tracker_class(**tracker_params))
    batch_count = 0
    batch_max = len(loader)
    for v, q, a, b, q_type, idx, q_len in loader:
        var_params = {
            'volatile': not train,
            'requires_grad': False,
        }
        v = Variable(v.cuda(async=True), **var_params)
        q = Variable(q.cuda(async=True), **var_params)
        a = Variable(a.cuda(async=True), **var_params)
        b = Variable(b.cuda(async=True), **var_params)
        q_len = Variable(q_len.cuda(async=True), **var_params)
        q_type = Variable(q_type.cuda(async=True), **var_params)

        if config.use_rl and train:
            net.eval()
            out, _, _ = net(v, b, q, q_len, q_type)
            acc = utils.batch_accuracy(out.data, a.data).cpu()
            baseline = []
            for i in range(acc.shape[0]):
                baseline.append(float(acc[i]))
            #float(acc.mean())
            net.train()
            utils.fix_batchnorm(net)
            out, rl_ls, _ = net(v, b, q, q_len, q_type)
            acc = utils.batch_accuracy(out.data, a.data).cpu()
            current = []
            for i in range(acc.shape[0]):
                current.append(float(acc[i]))
            #float(acc.mean())
            #print(baseline - current)
            rl_loss = []
            assert len(rl_ls) == len(baseline)
            for i in range(len(rl_ls)):
                rl_loss.append((baseline[i] - current[i]) * rl_ls[i])
            #(baseline - current) * sum(rl_ls) / len(rl_ls)
            #entropy_loss = sum(entropy_ls) / len(entropy_ls) * 1e-4
            loss = sum(rl_loss) #+ entropy_loss
        else:
            out, _, _ = net(v, b, q, q_len, q_type)
            if has_answers:
                nll = -F.log_softmax(out, dim=1)
                loss = (nll * a / 10).sum(dim=1).mean()
                acc = utils.batch_accuracy(out.data, a.data).cpu()

        if train:
            scheduler.step()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        else:
            # store information about evaluation of this minibatch
            _, answer = out.data.cpu().max(dim=1)
            answ.append(answer.view(-1))
            if has_answers:
                accs.append(acc.view(-1))
            idxs.append(idx.view(-1).clone())

        if has_answers:
            loss_tracker.append(loss.item())
            acc_tracker.append(acc.mean())
            fmt = '{:.4f}'.format
            loader.set_postfix(loss=fmt(loss_tracker.mean.value), acc=fmt(acc_tracker.mean.value))
        
        if train:
            writer.add_scalar('train/loss', loss.item(), epoch * batch_max + batch_count)
            writer.add_scalar('train/accu', acc.mean(), epoch * batch_max + batch_count)
            #writer.export_scalars_to_json("./log_board.json")
        else:
            writer.add_scalar('val/loss', loss.item(), epoch * batch_max + batch_count)
            writer.add_scalar('val/accu', acc.mean(), epoch * batch_max + batch_count)
            #writer.export_scalars_to_json("./log_board.json")
        batch_count += 1

    if not train:
        answ = list(torch.cat(answ, dim=0))
        if has_answers:
            accs = list(torch.cat(accs, dim=0))
        else:
            accs = []
        idxs = list(torch.cat(idxs, dim=0))
        return answ, accs, idxs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('name', nargs='*')
    parser.add_argument('--eval', dest='eval_only', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--resume', nargs='*')
    args = parser.parse_args()

    if args.test:
        args.eval_only = True
    src = open('model.py').read()
    if args.name:
        name = ' '.join(args.name)
    else:
        from datetime import datetime
        name = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    target_name = os.path.join('logs', '{}'.format(name))
    writer.add_text('Log Name:', name)
    if not args.test:
        # target_name won't be used in test mode
        print('will save to {}'.format(target_name))
    if args.resume:
        logs = torch.load(' '.join(args.resume))
        # hacky way to tell the VQA classes that they should use the vocab without passing more params around
        #data.preloaded_vocab = logs['vocab']

    cudnn.benchmark = True

    if not args.eval_only:
        train_loader = data.get_loader(train=True)
    if not args.test:
        val_loader = data.get_loader(val=True)
    else:
        val_loader = data.get_loader(test=True)

    net = model.Net(val_loader.dataset.num_tokens).cuda()
    # restore transfer learning
    # 'data/vgrel-29.tar' for 36
    # 'data/vgrel-19.tar' for 10-100
    if config.output_size == 36:
        print("load data/vgrel-29(transfer36).tar")
        ckpt = torch.load('data/vgrel-29(transfer36).tar')
    else:
        print("load data/vgrel-19(transfer110).tar")
        ckpt = torch.load('data/vgrel-19(transfer110).tar')
    
    utils.optimistic_restore(net.tree_lstm.gen_tree_net, ckpt['state_dict'])
    
    if config.use_rl:
        for p in net.parameters():
            p.requires_grad = False
        for p in net.tree_lstm.gen_tree_net.parameters():
            p.requires_grad = True
    
    optimizer = optim.Adam([p for p in net.parameters() if p.requires_grad], lr=config.initial_lr)
    scheduler = lr_scheduler.ExponentialLR(optimizer, 0.5**(1 / config.lr_halflife))
    start_epoch = 0
    if args.resume:
        net.load_state_dict(logs['weights'])
        #optimizer.load_state_dict(logs['optimizer'])
        start_epoch = int(logs['epoch']) + 1

    tracker = utils.Tracker()
    config_as_dict = {k: v for k, v in vars(config).items() if not k.startswith('__')}
    print(config_as_dict)
    best_accuracy = -1

    for i in range(start_epoch, config.epochs):
        if not args.eval_only:
            run(net, train_loader, optimizer, scheduler, tracker, train=True, prefix='train', epoch=i)
        if i % 1 != 0 or (i > 0 and i <20):
            r = [[-1], [-1], [-1]]
        else:
            r = run(net, val_loader, optimizer, scheduler, tracker, train=False, prefix='val', epoch=i, has_answers=not args.test)

        if not args.test:
            results = {
                'name': name,
                'tracker': tracker.to_dict(),
                'config': config_as_dict,
                'weights': net.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': i,
                'eval': {
                    'answers': r[0],
                    'accuracies': r[1],
                    'idx': r[2],
                },
                'vocab': val_loader.dataset.vocab,
                'src': src,
                'setting': exp_setting,
            }
            current_ac = sum(r[1]) / len(r[1])
            if current_ac >  best_accuracy:
                best_accuracy = current_ac
                print('update best model, current: ', current_ac)
                torch.save(results, target_name + '_best.pth')
            if i % 1 == 0:
                torch.save(results, target_name + '_' + str(i) + '.pth')

        else:
            # in test mode, save a results file in the format accepted by the submission server
            answer_index_to_string = {a:  s for s, a in val_loader.dataset.answer_to_index.items()}
            results = []
            for answer, index in zip(r[0], r[2]):
                answer = answer_index_to_string[answer.item()]
                qid = val_loader.dataset.question_ids[index]
                entry = {
                    'question_id': qid,
                    'answer': answer,
                }
                results.append(entry)
            with open('results.json', 'w') as fd:
                json.dump(results, fd)

        if args.eval_only:
            break


if __name__ == '__main__':
    main()
    writer.close()
    writer.export_scalars_to_json("./log_board.json")
