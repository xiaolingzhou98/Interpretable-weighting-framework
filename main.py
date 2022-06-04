import os
import copy
import argparse
from sympy import arg
import torch.optim
from meta import *
from model import *
from NLT_CIFAR import *
import torch.nn.functional as F
from loguru import logger
import csv

parser = argparse.ArgumentParser(description='Explainable_weighting_framework')
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--meta_net_hidden_size', type=int, default=100)
parser.add_argument('--meta_net_num_layers', type=int, default=1)

parser.add_argument('--lr', type=float, default=.1)
parser.add_argument('--momentum', type=float, default=.9)
parser.add_argument('--dampening', type=float, default=0.)
parser.add_argument('--nesterov', type=bool, default=False)
parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--meta_lr', type=float, default=1e-4)    
parser.add_argument('--meta_batch', type=int, default=100)   
parser.add_argument('--meta_weight_decay', type=float, default=5e-4)

parser.add_argument('--dataset', type=str, default='cifar10')
parser.add_argument('--num_meta', type=int, default=1000)
parser.add_argument('--imbalanced_factor', type=int, default=1)
parser.add_argument('--corruption_type', type=str, default="flip2")
parser.add_argument('--corruption_ratio', type=float, default=0.)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--max_epoch', type=int, default=200)

parser.add_argument('--meta_interval', type=int, default=20)
parser.add_argument('--paint_interval', type=int, default=20)
parser.add_argument('--sub_rate', type=float, default=0.2)
parser.add_argument('--add_rate', type=float, default=0.1)
parser.add_argument('--record_num', type=int, default=5)
parser.add_argument('--temperature', type=float, default=0.0001)
args = parser.parse_args()
logger.info(args)


def weight_feature(logits_vector,loss_vector,labels,index,weight_tensor_last1,weight_tensor_last2,loss_tensor_last, class_rate):
    labels_one_hot = F.one_hot(labels,num_classes=(args.dataset == 'cifar10' and 10 or 100)).float()
    class_rate = torch.mm(labels_one_hot,class_rate.unsqueeze(1))
    last_epoch_ave_loss = torch.mm(labels_one_hot,loss_tensor_last.unsqueeze(1))
    logits_labels = torch.sum(F.softmax(logits_vector,dim=1) * labels_one_hot,dim=1)
    logits_vector_grad = torch.norm(1- F.softmax(logits_vector,dim=1),dim=1)    
    logits_others_max =(F.softmax(logits_vector,dim=1)[labels_one_hot!=1].reshape(F.softmax(logits_vector,dim=1).size(0),-1)).max(dim=1).values
    logits_margin =  logits_labels - logits_others_max
    weight_last1 = weight_tensor_last1[index]
    weight_last2 = weight_tensor_last2[index]
    weight_sub = weight_last1 - weight_last2
    entropy =  torch.sum(F.softmax(logits_vector,dim=1)*F.log_softmax(logits_vector,dim=1),dim=1)
    
    feature = torch.cat([loss_vector.unsqueeze(1),
                        last_epoch_ave_loss,
                        logits_vector_grad.unsqueeze(1),
                        logits_margin.unsqueeze(1),
                        weight_last1.unsqueeze(1),
                        weight_last2.unsqueeze(1),
                        weight_sub.unsqueeze(1),
                        entropy.unsqueeze(1),
                        class_rate],dim=1)
    return feature


def compute_loss_accuracy(net, data_loader, criterion, device):
    net.eval()
    correct = 0
    total_loss = 0.
    

    with torch.no_grad():
        for batch_idx, (_,inputs, labels) in enumerate(data_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            total_loss += criterion(outputs, labels).item()
            _, pred = outputs.max(1)
            correct += pred.eq(labels).sum().item()

    return total_loss / (batch_idx + 1), correct / len(data_loader.dataset)


def explainable_weighting_framework():
    # meta_net = MLP(in_size = 9,hidden_size=args.meta_net_hidden_size, num_layers=args.meta_net_num_layers).to(device=args.device)
    meta_net = PRUNNRT(num_feature=9,temperature = args.temperature, mask_max=4).to(device=args.device)
    net = ResNet32(args.dataset == 'cifar10' and 10 or 100).to(device=args.device)
    criterion = nn.CrossEntropyLoss().to(device=args.device)
    optimizer = torch.optim.SGD(
        net.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        dampening=args.dampening,
        weight_decay=args.weight_decay,
        nesterov=args.nesterov,
    )
    meta_optimizer = torch.optim.Adam(meta_net.cut_points_list + [meta_net.leaf_score], lr=args.meta_lr, weight_decay=args.meta_weight_decay)
    lr = args.lr

    train_dataloader, meta_dataloader, test_dataloader, imbalanced_num_list = build_dataloader(
        seed=args.seed,
        dataset=args.dataset,
        num_meta_total=args.num_meta,
        imbalanced_factor=args.imbalanced_factor,
        corruption_type=args.corruption_type,
        corruption_ratio=args.corruption_ratio,
        batch_size=args.batch_size,
        meta_batch=args.meta_batch,
    )
    class_rate = torch.from_numpy(np.array(imbalanced_num_list)/sum(imbalanced_num_list)).cuda().float()
    meta_dataloader_iter = iter(meta_dataloader)
    iteration = 0
    best_acc = 0.
    
    # create weight dict 
    weight_tensor_current = torch.ones([50000]).cuda()
    weight_tensor_last1 = torch.ones([50000]).cuda()
    weight_tensor_last2 = torch.ones([50000]).cuda()
    loss_tensor_last = torch.ones([args.dataset == 'cifar10' and 10 or 100]).cuda()*5

    for epoch in range(args.max_epoch):
        loss_total = torch.zeros([args.dataset == 'cifar10' and 10 or 100]).cuda()
        class_total = torch.zeros([args.dataset == 'cifar10' and 10 or 100]).cuda()
        lr = args.lr * ((0.1 ** int(epoch >= 120)) * (0.1 ** int(epoch >= 160)))
       
        # if epoch >= 120 and epoch % 40 == 0:
        #     lr = lr / 10
        for group in optimizer.param_groups:
            group['lr'] = lr

        logger.info(f'Training epoch {epoch}')
        for iteration, (index, inputs, labels) in enumerate(train_dataloader):
            net.train()
            inputs, labels = inputs.to(args.device), labels.to(args.device)

            if (iteration + 1) % args.meta_interval == 0:
                pseudo_net = ResNet32(args.dataset == 'cifar10' and 10 or 100).to(args.device)
                pseudo_net.load_state_dict(net.state_dict())
                pseudo_net.train()
                
                pseudo_outputs = pseudo_net(inputs)
                pseudo_loss_vector = F.cross_entropy(pseudo_outputs, labels.long(), reduction='none')
                pseudo_feature = weight_feature(pseudo_outputs,pseudo_loss_vector,labels,index,weight_tensor_last1,weight_tensor_last2,loss_tensor_last, class_rate)
                pseudo_weight = meta_net(pseudo_feature)
                pseudo_loss = torch.mean(pseudo_weight * torch.reshape(pseudo_loss_vector, (-1, 1)))
                pseudo_grads = torch.autograd.grad(pseudo_loss, pseudo_net.parameters(), create_graph=True)
                pseudo_optimizer = MetaSGD(pseudo_net, pseudo_net.parameters(), lr=lr)
                pseudo_optimizer.load_state_dict(optimizer.state_dict())
                pseudo_optimizer.meta_step(pseudo_grads)

                del pseudo_grads

                try:
                    _, meta_inputs, meta_labels = next(meta_dataloader_iter)
                except StopIteration:
                    meta_dataloader_iter = iter(meta_dataloader)
                    _, meta_inputs, meta_labels = next(meta_dataloader_iter)

                meta_inputs, meta_labels = meta_inputs.to(args.device), meta_labels.to(args.device)
                meta_outputs = pseudo_net(meta_inputs)
                meta_loss = criterion(meta_outputs, meta_labels.long())
                
                

                meta_optimizer.zero_grad()
                meta_loss.backward()
                meta_optimizer.step()

            outputs = net(inputs)
            loss_vector = F.cross_entropy(outputs, labels.long(), reduction='none')
            loss_vector_reshape = torch.reshape(loss_vector, (-1, 1))

            with torch.no_grad():
                pseudo_feature = weight_feature(outputs,loss_vector,labels,index,weight_tensor_last1,weight_tensor_last2,loss_tensor_last,class_rate)
                weights = meta_net(pseudo_feature)
            
            weight_tensor_current.scatter_(0, index.cuda(), weights.squeeze(1))
            
            # loss_tensor_last
            labels_one_hot = F.one_hot(labels,num_classes=args.dataset == 'cifar10' and 10 or 100).float()
            class_total += torch.sum(labels_one_hot,dim=0)
            loss_total += torch.mm(loss_vector.detach().unsqueeze(0),labels_one_hot)[0]
            
            loss = torch.mean(weights * loss_vector_reshape)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            

        
        weight_tensor_last2 = copy.deepcopy(weight_tensor_last1)
        weight_tensor_last1 = copy.deepcopy(weight_tensor_current)
        loss_tensor_last = loss_total/(class_total + 1e-6)
        
        # pruning and growth
        meta_net.record_points(epoch=epoch)
        meta_net.statistics()
        
        logger.info('Computing Test Result...')

        test_loss, test_accuracy = compute_loss_accuracy(
            net=net,
            data_loader=test_dataloader,
            criterion=criterion,
            device=args.device,
        )
        #write_source_numpy = list(write_source_numpy)
        #print(write_source_numpy)
        
        if test_accuracy>best_acc:
            best_acc = test_accuracy
            torch.save(net.state_dict(),"best_net.pt")
            torch.save(meta_net.state_dict(),"best_meta.pt")
            
        logger.info('Epoch: {}, (Loss, Accuracy) Test: ({:.4f}, {:.2%}) LR: {}'.format(
            epoch,
            test_loss,
            test_accuracy,
            lr,
        ))
    logger.info(f'best_accuracy: {best_acc}')   
    os.rename("best_net.pt","best_net_"+str(best_acc)+"_"+str(args.corruption_type)+"_"+str(args.corruption_ratio)+"_"+str(args.imbalanced_factor)+".pt")
    os.rename("best_meta.pt","best_meta_"+str(best_acc)+"_"+str(args.corruption_type)+"_"+str(args.corruption_ratio)+"_"+str(args.imbalanced_factor)+".pt")

if __name__ == '__main__':
    explainable_weighting_framework()
