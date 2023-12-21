#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import sys
import logging
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.modules.loss import CrossEntropyLoss
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import random
import time
import torch, gc
import numpy as np
from tqdm import tqdm
from medpy.metric import dc,hd95
from scipy.ndimage import zoom

from utils.utils import powerset
from utils.utils import DiceLoss, calculate_dice_percase, val_single_volume
from utils.dataset_ACDC import ACDCdataset, RandomGenerator
from test_ACDC import inference
from lib.networks import MIST_CAM


# In[2]:


gc.collect()
torch.cuda.empty_cache()
os.environ['CUDA_VISIBLE_DEVICES']='0, 1'


# In[3]:


parser = argparse.ArgumentParser(description='Searching longest common substring. '
                    'Uses Ukkonen\'s suffix tree algorithm and generalized suffix tree. '
                    'Written by Ilya Stepanov (c) 2013')
parser.add_argument('strings', metavar='STRING', nargs='*', help='String for searching',)
parser.add_argument("--batch_size", default=12, help="batch size") #12
parser.add_argument("--lr", default=0.0001, help="learning rate")
parser.add_argument("--max_epochs", default=400)
parser.add_argument("--img_size", default=256)
parser.add_argument("--save_path", default="./model_pth/data")
parser.add_argument("--n_gpu", default=1)
parser.add_argument("--checkpoint", default=None)
parser.add_argument("--list_dir", default="./data/ACDC/lists_ACDC")
parser.add_argument("--root_dir", default="./data/ACDC/")
parser.add_argument("--volume_path", default="./data/ACDC/test")
parser.add_argument("--z_spacing", default=10)
parser.add_argument("--num_classes", default=4)
parser.add_argument('--test_save_dir', default='./predictions', help='saving prediction as nii!')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--seed', type=int,
                    default=2222, help='random seed')            
args = parser.parse_args("AAA".split())


# In[4]:


if not args.deterministic:
    cudnn.benchmark = True
    cudnn.deterministic = False
else:
    cudnn.benchmark = False
    cudnn.deterministic = True
    
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

args.is_pretrain = True
args.exp = 'MIST_CAM_loss_MUTATION_w3_7_' + str(args.img_size)
snapshot_path = "{}/{}/{}".format(args.save_path, args.exp, 'MIST_CAM_loss_MUTATION_w3_7')
snapshot_path = snapshot_path + '_pretrain' if args.is_pretrain else snapshot_path
snapshot_path = snapshot_path + '_epo' +str(args.max_epochs) if args.max_epochs != 30 else snapshot_path
snapshot_path = snapshot_path+'_bs'+str(args.batch_size)
snapshot_path = snapshot_path + '_lr' + str(args.lr) if args.lr != 0.01 else snapshot_path
snapshot_path = snapshot_path + '_'+str(args.img_size)
snapshot_path = snapshot_path + '_s'+str(args.seed) if args.seed!=1234 else snapshot_path

current_time = time.strftime("%H%M%S")
print("The current time is", current_time)
snapshot_path = snapshot_path +'_run'+current_time
    
if not os.path.exists(snapshot_path):
    os.makedirs(snapshot_path)

args.test_save_dir = os.path.join(snapshot_path, args.test_save_dir)
test_save_path = os.path.join(args.test_save_dir, args.exp)
if not os.path.exists(test_save_path):
    os.makedirs(test_save_path, exist_ok=True)


# In[5]:


net = MIST_CAM(n_class=args.num_classes, img_size_s1=(args.img_size,args.img_size), img_size_s2=(224,224), model_scale='small', decoder_aggregation='additive', interpolation='bilinear').cuda()

if args.checkpoint:
    net.load_state_dict(torch.load(args.checkpoint))

train_dataset = ACDCdataset(args.root_dir, args.list_dir, split="train", transform=
                                   transforms.Compose(
                                   [RandomGenerator(output_size=[args.img_size, args.img_size])]))
print("The length of train set is: {}".format(len(train_dataset)))
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
db_val=ACDCdataset(base_dir=args.root_dir, list_dir=args.list_dir, split="valid")
valloader=DataLoader(db_val, batch_size=1, shuffle=False)
db_test =ACDCdataset(base_dir=args.volume_path,list_dir=args.list_dir, split="test")
testloader = DataLoader(db_test, batch_size=1, shuffle=False)

if args.n_gpu > 1:
    net = nn.DataParallel(net)


# In[6]:


net = net.cuda()
net.train()
ce_loss = CrossEntropyLoss()
dice_loss = DiceLoss(args.num_classes)
#save_interval = args.n_skip

# iterator = tqdm(range(0, args.max_epochs), ncols=70)
# iter_num = 0

Loss = []
Test_Accuracy = []

Best_dcs = 0.80
Best_dcs_th = 0.865

logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')

max_iterations = args.max_epochs * len(train_loader)
base_lr = args.lr
optimizer = optim.AdamW(net.parameters(), lr=base_lr, weight_decay=0.0001)
#optimizer = optim.SGD(net.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)


# In[7]:


def val():
    logging.info("Validation ===>")
    dc_sum=0
    metric_list = 0.0
    net.eval()
    for i, val_sampled_batch in enumerate(valloader):
        val_image_batch, val_label_batch = val_sampled_batch["image"], val_sampled_batch["label"]

        val_image_batch, val_label_batch = val_image_batch.squeeze(0).cpu().detach().numpy(), val_label_batch.squeeze(0).cpu().detach().numpy()

        x, y = val_image_batch.shape[0], val_image_batch.shape[1]
        if x != args.img_size or y != args.img_size:
            val_image_batch = zoom(val_image_batch, (args.img_size / x, args.img_size / y), order=3) # not for double_maxvits
        val_image_batch = torch.from_numpy(val_image_batch).unsqueeze(0).unsqueeze(0).float().cuda()
        
        P = net(val_image_batch)
        #print(len(P))

        val_outputs = 0.0
        for idx in range(len(P)):
            val_outputs += P[idx]
        
        val_outputs = torch.softmax(val_outputs, dim=1)

        val_outputs = torch.argmax(val_outputs, dim=1).squeeze(0)
        val_outputs = val_outputs.cpu().detach().numpy()
        if x != args.img_size or y != args.img_size:
            val_outputs = zoom(val_outputs, (x / args.img_size, y / args.img_size), order=0)
        else:
            val_outputs = val_outputs

        dc_sum+=dc(val_outputs,val_label_batch[:])
    performance = dc_sum / len(valloader)
    logging.info('Testing performance in val model: mean_dice : %f, best_dice : %f' % (performance, Best_dcs))

    print('Testing performance in val model: mean_dice : %f, best_dice : %f' % (performance, Best_dcs))
    #print("val avg_dsc: %f" % (performance))
    return performance


# In[ ]:


l = [0, 1, 2, 3]
ss = [x for x in powerset(l)] # for mutation
#ss = [[0],[1],[2],[3]] # for only four-stage loss, no mutation
#print(ss)
iter_num = 0
best_state_dict=net.state_dict()
for epoch in tqdm(range(args.max_epochs)):
    net.train()
    train_loss = 0
    for i_batch, sampled_batch in enumerate(train_loader):
        image_batch, label_batch = sampled_batch["image"], sampled_batch["label"]
        image_batch, label_batch = image_batch.type(torch.FloatTensor), label_batch.type(torch.FloatTensor)
        image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
        
        P = net(image_batch)
        loss = 0.0
        lc1, lc2 = 0.3, 0.7
                  
        for s in ss:
            iout = 0.0
            if(s==[]):
                continue
            #print(s)
            for idx in range(len(s)):
                iout += P[s[idx]]
            loss_ce = ce_loss(iout, label_batch[:].long())
            loss_dice = dice_loss(iout, label_batch, softmax=True)
            loss += (lc1 * loss_ce + lc2 * loss_dice) 
           
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9 # We did not use this
        lr_ = base_lr
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_

        iter_num = iter_num + 1
        if iter_num%50 == 0:
            #logging.info('iteration %d : loss : %f lr_: %f' % (iter_num, loss.item(), lr_))
            print('iteration %d : loss : %f lr_: %f' % (iter_num, loss.item(), lr_))
        train_loss += loss.item()
    Loss.append(train_loss/len(train_dataset))
    #logging.info('iteration %d : loss : %f lr_: %f' % (iter_num, loss.item(), lr_))
    print('iteration %d : loss : %f lr_: %f' % (iter_num, loss.item(), lr_))
    
    save_model_path = os.path.join(snapshot_path, 'last.pth')
    torch.save(net.state_dict(), save_model_path)

    
    avg_dcs = val()
        
    if avg_dcs > Best_dcs:
        #save_model_path = os.path.join(snapshot_path, 'best.pth')
        best_state_dict=net.state_dict()
        #torch.save(net.state_dict(), save_model_path)
        #logging.info("save model to {}".format(save_model_path))
        print("save model to {}".format(save_model_path))

        Best_dcs = avg_dcs
        
#     if avg_dcs > Best_dcs_th or avg_dcs >= Best_dcs:
#         avg_test_dcs, avg_hd, avg_jacard, avg_asd = inference(args, net, testloader, args.test_save_dir)
#         print("test avg_dsc: %f" % (avg_test_dcs))
#         #logging.info("test avg_dsc: %f" % (avg_test_dcs))
#         Test_Accuracy.append(avg_test_dcs)  


    if epoch >= args.max_epochs - 1:
        save_model_path = os.path.join(snapshot_path,  'epoch={}_lr={}_avg_dcs={}.pth'.format(epoch, lr_, avg_dcs))
        torch.save(net.state_dict(), save_model_path)
        #logging.info("save model to {}".format(save_model_path))
        print("save model to {}".format(save_model_path))
        break
save_model_path = os.path.join(snapshot_path, 'best.pth')
torch.save(best_state_dict, save_model_path)


# In[9]:


current_time


# In[11]:


from utils.utils import test_single_volume
def inference(args, model, testloader, test_save_path=None):
    logging.info("{} test iterations per epoch".format(len(testloader)))
    model.eval()
    metric_list = 0.0
    with torch.no_grad():
        for i_batch, sampled_batch in tqdm(enumerate(testloader)):
            h, w = sampled_batch["image"].size()[2:]
            image, label, case_name = sampled_batch["image"], sampled_batch["label"], sampled_batch['case_name'][0]
            metric_i = test_single_volume(image, label, model, classes=args.num_classes, patch_size=[args.img_size, args.img_size],
                                          test_save_path=test_save_path, case=case_name, z_spacing=args.z_spacing)
            metric_list += np.array(metric_i)
            logging.info('idx %d case %s mean_dice %f mean_hd95 %f, mean_jacard %f mean_asd %f' % (i_batch, case_name, np.mean(metric_i, axis=0)[0], np.mean(metric_i, axis=0)[1], np.mean(metric_i, axis=0)[2], np.mean(metric_i, axis=0)[3]))
            print('idx %d case %s mean_dice %f mean_hd95 %f, mean_jacard %f mean_asd %f' % (i_batch, case_name, np.mean(metric_i, axis=0)[0], np.mean(metric_i, axis=0)[1], np.mean(metric_i, axis=0)[2], np.mean(metric_i, axis=0)[3]))
        metric_list = metric_list / len(testloader)
        for i in range(1, args.num_classes):
            logging.info('Mean class (%d) mean_dice %f mean_hd95 %f, mean_jacard %f mean_asd %f' % (i, metric_list[i-1][0], metric_list[i-1][1], metric_list[i-1][2], metric_list[i-1][3]))logging.info('Mean class (%d) mean_dice %f mean_hd95 %f, mean_jacard %f mean_asd %f' % (i, metric_list[i-1][0], metric_list[i-1][1], metric_list[i-1][2], metric_list[i-1][3]))
            print('Mean class (%d) mean_dice %f mean_hd95 %f, mean_jacard %f mean_asd %f' % (i, metric_list[i-1][0], metric_list[i-1][1], metric_list[i-1][2], metric_list[i-1][3]))
        performance = np.mean(metric_list, axis=0)[0]
        mean_hd95 = np.mean(metric_list, axis=0)[1]
        mean_jacard = np.mean(metric_list, axis=0)[2]
        mean_asd = np.mean(metric_list, axis=0)[3]
        logging.info('Testing performance in best val model: mean_dice : %f mean_hd95 : %f, mean_jacard : %f mean_asd : %f' % (performance, mean_hd95, mean_jacard, mean_asd))
        print('Testing performance in best val model: mean_dice : %f mean_hd95 : %f, mean_jacard : %f mean_asd : %f' % (performance, mean_hd95, mean_jacard, mean_asd))
        
        logging.info("Testing Finished!")
        print("Testing Finished!")
        return performance, mean_hd95, mean_jacard, mean_asd

if __name__ == "__main__":

    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    args.is_pretrain = True
    args.exp = 'MIST_CAM_loss_MUTATION_w3_7_' + str(args.img_size)
    snapshot_path = "{}/{}/{}".format(args.save_path, args.exp, 'MIST_CAM_loss_MUTATION_w3_7')
    snapshot_path = snapshot_path + '_pretrain' if args.is_pretrain else snapshot_path
    snapshot_path = snapshot_path + '_epo' +str(args.max_epochs) if args.max_epochs != 30 else snapshot_path
    snapshot_path = snapshot_path+'_bs'+str(args.batch_size)
    snapshot_path = snapshot_path + '_lr' + str(args.lr) if args.lr != 0.01 else snapshot_path
    snapshot_path = snapshot_path + '_'+str(args.img_size)
    snapshot_path = snapshot_path + '_s'+str(args.seed) if args.seed!=1234 else snapshot_path
    
    print("The current time is", current_time)
    snapshot_path = snapshot_path +'_run'+current_time # replace 'current_time' with the 'current_time' in your trained folder from 'model_pth' directory

    net = MIST_CAM(n_class=args.num_classes, img_size_s1=(args.img_size,args.img_size), img_size_s2=(224,224), model_scale='small', decoder_aggregation='additive', interpolation='bilinear').cuda()

    snapshot = os.path.join(snapshot_path, 'best.pth')
    print(snapshot)
    print(os.path.exists(snapshot))
    if not os.path.exists(snapshot): snapshot = snapshot.replace('last', 'epoch_'+str(args.max_epochs-1))
    net.load_state_dict(torch.load(snapshot))
    snapshot_name = snapshot_path.split('/')[-1]

    log_folder = 'test_log/test_log_' + args.exp
    os.makedirs(log_folder, exist_ok=True)
    logging.basicConfig(filename=log_folder + '/'+snapshot_name+".txt", level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    logging.info(snapshot_name)

    args.test_save_dir = os.path.join(snapshot_path, args.test_save_dir)
    test_save_path = os.path.join(args.test_save_dir, args.exp, snapshot_name)
    os.makedirs(test_save_path, exist_ok=True)
    
    
    db_test =ACDCdataset(base_dir=args.volume_path,list_dir=args.list_dir, split="test")
    testloader = DataLoader(db_test, batch_size=1, shuffle=False)
    
    results = inference(args, net, testloader, test_save_path)


# In[ ]:


img=cv2.imread()

