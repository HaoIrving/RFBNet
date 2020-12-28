from __future__ import print_function
import sys
import os
import pickle
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import numpy as np
from torch.autograd import Variable
from data import VOCroot,COCOroot 
from data import AnnotationTransform, COCODetection, VOCDetection, BaseTransform, VOC_300,VOC_512,COCO_300,COCO_512, COCO_mobile_300

import torch.utils.data as data
from layers.functions import Detect,PriorBox
from utils.nms_wrapper import nms
from utils.timer import Timer

import cv2

parser = argparse.ArgumentParser(description='Receptive Field Block Net')

parser.add_argument('-v', '--version', default='RFB_vgg',
                    help='RFB_vgg ,RFB_E_vgg or RFB_mobile version.')
parser.add_argument('-s', '--size', default='512',
                    help='300 or 512 input size.')
parser.add_argument('-d', '--dataset', default='COCO',
                    help='VOC or COCO version')
parser.add_argument('-m', '--trained_model', default='weights/RFB300_80_5.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--save_folder', default='eval/', type=str,
                    help='Dir to save results')
parser.add_argument('--cuda', default=True, type=bool,
                    help='Use cuda to train model')
parser.add_argument('--cpu', default=False, type=bool,
                    help='Use cpu nms')
parser.add_argument('--retest', default=False, type=bool,
                    help='test cache results')
parser.add_argument('--show_image', action="store_true", default=False, help='show detection results')
parser.add_argument('--vis_thres', default=0.5, type=float, help='visualization_threshold')
parser.add_argument('--confidence_threshold', default=0.05, type=float, help='confidence_threshold')
parser.add_argument('--nms_threshold', default=0.3, type=float, help='nms_threshold')
args = parser.parse_args()

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)

if args.dataset == 'VOC':
    cfg = (VOC_300, VOC_512)[args.size == '512']
else:
    cfg = (COCO_300, COCO_512)[args.size == '512']

if args.version == 'RFB_vgg':
    from models.RFB_Net_vgg import build_net
elif args.version == 'RFB_E_vgg':
    from models.RFB_Net_E_vgg import build_net
elif args.version == 'RFB_mobile':
    from models.RFB_Net_mobile import build_net
    cfg = COCO_mobile_300
else:
    print('Unkown version!')

priorbox = PriorBox(cfg)
with torch.no_grad():
    priors = priorbox.forward()
    if args.cuda:
        priors = priors.cuda()


def test_net(save_folder, net, detector, cuda, testset, transform, top_k, max_per_image=300, confidence_threshold=0.005, nms_threshold=0.4, AP_stats=None):

    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    # dump predictions and assoc. ground truth to text file for now
    num_images = len(testset)
    # num_classes = (21, 81)[args.dataset == 'COCO']
    num_classes = 2
    all_boxes = [[[] for _ in range(num_images)]
                 for _ in range(num_classes)]

    _t = {'im_detect': Timer(), 'misc': Timer()}
    det_file = os.path.join(save_folder, 'detections.pkl')

    if args.retest:
        f = open(det_file,'rb')
        all_boxes = pickle.load(f)
        print('Evaluating detections')
        testset.evaluate_detections(all_boxes, save_folder)
        return


    for i in range(num_images):
        img = testset.pull_image(i)
        scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        with torch.no_grad():
            x = transform(img).unsqueeze(0)
            if cuda:
                x = x.cuda()
                scale = scale.cuda()

        _t['im_detect'].tic()
        out = net(x)      # forward pass
        boxes, scores = detector.forward(out,priors)
        _t['im_detect'].toc()
        boxes = boxes[0]
        scores=scores[0]

        boxes *= scale
        boxes = boxes.cpu().numpy()
        scores = scores.cpu().numpy()
        # scale each detection back up to the image

        _t['misc'].tic()

        for j in range(1, num_classes):
            inds = np.where(scores[:, j] > confidence_threshold)[0]
            if len(inds) == 0:
                all_boxes[j][i] = np.empty([0, 5], dtype=np.float32)
                continue
            c_bboxes = boxes[inds]
            c_scores = scores[inds, j]

            # keep top-K before NMS
            order = c_scores.argsort()[::-1][:top_k]
            c_bboxes = c_bboxes[order]
            c_scores = c_scores[order]

            c_dets = np.hstack((c_bboxes, c_scores[:, np.newaxis])).astype(
                np.float32, copy=False)

            keep = nms(c_dets, nms_threshold, force_cpu=args.cpu)
            c_dets = c_dets[keep, :]
            all_boxes[j][i] = c_dets
        if max_per_image > 0:
            image_scores = np.hstack([all_boxes[j][i][:, -1] for j in range(1,num_classes)])
            if len(image_scores) > max_per_image:
                image_thresh = np.sort(image_scores)[-max_per_image]
                for j in range(1, num_classes):
                    keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
                    all_boxes[j][i] = all_boxes[j][i][keep, :]

        _t['misc'].toc()

        # print('im_detect: {:d}/{:d} forward_pass_time: {:.4f}s misc: {:.4f}s'.format(i + 1, num_images, _t['im_detect'].average_time, _t['misc'].average_time))
        
        if args.show_image:
            img_gt = img.astype(np.uint8)
            for b in all_boxes[1][i]:
                if b[4] < args.vis_thres:
                    continue
                text = "{:.4f}".format(b[4])
                b = list(map(int, b))
                cv2.rectangle(img_gt, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
                cx = b[0]
                cy = b[1] + 12
                cv2.putText(img_gt, text, (cx, cy),
                            cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))
            cv2.imshow('res', img_gt)
            cv2.waitKey(0)

    # with open(det_file, 'wb') as f:
    #     pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

    print('Evaluating detections')
    stats = testset.evaluate_detections(all_boxes, save_folder)
    AP_stats['ap50'].append(stats[1])
    AP_stats['ap_small'].append(stats[3])
    AP_stats['ap_medium'].append(stats[4])
    AP_stats['ap_large'].append(stats[5])


if __name__ == '__main__':
    from collections import OrderedDict
    # load data
    if args.dataset == 'VOC':
        testset = VOCDetection(
            VOCroot, [('2007', 'test')], None, AnnotationTransform())
    elif args.dataset == 'COCO':
        testset = COCODetection(COCOroot, [('sarship', 'test')], None)
    else:
        print('Only VOC and COCO dataset are supported now!')
    
    # args.show_image = True
    # args.vis_thres = 0.1
    # args.retest = True

    args.confidence_threshold = 0.01
    args.nms_threshold = 0.5
    # args.nms_threshold = 0.1
    # evaluation
    top_k = 1000
    keep_top_k = 500
    save_folder = os.path.join(args.save_folder, args.dataset)
    rgb_means = (98.13131, 98.13131, 98.13131)
    # load net
    img_dim = (300,512)[args.size=='512']
    num_classes = 2
    detector = Detect(num_classes,0,cfg)
    net = build_net('test', img_dim, num_classes)    # initialize detector
    
    ap_stats = {"ap50": [], "ap_small": [], "ap_medium": [], "ap_large": [], "epoch": []}

    start_epoch = 100; step = 10
    ToBeTested = ['weights/RFB_vgg_COCO_epoches_100.pth']
    # ToBeTested = [f'weights/RFB_vgg_COCO_epoches_{epoch}.pth' for epoch in range(start_epoch, 300, step)]
    ToBeTested.append('weights/Final_RFB_vgg_COCO.pth') # 68.5
    for index, model_path in enumerate(ToBeTested):
        args.trained_model = model_path
        state_dict = torch.load(args.trained_model)
        # create new OrderedDict that does not contain `module.`
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            head = k[:7]
            if head == 'module.':
                name = k[7:] # remove `module.`
            else:
                name = k
            new_state_dict[name] = v
        net.load_state_dict(new_state_dict)
        net.eval()
        print('Finished loading model!')
        # print(net)
        if args.cuda:
            net = net.cuda()
            cudnn.benchmark = True
        else:
            net = net.cpu()
        
        ap_stats['epoch'].append(start_epoch + index * step)
        print("evaluating epoch: {}".format(ap_stats['epoch'][-1]))
        test_net(save_folder, net, detector, args.cuda, testset,
                BaseTransform(net.size, rgb_means, (2, 0, 1)), top_k, 
                keep_top_k, confidence_threshold=args.confidence_threshold, nms_threshold=args.nms_threshold, AP_stats=ap_stats)
    print(ap_stats)
    if len(ap_stats['epoch']) != 0:
        from plot_curve import plot_map
        plot_map(ap_stats['ap50'])