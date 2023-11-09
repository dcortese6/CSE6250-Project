import os
import time
import numpy as np
import torch

from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score


class AverageMeter(object):
	"""Computes and stores the average and current value"""

	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count


# def compute_batch_accuracy(output, target):
#     """Computes the accuracy for a batch"""
#     with torch.no_grad():

#         batch_size = target.size(0)
#         # _, pred = output.max(1)
#         pred = torch.round(output)
#         print(list(output))
#         correct = pred.eq(target).sum()
#         print(correct)

#         return correct * 100.0 / batch_size

def compute_batch_accuracy(output, target, score_metric, threshold=0.5):

    with torch.no_grad():
        pred = np.array(output > threshold, dtype=float)
        mets = {'micro/precision': precision_score(y_true=target, y_pred=pred, average='micro', zero_division=np.nan),
                'micro/recall': recall_score(y_true=target, y_pred=pred, average='micro', zero_division=np.nan),
                'micro/f1': f1_score(y_true=target, y_pred=pred, average='micro', zero_division=np.nan),
                'macro/precision': precision_score(y_true=target, y_pred=pred, average='macro', zero_division=np.nan),
                'macro/recall': recall_score(y_true=target, y_pred=pred, average='macro', zero_division=np.nan),
                'macro/f1': f1_score(y_true=target, y_pred=pred, average='macro', zero_division=np.nan),
                'samples/precision': precision_score(y_true=target, y_pred=pred, average='samples', zero_division=np.nan),
                'samples/recall': recall_score(y_true=target, y_pred=pred, average='samples', zero_division=np.nan),
                'samples/f1': f1_score(y_true=target, y_pred=pred, average='samples', zero_division=np.nan),
                }
        return mets[score_metric]


def train(model, device, data_loader, criterion, optimizer, epoch, score_metric, print_freq=10):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    score = AverageMeter()
    # macro_f1 = AverageMeter()

    model.train()

    end = time.time()
    for i, (input, target) in enumerate(data_loader):
		# measure data loading time
        data_time.update(time.time() - end)

        if isinstance(input, tuple):
            input = tuple([e.to(device) if type(e) == torch.Tensor else e for e in input])
        else:
            input = input.to(device)
        target = target.to(device)

        optimizer.zero_grad()
        output = model(input)
        # print(torch.round(output))
        # print(target)
        loss = criterion(output, target)
        assert not np.isnan(loss.item()), 'Model diverged with loss = NaN'

        loss.backward()
        optimizer.step()

		# measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        losses.update(loss.item(), target.size(0))
        score.update(compute_batch_accuracy(output, target, score_metric).item(), target.size(0))
        # metrics = compute_batch_accuracy(output, target)
        # metrics = compute_batch_accuracy(output, target)
        # output_mets = {}
        # for m in metrics:
        #     met = AverageMeter()
        #     # key = 'macro/f1'
        #     # macro_f1_score = metrics[m]
        #     met.update(metrics[m], target.size(0))
        #     output_mets[m] = met

        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Score {score.val:.3f} ({score.avg:.3f})'.format(
                epoch, i, len(data_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, score=score))

    return losses.avg, score.avg


def evaluate(model, device, data_loader, criterion, score_metric, print_freq=10):
    batch_time = AverageMeter()
    losses = AverageMeter()
	# accuracy = AverageMeter()
    score = AverageMeter()

    # results = []

    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(data_loader):

            if isinstance(input, tuple):
                input = tuple([e.to(device) if type(e) == torch.Tensor else e for e in input])
            else:
                input = input.to(device)
            target = target.to(device)

            output = model(input)
            loss = criterion(output, target)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            losses.update(loss.item(), target.size(0))
            score.update(compute_batch_accuracy(output, target, score_metric).item(), target.size(0))
            # metrics = compute_batch_accuracy(output, target)
            # key = 'macro/f1'
            # macro_f1_score = metrics[key]
            # macro_f1.update(macro_f1_score, target.size(0))
            # output_mets = {}
            # for m in metrics:
            #     met = AverageMeter()
            #     # key = 'macro/f1'
            #     # macro_f1_score = metrics[m]
            #     met.update(metrics[m], target.size(0))
            #     output_mets[m] = met
            
            
            # y_true = target.detach().to('cpu').numpy().tolist()
            # y_pred = output.detach().to('cpu').max(1)[1].numpy().tolist()
            # results.extend(list(zip(y_true, y_pred)))

            if i % print_freq == 0:
                print('Test: [{0}/{1}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Accuracy {score.val:.3f} ({score.avg:.3f})'.format(
                        i, len(data_loader), batch_time=batch_time, loss=losses, score=score))
                    
                    
    return losses.avg, score.avg#, results
