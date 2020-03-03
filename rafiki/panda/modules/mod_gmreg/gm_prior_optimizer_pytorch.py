# This file contains all the classes that are related to
# GM-prior adaptive regularizer.
# =============================================================================
import torch
import numpy as np
from scipy.stats import norm as gaussian
import math


class GMOptimizer():
    '''
    introduce hyper-parameters for GM-regularization: a, b, alpha
    '''

    def __init__(self):
        self.weight_name_list = {}
        self.weight_dim_list = {}
        self.gmregularizers = {}

    def layer_wise_hyperpara(self, fea_num, hyperpara_list):
        print("layer_wise hyperpara_list: ", hyperpara_list)
        print("layer_wise fea_num: ", fea_num)
        alpha_val = fea_num ** (hyperpara_list[2])
        b_val = hyperpara_list[1] * fea_num
        a_val = 1. + b_val * hyperpara_list[0]
        return [a_val, b_val, alpha_val]

    # value is numpy here
    def gm_register(self, name, value, model_name, hyperpara_list, gm_num, gm_lambda_ratio_value, uptfreq):
        print("param name: ", name)
        print("param shape: ", value.shape)
        # only incude weights
        if np.ndim(value) >= 2:
            self.weight_name_list[name] = name
            dims = value.size
            print("dims: ", dims)
            self.weight_dim_list[name] = dims
            layer_hyperpara = self.layer_wise_hyperpara(dims, hyperpara_list)
            print("gm_register layer_hyperpara: ", layer_hyperpara)
            pi = [1.0 / gm_num for _ in range(gm_num)]
            k = 1.0 + gm_lambda_ratio_value
            print("gm_lambda_ratio_value: ", gm_lambda_ratio_value)
            # calculate base
            if 'mlp' in model_name:
                print("gm_register base mlp name: ", name)
                print("gm_register base mlp shape: ", value.shape)
                base = (value.shape[0] + value.shape[1]) / 40.0
            elif 'lstm' in model_name:
                if "weight_ih" in name or "weight_hh" in name:
                    print("gm_register base lstm ih and hh name: ", name)
                    print('gm_register base lstm ih and hh shape: ', value.shape)
                    base = 3.0 * 128.0 / 10.0  # hard-coded for lstm
                else:  # fc1
                    print("gm_register base lstm fc name: ", name)
                    print('gm_register base lstm fc shape: ', value.shape)
                    base = (value.shape[0] + value.shape[1]) / 40.0
            else:  # for resnet
                if 'conv' in name or 'downsample' in name:
                    print("gm_register base cnn conv downsample name: ", name)
                    print("gm_register base cnn conv downsample value shape: ", value.shape)
                    base = value.shape[0] * value.shape[2] * value.shape[3] / 20.0
                else:  # fc1
                    print("gm_register base cnn fc name: ", name)
                    print("gm_register base cnn fc value shape: ", value.shape)
                    base = 3.0 * value.shape[1] / 10.0
            print("base: ", base)
            # calculate GM initialized lambda (1/variance)
            if gm_lambda_ratio_value >= 0.0:
                reg_lambda = [base * math.pow(k, _) for _ in range(gm_num)]
            else:
                reg_lambda_range = base * float(gm_num)
                reg_lambda = np.arange(1.0, reg_lambda_range, reg_lambda_range / gm_num)
            print("pi: ", pi)
            print("reg_lambda: ", reg_lambda)
            self.gmregularizers[name] = GMRegularizer(hyperpara=layer_hyperpara, gm_num=gm_num, pi=pi,
                                                      reg_lambda=reg_lambda, uptfreq=uptfreq)

    def apply_GM_regularizer_constraint(self, labelnum, trainnum, epoch, weight_decay, f, name, step):
        # if np.ndim(tensor.to_numpy(value)) <= 2:
        if np.ndim(f.data.cpu().numpy()) < 2:
            # print ("apply adding weight decay: ", name)
            f.grad.data.add_(float(weight_decay), f.data)
        else:  # weight parameter
            # print ("not apply adding weight decay: ", name)
            # print ("self.gmregularizers[name]: ", self.gmregularizers[name])
            self.gmregularizers[name].apply(labelnum, trainnum, epoch, f, name, step)


class GMRegularizer():
    '''GM regularization
    Args:
        hyperparameters: a, b, alpha (like the coefficient of L2), uptfreq
    '''

    def __init__(self, hyperpara=None, gm_num=None, pi=None, reg_lambda=None, uptfreq=None):
        self.a, self.b, self.alpha, self.gm_num = hyperpara[0], hyperpara[1], hyperpara[2], gm_num
        print("init self.a, self.b, self.alpha, self.gm_num: ", self.a, self.b, self.alpha, self.gm_num)
        self.pi, self.reg_lambda = np.reshape(np.array(pi), (1, gm_num)), np.reshape(np.array(reg_lambda), (1, gm_num))
        print("init self.reg_lambda: ", self.reg_lambda)
        print("init self.pi: ", self.pi)
        self.gmuptfreq, self.paramuptfreq = uptfreq[0], uptfreq[1]
        print("init self.gmuptfreq, self.paramuptfreq: ", self.gmuptfreq, self.paramuptfreq)

    # calc the resposibilities for pj(wi)
    def calcResponsibility(self):
        # responsibility normalized with pi
        responsibility = gaussian.pdf(self.w_array, loc=np.zeros(shape=(1, self.gm_num)),
                                      scale=1 / np.sqrt(self.reg_lambda)) * self.pi
        # print ("responsibility shape: ", responsibility.shape)
        # responsibility normalized with summation(denominator)
        self.responsibility = responsibility / (np.sum(responsibility, axis=1).reshape(self.w_array.shape))
        # print ("np.sum(self.responsibility, axis=1): ", np.sum(self.responsibility, axis=1))

    def update_GM_Prior_EM(self, name, step):
        # update pi
        self.reg_lambda = (2 * (self.a - 1) + np.sum(self.responsibility, axis=0)) / (
                    2 * self.b + np.sum(self.responsibility * np.square(self.w_array), axis=0))
        if step % self.gmuptfreq == 0:
            print("name: ", name)
            print("np.sum(self.responsibility, axis=0): ", np.sum(self.responsibility, axis=0))
            print("np.sum(self.responsibility * np.square(self.w_array), axis=0): ",
                  np.sum(self.responsibility * np.square(self.w_array), axis=0))
            print("division: ",
                  np.sum(self.responsibility * np.square(self.w_array), axis=0) / np.sum(self.responsibility, axis=0))
        # update reg_lambda
        self.pi = (np.sum(self.responsibility, axis=0) + self.alpha - 1) / (
                    self.w_array.shape[0] + self.gm_num * (self.alpha - 1))
        # print ("self.w_array.shape[0]: ", self.w_array.shape[0])
        if step % self.gmuptfreq == 0:
            print("reg_lambda", self.reg_lambda)
            print("pi:", self.pi)

    def chunk_array(self, arr, chunks, dim):
        if dim == 0:
            chunk_array_list = []
            base = int(arr.shape[0] / chunks)
            for i in range(chunks):
                chunk_array_list.append(arr[i * base: (i + 1) * base])
        return chunk_array_list

    def apply(self, labelnum, trainnum, epoch, f, name, step):
        if "_first_gate" in name:
            w_array_chunk = self.chunk_array(f.data.cpu().numpy(), 4, 0)
            self.w_array = w_array_chunk[0].reshape((-1, 1))  # used for EM update also
        elif "_second_gate" in name:
            w_array_chunk = self.chunk_array(f.data.cpu().numpy(), 4, 0)
            self.w_array = w_array_chunk[1].reshape((-1, 1))  # used for EM update also
        elif "_third_gate" in name:
            w_array_chunk = self.chunk_array(f.data.cpu().numpy(), 4, 0)
            self.w_array = w_array_chunk[2].reshape((-1, 1))  # used for EM update also
        elif "_fourth_gate" in name:
            w_array_chunk = self.chunk_array(f.data.cpu().numpy(), 4, 0)
            self.w_array = w_array_chunk[3].reshape((-1, 1))  # used for EM update also
        else:
            # print ("name: ", name)
            # print ("self.w_array not first to fourth gate")
            self.w_array = f.data.cpu().numpy().reshape((-1, 1))  # used for EM update also
        if epoch < 2 or step % self.paramuptfreq == 0:
            self.calcResponsibility()
            if "_first_gate" in name:
                self.reg_grad_w = np.zeros(f.grad.data.cpu().numpy().shape)
                base = int(self.reg_grad_w.shape[0] / 4)
                self.reg_grad_w[0 * base: 1 * base] = (np.sum(self.responsibility * self.reg_lambda, axis=1).reshape(
                    self.w_array.shape) * self.w_array).reshape(base, -1)
            elif "_second_gate" in name:
                self.reg_grad_w = np.zeros(f.grad.data.cpu().numpy().shape)
                base = int(self.reg_grad_w.shape[0] / 4)
                self.reg_grad_w[1 * base: 2 * base] = (np.sum(self.responsibility * self.reg_lambda, axis=1).reshape(
                    self.w_array.shape) * self.w_array).reshape(base, -1)
            elif "_third_gate" in name:
                self.reg_grad_w = np.zeros(f.grad.data.cpu().numpy().shape)
                base = int(self.reg_grad_w.shape[0] / 4)
                self.reg_grad_w[2 * base: 3 * base] = (np.sum(self.responsibility * self.reg_lambda, axis=1).reshape(
                    self.w_array.shape) * self.w_array).reshape(base, -1)
            elif "_fourth_gate" in name:
                self.reg_grad_w = np.zeros(f.grad.data.cpu().numpy().shape)
                base = int(self.reg_grad_w.shape[0] / 4)
                self.reg_grad_w[3 * base: 4 * base] = (np.sum(self.responsibility * self.reg_lambda, axis=1).reshape(
                    self.w_array.shape) * self.w_array).reshape(base, -1)
            else:
                # print ("name: ", name)
                # print ("self.reg_grad_w not first to fourth gate")
                self.reg_grad_w = np.sum(self.responsibility * self.reg_lambda, axis=1).reshape(
                    self.w_array.shape) * self.w_array
        # print ("in apply f.data.cpu().numpy().shape: ", f.data.cpu().numpy().shape)
        normalization_coefficient = float(labelnum * trainnum)
        # print ("in apply normalization_coefficient: ", normalization_coefficient)
        # print ("in apply labelnum: ", labelnum)
        # print ("in apply trainnum: ", trainnum)
        reg_grad_w_dev = (torch.from_numpy(
            (self.reg_grad_w.reshape(f.data.cpu().numpy().shape)) / float(normalization_coefficient))).float()
        if (epoch == 0 and step < 50) or step % self.gmuptfreq == 0:
            print("step: ", step)
            print("name: ", name)
            print('data grad norm: ', np.linalg.norm(f.grad.data.cpu().numpy()))
            print('reg_grad_w_dev l2 norm: ', np.linalg.norm(reg_grad_w_dev.cpu().numpy()))
        f.grad.data.add_(1.0, reg_grad_w_dev.cuda())  # here3
        if (epoch == 0 and step < 50) or step % self.gmuptfreq == 0:
            print("delta w norm: ", np.linalg.norm(f.grad.data.cpu().numpy()))
            print("w norm: ", np.linalg.norm(f.data.cpu().numpy()))
        if epoch < 2 or step % self.gmuptfreq == 0:
            if epoch >= 2 and step % self.paramuptfreq != 0:
                self.calcResponsibility()
            self.update_GM_Prior_EM(name, step)
        # if 'conv1.weight' in name or 'fc.weight' in name:
        #     print ("in end apply name: ", name)
        #     print ("in end apply pi: ", self.pi)
        #     print ("in end apply reg_lambda: ", self.reg_lambda)


'''
class GMSGD(GMOptimizer, SGD):
    # The vallina Stochasitc Gradient Descent algorithm with momentum.
    # But this SGD has a GM regularizer

    def __init__(self, net=None, lr=None, momentum=None, weight_decay=None,
                 regularizer=None, constraint=None):
        GMOptimizer.__init__(self, net=net, lr=lr, momentum=momentum, weight_decay=weight_decay, regularizer=regularizer,
                                  constraint=constraint)
        SGD.__init__(self, lr=lr, momentum=momentum, weight_decay=weight_decay,
                 regularizer=regularizer, constraint=constraint)

    # compared with apply_with_lr, this need one more argument: isweight
    def apply_with_lr(self, dev, trainnum, net, epoch, lr, grad, value, name, step=-1):
        if grad.is_empty():
            return value
        ##### GM-prior: using gm_regularizer ##############
        grad = self.apply_GM_regularizer_constraint(dev=dev, trainnum=trainnum, net=net, epoch=epoch, value=value, grad=grad, name=name, step=step)
        ##### GM-prior: using gm_regularizer ##############
        if name is not None and name in self.learning_rate_multiplier:
            lr = lr * self.learning_rate_multiplier[name]
        self.opt.Apply(epoch, lr, name, grad.singa_tensor, value.singa_tensor)
        return value
'''
