import os
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau

from .arch import Architecture


class Model(object):

    def __init__(self, model, n_classes, load_model_path='', usegpu=True):

        self.model_name = model
        self.n_classes = n_classes
        self.load_model_path = load_model_path
        self.usegpu = usegpu

        if self.usegpu:
            self.device = torch.device('cuda')
            cudnn.benchmark = True
        else:
            self.device = torch.device('cpu')

        if load_model_path == '':
            self.model = Architecture(model, self.n_classes, pretrained=True)
        else:
            self.model = Architecture(model, self.n_classes, pretrained=False)

        self.model = self.model.to(self.device)

        self.__load_weights()

        #self.model = torch.nn.DataParallel(self.model, device_ids=range(self.ngpus))

        print((self.model))

    def __load_weights(self):

        def weights_initializer(m):
            """Custom weights initialization called on crnn"""
            classname = m.__class__.__name__
            if classname.find('Linear') != -1:
                m.weight.data.normal_(0.0, 0.001)
                m.bias.data.zero_()

        if self.load_model_path != '':
            assert os.path.isfile(self.load_model_path), 'Model : {} does not exists!'.format(self.load_model_path)
            print(('Loading model from {}'.format(self.load_model_path)))

            if self.usegpu:
                self.model.load_state_dict(torch.load(self.load_model_path))
            else:
                self.model.load_state_dict(torch.load(self.load_model_path, map_location=lambda storage, loc: storage))

            """if self.usegpu:
                loaded_model = torch.load(self.load_model_path)
            else:
                loaded_model = torch.load(self.load_model_path, map_location=lambda storage, loc: storage)
            loaded_model_layer_keys = loaded_model.keys()
            for layer_key in loaded_model_layer_keys:
                if layer_key.startswith('module.'):
                    new_layer_key = '.'.join(layer_key.split('.')[1:])
                    loaded_model[new_layer_key] = loaded_model.pop(layer_key)
            self.model.load_state_dict(loaded_model)"""
        else:
            self.model.apply(weights_initializer)

    def __define_input_variables(self, features, labels, volatile=False):

        if volatile:
            with torch.no_grad():
                features_var = Variable(features)
                labels_var = Variable(labels)
        else:
            features_var = Variable(features)
            labels_var = Variable(labels)

        features_var = features_var.to(self.device) #non_blocking=True
        labels_var = labels_var.to(self.device)     #non_blocking=True

        return features_var, labels_var

    def __define_criterion(self):
        self.criterion = torch.nn.CrossEntropyLoss()

        self.criterion = self.criterion.to(self.device)

    def __define_optimizer(self, learning_rate, weight_decay, lr_drop_factor, lr_drop_patience, optimizer='Adam'):
        assert optimizer in ['RMSprop', 'Adam', 'Adadelta', 'SGD']
        if optimizer == 'RMSprop':
            self.optimizer = optim.RMSprop(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        elif optimizer == 'Adadelta':
            self.optimizer = optim.Adadelta(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        elif optimizer == 'Adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        elif optimizer == 'SGD':
            self.optimizer = optim.SGD(self.model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)

        self.lr_scheduler = ReduceLROnPlateau(self.optimizer, mode='max', factor=lr_drop_factor, patience=lr_drop_patience, verbose=True)

    @staticmethod
    def __get_loss_averager():
        return averager()

    def __minibatch(self, train_test_iter, clip_grad_norm, mode='training'):
        assert mode in ['training', 'validation'], 'Mode must be either "training" or "validation"'
        if mode == 'training':
            for param in self.model.parameters():
                param.requires_grad = True
            self.model.train()
        else:
            for param in self.model.parameters():
                param.requires_grad = False
            self.model.eval()

        features, labels = next(train_test_iter)

        if mode == 'training':
            features, labels = self.__define_input_variables(features, labels)
        else:
            features, labels = self.__define_input_variables(features, labels, volatile=True)

        predictions = self.model(features)

        cost = self.criterion(predictions, labels)

        if mode == 'training':
            self.model.zero_grad()
            cost.backward()
            if clip_grad_norm != 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_grad_norm)

            self.optimizer.step()

        return cost, predictions, labels

    def __validate(self, test_loader):
        n_minibatches = len(test_loader)

        test_loss_averager = Model.__get_loss_averager()

        test_iter = iter(test_loader)
        n_correct = 0
        n_total = 0

        for minibatch_index in range(n_minibatches):
            cost, predictions, labels  = self.__minibatch(test_iter, 0.0, mode='validation')
            batch_size = float(predictions.size(0))
            test_loss_averager.add(cost)

            _, predictions = predictions.max(1)

            n_correct += torch.sum(predictions.data.cpu() == labels.data.cpu()).item()
            n_total += batch_size

        loss = test_loss_averager.val()
        accuracy = n_correct / float(n_total)

        print(('Validation Loss: {:.5f}, Accuracy: {:.5f}'.format(loss, accuracy)))

        return accuracy, loss

    def predict(self, features):

        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()

        with torch.no_grad():
            features = Variable(features)

        features = features.to(self.device)

        predictions = self.model(features)
        predictions = F.softmax(predictions, dim=1)

        return predictions.data.cpu().numpy()

    def test(self, test_loader):

        self.__define_criterion()

        n_minibatches = len(test_loader)
        test_iter = iter(test_loader)

        preds = []; labels = []
        for minibatch_index in range(n_minibatches):
            _, predictions, _labels = self.__minibatch(test_iter, 0.0, mode='validation')
            predictions = F.softmax(predictions, dim=1)
            predictions = predictions.data.cpu().numpy()
            _labels = _labels.data.cpu().numpy()

            labels.extend(_labels)
            preds.extend(predictions)

        return preds, labels

    def fit(self, learning_rate, weight_decay, clip_grad_norm, lr_drop_factor, lr_drop_patience, optimizer,
            n_epochs, train_loader, test_loader, model_save_path):

        training_log_file = open(os.path.join(model_save_path, 'training.log'), 'w')
        validation_log_file = open(os.path.join(model_save_path, 'validation.log'), 'w')

        training_log_file.write('Epoch,Loss,Accuracy\n')
        validation_log_file.write('Epoch,Loss,Accuracy\n')

        train_loss_averager = Model.__get_loss_averager()

        self.__define_criterion()
        self.__define_optimizer(learning_rate, weight_decay, lr_drop_factor, lr_drop_patience, optimizer=optimizer)

        self.__validate(test_loader)

        best_val_acc = 0.0
        best_val_loss = 1e10
        for epoch in range(n_epochs):
            epoch_start = time.time()

            train_iter = iter(train_loader)
            n_minibatches = len(train_loader)

            minibatch_index = 0
            train_n_correct = 0; train_n_total = 0
            while minibatch_index < n_minibatches:
                minibatch_cost, minibatch_predictions, minibatch_labels = self.__minibatch(train_iter, clip_grad_norm, 
                                                                                               mode='training')

                batch_size = float(minibatch_predictions.size(0))
                _, minibatch_predictions = minibatch_predictions.max(1)
                train_n_correct += torch.sum(minibatch_predictions.data.cpu() == minibatch_labels.data.cpu()).item()
                train_n_total += batch_size

                train_loss_averager.add(minibatch_cost)
                minibatch_index += 1

            train_accuracy = train_n_correct / float(train_n_total)
            train_loss = train_loss_averager.val()

            epoch_end = time.time()
            epoch_duration = epoch_end - epoch_start

            print(('[{:.5f}] [{}/{}] Loss : {:.5f} - Accuracy : {:.5f}'.format(epoch_duration, epoch, n_epochs, train_loss,
                                                                               train_accuracy)))

            val_accuracy, val_loss = self.__validate(test_loader)

            self.lr_scheduler.step(val_accuracy)

            if val_accuracy > best_val_acc:
                best_val_acc = val_accuracy
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), os.path.join(model_save_path,
                                                     'model_{:04d}_{:.5f}_{:.5f}.pth'.format(epoch, val_loss, val_accuracy)))
            elif val_accuracy == best_val_acc:
                if val_loss < best_val_loss:
                    best_val_acc = val_accuracy
                    best_val_loss = val_loss
                    torch.save(self.model.state_dict(), os.path.join(model_save_path,
                                                     'model_{:04d}_{:.5f}_{:.5f}.pth'.format(epoch, val_loss, val_accuracy)))


            training_log_file.write('{},{:.5f},{:.5f}\n'.format(epoch, train_loss, train_accuracy))
            validation_log_file.write('{},{:.5f},{:.5f}\n'.format(epoch, val_loss, val_accuracy))
            training_log_file.flush()
            validation_log_file.flush()

            train_loss_averager.reset()
            train_n_correct = 0; train_n_total = 0

        training_log_file.close()
        validation_log_file.close()

class averager(object):
    """Compute average for `torch.Variable` and `torch.Tensor`."""

    def __init__(self):
        self.reset()

    def add(self, v):
        if isinstance(v, Variable):
            count = v.data.numel()
            v = v.data.sum()
        elif isinstance(v, torch.Tensor):
            count = v.numel()
            v = v.sum()

        self.n_count += count
        self.sum += v

    def reset(self):
        self.n_count = 0
        self.sum = 0

    def val(self):
        res = 0
        if self.n_count != 0:
            res = self.sum / float(self.n_count)
        return res

if __name__ == "__main__":
    model = Model(4, load_model_path='', usegpu=False)
    x = torch.rand(2, 3, 128, 128)
    print(x.size())
    y = model.predict(x)
    print(y)
