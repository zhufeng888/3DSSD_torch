import torch
import time
import os
import numpy as np
import time
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_sched
from torch.autograd import Variable
np.set_printoptions(suppress=True)

class ModelTrainer():

    def train(self, model, train_dataset, val_dataset, cfg):

        optimizer = optim.Adam([{'params': model.parameters(), 'initial_lr': cfg.learning_rate}], cfg.learning_rate)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, cfg.STEPS, gamma=0.1, last_epoch=0)
        bn_decay = 0.5

        start_epoch = 0
        iter = 0
        if cfg.snapshot:
            checkpoint = torch.load(cfg.checkpoint)
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint['epoch']
            iter = checkpoint['iter']
            cfg.log.out('Load model successfully: %s' % (cfg.snapshot))

        num_batch = len(train_dataset)/cfg.batch_size
        mean_step_time = 0
        model.cuda()
        model.train()

        # Start training loop
        for epoch in range(start_epoch, cfg.epochs):
            if iter >= cfg.max_iter:
                break
            step = 0
            time1 = time.time()
            for i, batch in enumerate(train_dataset, 0):
                for key in batch:
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].cuda()


                # grad clear
                optimizer.zero_grad()

                # Forward pass
                loss_dict = model(cfg, batch, bn_decay=bn_decay)

                # Backward + optimize
                loss = loss_dict['offset_loss'] +loss_dict['angle_bin_loss'] +loss_dict['angle_res_loss'] +\
                       loss_dict['cls_loss']
                if loss_dict['corner_loss'] is not None:
                    loss += loss_dict['corner_loss']
                else:
                    loss_dict['corner_loss'] = 0

                if loss_dict['vote_loss'] is not None:
                    loss += loss_dict['vote_loss']
                else:
                    loss_dict['vote_loss'] = 0

                loss.backward()
                optimizer.step()

                time2 = time.time()
                mean_step_time = (mean_step_time*(epoch*num_batch + step) + (time2-time1))/(epoch*num_batch + step+1)

                if step % cfg.log_interval == 0:
                    cfg.log.out(
                        "iter:%6d; epoch:%3d; step:%3d; loss:%0.6f;  "
                        "offset_loss:%0.6f; angle_bin_loss:%0.6f; angle_res_loss:%0.6f; cls_loss:%0.6f; corner_loss:%0.6f; "
                        "vote_loss:%0.6f; "
                        "lr:%0.6f; step_time:%0.3fms; mean_step_time:%0.3fms"
                        % (iter,epoch, step, loss,  loss_dict['offset_loss'],loss_dict['angle_bin_loss'],loss_dict['angle_res_loss'],
                           loss_dict['cls_loss'], loss_dict['corner_loss'],loss_dict['vote_loss'],
                           scheduler.get_lr()[0], 1000*(time2 - time1), 1000*mean_step_time))

                    #cfg.writer.add_scalar('Train/Loss', loss.item(), step + epoch * num_batch) 待我研究一下补充

                time1 = time2
                step += 1
                iter +=1
                scheduler.step(iter)

                if iter >= cfg.STEPS[0]:
                    bn_decay=0.75

                if iter >= cfg.max_iter:
                    break

            # Validation
            '''model.eval()
            with torch.no_grad():
                self.validation(model, val_dataset, cfg, epoch)
            model.train()'''

            #保存模型
            if (epoch % cfg.check_interval == 0 and epoch != start_epoch) or iter >= cfg.max_iter:
                state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                         'iter': iter+1, 'epoch': epoch+1,}
                torch.save(state, cfg.checkpoint)

            

        cfg.log.out('Finished Training')
        # 第五步，移除某个创建的handler
        cfg.log.close()




    def validation(self, model, val_dataset, cfg, epoch):
        raise NotImplementedError('Please Use my_val_Kitti.py')






       
       
