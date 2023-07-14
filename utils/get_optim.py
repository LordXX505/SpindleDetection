from torch.optim import lr_scheduler
from torch import optim


def get_optimizer(option, params):
    opt_alg = 'sgd' if option.optim is None else option.optim
    if opt_alg == 'sgd':
        optimizer = optim.SGD(params,
                              lr=option.lr,
                              momentum=0.9,
                              nesterov=True,
                              )

    elif opt_alg == 'adam':
        optimizer = optim.Adam(params,
                               lr=option.lr,
                               betas=(0.9, 0.999),
                               )
    else:
        optimizer = optim.AdamW(params, lr=option.lr, betas=(0.9, 0.95))
    return optimizer


def get_scheduler(optimizer, opt):
    print('opt.lr_policy = [{}]'.format(opt.lr_policy))
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 1 + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        if opt.lr_decay_iters is None:
            opt.lr_decay_iters = 25
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.5)
    elif opt.lr_policy == 'step2':
        if opt.lr_decay_iters is None:
            opt.lr_decay_iters = 25
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        print('schedular=plateau')
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, threshold=0.001, patience=10)
    elif opt.lr_policy == 'plateau2':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.001, patience=5)
    elif opt.lr_policy == 'step_warmstart':
        def lambda_rule(epoch):
            #print(epoch)
            if epoch < 5:
                lr_l = 0.1
            elif 5 <= epoch < 100:
                lr_l = 1
            elif 100 <= epoch < 200:
                lr_l = 0.1
            elif 200 <= epoch:
                lr_l = 0.01
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step_warmstart2':
        def lambda_rule(epoch):
            #print(epoch)
            if epoch < 5:
                lr_l = 0.1
            elif 5 <= epoch < 50:
                lr_l = 1
            elif 50 <= epoch < 100:
                lr_l = 0.1
            elif 100 <= epoch:
                lr_l = 0.01
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    else:

        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler
