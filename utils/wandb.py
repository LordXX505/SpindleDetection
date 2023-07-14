import wandb

class wandb_logger:
    def __init__(self, config, job_id, k_fold):
        # wandb.login(key='b1436c68efb97170187ddd062dbde65e9c173745')
        name = f"experiment_{job_id}_{k_fold}"
        # Track hyperparameters and run metadata
        wandb.init(
            project="SpindleDetection",
            anonymous="allow",
            name=name,
            config=config,
            reinit=True)

        self.config = config
        self.step = None

    #用于记录参数值的接口
    def log(self, name, data, step=None):
        if step is None:
            wandb.log({name: data})
        else:
            wandb.log({name: data}, step=step)
            self.step = step

    #用于保存模型的接口
    def watch_model(self, *args, **kwargs):
        wandb.watch(*args, **kwargs)


    #用于显示图像的接口
    def log_image(self, name, fig):
        if self.step is None:
            wandb.log({name: wandb.Image(fig)})
        else:
            wandb.log({name: wandb.Image(fig)}, step=self.step)

    def finish(self):
        wandb.finish(quiet=True)