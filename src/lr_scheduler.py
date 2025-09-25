from timm.scheduler.cosine_lr import CosineLRScheduler

def build_scheduler(config, optimizer, n_iter_per_epoch):
    num_steps = int(config.num_epochs * n_iter_per_epoch) # Total number of training steps
    warmup_steps = config.warmup_t if not config.t_in_epochs else int(config.warmup_t * n_iter_per_epoch) # Total number of warmup steps
 
    lr_scheduler = CosineLRScheduler(
        optimizer,
        t_initial=(
            (num_steps - warmup_steps) 
            if config.lr_scheduler.warmup_prefix else num_steps
        ),  # total steps for LR decay in one cycle
        cycle_mul=1.,
        lr_min=config.lr_scheduler.lr_min, # minimum learning rate
        warmup_lr_init=config.lr_scheduler.warmup_lr_init, # initial LR during warmup
        warmup_t=config.warmup_t, # warmup duration
        cycle_limit=1, # number of cycles
        t_in_epochs=config.t_in_epochs, # False = step-based, True = epoch-based
        warmup_prefix=config.lr_scheduler.warmup_prefix,
    )

    return lr_scheduler
