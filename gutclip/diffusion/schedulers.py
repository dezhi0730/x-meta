from diffusers import DDPMScheduler, DDIMScheduler, DPMSolverMultistepScheduler

def get_scheduler(name: str, num_train_steps: int = 1000):
    if name == "ddpm":
        return DDPMScheduler(num_train_timesteps=num_train_steps)
    if name == "ddim":
        return DDIMScheduler(num_train_timesteps=num_train_steps)
    if name == "dpm_solver":
        return DPMSolverMultistepScheduler(num_train_timesteps=num_train_steps)
    raise ValueError(name)