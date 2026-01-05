class EnvironmentSettings:
    def __init__(self):
        self.workspace_dir = ''  # Base directory for saving network checkpoints.
        self.tensorboard_dir = self.workspace_dir + '/tensorboard'    # Directory for tensorboard files.

        self.coesot_dir = ''
        self.coesot_val_dir = ''
        self.felt_dir = ''
        self.felt_val_dir = ''
        self.fe108_dir = ''
        self.fe108_val_dir = ''
