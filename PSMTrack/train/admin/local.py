class EnvironmentSettings:
    def __init__(self):
        self.workspace_dir = '.../PSMTrack'  # Base directory for saving network checkpoints.
        self.tensorboard_dir = self.workspace_dir + '/tensorboard'    # Directory for tensorboard files
        
        self.eventvot_dir = '.../'
        self.eventvot_val_dir = '.../'
        # self.coesot_dir = ''
        # self.coesot_val_dir = ''
        # self.fe240_dir = ''
        # self.fe240_val_dir = ''
