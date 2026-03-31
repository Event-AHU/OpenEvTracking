from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.


    settings.coesot_path = '/data/zhangfan/CEUTrack_COESOT_rgbe/data/COESOT'
    settings.fe108_path = '/data/zhangfan/CEUTrack_COESOT_rgbe/data/FE108'
    settings.prj_dir     = "/data/zhangfan/CEUTrack_COESOT_rgbe"
    settings.network_path = '/data/zhangfan/CEUTrack_COESOT_rgbe/test/networks'    # Where tracking networks are stored.
    settings.save_dir    = "/data/zhangfan/CEUTrack_COESOT_rgbe/output"
    settings.result_plot_path = '/data/zhangfan/CEUTrack_COESOT_rgbe/test/result_plots'
    settings.results_path = '/data/zhangfan/CEUTrack_COESOT_rgbe/test/tracking_results'    # Where to store tracking results
    settings.save_dir = '/data/zhangfan/CEUTrack_COESOT_rgbe'
    settings.segmentation_path = '/data/zhangfan/CEUTrack_COESOT_rgbe/test/segmentation_results'
    settings.visevent_path = '/data/zhangfan/CEUTrack_COESOT_rgbe/data/visevent'

    return settings

