import numpy as np
import yaml

class FS_Params:
    def __init__(self):
        super(FS_Params, self).__init__()

        self.BACKBONE = None
        self.MEMB_TYPE = None

        self.Bspline_t = None
        self.Bspline_c = None
        self.Bspline_k = None

        self.LR_b0 = None
        self.LR_b1 = None

        self.EXP_0 = None
        self.EXP_1 = None
        self.EXP_2 = None
        self.EXP_3 = None
        
        self.threshd = None
        self.Epi = None
        self.Max_score = None

        self.ALPHA_I_L = None
        self.ALPHA_U_L = None
        self.ALPHA_N = None

    def set_static_fs_params(self, backbone, memb_type, dataname):
        self.BACKBONE = backbone
        self.MEMB_TYPE = memb_type
        if backbone == 'QTO':
            if dataname in ['nell', 'fb15k-237']:
                self.Epi = 0.0001
                self.Max_score = 9.210441

            elif dataname == 'fb15k':
                self.Epi = 0.0005
                self.Max_score = 7.6014023

    def load_membership_params(self, dataname):
        if self.MEMB_TYPE == 'symbolic':
            params_fn = f'./params/{dataname}_{self.BACKBONE}_symbolic_params.yml'
        else:
            params_fn = f'./params/{dataname}_{self.BACKBONE}_params.yml'
        print(f'Loading params from {params_fn} ...')

        with open(params_fn, 'r') as fin:
            params_data = fin.read()
        params_data = yaml.load(params_data, Loader=yaml.Loader)

        if self.MEMB_TYPE == 'symbolic' and self.BACKBONE == 'CQD':
            self.LR_b0 = float(params_data['b0'][0])
            self.LR_b1 = float(params_data['b1'][0,0])
        elif self.MEMB_TYPE == 'symbolic' and self.BACKBONE == 'QTO':
            self.EXP_0 = float(params_data['a'])
            self.EXP_1 = float(params_data['b'])
            self.EXP_2 = float(params_data['c'])
            self.EXP_3 = float(params_data['d'])
        else:
            self.Bspline_t = np.array(params_data['t'])
            self.Bspline_c = np.array(params_data['c'])
            self.Bspline_k = int(params_data['k'])

        self.threshd = float(params_data['threshd'])

    def load_rule_params(self, dataname):
        if self.MEMB_TYPE == 'symbolic':
            rule_params_fn = f'./params/{dataname}_{self.BACKBONE}_rule_symbolic_params.yml'
        else:
            rule_params_fn = f'./params/{dataname}_{self.BACKBONE}_rule_params.yml'
        print(f'Loading rule params from {rule_params_fn} ...')

        with open(rule_params_fn, 'r') as fin:
            params_data = fin.read()
        params_data = yaml.load(params_data, Loader=yaml.Loader)

        self.ALPHA_I_L = tuple(params_data['I'])
        self.ALPHA_U_L = tuple(params_data['U'])
        self.ALPHA_N = tuple(params_data['N'])


FS_PARAMS = FS_Params()

DEFUZZ = 'mean' # 'mean' / 'max'

GRID = False

ADJ = None
