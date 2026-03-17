import numpy as np; np.set_printoptions(precision=2); np.random.seed(0)
import torch; torch.set_printoptions(precision=2)
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib 
from matplotlib.font_manager import FontProperties
from mpl_toolkits import mplot3d

def get_default_hp():
    '''Get a default hp.

    Returns:
        hp : a dictionary containing training hyperparameters
        optimizer: the type of optimizer (needs to be instantiated later)
        loss_fnc: the type of loss function
    '''
#     num_ring = task.get_num_ring(ruleset)
#     n_rule   = task.get_num_rule(ruleset)

#     n_eachring = 32
#     n_input, n_output = 1+num_ring*n_eachring+n_rule, n_eachring+1


    hp = {
            # Type of loss functions
            'loss_type': 'mse',
            # initialization: diagonal, orthogonal, kaiming_normal, kaiming_uniform, normal, uniform, constant
            'initialization_weights': 'orthogonal',
            'initialization_bias': 'zero',
            # Optimizer
            'optimizer': 'adam',
            # Type of activation runctions, relu, softplus, tanh, elu
            'activation': 'relu',
            'k_relu_satu': 10,    # the saturating bound if using the saturating ReLU function
            # Time constant (ms)
            'tau': 100,
            # discretization time step (ms)
            'dt': 10,
            # discretization time step/time constant
#             'alpha': 0.2,
            # recurrent noise
#             'sigma_rec': 0.05,
            # input noise
#             'sigma_x': 0.01,
            # leaky_rec weight initialization, diag, randortho, randgauss
#             'w_rec_init': 'randortho',
            # a default weak regularization prevents instability
            'l1_h': 1e-3*0,
            # l2 regularization on activity
            'l2_h': 1e-3*0,
            # l2 regularization on weight
            'l1_weight': 1e-3*0,
            # l2 regularization on weight
            'l2_weight': 1e-3*0,
            # l2 regularization on recurrent E synapses of the SR network
            'l2_rec_e_weight_sr': 0,
            # l2 regularization on the neural activity of the E neurons in the SR network
            'l2_h_sr': 0,
            # l2 regularization on the neural activity of the E neurons in the PFC network
            'l1_h_sredend': 0,
            # l1 regularization on the activity of SR E dendrite
            'l2_h_pfc': 0,
            # l2 regularization on deviation from initialization
#             'l2_weight_init': 0,
            # proportion of weights to train, None or float between (0, 1)
#             'p_weight_train': None,
            # Stopping performance
            'target_perf': 1,
            # number of units each ring
#             'n_eachring': n_eachring,
            # number of rings
#             'num_ring': num_ring,
            # number of rules
#             'n_rule': n_rule,
            # first input index for rule units
#             'rule_start': 1+num_ring*n_eachring,
            # number of input units
            'n_input': 16,    # for cxtdm: 5, for wcst: 16
            # number of input units for rule cue
            'n_input_rule_cue': 4,
            # number of output units
            'n_output': 3,
            # number of PFC readout units
            'n_output_rule': 2,
            # number of recurrent units
            'cell_group_list': ['sr_esoma', 'sr_edend', 'sr_pv', 'sr_sst', 'sr_vip', 'pfc_esoma', 'pfc_edend', 'pfc_pv', 'pfc_sst', 'pfc_vip'],
            'n_sr_esoma': int(70),     # default: 70
            'n_sr_edend': int(140),     # default: 140
            'n_sr_pv': int(10),     # default: 10
            'n_sr_sst': int(10),     # default: 10
            'n_sr_vip': int(10),    # default: 10
            'n_pfc_esoma': int(70),     # default: 70
            'n_pfc_edend': int(140),     # default: 140
            'n_pfc_pv': int(10),     # default: 10
            'n_pfc_sst': int(10),     # default: 10
            'n_pfc_vip': int(10),    # default: 10
            # number of input units
#             'ruleset': ruleset,
            # if save model and figure
            'save_model': False,
            'save_figures': True,
            # name to save
            'save_name': 'NA',
            # learning rate
            'learning_rate': 1e-3,
            # gradient clipping: max L2 norm of all parameter gradients (None = no clipping)
            'grad_clip_max_norm': 1.0,
            # intelligent synapses parameters, tuple (c, ksi)
#             'c_intsyn': 0,
#             'ksi_intsyn': 0,
            'explicit_rule': False,
            'train_rule': True,
            'block_len': 20,
            'n_switches': 3,
            'n_batches_per_block': int(2e8),
            'n_blocks': int(1),
            'batch_size': int(50),
#             'batch_size_test': 1,
            'network_noise': 0.01,
            'input_noise_perceptual': 0.01,
            'input_noise_rule': 0.01,
            'switch_every': 10,    # switch every x batches
            'test': True,    # whether to freeze the weight and test once in a while during training
            'n_branches': 2,    # number of dendritic branches for each E cell
            'mglur': False,    # metabotropic glutamate receptor
            'divide_sr_sst_vip': False,     # two subgroups of SR SST and SR VIP (to encourage gating)
            'no_pfcesoma_to_srsst': False,
            'no_pfcesoma_to_sredend': False,
            'no_pfcesoma_to_srpv': False,
            'no_srsst_to_srvip': False,
            'sr_sst_high_bias': False,
            'fdbk_to_vip': False,
            'exc_to_vip': False,
            'dend_nonlinearity': 'old',    # old, v2, v3
            'trainable_dend2soma': False,
#             'divisive_dend_inh': False,
#             'divisive_dend_ei': False,
#             'divisive_dend_nonlinear': False,
            'dendrite_type': 'additive',    # none/additive/divisive_nonlinear/divisive_ei/divisive_inh
#             'scale_down_init_wexc': False,
            'grad_remove_history': True,
            'plot_during_training': True,
            'structured_sr_sst_to_sr_edend': False,
            'structured_sr_sst_to_sr_edend_branch_specific': False,
            'sparse_pfcesoma_to_srvip': 0,
            'sparse_srsst_to_sredend': 0.8,
            'pos_wout': False,    # whether the readout weight for response is positive
            'pos_wout_rule': False,    # whether the readout weight for rule is positive
            'task': 'wcst',
            'jobname': 'testjob',    # determined by the batch file
            'timeit_print': False,
            'resp_cue': False,    # whether or not to have an external cue to indicate the start of response (might be important for generalization across dt)
            'torch_seed': 1,
            'bpx1tr': False,
            'record_recent_rnn_activity': False,
            'check_explode_cg': False,
            'scale_down_wexc_init': True    # scale down the outgoing weights from exc cells at initialization to prevent exploding dynamics
            }



#     if hp['optimizer']=='adam':
#         optimizer = torch.optim.Adam
#     elif hp['optimizer']=='Rprop':
#         optimizer = torch.optim.Rprop
#     else:
#         raise NotImplementedError
    optimizer = hp['optimizer']

    if hp['loss_type']=='mse':
        loss_fnc = nn.MSELoss()
    else:
        raise NotImplementedError

    return hp, optimizer, loss_fnc