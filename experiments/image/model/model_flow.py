from .dense_flow import DenseFlow
from .h_flow import HFlow

def add_model_args(parser):

    # Flow params
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--growth_rate', type=int, default=None)
    parser.add_argument('--checkpointing', action='store_true', default=False)
    parser.add_argument('--block_conf', nargs='+', type=int)
    parser.add_argument('--layer_mid_chnls', nargs='+', type=int)
    parser.add_argument('--layers_conf', nargs='+', type=int)
    parser.add_argument('--beta', type=float, default=100)
    parser.add_argument('--sig2', type=float, default=0.1)
    parser.add_argument('--block_conf_h', nargs='+', type=int)
    parser.add_argument('--layers_conf_h', nargs='+', type=int)
    parser.add_argument('--layer_mid_chnls_h', nargs='+', type=int)
    parser.add_argument('--growth_rate_h', type=int, default=None)


def get_model_id(args):
    return 'densenet-flow'


def get_model(args, data_shape):

    return DenseFlow(
        data_shape=data_shape, block_config=args.block_conf, layers_config=args.layers_conf,
        layer_mid_chnls=args.layer_mid_chnls, growth_rate=args.growth_rate,  checkpointing=args.checkpointing)

def get_model_h(args, data_shape):
    return HFlow(
        data_shape=data_shape, block_config=args.block_conf_h, layers_config=args.layers_conf_h,
        layer_mid_chnls=args.layer_mid_chnls_h, growth_rate=args.growth_rate_h,  checkpointing=args.checkpointing)
