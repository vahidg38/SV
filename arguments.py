import argparse


def get_args():
    parser = argparse.ArgumentParser(description="Autoencoders for data reconstruction",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-bs', '--batch', type=int, default=1,
                        help='Number of samples that will be propagated through the network')

    parser.add_argument('-e', '--epochs', type=int, default=50,
                        help='max num of epochs')

    parser.add_argument('-pt', '--patience', type=int, default=5,
                        help='patienece')

    parser.add_argument('-pa', '--path', type=str, default='IAQ_2month_Vah.mat',
                        help='data path')

    parser.add_argument('-md', '--memdim', type=int, default=200,
                        help='memory dimension')

    parser.add_argument('-mo', '--model', type=str, default='AE',
                        choices=[
                            'AE', 'DAE', 'MAE', 'VAE', 'MVAE'
                        ],
                        help=' model selection')

    return parser.parse_args()
