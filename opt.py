import argparse

class Opt:
    parser = argparse.ArgumentParser()

    # model
    parser.add_argument('--model', type=str, default='CNN', help='BERT, BILSTM, ELECTRA, CNN')
    parser.add_argument('--sent_embedding', type=int, default=0, help='0: CLS, 1: 4-layer concat')
    parser.add_argument('--hidden_dim', type=int, default=768, help='BERT or ELECTRA: hidden dimension of classifier, BILSTM: hidden dimension of lstm')
    parser.add_argument('--num_layer', type=int, default=2, help='BILSTM: number of layers of lstm')
    parser.add_argument("--embedding_dim", type=int, default=256, help='embedding dimension of CNN')
    parser.add_argument("--kernel_sizes", nargs='+', default=[3, 4, 5], type=int, help='kernel sizes of CNN')
    parser.add_argument("--kernel_depth", default=500, type=int, help='kernel depth of CNN')
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout ratio')

    # training
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--gpu', type=int, default=0, help='0,1,2,3')
    parser.add_argument('--max_epoch', type=int, default=50)
    parser.add_argument('--save', type=int, default=1, help='0: false, 1:true')
    parser.add_argument('--lr_pretrained', type=float, default=1e-05, help='learning rate, 5e-5, 3e-5 or 2e-5')
    parser.add_argument('--lr_clf', type=float, default=0.0001, help='learning rate, 5e-5, 3e-5 or 2e-5')
    parser.add_argument('--freeze_pretrained', type=int, default=0, help='0: false, 1:true')
    parser.add_argument('--eps', type=float, default=1e-8, help='epsilon for AdamW, 1e-8')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay for AdamW, 5e-4')

    # dataset
    parser.add_argument('--data_path', type=str, default='./Dataset')
    parser.add_argument('--save_model_path', type=str, default='./Saved_models')
    parser.add_argument('--save_submission_path', type=str, default='./Submissions')
    parser.add_argument('--max_len', type=int, default=50, help='max length of the sentence')
    parser.add_argument('--aug', type=int, default=0, help='0: false, 1: true(ru)')
    parser.add_argument('--split_ratio', type=int, default=1, help='k/10, k in [1,2,3]')
    parser.add_argument('--author', type=str, default='jh')