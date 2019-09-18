from sacred import Experiment

ex = Experiment(
    "Learning Dynamics of Attention: Human Prior for Interpretable Machine Reasoning"
)


@ex.config
def config():
    root = None
    task_root = None
    dataset_name = None
    load_seed = None
    print_freq = 10

    workers = 8
    epochs = 256
    epoch = 0
    max_step = 12
    data_fraction = 1.0
    penalty_lambda = 0.1
    humans_with_original = False

    dim = 512
    start_epoch = 0
    batch_size = 64
    val_batch_size = 64
    learning_rate = 1e-4
    momentum = 0.9
    weight_decay = 0
    grad_clip = 8

    deterministic = False
    solver_tol = 1e-3
    read_dropout = 0.15
    img_enc_dropout = 0.18
    emb_dropout = 0.15
    question_dropout = 0.08
    classifier_dropout = 0.15

    embed_hidden = 300
    use_daft = False
    debug = False

    visualization_split = "val"


@ex.named_config
def clevr():
    task_root = "clevr"
    dataset_name = "clevr"


@ex.named_config
def gqa():
    task_root = "gqa"
    dataset_name = "gqa"
    img_enc_dropout = 0.2
    emb_dropout = 0.2
