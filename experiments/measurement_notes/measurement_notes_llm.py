import os
import sys
import warnings
import wandb

import numpy as np

import torch
from torch import optim
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision import transforms

from torchmimic.utils import pad_colalte
from torchmimic.data.preprocessing import Normalizer
from torchmimic.data import IHMDataset
from torchmimic.loggers import IHMLogger

from llm_argparser import parser

currDir = os.path.dirname(os.path.realpath(__file__))
rootDir = os.path.abspath(os.path.join(currDir, "../.."))

if rootDir not in sys.path:  # add parent dir to paths
    sys.path.append(rootDir)


warnings.filterwarnings("ignore", category=ResourceWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from multimodal_clinical_pretraining.data.utils import ShuffleTransform
from multimodal_clinical_pretraining.utils import load_pretrained_model
from multimodal_clinical_pretraining.scheduler import create_scheduler
from multimodal_clinical_pretraining.models import create_model

from sklearn.metrics import (
    f1_score,
    roc_auc_score,
    accuracy_score,
    balanced_accuracy_score,
    mean_absolute_error
)


def balanced_accuracy(true, pred):
    """
    Returns the Balanced Accuracy for the provided true and predicted values

    :param true: true values
    :type true: np.array
    :param pred: predicted values
    :type pred: np.array
    :return: Balanced Accuracy score
    :rtype: int
    """
    return balanced_accuracy_score(true, pred)

def mae(true, pred):
    """
    Returns the Mean Absolute Error/Deviation for the provided
    true and predicted values

    :param true: true values
    :type true: np.array
    :param pred: predicted values
    :type pred: np.array
    :return: MAE/MAD score
    :rtype: int
    """
    one_hot = np.zeros((true.size, true.max() + 1))
    for i in np.arange(true.size):
        one_hot[np.arange(true.size), true[i]] = 1
    return mean_absolute_error(one_hot, pred)

class ClinicalMultiModal(nn.Module):
    def __init__(self, base_model, measurement_emb_size, n_classes):
        super(ClinicalMultiModal, self).__init__()
        self.base_model=base_model
        self.classifier=nn.Linear(measurement_emb_size, n_classes)
        self.classifier.weight.data.normal_(mean=0.0, std=0.01)
        self.classifier.bias.data.zero_()
    def forward(self,x):
        features=self.base_model(x)
        logits=self.classifier(features[:,0,:])
        return logits
    

def evaluate_model(model, dataloader, device, criterion, logger, epoch, wandb, threshold=0.5, task="IHM", use_measurements=True):
    """
    Evaluate the model on the provided DataLoader.

    Args:
        model (torch.nn.Module): The model to evaluate.
        dataloader (torch.utils.data.DataLoader): DataLoader for the evaluation dataset.
        device (torch.device): Device to perform evaluation on.
        criterion (torch.nn.Module): Loss function.
        logger (Logger): Logger to track metrics.
        threshold (float, optional): Threshold for binary classification. Defaults to 0.5.
        task (str, optional): Specific task identifier. Defaults to "IHM".
        use_measurements (bool, optional): Flag to use measurement data. Defaults to True.

    Returns:
        dict: Dictionary containing evaluation metrics.
    """
    model.eval()
    ys = []
    preds = []

    with torch.no_grad():
        for values, labels, seq_lengths, _ in dataloader:
            input_list = []

            # Prepare measurement data
            measurement_x = values.to(device)
            labels = labels.to(device)

            if use_measurements:
                input_list.append({"x": measurement_x})

            logits = model(input_list)
            # logits = torch.sigmoid(logits)

            if task == "IHM":
                logits = logits[:, 0]

            y = labels
            loss = criterion(logits, y)
            pred = logits

            ys.extend(y.detach().cpu().numpy())
            preds.extend(pred.detach().cpu().numpy())

        binary_preds = (1/(1+np.exp(-np.array(preds))) > 0.5).astype(int)

        wandb.log(
            {"Test Epoch": epoch, 
            "Test Loss": loss.item(),
            "Test AUC-ROC": roc_auc_score(ys, preds), 
            "Test Accuracy": accuracy_score(ys, binary_preds), 
            "Test F1": f1_score(ys, binary_preds, average="weighted")
            })
            
        logger.update(pred, y, loss)

    logger.print_metrics(epoch,split="Test")

    # preds = np.array(preds)
    # binary_preds = (preds > threshold).astype(int)  # Apply threshold for binary classification

    # Compute precision, recall, and F1-score
    # accuracy = accuracy_score(ys, preds)
    # f1=f1_score(ys,preds)
    # auc_roc=roc_auc_score(ys,preds,multi_class="ovr", average=None)


def train(args, train_dataloader, test_dataloader):

    wandb.login(key=os.getenv("WANDB_API_KEY"))
    exp_path = os.path.join("exp_outputs", args.exp_name)

    wandb.init(
        project="Multimodal Clinical down-streaming task",
        name=args.exp_name,
        config=args,
        save_code=True,
        settings=wandb.Settings(code_dir=exp_path),
    )

    base_model = create_model(args)

    if args.pretrained_path is not None:
        base_model = load_pretrained_model(base_model, args)

    model = ClinicalMultiModal(
        base_model,
        args.measurement_emb_size,
        args.n_classes
    ).to(args.device)


    params = model.classifier.parameters()
    model.base_model.eval()

    optimizer = optim.AdamW(
        params,
        lr=args.lr,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=args.weight_decay,
    )

    lr_scheduler, _ = create_scheduler(args, optimizer)

    if args.task == "IHM":
        logger = IHMLogger(args.exp_name, args, log_wandb=False)

    # criteria = nn.BCELoss()
    criteria = nn.BCEWithLogitsLoss()

    for epoch in range(args.epochs):
        ys = []
        preds = []
        print(f"LR = {optimizer.param_groups[0]['lr']}")
        
        model.classifier.train()
        logger.reset()

        for values, labels, seq_lengths, _ in train_dataloader:
            optimizer.zero_grad()

            # Prepare measurement data
            measurement_x = values.to(args.device)
            labels = labels.to(args.device)

            input_list = []

            if args.use_measurements:
                input_list.append(
                    {
                        "x": measurement_x,
                    }
                )

            seq_lengths = torch.LongTensor(seq_lengths)
            logits = model(input_list)
            # logits = F.sigmoid(logits)

            if args.task == "IHM":
                logits = logits[:, 0]

            y = labels

            loss = criteria(logits, y)

            loss.backward()
            optimizer.step()

            pred = logits

            ys.extend(y.detach().cpu().numpy())
            preds.extend(pred.detach().cpu().numpy())

            binary_preds = (1/(1+np.exp(-np.array(preds))) > 0.5).astype(int)

            wandb.log(
                {"Training Epoch": epoch,
                #  "Training batch": len(ys),
                 "Training Loss": loss.item(), 
                 "Training AUC-ROC": roc_auc_score(ys, preds), 
                 "Training Accuracy": accuracy_score(ys, binary_preds), 
                 "Training F1": f1_score(ys, binary_preds, average="weighted")
                })

            logger.update(pred, y, loss)

        lr_scheduler.step(epoch + 1)
        logger.print_metrics(epoch, split="Train")
        logger.reset()

        # preds=np.array(preds)
        # I do not think it is a good idea to convert the predictions to binary
        # binary_preds=(preds>=0.5).astype(int)

        # accuracy=accuracy_score(ys,preds)
        # f1=f1_score(ys,preds)

        evaluate_model(model,test_dataloader,args.device,criteria,logger,epoch, wandb)
    
    wandb.finish()
        

if __name__ == "__main__":
    activation_map = {"GELU": nn.GELU(), "ReLU": nn.ReLU()}
    args = parser()
    args.measurement_activation = activation_map[args.measurement_activation]

    args.use_projector = False

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.deterministic = True

    train_measurement_transform = transforms.Compose(
        [
            ShuffleTransform(args.measurement_max_seq_len),
        ]
    )

    test_measurement_transform = transforms.Compose(
        [
            ShuffleTransform(args.measurement_max_seq_len),
        ]
    )

    listfile=""
    if args.task == "IHM":
        root = os.path.join("/workspaces/multimodal-mimic/", "in-hospital-mortality-6")
        train_listfile = listfile + "train_listfile.csv"
        test_listfile = "test_listfile.csv"
        train_dataset = IHMDataset(
            root, customListFile=os.path.join(root, train_listfile), train=True
        )
        test_dataset = IHMDataset(
            root, customListFile=os.path.join(root, test_listfile), train=False
        )

    discretizer_header = train_dataset.discretizer.transform(
        train_dataset.reader.read_example(0)["X"]
    )[1].split(",")
    cont_channels = [
        i for (i, x) in enumerate(discretizer_header) if x.find("->") == -1
    ]

    train_dataset.normalizer = Normalizer(fields=cont_channels)
    train_dataset.normalizer.load_params(
        "./multimodal_clinical_pretraining/resources/normalizer_params"
    )

    test_dataset.normalizer = Normalizer(fields=cont_channels)
    test_dataset.normalizer.load_params(
        "./multimodal_clinical_pretraining/resources/normalizer_params"
    )

    print(f"Length of training dataset = {len(train_dataset)}")
    print(f"Length of test dataset = {len(test_dataset)}")

    args.n = len(train_dataset)
    args.n_features = 76

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=0,
        collate_fn=pad_colalte,
        pin_memory=True,
        shuffle=True,
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size * 2,
        num_workers=0,
        collate_fn=pad_colalte,
        pin_memory=True,
        shuffle=False,
    )

    args.use_measurements = True
    args.use_notes = True
    args.vocab_size = 30522
    # tokenizer=AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT") # tokenizer for notes
    args.pad_token_id = 0

    if args.task == "IHM":
        args.n_classes = 1

    torch.manual_seed(42)
    np.random.seed(42)

    output = train(
        args,
        train_dataloader,
        test_dataloader,
    )
