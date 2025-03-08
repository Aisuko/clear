import os, re, random

import pickle
import torch
import numpy as np
import pandas as pd
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


from transformers import AutoTokenizer


def timeseries_notes_collate(batch, max_len, padding_token_id):
    (
        visit_occurrences,
        person_ids,
        note_type_concept_ids,
        note_datetimes,
        note_texts,
        note_texts_tokenized,
        death_times,
        visit_start_datetimes,
        visit_end_datetimes,
        seq_len,
    ) = zip(*batch)

    for idx in range(len(note_texts_tokenized)):
        note_texts_tokenized[idx] = torch.LongTensor()

    note_texts_tokenized = pad_sequence(
        note_texts_tokenized, batch_first=True, padding_value=padding_token_id
    )

    output_dic = {
        "visit_occurrences": visit_occurrences,
        "person_ids": person_ids,
        "note_type_concept_ids": note_type_concept_ids,
        "note_datetime": note_datetimes,
        "note_texts": note_texts,
        "note_texts_tokenized": note_texts_tokenized,
        "death_times": death_times,
        "visit_start_datetime": visit_start_datetimes,
        "visit_end_datetime": visit_end_datetimes,
        "seq_len": torch.LongTensor(seq_len),
    }

    return output_dic


def concat_notes_collate(batch, max_len, padding_token_id):
    (
        stay_ids,
        note_texts,
        note_texts_tokenized,
        seq_len,
        sample_idx,
    ) = zip(*batch)

    note_texts = [" ".join(nt) for nt in note_texts]

    note_texts_tokenized = pad_sequence(
        note_texts_tokenized, batch_first=True, padding_value=padding_token_id
    )

    sample_idx = torch.LongTensor(sample_idx)

    output_dic = {
        "stay_ids": stay_ids,
        "note_texts": note_texts,
        "note_texts_tokenized": note_texts_tokenized,
        "seq_len": torch.LongTensor(seq_len),
        "sample_idx": sample_idx,
    }

    return output_dic


class MIMICIIINoteDataset(Dataset):
    """
    The fusion is achieved implicitly by alignging the note data with
    the time-series(measurement) data using the common ICU stay ID.

    1. Alinging by ICU stay
    2. Random Note Selection
    3. Note processing and tokenization
    4. Retrieving Time-series data
    5. Fusion for training

    """

    def __init__(
        self,
        root,
        max_window_size=None,
        max_seq_len=256,
        transform=None,
        split="train",
        collate_fn=concat_notes_collate,
        mask_rate=0,
        exclude_concept_ids=[],
        max_instances=5,
        used_note_types=None,
        measurement_dataset=None,
    ):
        self.modality = "note"
        self.exclude_concept_ids = exclude_concept_ids
        self.measurement_dataset = measurement_dataset

        if used_note_types is None:
            self.used_note_types = [
                "Echo",
                "ECG",
                "Nursing",
                "Physician ",
                "Rehab Services",
                "Case Management ",
                "Respiratory ",
                "Nutrition",
                "General",
                "Social Work",
                "Pharmacy",
                "Consult",
                "Radiology",
                "Nursing/other",
            ]
        else:
            self.used_note_types = used_note_types

        self.tokenizer = AutoTokenizer.from_pretrained(
            "emilyalsentzer/Bio_ClinicalBERT"
        )
        self.max_seq_len = max_seq_len
        self.collate_fn = collate_fn
        self.mask_rate = mask_rate
        self.transform = transform
        self.max_instances = max_instances

        self.load_data(root)

        self.representations = {}
        self.split = split

        self.stay_ids = list(self.data["ICUSTAY_ID"].unique())

    def load_data(self, root):
        self.data = pd.read_csv(os.path.join(root, "NOTEEVENTS.csv"), low_memory=False)

        stay_data = pd.read_csv(
            os.path.join(root, "ICUSTAYS.csv"),
            usecols=[
                "HADM_ID",
                "ICUSTAY_ID",
                "INTIME",
                "OUTTIME",
            ],
        )

        self.data = self.data.merge(stay_data, on="HADM_ID").drop_duplicates("ROW_ID")

        original_data = self.data

        discharge_data = self.data[self.data["CATEGORY"] == "Discharge summary"]

        self.data = self.data[
            (self.data["CHARTTIME"] >= self.data["INTIME"])
            & (self.data["CHARTTIME"] <= self.data["OUTTIME"])
        ]

        self.data = self.data[self.data["CATEGORY"].isin(self.used_note_types)]
        self.data = pd.concat([self.data, discharge_data], axis=0)
        self.data = self.data[
            self.data["ICUSTAY_ID"].isin(self.measurement_dataset.stay_ids)
        ].reset_index(drop=True)

    def transform_vo_frame(self, df):
        new_frame = pd.DataFrame(
            np.empty((self.metadata.index.shape[0], 1)),
            indices=self.metadata.index,
            columns=["measurement_datetime"],
        )

        for row_idx, row in df.iterrows():
            new_frame[row_idx] = row

        return new_frame

    def __len__(self):
        """
        It is a internal method of the class that returns the length of the dataset.

        self.stay_ids is a list of unique ICU stay IDs.
        """
        return len(self.stay_ids)

    def __getitem__(self, idx):
        """
        Internal method of the class that returns a sample from the dataset.
        It is called by PyTorch's DataLoader class. And it will be called for each
        sample in the dataset. It will be called automatically when the DataLoader
        is used to iterate over the dataset.

        Args:
        # idx: the index of the current sample

        Returns:
        # note_text: the original note text
        # indexed_tokens: the tokenized note text
        # measurement_data: the time-series data
        # idx: the index of the current sample
        # measurement_idx: the index of the time-series data
        """
        # get the ICU stay ID for this sample
        stay_id = int(self.stay_ids[idx])

        # retrive all note record corresponding to this ICU stay ID
        data = self.data[self.data["ICUSTAY_ID"] == stay_id].reset_index(drop=True)

        # randomly select one note from the available records
        note_idx = random.choice(list(range(len(data))))
        data = data.iloc[note_idx]
        hadm_id = data["HADM_ID"]

        note_type_concept_id = np.zeros(data.shape[0])
        note_datetime = np.empty((data.shape[0]), dtype="datetime64[s]")
        note_text = []
        note_text_tokenized = []
        representations = []

        # Process the note text: remove extra spaces and underscores
        note_text = data["TEXT"]
        note_text = re.sub(" +", " ", note_text)
        note_text = re.sub("_+", "_", note_text)

        # adding tokenized notes
        indexed_tokens = self.tokenizer.encode(note_text)
        indexed_tokens = [token for token in indexed_tokens if token != 115]
        indexed_tokens = indexed_tokens[: self.max_seq_len]
        indexed_tokens = torch.LongTensor(indexed_tokens)

        # Retrieve the associated time-series (measurement) data for this ICU stay
        if self.measurement_dataset is not None:
            (
                measurement_data,
                measurement_idx,
            ) = self.measurement_dataset.get_by_stayid(stay_id)

        else:
            measurement_data = None

        # Return a tuple of the note text, tokenized note text, measurement data, and the index
        # of the current sample

        return (
            note_text,
            indexed_tokens,
            measurement_data,
            idx,
            measurement_idx,
        )

    def collate(self, batch):
        return self.collate_fn(batch, self.max_seq_len, self.tokenizer.pad_token_id)
    
    def save_to_disk(self, filepath):
        """
        Serialize the dataset instance to a Pickle file on disk.
        """
        try:
            # os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            with open(filepath, "wb") as f:
                pickle.dump(self, f)
            print(f"Dataset instance saved to {filepath}")
        except Exception as e:
            print(f"Error: Directory for path '{filepath}' not found.")
        except pickle.PicklingError:
            print("Error: This dataset contains unpicklable objects and cannot be saved.")
        except Exception as e:
            print(f"Error: An unexpected issue occurred during save: {e}")

    @classmethod
    def load_from_disk(cls, filepath):
        """
        Load a serialized dataset instance from a Pickle file on disk.
        """
        try:
            with open(filepath, "rb") as f:
                obj= pickle.load(f)
            if not isinstance(obj, cls):
                raise TypeError(f"Error: Loaded object is not an instance of {cls.__name__}")
            print(f"Dataset instance loaded from {filepath}")
            return obj
        except FileNotFoundError:
            print(f"Error: File '{filepath}' not found.")
        except pickle.UnpicklingError:
            print("Error: Failed to unpickle data (file may be corrupted or not a valid Pickle).")
        except Exception as e:
            print(f"Error: An unexpected issue occurred during load: {e}")
