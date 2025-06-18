from google.colab import drive
drive.mount('/content/drive')

!pip install jamo jiwer jamotools

!pip install --upgrade transformers

from jamo import hangul_to_jamo
from jamo import h2j, j2hcj
import re

def text_to_jamo(text):
    return ''.join(j2hcj(h2j(text)))

def clean_jamo_text(jamo_text):
    return re.sub(r'[^\u3131-\u3163\u1100-\u11FF\uAC00-\uD7A3 ]', '', jamo_text)

import os
import pandas as pd
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from datasets import Dataset, Audio
from transformers import AutoProcessor, Wav2Vec2ForCTC, AutoModelForCTC

processor = AutoProcessor.from_pretrained("student-47/wav2vec2-large-xlrs-korean-v5")
model = AutoModelForCTC.from_pretrained("student-47/wav2vec2-large-xlrs-korean-v5")

config = model.config
print(config)

# Read data
df = pd.read_csv('./labels_speech_recognition_final.csv')
#df['audio_path'] = '../../' + df['path']
#data = df[df['audio_path'].apply(os.path.exists)].copy()
df['audio_path'] = './augmented/' + df['path'].astype(str) + '.wav'

# Filter out rows where the audio file does not exist
data = df[df['audio_path'].apply(os.path.exists)].copy()

# Label code
le_gender = LabelEncoder().fit(data['gender'])
le_age = LabelEncoder().fit(data['age'])
le_accent = LabelEncoder().fit(data['accents'])

data['gender_label'] = le_gender.transform(data['gender'])
data['age_label'] = le_age.transform(data['age'])
data['accent_label'] = le_accent.transform(data['accents'])

train_val, test = train_test_split(data, test_size=0.1, random_state=47)
train, val = train_test_split(train_val, test_size=2/9, random_state=47)

print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")

# ====== Convert to Huggingface Dataset and preprocess ======
train_dataset = Dataset.from_pandas(train.reset_index(drop=True))
val_dataset = Dataset.from_pandas(val.reset_index(drop=True))
test_dataset = Dataset.from_pandas(test.reset_index(drop=True))

# NOTE: The audio path field is now audio_path
train_dataset = train_dataset.cast_column("audio_path", Audio(sampling_rate=16000))
val_dataset = val_dataset.cast_column("audio_path", Audio(sampling_rate=16000))
test_dataset = test_dataset.cast_column("audio_path", Audio(sampling_rate=16000))

MAX_AUDIO_LEN = 16000 * 10

def preprocess_function(batch):
    audio = batch["audio_path"]
    #audio_array = audio["array"][:MAX_AUDIO_LEN]
    audio_array = audio["array"]
    jamo_text = text_to_jamo(batch["sentence"])
    jamo_text = clean_jamo_text(jamo_text)
    inputs = processor(
        [audio_array],
        sampling_rate=audio["sampling_rate"],
        text=[jamo_text],
        return_tensors="pt",
        padding="longest"
    )

    input_values = inputs.input_values[0]
    attention_mask = inputs.attention_mask[0]
    labels = inputs.labels[0]
    return {
        "input_values": input_values,
        "attention_mask": attention_mask,
        "labels": labels,
        "age_label": batch["age_label"],
        "gender_label": batch["gender_label"],
        "accent_label": batch["accent_label"]
    }

processed_train = train_dataset.map(preprocess_function, remove_columns=train_dataset.column_names)
processed_val = val_dataset.map(preprocess_function, remove_columns=val_dataset.column_names)
processed_test = test_dataset.map(preprocess_function, remove_columns=test_dataset.column_names)

# Data loader
from torch.utils.data import DataLoader

def collate_fn(batch):
    input_values = [torch.tensor(item['input_values']) for item in batch]
    attention_mask = [torch.tensor(item['attention_mask']) for item in batch]
    labels = [torch.tensor(item['labels']) for item in batch]
    age_labels = torch.tensor([item['age_label'] for item in batch], dtype=torch.long)
    gender_labels = torch.tensor([item['gender_label'] for item in batch], dtype=torch.long)
    accent_labels = torch.tensor([item['accent_label'] for item in batch], dtype=torch.long)

    input_values = torch.nn.utils.rnn.pad_sequence(input_values, batch_first=True)
    attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True)
    labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)

    return {
        'input_values': input_values,
        'attention_mask': attention_mask,
        'labels': labels,
        'age_label': age_labels,
        'gender_label': gender_labels,
        'accent_label': accent_labels
    }

train_batch_size = 8
eval_batch_size = 8

train_loader = DataLoader(processed_train, batch_size=train_batch_size, shuffle=True, collate_fn=collate_fn, pin_memory=True, num_workers=4)
val_loader = DataLoader(processed_val, batch_size=eval_batch_size, shuffle=False, collate_fn=collate_fn, pin_memory=True, num_workers=4)
test_loader = DataLoader(processed_test, batch_size=eval_batch_size, shuffle=False, collate_fn=collate_fn, pin_memory=True, num_workers=4)

import torch
import torch.nn as nn
from transformers import Wav2Vec2ForCTC

class MultiTaskWav2Vec2(nn.Module):
    def __init__(self, base_model_name, num_age, num_gender, num_accent):
        super().__init__()
        self.asr = Wav2Vec2ForCTC.from_pretrained(
            base_model_name,
            # --- SpecAugment Setting ---
            mask_time_prob=0.05,        # The fraction of time steps to which the temporal mask is applied
            mask_time_length=10,        # The average length of each temporal mask
            mask_feature_prob=0.05,      # The proportion of frequency channels to apply the frequency mask to
            mask_feature_length=64,      # The average length of each frequency mask
        )

        hidden_size = self.asr.config.hidden_size

        self.age_head = nn.Sequential(
            nn.Linear(hidden_size, 128), nn.ReLU(), nn.Linear(128, num_age)
        )
        self.gender_head = nn.Sequential(
            nn.Linear(hidden_size, 64), nn.ReLU(), nn.Linear(64, num_gender)
        )
        self.accent_head = nn.Sequential(
            nn.Linear(hidden_size, 128), nn.ReLU(), nn.Linear(128, num_accent)
        )

    def forward(self, input_values, attention_mask=None):
        outputs = self.asr.wav2vec2(input_values, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state  # [B, T, H]
        pooled = hidden_states.mean(dim=1)  # [B, H]
        age_logits = self.age_head(pooled)
        gender_logits = self.gender_head(pooled)
        accent_logits = self.accent_head(pooled)
        ctc_logits = self.asr.lm_head(hidden_states)
        return {
            'logits': ctc_logits,
            'age_logits': age_logits,
            'gender_logits': gender_logits,
            'accent_logits': accent_logits,
        }

import torch.nn.functional as F
import pandas as pd
import jiwer
from jiwer import wer
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm.auto import tqdm
from typing import List

def manual_compute_measures(references, predictions, show_progress=False):
    if len(references) != len(predictions):
        return {
            'wer': float('inf'),
            'substitutions': 0,
            'deletions': 0,
            'insertions': 0,
            'hits': 0
        }

    total_substitutions = 0
    total_deletions = 0
    total_insertions = 0
    total_hits = 0
    total_ref_words = 0

    # Decide whether to wrap the loop based on the show_progress parameter
    # Creating a basic iterator
    iterator = zip(references, predictions)
    # If necessary, wrap it with tqdm
    if show_progress:
        iterator = tqdm(iterator, total=len(references), desc="Caculate WER", unit="sentence")

    #for ref, hyp in zip(references, predictions):
    for ref, hyp in iterator:
        ref_words = ref.strip().split()
        hyp_words = hyp.strip().split()

        total_ref_words += len(ref_words)

        if len(ref_words) == 0:
            total_insertions += len(hyp_words)
            continue

        # Dynamic programming to calculate edit distance and operation type
        d = [[0] * (len(hyp_words) + 1) for _ in range(len(ref_words) + 1)]

        # Initialize Bounds
        for i in range(len(ref_words) + 1):
            d[i][0] = i
        for j in range(len(hyp_words) + 1):
            d[0][j] = j

        # Filling the Matrix
        for i in range(1, len(ref_words) + 1):
            for j in range(1, len(hyp_words) + 1):
                if ref_words[i-1] == hyp_words[j-1]:
                    d[i][j] = d[i-1][j-1]       # Match, no action
                else:
                    d[i][j] = min(
                        d[i-1][j] + 1,      # Delete
                        d[i][j-1] + 1,      # Insert
                        d[i-1][j-1] + 1      # Replace
                    )

        # Backtrack to calculate the specific number of operations
        i, j = len(ref_words), len(hyp_words)
        while i > 0 or j > 0:
            if i > 0 and j > 0 and ref_words[i-1] == hyp_words[j-1]:
                # Match
                total_hits += 1
                i -= 1
                j -= 1
            elif i > 0 and j > 0 and d[i][j] == d[i-1][j-1] + 1:
                # Replace
                total_substitutions += 1
                i -= 1
                j -= 1
            elif i > 0 and d[i][j] == d[i-1][j] + 1:
                # Delete
                total_deletions += 1
                i -= 1
            elif j > 0 and d[i][j] == d[i][j-1] + 1:
                # Insert
                total_insertions += 1
                j -= 1
            else:
                break

    # Calculating WER
    wer = (total_substitutions + total_deletions + total_insertions) / total_ref_words if total_ref_words > 0 else float('inf')

    return {
        'wer': wer,
        'substitutions': total_substitutions,
        'deletions': total_deletions,
        'insertions': total_insertions,
        'hits': total_hits
    }

def calculate_wer(predictions: List[str], references: List[str], show_progress: bool = False) -> dict:
    if len(predictions) != len(references) or len(predictions) == 0:
        return {
            'wer': float('inf'),
            'substitutions': 0,
            'deletions': 0,
            'insertions': 0,
            'hits': 0
        }

    # Clear data
    clean_predictions = []
    clean_references = []

    for pred, ref in zip(predictions, references):
        pred_clean = str(pred).strip() if pred is not None else ""
        ref_clean = str(ref).strip() if ref is not None else ""

        if pred_clean and ref_clean:
            clean_predictions.append(pred_clean)
            clean_references.append(ref_clean)

    if len(clean_predictions) == 0:
        return {
            'wer': float('inf'),
            'substitutions': 0,
            'deletions': 0,
            'insertions': 0,
            'hits': 0
        }

    try:
        measures = {}
        # Add a "wait-style" progress bar to jiwer
        if show_progress:
            with tqdm(total=1, desc="Use jiwer to cacluate WER") as pbar:
                measures = jiwer.compute_measures(clean_references, clean_predictions)
                pbar.update(1)
        else:
            # If the progress bar is not displayed, the calculation is performed directly
            measures = jiwer.compute_measures(clean_references, clean_predictions)
        # Calculate detailed indicators
        #measures = jiwer.compute_measures(clean_references, clean_predictions)

        return {
            'wer': measures['wer'],
            'substitutions': measures['substitutions'],
            'deletions': measures['deletions'],
            'insertions': measures['insertions'],
            'hits': measures['hits']
        }

    except Exception as e:
        print(f"jiwer failed, using manual calculation: {e}")
        return manual_compute_measures(clean_references, clean_predictions, show_progress=show_progress)
        #return manual_compute_measures(clean_references, clean_predictions)

def compute_loss(batch, outputs, processor, epoch):
    batch_size, max_seq_len, _ = outputs['logits'].shape
    input_lengths = torch.full(
        size=(batch_size,),
        fill_value=max_seq_len,
        dtype=torch.long,
        device=outputs['logits'].device
    )
    target_lengths = (batch['labels'] != -100).sum(dim=1)
    labels_flattened = []
    for i in range(batch_size):
        valid = batch['labels'][i][batch['labels'][i] != -100]
        labels_flattened.append(valid)
    labels_flattened = torch.cat(labels_flattened)

    # CTC Loss
    ctc_loss = F.ctc_loss(
        outputs['logits'].log_softmax(-1).transpose(0, 1),
        labels_flattened,
        input_lengths=input_lengths,
        target_lengths=target_lengths,
        blank=processor.tokenizer.pad_token_id
    )

    # Classification Loss
    age_loss = F.cross_entropy(outputs['age_logits'], batch['age_label'])
    gender_loss = F.cross_entropy(outputs['gender_logits'], batch['gender_label'])
    accent_loss = F.cross_entropy(outputs['accent_logits'], batch['accent_label'])

    # Dynamic task weights
    if 'task_weights' in outputs:
        # Perform softmax normalization on task weights
        task_weights = F.softmax(outputs['task_weights'], dim=0)

        # Weighted auxiliary task loss
        aux_loss = (task_weights[0] * age_loss +
                   task_weights[1] * gender_loss +
                   task_weights[2] * accent_loss)

        # Total loss = CTC loss + weighted auxiliary loss
        total_loss = ctc_loss + aux_loss

        # Print weight information (for debugging)
        if epoch % 10 == 0:
            print(f"Task weights - Age: {task_weights[0]:.3f}, "
                  f"Gender: {task_weights[1]:.3f}, "
                  f"Accent: {task_weights[2]:.3f}")
    else:
        # Fixed weights as a fallback
        total_loss = ctc_loss + 0.5 * age_loss + 0.5 * gender_loss + 0.5 * accent_loss

    return total_loss, ctc_loss, age_loss, gender_loss, accent_loss

def train_one_epoch(model, dataloader, optimizer, processor, epoch, lr_scheduler, gradient_accumulation_steps, device):
    model.train()
    total_loss = 0
    step = 0
    accumulated_step = 0

    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1} Training", unit="batch")

    #for batch in dataloader:
    for batch in progress_bar:
        torch.cuda.empty_cache()
        step += 1
        for k in batch:
            if isinstance(batch[k], torch.Tensor):
                batch[k] = batch[k].to(device)
        outputs = model(
            input_values=batch['input_values'],
            attention_mask=batch['attention_mask']
        )

        try:
            loss, xlsr_loss, age_loss, gender_loss, accent_loss = compute_loss(batch, outputs, processor, epoch)
                # Gradient accumulation: loss divided by the number of accumulated steps
            loss = loss / gradient_accumulation_steps
        except RuntimeError as e:
            if "Expected tensor to have size" in str(e):
                print(f"Skip abnormal batch: {e}")
                continue
            else: raise

        loss.backward()

        # Gradient accumulation: updated every gradient_accumulation_steps steps
        if step % gradient_accumulation_steps == 0:
            # Optimizer Updates
            optimizer.step()

            # Learning rate scheduler update
            lr_scheduler.step()

            # Zero gradient
            optimizer.zero_grad()

            accumulated_step += 1

        # Cumulative loss (recovering the true loss value)
        total_loss += loss.item() * gradient_accumulation_steps

    # If there are still unupdated gradients at the end
    if step % gradient_accumulation_steps != 0:
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        accumulated_step += 1

    res = total_loss / step

    '''try:
            loss, xlsr_loss, age_loss, gender_loss, accent_loss = compute_loss(batch, outputs, processor, epoch)
        except RuntimeError as e:
            if "Expected tensor to have size" in str(e):
                print(f"Skip abnormal batch: {e}")
                continue
            else: raise

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        res = total_loss / len(dataloader)'''

    return {
        'train_loss': res,
        'xlsr_train_loss': xlsr_loss.item(),
        'age_train_loss': age_loss.item(),
        'gender_train_loss': gender_loss.item(),
        'accent_train_loss': accent_loss.item(),
        'step': step
    }

def evaluate(model, dataloader, processor, epoch, device, detailed_analysis=False):
    model.eval()
    total_loss = 0
    correct_age, correct_gender, correct_accent = 0, 0, 0
    total = 0
    all_predictions, all_references = [], []

    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1} Evaluation", unit="batch")

    with torch.no_grad():
        #for batch in dataloader:
        for batch in progress_bar:
            for k in batch:
                if isinstance(batch[k], torch.Tensor):
                    batch[k] = batch[k].to(device)
            outputs = model(
                input_values=batch['input_values'],
                attention_mask=batch['attention_mask']
            )

            try:
                loss, xlsr_loss, age_loss, gender_loss, accent_loss = compute_loss(batch, outputs, processor, epoch)
            except RuntimeError as e:
                if "Expected tensor to have size" in str(e):
                    print(f"Skip abnormal batch: {e}")
                    continue
                else:
                    raise

            total_loss += loss.item()
            correct_age += (outputs['age_logits'].argmax(dim=1) == batch['age_label']).sum().item()
            correct_gender += (outputs['gender_logits'].argmax(dim=1) == batch['gender_label']).sum().item()
            correct_accent += (outputs['accent_logits'].argmax(dim=1) == batch['accent_label']).sum().item()
            total += batch['age_label'].size(0)

            # WER
            pred_ids = outputs['logits'].argmax(dim=-1)
            predictions = processor.batch_decode(pred_ids, skip_special_tokens=True)

            labels = batch['labels'].cpu().numpy()
            references = []
            for label in labels:
                label = label[label != -100]    # Remove padding
                references.append(processor.tokenizer.decode(label, skip_special_tokens=True))

            all_predictions.extend(predictions)
            all_references.extend(references)

    val_loss = total_loss / len(dataloader)
    wer_score = calculate_wer(all_predictions, all_references)

    res = {
        'val_loss': val_loss,
        'xlsr_val_loss': xlsr_loss.item(),
        'age_val_loss': age_loss.item(),
        'gender_val_loss': gender_loss.item(),
        'accent_val_loss': accent_loss.item(),
        'wer': wer_score['wer'],
        'age_acc': correct_age / total,
        'gender_acc': correct_gender / total,
        'accent_acc': correct_accent / total
    }

    if detailed_analysis:
        detailed_metrics = calculate_wer(all_predictions, all_references, show_progress=True)
        res['detailed_wer'] = detailed_metrics

        print(f"\n=== Detailed WER Analysis (Epoch {epoch+1}) ===")
        print(f"WER: {detailed_metrics['wer']:.4f}")
        print(f"Substitutions: {detailed_metrics['substitutions']}")
        print(f"Deletions: {detailed_metrics['deletions']}")
        print(f"Insertions: {detailed_metrics['insertions']}")
        print(f"Hits: {detailed_metrics['hits']}")
        print("=" * 50)

    return res

import sys
import time

# ========== Start time ==========
start_time = time.perf_counter()

# ====== Train ======
from torch.optim.lr_scheduler import LinearLR
from torch.cuda.amp import GradScaler, autocast
import random
from torch.optim import Adam
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_age = len(le_age.classes_)
num_gender = len(le_gender.classes_)
num_accent = len(le_accent.classes_)

file_path = "./saved_model/best_model.txt"
save_path = "./saved_model/best_model.pt"

model = MultiTaskWav2Vec2("student-47/wav2vec2-large-xlrs-korean-v5", num_age, num_gender, num_accent)
model.to(device)
model.load_state_dict(torch.load(save_path, map_location=device))
#optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
lr = 0.0001
seed = 47
gradient_accumulation_steps = 2

optimizer = Adam(
    model.parameters(),
    lr=lr,
    betas=(0.9, 0.999),
    eps=1e-08
)

lr_scheduler = LinearLR(
    optimizer,
    start_factor=0.1,
    total_iters=1000
)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(seed)

def print_shape_or_len(arr, name):
    if hasattr(arr, 'shape'):
        print(f"{name} shape:", arr.shape)
    else:
        print(f"{name} len:", len(arr))

num_epochs = 13

best_t_loss = float('inf')
best_t_xlsr_loss = float('inf')
best_t_age_loss = float('inf')
best_t_gender_loss = float('inf')
best_t_accent_loss = float('inf')

best_v_loss = float('inf')
best_v_xlsr_loss = float('inf')
best_v_age_loss = float('inf')
best_v_gender_loss = float('inf')
best_v_accent_loss = float('inf')

best_wer = float('inf')
best_age_acc = 0.0
best_gender_acc = 0.0
best_accent_acc = 0.0
best_epoch = 0
best_step = 0

best_metrics = {}

flag = False
is_best = False

try:
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()

            if not line or line.startswith('#') or line.startswith("'''"): continue

            if line.startswith("Best model at epoch"):
                best_epoch = float(line.split()[-1])
            elif line.startswith("Train Loss:"):
                best_t_loss = float(line.split(":")[1].strip())
            elif line.startswith("Xlsr Train Loss:"):
                best_t_xlsr_loss = float(line.split(":")[1].strip())
            elif line.startswith("Age Train Loss:"):
                best_t_age_loss = float(line.split(":")[1].strip())
            elif line.startswith("Gender Train Loss:"):
                best_t_gender_loss = float(line.split(":")[1].strip())
            elif line.startswith("Accent Train Loss:"):
                best_t_accent_loss = float(line.split(":")[1].strip())

            elif line.startswith("Val Loss:"):
                best_v_loss = float(line.split(":")[1].strip())
            elif line.startswith("Xlsr Val Loss:"):
                best_v_xlsr_loss = float(line.split(":")[1].strip())
            elif line.startswith("Age Val Loss:"):
                best_v_age_loss = float(line.split(":")[1].strip())
            elif line.startswith("Gender Val Loss:"):
                best_v_gender_loss = float(line.split(":")[1].strip())
            elif line.startswith("Accent Val Loss:"):
                best_v_accent_loss = float(line.split(":")[1].strip())

            elif line.startswith("WER:"):
                best_wer = float(line.split(":")[1].strip())
            elif line.startswith("Age Acc:"):
                best_age_acc = float(line.split(":")[1].strip())
            elif line.startswith("Gender Acc:"):
                best_gender_acc = float(line.split(":")[1].strip())
            elif line.startswith("Accent Acc:"):
                best_accent_acc = float(line.split(":")[1].strip())
            elif line.startswith("Step"):
                best_step = int(line.split()[-1])
except FileNotFoundError:
    pass

best_metrics = {
    'train_loss': best_t_loss,
    'xlsr_train_loss': best_t_xlsr_loss,
    'age_train_loss': best_t_age_loss,
    'gender_train_loss': best_t_gender_loss,
    'accent_train_loss': best_t_accent_loss,

    'val_loss': best_v_loss,
    'xlsr_val_loss': best_v_xlsr_loss,
    'age_val_loss': best_v_age_loss,
    'gender_val_loss': best_v_gender_loss,
    'accent_val_loss': best_v_accent_loss,

    'wer': best_wer,
    'age_acc': best_age_acc,
    'gender_acc': best_gender_acc,
    'accent_acc': best_accent_acc,
    'epoch': best_epoch,
    'step': best_step
}

print(best_metrics)

try:
    import jiwer
    jiwer_available = True
except ImportError:
    jiwer_available = False

print(jiwer_available)

import gc
torch.cuda.empty_cache()
gc.collect()

for epoch in range(num_epochs):
    print(f"\nEpoch {epoch+1}")
    sample = processed_train[0]
    train_metrics = train_one_epoch(model, train_loader, optimizer, processor, epoch, lr_scheduler, gradient_accumulation_steps, device)
    detailed = (epoch % 10 == 0)
    val_metrics = evaluate(model, val_loader, processor, epoch, device, detailed_analysis=detailed)

    train_loss = train_metrics['train_loss']
    xlsr_train_loss = train_metrics['xlsr_train_loss']
    age_train_loss = train_metrics['age_train_loss']
    gender_train_loss = train_metrics['gender_train_loss']
    accent_train_loss = train_metrics['accent_train_loss']
    train_step = train_metrics['step']

    val_loss = val_metrics['val_loss']
    xlsr_val_loss = val_metrics['xlsr_val_loss']
    age_val_loss = val_metrics['age_val_loss']
    gender_val_loss = val_metrics['gender_val_loss']
    accent_val_loss = val_metrics['accent_val_loss']

    wer = val_metrics['wer']
    age_acc = val_metrics['age_acc']
    gender_acc = val_metrics['gender_acc']
    accent_acc = val_metrics['accent_acc']

    print(f"Train Loss: {train_metrics['train_loss']:.4f} | "
          f"XLSR Train Loss: {train_metrics['xlsr_train_loss']:.4f} | "
          f"Age Train Loss: {train_metrics['age_train_loss']:.4f} | "
          f"Gender Train Loss: {train_metrics['gender_train_loss']:.4f} | "
          f"Accent Train Loss: {train_metrics['accent_train_loss']:.4f} | "
         )
    print(f"Val Loss: {val_metrics['val_loss']:.4f} | "
          f"XLSR Val Loss: {val_metrics['xlsr_val_loss']:.4f} | "
          f"Age Val Loss: {val_metrics['age_val_loss']:.4f} | "
          f"Gender Val Loss: {val_metrics['gender_val_loss']:.4f} | "
          f"Accent Val Loss: {val_metrics['accent_val_loss']:.4f} | "
          f"WER: {wer:.4f} | "
          f"Age Acc: {val_metrics['age_acc']:.4f} | "
          f"Gender Acc: {val_metrics['gender_acc']:.4f} | "
          f"Accent Acc: {val_metrics['accent_acc']:.4f} | "
        )

    if age_acc > 0.99 and gender_acc > 0.99 and accent_acc > 0.99:
        flag = (wer <= best_wer + 1.0)

    else:
        flag = (
            wer <= best_wer + 1.0 and
            age_acc >= best_age_acc and
            gender_acc >= best_gender_acc and
            accent_acc >= best_accent_acc
        )

    if flag:
        is_best = True

        best_t_loss = train_loss
        best_t_xlsr_loss = xlsr_train_loss
        best_t_age_loss = age_train_loss
        best_t_gender_loss = gender_train_loss
        best_t_accent_loss = accent_train_loss

        best_v_loss = val_loss
        best_v_xlsr_loss = xlsr_val_loss
        best_v_age_loss = age_val_loss
        best_v_gender_loss = gender_val_loss
        best_v_accent_loss = accent_val_loss

        best_wer = wer
        best_age_acc = age_acc
        best_gender_acc = gender_acc
        best_accent_acc = accent_acc
        best_epoch = epoch + 1
        best_step = train_step

        best_metrics = {
            'train_loss': best_t_loss,
            'xlsr_train_loss': best_t_xlsr_loss,
            'age_train_loss': best_t_age_loss,
            'gender_train_loss': best_t_gender_loss,
            'accent_train_loss': best_t_accent_loss,

            'val_loss': best_v_loss,
            'xlsr_val_loss': best_v_xlsr_loss,
            'age_val_loss': best_v_age_loss,
            'gender_val_loss': best_v_gender_loss,
            'accent_val_loss': best_v_accent_loss,

            'wer': best_wer,
            'age_acc': best_age_acc,
            'gender_acc': best_gender_acc,
            'accent_acc': best_accent_acc,
            'epoch': best_epoch,
            'step': best_step
        }
        torch.save(model.state_dict(), save_path)
        print(f"*** Best model saved at epoch {epoch+1} ***")

if is_best:
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(f"Best model at epoch {best_metrics['epoch']}\n\n")
        f.write(f"Train Loss: {best_metrics['train_loss']:.4f}\n")
        f.write(f"XLSR Train Loss: {best_metrics['xlsr_train_loss']:.4f}\n")
        f.write(f"Age Train Loss: {best_metrics['age_train_loss']:.4f}\n")
        f.write(f"Gender Train Loss: {best_metrics['gender_train_loss']:.4f}\n")
        f.write(f"Accent Train Loss: {best_metrics['accent_train_loss']:.4f}\n\n")
        f.write(f"Val Loss: {best_metrics['val_loss']:.4f}\n")
        f.write(f"XLSR Val Loss: {best_metrics['xlsr_val_loss']:.4f}\n")
        f.write(f"Age Val Loss: {best_metrics['age_val_loss']:.4f}\n")
        f.write(f"Gender Val Loss: {best_metrics['gender_val_loss']:.4f}\n")
        f.write(f"Accent Val Loss: {best_metrics['accent_val_loss']:.4f}\n\n")
        f.write(f"WER: {best_metrics['wer']:.4f}\n")
        f.write(f"Age Acc: {best_metrics['age_acc']:.4f}\n")
        f.write(f"Gender Acc: {best_metrics['gender_acc']:.4f}\n")
        f.write(f"Accent Acc: {best_metrics['accent_acc']:.4f}\n\n")
        f.write(f"Step: {best_metrics['step']}\n")

torch.cuda.empty_cache()

end_time = time.perf_counter()
runtime = end_time - start_time
if runtime < 60:
    print(f"Total runtime: {runtime:.2f} seconds")
elif runtime < 3600:
    print(f"Total runtime: {runtime/60:.2f} minutes")
else:
    print(f"Total runtime: {runtime/3600:.2f} hours")
print()

import jamotools

def predict(model, processor, batch, le_age, le_gender, le_accent, device):
    model.eval()
    with torch.no_grad():
        for k in batch:
            if isinstance(batch[k], torch.Tensor):
                batch[k] = batch[k].to(device)
        outputs = model(
            input_values=batch['input_values'],
            attention_mask=batch['attention_mask']
        )
        pred_ids = torch.argmax(outputs['logits'], dim=-1)
        transcriptions = processor.batch_decode(pred_ids, skip_special_tokens=True)

        age_preds = le_age.inverse_transform(outputs['age_logits'].argmax(dim=1).cpu().numpy())
        gender_preds = le_gender.inverse_transform(outputs['gender_logits'].argmax(dim=1).cpu().numpy())
        accent_preds = le_accent.inverse_transform(outputs['accent_logits'].argmax(dim=1).cpu().numpy())

        return transcriptions, age_preds, gender_preds, accent_preds

# Randomly sample 7 indexes
num_samples = 7
total_samples = len(test_dataset)
random_indices = random.sample(range(total_samples), num_samples)

# Sampling and saving raw information
samples = [test_dataset[i] for i in random_indices]
paths = [sample['path'] for sample in samples]
true_texts = [sample['sentence'] for sample in samples]
true_ages = [sample['age'] for sample in samples]
true_genders = [sample['gender'] for sample in samples]
true_accents = [sample['accents'] for sample in samples]

processed_samples = [preprocess_function(sample) for sample in samples]
batch = collate_fn(processed_samples)

model.load_state_dict(torch.load(save_path, map_location=device))
transcriptions, age_preds, gender_preds, accent_preds = predict(
    model, processor, batch, le_age, le_gender, le_accent, device
)

print("=" * 60)
print("TESTING PHASE - RANDOM SAMPLE PREDICTIONS")
print("=" * 60)

for i in range(num_samples):
    print(f"Sample {i+1}:")
    print(f"Path: {paths[i]}\n")
    print()
    print("=== Ground Truth ===")
    print(f"Text   : {true_texts[i]}")
    print(f"Age    : {true_ages[i]}")
    print(f"Gender : {true_genders[i]}")
    print(f"Accent : {true_accents[i]}\n")
    print()
    print("=== Model Prediction ===")
    print(f"Text   : {jamotools.join_jamos(transcriptions[i])}")
    print(f"Jamo   : {transcriptions[i]}")
    print(f"Age    : {age_preds[i]}")
    print(f"Gender : {gender_preds[i]}")
    print(f"Accent : {accent_preds[i]}")

    sample_wer = jiwer.wer(true_texts[i], jamotools.join_jamos(transcriptions[i]))
    print(f"WER    : {sample_wer:.4f}")
    print("-" * 60)

print("Finished!")