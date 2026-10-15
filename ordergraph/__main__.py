import argparse
import json
from collections import deque
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.nn import functional as F
from torchvision.models import ResNet50_Weights, resnet50
from torchvision import transforms


class Classifier(nn.Module):
    def __init__(self, classes):
        super().__init__()
        self.encoder = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.encoder.fc = nn.Identity()
        self.head = nn.Linear(2048, classes)

    def forward(self, x):
        z = F.normalize(self.encoder(x), dim=1)
        return self.head(z), z


def images(paths, training, device):
    operations = [transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip()] if training else [transforms.Resize(256), transforms.CenterCrop(224)]
    transform = transforms.Compose(operations + [transforms.ToTensor(), transforms.Normalize([.485, .456, .406], [.229, .224, .225])])
    return torch.stack([transform(Image.open(path).convert('RGB')) for path in paths]).to(device)


def prepare(c, out, source):
    source = Path(source)
    d = pd.read_csv(source)
    if not {'image', 'label', 'split'} <= set(d):
        raise ValueError('Manifest requires image,label,split')
    d.image = d.image.map(lambda p: str((source.parent / p).resolve()))
    if d.image.duplicated().any():
        raise ValueError('Image identities cross partitions or repeat')
    d['clean_label'] = d['clean_label'] if 'clean_label' in d else d.label
    if 'ambiguous' not in d:
        d['ambiguous'] = False
    rng = np.random.default_rng(c['seed'])
    eligible = np.flatnonzero((d.split == 'train') & ~d.ambiguous.astype(bool))
    chosen = rng.choice(eligible, round(c['noise_rate'] * len(eligible)), replace=False)
    for index in chosen:
        original = int(d.loc[index, 'label'])
        if c['noise'] == 'symmetric':
            d.loc[index, 'label'] = (original + int(rng.integers(1, c['classes']))) % c['classes']
        elif c['noise'] == 'pair-flip':
            d.loc[index, 'label'] = (original + 1) % c['classes']
        elif c['noise'] == 'provided':
            d['clean_label'] = d.get('clean_label', d.label)
            break
        else:
            raise ValueError('noise must be symmetric, pair-flip or provided')
    d['corrupted'] = d.label != d.clean_label
    d.to_csv(out / 'manifest.csv', index=False)


def scan(model, data, c):
    logits, views, features = [], [], []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(data), c['batch_size']):
            batch = data.iloc[start:start + c['batch_size']]
            x = images(batch.image, False, c['device'])
            p, z = model(x)
            second, _ = model(torch.flip(x, [-1]))
            logits.append(p.cpu())
            views.append(second.cpu())
            features.append(z.cpu())
    return torch.cat(logits), torch.cat(views), torch.cat(features)


def train(c, out):
    torch.manual_seed(c['seed'])
    data = pd.read_csv(out / 'manifest.csv')
    training = data[data.split == 'train'].reset_index(drop=True)
    validation = data[data.split == 'val'].reset_index(drop=True)
    y = torch.tensor(training.label.to_numpy(), dtype=torch.long)
    model = Classifier(c['classes']).to(c['device'])
    optimizer = torch.optim.SGD(model.parameters(), lr=c['learning_rate'], momentum=.9, weight_decay=.0005)
    history = deque(maxlen=c['history'])
    ema_loss = torch.zeros(len(training))
    best, curve = -1, []
    posterior = torch.zeros(len(training), 3)
    posterior[:, 0] = 1
    for epoch in range(c['epochs']):
        logits, view, z = scan(model, training, c)
        anchors = torch.stack([z[y == label].mean(0) if (y == label).any() else torch.zeros(z.shape[1]) for label in range(c['classes'])])
        anchors = F.normalize(anchors, dim=1)
        similarity = z @ anchors.T
        if epoch % c['history'] == 0:
            candidates = similarity.topk(min(c['neighbors'], c['classes']), dim=1).indices
            reference = similarity.gather(1, candidates)
            history.clear()
        local = similarity.gather(1, candidates)
        second_features = view.softmax(-1).gather(1, candidates)
        history.append((local, second_features))
        reference = c['reference_ema'] * reference + (1 - c['reference_ema']) * local
        order = reference.argsort(1, descending=True)
        pair_left, pair_right = torch.triu_indices(candidates.shape[1], candidates.shape[1], 1)
        left = order[:, pair_left]
        right = order[:, pair_right]
        agreements = [(v.gather(1, left) > v.gather(1, right)).float().mean(1) for pair in history for v in pair]
        survival = torch.stack(agreements).mean(0)
        label_rank = torch.where(candidates.gather(1, order) == y[:, None],
                                 torch.arange(1, candidates.shape[1] + 1)[None], candidates.shape[1] + 1).min(1).values
        current_loss = F.cross_entropy(logits, y, reduction='none')
        ema_loss = .9 * ema_loss + .1 * current_loss
        view_agreement = (logits.argmax(1) == view.argmax(1)).float()
        signals = np.stack([survival, label_rank.float(), ema_loss, view_agreement], 1)
        if c['variant'] == 'no-temporal':
            signals[:, 0] = 0
        elif c['variant'] == 'confidence-only':
            signals = np.stack([logits.softmax(-1).max(1).values, current_loss], 1)
        if epoch >= c['warmup'] and c['variant'] != 'ce':
            components = 2 if c['variant'] == 'binary' else 3
            standardized = StandardScaler().fit_transform(signals)
            mixture = GaussianMixture(components, covariance_type='diag', reg_covar=1e-4, random_state=c['seed']).fit(standardized)
            raw = mixture.predict_proba(standardized)
            contradiction = np.array([(label_rank.numpy() * raw[:, k]).sum() / raw[:, k].sum() +
                                      (ema_loss.numpy() * raw[:, k]).sum() / raw[:, k].sum() for k in range(components)])
            clean, corrupt = int(contradiction.argmin()), int(contradiction.argmax())
            posterior = torch.zeros(len(training), 3)
            posterior[:, 0] = torch.tensor(raw[:, clean], dtype=torch.float32)
            posterior[:, 2] = torch.tensor(raw[:, corrupt], dtype=torch.float32)
            if components == 3:
                ambiguous = ({0, 1, 2} - {clean, corrupt}).pop()
                posterior[:, 1] = torch.tensor(raw[:, ambiguous], dtype=torch.float32)
        stable = posterior[:, 0] >= c['pseudo_threshold']
        soft = torch.zeros_like(logits)
        if stable.any():
            stable_ids = torch.nonzero(stable).flatten()
            for ids in torch.arange(len(z)).split(256):
                similarity_to_stable = z[ids] @ z[stable_ids].T
                similarity_to_stable[ids[:, None] == stable_ids[None]] = -torch.inf
                nearest = similarity_to_stable.topk(min(c['stable_neighbors'], len(stable_ids)), dim=1).indices
                soft[ids] = logits[stable_ids[nearest]].softmax(-1).mean(1)
        model.train()
        losses = []
        for ids in torch.randperm(len(training)).split(c['batch_size']):
            batch = training.iloc[ids.numpy()]
            first, embedding = model(images(batch.image, True, c['device']))
            _, second = model(images(batch.image, True, c['device']))
            labels = y[ids].to(c['device'])
            ce = F.cross_entropy(first, labels, reduction='none')
            scores = embedding @ anchors.to(c['device']).T
            local_scores = scores.gather(1, candidates[ids].to(c['device']))
            pair_difference = local_scores.gather(1, left[ids].to(c['device'])) - local_scores.gather(1, right[ids].to(c['device']))
            relational = F.softplus(-pair_difference).sum(1)
            if c['variant'] == 'no-order-loss':
                relational = ce
            contrastive = F.cross_entropy(embedding @ second.T / c['contrastive_temperature'], torch.arange(len(ids), device=first.device), reduction='none')
            pseudo = -(soft[ids].to(first.device) * first.log_softmax(-1)).sum(1)
            trusted = posterior[ids, 2].to(first.device) >= c['pseudo_threshold']
            weights = posterior[ids].to(first.device)
            loss = (weights[:, 0] * ce + weights[:, 1] * relational + weights[:, 2] * (contrastive + trusted * pseudo)).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        validation_logits = scan(model, validation, c)[0]
        accuracy = accuracy_score(validation.label, validation_logits.argmax(1))
        curve.append(dict(epoch=epoch, loss=np.mean(losses), validation_accuracy=accuracy))
        if accuracy > best:
            best = accuracy
            torch.save(model.state_dict(), out / 'model.pt')
            np.savez(out / 'states.npz', posterior=posterior.numpy(), survival=survival.numpy(),
                     corrupted=training.corrupted.to_numpy(), ambiguous=training.ambiguous.to_numpy())
        for group in optimizer.param_groups:
            group['lr'] = c['learning_rate'] * .5 * (1 + np.cos(np.pi * (epoch + 1) / c['epochs']))
    pd.DataFrame(curve).to_csv(out / 'learning.csv', index=False)


def run(c, out):
    model = Classifier(c['classes']).to(c['device'])
    model.load_state_dict(torch.load(out / 'model.pt', map_location=c['device'], weights_only=True))
    frame = pd.read_csv(out / 'manifest.csv')
    test = frame[frame.split == 'test'].reset_index(drop=True)
    probability = scan(model, test, c)[0].softmax(-1).numpy()
    np.savez(out / 'predictions.npz', probability=probability, y=test.label.to_numpy(),
             boundary=test.boundary.to_numpy() if 'boundary' in test else np.zeros(len(test), dtype=bool))


def evaluate(c, out):
    d = np.load(out / 'predictions.npz')
    state = np.load(out / 'states.npz')
    p, y = d['probability'], d['y']
    confidence, correct = p.max(1), p.argmax(1) == y
    result = dict(method=c['variant'], accuracy=float(correct.mean()))
    result['ece'] = sum(len(ids) / len(y) * abs(correct[ids].mean() - confidence[ids].mean()) for ids in np.array_split(np.argsort(confidence), min(15, len(y))))
    if len(np.unique(state['corrupted'])) == 2:
        result['noise_auroc'] = roc_auc_score(state['corrupted'], state['posterior'][:, 2])
    if state['ambiguous'].any():
        result['ambiguity_retention'] = float((state['posterior'][state['ambiguous'], :2].sum(1) >= c['pseudo_threshold']).mean())
    if d['boundary'].any():
        result['boundary_accuracy'] = float(correct[d['boundary']].mean())
    pd.DataFrame([result]).to_csv(out / 'metrics.csv', index=False)


def plot(out):
    d = pd.read_csv(out / 'learning.csv')
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(d.epoch, d.validation_accuracy)
    ax.set(xlabel='Epoch', ylabel='Validation accuracy')
    fig.tight_layout()
    fig.savefig(out / 'learning.png', dpi=180)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('command', choices=['prepare', 'train', 'run', 'evaluate', 'plot'])
    p.add_argument('--input')
    p.add_argument('--config', default='config.json')
    p.add_argument('--output', default='runs/default')
    p.add_argument('--variant')
    args = p.parse_args()
    c = json.loads(Path(args.config).read_text(encoding='utf-8'))
    if args.variant:
        c['variant'] = args.variant
    if c['variant'] not in ['full', 'ce', 'binary', 'no-temporal', 'no-order-loss', 'confidence-only']:
        p.error('Unknown variant')
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    settings = out / 'settings.json'
    if args.variant is None and settings.exists():
        c['variant'] = json.loads(settings.read_text(encoding='utf-8')).get('variant', c.get('variant', 'full'))
    settings.write_text(json.dumps(c, indent=2))
    if args.command == 'prepare':
        if not args.input:
            p.error('--input required')
        prepare(c, out, args.input)
    elif args.command == 'plot':
        plot(out)
    else:
        globals()[args.command](c, out)


if __name__ == '__main__':
    main()
