import os
import pickle
import datasets
import random
import transformers

class TokenizerWrapper:
    def __init__(self, input_ids):
        self.input_ids = input_ids


def _dataset_source(local_path, hub_id):
    return local_path if os.path.exists(local_path) else hub_id


def get_wikitext2(nsamples, seqlen, tokenizer, eval_mode=False):
    if eval_mode:
        testdata = datasets.load_dataset(_dataset_source('./datasets/wikitext', 'Salesforce/wikitext'), 'wikitext-2-raw-v1', split='test')
        testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')
        return testenc
    else:
        traindata = datasets.load_dataset(_dataset_source('./datasets/wikitext', 'Salesforce/wikitext'), 'wikitext-2-raw-v1', split='train')
        traindata = traindata.filter(lambda x: len(x['text']) > 0)
        traindata = traindata.map(lambda x : {'text': x['text'].strip()})
        trainenc = tokenizer("\n\n".join(traindata['text']), return_tensors='pt')    
        trainloader = []
        for _ in range(nsamples):
            i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            inp = trainenc.input_ids[:, i:j]
            tar = inp.clone()
            tar[:, :-1] = -100
            trainloader.append((inp, tar))
        return trainloader


def get_c4_new(nsamples, seqlen, tokenizer, eval_mode=False):
    if eval_mode:
        valdata = datasets.load_dataset(
        _dataset_source('./datasets/allenai/c4', 'allenai/c4'), data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation')
        valenc = tokenizer(' '.join(valdata[:1100]['text']), return_tensors='pt')
        valenc = valenc.input_ids[:, :(256 * seqlen)]
        valenc = TokenizerWrapper(valenc)
        return valenc
    else:
        traindata = datasets.load_dataset(
            _dataset_source('./datasets/allenai/c4', 'allenai/c4'), data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train')
        trainloader = []
        for _ in range(nsamples):
            while True:
                i = random.randint(0, len(traindata) - 1)
                trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
                if trainenc.input_ids.shape[1] >= seqlen:
                    break
            i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            inp = trainenc.input_ids[:, i:j]
            tar = inp.clone()
            tar[:, :-1] = -100
            trainloader.append((inp, tar))
        return trainloader


def get_ptb_new(nsamples, seqlen, tokenizer, eval_mode=False):
    if eval_mode:
        testdata = datasets.load_dataset(_dataset_source('./datasets/ptb_text_only', 'ptb_text_only'), 'penn_treebank', split='test')
        testenc = tokenizer(" ".join(testdata['sentence']), return_tensors='pt')
        return testenc
    else:
        traindata = datasets.load_dataset(_dataset_source('./datasets/ptb_text_only', 'ptb_text_only'), 'penn_treebank', split='train')
        trainenc = tokenizer(" ".join(traindata['sentence']), return_tensors='pt')
        trainloader = []
        for _ in range(nsamples):
            i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            inp = trainenc.input_ids[:, i:j]
            tar = inp.clone()
            tar[:, :-1] = -100
            trainloader.append((inp, tar))
        return trainloader


def get_pile(nsamples, seqlen, tokenizer):
    traindata = datasets.load_dataset("./datasets/pile-val-backup", split="validation")
    trainenc = tokenizer("\n\n".join(traindata['text'][:1000]), return_tensors='pt')
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader


def get_loaders(
    args, name, tokenizer, nsamples=128, seqlen=2048, eval_mode=False
):
    if 'wikitext2' in name:
        dataset = get_wikitext2(nsamples, seqlen, tokenizer, eval_mode)
    elif 'ptb' in name:
        dataset = get_ptb_new(nsamples, seqlen, tokenizer, eval_mode)
    elif 'c4' in name:
        dataset = get_c4_new(nsamples, seqlen, tokenizer, eval_mode)
    elif 'pile' in name:
        dataset = get_pile(nsamples, seqlen, tokenizer)

    if 'c4' in name and eval_mode:
        dataset = dataset.input_ids
        dataset = TokenizerWrapper(dataset)
    return dataset


def _find_image_column(dataset):
    """Return the name of the column that holds PIL images (or image dicts)."""
    from PIL.Image import Image as PILImage

    features = getattr(dataset, "features", {}) or {}
    for name in ("image", "img", "images", "jpg", "png"):
        if name in features:
            return name
    # Fall back to probing the first row for a PIL image.
    first = dataset[0]
    for key, value in first.items():
        if isinstance(value, PILImage):
            return key
    raise ValueError("Could not locate an image column in the vision calibration dataset.")


def _extract_image(example, column):
    image = example[column]
    if isinstance(image, dict) and "bytes" in image:
        import io
        from PIL import Image
        image = Image.open(io.BytesIO(image["bytes"]))
    return image.convert("RGB")


def get_vision_calib_loader(dataset_name, processor, nsamples=128, seed=42):
    """Yield ``(pixel_values, image_grid_thw)`` tensors for vision FlatQuant calibration.

    Images are pulled from a HuggingFace dataset and run through the model's own image
    processor so the patch layout (and therefore ``grid_thw``) matches inference. Only
    the image branch of the processor is used -- no text/tokenizer is required.
    """
    image_processor = getattr(processor, "image_processor", processor)

    dataset = None
    last_err = None
    for split in ("train", "validation", "test"):
        try:
            dataset = datasets.load_dataset(dataset_name, split=split)
            break
        except Exception as err:  # split may not exist; try the next one
            last_err = err
    if dataset is None:
        raise RuntimeError(f"Could not load vision calibration dataset {dataset_name}: {last_err}")

    column = _find_image_column(dataset)
    indices = list(range(len(dataset)))
    random.Random(seed).shuffle(indices)

    samples = []
    for idx in indices:
        if len(samples) >= nsamples:
            break
        try:
            image = _extract_image(dataset[idx], column)
        except Exception:
            continue
        processed = image_processor(images=image, return_tensors="pt")
        pixel_values = processed["pixel_values"]
        grid_thw = processed.get("image_grid_thw")
        if grid_thw is None:
            continue
        samples.append((pixel_values, grid_thw))
    if not samples:
        raise RuntimeError(f"No usable calibration images found in {dataset_name}.")
    return samples
