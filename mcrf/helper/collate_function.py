import torch


def weibo_collate_fn(device, batch):
    data = {
        "inputs": [],
        "mask": [],
        "segs": [],
        "tags": [],
    }
    for b in batch:
        data["inputs"].append(b["inputs"])
        data["mask"].append(b["mask"])
        if "segs" in b:
            data["segs"].append(b["segs"])
        if "tags" in b:
            data["tags"].append(b["tags"])

    for key in data:
        data[key] = torch.tensor(data[key], dtype=torch.long, requires_grad=False, device=device)
    return data
