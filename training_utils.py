import torch


def load_saved(model, path, exact=True):
    # try:
    #     checkpoint = torch.load(path, map_location={'cuda:0': 'cuda:2'})
    # except:
    checkpoint = torch.load(path, map_location=torch.device('cpu'))
    if "q_model_state_dict" in checkpoint:
        state_dict = checkpoint['q_model_state_dict']
    elif 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    # state_dict.pop('module.proj.weight')
    # state_dict.pop('module.proj.bias')

    def filter(x):
        return x[7:] if x.startswith('module.') else x

    if exact:
        state_dict = {filter(k): v for (k, v) in state_dict.items()}
        # model_state_dict = {filter(k): v for (k, v) in model.state_dict().items()}
    else:
        state_dict = {filter(k): v for (k, v) in state_dict.items() if filter(k) in model.state_dict()}
        # model_state_dict = {filter(k): v for (k, v) in model.state_dict().items()}
    
    # print("state", state_dict['bert-base'].keys())
    # print("*"*100)
    # print("model", model.state_dict().keys())
    # state_dict = {"encoder."+k: v for (k, v) in state_dict.items()}

    # if "encoder.embeddings.position_ids" not in state_dict.keys():
    #     # state_dict["encoder.embeddings.position_ids"] = torch.arange(0, 514, device=None).view(1, -1).cuda()
    #     state_dict["encoder.embeddings.position_ids"] = torch.arange(0, 512, device=None).view(1, -1).cuda()
    # # print(state_dict["encoder.embeddings.position_ids"])

    # state_dict["encoder.embeddings.word_embeddings.weight"] = torch.cat([state_dict["encoder.embeddings.word_embeddings.weight"], model.state_dict()["encoder.embeddings.word_embeddings.weight"][-2:,:]], 0)

    model.load_state_dict(state_dict, strict=False)
    if 'model_state_dict' in checkpoint:
        return model, checkpoint
    elif "k_model_state_dict" in checkpoint:
        return model, checkpoint
    else:
        return model


def move_to_cuda(sample):
    if len(sample) == 0:
        return {}

    def _move_to_cuda(maybe_tensor):
        if torch.is_tensor(maybe_tensor):
            return maybe_tensor.cuda()
        elif isinstance(maybe_tensor, dict):
            return {
                key: _move_to_cuda(value)
                for key, value in maybe_tensor.items()
            }
        elif isinstance(maybe_tensor, list):
            return [_move_to_cuda(x) for x in maybe_tensor]
        else:
            return maybe_tensor

    return _move_to_cuda(sample)


def move_to_ds_cuda(sample, device):
    if len(sample) == 0:
        return {}

    def _move_to_cuda(maybe_tensor, device):
        if torch.is_tensor(maybe_tensor):
            return maybe_tensor.to(device)
        elif isinstance(maybe_tensor, dict):
            return {
                key: _move_to_cuda(value, device)
                for key, value in maybe_tensor.items()
            }
        elif isinstance(maybe_tensor, list):
            return [_move_to_cuda(x, device) for x in maybe_tensor]
        else:
            return maybe_tensor

    return _move_to_cuda(sample, device)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


