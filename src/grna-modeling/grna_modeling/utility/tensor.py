import torch


def padcat(x, val, dim=0):
    pad_dims = list(range(len(x[0].shape)))
    pad_dims.remove(dim)

    for pad_dim in pad_dims:
        max_dim_size = max([x_i.shape[pad_dim] for x_i in x])
        for x_i, x_ in enumerate(x):
            if not x_.shape[pad_dim] == max_dim_size:
                pad_shape = list(x_.shape)
                pad_shape[pad_dim] = max_dim_size - x_.shape[pad_dim]
                pad_i = torch.tensor(val).expand(pad_shape).to(x_.device)
                x[x_i] = torch.cat([x_, pad_i], dim=pad_dim)

    x = torch.cat(x, dim=dim)

    return x