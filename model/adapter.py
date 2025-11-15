# coding=utf-8

import torch


class Adapter(torch.nn.Module):
    def __init__(
        self,
        image=True,
        hidden: int = 768,
        out: int = 1024
    ) -> None:
        super(Adapter, self).__init__()
        if image:
            self.linear1 = torch.nn.Linear(1536, out)
        else:
            self.linear1 = torch.nn.Linear(hidden, out)
        self.gelu = torch.nn.GELU()
        self.linear2 = torch.nn.Linear(out, out)

    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        y = self.linear1(x)
        y = self.gelu(y)
        out = self.linear2(y)
        return out
