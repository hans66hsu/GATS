from typing import NamedTuple
import abc
import numpy as np
import torch
import torch.nn.functional as nnf
from torch import nn, Tensor, LongTensor, BoolTensor
from KDEpy import FFTKDE


# ref: https://stackoverflow.com/a/71801795
# do partial sums along dim 0 of tensor t
def partial_sums(t: Tensor, lens: LongTensor) -> Tensor:
    device = t.device
    elems, parts = t.size(0), len(lens)
    ind_x = torch.repeat_interleave(torch.arange(parts, device=device), lens)
    total = len(ind_x)
    ind_mat = torch.sparse_coo_tensor(
        torch.stack((ind_x, torch.arange(total, device=device)), dim=0),
        torch.ones(total, device=device, dtype=t.dtype),
        (parts, elems),
        device=device)
    return torch.mv(ind_mat, t)


class Reliability(NamedTuple):
    conf: Tensor
    acc: Tensor
    count: LongTensor


class ECE(nn.Module):
    binning_schemes = ('equal_width', 'uniform_mass')

    @staticmethod
    def equal_width_binning(
            confs: Tensor, corrects: BoolTensor, bins: int
    ) -> Reliability:
        sortedconfs, sortindices = torch.sort(confs)
        binidx = (sortedconfs * bins).long()
        binidx[binidx == bins] = bins - 1
        bincounts = binidx.bincount(minlength=bins)
        bincumconfs = partial_sums(sortedconfs, bincounts)
        bincumcorrects = partial_sums(
            corrects[sortindices].to(dtype=torch.get_default_dtype()),
            bincounts)
        return Reliability(
            conf=bincumconfs, acc=bincumcorrects, count=bincounts)

    @staticmethod
    def uniform_mass_binning(
            confs: Tensor, corrects: BoolTensor, bins: int
    ) -> Reliability:
        device = confs.device
        sortedconfs, sortindices = torch.sort(confs)
        indices = torch.div(
            torch.arange(bins + 1, device=device) * len(corrects),
            bins,
            rounding_mode='floor')
        bincounts = indices[1:] - indices[:-1]
        bincumconfs = partial_sums(sortedconfs, bincounts)
        bincumcorrects = partial_sums(
            corrects[sortindices].to(dtype=torch.get_default_dtype()),
            bincounts)
        return Reliability(
            conf=bincumconfs, acc=bincumcorrects, count=bincounts)

    def __init__(self, bins: int = 20, scheme: str = 'equal_width', norm=1):
        """
        bins: int, number of bins
        scheme: str, binning scheme
        norm: int or float, norm of error terms

        defaults follows:
        "On Calibration of Modern Neural Networks, Gou et. al., 2017"
        """
        assert scheme in ECE.binning_schemes
        super().__init__()
        self.bins = bins
        self.scheme = scheme
        self.norm = norm

    def binning(
            self, confs: Tensor, corrects: BoolTensor
    ) -> Reliability:
        scheme = self.scheme
        if scheme == 'equal_width':
            return ECE.equal_width_binning(confs, corrects, self.bins)
        elif scheme == 'uniform_mass':
            return ECE.uniform_mass_binning(confs, corrects, self.bins)
        else:
            raise ValueError(f'unrecognized binning scheme: {scheme}')

    def forward(self, confs: Tensor, corrects: BoolTensor) -> Tensor:
        bincumconfs, bincumcorrects, bincounts = self.binning(confs, corrects)
        # numerical trick to make 0/0=0 and other values untouched
        errs = (bincumconfs - bincumcorrects).abs() / (
            bincounts + torch.finfo().tiny)
        return ((errs ** self.norm) * bincounts / bincounts.sum()).sum()


class NodewiseMetric(nn.Module, metaclass=abc.ABCMeta):
    # edge_index - shape: (2, E), dtype: long
    def __init__(self, node_index: LongTensor):
        super().__init__()
        self.node_index = node_index

    @abc.abstractmethod
    def forward(self, logits: Tensor, gts: LongTensor) -> Tensor:
        raise NotImplementedError


class NodewiseNLL(NodewiseMetric):
    def forward(self, logits: Tensor, gts: LongTensor) -> Tensor:
        nodelogits = logits[self.node_index]
        nodegts = gts[self.node_index]
        return nnf.cross_entropy(nodelogits, nodegts)


class NodewiseBrier(NodewiseMetric):
    def forward(self, logits: Tensor, gts: LongTensor) -> Tensor:
        nodeprobs = torch.softmax(logits[self.node_index], -1)
        nodeconfs = torch.gather(
            nodeprobs, -1, gts[self.node_index].unsqueeze(-1)).squeeze(-1)
        return (nodeprobs.square().sum(dim=-1) - 2.0 * nodeconfs
                ).mean().add(1.0)


class NodewiseECE(NodewiseMetric):
    def __init__(
            self, node_index: LongTensor, bins: int = 15,
            scheme: str = 'equal_width', norm=1):
        super().__init__(node_index)
        self.ece_loss = ECE(bins, scheme, norm)

    def get_reliability(self, logits: Tensor, gts: LongTensor) -> Reliability:
        nodelogits, nodegts = logits[self.node_index], gts[self.node_index]
        nodeconfs, nodepreds = torch.softmax(nodelogits, -1).max(dim=-1)
        nodecorrects = (nodepreds == nodegts)
        return self.ece_loss.binning(nodeconfs, nodecorrects)

    def forward(self, logits: Tensor, gts: LongTensor) -> Tensor:
        nodelogits, nodegts = logits[self.node_index], gts[self.node_index]
        nodeconfs, nodepreds = torch.softmax(nodelogits, -1).max(dim=-1)
        nodecorrects = (nodepreds == nodegts)
        return self.ece_loss(nodeconfs, nodecorrects)

class NodewiswClassECE(NodewiseMetric):
    def __init__(
            self, node_index: LongTensor, bins: int = 15,
            scheme: str = 'equal_width', norm=1):
        super().__init__(node_index)
        self.ece_loss = ECE(bins, scheme, norm)

    def forward(self, logits: Tensor, gts: LongTensor) -> Tensor:
        nodelogits, nodegts = logits[self.node_index], gts[self.node_index]
        nodeconfs = torch.softmax(nodelogits, -1)
        num_classes = logits.size(1)
        class_ece = torch.zeros(num_classes, device=logits.device)
        for i in range(num_classes):
            classconfs = nodeconfs[:,i]
            frequency = nodegts.eq(i)
            assert classconfs.size() == frequency.size()
            class_ece[i] = self.ece_loss(classconfs, frequency)
        return torch.mean(class_ece)

class NodewiseKDE(NodewiseMetric):
    def __init__(self, node_index: LongTensor, norm=1):
        super().__init__(node_index)
        self.norm = norm

    def forward(self, logits: Tensor, gts: LongTensor) -> Tensor:
        nodelogits, nodegts = logits[self.node_index], gts[self.node_index]
        nodeconfs, nodepreds = torch.softmax(nodelogits, -1).max(dim=-1)
        nodecorrects = (nodepreds == nodegts)
        return KDE.ece_kde(nodeconfs, nodecorrects, norm=self.norm)


class KDE: 
    """
    Code adapted from https://github.com/zhang64-llnl/Mix-n-Match-Calibration
    """
    @staticmethod
    def mirror_1d(d, xmin=None, xmax=None):
        """If necessary apply reflecting boundary conditions."""
        if xmin is not None and xmax is not None:
            xmed = (xmin+xmax)/2
            return np.concatenate(((2*xmin-d[d < xmed]).reshape(-1,1), d, (2*xmax-d[d >= xmed]).reshape(-1,1)))
        elif xmin is not None:
            return np.concatenate((2*xmin-d, d))
        elif xmax is not None:
            return np.concatenate((d, 2*xmax-d))
        else:
            return d

    @staticmethod
    def density_estimator(conf, x_int, kbw, method='triweight'):
        # Compute KDE using the bandwidth found, and twice as many grid points
        low_bound, up_bound = 0.0, 1.0
        pp = FFTKDE(bw=kbw, kernel=method).fit(conf).evaluate(x_int)
        pp[x_int<=low_bound] = 0  # Set the KDE to zero outside of the domain
        pp[x_int>=up_bound] = 0  # Set the KDE to zero outside of the domain
        return pp * 2  # Double the y-values to get integral of ~1       
        
    @staticmethod
    @torch.no_grad()
    def ece_kde(confidence, correct, norm=1, kbw_choice='correct'):
        confidence = torch.clip(confidence,1e-256,1-1e-256)
        x_int = np.linspace(-0.6, 1.6, num=2**14)
        correct_conf = (confidence[correct==1].view(-1,1)).cpu().numpy()
        N = confidence.size(0)

        if kbw_choice == 'correct':
            kbw = np.std(correct_conf)*(N*2)**-0.2
        else:
            kbw = np.std(confidence.cpu().numpy())*(N*2)**-0.2
        # Mirror the data about the domain boundary
        low_bound = 0.0
        up_bound = 1.0
        dconf_1m = KDE.mirror_1d(correct_conf,low_bound,up_bound)
        pp1 = KDE.density_estimator(dconf_1m, x_int, kbw)
        pp1 = torch.from_numpy(pp1).to(confidence.device)

        pred_b_intm = KDE.mirror_1d(confidence.view(-1,1).cpu().numpy(),low_bound,up_bound)
        pp2 = KDE.density_estimator(pred_b_intm, x_int, kbw)
        pp2 = torch.from_numpy(pp2).to(confidence.device)

        # Accuracy (confidence)
        perc = torch.mean(correct.float())
        x_int = torch.from_numpy(x_int).to(confidence.device)
        integral = torch.zeros_like(x_int)

        conf = x_int
        accu = perc*pp1/pp2
        accu = torch.where((accu < 1.0), accu ,1.0)
        thre = ( pp1 > 1e-6) | (pp2 > 1e-6 ) 
        accu_notnan = ~torch.isnan(accu)
        integral[thre & accu_notnan] = torch.abs(conf[thre & accu_notnan]-accu[thre & accu_notnan])**norm*pp2[thre & accu_notnan]
        # Dont integrate the first sample 
        fail_thre_index = torch.nonzero(~thre)[1:]
        integral[fail_thre_index] = integral[fail_thre_index-1]

        ind = (x_int >= 0.0) & (x_int <= 1.0)
        return torch.trapz(integral[ind],x_int[ind]) / torch.trapz(pp2[ind],x_int[ind])