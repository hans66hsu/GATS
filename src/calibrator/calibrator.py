from typing import Sequence
import numpy as np
import scipy
from scipy.interpolate import interp1d
from sklearn.isotonic import IsotonicRegression
import copy
import torch
from torch import nn, optim
from torch.nn import functional as F

from src.calibrator.attention_ts import CalibAttentionLayer
from src.model.model import GCN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def intra_distance_loss(output, labels):
    """
    Marginal regularization from CaGCN (https://github.com/BUPT-GAMMA/CaGCN)
    """
    output = F.softmax(output, dim=1)
    pred_max_index = torch.max(output, 1)[1]
    correct_i = pred_max_index==labels
    incorrect_i = pred_max_index!=labels
    output = torch.sort(output, dim=1, descending=True)
    pred,sub_pred = output[0][:,0], output[0][:,1]
    incorrect_loss = torch.sum(pred[incorrect_i]-sub_pred[incorrect_i]) / labels.size(0)
    correct_loss = torch.sum(1- pred[correct_i] + sub_pred[correct_i]) / labels.size(0)
    return incorrect_loss + correct_loss

def fit_calibration(temp_model, eval, data, train_mask, test_mask, patience = 100):
    """
    Train calibrator
    """    
    vlss_mn = float('Inf')
    with torch.no_grad():
        logits = temp_model.model(data.x, data.edge_index)
        labels = data.y
        edge_index = data.edge_index
        model_dict = temp_model.state_dict()
        parameters = {k: v for k,v in model_dict.items() if k.split(".")[0] != "model"}
    for epoch in range(2000):
        temp_model.optimizer.zero_grad()
        temp_model.train()
        # Post-hoc calibration set the classifier to the evaluation mode
        temp_model.model.eval()
        assert not temp_model.model.training
        calibrated = eval(logits)
        loss = F.cross_entropy(calibrated[train_mask], labels[train_mask])
        # dist_reg = intra_distance_loss(calibrated[train_mask], labels[train_mask])
        # margin_reg = 0.
        # loss = loss + margin_reg * dist_reg
        loss.backward()
        temp_model.optimizer.step()

        with torch.no_grad():
            temp_model.eval()
            calibrated = eval(logits)
            val_loss = F.cross_entropy(calibrated[test_mask], labels[test_mask])
            # dist_reg = intra_distance_loss(calibrated[train_mask], labels[train_mask])
            # val_loss = val_loss + margin_reg * dist_reg
            if val_loss <= vlss_mn:
                state_dict_early_model = copy.deepcopy(parameters)
                vlss_mn = np.min((val_loss.cpu().numpy(), vlss_mn))
                curr_step = 0
            else:
                curr_step += 1
                if curr_step >= patience:
                    break
    model_dict.update(state_dict_early_model)
    temp_model.load_state_dict(model_dict)


class TS(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, x, edge_index):
        logits = self.model(x, edge_index)
        temperature = self.temperature_scale(logits)
        return logits / temperature

    def temperature_scale(self, logits):
        """
        Expand temperature to match the size of logits
        """
        temperature = self.temperature.unsqueeze(1).expand(logits.size(0), logits.size(1))
        return temperature

    def fit(self, data, train_mask, test_mask, wdecay):
        self.to(device)
        def eval(logits):
            temperature = self.temperature_scale(logits)
            calibrated = logits / temperature
            return calibrated

        self.train_param = [self.temperature]
        self.optimizer = optim.Adam(self.train_param, lr=0.01, weight_decay=wdecay)
        fit_calibration(self, eval, data, train_mask, test_mask)
        return self


class VS(nn.Module):
    def __init__(self, model, num_classes):
        super().__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(num_classes))
        self.bias = nn.Parameter(torch.ones(num_classes))

    def forward(self, x, edge_index):
        logits = self.model(x, edge_index)
        temperature = self.vector_scale(logits)
        return logits * temperature + self.bias

    def vector_scale(self, logits):
        """
        Expand temperature to match the size of logits
        """
        temperature = self.temperature.unsqueeze(0).expand(logits.size(0), logits.size(1))
        return temperature

    def fit(self, data, train_mask, test_mask, wdecay):
        self.to(device)
        def eval(logits):
            temperature = self.vector_scale(logits)
            calibrated = logits * temperature + self.bias
            return calibrated

        self.train_param = [self.temperature]
        self.optimizer = optim.Adam(self.train_param, lr=0.01, weight_decay=wdecay)
        fit_calibration(self, eval, data, train_mask, test_mask)
        return self


class ETS(nn.Module):
    def __init__(self, model, num_classes):
        super().__init__()
        self.model = model
        self.weight1 = nn.Parameter(torch.ones(1))
        self.weight2 = nn.Parameter(torch.zeros(1))
        self.weight3 = nn.Parameter(torch.zeros(1))
        self.num_classes = num_classes
        self.temp_model = TS(model)

    def forward(self, x, edge_index):
        logits = self.model(x, edge_index)
        temp = self.temp_model.temperature_scale(logits)
        p = self.w1 * F.softmax(logits / temp, dim=1) + self.w2 * F.softmax(logits, dim=1) + self.w3 * 1/self.num_classes
        return torch.log(p)

    def fit(self, data, train_mask, test_mask, wdecay):
        self.to(device)
        self.temp_model.fit(data, train_mask, test_mask, wdecay)
        torch.cuda.empty_cache()
        logits = self.model(data.x, data.edge_index)[train_mask]
        label = data.y[train_mask]
        one_hot = torch.zeros_like(logits)
        one_hot.scatter_(1, label.unsqueeze(-1), 1)
        temp = self.temp_model.temperature.cpu().detach().numpy()
        w = self.ensemble_scaling(logits.cpu().detach().numpy(), one_hot.cpu().detach().numpy(), temp)
        self.w1, self.w2, self.w3 = w[0], w[1], w[2]
        return self

    def ensemble_scaling(self, logit, label, t):
        """
        Official ETS implementation from Mix-n-Match: Ensemble and Compositional Methods for Uncertainty Calibration in Deep Learning
        Code taken from (https://github.com/zhang64-llnl/Mix-n-Match-Calibration)
        Use the scipy optimization because PyTorch does not have constrained optimization.
        """
        p1 = np.exp(logit)/np.sum(np.exp(logit),1)[:,None]
        logit = logit/t
        p0 = np.exp(logit)/np.sum(np.exp(logit),1)[:,None]
        p2 = np.ones_like(p0)/self.num_classes
        

        bnds_w = ((0.0, 1.0),(0.0, 1.0),(0.0, 1.0),)
        def my_constraint_fun(x): return np.sum(x)-1
        constraints = { "type":"eq", "fun":my_constraint_fun,}
        w = scipy.optimize.minimize(ETS.ll_w, (1.0, 0.0, 0.0), args = (p0,p1,p2,label), method='SLSQP', constraints = constraints, bounds=bnds_w, tol=1e-12, options={'disp': False})
        w = w.x
        return w

    @staticmethod
    def ll_w(w, *args):
    ## find optimal weight coefficients with Cros-Entropy loss function
        p0, p1, p2, label = args
        p = (w[0]*p0+w[1]*p1+w[2]*p2)
        N = p.shape[0]
        ce = -np.sum(label*np.log(p))/N
        return ce   


class CaGCN(nn.Module):
    def __init__(self, model, num_nodes, num_class, dropout_rate):
        super().__init__()
        self.model = model
        self.num_nodes = num_nodes
        self.cagcn = GCN(num_class, 1, 16, drop_rate=dropout_rate, num_layers=2)

    def forward(self, x, edge_index):
        logits = self.model(x, edge_index)
        temperature = self.graph_temperature_scale(logits, edge_index)
        return logits * F.softplus(temperature)

    def graph_temperature_scale(self, logits, edge_index):
        """
        Perform graph temperature scaling on logits
        """
        temperature = self.cagcn(logits, edge_index)
        return temperature

    def fit(self, data, train_mask, test_mask, wdecay):
        self.to(device)
        def eval(logits):
            temperature = self.graph_temperature_scale(logits, data.edge_index)
            calibrated = logits * F.softplus(temperature)
            return calibrated

        self.train_param = self.cagcn.parameters()
        self.optimizer = optim.Adam(self.train_param, lr=0.01, weight_decay=wdecay)
        fit_calibration(self, eval, data, train_mask, test_mask)
        return self


class GATS(nn.Module):
    def __init__(self, model, edge_index, num_nodes, train_mask, num_class, dist_to_train, gats_args):
        super().__init__()
        self.model = model
        self.num_nodes = num_nodes
        self.cagat = CalibAttentionLayer(in_channels=num_class,
                                         out_channels=1,
                                         edge_index=edge_index,
                                         num_nodes=num_nodes,
                                         train_mask=train_mask,
                                         dist_to_train=dist_to_train,
                                         heads=gats_args.heads,
                                         bias=gats_args.bias)

    def forward(self, x, edge_index):
        logits = self.model(x, edge_index)
        temperature = self.graph_temperature_scale(logits)
        return logits / temperature

    def graph_temperature_scale(self, logits):
        """
        Perform graph temperature scaling on logits
        """
        temperature = self.cagat(logits).view(self.num_nodes, -1)
        return temperature.expand(self.num_nodes, logits.size(1))

    def fit(self, data, train_mask, test_mask, wdecay):
        self.to(device)
        def eval(logits):
            temperature = self.graph_temperature_scale(logits)
            calibrated = logits / temperature
            return calibrated

        self.train_param = self.cagat.parameters()
        self.optimizer = optim.Adam(self.train_param, lr=0.01, weight_decay=wdecay)
        fit_calibration(self, eval, data, train_mask, test_mask)
        return self


# multiclass isotonic regression
# c.f.: https://github.com/zhang64-llnl/Mix-n-Match-Calibration
class IRM(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.irm = IsotonicRegression(out_of_bounds='clip')

    @torch.no_grad()
    def forward(self, x, edge_index):
        logits = self.model(x, edge_index)
        probs = torch.softmax(logits, -1).cpu().numpy()
        p_calib = self.irm.predict(
            probs.flatten()).reshape(probs.shape) + 1e-9 * probs
        return torch.log(
            torch.from_numpy(p_calib).to(device) + torch.finfo().tiny)

    @torch.no_grad()
    def fit(self, data, train_mask, test_mask, wdecay):
        self.to(device)
        logits = self.model(data.x, data.edge_index)
        labels = data.y
        train_p = torch.softmax(logits[train_mask], -1).cpu().numpy()
        train_y = F.one_hot(
            labels[train_mask], train_p.shape[-1]).cpu().numpy()
        self.irm.fit_transform(train_p.flatten(), (train_y.flatten()))
        return self


class Dirichlet(nn.Module):
    def __init__(self, model, nclass: int):
        super().__init__()
        self.model = model
        self.dir = nn.Linear(nclass, nclass)

    def forward(self, x, edge_index):
        return self.calibrate(self.model(x, edge_index))

    def calibrate(self, logits):
        return self.dir(torch.log_softmax(logits, -1))

    def odir_loss(self, lamb: float = 0., mu: float = 0.):
        w, b = self.dir.weight, self.dir.bias
        loss = 0
        if lamb:
            k = len(b)
            assert k >= 2
            loss += lamb / (k * (k-1)) * (
                (w ** 2).sum() - (torch.diagonal(w) ** 2).sum())
        if mu:
            loss += mu * (b ** 2).mean()
        return loss

    def fit(
            self, data, train_mask, test_mask, wdecay, patience=100):
        self.to(device)
        optimizer = optim.Adam(self.dir.parameters(), lr=0.01)
        vlss_mn = float('Inf')

        with torch.no_grad():
            logits = self.model(data.x, data.edge_index)
            labels = data.y
            model_dict = self.state_dict()
            parameters = {k: v for k, v in model_dict.items() if
                          k.split(".")[0] != "model"}

        for epoch in range(2000):
            self.train()
            self.model.eval()
            optimizer.zero_grad()
            calibrated = self.calibrate(logits)
            loss = F.cross_entropy(
                calibrated[train_mask], labels[train_mask]
            ) + self.odir_loss(wdecay, wdecay)
            loss.backward()
            optimizer.step()

            self.eval()
            with torch.no_grad():
                val_loss = F.cross_entropy(
                    calibrated[test_mask], labels[test_mask]
                ) + self.odir_loss(wdecay, wdecay)
                if val_loss <= vlss_mn:
                    state_dict_early_model = copy.deepcopy(parameters)
                    vlss_mn = np.min((val_loss.cpu().numpy(), vlss_mn))
                    curr_step = 0
                else:
                    curr_step += 1
                    if curr_step >= patience:
                        break
        model_dict.update(state_dict_early_model)
        self.load_state_dict(model_dict)
        return self


# calibration with spline
# c.f.: https://github.com/kartikgupta-at-anu/spline-calibration
#
# authors didn't provide ways to calibrate the probabilistic prediction
#
# we rescale the non-prediction part, which preserve ECE and ACC in most cases
class SplineCalib(nn.Module):
    def __init__(self, model, knots=7):
        super().__init__()
        self.model = model
        self.knots = knots
        # identity as default
        self.calibfn = interp1d(np.asarray([0., 1.]), np.asarray([0., 1.]))

    @torch.no_grad()
    def forward(self, x, edge_index):
        logits = self.model(x, edge_index)
        probs = torch.softmax(logits, -1)
        scores, preds = probs.max(-1)
        p_calib = self.calibfn(scores.cpu().numpy()).clip(min=0.0, max=1.0)
        p_calib = torch.from_numpy(p_calib).to(
            device=probs.device, dtype=probs.dtype).unsqueeze(-1)
        new_probs = probs / (1 - scores.unsqueeze(-1))
        # mask out inf and nan
        ok_mask = torch.any((new_probs < torch.finfo().max), dim=-1)
        new_probs[~ok_mask, :] = (1. / (logits.shape[-1] - 1))
        new_probs = new_probs * (1 - p_calib)
        new_probs = torch.scatter(new_probs, -1, preds.unsqueeze(-1), p_calib)
        return torch.log(new_probs + torch.finfo().tiny)

    @torch.no_grad()
    def fit(self, data, train_mask, test_mask, wdecay):
        logits = self.model(data.x, data.edge_index)[train_mask]
        scores, preds = torch.softmax(logits, -1).max(-1)
        corrects = torch.eq(preds, data.y[train_mask])
        scores_sorted, sort_idx = scores.sort()
        corrects_sorted = corrects[sort_idx].cpu().numpy().astype(np.float32)
        scores_sorted = scores_sorted.cpu().numpy()
        del logits, scores, preds, corrects, sort_idx

        # merge duplicates
        scores_sorted, idx = np.unique(scores_sorted, return_inverse=True)
        corrects_sorted = np.bincount(
            idx, weights=corrects_sorted) / np.bincount(idx)
        del idx

        # Accumulate and normalize by dividing by num samples
        nsamples = len(scores_sorted)
        integrated_accuracy = np.cumsum(corrects_sorted) / nsamples
        integrated_scores = np.cumsum(scores_sorted) / nsamples
        percentile = np.linspace(0.0, 1.0, nsamples)

        # Now, try to fit a spline to the accumulated accuracy
        kx = np.linspace(0.0, 1.0, self.knots)
        spline = Spline(
            percentile, integrated_accuracy - integrated_scores, kx)

        # Evaluate the spline to get the accuracy
        calib_scores = scores_sorted + spline.evaluate_deriv(percentile)

        # Return the interpolating function -- uses full (not decimated) scores and
        # accuracy
        self.calibfn = interp1d(
            scores_sorted, calib_scores, fill_value='extrapolate')
        return self


# c.f.: https://github.com/kartikgupta-at-anu/spline-calibration
class Spline:

    # Initializer
    def __init__(self, x, y, kx, runout='natural'):

        # This calculates and initializes the spline

        # Store the values of the knot points
        self.kx = kx
        self.delta = kx[1] - kx[0]
        self.nknots = len(kx)
        self.runout = runout

        # Now, compute the other matrices
        m_from_ky = self.ky_to_M()  # Computes second derivatives from knots
        my_from_ky = np.concatenate([m_from_ky, np.eye(len(kx))], axis=0)
        y_from_my = self.my_to_y(x)
        y_from_ky = y_from_my @ my_from_ky

        # Now find the least squares solution
        ky = np.linalg.lstsq(y_from_ky, y, rcond=-1)[0]

        # Return my
        self.ky = ky
        self.my = my_from_ky @ ky

    def my_to_y(self, vecx):
        # Makes a matrix that computes y from M
        # The matrix will have one row for each value of x

        # Make matrices of the right size
        ndata = len(vecx)
        nknots = self.nknots
        delta = self.delta

        mM = np.zeros((ndata, nknots))
        my = np.zeros((ndata, nknots))

        for i, xx in enumerate(vecx):
            # First work out which knots it falls between
            j = int(np.floor((xx - self.kx[0]) / delta))
            if j >= self.nknots - 1: j = self.nknots - 2
            if j < 0: j = 0
            x = xx - j * delta

            # Fill in the values in the matrices
            mM[i, j] = -x ** 3 / (
                        6.0 * delta) + x ** 2 / 2.0 - 2.0 * delta * x / 6.0
            mM[i, j + 1] = x ** 3 / (6.0 * delta) - delta * x / 6.0
            my[i, j] = -x / delta + 1.0
            my[i, j + 1] = x / delta

        # Now, put them together
        M = np.concatenate([mM, my], axis=1)

        return M

    # -------------------------------------------------------------------------------

    def my_to_dy(self, vecx):
        # Makes a matrix that computes y from M for a sequence of values x
        # The matrix will have one row for each value of x in vecx
        # Knots are at evenly spaced positions kx

        # Make matrices of the right size
        ndata = len(vecx)
        h = self.delta

        mM = np.zeros((ndata, self.nknots))
        my = np.zeros((ndata, self.nknots))

        for i, xx in enumerate(vecx):
            # First work out which knots it falls between
            j = int(np.floor((xx - self.kx[0]) / h))
            if j >= self.nknots - 1: j = self.nknots - 2
            if j < 0: j = 0
            x = xx - j * h

            mM[i, j] = -x ** 2 / (2.0 * h) + x - 2.0 * h / 6.0
            mM[i, j + 1] = x ** 2 / (2.0 * h) - h / 6.0
            my[i, j] = -1.0 / h
            my[i, j + 1] = 1.0 / h

        # Now, put them together
        M = np.concatenate([mM, my], axis=1)

        return M

    # -------------------------------------------------------------------------------

    def ky_to_M(self):

        # Make a matrix that computes the
        A = 4.0 * np.eye(self.nknots - 2)
        b = np.zeros(self.nknots - 2)
        for i in range(1, self.nknots - 2):
            A[i - 1, i] = 1.0
            A[i, i - 1] = 1.0

        # For parabolic run-out spline
        if self.runout == 'parabolic':
            A[0, 0] = 5.0
            A[-1, -1] = 5.0

        # For cubic run-out spline
        if self.runout == 'cubic':
            A[0, 0] = 6.0
            A[0, 1] = 0.0
            A[-1, -1] = 6.0
            A[-1, -2] = 0.0

        # The goal
        delta = self.delta
        B = np.zeros((self.nknots - 2, self.nknots))
        for i in range(0, self.nknots - 2):
            B[i, i] = 1.0
            B[i, i + 1] = -2.0
            B[i, i + 2] = 1.0

        B = B * (6 / delta ** 2)

        # Now, solve
        Ainv = np.linalg.inv(A)
        AinvB = Ainv @ B

        # Now, add rows of zeros for M[0] and M[n-1]

        # This depends on the type of spline
        if (self.runout == 'natural'):
            z0 = np.zeros((1, self.nknots))  # for natural spline
            z1 = np.zeros((1, self.nknots))  # for natural spline

        if (self.runout == 'parabolic'):
            # For parabolic runout spline
            z0 = AinvB[0]
            z1 = AinvB[-1]

        if (self.runout == 'cubic'):
            # For cubic runout spline

            # First and last two rows
            z0 = AinvB[0]
            z1 = AinvB[1]
            zm1 = AinvB[-1]
            zm2 = AinvB[-2]

            z0 = 2.0 * z0 - z1
            z1 = 2.0 * zm1 - zm2

        # Reshape to (1, n) matrices
        z0 = z0.reshape((1, -1))
        z1 = z1.reshape((1, -1))

        AinvB = np.concatenate([z0, AinvB, z1], axis=0)

        return AinvB

    # -------------------------------------------------------------------------------

    def evaluate(self, x):
        # Evaluates the spline at a vector of values
        y = self.my_to_y(x) @ self.my
        return y

    # -------------------------------------------------------------------------------

    def evaluate_deriv(self, x):

        # Evaluates the spline at a vector (or single) point
        y = self.my_to_dy(x) @ self.my
        return y


# c.f.: https://github.com/AmirooR/IntraOrderPreservingCalibration
class OrderInvariantCalib(nn.Module):
    def __init__(self, model, nclass: int, nhiddens: Sequence[int] = None):
        super().__init__()
        self.model = model
        self.nclass = nclass
        self.nhiddens = (nclass,) if nhiddens is None else nhiddens
        self.base_calib = self._build_base_calib()
        self.invariant = True

    def _build_base_calib(self) -> nn.Module:
        sizes = [self.nclass] + list(self.nhiddens)
        layers = []
        for ni, no in zip(sizes[:-1], sizes[1:]):
            layers.append(nn.Linear(ni, no))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(self.nhiddens[-1], self.nclass))
        return nn.Sequential(*layers)

    @staticmethod
    def compute_u(sorted_logits):
        diffs = sorted_logits[:, :-1] - sorted_logits[:, 1:]
        diffs = torch.cat((
            diffs,
            torch.ones(
                (diffs.shape[0], 1), dtype=diffs.dtype, device=diffs.device)
        ), dim=1)
        return diffs.flip([1])

    def calibrate(self, logits):
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        unsorted_indices = torch.argsort(sorted_indices, descending=False)
        #[B, C]
        u = self.compute_u(sorted_logits)
        inp = sorted_logits if self.invariant else logits
        m = self.base_calib(inp)
        m[:, 1:] = F.softplus(m[:, 1:].clone())
        m[:, 0] = 0
        um = torch.cumsum(u*m, 1).flip([1])
        out = torch.gather(um, 1, unsorted_indices)
        return out

    def forward(self, x, edge_index):
        return self.calibrate(self.model(x, edge_index))

    def fit(self, data, train_mask, test_mask, wdecay):
        self.to(device)

        self.train_param = self.base_calib.parameters()
        self.optimizer = optim.Adam(
            self.train_param, lr=0.01, weight_decay=wdecay)
        fit_calibration(
            self, self.calibrate, data, train_mask, test_mask)
        return self
