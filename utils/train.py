# -*- coding: utf-8 -*-
import torch, numpy as np
from sklearn.metrics import accuracy_score, f1_score
from tqdm.auto import tqdm

class Trainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.crit = torch.nn.CrossEntropyLoss()
        self.opt  = None
        self.sch  = None

    def fit(self, model, dl_tr, dl_va, meta):
        model.to(self.device)
        self.opt = torch.optim.Adam(model.parameters(), lr=self.cfg["train"]["lr"])
        self.sch = torch.optim.lr_scheduler.ReduceLROnPlateau(self.opt, factor=0.5, patience=2)

        best, best_state = -1, None
        for ep in range(1, self.cfg["train"]["epochs"]+1):
            tr = self._run_epoch(model, dl_tr, train=True,  desc=f"train ep{ep:02d}")
            va = self._run_epoch(model, dl_va, train=False, desc=f"valid ep{ep:02d}")
            self.sch.step(va["loss"])
            score = 0.5*(va["f1_motor"] + va["f1_state"])
            tqdm.write(f"[{ep:02d}] loss_tr={tr['loss']:.4f} f1M={tr['f1_motor']:.3f} f1S={tr['f1_state']:.3f} "
                       f"| loss_va={va['loss']:.4f} f1M={va['f1_motor']:.3f} f1S={va['f1_state']:.3f}")
            if score > best:
                best, best_state = score, {k:v.cpu() for k,v in model.state_dict().items()}
        if best_state:
            model.load_state_dict(best_state)
            torch.save({"score":best, "state":best_state}, "best_model.pt")
            tqdm.write(f"[OK] Melhor score médio (F1): {best:.3f} — salvo em best_model.pt")

    def _run_epoch(self, model, dl, train, desc):
        model.train(train)
        losses = []
        yM_true, yM_pred = [], []
        yS_true, yS_pred = [], []
        for xb, y_m, y_s, k in tqdm(dl, desc=desc, leave=False):
            xb, y_m, y_s, k = xb.to(self.device), y_m.to(self.device), y_s.to(self.device), k.to(self.device)
            with torch.set_grad_enabled(train):
                lm, ls = model(xb, motor_k=k)
                loss = self.crit(lm, y_m) + self.crit(ls, y_s)
                if train:
                    self.opt.zero_grad(); loss.backward(); self.opt.step()
            losses.append(loss.item())
            yM_true.extend(y_m.cpu().numpy().tolist())
            yM_pred.extend(torch.argmax(lm, dim=1).cpu().numpy().tolist())
            yS_true.extend(y_s.cpu().numpy().tolist())
            yS_pred.extend(torch.argmax(ls, dim=1).cpu().numpy().tolist())

        f1M = f1_score(yM_true, yM_pred, average="macro", zero_division=0)
        f1S = f1_score(yS_true, yS_pred, average="macro", zero_division=0)
        return {"loss": float(np.mean(losses)), "f1_motor": float(f1M), "f1_state": float(f1S)}
