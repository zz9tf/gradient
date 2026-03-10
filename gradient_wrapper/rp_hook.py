import torch


class RepresentationHookManager:
    def __init__(self, detach: bool = False, retain_grad: bool = False):
        self.detach = detach
        self.retain_grad = retain_grad
        self.repr_cache = {}
        self.grad_cache = {}
        self._hook_handles = []

    def clear(self):
        self.repr_cache.clear()
        self.grad_cache.clear()

    def remove_hooks(self):
        for h in self._hook_handles:
            h.remove()
        self._hook_handles = []

    def _normalize_output(self, out):
        if torch.is_tensor(out):
            return out
        if isinstance(out, (tuple, list)):
            for x in out:
                if torch.is_tensor(x):
                    return x
        if isinstance(out, dict):
            for _, x in out.items():
                if torch.is_tensor(x):
                    return x
        return None

    def make_hook(self, name):
        def fn(module, inp, out):
            x = self._normalize_output(out)
            if x is None:
                return

            if self.retain_grad and x.requires_grad:
                x.retain_grad()

            self.repr_cache[name] = x.detach() if self.detach else x
        return fn

    def register_by_name(self, model, target_modules):
        self.remove_hooks()
        named_modules = dict(model.named_modules())
        for name in target_modules:
            h = named_modules[name].register_forward_hook(self.make_hook(name))
            self._hook_handles.append(h)

    def collect_grads(self):
        self.grad_cache = {}
        for name, x in self.repr_cache.items():
            if torch.is_tensor(x) and hasattr(x, "grad") and x.grad is not None:
                self.grad_cache[name] = x.grad.detach()
        return self.grad_cache