from .base_tool import BaseTool
from .tool_registry import register_tool


@register_tool("NumericalCalculusTool")
class NumericalCalculusTool(BaseTool):
    """
    Tool for approximating derivatives and definite integrals using base Python methods.
    Provides:
      - forward, backward, central difference derivative approximations
      - trapezoidal and Simpson's rule numerical integration
    """

    def __init__(self, config=None):
        super().__init__(config)

    def run(self, args):
        mode = args.get("mode")
        func = args.get("function")
        a = args.get("a")
        b = args.get("b")
        h = args.get("h", 1e-5)
        n = args.get("n", 100)

        if not mode or not func:
            return {"error": "`mode` and `function` are required."}

        try:
            if mode == "derivative_forward":
                return {"result": self.forward_diff(func, a, h)}
            elif mode == "derivative_central":
                return {"result": self.central_diff(func, a, h)}
            elif mode == "derivative_backward":
                return {"result": self.backward_diff(func, a, h)}
            elif mode == "integrate_trapezoid":
                return {"result": self.trapezoidal_integral(func, a, b, n)}
            elif mode == "integrate_simpson":
                return {"result": self.simpsons_integral(func, a, b, n)}
            else:
                return {"error": f"Unknown mode `{mode}`."}
        except Exception as e:
            return {"error": str(e)}

    # ---------------- Derivative approximations ----------------

    def forward_diff(self, f_expr, x, h):
        """Forward finite difference: f'(x) ≈ (f(x+h) - f(x)) / h"""
        f = self._parse_function(f_expr)
        return (f(x + h) - f(x)) / h

    def backward_diff(self, f_expr, x, h):
        """Backward finite difference: f'(x) ≈ (f(x) - f(x-h)) / h"""
        f = self._parse_function(f_expr)
        return (f(x) - f(x - h)) / h

    def central_diff(self, f_expr, x, h):
        """Central difference: f'(x) ≈ (f(x+h) - f(x-h)) / (2h)"""
        f = self._parse_function(f_expr)
        return (f(x + h) - f(x - h)) / (2 * h)

    # ---------------- Integration approximations ----------------

    def trapezoidal_integral(self, f_expr, a, b, n):
        """Approximate ∫f(x)dx from a→b using the trapezoidal rule."""
        f = self._parse_function(f_expr)
        h = (b - a) / n
        total = 0.5 * (f(a) + f(b))
        for i in range(1, n):
            total += f(a + i * h)
        return total * h

    def simpsons_integral(self, f_expr, a, b, n):
        """Approximate ∫f(x)dx using Simpson's rule (n must be even)."""
        if n % 2 == 1:
            n += 1  # make even
        f = self._parse_function(f_expr)
        h = (b - a) / n
        total = f(a) + f(b)
        for i in range(1, n):
            coef = 4 if i % 2 == 1 else 2
            total += coef * f(a + i * h)
        return total * h / 3

    # ---------------- Helper ----------------

    def _parse_function(self, f_expr):
        """
        Convert a string expression (e.g. "x**2 + 3*x") to a callable f(x).
        Uses eval() safely in a restricted environment.
        """
        allowed = {"x": 0, "abs": abs, "pow": pow, "sqrt": lambda x: x ** 0.5}
        def f(x):
            return eval(f_expr, {"__builtins__": {}}, {**allowed, "x": x})
        return f
