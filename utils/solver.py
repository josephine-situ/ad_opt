"""
Unified MIP solver interface for Gurobi and SCIP.

Provides a thin abstraction layer that normalizes the API differences between
gurobipy and PySCIPOpt so the optimization code can switch solvers via a flag.

Usage:
    solver = MIPSolver("my_model", solver="gurobi")  # or solver="scip"
    x = solver.add_var(lb=0.0, name="x")
    solver.add_constr(x <= 10, name="bound")
    solver.set_objective(x, sense="maximize")
    solver.optimize()
    print(solver.get_val(x))
"""


class MIPSolver:
    """Unified MIP solver interface for Gurobi and SCIP."""

    GUROBI = "gurobi"
    SCIP = "scip"

    def __init__(self, name: str, solver: str = "gurobi"):
        self.solver_type = solver.lower()

        if self.solver_type == self.GUROBI:
            import gurobipy as gp

            self._gp = gp
            self._model = gp.Model(name)
        elif self.solver_type == self.SCIP:
            from pyscipopt import Model

            self._model = Model(name)
            self._apply_scip_defaults()
        else:
            raise ValueError(f"Unknown solver: {solver!r}. Choose 'gurobi' or 'scip'.")

    # ------------------------------------------------------------------
    # Parameters
    # ------------------------------------------------------------------
    _PARAM_MAP_SCIP = {
        "TimeLimit": "limits/time",
        "MIPGap": "limits/gap",
        "OutputFlag": "display/verblevel",
        # Gurobi-specific params with no SCIP equivalent
        "DualReductions": None,
        "InfUnbdInfo": None,
    }

    # SCIP defaults that prevent hangs on large XGBoost-embedded MIPs
    _SCIP_INIT_PARAMS = {
        "misc/usesymmetry": 0,       # disable symmetry detection (RAM-intensive, not useful here)
        "presolving/maxrounds": 0,    # skip heavy presolve rounds
    }

    def _apply_scip_defaults(self):
        """Apply SCIP-specific defaults at model creation."""
        for k, v in self._SCIP_INIT_PARAMS.items():
            self._model.setParam(k, v)

    def set_param(self, key, value):
        if self.solver_type == self.GUROBI:
            self._model.setParam(key, value)
        else:
            scip_key = self._PARAM_MAP_SCIP.get(key, key)
            if scip_key is None:
                return  # silently skip unsupported params
            if key == "OutputFlag":
                value = 4 if value else 0
            self._model.setParam(scip_key, value)

    # ------------------------------------------------------------------
    # Variables
    # ------------------------------------------------------------------
    @property
    def infinity(self):
        if self.solver_type == self.GUROBI:
            from gurobipy import GRB

            return GRB.INFINITY
        return 1e20  # PySCIPOpt convention

    def add_var(self, lb=0.0, ub=None, vtype="C", name=""):
        """Add a variable.  vtype: 'C' (continuous), 'B' (binary), 'I' (integer)."""
        if self.solver_type == self.GUROBI:
            from gurobipy import GRB

            vtype_map = {"C": GRB.CONTINUOUS, "B": GRB.BINARY, "I": GRB.INTEGER}
            kwargs = {"name": name, "vtype": vtype_map.get(vtype, vtype)}
            if lb is not None:
                kwargs["lb"] = lb
            if ub is not None:
                kwargs["ub"] = ub
            return self._model.addVar(**kwargs)
        else:
            kwargs = {"name": name, "vtype": vtype}
            if lb is not None:
                kwargs["lb"] = lb
            if ub is not None:
                kwargs["ub"] = ub
            return self._model.addVar(**kwargs)

    # ------------------------------------------------------------------
    # Constraints
    # ------------------------------------------------------------------
    def add_constr(self, constraint, name=""):
        if self.solver_type == self.GUROBI:
            self._model.addConstr(constraint, name=name)
        else:
            self._model.addCons(constraint, name=name)

    # ------------------------------------------------------------------
    # Expressions
    # ------------------------------------------------------------------
    def quicksum(self, terms):
        """Efficient summation of variables / linear expressions."""
        if self.solver_type == self.GUROBI:
            return self._gp.quicksum(terms)
        else:
            from pyscipopt import quicksum

            return quicksum(terms)

    def lin_expr(self, coeffs, variables):
        """Build ``sum(c_i * x_i)`` from coefficient and variable lists."""
        if self.solver_type == self.GUROBI:
            return self._gp.LinExpr(coeffs, variables)
        else:
            from pyscipopt import quicksum

            return quicksum(c * v for c, v in zip(coeffs, variables))

    # ------------------------------------------------------------------
    # Objective
    # ------------------------------------------------------------------
    def set_objective(self, expr, sense="maximize"):
        if self.solver_type == self.GUROBI:
            from gurobipy import GRB

            sense_map = {"maximize": GRB.MAXIMIZE, "minimize": GRB.MINIMIZE}
            self._model.setObjective(expr, sense_map[sense])
        else:
            self._model.setObjective(expr, sense)

    # ------------------------------------------------------------------
    # Solve
    # ------------------------------------------------------------------
    def update(self):
        if self.solver_type == self.GUROBI:
            self._model.update()
        # SCIP does not need an explicit update call

    def optimize(self):
        self._model.optimize()

    # ------------------------------------------------------------------
    # Status queries
    # ------------------------------------------------------------------
    @property
    def raw_status(self):
        if self.solver_type == self.GUROBI:
            return self._model.status
        return self._model.getStatus()

    def is_optimal(self):
        if self.solver_type == self.GUROBI:
            from gurobipy import GRB

            return self._model.status == GRB.OPTIMAL
        return self._model.getStatus() == "optimal"

    def is_time_limit(self):
        if self.solver_type == self.GUROBI:
            from gurobipy import GRB

            return self._model.status == GRB.TIME_LIMIT
        return self._model.getStatus() == "timelimit"

    def is_optimal_or_limit(self):
        return self.is_optimal() or self.is_time_limit()

    def is_infeasible(self):
        if self.solver_type == self.GUROBI:
            from gurobipy import GRB

            return self._model.status == GRB.INFEASIBLE
        return self._model.getStatus() == "infeasible"

    def is_inf_or_unbd(self):
        if self.solver_type == self.GUROBI:
            from gurobipy import GRB

            return self._model.status == GRB.INF_OR_UNBD
        return self._model.getStatus() in ("infeasible", "unbounded", "inforunbd")

    # ------------------------------------------------------------------
    # Solution access
    # ------------------------------------------------------------------
    def get_obj_val(self):
        if self.solver_type == self.GUROBI:
            return self._model.ObjVal
        return self._model.getObjVal()

    def get_val(self, var):
        if self.solver_type == self.GUROBI:
            return var.X
        return self._model.getVal(var)

    def get_vals(self, var_list):
        return [self.get_val(v) for v in var_list]

    # ------------------------------------------------------------------
    # Diagnostics (Gurobi-specific; best-effort on SCIP)
    # ------------------------------------------------------------------
    def compute_iis(self):
        if self.solver_type == self.GUROBI:
            self._model.computeIIS()
        else:
            print("[Info] IIS computation is not available in SCIP.")

    def write(self, path: str):
        if self.solver_type == self.GUROBI:
            self._model.write(path)
        else:
            self._model.writeProblem(path)

    # ------------------------------------------------------------------
    # Context manager (Gurobi models benefit from explicit dispose)
    # ------------------------------------------------------------------
    def dispose(self):
        if self.solver_type == self.GUROBI:
            self._model.dispose()
        elif self.solver_type == self.SCIP:
            self._model.freeProb()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.dispose()
        return False
