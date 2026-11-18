"""
Kernel (.kn) random program generator — v1.

Generates syntactically valid, checker-passing Kernel programs for the LLM
training corpus. v1 scope: int-only arithmetic, if/elif/else, for-in-range
loops, and bounded while loops (counter-decrement pattern, guaranteed to
terminate). No structs/unsafe yet in the generator — those exist in the
language already (see kernel-lang/examples/) and get added here in a v1.1
pass once the base pipeline is proven out.

Every program generated here is designed to pass the ownership/safety
checker by construction (int-only values are always Copy, so there's
nothing to move-check), which matters because the corpus builder discards
anything `kernel trace` rejects.
"""

import random
import string


class ProgramGenerator:
    def __init__(self, seed=None):
        self.rng = random.Random(seed)
        self.var_counter = 0

    def fresh_var(self):
        name = f"v{self.var_counter}"
        self.var_counter += 1
        return name

    def gen_program(self, max_depth=3, max_stmts=10):
        self.var_counter = 0
        vars_in_scope = []
        body = self.gen_block(vars_in_scope, depth=0, max_depth=max_depth, max_stmts=max_stmts)
        lines = ["fn main():"]
        for stmt in body:
            lines.append(self.indent(stmt, 1))
        return "\n".join(lines) + "\n"

    def indent(self, stmt_lines, level):
        pad = "    " * level
        return "\n".join(pad + line if line else "" for line in stmt_lines.split("\n"))

    def gen_block(self, vars_in_scope, depth, max_depth, max_stmts):
        """Returns a list of statement strings (each may be multi-line)."""
        stmts = []
        n = self.rng.randint(2, max_stmts)
        local_vars = list(vars_in_scope)
        for _ in range(n):
            choices = ["let", "print"]
            if local_vars:
                choices += ["assign", "assign"]
            if depth < max_depth:
                choices += ["if", "for"]
                if depth < max_depth - 1:
                    choices += ["while"]
            kind = self.rng.choice(choices)

            if kind == "let":
                name = self.fresh_var()
                expr = self.gen_expr(local_vars)
                stmts.append(f"let {name} = {expr}")
                local_vars.append(name)
            elif kind == "assign":
                name = self.rng.choice(local_vars)
                op = self.rng.choice(["=", "+=", "-="])
                expr = self.gen_expr(local_vars)
                stmts.append(f"{name} {op} {expr}")
            elif kind == "print":
                if local_vars and self.rng.random() < 0.7:
                    name = self.rng.choice(local_vars)
                    stmts.append(f'print("{name} =", {name})')
                else:
                    stmts.append(f'print({self.rng.randint(0, 999)})')
            elif kind == "if":
                stmts.append(self.gen_if(local_vars, depth, max_depth, max_stmts))
            elif kind == "for":
                stmts.append(self.gen_for(local_vars, depth, max_depth, max_stmts))
            elif kind == "while":
                stmts.append(self.gen_while(local_vars, depth, max_depth, max_stmts))
        return stmts

    def gen_expr(self, local_vars, allow_var=True):
        """A small int-valued expression: literal, var, or a binary op of two."""
        choice = self.rng.random()
        if not local_vars or choice < 0.35:
            return str(self.rng.randint(0, 100))
        if allow_var and choice < 0.65:
            return self.rng.choice(local_vars)
        op = self.rng.choice(["+", "-", "*", "%"])
        left = self.rng.choice(local_vars) if local_vars else str(self.rng.randint(1, 50))
        right = str(self.rng.randint(1, 20)) if op in ("%", "*") else (
            self.rng.choice(local_vars) if local_vars and self.rng.random() < 0.5 else str(self.rng.randint(0, 50))
        )
        return f"({left} {op} {right})"

    def gen_cond(self, local_vars):
        op = self.rng.choice(["<", ">", "<=", ">=", "==", "!="])
        left = self.gen_expr(local_vars)
        right = self.gen_expr(local_vars)
        return f"{left} {op} {right}"

    def gen_if(self, local_vars, depth, max_depth, max_stmts):
        lines = [f"if {self.gen_cond(local_vars)}:"]
        body = self.gen_block(local_vars, depth + 1, max_depth, max(2, max_stmts - 3))
        for s in body:
            lines.append(self.indent(s, 1))
        if self.rng.random() < 0.4:
            lines.append(f"elif {self.gen_cond(local_vars)}:")
            body2 = self.gen_block(local_vars, depth + 1, max_depth, max(2, max_stmts - 3))
            for s in body2:
                lines.append(self.indent(s, 1))
        if self.rng.random() < 0.6:
            lines.append("else:")
            body3 = self.gen_block(local_vars, depth + 1, max_depth, max(2, max_stmts - 3))
            for s in body3:
                lines.append(self.indent(s, 1))
        return "\n".join(lines)

    def gen_for(self, local_vars, depth, max_depth, max_stmts):
        loop_var = self.fresh_var()
        lo = self.rng.randint(0, 5)
        hi = lo + self.rng.randint(1, 8)
        lines = [f"for {loop_var} in {lo}..{hi}:"]
        inner_vars = local_vars + [loop_var]
        body = self.gen_block(inner_vars, depth + 1, max_depth, max(2, max_stmts - 3))
        for s in body:
            lines.append(self.indent(s, 1))
        return "\n".join(lines)

    def gen_while(self, local_vars, depth, max_depth, max_stmts):
        # guaranteed-terminating pattern: a fresh counter that only this
        # loop decrements, bounded to a small constant.
        counter = self.fresh_var()
        bound = self.rng.randint(1, 8)
        lines = [f"let {counter} = {bound}"]
        loop_lines = [f"while {counter} > 0:"]
        body = self.gen_block(local_vars, depth + 1, max_depth, max(1, max_stmts - 4))
        for s in body:
            loop_lines.append(self.indent(s, 1))
        loop_lines.append(self.indent(f"{counter} -= 1", 1))
        return "\n".join(lines) + "\n" + "\n".join(loop_lines)


def gen_batch(n, seed_start=0, max_depth=3, max_stmts=10):
    programs = []
    for i in range(n):
        gen = ProgramGenerator(seed=seed_start + i)
        programs.append(gen.gen_program(max_depth=max_depth, max_stmts=max_stmts))
    return programs


if __name__ == "__main__":
    for p in gen_batch(3, seed_start=42):
        print("=" * 40)
        print(p)
