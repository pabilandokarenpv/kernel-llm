KERNEL_SPEC = """
Kernel (.kn) is a Python-like systems language. ONLY use this exact
supported syntax subset, nothing else (no classes, no imports, no structs,
no lists/arrays, no string formatting beyond print's comma form):

- Program shape: always starts with `fn main():` then an indented body
  (4-space indent, no braces, no semicolons).
- Variables: `let name = expr` to declare, `name = expr` / `name += expr` /
  `name -= expr` to reassign an existing variable. Only integers exist.
- Printing: `print(expr)` prints one value. `print("label", expr)` prints a
  label then a value, comma-separated (this is the ONLY multi-arg form).
- If/elif/else: `if cond:` / `elif cond:` / `else:`, each with an indented
  body. Conditions use `<  >  <=  >=  ==  !=` comparing two expressions.
- For loop: `for v in lo..hi:` iterates v from lo up to (not including) hi.
- While loop: `while cond:` — always pair with a counter that gets
  decremented inside the body so it provably terminates, e.g.:
    let counter = 5
    while counter > 0:
        ...
        counter -= 1
- Expressions: integer literals, variable names, and binary ops
  `+  -  *  %` between two expressions, e.g. `(a + b)`, `(x % 10)`.
- Functions besides main: `fn name(param: int) -> int:` then an indented
  body with a `return expr` statement.

Example of fully valid Kernel code:
fn add(a: int, b: int) -> int:
    return a + b

fn main():
    let x = 5
    let y = 10
    print(add(x, y))
    for i in 0..3:
        print("i =", i)
    let counter = 3
    while counter > 0:
        print(counter)
        counter -= 1
"""
