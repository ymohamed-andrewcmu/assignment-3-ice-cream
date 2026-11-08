"""Run this notebook's Python cells sequentially without a socket-based kernel.

The actual Python stdout and final expression are captured as notebook outputs.
Only use with a trusted notebook; cells execute with normal Python permissions.
"""
import ast
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import os

import nbformat


def main():
    root = Path(__file__).resolve().parent
    os.chdir(root)
    path = root / "train_model.ipynb"
    notebook = nbformat.read(path, as_version=4)
    namespace = {"__name__": "__main__"}
    count = 0
    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        count += 1
        tree = ast.parse(cell.source)
        final_expression = tree.body.pop().value if tree.body and isinstance(tree.body[-1], ast.Expr) else None
        stdout = StringIO()
        with redirect_stdout(stdout):
            exec(compile(tree, f"notebook-cell-{count}", "exec"), namespace)
            result = eval(compile(ast.Expression(final_expression), f"notebook-cell-{count}", "eval"), namespace) if final_expression else None
        cell.execution_count = count
        cell.outputs = []
        if stdout.getvalue():
            cell.outputs.append(nbformat.v4.new_output("stream", name="stdout", text=stdout.getvalue()))
        if result is not None:
            data = {"text/plain": repr(result)}
            if hasattr(result, "_repr_html_"):
                html = result._repr_html_()
                if html:
                    data["text/html"] = html
            cell.outputs.append(nbformat.v4.new_output("execute_result", execution_count=count, data=data, metadata={}))
        print(f"Executed cell {count}")
    nbformat.validate(notebook)
    nbformat.write(notebook, path)
    print(f"Saved {path.name} with real outputs from all {count} code cells.")


if __name__ == "__main__":
    main()
