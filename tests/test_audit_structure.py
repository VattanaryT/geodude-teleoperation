"""Structural audit: packaging, dependency declaration, layering, import side effects.

These tests encode expectations that hold for *any* installable Python package,
so a failure here is a packaging/cohesion defect rather than a taste question.
"""
import ast
import pathlib
import subprocess
import sys
import tomllib
import zipfile

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
PKG = REPO / "src" / "geodude_teleoperation"
EXAMPLES = REPO / "examples"

# Import name -> PyPI distribution name, for the imports this repo actually uses.
_DIST_FOR_IMPORT = {
    "gafropy": "gafropy",
    "numpy": "numpy",
    "scipy": "scipy",
    "trimesh": "trimesh",
    "rtree": "rtree",
    "viser": "viser",
    "pyspacemouse": "pyspacemouse",
    "pymeshlab": "pymeshlab",
    "robust_laplacian": "robust-laplacian",
    "yaml": "pyyaml",
    "matplotlib": "matplotlib",
    "rtde_receive": "ur-rtde",
    "rtde_io": "ur-rtde",
}

_STDLIB = set(sys.stdlib_module_names)


def _py_files(root):
    return sorted(p for p in root.rglob("*.py") if "__pycache__" not in p.parts)


def _module_level_imports(path):
    """Top-level (unconditional, module-scope) third-party import roots."""
    tree = ast.parse(path.read_text(), filename=str(path))
    roots = set()
    for node in tree.body:  # module scope only: lazy in-function imports are opt-in
        if isinstance(node, ast.Import):
            for alias in node.names:
                roots.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module.split(".")[0])
    return {r for r in roots if r not in _STDLIB and r != "geodude_teleoperation"}


@pytest.fixture(scope="module")
def declared_deps():
    data = tomllib.loads((REPO / "pyproject.toml").read_text())
    deps = list(data["project"].get("dependencies", []))
    for extra in data["project"].get("optional-dependencies", {}).values():
        deps.extend(extra)
    return {d.split("[")[0].split(">")[0].split("=")[0].split(";")[0].strip().lower()
            for d in deps}


def test_pyproject_declares_every_module_level_library_import(declared_deps):
    """A `pip install geodude-teleoperation` must be able to `import` the package.

    Anything imported at module scope inside src/ is a hard runtime dependency;
    if it is not in [project.dependencies] the fresh install raises ImportError
    on the very first import.
    """
    missing = {}
    for path in _py_files(PKG):
        for root in _module_level_imports(path):
            dist = _DIST_FOR_IMPORT.get(root, root).lower()
            if dist not in declared_deps:
                missing.setdefault(dist, []).append(
                    f"{path.relative_to(REPO)} imports {root!r}")
    assert not missing, (
        "undeclared hard dependencies -- `pip install .` then `import "
        f"geodude_teleoperation` will fail: {missing}")


def test_library_never_imports_from_examples():
    """examples/ is the top of the stack; src/ must not depend on it."""
    offenders = [
        str(p.relative_to(REPO)) for p in _py_files(PKG)
        if "examples" in {r for r in _module_level_imports(p)}
    ]
    assert offenders == [], f"library modules importing examples/: {offenders}"


def test_no_import_cycles_inside_the_package():
    graph = {}
    for path in _py_files(PKG):
        name = ".".join(path.relative_to(PKG.parent).with_suffix("").parts)
        name = name.removesuffix(".__init__")
        tree = ast.parse(path.read_text(), filename=str(path))
        deps = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and \
                    node.module.startswith("geodude_teleoperation"):
                deps.add(node.module)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith("geodude_teleoperation"):
                        deps.add(alias.name)
        graph[name] = deps

    cycles = []
    state = {}

    def visit(node, stack):
        if state.get(node) == "done":
            return
        if state.get(node) == "open":
            cycles.append(stack[stack.index(node):] + [node])
            return
        state[node] = "open"
        for dep in sorted(graph.get(node, ())):
            if dep in graph:
                visit(dep, stack + [node])
        state[node] = "done"

    for node in sorted(graph):
        visit(node, [])
    assert cycles == [], f"import cycles inside the package: {cycles}"


def test_importing_the_package_has_no_side_effects():
    """No threads, no subprocesses, no device/window/file handles at import time.

    A library that opens hardware or spawns a thread on import cannot be used
    from a notebook, a test runner, or a headless CI box.
    """
    script = (
        "import threading, os, sys\n"
        "before = threading.active_count()\n"
        "import geodude_teleoperation as gt\n"
        "after = threading.active_count()\n"
        "assert after == before, f'import started {after - before} thread(s)'\n"
        "print('OK')\n"
    )
    env = {"PYTHONPATH": str(PKG.parent), "PATH": "/usr/bin:/bin",
           "HOME": str(REPO)}
    proc = subprocess.run([sys.executable, "-c", script], capture_output=True,
                          text=True, env=env, timeout=180)
    assert proc.returncode == 0, (
        f"importing geodude_teleoperation failed or had side effects:\n"
        f"stdout={proc.stdout}\nstderr={proc.stderr[-3000:]}")


def _library_asset_paths():
    """Module-level constants in src/ that point at files on disk."""
    from geodude_teleoperation import ee_viz, laser_frames

    return {
        "ee_viz.CHAMFER_MESH": ee_viz.CHAMFER_MESH,
        "laser_frames.DEFAULT_T_LEFT": laser_frames.DEFAULT_T_LEFT,
        "laser_frames.DEFAULT_T_RIGHT": laser_frames.DEFAULT_T_RIGHT,
        "laser_frames.DEFAULT_CROP_META": laser_frames.DEFAULT_CROP_META,
        "laser_frames.DEFAULT_TAG0_OFFSET": laser_frames.DEFAULT_TAG0_OFFSET,
        "laser_frames.DEFAULT_LASER_OFFSET": laser_frames.DEFAULT_LASER_OFFSET,
    }


def test_library_default_asset_paths_live_inside_the_package():
    """Assets a *library* module reads must ship with the library.

    Both ee_viz and laser_frames build their default paths as
    `Path(__file__).parent.parent.parent / "assets" / ...`, which only lands on
    the repo's assets/ when running from a source checkout. From site-packages
    the same expression points at the venv's lib dir.
    """
    pkg_dir = pathlib.Path(
        pytest.importorskip("geodude_teleoperation").__file__).resolve().parent
    outside = {
        name: str(path) for name, path in _library_asset_paths().items()
        if pkg_dir not in pathlib.Path(path).resolve().parents
    }
    assert not outside, (
        "library asset paths resolve outside the installed package -- these "
        f"break after `pip install`: {outside}")


@pytest.fixture(scope="module")
def built_wheel(tmp_path_factory):
    """Build the project's own wheel exactly as `pip install .` would."""
    pytest.importorskip("setuptools")
    out = tmp_path_factory.mktemp("wheel")
    proc = subprocess.run(
        [sys.executable, "-m", "pip", "wheel", "--no-deps",
         "--no-build-isolation", "-w", str(out), str(REPO)],
        capture_output=True, text=True, timeout=600)
    wheels = list(out.glob("*.whl"))
    if not wheels:
        pytest.skip(f"could not build a wheel here: {proc.stderr[-800:]}")
    return wheels[0]


def test_wheel_ships_every_asset_the_library_reads(built_wheel):
    names = zipfile.ZipFile(built_wheel).namelist()
    shipped_assets = [n for n in names if "/assets/" in n or n.endswith(".stl")
                      or n.endswith(".json")]
    needed = _library_asset_paths()
    assert shipped_assets, (
        "the built wheel contains no asset files at all, yet library modules "
        f"read {sorted(needed)} at import time. pyproject.toml has no "
        "[tool.setuptools.package-data]/MANIFEST.in and assets/ sits outside "
        f"src/, so nothing under assets/ is packaged. Wheel contents: {names}")


def test_wheel_ships_every_python_module_in_the_package(built_wheel):
    names = set(zipfile.ZipFile(built_wheel).namelist())
    expected = {
        "/".join(p.relative_to(PKG.parent).parts) for p in _py_files(PKG)}
    missing = sorted(expected - names)
    assert not missing, f"modules present in src/ but absent from the wheel: {missing}"
