import argparse
import gc
import os
import platform
import subprocess
import time

import numpy as np
import scipy
import scipy.sparse as sparse
import scipy.sparse.linalg as sparse_linalg
import qutip
import threadpoolctl
from qutip import piqs


OMEGA = 0.70
DETUNING = 0.23
GAMMA_DOWN = 0.31
GAMMA_UP = 0.09
VALIDATION_TOLERANCE = 5e-7


def active_threadpools():
    pools = threadpoolctl.threadpool_info()
    if not pools:
        return "none reported"
    return ";".join(
        f"{pool.get('internal_api', 'unknown')}:"
        f"{pool.get('prefix', 'unknown')}:"
        f"threads={pool.get('num_threads', 'unknown')}:"
        f"version={pool.get('version', 'unknown')}"
        for pool in pools
    )


ACTIVE_THREADPOOLS = active_threadpools()

COLUMNS = (
    "schema_version", "generated_unix_time", "package", "package_version",
    "dependency_versions",
    "language", "language_version", "os", "arch", "cpu",
    "logical_cpu_threads", "numerical_threads", "git_commit", "git_dirty",
    "N", "local_dimension", "physical_hilbert_dimension",
    "dicke_hilbert_dimension", "retained_operator_dimension",
    "embedded_liouville_dimension", "generator_nnz", "representation",
    "solver", "phase", "sample", "seconds", "omega", "detuning",
    "gamma_down", "gamma_up", "solver_atol", "solver_rtol",
    "validation_tolerance", "physical_residual_inf", "trace_error",
    "observable", "observable_value", "expected_observable",
    "observable_abs_error", "minimum_eigenvalue", "hermiticity_error",
    "iterations", "invariant_subspace_leakage", "validation_passed", "notes",
    "generator_probe_norm", "generator_probe_checksum_real",
    "generator_probe_checksum_imag",
)


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--sizes", default="8,16,24,32,40")
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    result = parser.parse_args()
    result.sizes = [int(value) for value in result.sizes.split(",") if value]
    if not result.sizes or min(result.sizes) <= 0:
        parser.error("--sizes must contain positive integers")
    if result.samples <= 0 or result.warmups < 0:
        parser.error("--samples must be positive and --warmups nonnegative")
    return result


def git_output(*parts):
    repository = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    try:
        return subprocess.check_output(
            ["git", "-C", repository, *parts], text=True,
            stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.SubprocessError):
        return "unavailable"


def expected_jz_per_particle():
    gamma_one = GAMMA_DOWN + GAMMA_UP
    gamma_two = gamma_one / 2
    equilibrium_z = (GAMMA_UP - GAMMA_DOWN) / gamma_one
    denominator = (gamma_one * (gamma_two**2 + DETUNING**2)
                   + OMEGA**2 * gamma_two)
    return (equilibrium_z * gamma_one * (gamma_two**2 + DETUNING**2)
            / (2 * denominator))


def physical_indices(N, nds):
    indices = []
    trace_indices = []
    degeneracies = []
    block_slices = []
    start = 0
    retained_start = 0
    for block_size in range(N + 1, 0, -2):
        j = (block_size - 1) / 2
        degeneracy = piqs.state_degeneracy(N, j)
        for column in range(block_size):
            for row in range(block_size):
                indices.append(start + row + nds * (start + column))
                degeneracies.append(degeneracy)
                if row == column:
                    trace_indices.append(len(indices) - 1)
        block_length = block_size * block_size
        block_slices.append((slice(retained_start, retained_start + block_length),
                             block_size, degeneracy))
        retained_start += block_length
        start += block_size
    return (np.asarray(indices, dtype=np.int64), trace_indices,
            np.asarray(degeneracies, dtype=np.float64), block_slices)


def prepare_case(N):
    jx, _, jz = piqs.jspin(N)
    hamiltonian = OMEGA * jx + DETUNING * jz
    ensemble = piqs.Dicke(
        N,
        hamiltonian=hamiltonian,
        emission=GAMMA_DOWN,
        pumping=GAMMA_UP,
    )
    liouvillian = ensemble.liouvillian()
    full_generator = liouvillian.data.as_scipy().tocsc()
    indices, trace_indices, degeneracies, block_slices = physical_indices(
        N, ensemble.nds)
    native_generator = full_generator[indices, :][:, indices].tocsc()

    # PIQS stores multiplicity-weighted blocks M_j=f_j*rho_j. PID's
    # equation-(7) coordinate is C_j=sqrt(f_j)*rho_j. Apply the diagonal
    # similarity x_PIQS=sqrt(f)*x_PID once during setup so both timed solvers
    # act on the same normalized coordinates.
    coordinate_scales = np.sqrt(degeneracies)
    generator = (
        sparse.diags(1 / coordinate_scales)
        @ native_generator
        @ sparse.diags(coordinate_scales)
    ).tocsc()

    mask = np.ones(full_generator.shape[0], dtype=bool)
    mask[indices] = False
    leakage_block = full_generator[mask, :][:, indices]
    leakage = (float(np.max(np.abs(leakage_block.data)))
               if leakage_block.nnz else 0.0)

    trace_vector = np.zeros(len(indices), dtype=np.complex128)
    trace_vector[trace_indices] = coordinate_scales[trace_indices]
    augmented = generator.tolil(copy=True)
    augmented[0, :] = trace_vector
    augmented = augmented.tocsc()
    right_hand_side = np.zeros(len(indices), dtype=np.complex128)
    right_hand_side[0] = 1
    observable = (
        jz.data.as_scipy().toarray().reshape(-1, order="F")[indices]
        * coordinate_scales
    )
    return {
        "N": N,
        "nds": ensemble.nds,
        "liouvillian": liouvillian,
        "full_generator_nnz": full_generator.nnz,
        "indices": indices,
        "outside_mask": mask,
        "generator": generator,
        "augmented": augmented,
        "rhs": right_hand_side,
        "trace": trace_vector,
        "degeneracies": degeneracies,
        "coordinate_scales": coordinate_scales,
        "blocks": block_slices,
        "observable": observable,
        "leakage": leakage,
    }


def solve_case(case):
    return sparse_linalg.spsolve(
        case["augmented"], case["rhs"], use_umfpack=False)


def prepare_splu_case(N):
    case = prepare_case(N)
    case["factorization"] = sparse_linalg.splu(case["augmented"])
    return case


def solve_splu_case(case):
    return case["factorization"].solve(case["rhs"])


def solve_public_case(case):
    density = qutip.steadystate(
        case["liouvillian"], method="direct", solver="spsolve", use_rcm=True,
        use_umfpack=False)
    vector = qutip.operator_to_vector(density).full().ravel()
    outside = vector[case["outside_mask"]]
    leakage = float(np.max(np.abs(outside))) if len(outside) else 0.0
    return vector[case["indices"]] / case["coordinate_scales"], leakage


def validate_case(case, state):
    residual_vector = case["generator"] @ state
    physical_residual = float(np.max(
        np.abs(residual_vector) / case["coordinate_scales"]))
    trace_error = float(abs(np.vdot(case["trace"], state) - 1))
    value = float(np.real(np.vdot(case["observable"], state)) / case["N"])
    expected = expected_jz_per_particle()
    observable_error = abs(value - expected)
    dimension = len(state)
    indices = np.arange(1, dimension + 1, dtype=np.float64)
    probe = (np.sin(0.37 * indices) + 1j * np.cos(0.19 * indices)) \
        / np.sqrt(dimension)
    weight = (np.cos(0.11 * indices) + 1j * np.sin(0.29 * indices)) \
        / np.sqrt(dimension)
    probe_image = case["generator"] @ probe
    generator_probe_norm = float(np.linalg.norm(probe_image))
    generator_probe_checksum = np.vdot(weight, probe_image)
    minimum_eigenvalue = np.inf
    hermiticity_error = 0.0
    for block_slice, block_size, degeneracy in case["blocks"]:
        weighted = state[block_slice].reshape(
            (block_size, block_size), order="F")
        physical = weighted / np.sqrt(degeneracy)
        scale = max(float(np.linalg.norm(physical)), np.finfo(float).tiny)
        hermiticity_error = max(
            hermiticity_error,
            float(np.linalg.norm(physical - physical.conj().T) / scale),
        )
        hermitian = (physical + physical.conj().T) / 2
        minimum_eigenvalue = min(
            minimum_eigenvalue, float(np.linalg.eigvalsh(hermitian)[0]))
    passed = (
        physical_residual <= VALIDATION_TOLERANCE
        and trace_error <= VALIDATION_TOLERANCE
        and observable_error <= VALIDATION_TOLERANCE
        and minimum_eigenvalue >= -VALIDATION_TOLERANCE
        and hermiticity_error <= VALIDATION_TOLERANCE
        and case["leakage"] == 0
    )
    if not passed:
        raise RuntimeError(
            f"QuTiP validation failed at N={case['N']}: "
            f"residual={physical_residual}, trace error={trace_error}, "
            f"observable error={observable_error}, "
            f"minimum eigenvalue={minimum_eigenvalue}, "
            f"Hermiticity error={hermiticity_error}, "
            f"subspace leakage={case['leakage']}")
    return {
        "residual": physical_residual,
        "trace_error": trace_error,
        "value": value,
        "expected": expected,
        "observable_error": observable_error,
        "minimum_eigenvalue": minimum_eigenvalue,
        "hermiticity_error": hermiticity_error,
        "passed": passed,
        "generator_probe_norm": generator_probe_norm,
        "generator_probe_checksum": generator_probe_checksum,
    }


def timed_setup(N, prepared=False):
    gc.collect()
    start = time.perf_counter_ns()
    case = prepare_splu_case(N) if prepared else prepare_case(N)
    return case, (time.perf_counter_ns() - start) / 1e9


def timed_solve(case, prepared=False):
    gc.collect()
    start = time.perf_counter_ns()
    state = solve_splu_case(case) if prepared else solve_case(case)
    return state, (time.perf_counter_ns() - start) / 1e9


def timed_time_to_solution(N):
    gc.collect()
    start = time.perf_counter_ns()
    case = prepare_case(N)
    state = solve_case(case)
    return case, state, (time.perf_counter_ns() - start) / 1e9


def result_row(case, validation, phase, sample, seconds, public=False,
               prepared=False, extra_note=""):
    status = git_output("status", "--porcelain")
    return {
        "schema_version": 1,
        "generated_unix_time": time.time(),
        "package": (
            "QuTiP-public" if public else
            "QuTiP-prepared" if prepared else "QuTiP"),
        "package_version": qutip.__version__,
        "dependency_versions": (
            f"NumPy={np.__version__},SciPy={scipy.__version__},"
            f"threadpoolctl={threadpoolctl.__version__}"),
        "language": "Python",
        "language_version": platform.python_version(),
        "os": platform.system(),
        "arch": platform.machine(),
        "cpu": platform.processor() or platform.machine(),
        "logical_cpu_threads": os.cpu_count(),
        "numerical_threads": ACTIVE_THREADPOOLS,
        "git_commit": git_output("rev-parse", "HEAD"),
        "git_dirty": "unknown" if status == "unavailable" else str(bool(status)).lower(),
        "N": case["N"],
        "local_dimension": 2,
        "physical_hilbert_dimension": 2 ** case["N"],
        "dicke_hilbert_dimension": case["nds"],
        "retained_operator_dimension": (
            case["nds"] ** 2 if public else len(case["rhs"])),
        "embedded_liouville_dimension": case["nds"] ** 2,
        "generator_nnz": (
            case["full_generator_nnz"] if public else case["generator"].nnz),
        "representation": (
            "PIQS_untrimmed_concatenated_Dicke_embedding" if public else
            "PIQS_physical_blocks_in_PID_equation_7_coordinates"
        ),
        "solver": (
            "qutip.steadystate_direct_SuperLU_use_rcm" if public else
            "scipy.sparse.linalg.splu_prepared_SuperLU" if prepared else
            "scipy.sparse.linalg.spsolve_SuperLU_use_umfpack_false"
        ),
        "phase": phase,
        "sample": sample,
        "seconds": seconds,
        "omega": OMEGA,
        "detuning": DETUNING,
        "gamma_down": GAMMA_DOWN,
        "gamma_up": GAMMA_UP,
        "solver_atol": "NA_direct",
        "solver_rtol": "NA_direct",
        "validation_tolerance": VALIDATION_TOLERANCE,
        "physical_residual_inf": validation["residual"],
        "trace_error": validation["trace_error"],
        "observable": "real(<Jz>)/N",
        "observable_value": validation["value"],
        "expected_observable": validation["expected"],
        "observable_abs_error": validation["observable_error"],
        "minimum_eigenvalue": validation["minimum_eigenvalue"],
        "hermiticity_error": validation["hermiticity_error"],
        "iterations": "NA_direct",
        "invariant_subspace_leakage": validation.get(
            "state_subspace_leakage", case["leakage"]),
        "validation_passed": str(validation["passed"]).lower(),
        "notes": (
            (
                f"official qutip.steadystate on the untrimmed PIQS embedding; "
                f"SciPy {scipy.__version__}; "
            ) if public else (
                f"QuTiP PIQS public generator; exact block-diagonal invariant "
                f"restriction and diagonal conversion to PID equation-(7) "
                f"coordinates; SciPy {scipy.__version__}; setup includes full "
                f"PIQS construction/restriction/conversion"
                + (f" and SuperLU factorization; solve is a prepared "
                   f"triangular solve; " if prepared else
                   f"; time_to_solution is fresh setup plus first spsolve; "
                   f"solve reuses the CSC system but spsolve refactors it on "
                   f"every call; use_umfpack=False forces SuperLU; ")
                + f"timed solve stops when the numerical solver returns and "
                f"common validation follows outside the clock, making the "
                f"reported solve ratio conservative for PID; "
            )
        ) + extra_note,
        "generator_probe_norm": validation["generator_probe_norm"],
        "generator_probe_checksum_real": float(
            np.real(validation["generator_probe_checksum"])),
        "generator_probe_checksum_imag": float(
            np.imag(validation["generator_probe_checksum"])),
    }


def write_rows(path, rows):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as stream:
        stream.write("\t".join(COLUMNS) + "\n")
        for row in rows:
            values = []
            for key in COLUMNS:
                value = str(row[key])
                if "\t" in value or "\n" in value:
                    raise ValueError("TSV value contains a tab or newline")
                values.append(value)
            stream.write("\t".join(values) + "\n")


def main():
    options = arguments()
    rows = []

    compilation_case = prepare_case(options.sizes[0])
    solve_case(compilation_case)

    for N in options.sizes:
        time_to_solution_seconds = []
        time_to_solution_case = None
        time_to_solution_state = None
        for _ in range(options.samples):
            (time_to_solution_case, time_to_solution_state,
             seconds) = timed_time_to_solution(N)
            time_to_solution_seconds.append(seconds)
        time_to_solution_validation = validate_case(
            time_to_solution_case, time_to_solution_state)

        setup_seconds = []
        case = None
        for _ in range(options.samples):
            case, seconds = timed_setup(N)
            setup_seconds.append(seconds)
        for _ in range(options.warmups):
            solve_case(case)
        solve_seconds = []
        state = None
        for _ in range(options.samples):
            state, seconds = timed_solve(case)
            solve_seconds.append(seconds)
        validation = validate_case(case, state)
        rows.extend(
            result_row(time_to_solution_case, time_to_solution_validation,
                       "time_to_solution", index, seconds)
            for index, seconds in enumerate(
                time_to_solution_seconds, start=1)
        )
        rows.extend(
            result_row(case, validation, "setup", index, seconds)
            for index, seconds in enumerate(setup_seconds, start=1)
        )
        rows.extend(
            result_row(case, validation, "solve", index, seconds)
            for index, seconds in enumerate(solve_seconds, start=1)
        )
        print(
            f"QuTiP N={N} retained={len(case['rhs'])} "
            f"residual={validation['residual']} "
            f"observable={validation['value']}",
            flush=True,
        )

        prepared_setup_seconds = []
        prepared_case = None
        for _ in range(options.samples):
            prepared_case, seconds = timed_setup(N, prepared=True)
            prepared_setup_seconds.append(seconds)
        for _ in range(options.warmups):
            solve_splu_case(prepared_case)
        prepared_solve_seconds = []
        prepared_state = None
        for _ in range(options.samples):
            prepared_state, seconds = timed_solve(
                prepared_case, prepared=True)
            prepared_solve_seconds.append(seconds)
        prepared_validation = validate_case(prepared_case, prepared_state)
        rows.extend(
            result_row(prepared_case, prepared_validation, "setup", index,
                       seconds, prepared=True)
            for index, seconds in enumerate(
                prepared_setup_seconds, start=1)
        )
        rows.extend(
            result_row(prepared_case, prepared_validation, "solve", index,
                       seconds, prepared=True)
            for index, seconds in enumerate(
                prepared_solve_seconds, start=1)
        )
        print(
            f"QuTiP prepared SuperLU N={N} "
            f"residual={prepared_validation['residual']}",
            flush=True,
        )

        # Probe the official high-level API on QuTiP's untrimmed embedding.
        # It can be singular because that embedding includes cross-sector
        # coherences outside physical PI density operators. Record successful
        # timings separately and record failures instead of hiding them.
        try:
            probe_state, state_leakage = solve_public_case(case)
            probe_validation = validate_case(case, probe_state)
            probe_validation["state_subspace_leakage"] = state_leakage
            if state_leakage > VALIDATION_TOLERANCE:
                raise RuntimeError(
                    f"public steady state leaked outside the physical PI "
                    f"subspace by {state_leakage}")
            for _ in range(options.warmups):
                solve_public_case(case)
            public_samples = []
            for _ in range(options.samples):
                gc.collect()
                start = time.perf_counter_ns()
                public_samples.append((
                    solve_public_case(case),
                    (time.perf_counter_ns() - start) / 1e9,
                ))
            public_state, state_leakage = public_samples[-1][0]
            public_validation = validate_case(case, public_state)
            public_validation["state_subspace_leakage"] = state_leakage
            if state_leakage > VALIDATION_TOLERANCE:
                raise RuntimeError(
                    f"timed public steady state leaked outside the physical "
                    f"PI subspace by {state_leakage}")
            rows.extend(
                result_row(
                    case, public_validation, "solve", index, seconds,
                    public=True,
                    extra_note=(
                        "official high-level baseline retained only where "
                        "the untrimmed solve succeeded and validated"
                    ),
                )
                for index, (_, seconds) in enumerate(public_samples, start=1)
            )
            print(
                f"QuTiP public N={N} succeeded; "
                f"residual={public_validation['residual']}", flush=True)
        except Exception as error:
            failed = {
                "residual": float("nan"),
                "trace_error": float("nan"),
                "value": float("nan"),
                "expected": expected_jz_per_particle(),
                "observable_error": float("nan"),
                "minimum_eigenvalue": float("nan"),
                "hermiticity_error": float("nan"),
                "passed": False,
                "state_subspace_leakage": float("nan"),
                "generator_probe_norm": float("nan"),
                "generator_probe_checksum": complex(float("nan"),
                                                       float("nan")),
            }
            message = " ".join(str(error).split())
            rows.append(result_row(
                case, failed, "unavailable", 0, float("nan"), public=True,
                extra_note=(
                    "official high-level probe failed; no timing ratio is "
                    f"reported: {type(error).__name__}: {message}"
                ),
            ))
            print(
                f"QuTiP public N={N} unavailable: "
                f"{type(error).__name__}: {message}", flush=True)
    write_rows(options.output, rows)
    print(f"wrote {options.output}")


if __name__ == "__main__":
    main()
