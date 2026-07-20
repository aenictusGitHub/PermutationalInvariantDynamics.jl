# Canonical public spellings and retained compatibility aliases. Keep these
# tables centralized so high-level commands, scans, estimates, and low-level
# solvers cannot silently drift apart.
const _STATIONARY_ALGORITHM_NAMES=(
    :auto,:direct,:svd,:eigen,:shiftinvert,:shift_invert,
    :inverse_iteration,:gmres,:krylov)
const _SPECTRUM_ALGORITHM_NAMES=(
    :auto,:dense,:arnoldi,:krylov,:ordinary_arnoldi,
    :block_arnoldi,:block,:harmonic,:iram,:implicit_qr,
    :jd,:jacobi_davidson)

function _canonical_stationary_algorithm(method::Symbol)
    method in (:shift_invert,:inverse_iteration)&&return :shiftinvert
    method in (:gmres,:krylov)&&return :gmres
    method in (:auto,:direct,:svd,:eigen,:shiftinvert)&&return method
    throw(ArgumentError(
        "unsupported stationary-state algorithm $method; choose one of $(_STATIONARY_ALGORITHM_NAMES)"))
end

function _stationary_solver_method(method::Symbol)
    canonical=_canonical_stationary_algorithm(method)
    canonical===:gmres ? :krylov : canonical
end

function _canonical_spectrum_algorithm(method::Symbol)
    method in (:krylov,:ordinary_arnoldi)&&return :arnoldi
    method===:block&&return :block_arnoldi
    method===:implicit_qr&&return :iram
    method===:jacobi_davidson&&return :jd
    method in (:auto,:dense,:arnoldi,:block_arnoldi,:harmonic,:iram,:jd)&&
        return method
    throw(ArgumentError(
        "unsupported spectrum algorithm $method; choose one of $(_SPECTRUM_ALGORITHM_NAMES)"))
end

function _canonical_dynamics_algorithm(method::Symbol)
    method in (:adaptive_or_rk4,:dynamics)&&return :rk4
    method in (:auto,:rk4)&&return method
    throw(ArgumentError(
        "unsupported dynamics algorithm $method; choose :auto or :rk4"))
end
