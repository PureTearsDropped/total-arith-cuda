# bench_libs.jl — MultiU32 against the existing multi-word libraries, same operands.
#   julia julia/bench_libs.jl      (needs MultiFloats, DoubleFloats, Quadmath and CUDA — installed by you,
#                                   not vendored; the GPU section needs a CUDA device)
#   CPU: F128 vs Float128 (Quadmath = libquadmath) / Double64 (DoubleFloats) / Float64x2 (MultiFloats) /
#        MPFR@113;  F256 vs Float64x4 / MPFR@237;  F512 vs MPFR@489 (MultiFloats 3.3.1 ships kernels for
#        ≤ 4 words only).  MPFR two ways: Julia's BigFloat wrapper (allocates every result) and the C
#        library itself through ccall into a preallocated result — the latter is "MPFR on one core".
#   Accuracy: the maximum error of each library's four operations against MPFR at 600 bits, in units of
#        2^-(its precision): correctly rounded ⟹ ≤ 1.
#   GPU: F128/F256/F512 (AoS, the like-for-like kernel; the SoA numbers are in the README) vs
#        Float64x2/x4 in the same one-thread-one-operation kernel; every GPU result is compared with the
#        CPU result of the same library.
using Random, Printf
include(joinpath(@__DIR__, "MultiU32.jl"))
using .MultiU32
using MultiFloats, DoubleFloats, Quadmath

const NP = 20_000
rng = MersenneTwister(20260903)

# operands: full-precision random significands (500 bits), random signs, exponents −8…8, as BigFloat sources
function sources(n)
    setprecision(BigFloat, 500) do
        [(rand(rng, Bool) ? -1 : 1) * ldexp(BigFloat(1) + rand(rng, BigFloat), rand(rng, -8:8)) for _ in 1:n]
    end
end
srcx = sources(NP); srcy = sources(NP)

conv(::Type{T}, v) where {T} = T(v)
conv(::Type{BigFloat}, v) = BigFloat(v)          # at the current precision

function time_op(f, xs::Vector{T}, ys::Vector{T}) where {T}
    z = similar(xs)
    best = Inf
    for _ in 1:5
        t = @elapsed begin
            @inbounds for i in eachindex(xs)
                z[i] = f(xs[i], ys[i])
            end
        end
        best = min(best, t)
    end
    1e9 * best / length(xs)
end

const OPS = (("add", +), ("mul", *), ("div", /), ("sqrt", (a, b) -> sqrt(abs(a))))

function row(label, ::Type{T}, prec) where {T}
    xs = setprecision(BigFloat, prec) do; [conv(T, v) for v in srcx]; end
    ys = setprecision(BigFloat, prec) do; [conv(T, v) for v in srcy]; end
    ts = setprecision(BigFloat, prec) do
        [time_op(f, xs, ys) for (_, f) in OPS]
    end
    @printf("  %-30s %6.1f %6.1f %7.1f %7.1f\n", label, ts...)
end

# MPFR, the C library: mpfr_add / mul / div / sqrt into a preallocated result (no allocation in the loop)
const libmpfr = Base.MPFR.libmpfr
const RN = Base.MPFR.MPFRRoundNearest
mpfr_add!(z, x, y) = ccall((:mpfr_add, libmpfr), Int32, (Ref{BigFloat}, Ref{BigFloat}, Ref{BigFloat}, Base.MPFR.MPFRRoundingMode), z, x, y, RN)
mpfr_mul!(z, x, y) = ccall((:mpfr_mul, libmpfr), Int32, (Ref{BigFloat}, Ref{BigFloat}, Ref{BigFloat}, Base.MPFR.MPFRRoundingMode), z, x, y, RN)
mpfr_div!(z, x, y) = ccall((:mpfr_div, libmpfr), Int32, (Ref{BigFloat}, Ref{BigFloat}, Ref{BigFloat}, Base.MPFR.MPFRRoundingMode), z, x, y, RN)
mpfr_sqrt!(z, x, y) = ccall((:mpfr_sqrt, libmpfr), Int32, (Ref{BigFloat}, Ref{BigFloat}, Base.MPFR.MPFRRoundingMode), z, x, RN)
mpfr_prec!(z, x, y) = ccall((:mpfr_get_prec, libmpfr), Clong, (Ref{BigFloat},), x)      # the ccall floor
function time_c(f!, xs::Vector{BigFloat}, ys::Vector{BigFloat})
    z = [BigFloat() for _ in 1:length(xs)]
    best = Inf
    for _ in 1:5
        t = @elapsed begin
            @inbounds for i in eachindex(xs)
                f!(z[i], xs[i], ys[i])
            end
        end
        best = min(best, t)
    end
    1e9 * best / length(xs)
end
function row_c(prec)
    xs = setprecision(BigFloat, prec) do; [abs(BigFloat(v)) for v in srcx]; end
    ys = setprecision(BigFloat, prec) do; [BigFloat(v) for v in srcy]; end
    ts = setprecision(BigFloat, prec) do
        [time_c(f!, xs, ys) for f! in (mpfr_add!, mpfr_mul!, mpfr_div!, mpfr_sqrt!)]
    end
    floor = time_c(mpfr_prec!, xs, ys)
    @printf("  %-30s %6.1f %6.1f %7.1f %7.1f   (ccall floor %.1f ns)\n", "MPFR C library ($prec)", ts..., floor)
end

println("CPU ($(Sys.cpu_info()[1].model)), single thread, ns per operation, min of 5 passes over $NP random pairs")
println("(random signs, exponents −8…8, 500-bit significands; a loop z[i] = f(x[i], y[i]), so LLVM may vectorize the pure-Float64 libraries)")
println("  library (bits of precision)     add    mul     div    sqrt")
println("~128 bits")
row("MultiU32 F128 (113)", F128, 113)
row("Quadmath Float128 (113)", Float128, 113)
row("DoubleFloats Double64 (106)", Double64, 106)
row("MultiFloats Float64x2 (106)", Float64x2, 106)
row("MPFR via BigFloat (113)", BigFloat, 113)
row_c(113)
println("~256 bits")
row("MultiU32 F256 (237)", F256, 237)
row("MultiFloats Float64x4 (212)", Float64x4, 212)
row("MPFR via BigFloat (237)", BigFloat, 237)
row_c(237)
println("~512 bits")
row("MultiU32 F512 (489)", F512, 489)
row("MPFR via BigFloat (489)", BigFloat, 489)
row_c(489)

# accuracy: max relative error of each library's four operations against MPFR at 600 bits, in units of 2^-prec
println()
println("max relative error against MPFR@600 over 2000 operands, in units of 2^-(bits of precision) — correctly rounded ⟹ ≤ 1:")
println("  library (bits of precision)      add     mul     div    sqrt")
function accuracy(label, ::Type{T}, prec) where {T}
    xs = setprecision(BigFloat, prec) do; [conv(T, v) for v in srcx[1:2000]]; end
    ys = setprecision(BigFloat, prec) do; [conv(T, v) for v in srcy[1:2000]]; end
    worst = zeros(4)
    setprecision(BigFloat, 600) do
        for i in eachindex(xs)
            bx = BigFloat(xs[i]); by = BigFloat(ys[i])
            for (k, (r, e)) in enumerate(((BigFloat(xs[i] + ys[i]), bx + by), (BigFloat(xs[i] * ys[i]), bx * by),
                                          (BigFloat(xs[i] / ys[i]), bx / by), (BigFloat(sqrt(abs(xs[i]))), sqrt(abs(bx)))))
                e == 0 && continue
                worst[k] = max(worst[k], Float64(abs(r - e) / abs(e)) * 2.0^prec)
            end
        end
    end
    @printf("  %-30s %7.3g %7.3g %7.3g %7.3g\n", label, worst...)
end
accuracy("MultiU32 F128 (113)", F128, 113)
accuracy("Quadmath Float128 (113)", Float128, 113)
accuracy("DoubleFloats Double64 (106)", Double64, 106)
accuracy("MultiFloats Float64x2 (106)", Float64x2, 106)
accuracy("MultiU32 F256 (237)", F256, 237)
accuracy("MultiFloats Float64x4 (212)", Float64x4, 212)
accuracy("MultiU32 F512 (489)", F512, 489)

# ---- GPU ----
using CUDA
function kern!(f, z, x, y)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i ≤ length(z)
        @inbounds z[i] = f(x[i], y[i])
    end
    return
end
const NG = 2^22
function gpu_row(label, ::Type{T}, prec) where {T}
    xs = setprecision(BigFloat, prec) do; [conv(T, v) for v in srcx]; end
    ys = setprecision(BigFloat, prec) do; [conv(T, v) for v in srcy]; end
    dx = CuArray(repeat(xs, cld(NG, NP))[1:NG]); dy = CuArray(repeat(ys, cld(NG, NP))[1:NG]); dz = similar(dx)
    ts = Float64[]; regs = Int[]; bad = Int[]
    for (name, f) in OPS
        try
            k = @cuda launch=false kern!(f, dz, dx, dy)
            blocks = cld(NG, 128)
            t0 = time(); n = 0
            while n < 10 || time() - t0 < 1.0
                k(f, dz, dx, dy; threads = 128, blocks); n += 1
                n % 10 == 0 && CUDA.synchronize()
            end
            CUDA.synchronize()
            best = Inf
            for _ in 1:10
                best = min(best, CUDA.@elapsed k(f, dz, dx, dy; threads = 128, blocks))
            end
            push!(ts, 1e9 * best / NG); push!(regs, CUDA.registers(k))
            # the GPU result must equal the CPU result of the same library (first NP elements)
            hz = Array(dz)[1:NP]
            push!(bad, count(i -> !(hz[i] == f(xs[i], ys[i])), 1:NP))
        catch err
            println("  $label $name: does not run on the GPU — ", first(sprint(showerror, err), 300))
            push!(ts, NaN); push!(regs, 0); push!(bad, -1)
        end
    end
    @printf("  %-30s %6.3f %6.3f %7.3f %7.3f   regs %s   GPU≠CPU %s\n", label, ts..., join(regs, "/"), join(bad, "/"))
end
println()
println("GPU ($(CUDA.name(CUDA.device()))), array of structs, N = $NG, ns per operation (≥ 1 s warm-up, min of 10):")
println("  library (bits of precision)     add    mul     div    sqrt")
gpu_row("MultiU32 F128 (113)", F128, 113)
gpu_row("MultiFloats Float64x2 (106)", Float64x2, 106)
gpu_row("MultiU32 F256 (237)", F256, 237)
gpu_row("MultiFloats Float64x4 (212)", Float64x4, 212)
gpu_row("MultiU32 F512 (489)", F512, 489)
