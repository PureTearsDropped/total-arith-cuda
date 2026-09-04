# bench_multif32_cuda.jl — MultiF32 / MultiU32 on the GPU: the NTuple kernels of MultiF32.jl or
# MultiU32.jl, unchanged, inside a CUDA.jl kernel (one thread = one F128/F256/F512 operation).
#
#   julia julia/bench_multif32_cuda.jl [N] [f32|u32] [aos|soa|both]
#       N      elements per array (default 2^20; needs CUDA.jl)
#       engine f32 = MultiF32.jl (Float32 digits), u32 = MultiU32.jl (29-bit integer digits)
#       layout aos = one array of structs (CuArray{F512}), soa = one array per field with the
#              digits as an N×K matrix, both (default) = the two side by side
#
# Prints ns per operation (kernel time / N by CUDA events: ≥ 1 s of warm-up launches, then the minimum
# of 10) with the kernel's register count and local (per-thread stack) memory, for copy / add / mul /
# div / sqrt at 128 / 256 / 512 bits, and checks every GPU result bit for bit against the CPU result of
# the same operands (the CPU results are the MPFR-verified ones).  The warm-up is by time, not by
# count: after the CPU verification pass idles the GPU for seconds, 10 launches of a kernel shorter
# than ~10 ms still measure at a reduced clock (F512 add 4.05 instead of 2.46 ns/op, F128 mul 1.01
# instead of 0.59) — ≥ 1 s of launches gives min ≈ max within 2 % for every kernel.
# Operands: random values with random signs and exponents −8…8, plus 1/64 of range edges (0, ±MAX, MIN,
# ±MAX·GE), so the cancellation, saturation, flag and zero paths run on the GPU too.  Measured cost of
# those paths (RTX 5090, F512): a zero in 1 lane of 32 costs 20–40 %, a sign flip in 1 lane of 32 costs
# 40 % on add — warp divergence, the SIMT tax on data-dependent branches.
#
# Layouts.  AoS: thread i reads struct i, so the 32 lanes of a warp that load the same field touch
# 32 addresses 84 bytes apart (F512) — 21 cache lines per load instruction instead of one; the copy
# kernel reaches 41 % of the device-to-device copy bandwidth printed as the reference (RTX 5090:
# 1518 GB/s).  SoA: one array per field, the digits as a matrix `d[i, j]` with the element index
# first, so a warp's load of digit j is 32 consecutive words = one 128-byte line; the copy reaches
# 96 % of the reference and add / mul gain 1.4–1.7× / 1.15–1.4×, div / sqrt nothing (register-bound).
# The arithmetic kernels are the same functions; only the load / store glue differs.  Use N ≥ 2^22
# for the layout comparison: at 2^20 two F128 / F256 arrays fit in the 96 MB L2 and the SoA copy
# measures above DRAM bandwidth.
using CUDA, Random, Printf
using Base.Cartesian: @nexprs

const N = length(ARGS) ≥ 1 ? parse(Int, ARGS[1]) : 2^20
const ENGINE = length(ARGS) ≥ 2 ? ARGS[2] : "f32"
const LAYOUT = length(ARGS) ≥ 3 ? ARGS[3] : "both"
if ENGINE == "u32"
    include(joinpath(@__DIR__, "MultiU32.jl")); using .MultiU32; const M = MultiU32
else
    include(joinpath(@__DIR__, "MultiF32.jl")); using .MultiF32; const M = MultiF32
end

# ---- AoS: CuArray{T} ------------------------------------------------------------------------
function kern!(f, z, x, y)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i ≤ length(z)
        @inbounds z[i] = f(x[i], y[i])
    end
    return
end

# ---- SoA: one CuArray per field, digits as an N×K matrix (element index first) ---------------
struct SoA{TN, TE, TD, TF}
    neg::TN; ex::TE; d::TD; flag::TF
end
function SoA(xs::Vector{T}) where {T}
    K = M.ndig(T)
    d = Matrix{eltype(fieldtype(T, :d))}(undef, length(xs), K)
    for (i, x) in enumerate(xs), j in 1:K
        d[i, j] = x.d[j]
    end
    SoA(CuArray([x.neg for x in xs]), CuArray([x.ex for x in xs]), CuArray(d), CuArray([x.flag for x in xs]))
end
function gather(::Type{T}, s::SoA) where {T}                                   # SoA → Vector{T} (host)
    neg = Array(s.neg); ex = Array(s.ex); d = Array(s.d); flag = Array(s.flag); K = M.ndig(T)
    [T(neg[i], ex[i], ntuple(j -> d[i, j], Val(K)), flag[i]) for i in eachindex(neg)]
end
bytes_per_element(::Type{T}) where {T} = 1 + 4 + 4 * M.ndig(T) + 1              # what SoA moves per element

# (inlined: a tuple returned from or passed to a non-inlined function goes through the stack —
#  local memory on the GPU, 160–936 B per thread and the SoA kernels 1.5–2× slower than AoS)
@generated function load_digits(d, i, ::Val{K}) where {K}
    :(Base.@_inline_meta; @inbounds ($([:(d[i, $j]) for j in 1:K]...),))
end
@generated function store_digits!(d, i, t::NTuple{K}) where {K}
    quote
        Base.@_inline_meta
        @inbounds @nexprs $K j -> d[i, j] = t[j]
        nothing
    end
end
@inline function soa_load(::Type{T}, neg, ex, d, flag, i) where {T}
    @inbounds T(neg[i], ex[i], load_digits(d, i, Val(M.ndig(T))), flag[i])
end
function kern_soa!(f, ::Val{T}, zn, ze, zd, zf, xn, xe, xd, xf, yn, ye, yd, yf) where {T}
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i ≤ length(zn)
        r = f(soa_load(T, xn, xe, xd, xf, i), soa_load(T, yn, ye, yd, yf, i))
        @inbounds begin
            zn[i] = r.neg; ze[i] = r.ex; zf[i] = r.flag
        end
        store_digits!(zd, i, r.d)
    end
    return
end

same(a::T, b::T) where {T} = a.neg === b.neg && a.ex === b.ex && a.flag === b.flag && a.d === b.d   # bitwise

# launch k with args, ≥ 1 s of warm-up, then the minimum of 10 timed launches
function timed(k, args; threads = 128, warm_s = 1.0)
    blocks = cld(N, threads)
    t0 = time(); n = 0
    while n < 10 || time() - t0 < warm_s                                      # warm-up: ≥ 10 launches AND ≥ 1 s
        k(args...; threads, blocks); n += 1                                  # (the GPU clock ramps over ~1 s
        n % 10 == 0 && CUDA.synchronize()                                    #  after the CPU pass idled it)
    end
    CUDA.synchronize()
    best = Inf
    for _ in 1:10
        best = min(best, CUDA.@elapsed k(args...; threads, blocks))
    end
    best, CUDA.registers(k), CUDA.memory(k).local
end
function gpu_time(f, dz, dx, dy)
    k = @cuda launch=false kern!(f, dz, dx, dy)
    timed(k, (f, dz, dx, dy))
end
function gpu_time_soa(f, ::Type{T}, sz, sx, sy) where {T}
    k = @cuda launch=false kern_soa!(f, Val(T), sz.neg, sz.ex, sz.d, sz.flag, sx.neg, sx.ex, sx.d, sx.flag, sy.neg, sy.ex, sy.d, sy.flag)
    timed(k, (f, Val(T), sz.neg, sz.ex, sz.d, sz.flag, sx.neg, sx.ex, sx.d, sx.flag, sy.neg, sy.ex, sy.d, sy.flag))
end

function operands(::Type{T}, rng, n) where {T}
    # random significands and signs (half the pairs have opposite signs: the cancellation path of add runs)
    rnd() = (v = M.rand_val(T, rng, rand(rng, -8:8)); rand(rng, Bool) ? T(true, v.ex, v.d, v.flag) : v)
    xs = [rnd() for _ in 1:n]
    # a few range edges so that the flag / saturation paths run on the GPU too
    edge = (zero(T), floatmax(T), floatmin(T), typemax(T), typemin(T))
    for _ in 1:max(4, n ÷ 64)
        xs[rand(rng, 1:n)] = rand(rng, edge)
    end
    xs
end

CUDA.functional() || error("CUDA.jl is not functional on this machine")
dev = CUDA.device()
println("GPU: $(CUDA.name(dev)), compute capability $(CUDA.capability(dev)), CUDA runtime $(CUDA.runtime_version()), driver $(CUDA.driver_version()), CUDA.jl $(pkgversion(CUDA))")
println("engine: $(M) — layout: $LAYOUT — N = $N elements per array; ns/op = kernel time / N (≥ 1 s warm-up, min of 10 launches); every result compared bitwise with the CPU")
println("GB/s = bytes moved per operation (2 arrays for copy, 3 for the rest; SoA bytes per element, i.e. without struct padding) / time")
println()
rng = MersenneTwister(20260903)
allsame = true
# bandwidth reference on this machine: device-to-device copy and a plain UInt32 copy kernel, 256 MB
let n = 2^26
    a = CUDA.rand(UInt32, n); b = similar(a)
    tm = Inf; tk = Inf
    k = @cuda launch=false kern!((x, y) -> x, b, a, a)
    for _ in 1:20
        tm = min(tm, CUDA.@elapsed copyto!(b, a))
        tk = min(tk, CUDA.@elapsed k((x, y) -> x, b, a, a; threads = 128, blocks = cld(n, 128)))
    end
    @printf("bandwidth reference (2 × 256 MB moved): copyto! D2D %.0f GB/s, UInt32 copy kernel (4 B per thread) %.0f GB/s\n\n", 8n / tm / 1e9, 8n / tk / 1e9)
end
const OPS0 = (("copy", (a, b) -> a), ("add", +), ("mul", *), ("div", /), ("sqrt", (a, b) -> sqrt(abs(a))))
# exp / log exist in MultiU32 only (2026-09-03); the operands' exponents −8…8 keep exp away from saturation
const OPS = ENGINE == "u32" ? (OPS0..., ("exp", (a, b) -> exp(a)), ("log", (a, b) -> log(abs(a)))) : OPS0
for T in (F128, F256, F512)
    global allsame
    xs = operands(T, rng, N); ys = operands(T, rng, N)
    @printf("%s  (P=%d bits, K=%d digits, %d bytes/element as a struct, %d as fields)\n", T, M.prec(T), M.ndig(T), sizeof(T), bytes_per_element(T))
    if LAYOUT != "soa"
        dx = CuArray(xs); dy = CuArray(ys); dz = similar(dx)
    end
    if LAYOUT != "aos"
        sx = SoA(xs); sy = SoA(ys); sz = SoA(fill(zero(T), N))
    end
    for (name, f) in OPS
        zc = [f(xs[i], ys[i]) for i in eachindex(xs)]
        nb = (name == "copy" ? 2 : 3) * bytes_per_element(T)
        if LAYOUT != "soa"
            t, regs, loc = gpu_time(f, dz, dx, dy)
            zg = Array(dz)
            nbad = count(i -> !same(zg[i], zc[i]), eachindex(zc))
            allsame &= nbad == 0
            @printf("  %-4s AoS %8.3f ns/op  %7.0f GB/s   regs %3d  local %5d B   GPU≠CPU: %d of %d\n",
                    name, 1e9 * t / N, nb / (1e9 * t / N), regs, loc, nbad, N)
        end
        if LAYOUT != "aos"
            t, regs, loc = gpu_time_soa(f, T, sz, sx, sy)
            zs = gather(T, sz)
            nbad = count(i -> !same(zs[i], zc[i]), eachindex(zc))
            allsame &= nbad == 0
            @printf("  %-4s SoA %8.3f ns/op  %7.0f GB/s   regs %3d  local %5d B   GPU≠CPU: %d of %d\n",
                    name, 1e9 * t / N, nb / (1e9 * t / N), regs, loc, nbad, N)
        end
    end
    println()
end
println(allsame ? "PASS: all GPU results bit-identical to the CPU" : "FAIL: GPU/CPU differences")
