# ⚠️ AI-assisted; verify. / 生成AI使用・要検証
"""
TotArith — a scalar total-arithmetic Number for Julia.

  `TotNum <: Real`: value + a flag that names when the true value left the machine's
  representable range, WITH direction.  Overflow → ±MAX + GE (|true| ≥ |val|);
  underflow → ±MIN + LE (|true| ≤ |val|); a/0 = 0; NaN/Inf are never produced.

  Because it subtypes `Real` and overloads `Base.:+ - * /` etc., *existing generic
  Julia code runs on it unchanged* — the flag flows through any library that is written
  against `Number`/`Real` (ODE solvers, linear algebra, ...).  That is the whole point.
"""
module TotArith

export TotNum, GE, LE, SUNK, isflagged, flag_of, MAXF, MINF

const GE   = 0x01
const LE   = 0x02
const SUNK = 0x04
const MAXF = floatmax(Float64)
const MINF = floatmin(Float64)

struct TotNum <: Real
    val::Float64
    flag::UInt8
end
TotNum(x::Real) = _entry(Float64(x))          # entry totalization at construction
TotNum(x::TotNum) = x

flag_of(a::TotNum) = a.flag
isflagged(a::TotNum) = a.flag != 0x00
Base.Float64(a::TotNum) = a.val
Base.Float32(a::TotNum) = Float32(a.val)
Base.float(a::TotNum) = a.val
Base.AbstractFloat(a::TotNum) = a.val
(::Type{T})(a::TotNum) where {T<:AbstractFloat} = T(a.val)
Base.float(::Type{TotNum}) = TotNum
Base.big(a::TotNum) = big(a.val)

# ---- totalize a raw Float64 into (val, flag). Never NaN/Inf. ----
@inline function _sat(raw::Float64)
    isnan(raw) && return TotNum(0.0, GE | LE | SUNK)
    s = sign(raw); a = abs(raw)
    if a > MAXF || isinf(raw); return TotNum(s * MAXF, GE); end
    if a > 0 && a < MINF;      return TotNum(s * MINF, LE); end
    return TotNum(raw, 0x00)
end
@inline _entry(x::Float64) = _sat(x)

@inline function _addflag(fa, fb, va, vb)
    fin = fa | fb
    (fin == 0x00) && return 0x00
    same = sign(va) * sign(vb) > 0
    known = (fin & SUNK) == 0x00
    (known && same) ? fin : (GE | LE | SUNK)   # cancellation-safe
end
@inline function _mulflag(fa, fb)
    ga = fa & GE; la = (fa >> 1) & 0x01
    gb = fb & GE; lb = (fb >> 1) & 0x01
    ge = ((ga | gb) & ~(la | lb)) & 0x01
    le = ((la | lb) & ~(ga | gb)) & 0x01
    nb = ((ga | gb) & (la | lb)) & 0x01
    (ge * GE) | (le * LE) | (nb * (GE | LE)) | ((fa | fb) & SUNK)
end

# ---- the operator overloads: THIS is the bridge to the whole ecosystem ----
function Base.:+(a::TotNum, b::TotNum)
    r = _sat(a.val + b.val)
    TotNum(r.val, r.flag | _addflag(a.flag, b.flag, a.val, b.val))
end
function Base.:-(a::TotNum, b::TotNum)
    r = _sat(a.val - b.val)
    TotNum(r.val, r.flag | _addflag(a.flag, b.flag, a.val, -b.val))
end
Base.:-(a::TotNum) = TotNum(-a.val, a.flag)
function Base.:*(a::TotNum, b::TotNum)
    r = _sat(a.val * b.val)
    tz = (a.val == 0 && (a.flag & GE) == 0) || (b.val == 0 && (b.flag & GE) == 0)
    tz ? TotNum(0.0, 0x00) : TotNum(r.val, r.flag | _mulflag(a.flag, b.flag))
end
function Base.:/(a::TotNum, b::TotNum)
    bz = b.val == 0
    raw = bz ? 0.0 : a.val / b.val               # a/0 = 0
    r = _sat(raw)
    fin = a.flag | b.flag
    nb = (fin & (GE | LE)) > 0
    dz = (a.val == 0 && (a.flag & GE) > 0) || (b.val == 0 && (b.flag & GE) > 0)
    f = r.flag | (nb ? (GE | LE) : 0x00) | (fin & SUNK) | (dz ? SUNK : 0x00)
    TotNum(r.val, f)
end

# ---- the glue that lets generic Number/Real code accept TotNum ----
Base.promote_rule(::Type{TotNum}, ::Type{<:Real}) = TotNum
Base.convert(::Type{TotNum}, x::Real) = TotNum(x)
Base.zero(::Type{TotNum}) = TotNum(0.0, 0x00)
Base.one(::Type{TotNum})  = TotNum(1.0, 0x00)
Base.zero(::TotNum) = zero(TotNum)
Base.one(::TotNum)  = one(TotNum)
# comparisons + basics the ecosystem calls
Base.:<(a::TotNum, b::TotNum) = a.val < b.val
Base.:<=(a::TotNum, b::TotNum) = a.val <= b.val
Base.:(==)(a::TotNum, b::TotNum) = a.val == b.val
Base.isless(a::TotNum, b::TotNum) = a.val < b.val
Base.abs(a::TotNum) = TotNum(abs(a.val), a.flag)
Base.sign(a::TotNum) = TotNum(sign(a.val), (a.flag & SUNK))
function Base.sqrt(a::TotNum)
    r = _sat(sqrt(max(a.val, 0.0)))
    TotNum(r.val, r.flag | a.flag)               # monotone: bound flags pass through
end
# scalar transcendentals the solver machinery reaches for — each totalized (wrap + flag).
# ^ / exp overflow easily, so totalizing them is not just glue but honest range naming.
const CPLX = 0x08                                 # NEW: "real欄に置けない"(√-1型) — 複素へ行け
function Base.:^(a::TotNum, b::TotNum)
    x, y = a.val, b.val
    inflag = a.flag | b.flag
    # 指数=0 の 二種を 分ける（"0を予約語に"の 帰結）: 本物の0 だけが 空の積=1。
    if y == 0
        if b.flag == 0x00                          # 指数が **本物の0** → 空の積 → 1 (0^0 も 1)
            return TotNum(1.0, a.flag & SUNK)      # (底の 符号不明だけは 伝播・大きさは 1 で 確定)
        else                                       # 指数が **潰れた≈0(±MIN)** = 微小な非ゼロ → 空の積でない
            if x == 0 && a.flag == 0x00            #   0^(微小): 符号+なら0/−なら∞ → 割れる
                return TotNum(0.0, GE | LE | SUNK) #   確定できない → 境界なし+符号不明
            else                                   #   有限底: a^(微小) ≈ 1 (連続)
                return TotNum(1.0, inflag)
            end
        end
    end
    if x < 0 && !isinteger(y)                     # (負)^(非整数) = 実数の範囲外 → 型が違う
        return TotNum(0.0, inflag | CPLX)         # NaN でなく "複素へ" と名指し
    end
    s = (x < 0 && isodd(Int(round(y)))) ? -1.0 : 1.0
    r = _sat(s * abs(x)^y)                         # 溢れ→±MAX·GE / 潰れ→±MIN·LE を _sat が担当
    TotNum(r.val, r.flag | inflag)
end
Base.:^(a::TotNum, n::Integer) = (r = _sat(a.val^n); TotNum(r.val, r.flag | a.flag))
Base.literal_pow(::typeof(^), a::TotNum, ::Val{N}) where {N} = (r = _sat(a.val^N); TotNum(r.val, r.flag | a.flag))
Base.exp(a::TotNum) = (r = _sat(exp(a.val)); TotNum(r.val, r.flag | a.flag))
Base.log(a::TotNum) = (r = _sat(log(max(a.val, MINF))); TotNum(r.val, r.flag | a.flag))
Base.sin(a::TotNum) = TotNum(sin(a.val), a.flag)   # bounded [-1,1]: no new range flag
Base.cos(a::TotNum) = TotNum(cos(a.val), a.flag)
Base.inv(a::TotNum) = one(TotNum) / a
Base.:*(a::TotNum, b::Bool) = b ? a : zero(TotNum)   # solvers multiply by Bool masks
Base.:*(b::Bool, a::TotNum) = b ? a : zero(TotNum)
Base.nextfloat(a::TotNum) = TotNum(nextfloat(a.val), a.flag)
Base.prevfloat(a::TotNum) = TotNum(prevfloat(a.val), a.flag)
Base.eps(::Type{TotNum}) = TotNum(eps(Float64), 0x00)
Base.typemax(::Type{TotNum}) = TotNum(MAXF, GE)
Base.typemin(::Type{TotNum}) = TotNum(-MAXF, GE)
Base.isnan(::TotNum) = false                     # never — by construction
Base.isinf(::TotNum) = false
Base.isfinite(::TotNum) = true
Base.show(io::IO, a::TotNum) =
    print(io, a.flag == 0 ? string(a.val) :
              string(a.val, "⟦", (a.flag & GE)>0 ? "≥" : "", (a.flag & LE)>0 ? "≤" : "",
                     (a.flag & SUNK)>0 ? "±" : "", (a.flag & CPLX)>0 ? "ℂ" : "", "⟧"))

end # module
