export SimpleCTC
export ViterbiSimpleCTC

export SimpleCTCGreedySearch
export SimpleCTCGreedySearchWithTimestamp


LogZero(T::DataType) = - floatmax(T)

"""
    logsum2exp(a::Real, b::Real) -> max(a,b) + log(1.0 + exp(-abs(a-b)))

logsum2exp(log(a), log(b)) isequal to log(a + b)

```julia
julia> logsum2exp(Float32(1.2),Float64(3.3))
3.4155195283818967

julia> logsum2exp(log(1.0), log(2.0)) ≈ log(1.0 + 2.0)
true
```
"""
function logsum2exp(a::Real, b::Real)
    isinf(a) && return b
    isinf(b) && return a
    if a < b
        a, b = b, a
    end
    return (a + log(1.0 + exp(b-a)))
end


function seqsimplectc(seq::VecInt, blank::Int=1)
    if seq[1] == 0
        return [blank]
    end
    L = length(seq) # sequence length
    N = 2 * L + 1   # topology length
    label = zeros(Int, N)
    label[1:2:N] .= blank
    label[2:2:N] .= seq
    return label
end


"""
    SimpleCTC(p::Array{T,2}, seqlabel::VecInt; blank::Int=1)

# Topology Example
     ┌─►─┐    ┌─►─┐    ┌─►─┐    ┌─►─┐    ┌─►─┐    ┌─►─┐    ┌─►─┐
    ┌┴───┴┐  ┌┴───┴┐  ┌┴───┴┐  ┌┴───┴┐  ┌┴───┴┐  ┌┴───┴┐  ┌┴───┴┐
    │blank├─►│  C  ├─►│blank├─►│  A  ├─►│blank├─►│  T  ├─►│blank│
    └─────┘  └─────┘  └─────┘  └─────┘  └─────┘  └─────┘  └─────┘
"""
function SimpleCTC(p::Array{TYPE,2}, seqlabel::VecInt; blank::Int=1) where TYPE
    seq  = seqsimplectc(seqlabel, blank)
    ZERO = TYPE(0)                               # typed zero,e.g. Float32(0)
    S, T = size(p)                               # assert p is a 2-D tensor
    L = length(seq)                              # topology length with blanks
    r = fill!(Array{TYPE,2}(undef,S,T), ZERO)    # 𝜸 = p(s[k,t] | x[1:T]), k in softmax's indexing

    if L == 1
        r[blank,:] .= TYPE(1)
        return r, - sum(log.(p[blank,:]))
    end

    Log0 = LogZero(TYPE)                         # approximate -Inf of TYPE
    a = fill!(Array{TYPE,2}(undef,L,T), Log0)    # 𝜶 = p(s[k,t], x[1:t]), k in SimpleCTC topology's indexing
    b = fill!(Array{TYPE,2}(undef,L,T), Log0)    # 𝛃 = p(x[t+1:T] | s[k,t]), k in SimpleCTC topology's indexing
    a[1,1] = log(p[seq[1],1])
    a[2,1] = log(p[seq[2],1])
    b[L-1,T] = ZERO
    b[L  ,T] = ZERO

    # --- forward in log scale ---
    for t = 2:T
        τ = t-1
        first = max(1, t-T+L-1)
        lasst = min(1+t, L)
        for s = first:lasst
            if s≠1
                a[s,t] = logsum2exp(a[s,τ], a[s-1,τ])
            else
                a[s,t] = a[s,τ]
            end
            a[s,t] += log(p[seq[s],t])
        end
    end

    # --- backward in log scale ---
    for t = T-1:-1:1
        τ = t+1
        first = max(1, t-T+L-1)
        lasst = min(1+t, L)
        for s = first:lasst
            Q = b[s,τ] + log(p[seq[s],τ])
            if s≠L
                b[s,t] = logsum2exp(Q, b[s+1,τ] + log(p[seq[s+1],τ]))
            else
                b[s,t] = Q
            end
        end
    end

    logsum = logsum2exp(a[1,1] + b[1,1], a[2,1] + b[2,1])
    g = exp.((a + b) .- logsum)

    # reduce first line of g
    r[blank,:] .+= g[1,:]
    # reduce rest lines of g
    for n = 1:div(L-1,2)
        s = n<<1
        r[seq[s],:] .+= g[s,  :]
        r[blank, :] .+= g[s+1,:]
    end

    return r, -logsum
end


"""
    SimpleCTCGreedySearch(x::Array; blank::Int=1, dims=1) -> hypothesis
remove repeats and blanks of argmax(x, dims=dims)
"""
function SimpleCTCGreedySearch(x::Array; blank::Int=1, dims::Dimtype=1)
    hyp = Vector{Int}(undef, 0)
    idx = argmax(x, dims=dims)

    # first time-step
    previous = 0
    current  = idx[1][1]
    if current ≠ blank
        push!(hyp, current)
    end
    # rest time-steps
    for t = 2:length(idx)
        previous = current
        current  = idx[t][1]
        if !(current==previous || current==blank)
            push!(hyp, current)
        end
    end
    return hyp
end


"""
    SimpleCTCGreedySearchWithTimestamp(x::Array; blank::Int=1, dim::Int=1) -> hypothesis, timestamp
"""
function SimpleCTCGreedySearchWithTimestamp(x::Array; blank::Int=1, dims::Dimtype=1)
    hyp = Vector{Int}(undef, 0)
    stp = Vector{Float32}(undef, 0)
    idx = argmax(x; dims)
    T   = length(idx)

    # first time-step
    previous = 0
    current  = idx[1][1]
    if current ≠ blank
        push!(hyp, current)
        push!(stp, 1 / T)
    end
    # rest time-steps
    for t = 2:T
        previous = current
        current  = idx[t][1]
        if !(current==previous || current==blank)
            push!(hyp, current)
            push!(stp, t / T)
        end
    end
    return hyp, stp
end




"""
    ViterbiSimpleCTC(p::Array{F,2}, seqlabel::VecInt; blank::Int=1, eps::Real=1f-5)
force alignment by viterbi algorithm. eps is a label smoothing value, so one is slightly less than 1
and zero used is slightly bigger than 0.

# Topology Example
     ┌─►─┐    ┌─►─┐    ┌─►─┐    ┌─►─┐    ┌─►─┐    ┌─►─┐    ┌─►─┐
    ┌┴───┴┐  ┌┴───┴┐  ┌┴───┴┐  ┌┴───┴┐  ┌┴───┴┐  ┌┴───┴┐  ┌┴───┴┐
    │blank├─►│  S  ├─►│blank├─►│  U  ├─►│blank├─►│  N  ├─►│blank│
    └─────┘  └─────┘  └─────┘  └─────┘  └─────┘  └─────┘  └─────┘
"""
function ViterbiSimpleCTC(p::Array{TYPE,2}, seqlabel::VecInt; blank::Int=1, eps::Real=1f-5) where TYPE
    S, T = size(p)                               # assert p is a 2-D tensor
    seq  = seqsimplectc(seqlabel, blank)         # extend by topology constraint
    Log0 = LogZero(TYPE)                         # approximate -Inf of TYPE
    ZERO = TYPE(eps / (S-1))                     # closing to 0 but slightly bigger than 0
    ONE  = TYPE(1 - eps)                         # closing to 1 but slightly less than 1
    lnp  = TYPE(0)
    L = length(seq)                              # topology length with blanks, assert L ≤ T
    r = fill!(Array{TYPE,2}(undef,S,T), ZERO)    # 𝜸 = p(s[k,t] | x[1:T]), k in softmax's indexing

    if L == 1
        r[blank,:] .= ONE
        return r, - sum(log.(p[blank,:]))
    end

    d = fill!(Array{TYPE,2}(undef,L,T), Log0)    # accumulated scores when forward
    ϕ = zeros(UInt, L, T-1)                      # recording the best upstream nodes for each node
    h = zeros(UInt, T)                           # best path after backtracing

    # ══ init at fisrt timestep ══
    d[1,1] = log(p[seq[1],1])
    d[2,1] = log(p[seq[2],1])

    # ══ viterbi in log scale ══
    for t = 2:T
        τ = t-1
        first = max(1, t-T+L-1)
        lasst = min(1+t, L)
        if first ≠ 1 # then each node has two kids
            for s = first:lasst
                i = ifelse(d[s-1,τ] > d[s,τ], s-1, s)
                d[s,t] = d[i,τ] + log(p[seq[s],t])
                ϕ[s,τ] = i
            end
        else
            d[first,t] = d[first,τ] + log(p[blank,t])
            ϕ[first,τ] = 1
            for s = first+1:lasst
                i = ifelse(d[s-1,τ] > d[s,τ], s-1, s)
                d[s,t] = d[i,τ] + log(p[seq[s],t])
                ϕ[s,τ] = i
            end
        end
    end
    # ══ backtrace ══
    h[T] = ifelse(d[L,T] > d[L-1,T], L, L-1)
    for t = T-1:-1:1
        h[t] = ϕ[h[t+1],t]
    end
    # ══ one-hot assignment ══
    for t = 1:T
        i = seq[h[t]]
        r[i,t] = ONE
        lnp += log(p[i,t])
    end
    return r, -lnp
end
