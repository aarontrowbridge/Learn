using LinearAlgebra, Random

σ(z) = 1 / (1 + exp(-z))
dσ(a) = a * (1 - a)

quadratic_cost(h, y) = 0.5norm(h - y)^2

cross(h, y) = y * log(h) + (1 - y) * log(1 - h)
cross_entropy_cost(h, y) = -sum(cross.(h, y)) / length(y)

#
# data structures
#
const Data = Tuple{Vector{Float64}, Vector{Float64}}

struct Method
    cost::Function
    prop::Function
    function Method(alg::Symbol)
        if alg == :cren
            new(cross_entropy_cost, cross_entropy)
        elseif alg == :quad
            new(quadratic_cost, quadratic)
        end
    end
end

mutable struct Network
    w::Vector{Matrix{Float64}}
    b::Vector{Vector{Float64}}
    a::Vector{Vector{Float64}}
    err::Float64
    function Network(l::Vector{Int})
        w = [randn(r, c) / sqrt(l[1]) for (r, c) in zip(l[2:end], l[1:end-1])]
        b = [randn(n) for n in l[2:end]]
        a = [zeros(n) for n in l[2:end]]
        new(w, b, a, 0)
    end
end

#
# training functions
#
function train!(net::Network, btch::Vector{Data})
    net.err = 0
    Δw = [zeros(size(W)) for W in net.w]
    Δb = [zeros(size(b)) for b in net.b]
    for (x, y) in btch
        feed!(net, x)
        net.err += algo.cost(net.a[end], y)
        (dW, db) = algo.prop(net, y)
        popfirst!(net.a)
        Δw .+= dW
        Δb .+= db
    end
    net.w .-= η * Δw
    net.b .-= η * Δb
    net.err /= btchsz
end

function train_regularize!(net::Network, btch::Vector{Data})
    net.err = 0
    Δw = [zeros(size(W)) for W in net.w]
    Δb = [zeros(size(b)) for b in net.b]
    for (x, y) in btch
        feed!(net, x)
        net.err += algo.cost(net.a[end], y) + 0.5λ*norm(net.w[end])^2 / 60000
        (dW, db) = algo.prop(net, y)
        popfirst!(net.a)
        Δw .+= dW
        Δb .+= db
    end
    net.w .-= η * (Δw .+ (λ .* net.w) ./ 60000)
    net.b .-= η * Δb
    net.err /= btchsz
end

#
# feed forward function
#
function feed!(net::Network, x::Vector{Float64}, test=false)
    pushfirst!(net.a, x)
    for (W, b, l) in zip(net.w, net.b, 2:L)
        z = W * net.a[l-1] .+ b
        net.a[l] .= σ.(z)
    end
    if test popfirst!(net.a) end
end

#
# error propagation methods
#
function quadratic(net::Network, y::Vector{Float64})
    dW = [zeros(size(W)) for W in net.w]
    db = [zeros(size(b)) for b in net.b]
    h = net.a[end]
    Δ = (h - y) .* dσ.(h)
    for (W, a, l) in zip(reverse(net.w), net.a[end-1:-1:1], L-1:-1:1)
        dW[l] = Δ * a'
        db[l] = Δ
        Δ = W' * Δ .* dσ.(a)
    end
    (dW, db)
end

function cross_entropy(net::Network, y::Vector{Float64})
    dW = [zeros(size(W)) for W in net.w]
    db = [zeros(size(b)) for b in net.b]
    h = net.a[end]
    Δ = h - y
    for (W, a, l) in zip(reverse(net.w), net.a[end-1:-1:1], L-1:-1:1)
        dW[l] = Δ * a'
        db[l] = Δ
        Δ = W' * Δ .* dσ.(a)
    end
    (dW, db)
end

#
# data fetching functions
#
normalize(data, max=255) = data / max

function load_data()
    images = open("train-images-idx3-ubyte", "r")
    labels = open("train-labels-idx1-ubyte", "r")
    seek(images, 16)
    seek(labels, 8)

    imgs = Vector{Vector{Float64}}()
    labs = Vector{Vector{Float64}}()
    for i in 1:smplsz
        img = convert(Vector{Float64}, read(images, dim))
        img .= normalize.(img)
        push!(imgs, img)
        lab = fill(0.0, 10)
        lab[read(labels, 1)[1] + 1] = 1
        push!(labs, lab)
    end
    data = collect(zip(imgs, labs))
    if double data = [data; data] end
    shuffle!(data)

    test_imgs = open("t10k-images-idx3-ubyte", "r")
    test_labs = open("t10k-labels-idx1-ubyte", "r")
    seek(test_imgs, 16)
    seek(test_labs, 8)

    imgs = Vector{Vector{Float64}}()
    labs = Vector{Vector{Float64}}()
    for i in 1:testsz
        img = convert(Vector{Float64}, read(test_imgs, dim))
        img .= normalize.(img)
        push!(imgs, img)
        lab = fill(0.0, 10)
        lab[read(test_labs, 1)[1] + 1] = 1
        push!(labs, lab)
    end
    test = collect(zip(imgs, labs))
    shuffle!(test)

    btchs = [data[i:i-1+btchsz] for i in 1:btchsz:length(data)]
    (btchs, test)
end

#
# test function
#
function test(net::Network, test::Vector{Data})
    c = 0
    for (x, y) in test
        feed!(net, x, true)
        if findmax(net.a[end])[2] == findmax(y)[2] c += 1 end
    end
    c / testsz
end

#input dimension
const dim = 28^2

# double and shuffle data
const double = false

# data parameters
const smplsz = 60000
const btchsz = 10
const testsz = 10000
const btchn  = div(double ? 2smplsz : smplsz, btchsz)

# methed definition <mthd> ∈ {Quadratic Cost <quad>, Cross-Entropy <cren>}
# in shell use "julia neural.jl <mthd>"
const algo = Method(Symbol(ARGS[1]))

#
# hyperparameters
#
const η  = 0.25
const λ  = 4.00
const α  = 0.5
const ls = [dim, 20, 20, 10]

# function constants
const L  = length(ls)

# regularization boolean
const regularize = false

# batchs to iterate over for error avgerr calculation
const epoch = 100

function main()
    println("initializing\n")
    net::Network = Network(ls)
    (btchs, tst) = load_data()

    if regularize
        err = open("err_$(ARGS[1])_reg.dat", "w")
    else
        err = open("err_$(ARGS[1]).dat", "w")
    end

    avgerr = 0
    println("i = 0 : beginning training routine")

    for (btch, i) in zip(btchs, 1:btchn)
        regularize ? train_regularize!(net, btch) : train!(net, btch)
        avgerr += net.err
        if i % epoch == 0
            avgerr /= epoch
            println(err, "$(i*btchsz) $avgerr")
            div(i, epoch) % 5 == 0 ? println("i = $i") : continue
            i == btchn ? continue : avgerr = 0
        end
    end

    pc = test(net, tst)
    println()
    println("terminating avg error: $avgerr")
    println("% correct in test set: $(pc*100)\n")
end

@time main()

